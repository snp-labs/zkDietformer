from model_wrapper import ModelForMaskedLM
from dataset import CorpusDataset, CorpusDatasetV2
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import time
import os
import json
import pickle
import numpy as np
import argparse
import utils
import lz4.frame
# (추가)
from attention_nystrom import NystromAttention
from attn_witness_pipeline import process_attn_debug_npz

parser = argparse.ArgumentParser()
parser.add_argument("--use_data", type = utils.str2bool, help = "use_data", dest = "use_data", default = False)
# (추가/어텐션 수 기록에 필요한 것들)
parser.add_argument("--save_attn_npz", type = str, help = "prefix path to save attention intermediates npz (optional). Example: /tmp/attn_debug", default = None)
parser.add_argument("--witness_v_scale", type=int, default=1)
parser.add_argument("--witness_inv_scale", type=int, default=1)
parser.add_argument("--witness_scale_input", type=int, default=(1<<8))
parser.add_argument("--witness_frac_bits", type=int, default=16)
parser.add_argument("--witness_k_segments", type=int, default=2)
parser.add_argument("--witness_limb_bits", type=int, default=4)
parser.add_argument("--witness_tables_outdir", type=str, default="tmp_tables")
args = parser.parse_args()

with open("/model/config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = config["model_checkpoints"]
data_folder = config["data_folder"]

os.makedirs(checkpoint_dir, exist_ok=True)

if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))
else:
    device_ids = [torch.device("cpu")]

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Datasets ###########################

if "dataset" not in config:
    config["dataset"] = None

tokenizer = RobertaTokenizerFast.from_pretrained('/code/roberta-base', model_max_length = model_config["max_seq_len"])
tokenizer.model_max_length = model_config["max_seq_len"]
tokenizer.init_kwargs['model_max_length'] = model_config["max_seq_len"]
model_config["vocab_size"] = len(tokenizer.get_vocab())

if not args.use_data:
    dataset = CorpusDataset(folder_path = data_folder, file_json = "train.json", option = config["dataset"])
    # DataLoader 생성 전에 데이터를 먼저 로드하여 len이 0이 되지 않게 함 
    dataset.load_files()
    print(f"DEBUG: Dataset length is {len(dataset)}") # 이 값이 0이면 에러 발생
    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어있습니다. 경로와 전처리 파일을 확인하세요.")
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
    data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], shuffle = True, collate_fn = data_collator)
    pretrain_dataloader_iter = enumerate(data_loader)

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

if torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids = device_ids)
else:
    print("Running on CPU...")

# define wrapper for safe module access (works for CPU and DataParallel)
wrapper = model.module if hasattr(model, "module") else model

if "from_cp" in config:

    from_cp = config["from_cp"]
    checkpoint = torch.load(from_cp, map_location = 'cpu')

    cp_pos_encoding = checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'].data.numpy()
    cp_max_seq_len, embedding_dim = cp_pos_encoding.shape
    assert model_config["max_seq_len"] >= (cp_max_seq_len - 2)
    assert model_config["max_seq_len"] % (cp_max_seq_len - 2) == 0
    num_copy = model_config["max_seq_len"] // (cp_max_seq_len - 2)
    pos_encoding = np.concatenate([cp_pos_encoding[:2, :]] + [cp_pos_encoding[2:, :]] * num_copy, axis = 0)
    checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'] = torch.tensor(pos_encoding, dtype = torch.float)

    missing_keys, unexpected_keys = wrapper.load_state_dict(checkpoint['model_state_dict'], strict = False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    print("Model initialized", from_cp, flush = True)

elif "roberta_path" in config:

    roberta_path = config["roberta_path"]
    with open(roberta_path, "rb") as f:
        weights = pickle.load(f)
    weights = utils.convert_weight_keys(weights)

    assert weights["model.embeddings.position_embeddings.weight"].shape[0] == (model_config["max_seq_len"] + 2)

    for key in weights:
        weights[key] = torch.tensor(weights[key])

    missing_keys, unexpected_keys = wrapper.load_state_dict(weights, strict = False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    print("Model initialized from RoBERTa", flush = True)

else:
    print("Model randomly initialized", flush = True)

# 추가 : 모델을 한 번 forward 실행해서 각 NystromAttention 모듈에 실제로 들어간 Q, K, V, mask 값을 캡처하고 이를 npz 파일로 저장
def _capture_and_save_attention_intermediates(model, data_loader_iter, save_prefix):
    
    wrapper_local = model.module if hasattr(model, "module") else model

    # Find Nystrom attention modules
    nystrom_modules = []
    for name, m in wrapper_local.named_modules():
        if isinstance(m, NystromAttention):
            nystrom_modules.append((name, m))
    if len(nystrom_modules) == 0:
        print("No NystromAttention modules found in model — skipping attention intermediate capture.")
        return

    # Obtain one batch
    if data_loader_iter is None:
        print("No data loader iterator available (use_data=True?). Cannot capture batch automatically.")
        return

    try:
        _, batch = next(data_loader_iter)
    except Exception as e:
        print("Failed to fetch a batch for attention capture:", e)
        return

    # Move inputs to device consistent with model parameters
    # If module has parameters use their device, else default to cpu
    try:
        param0 = next(wrapper_local.parameters())
        device = param0.device
    except StopIteration:
        device = torch.device("cpu")

    inputs = {k: v.to(device) for k, v in batch.items()}

    # We will register forward hooks on each Nystrom module to capture the arguments they receive.
    saved_inputs = {}  # map module_name -> (Q_cpu, K_cpu, V_cpu, mask_cpu)

    def make_hook(mod_name):
        def hook_fn(module, hook_inputs, hook_output):
            # Expect hook_inputs to be tuple (Q, K, V, mask, ...) as in attention forward
            # 캠쳐하는 데이터
            try:
                Q_in = hook_inputs[0].detach().cpu()
                K_in = hook_inputs[1].detach().cpu()
                V_in = hook_inputs[2].detach().cpu()
                mask_in = hook_inputs[3].detach().cpu()
                saved_inputs[mod_name] = (Q_in, K_in, V_in, mask_in)
            except Exception as e:
                print(f"Warning: failed to capture inputs for module {mod_name}: {e}")
        return hook_fn

    handles = [] # 훅달기
    for name, mod in nystrom_modules:
        h = mod.register_forward_hook(make_hook(name))
        handles.append(h)

    # Run a single forward (eval, no grad) to trigger hooks
    wrapper_local.eval()
    with torch.no_grad():
        try:
            _ = wrapper_local(**inputs) if hasattr(wrapper_local, "__call__") else model(**inputs)
        except Exception as e:
            print("Warning: model forward during attention capture failed:", e)

    # remove hooks
    for h in handles:
        h.remove()

    # Re-run each Nystrom module directly with collect_intermediates=True and save debug npz
    for name, mod in nystrom_modules:
        if name not in saved_inputs:
            print(f"No saved inputs for module {name}; skipping.")
            continue
        Q_cpu, K_cpu, V_cpu, mask_cpu = saved_inputs[name]
        try:
            param0 = next(mod.parameters())
            mod_device = param0.device
        except StopIteration:
            mod_device = device
        Q = Q_cpu.to(mod_device)
        K = K_cpu.to(mod_device)
        V = V_cpu.to(mod_device)
        mask = mask_cpu.to(mod_device)
        saved_path = None
        try:
            # call forward with collect flag (attention implementation must support it)
            save_path = f"{save_prefix}-{name.replace('/', '_')}.npz"
            out = mod(Q, K, V, mask, collect_intermediates=True, save_path=save_path)
            saved_path = save_path
            print(f"Saved intermediates for module {name} to {save_path}")
        except TypeError:
            # attention module may not accept collect_intermediates (older implementation). fallback: try without save_path
            try:
                out = mod(Q, K, V, mask, collect_intermediates=True)
                if isinstance(out, tuple) and len(out) == 2: # 어텐션 재실행 후 중간값들까지 모두 저장
                    debug = out[1]
                    save_path = f"{save_prefix}-{name.replace('/', '_')}.npz"
                    np_save = {}
                    for k_debug, v_debug in debug.items():
                        if isinstance(v_debug, torch.Tensor):
                            np_save[k_debug] = v_debug.detach().cpu().numpy()
                        else:
                            try:
                                np_save[k_debug] = np.array(v_debug)
                            except Exception:
                                pass
                    np.savez_compressed(save_path, **np_save)
                    saved_path = save_path
                    print(f"Saved intermediates (fallback) for module {name} to {save_path}")
                else:
                    print(f"Module {name} returned unexpected output shape for collect_intermediates; skipping save.")
            except Exception as e:
                print(f"Could not collect intermediates for module {name}: {e}")

        # If we saved a debug npz, process it into witness npz
        if saved_path is not None:
            try:
                witness_out = saved_path.replace(".npz", ".witness.npz")
                process_attn_debug_npz( # 영지식 회로에 들어갈 수 있도록 어텐션 중간값들을 변환
                    saved_path,
                    witness_out,
                    scale_input = args.witness_scale_input,
                    frac_bits = args.witness_frac_bits,
                    k_segments = args.witness_k_segments,
                    limb_bits = args.witness_limb_bits,
                    v_scale = args.witness_v_scale,
                    inv_scale = args.witness_inv_scale,
                    tables_outdir = args.witness_tables_outdir,
                    multiplier = 1,
                )
            except Exception as e:
                print(f"Failed to run witness pipeline on {saved_path}: {e}")

    # restore model train mode
    wrapper_local.train()

########################### 앞에서 구현한 함수를 실제로 쓰는 곳 ###########################
if args.save_attn_npz is not None: #앞에서 
    try:
        if not args.use_data:
            capture_data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], shuffle = True, collate_fn = data_collator)
            capture_iter = iter(capture_data_loader)
            _capture_and_save_attention_intermediates(model, capture_iter, args.save_attn_npz)
        else:
            print("save_attn_npz requested but use_data=True; please run capture from a data-loading context or disable use_data.")
    except Exception as e:
        print("Failed to capture/save attention intermediates:", e)

########################### Continue with training loop(원래 코드로 돌아옴) ###########################

optimizer = torch.optim.AdamW( # 학습 옵티마이저
    model.parameters(),
    lr = pretraining_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = pretraining_config["learning_rate"],
    pct_start = pretraining_config["warmup"],
    anneal_strategy = "linear",
    epochs = pretraining_config["epoch"],
    steps_per_epoch = pretraining_config["batches_per_epoch"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

start_epoch = 0
inst_pass = 0
for epoch in reversed(range(pretraining_config["epoch"])):
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        # restore into wrapper safely
        wrapper.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        inst_pass = checkpoint["inst_pass"]
        print("Model restored", checkpoint_path)
        break

########################### Running Model ###########################

log_f = open(os.path.join(checkpoint_dir, "pretrain_output.log"), "a+")

accumu_steps = utils.compute_accumu_step(pretraining_config["batch_size"], len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

model.train()

init_t = time.time()
for epoch in range(start_epoch, pretraining_config["epoch"]):
    if args.use_data:
        with lz4.frame.open(f"/model/data/batches-{epoch:04}.plz4", "rb") as f:
            data_batches = pickle.load(f)
        print(f"/model/data/batches-{epoch:04}.plz4", flush = True)

    for batch_idx in range(pretraining_config["batches_per_epoch"]):

        t0 = time.time()

        optimizer.zero_grad()

        if not args.use_data:
            _, batch = next(pretrain_dataloader_iter)
        else:
            batch = data_batches[batch_idx]
            data_batches[batch_idx] = None

        inst_pass += list(batch.values())[0].size(0)
        summary = {}

        for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):
            outputs = model(**inputs)
            for key in outputs:
                outputs[key] = outputs[key].mean() * percent
            utils.backward(outputs["loss"], amp_scaler)
            utils.add_output_to_summary(outputs, summary)

        utils.optimizer_step(optimizer, lr_scheduler, amp_scaler)
        del batch

        t1 = time.time()

        summary["idx"] = epoch * pretraining_config["batches_per_epoch"] + batch_idx
        summary["batch_idx"] = batch_idx
        summary["epoch"] = epoch
        summary["time"] = round(t1 - t0, 4)
        summary["inst_pass"] = inst_pass
        summary["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)
        summary["time_since_start"] = round(time.time() - init_t, 4)

        log_f.write(json.dumps(summary, sort_keys = True) + "\n")

        if batch_idx % pretraining_config["batches_per_report"] == 0:
            print(json.dumps(summary, sort_keys = True), flush = True)
            log_f.flush()

    dump_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")

    save_model = wrapper
    torch.save({
        "model_state_dict":save_model.state_dict()
    }, dump_path.replace(".cp", ".model"))
    torch.save({
        "model_state_dict":save_model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "lr_scheduler_state_dict":lr_scheduler.state_dict(),
        "epoch":epoch,
        "inst_pass":inst_pass
    }, dump_path)
    print(f"Dump {dump_path}", flush = True)