#!/usr/bin/env python3
from model_wrapper import ModelForMaskedLM
from dataset import CorpusDataset
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import utils
# (추가)
from attention_nystrom import NystromAttention
from attention_pipeline import process_attn_debug_npz
from normalization_pipeline import process_normalization_debug_npz
from model import Approx_RMSNorm

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type = int, help = "checkpoint", dest = "checkpoint", default = None)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", default = None)
parser.add_argument("--num_batch", type = int, help = "number of batches", dest = "num_batch", default = None)
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

# modify
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..", "..", "bookcorpus-512", "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]

relative_data_path = config["data_folder"].lstrip("/")
data_folder = os.path.abspath(os.path.join(base_dir, "..", "..", relative_data_path))

relative_cp_path = config["model_checkpoints"].lstrip("/")
checkpoint_dir = os.path.abspath(os.path.join(base_dir, "..", "..", relative_cp_path))

print("Using data folder:", data_folder)
print("Using checkpoint folder:", checkpoint_dir)

if args.batch_size is not None:
    pretraining_config["batch_size"] = args.batch_size

if args.num_batch is not None:
    pretraining_config["validate_batches_per_epoch"] = args.num_batch

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Dataset ###########################

tokenizer_path = os.path.join(base_dir, "..", "roberta-base")
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, model_max_length=model_config["max_seq_len"])
# tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())

if "dataset" not in config:
    config["dataset"] = None

dataset = CorpusDataset(folder_path = data_folder, file_json = "dev.json", files_per_batch = 128, option = config["dataset"])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], collate_fn = data_collator)
pretrain_dataloader_iter = enumerate(data_loader)

########################### Loading Model ###########################
device = torch.device("cpu")

model = ModelForMaskedLM(model_config)
print(model)

# model = model.cuda()
# model = nn.DataParallel(model, device_ids = device_ids)
model = model.to(device)
########################### Running Model ###########################
# modify
# accumu_steps = utils.compute_accumu_step(pretraining_config["batch_size"], len(device_ids), gpu_config["inst_per_gpu"])
# print("accumu_steps", accumu_steps)
accumu_steps = 1
print("accumu_steps (CPU mode):", accumu_steps)

# === Robust log handling: replace the previous fragile utils.read_data usage ===
if args.checkpoint is None:
    log_f_path = os.path.join(checkpoint_dir, "validation_output.log")
    # ensure the file exists and open for append
    try:
        open(log_f_path, "a").close()
    except Exception:
        # if directory doesn't exist yet, try to create it
        os.makedirs(os.path.dirname(log_f_path) or ".", exist_ok=True)
        open(log_f_path, "a").close()

    log_f = open(log_f_path, "a+")
    start_epoch = 0
    try:
        log = utils.read_data(log_f_path)
        if log is None:
            start_epoch = 0
        else:
            # handle several possible types (pandas DataFrame, dict, numpy arrays, etc.)
            try:
                if hasattr(log, "columns") and "epoch" in log.columns:
                    max_epoch = log["epoch"].max()
                    start_epoch = int(max_epoch)
                elif isinstance(log, dict) and "epoch" in log:
                    # dict likely contains a list/array under "epoch" key
                    try:
                        start_epoch = int(np.max(log["epoch"]))
                    except Exception:
                        # fallback: last element
                        ep_list = list(log["epoch"])
                        start_epoch = int(ep_list[-1]) if len(ep_list) > 0 else 0
                else:
                    # last resort: try to extract numeric epoch values
                    epoch_vals = None
                    if isinstance(log, (list, tuple)) and len(log) > 0:
                        # maybe a list of dicts
                        try:
                            epoch_vals = [item.get("epoch") for item in log if isinstance(item, dict) and "epoch" in item]
                        except Exception:
                            epoch_vals = None
                    if epoch_vals:
                        start_epoch = int(np.max(epoch_vals))
                    else:
                        start_epoch = 0
            except Exception:
                start_epoch = 0
    except Exception as e:
        print("Failed to read existing validation log, starting from epoch 0:", e)
        start_epoch = 0

    end_epoch = pretraining_config["epoch"]
else:
    log_f_path = os.path.join(checkpoint_dir, f"validation_output-checkpoint-{args.checkpoint}.log")
    log_f = open(log_f_path, "w")
    start_epoch = args.checkpoint
    end_epoch = args.checkpoint + 1
# === end robust log handling ===

print(f"Starts from Epoch: {start_epoch}, End at Epoch: {end_epoch}", flush = True)

def _capture_and_process_all_intermediates(model, data_loader, save_prefix):
    """
    NystromAttention과 Approx_RMSNorm의 중간값을 모두 캡처하고 
    각각의 Witness 파이프라인을 실행합니다.
    """
    wrapper = model.module if hasattr(model, "module") else model
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 캡처 플래그 활성화
    for name, m in wrapper.named_modules():
        if hasattr(m, "capture_intermediates"):
            m.capture_intermediates = True
            m.intermediates = []

    # 2. 데이터 한 배치 준비
    try:
        capture_iter = enumerate(data_loader)
        _, batch = next(capture_iter)
        param0 = next(wrapper.parameters())
        device_local = param0.device
        inputs = {k: v.to(device_local) for k, v in batch.items()}
    except Exception as e:
        print(f"Failed to prepare batch for capture: {e}")
        return

    # 3. Attention 입력을 위한 Hook 설정
    saved_attn_inputs = {}
    nystrom_modules = [(n, m) for n, m in wrapper.named_modules() if isinstance(m, NystromAttention)]
    
    def make_attn_hook(mod_name):
        def hook_fn(module, hook_inputs, hook_output):
            try:
                # Q, K, V, mask 순서
                saved_attn_inputs[mod_name] = [x.detach().cpu() for x in hook_inputs[:4]]
            except Exception as e:
                print(f"Hook failed for {mod_name}: {e}")
        return hook_fn

    handles = [mod.register_forward_hook(make_attn_hook(name)) for name, mod in nystrom_modules]

    # 4. Forward Pass 실행 (모든 캡처 트리거)
    wrapper.eval()
    with torch.no_grad():
        try:
            _ = wrapper(**inputs)
        except Exception as e:
            print(f"Model forward failed: {e}")

    for h in handles: h.remove()

    # 5. 수집된 데이터 처리 및 Witness 생성
    for name, m in wrapper.named_modules():
        # (A) Nystrom Attention 처리
        if isinstance(m, NystromAttention) and name in saved_attn_inputs:
            Q, K, V, mask = [x.to(device_local) for x in saved_attn_inputs[name]]
            file_base = f"{save_prefix}-{name}.npz"
            debug_path = os.path.join(output_dir, file_base)
            
            # collect_intermediates=True로 재실행하여 내부 값 저장
            m(Q, K, V, mask, collect_intermediates=True, save_path=debug_path)
            
            if os.path.exists(debug_path):
                witness_path = debug_path.replace(".npz", ".witness.npz")
                process_attn_debug_npz(debug_path, witness_path, k_bits=22)
                print(f"Generated Attention Witness: {witness_path}")

        # (B) Approx_RMSNorm 처리
        elif isinstance(m, Approx_RMSNorm):
            if hasattr(m, "intermediates") and len(m.intermediates) > 0:
                inter = m.intermediates[-1]
                file_base = f"{save_prefix}-{name}.npz"
                debug_path = os.path.join(output_dir, file_base)
                
                # 원본 데이터 저장
                np.savez_compressed(debug_path, **inter)
                
                # Witness 파이프라인 실행
                witness_path = debug_path.replace(".npz", ".witness.npz")
                process_normalization_debug_npz(debug_path, witness_path, k_bits=22)
                print(f"Generated RMSNorm Witness: {witness_path}")

    # 6. 캡처 비활성화
    for name, m in wrapper.named_modules():
        if hasattr(m, "capture_intermediates"):
            m.capture_intermediates = False

########################### Main Loop ###########################

for epoch in range(start_epoch, end_epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.model")
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Model restored", checkpoint_path, flush = True)

    # 중간값 캡처 및 파이프라인 실행
    if args.save_attn_npz is not None:
        try:
            _capture_and_process_all_intermediates(model, data_loader, args.save_attn_npz)
        except Exception as e:
            print(f"Failed during intermediate processing: {e}")

    model.eval()
    epoch_summary = {}

    with torch.no_grad():
        for batch_idx in range(pretraining_config["validate_batches_per_epoch"]):
            t0 = time.time()
            _, batch = next(pretrain_dataloader_iter)
            summary = {}
            for percent, inputs in utils.partition_inputs(batch, 1, True):
                inputs = {k: v.to(device) for k, v in inputs.items()} 
                outputs = model(**inputs)
                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                utils.add_output_to_summary(outputs, summary)
            t1 = time.time()
            # [이후 로그 기록 로직 원본 유지]
            for key in summary:
                if key not in epoch_summary: epoch_summary[key] = []
                epoch_summary[key].append(summary[key])
            summary["idx"] = epoch * pretraining_config["validate_batches_per_epoch"] + batch_idx
            summary["batch_idx"] = batch_idx
            summary["epoch"] = epoch
            summary["time"] = round(t1 - t0, 4)
            log_f.write(json.dumps(summary, sort_keys = True) + "\n")
            if batch_idx % pretraining_config["batches_per_report"] == 0:
                print(json.dumps(summary, sort_keys = True), flush = True)
                log_f.flush()

    for key in epoch_summary:
        epoch_summary[key] = np.mean(epoch_summary[key])
    log_f.write(json.dumps(epoch_summary, sort_keys = True) + "\n")
    log_f.flush()
    print(json.dumps(epoch_summary, sort_keys = True), flush = True)