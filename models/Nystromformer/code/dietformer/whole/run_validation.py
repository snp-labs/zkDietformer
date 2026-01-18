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
from attn_witness_pipeline import process_attn_debug_npz

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type = int, help = "checkpoint", dest = "checkpoint", default = None)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", default = None)
parser.add_argument("--num_batch", type = int, help = "number of batches", dest = "num_batch", default = None)
# new options: save attention intermediates and witness pipeline parameters
parser.add_argument("--save_attn_npz", type = str, help = "prefix path to save attention intermediates npz (optional). Example: /tmp/attn_debug", default = None)
parser.add_argument("--witness_v_scale", type=int, default=1)
parser.add_argument("--witness_inv_scale", type=int, default=1)
parser.add_argument("--witness_scale_input", type=int, default=(1<<8))
parser.add_argument("--witness_frac_bits", type=int, default=16)
parser.add_argument("--witness_k_segments", type=int, default=2)
parser.add_argument("--witness_limb_bits", type=int, default=2)
parser.add_argument("--witness_tables_outdir", type=str, default="tmp_tables")
args = parser.parse_args()

# modify
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..", "bookcorpus-512", "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
# modify
data_folder = os.path.join(base_dir, "..", config["data_folder"].lstrip("/"))
data_folder = os.path.normpath(data_folder)

checkpoint_dir = os.path.join(base_dir, "..", config["model_checkpoints"].lstrip("/"))
checkpoint_dir = os.path.normpath(checkpoint_dir)

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

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
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

if args.checkpoint is None:
    log_f_path = os.path.join(checkpoint_dir, "validation_output.log")
    log_f = open(log_f_path, "a+")
    try:
        log = utils.read_data(log_f_path)
        start_epoch = int(log["epoch"].max().item())
    except Exception as e:
        print(e)
        start_epoch = 0
    end_epoch = pretraining_config["epoch"]
else:
    log_f_path = os.path.join(checkpoint_dir, f"validation_output-checkpoint-{args.checkpoint}.log")
    log_f = open(log_f_path, "w")
    start_epoch = args.checkpoint
    end_epoch = args.checkpoint + 1
print(f"Starts from Epoch: {start_epoch}, End at Epoch: {end_epoch}", flush = True)

def _capture_and_save_attention_intermediates(model, data_loader, save_prefix):
    # run_pretrain.py쪽과 비슷함.
    wrapper = model.module if hasattr(model, "module") else model

    # Find Nystrom attention modules
    nystrom_modules = []
    for name, m in wrapper.named_modules():
        if isinstance(m, NystromAttention):
            nystrom_modules.append((name, m))
    if len(nystrom_modules) == 0:
        print("No NystromAttention modules found in model — skipping attention intermediate capture.")
        return

    # Obtain one batch (create an iterator so we don't consume outer iterator)
    try:
        capture_iter = iter(data_loader)
        _, batch = next(capture_iter)
    except Exception as e:
        print("Failed to fetch a batch for attention capture:", e)
        return

    # Move inputs to device consistent with model parameters (safe extraction)
    try:
        param0 = next(wrapper.parameters())
        device_local = param0.device
    except StopIteration:
        device_local = torch.device("cpu")
    inputs = {k: v.to(device_local) for k, v in batch.items()}

    # register forward hooks on each Nystrom module to capture the arguments they receive.
    saved_inputs = {}  # map module_name -> (Q_cpu, K_cpu, V_cpu, mask_cpu)

    def make_hook(mod_name):
        def hook_fn(module, hook_inputs, hook_output):
            # Expect hook_inputs to be tuple (Q, K, V, mask, ...) as in attention forward
            try:
                Q_in = hook_inputs[0].detach().cpu()
                K_in = hook_inputs[1].detach().cpu()
                V_in = hook_inputs[2].detach().cpu()
                mask_in = hook_inputs[3].detach().cpu()
                saved_inputs[mod_name] = (Q_in, K_in, V_in, mask_in)
            except Exception as e:
                print(f"Warning: failed to capture inputs for module {mod_name}: {e}")
        return hook_fn

    handles = []
    for name, mod in nystrom_modules:
        h = mod.register_forward_hook(make_hook(name))
        handles.append(h)

    # Run a single forward (eval, no grad) to trigger hooks - call wrapper to ensure hooks trigger
    wrapper.eval()
    with torch.no_grad():
        try:
            _ = wrapper(**inputs)
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
        mod_device = next(mod.parameters()).device if any(p.requires_grad for p in mod.parameters()) else device_local
        Q = Q_cpu.to(mod_device)
        K = K_cpu.to(mod_device)
        V = V_cpu.to(mod_device)
        mask = mask_cpu.to(mod_device)
        saved_path = None
        try:
            save_path = f"{save_prefix}-{name.replace('/', '_')}.npz"
            out = mod(Q, K, V, mask, collect_intermediates=True, save_path=save_path)
            saved_path = save_path
            print(f"Saved intermediates for module {name} to {save_path}")
        except TypeError:
            # fallback if module doesn't accept collect_intermediates or save_path
            try:
                out = mod(Q, K, V, mask, collect_intermediates=True)
                if isinstance(out, tuple) and len(out) == 2:
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
                process_attn_debug_npz(
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
                    r = 3,
                )
            except Exception as e:
                print(f"Failed to run witness pipeline on {saved_path}: {e}")

    # restore model train mode
    wrapper.train()

for epoch in range(start_epoch, end_epoch):

    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.model")
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    # modify
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model restored", checkpoint_path, flush = True)

    # (추가) capture attention intermediates for this restored model before running validation
    if args.save_attn_npz is not None:
        try:
            _capture_and_save_attention_intermediates(model, data_loader, args.save_attn_npz)
        except Exception as e:
            print("Failed to capture/save attention intermediates during validation:", e)

    model.eval()

    epoch_summary = {}

    with torch.no_grad():
        for batch_idx in range(pretraining_config["validate_batches_per_epoch"]):

            t0 = time.time()
            _, batch = next(pretrain_dataloader_iter)
            summary = {}
            for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):
                # modify
                inputs = {k: v.to(device) for k, v in inputs.items()} 
                outputs = model(**inputs)
                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                utils.add_output_to_summary(outputs, summary)
            t1 = time.time()

            for key in summary:
                if key not in epoch_summary:
                    epoch_summary[key] = []
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