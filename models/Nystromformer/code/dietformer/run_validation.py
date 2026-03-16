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
from model import Approx_RMSNorm, Approx_Linear

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

tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base", 
    model_max_length=model_config["max_seq_len"]
)
model_config["vocab_size"] = len(tokenizer.get_vocab())

if "dataset" not in config:
    config["dataset"] = None

dataset = CorpusDataset(folder_path = data_folder, file_json = "dev.json", files_per_batch = 128, option = config["dataset"])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], collate_fn = data_collator)

########################### Loading Model ###########################
device = torch.device("cpu")

model = ModelForMaskedLM(model_config)
model = model.to(device)

accumu_steps = 1
print("accumu_steps (CPU mode):", accumu_steps)

# === Robust log handling ===
if args.checkpoint is None:
    log_f_path = os.path.join(checkpoint_dir, "validation_output.log")
    try:
        open(log_f_path, "a").close()
    except Exception:
        os.makedirs(os.path.dirname(log_f_path) or ".", exist_ok=True)
        open(log_f_path, "a").close()

    log_f = open(log_f_path, "a+")
    start_epoch = 0
    try:
        log = utils.read_data(log_f_path)
        if log is not None:
            if hasattr(log, "columns") and "epoch" in log.columns:
                start_epoch = int(log["epoch"].max())
            elif isinstance(log, dict) and "epoch" in log:
                start_epoch = int(np.max(log["epoch"]))
    except Exception as e:
        print("Failed to read existing validation log, starting from epoch 0:", e)
        start_epoch = 0

    end_epoch = pretraining_config["epoch"]
else:
    log_f_path = os.path.join(checkpoint_dir, f"validation_output-checkpoint-{args.checkpoint}.log")
    log_f = open(log_f_path, "w")
    start_epoch = args.checkpoint
    end_epoch = args.checkpoint + 1

print(f"Starts from Epoch: {start_epoch}, End at Epoch: {end_epoch}", flush = True)

def _capture_and_process_all_intermediates(model, current_batch, batch_idx, save_prefix):
    """
    현재 배치(current_batch)에 대해 중간값을 캡처하고 
    batch_idx를 포함한 파일명으로 저장합니다.
    """
    wrapper = model.module if hasattr(model, "module") else model
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 캡처 플래그 활성화
    for name, m in wrapper.named_modules():
        if hasattr(m, "capture_intermediates"):
            m.capture_intermediates = True
            m.intermediates = []

    # 2. 메인 루프에서 전달받은 현재 배치 사용 (next 호출 제거)
    param0 = next(wrapper.parameters())
    device_local = param0.device
    inputs = {k: v.to(device_local) for k, v in current_batch.items()}

    # 3. Forward Hook 설정
    saved_attn_inputs = {}
    def make_attn_hook(mod_name):
        def hook_fn(module, hook_inputs, hook_output):
            saved_attn_inputs[mod_name] = [x.detach().cpu() for x in hook_inputs[:4]]
        return hook_fn

    nystrom_modules = [(n, m) for n, m in wrapper.named_modules() if isinstance(m, NystromAttention)]
    handles = [mod.register_forward_hook(make_attn_hook(name)) for name, mod in nystrom_modules]

    # 4. Forward Pass 실행
    wrapper.eval()
    with torch.no_grad():
        _ = wrapper(**inputs)

    for h in handles: h.remove()

    # 5. 수집된 데이터 처리 및 Witness 생성
    for name, m in wrapper.named_modules():
        if isinstance(m, NystromAttention) and name in saved_attn_inputs:
            Q, K, V, mask = [x.to(device_local) for x in saved_attn_inputs[name]]
            file_base = f"{save_prefix}-batch{batch_idx}-{name}.npz"
            debug_path = os.path.join(output_dir, file_base)
            
            m(Q, K, V, mask, collect_intermediates=True, save_path=debug_path)
            
            if os.path.exists(debug_path):
                witness_path = debug_path.replace(".npz", ".witness.npz")
                process_attn_debug_npz(debug_path, witness_path, k_bits=22)
                print(f"[Batch {batch_idx}] Generated Attention Witness: {witness_path}")

        elif isinstance(m, Approx_RMSNorm) and "norm1" in name:
            block_name = name.replace(".norm1", "")
            named_mods = dict(wrapper.named_modules())
            ff1_mod, act_mod, ff2_mod, norm2_mod = [named_mods.get(f"{block_name}.{x}") for x in ["ff1", "act", "ff2", "norm2"]]
            
            if all(hasattr(mod, "intermediates") and len(mod.intermediates) > 0 for mod in [m, ff1_mod, act_mod, ff2_mod, norm2_mod] if mod is not None):
                file_base = f"{save_prefix}-batch{batch_idx}-{block_name}.norm_ff_combined.npz"
                debug_path = os.path.join(output_dir, file_base)
                
                combined_data = {
                    "rms_input": m.intermediates[-1]["rms_input"],
                    "ms": m.intermediates[-1]["ms"],
                    "rms_weight": m.intermediates[-1]["rms_weight"],
                    "rms_output": m.intermediates[-1]["rms_output"],
                    "linear_weight": ff1_mod.intermediates[-1]["linear_weight"],
                    "linear_bias": ff1_mod.intermediates[-1]["linear_bias"],
                    "linear_output": ff1_mod.intermediates[-1]["linear_output"],
                    "gelu_output": act_mod.intermediates[-1]["gelu_output"],
                    "linear_weight_ff2": ff2_mod.intermediates[-1]["linear_weight"],
                    "linear_bias_ff2": ff2_mod.intermediates[-1].get("linear_bias"),
                    "linear_output_ff2": ff2_mod.intermediates[-1]["linear_output"],
                    "rms_input_rms2": norm2_mod.intermediates[-1]["rms_input"],
                    "ms_rms2": norm2_mod.intermediates[-1]["ms"],
                    "rms_weight_rms2": norm2_mod.intermediates[-1]["rms_weight"],
                    "rms_output_rms2": norm2_mod.intermediates[-1]["rms_output"],
                }
                np.savez_compressed(debug_path, **combined_data)
                witness_path = debug_path.replace(".npz", ".witness.npz")
                process_normalization_debug_npz(debug_path, witness_path, k_bits=22)
                print(f"[Batch {batch_idx}] Generated MLP Witness: {witness_path}")

    for name, m in wrapper.named_modules():
        if hasattr(m, "capture_intermediates"):
            m.capture_intermediates = False

########################### 통합된 Main Loop ###########################

for epoch in range(start_epoch, end_epoch): 
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.model")
    if not os.path.exists(checkpoint_path): continue
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'], strict=False)
    print(f"Model restored from {checkpoint_path}")

    # 데이터 로더 초기화
    pretrain_dataloader_iter = enumerate(data_loader)
    model.eval()
    epoch_summary = {}

    # 단일 루프로 통합: 캡처와 성능 측정을 하나의 배치에 대해 순차 실행
    for batch_idx in range(pretraining_config["validate_batches_per_epoch"]):
        try:
            _, batch = next(pretrain_dataloader_iter)
        except StopIteration:
            break

        # 1. 캡처 수행 (Witness 생성)
        if args.save_attn_npz is not None:
            _capture_and_process_all_intermediates(model, batch, batch_idx, args.save_attn_npz)

        # 2. 성능 측정 및 로그 기록 (동일한 batch 사용)
        t0 = time.time()
        summary = {}
        with torch.no_grad():
            for percent, inputs in utils.partition_inputs(batch, 1, True):
                inputs = {k: v.to(device) for k, v in inputs.items()} 
                outputs = model(**inputs)
                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                utils.add_output_to_summary(outputs, summary)
        t1 = time.time()

        # 결과 정리
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

    # 에폭 요약 기록
    for key in epoch_summary:
        epoch_summary[key] = np.mean(epoch_summary[key])
    log_f.write(json.dumps(epoch_summary, sort_keys = True) + "\n")
    log_f.flush()
    print(json.dumps(epoch_summary, sort_keys = True), flush = True)