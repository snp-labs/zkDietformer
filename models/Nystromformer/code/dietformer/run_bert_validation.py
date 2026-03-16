#!/usr/bin/env python3
import torch
import torch.nn as nn
import time
import os
import json
import math
import numpy as np
import argparse
import utils

# BERT/RoBERTa 환경을 위한 라이브러리
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from torch.utils.data import DataLoader

# 데이터셋 로드
from dataset import CorpusDataset
from bert_attention_pipeline import process_standard_attn_debug_npz
from bert_normalization_pipeline import process_standard_normalization_debug_npz

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=int, help="checkpoint index", default=None)
parser.add_argument("--batch_size", type=int, help="batch size", default=None)
parser.add_argument("--num_batch", type=int, help="number of batches", default=None)
parser.add_argument("--save_attn_npz", type=str, help="prefix path for saving npz", default=None)
args = parser.parse_args()

# 설정 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "..", "..", "bookcorpus-512", "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]

if args.batch_size is not None: pretraining_config["batch_size"] = args.batch_size
if args.num_batch is not None: pretraining_config["validate_batches_per_epoch"] = args.num_batch

# 경로 설정
relative_data_path = config["data_folder"].lstrip("/")
data_folder = os.path.abspath(os.path.join(base_dir, "..", "..", relative_data_path))
relative_cp_path = config["model_checkpoints"].lstrip("/")
checkpoint_dir = os.path.abspath(os.path.join(base_dir, "..", "..", relative_cp_path))

########################### Loading Dataset ###########################

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", model_max_length=model_config["max_seq_len"])
dataset = CorpusDataset(folder_path=data_folder, file_json="dev.json", files_per_batch=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
data_loader = DataLoader(dataset, batch_size=pretraining_config["batch_size"], collate_fn=data_collator)

########################### Loading BERT Model ###########################

device = torch.device("cpu") # 실험을 위해 CPU 설정 (GPU 사용 시 "cuda"로 변경)

# Baseline용 표준 RoBERTa 모델 로드
model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)

# 로그 설정
log_f_path = os.path.join(checkpoint_dir, "bert_validation_output.log")
os.makedirs(os.path.dirname(log_f_path), exist_ok=True)
log_f = open(log_f_path, "a+")

def _capture_and_process_bert_intermediates(model, current_batch, batch_idx, save_prefix):
    output_dir = "outputs_bert"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Hook을 통해 캡처할 타겟 모듈 설정
    target_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, RobertaSelfAttention)]

    # attention_mask를 배치에서 직접 추출합니다.
    # transformers 4.x에서 attention_mask는 keyword arg로 전달되므로
    # register_forward_hook의 hook_inputs에 포함되지 않습니다.
    # 따라서 배치의 원본 이진 마스크(1=valid, 0=pad)를 클로저로 캡처합니다.
    if 'attention_mask' in current_batch and current_batch['attention_mask'] is not None:
        # [batch, seq_len], int (1=valid, 0=padding)
        batch_mask_np = current_batch['attention_mask'].cpu().numpy().astype(np.int64)
    else:
        input_ids = current_batch['input_ids']
        batch_mask_np = np.ones((input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)

    saved_inputs = {}

    def make_hook(mod_name):
        def hook_fn(module, hook_inputs, hook_output):
            hidden_states = hook_inputs[0]
            num_heads = module.num_attention_heads
            head_dim = module.attention_head_size

            # 1. Q, K, V 프로젝션 후 [batch, heads, seq, head_dim] 형태로 변환
            with torch.no_grad():
                q_proj = module.query(hidden_states)  # [batch, seq, hidden]
                k_proj = module.key(hidden_states)
                v_proj = module.value(hidden_states)

            def transpose_for_scores(x):
                b, s, _ = x.shape
                return x.view(b, s, num_heads, head_dim).permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]

            q_heads = transpose_for_scores(q_proj)
            k_heads = transpose_for_scores(k_proj)
            v_heads = transpose_for_scores(v_proj)

            # 2. 마스크와 스케일 적용 (attention_bert.py와 동일한 규격)
            # batch_mask_np: [batch, seq_len] int (1/0) → 클로저로 캡처
            mask_t = torch.tensor(batch_mask_np, dtype=hidden_states.dtype, device=hidden_states.device)
            scale = 1.0 / math.sqrt(math.sqrt(head_dim))
            q_proc = q_heads * mask_t[:, None, :, None] * scale  # [batch, heads, seq, head_dim]
            k_proc = k_heads * mask_t[:, None, :, None] * scale

            # 3. 출력 X: context_layer [batch, seq, hidden] → [batch, heads, seq, head_dim]
            x_merged = hook_output[0] if isinstance(hook_output, tuple) else hook_output
            b, s, _ = x_merged.shape
            x_per_head = x_merged.view(b, s, num_heads, head_dim).permute(0, 2, 1, 3)

            # 4. 저장 (None 값 없이 모두 ndarray)
            saved_inputs[mod_name] = {
                "Q_proc": q_proc.detach().cpu().numpy(),   # [batch, heads, seq, head_dim]
                "K_proc": k_proc.detach().cpu().numpy(),
                "V":      v_heads.detach().cpu().numpy(),
                "mask":   batch_mask_np,                    # [batch, seq_len], int64
                "X":      x_per_head.detach().cpu().numpy(),
            }
        return hook_fn

    handles = [m.register_forward_hook(make_hook(n)) for n, m in target_modules]

    # 2. Forward Pass
    inputs = {k: v.to(device) for k, v in current_batch.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    for h in handles: h.remove()

    # 3. 데이터 저장 및 파이프라인 실행
    for name, data in saved_inputs.items():
        file_base = f"{save_prefix}-batch{batch_idx}-{name}.npz"
        debug_path = os.path.join(output_dir, file_base)

        np.savez_compressed(debug_path, **data)

        if os.path.exists(debug_path):
            witness_path = debug_path.replace(".npz", ".witness.npz")
            # BERT 전용 표준 어텐션 파이프라인 호출
            process_standard_attn_debug_npz(debug_path, witness_path, k_bits=22)
            print(f"[BERT Batch {batch_idx}] Generated Standard Attention Witness: {witness_path}")

########################### Main Loop ###########################

print(f"Starting BERT Baseline Validation for {pretraining_config['validate_batches_per_epoch']} batches...")
pretrain_dataloader_iter = enumerate(data_loader)

for epoch in range(1): # Baseline은 보통 학습된 모델 1회 평가
    epoch_summary = {}

    for batch_idx in range(pretraining_config["validate_batches_per_epoch"]):
        try:
            _, batch = next(pretrain_dataloader_iter)
        except StopIteration:
            break

        # 1. Witness 캡처 (BERT용)
        if args.save_attn_npz is not None:
            _capture_and_process_bert_intermediates(model, batch, batch_idx, args.save_attn_npz)

        # 2. 성능 측정
        t0 = time.time()
        summary = {}
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            # Loss 및 Accuracy 추출 (BERT 기본 출력 사용)
            summary["loss"] = outputs.loss.item() if outputs.loss is not None else 0.0
            # MLM 정확도 계산 생략(필요시 추가)
        t1 = time.time()

        summary["batch_idx"] = batch_idx
        summary["time"] = round(t1 - t0, 4)
        
        log_f.write(json.dumps(summary, sort_keys=True) + "\n")
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Time {summary['time']}s, Loss {summary['loss']:.4f}")
            log_f.flush()

log_f.close()
print("BERT Baseline Validation Completed.")