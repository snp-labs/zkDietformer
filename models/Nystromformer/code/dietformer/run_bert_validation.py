#!/usr/bin/env python3
"""
BERT-base-uncased baseline evaluation — fixture 기반 입력 + GELU swap.

작업 1C 변경:
- CorpusDataset 의존 제거 → evaluation_fixtures.SAMPLE_TEXTS 사용
  (이로써 RoBERTa BPE pickle vs BERT vocab 토크나이저 불일치 버그 해결)
- BertTokenizerFast 로 on-the-fly 인코딩 (padding=max_length, max_length=510)
- random seed 42 (torch + numpy + random)
- replace_activation_with_sigmoid_gelu(model, "bert") — 회로 lookup 정합성
- hook 메커니즘 / npz 키 (ln_*) 은 Step C 결과 그대로 보존

이전 단계와의 정렬:
- Step B: RoBERTa → BERT 교체 완료
- Step C: LayerNorm 전환, ln_* 키 컨벤션
- Phase 4: layernorm_table.py VAR_SCALE=256 (이미 적용)

회로 verify 는 Phase 5 (Rust 가젯 enforce_layernorm 추가) 후에야 통과 가능.
"""
import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn

from transformers import BertForMaskedLM, BertTokenizerFast
from transformers.models.bert.modeling_bert import BertSelfAttention

from evaluation_fixtures import (
    SAMPLE_TEXTS, NUM_SAMPLES, MAX_SEQ_LEN, assert_fixture_invariants,
)
from gelu_swap import replace_activation_with_sigmoid_gelu, SigmoidGELU
from bert_attention_pipeline import process_standard_attn_debug_npz, load_small_bert
from bert_normalization_pipeline import process_standard_normalization_debug_npz
from layernorm_table import EPS_BERT


# ── 재현성 seed ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── argparse ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=int, default=None,
    help="(deprecated, 무시됨) HF 사전학습 모델을 사용하므로 체크포인트 인덱스 불필요.",
)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_batch", type=int, default=1)
parser.add_argument(
    "--save_attn_npz", type=str, default=None,
    help="prefix string for npz files. 예: 'run1' 또는 'outputs_bert'",
)
parser.add_argument(
    "--hidden_size", type=int, default=None,
    help="Random init 모델 hidden_size. 지정 시 pretrained 대신 random init 사용. "
         "dim sweep 예: 256, 384, 512.",
)
parser.add_argument(
    "--num_attention_heads", type=int, default=None,
    help="Random init 모델 head 수. hidden_size 를 나눌 수 있어야 함. "
         "미지정 시 자동 결정 (hidden_size // 64, 최소 4).",
)
parser.add_argument(
    "--num_hidden_layers", type=int, default=None,
    help="Random init 모델 layer 수. 미지정 시 함수 default (4) 사용.",
)
args = parser.parse_args()

# fixture 무결성 self-check
assert_fixture_invariants()

total_needed = args.num_batch * args.batch_size
if total_needed > NUM_SAMPLES:
    raise ValueError(
        f"requested {total_needed} samples (num_batch={args.num_batch} × "
        f"batch_size={args.batch_size}) but fixture has only {NUM_SAMPLES}. "
        f"increase NUM_SAMPLES in evaluation_fixtures.py 또는 명령 인자를 줄이세요."
    )

# ── 출력 디렉터리 자동 분기 ────────────────────────────────────────────────
if args.save_attn_npz:
    output_dir_name = args.save_attn_npz
elif args.hidden_size is not None:
    n_layers = args.num_hidden_layers or 4
    output_dir_name = f"outputs_n{MAX_SEQ_LEN}_L{n_layers}_h{args.hidden_size}_bert"
else:
    output_dir_name = f"outputs_n{MAX_SEQ_LEN}_bert"
save_prefix = output_dir_name

# ── tokenizer + model ──────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased", model_max_length=MAX_SEQ_LEN,
)

device = torch.device("cpu")  # 실험을 위해 CPU. GPU는 "cuda"로 변경.

# ── 모델 로드: --hidden_size 지정 시 random init, 미지정 시 pretrained ───
if args.hidden_size is not None:
    n_heads = args.num_attention_heads
    if n_heads is None:
        n_heads = max(4, args.hidden_size // 64)
    model = load_small_bert(
        hidden_size=args.hidden_size,
        num_attention_heads=n_heads,
        num_hidden_layers=args.num_hidden_layers or 4,
        seq_len=MAX_SEQ_LEN,
    )
    print(f"[BERT] RANDOM INIT — hidden_size={args.hidden_size}, heads={n_heads}")
else:
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.to(device).eval()  # 중요: dropout 비활성화로 결정성 보장

# GELU swap (회로 lookup 정합성 — gelu_table.py 가 σ(1.702x) 가정)
n_swapped = replace_activation_with_sigmoid_gelu(model, "bert")
expected_layers = model.config.num_hidden_layers
assert n_swapped == expected_layers, (
    f"GELU swap incomplete: {n_swapped}/{expected_layers}"
)
print(f"[BERT] swapped {n_swapped}/{expected_layers} layer activations to SigmoidGELU")
print(f"[BERT] hidden_size={model.config.hidden_size}, "
      f"num_hidden_layers={model.config.num_hidden_layers}, "
      f"num_attention_heads={model.config.num_attention_heads}")

# ── batch 생성: fixture → on-the-fly tokenization ──────────────────────────
def _make_batches() -> list[dict]:
    """SAMPLE_TEXTS 의 앞 total_needed 개 를 batch_size 단위로 묶음.

    각 batch shape = [batch_size, MAX_SEQ_LEN] (padding=max_length).
    """
    batches = []
    texts = SAMPLE_TEXTS[:total_needed]
    for i in range(0, len(texts), args.batch_size):
        chunk = list(texts[i : i + args.batch_size])
        enc = tokenizer(
            chunk,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        batches.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        })
    return batches


# ── 출력 디렉터리 + 로그 ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, output_dir_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_f_path = os.path.join(OUTPUT_DIR, "bert_validation_output.log")
log_f = open(log_f_path, "a+")


# ── capture 함수 (Step C 그대로, ln_* 키) ──────────────────────────────────
def _capture_and_process_bert_intermediates(model, current_batch, batch_idx, save_prefix, output_dir_abs):
    os.makedirs(output_dir_abs, exist_ok=True)

    # 1. Attention Hook 타겟 모듈 설정
    target_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, BertSelfAttention)]

    # attention_mask: transformers 4.x에서 keyword arg이므로 클로저로 캡처
    if 'attention_mask' in current_batch and current_batch['attention_mask'] is not None:
        batch_mask_np = current_batch['attention_mask'].cpu().numpy().astype(np.int64)
    else:
        input_ids = current_batch['input_ids']
        batch_mask_np = np.ones((input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)

    saved_inputs = {}
    saved_norm = {}  # key: "layer_{i}" → dict of normalization intermediates

    # ── Attention Hook ──────────────────────────────────────────────────────
    def make_hook(mod_name):
        def hook_fn(module, hook_inputs, hook_output):
            hidden_states = hook_inputs[0]
            num_heads = module.num_attention_heads
            head_dim = module.attention_head_size

            with torch.no_grad():
                q_proj = module.query(hidden_states)
                k_proj = module.key(hidden_states)
                v_proj = module.value(hidden_states)

            def transpose_for_scores(x):
                b, s, _ = x.shape
                return x.view(b, s, num_heads, head_dim).permute(0, 2, 1, 3)

            q_heads = transpose_for_scores(q_proj)
            k_heads = transpose_for_scores(k_proj)
            v_heads = transpose_for_scores(v_proj)

            mask_t = torch.tensor(batch_mask_np, dtype=hidden_states.dtype, device=hidden_states.device)
            scale = 1.0 / math.sqrt(math.sqrt(head_dim))
            q_proc = q_heads * mask_t[:, None, :, None] * scale
            k_proc = k_heads * mask_t[:, None, :, None] * scale

            x_merged = hook_output[0] if isinstance(hook_output, tuple) else hook_output
            b, s, _ = x_merged.shape
            x_per_head = x_merged.view(b, s, num_heads, head_dim).permute(0, 2, 1, 3)

            saved_inputs[mod_name] = {
                "Q_proc": q_proc.detach().cpu().numpy(),
                "K_proc": k_proc.detach().cpu().numpy(),
                "V":      v_heads.detach().cpu().numpy(),
                "mask":   batch_mask_np,
                "X":      x_per_head.detach().cpu().numpy(),
            }
        return hook_fn

    # ── Normalization / FFN Hooks (Step C 결과 — ln_* 키) ────────────────────
    def make_norm_hooks(layer_key):
        # Hook 1: post-attention LayerNorm (BertSelfOutput.LayerNorm)
        def hook_attn_layernorm(module, inputs, output):
            x = inputs[0].detach().cpu()
            saved_norm.setdefault(layer_key, {}).update({
                "ln_input":  x.numpy(),
                "mean":      x.mean(dim=-1, keepdim=True).numpy(),
                "var":       x.var(dim=-1, unbiased=False, keepdim=True).numpy(),
                "ln_weight": module.weight.detach().cpu().numpy(),
                "ln_bias":   module.bias.detach().cpu().numpy(),
                "ln_output": output.detach().cpu().numpy(),
            })

        # Hook 2: FFN up-projection dense (BertIntermediate.dense, before GeLU)
        def hook_ff1_dense(module, _inputs, output):
            entry = {
                "linear_weight": module.weight.detach().cpu().numpy(),
                "linear_output": output.detach().cpu().numpy(),
            }
            if module.bias is not None:
                entry["linear_bias"] = module.bias.detach().cpu().numpy()
            saved_norm.setdefault(layer_key, {}).update(entry)

        # Hook 3: full BertIntermediate output (after GeLU activation)
        def hook_intermediate(_module, _inputs, output):
            saved_norm.setdefault(layer_key, {})["gelu_output"] = output.detach().cpu().numpy()

        # Hook 4: FFN down-projection dense (BertOutput.dense, before residual+LayerNorm)
        def hook_ff2_dense(module, _inputs, output):
            entry = {
                "linear_weight_ff2": module.weight.detach().cpu().numpy(),
                "linear_output_ff2": output.detach().cpu().numpy(),
            }
            if module.bias is not None:
                entry["linear_bias_ff2"] = module.bias.detach().cpu().numpy()
            saved_norm.setdefault(layer_key, {}).update(entry)

        # Hook 5: post-FFN LayerNorm (BertOutput.LayerNorm)
        def hook_ffn_layernorm(module, inputs, output):
            x = inputs[0].detach().cpu()
            saved_norm.setdefault(layer_key, {}).update({
                "ln_input_2":  x.numpy(),
                "mean_2":      x.mean(dim=-1, keepdim=True).numpy(),
                "var_2":       x.var(dim=-1, unbiased=False, keepdim=True).numpy(),
                "ln_weight_2": module.weight.detach().cpu().numpy(),
                "ln_bias_2":   module.bias.detach().cpu().numpy(),
                "ln_output_2": output.detach().cpu().numpy(),
            })

        return hook_attn_layernorm, hook_ff1_dense, hook_intermediate, hook_ff2_dense, hook_ffn_layernorm

    # Attention hook 등록
    handles = [m.register_forward_hook(make_hook(n)) for n, m in target_modules]

    # Normalization hook 등록 (모든 레이어에 대해)
    norm_handles = []
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        layer = model.bert.encoder.layer[i]
        layer_key = f"layer_{i}"
        h_aln, h_ff1, h_int, h_ff2, h_fln = make_norm_hooks(layer_key)
        norm_handles.extend([
            layer.attention.output.LayerNorm.register_forward_hook(h_aln),
            layer.intermediate.dense.register_forward_hook(h_ff1),
            layer.intermediate.register_forward_hook(h_int),
            layer.output.dense.register_forward_hook(h_ff2),
            layer.output.LayerNorm.register_forward_hook(h_fln),
        ])

    # 2. Forward pass — 단일 forward 로 12 layer × 모든 hook 캡처
    model.eval()
    with torch.no_grad():
        model(**current_batch)

    for h in handles:
        h.remove()
    for h in norm_handles:
        h.remove()

    # 3. 데이터 저장 및 파이프라인 실행
    for name, data in saved_inputs.items():
        file_base = f"{save_prefix}-batch{batch_idx}-{name}.npz"
        debug_path = os.path.join(output_dir_abs, file_base)
        np.savez_compressed(debug_path, **data)

        if os.path.exists(debug_path):
            # Attention witness
            witness_path = debug_path.replace(".npz", ".witness.npz")
            process_standard_attn_debug_npz(debug_path, witness_path, k_bits=22)
            print(f"[BERT Batch {batch_idx}] Generated Standard Attention Witness: {witness_path}")

            # Normalization witness: 어텐션 모듈명에서 레이어 인덱스 추출하여 매핑
            m = re.search(r'layer\.(\d+)', name)
            if m:
                layer_key = f"layer_{m.group(1)}"
                if layer_key in saved_norm:
                    norm_debug_path = debug_path.replace(".npz", ".norm_debug.npz")
                    norm_data = {k: np.asarray(v) for k, v in saved_norm[layer_key].items()}
                    np.savez_compressed(norm_debug_path, **norm_data)

                    norm_witness_path = norm_debug_path.replace(".norm_debug.npz", ".norm.witness.npz")
                    process_standard_normalization_debug_npz(
                        norm_debug_path, norm_witness_path,
                        k_bits=22, eps=EPS_BERT, table_dir=output_dir_abs,
                    )
                    print(f"[BERT Batch {batch_idx}] Generated Normalization Witness: {norm_witness_path}")


# ── Main loop ──────────────────────────────────────────────────────────────
print(
    f"Starting BERT Baseline Validation: {args.num_batch} batches × "
    f"{args.batch_size} samples (max_seq_len={MAX_SEQ_LEN}, "
    f"output_dir={output_dir_name})"
)

batches = _make_batches()

for batch_idx, batch in enumerate(batches):
    # 1. Capture (witness 생성)
    _capture_and_process_bert_intermediates(
        model, batch, batch_idx, save_prefix, OUTPUT_DIR,
    )

    # 2. 성능 측정
    t0 = time.time()
    summary = {}
    with torch.no_grad():
        outputs = model(**batch)
        summary["loss"] = (
            outputs.loss.item()
            if hasattr(outputs, "loss") and outputs.loss is not None
            else 0.0
        )
    t1 = time.time()

    summary["batch_idx"] = batch_idx
    summary["time"] = round(t1 - t0, 4)

    log_f.write(json.dumps(summary, sort_keys=True) + "\n")
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}: Time {summary['time']}s, Loss {summary['loss']:.4f}")
        log_f.flush()

log_f.close()
print(f"BERT Baseline Validation Completed. Output: {OUTPUT_DIR}/")
