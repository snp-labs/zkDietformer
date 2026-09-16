import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch

from transformers import AutoTokenizer

from evaluation_fixtures import (
    SAMPLE_TEXTS, NUM_SAMPLES, MAX_SEQ_LEN, assert_fixture_invariants,
)
from gelu_swap import replace_activation_with_sigmoid_gelu
from nystrom_attention_capture import (
    load_nystromformer_with_landmarks_override,
    load_small_nystromformer,
    apply_monkey_patch,
    valid_landmark_choices,
)
from attention_pipeline import process_attn_debug_npz
from normalization_pipeline import process_standard_normalization_debug_npz
from layernorm_table import EPS_NYSTROMFORMER


# ── 재현성 seed ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── argparse ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=int, default=None,
    help="(deprecated, 무시됨) HF 사전학습 모델 사용으로 체크포인트 인덱스 불필요.",
)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_batch", type=int, default=1)
parser.add_argument(
    "--save_attn_npz", type=str, default=None,
    help="prefix string + 출력 dir 명시. 미지정 시 outputs_m{num_landmarks}/ 자동 분기.",
)
parser.add_argument(
    "--num_landmarks", type=int, default=64,
    help=f"Nyström landmark count. seq_len={MAX_SEQ_LEN}의 약수여야 함. "
         f"sweep 권장: 32, 64, 112. 가능 후보: {valid_landmark_choices(MAX_SEQ_LEN)}.",
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
parser.add_argument(
    "--seq_len", type=int, default=None,
    help="Sequence length (default: MAX_SEQ_LEN from evaluation_fixtures).",
)
args = parser.parse_args()

# Use --seq_len if provided, otherwise fall back to evaluation_fixtures constant.
SEQ_LEN = args.seq_len if args.seq_len is not None else MAX_SEQ_LEN

# fixture 무결성 self-check
assert_fixture_invariants()

total_needed = args.num_batch * args.batch_size
if total_needed > NUM_SAMPLES:
    raise ValueError(
        f"requested {total_needed} samples (num_batch={args.num_batch} × "
        f"batch_size={args.batch_size}) but fixture has only {NUM_SAMPLES}."
    )

# ── 출력 디렉터리 자동 분기 ────────────────────────────────────────────────
if args.save_attn_npz:
    output_dir_name = args.save_attn_npz
elif args.hidden_size is not None:
    n_layers = args.num_hidden_layers or 4
    output_dir_name = f"outputs_n{SEQ_LEN}_m{args.num_landmarks}_L{n_layers}_h{args.hidden_size}_nystrom"
else:
    output_dir_name = f"outputs_n{SEQ_LEN}_m{args.num_landmarks}_nystrom"
save_prefix = output_dir_name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, output_dir_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── tokenizer + model ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")

device = torch.device("cpu")  # 실험을 위해 CPU. GPU 사용 시 "cuda"로 변경.

# ── 모델 로드: --hidden_size 지정 시 random init, 미지정 시 pretrained ───
if args.hidden_size is not None:
    n_heads = args.num_attention_heads
    if n_heads is None:
        n_heads = max(4, args.hidden_size // 64)
    model = load_small_nystromformer(
        hidden_size=args.hidden_size,
        num_attention_heads=n_heads,
        num_hidden_layers=args.num_hidden_layers or 4,
        num_landmarks=args.num_landmarks,
        seq_len=SEQ_LEN,
    )
    print(f"[Nystromformer] RANDOM INIT — hidden_size={args.hidden_size}, heads={n_heads}")
else:
    model = load_nystromformer_with_landmarks_override(
        model_id="uw-madison/nystromformer-512",
        num_landmarks=args.num_landmarks,
        seq_len=SEQ_LEN,
    )
model.to(device).eval()  # 중요: dropout 비활성화로 결정성 보장

# GELU swap (회로 lookup 정합성 — gelu_table.py가 σ(1.702x) 가정)
n_swapped = replace_activation_with_sigmoid_gelu(model, "nystromformer")
expected_layers = model.config.num_hidden_layers
assert n_swapped == expected_layers, (
    f"GELU swap incomplete: {n_swapped}/{expected_layers}"
)
print(f"[Nystromformer] swapped {n_swapped}/{expected_layers} layer activations to SigmoidGELU")
print(
    f"[Nystromformer] hidden_size={model.config.hidden_size}, "
    f"num_hidden_layers={model.config.num_hidden_layers}, "
    f"num_attention_heads={model.config.num_attention_heads}, "
    f"num_landmarks={model.config.num_landmarks}, "
    f"segment_means_seq_len={model.config.segment_means_seq_len}, "
    f"output_dir={output_dir_name}"
)

# Monkey-patch 적용 — registry 가 12 layer × 17 키 캡처
registry = apply_monkey_patch(model)


# ── batch 생성: fixture → on-the-fly tokenization ──────────────────────────
def _make_batches() -> list:
    """SAMPLE_TEXTS 의 앞 total_needed 개를 batch_size 단위로 묶음.

    각 batch shape = [batch_size, SEQ_LEN] (padding=max_length).
    """
    batches = []
    texts = SAMPLE_TEXTS[:total_needed]
    for i in range(0, len(texts), args.batch_size):
        chunk = list(texts[i : i + args.batch_size])
        enc = tokenizer(
            chunk,
            padding="max_length",
            max_length=SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        batches.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        })
    return batches


# ── log file ───────────────────────────────────────────────────────────────
log_f_path = os.path.join(OUTPUT_DIR, "nystromformer_validation_output.log")
log_f = open(log_f_path, "a+")


# ── Capture function: monkey-patch (attention) + LayerNorm hook (BERT 패턴) ─
def _capture_and_process_nystrom_intermediates(
    model, current_batch, batch_idx, save_prefix, output_dir_abs,
):
    """단일 forward 로 12 layer × (17 attention + 12 norm) 키 모두 캡처.

    - Attention: monkey-patch 가 registry 에 push (작업 2A)
    - LayerNorm: forward hook 으로 saved_norm dict 채움 (BERT Step C 동일 패턴)
    - 두 캡처를 합쳐 npz 저장 + witness 파이프라인 호출
    """
    os.makedirs(output_dir_abs, exist_ok=True)

    saved_norm = {}  # layer_key → dict (LayerNorm 측 캡처)

    # ── LayerNorm / FFN hooks (Step C 패턴 재사용 — ln_* 키) ────────────
    def make_norm_hooks(layer_key):
        # Hook 1: post-attention LayerNorm (NystromformerSelfOutput.LayerNorm)
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

        # Hook 2: FFN up-projection dense (NystromformerIntermediate.dense, before SigmoidGELU)
        def hook_ff1_dense(module, _inputs, output):
            entry = {
                "linear_weight": module.weight.detach().cpu().numpy(),
                "linear_output": output.detach().cpu().numpy(),
            }
            if module.bias is not None:
                entry["linear_bias"] = module.bias.detach().cpu().numpy()
            saved_norm.setdefault(layer_key, {}).update(entry)

        # Hook 3: full NystromformerIntermediate output (after SigmoidGELU)
        def hook_intermediate(_module, _inputs, output):
            saved_norm.setdefault(layer_key, {})["gelu_output"] = output.detach().cpu().numpy()

        # Hook 4: FFN down-projection dense (NystromformerOutput.dense, before residual+LayerNorm)
        def hook_ff2_dense(module, _inputs, output):
            entry = {
                "linear_weight_ff2": module.weight.detach().cpu().numpy(),
                "linear_output_ff2": output.detach().cpu().numpy(),
            }
            if module.bias is not None:
                entry["linear_bias_ff2"] = module.bias.detach().cpu().numpy()
            saved_norm.setdefault(layer_key, {}).update(entry)

        # Hook 5: post-FFN LayerNorm (NystromformerOutput.LayerNorm)
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

    # Norm hook 등록 (12 layer × 5 hook = 60개)
    norm_handles = []
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        layer = model.nystromformer.encoder.layer[i]
        layer_key = f"layer_{i}"
        h_aln, h_ff1, h_int, h_ff2, h_fln = make_norm_hooks(layer_key)
        norm_handles.extend([
            layer.attention.output.LayerNorm.register_forward_hook(h_aln),
            layer.intermediate.dense.register_forward_hook(h_ff1),
            layer.intermediate.register_forward_hook(h_int),
            layer.output.dense.register_forward_hook(h_ff2),
            layer.output.LayerNorm.register_forward_hook(h_fln),
        ])

    # ── Forward 1회 — 12 layer × 17 attention 키 + 12 layer × 12 norm 키 캡처 ─
    registry.reset()
    inputs = {k: v.to(device) for k, v in current_batch.items()}
    model.eval()
    with torch.no_grad():
        model(**inputs)

    # hook 제거
    for h in norm_handles:
        h.remove()

    # ── 데이터 저장 + witness 변환 ─────────────────────────────────────
    batch_mask_np = current_batch['attention_mask'].cpu().numpy().astype(np.int64)

    captured_attn = registry.all()  # {layer_name: {key: array}}
    for layer_name, attn_data in captured_attn.items():
        # mask 키는 caller 가 추가 (apply_monkey_patch 명세 — Q3 결정)
        attn_data = dict(attn_data)  # 사본 (불변 보존)
        attn_data['mask'] = batch_mask_np

        file_base = f"{save_prefix}-batch{batch_idx}-{layer_name}.npz"
        debug_path = os.path.join(output_dir_abs, file_base)
        np.savez_compressed(debug_path, **attn_data)

        if os.path.exists(debug_path):
            # Attention witness (Nystrom 3-kernel pipeline)
            witness_path = debug_path.replace(".npz", ".witness.npz")
            process_attn_debug_npz(debug_path, witness_path, k_bits=26)
            print(f"[Nystromformer Batch {batch_idx}] Generated Attention Witness: {witness_path}")

            # LayerNorm witness — module name 의 "layer.{i}" 로부터 layer_key 매핑
            m = re.search(r'layer\.(\d+)', layer_name)
            if m:
                layer_key = f"layer_{m.group(1)}"
                if layer_key in saved_norm:
                    norm_debug_path = debug_path.replace(".npz", ".norm_debug.npz")
                    norm_data = {k: np.asarray(v) for k, v in saved_norm[layer_key].items()}
                    np.savez_compressed(norm_debug_path, **norm_data)

                    norm_witness_path = norm_debug_path.replace(".norm_debug.npz", ".norm.witness.npz")
                    process_standard_normalization_debug_npz(
                        norm_debug_path, norm_witness_path,
                        k_bits=26,
                        eps=EPS_NYSTROMFORMER,        # 1e-5 — Nystrom 사전학습 정합
                        table_dir=output_dir_abs,     # 작업 2C 일반화 인자 활용
                    )
                    print(f"[Nystromformer Batch {batch_idx}] Generated Normalization Witness: {norm_witness_path}")


# ── Main loop ──────────────────────────────────────────────────────────────
print(
    f"Starting Nystromformer Validation: {args.num_batch} batches × "
    f"{args.batch_size} samples (max_seq_len={SEQ_LEN}, "
    f"num_landmarks={args.num_landmarks})"
)

batches = _make_batches()

for batch_idx, batch in enumerate(batches):
    # 1. Capture forward (witness 생성)
    _capture_and_process_nystrom_intermediates(
        model, batch, batch_idx, save_prefix, OUTPUT_DIR,
    )

    # 2. Performance forward (loss + 시간 측정)
    t0 = time.time()
    summary = {}
    with torch.no_grad():
        outputs = model(**batch)
        # labels 없이 호출하므로 outputs.loss 는 None (HF 기본 동작)
        summary["loss"] = (
            outputs.loss.item()
            if hasattr(outputs, "loss") and outputs.loss is not None
            else 0.0
        )
    t1 = time.time()

    summary["batch_idx"] = batch_idx
    summary["num_landmarks"] = args.num_landmarks
    summary["time"] = round(t1 - t0, 4)

    log_f.write(json.dumps(summary, sort_keys=True) + "\n")
    if batch_idx % 10 == 0:
        print(
            f"Batch {batch_idx}: Time {summary['time']}s, "
            f"Loss {summary['loss']:.4f}, "
            f"num_landmarks={args.num_landmarks}"
        )
        log_f.flush()

log_f.close()
print(f"Nystromformer Validation Completed. Output: {OUTPUT_DIR}/")
