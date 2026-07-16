#!/usr/bin/env python3
"""
양자화 오차 평가 스크립트.

서킷의 양자화된 attention 출력과 실제 모델의 float 출력을 비교하여:
1. 양자화 오차 (MSE, RMSE, MaxAbs, CosSim)
2. Top-1 / Top-5 prediction agreement

를 레이어별 + 전체 평균으로 보고.

사용법:
  # BERT (witness 생성 후 실행)
  python evaluate_quantization.py --model bert --witness_dir outputs_n128_bert

  # Nystromformer
  python evaluate_quantization.py --model nystrom --witness_dir outputs_n128_m64_nystrom

  # 특정 배치만
  python evaluate_quantization.py --model bert --witness_dir outputs_n128_bert --batch_idx 0
"""
import argparse
import glob
import json
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn

from evaluation_fixtures import SAMPLE_TEXTS, NUM_SAMPLES, MAX_SEQ_LEN, assert_fixture_invariants
from gelu_swap import replace_activation_with_sigmoid_gelu

# ── seed ──────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── argparse ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate quantization error & accuracy")
parser.add_argument("--model", type=str, required=True, choices=["bert", "nystrom"],
                    help="Model type: bert or nystrom")
parser.add_argument("--witness_dir", type=str, required=True,
                    help="Directory containing witness NPZ files")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_batch", type=int, default=1)
parser.add_argument("--batch_idx", type=int, default=None,
                    help="Evaluate a specific batch only (default: all)")
parser.add_argument("--hidden_size", type=int, default=None)
parser.add_argument("--num_attention_heads", type=int, default=None)
parser.add_argument("--num_hidden_layers", type=int, default=None)
parser.add_argument("--num_landmarks", type=int, default=64)
args = parser.parse_args()

assert_fixture_invariants()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WITNESS_DIR = os.path.join(SCRIPT_DIR, args.witness_dir)

# ── Auto-detect model params from witness_dir name ───────────────────────
# Pattern: outputs_n{seq}_m{landmarks}_L{layers}_h{hidden}_{model}
_dir_match = re.match(
    r'outputs_n(\d+)(?:_m(\d+))?(?:_L(\d+))?(?:_h(\d+))?_',
    os.path.basename(args.witness_dir),
)
if _dir_match:
    if _dir_match.group(2) and args.num_landmarks == 64:  # not explicitly set
        args.num_landmarks = int(_dir_match.group(2))
    if _dir_match.group(3) and args.num_hidden_layers is None:
        args.num_hidden_layers = int(_dir_match.group(3))
    if _dir_match.group(4) and args.hidden_size is None:
        args.hidden_size = int(_dir_match.group(4))
    print(f"[Auto-detect] hidden_size={args.hidden_size}, "
          f"num_hidden_layers={args.num_hidden_layers}, "
          f"num_landmarks={args.num_landmarks}")

device = torch.device("cpu")


# ── Model & Tokenizer Loading ────────────────────────────────────────────
def load_model_and_tokenizer():
    if args.model == "bert":
        from transformers import BertForMaskedLM, BertTokenizerFast
        from bert_attention_pipeline import load_small_bert

        tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased", model_max_length=MAX_SEQ_LEN,
        )
        if args.hidden_size is not None:
            n_heads = args.num_attention_heads or max(4, args.hidden_size // 64)
            model = load_small_bert(
                hidden_size=args.hidden_size,
                num_attention_heads=n_heads,
                num_hidden_layers=args.num_hidden_layers or 4,
                seq_len=MAX_SEQ_LEN,
            )
        else:
            model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        model.to(device).eval()
        n_swapped = replace_activation_with_sigmoid_gelu(model, "bert")
        print(f"[BERT] GELU swapped: {n_swapped} layers")
        return model, tokenizer

    else:  # nystrom
        from transformers import AutoTokenizer
        from nystrom_attention_capture import (
            load_nystromformer_with_landmarks_override,
            load_small_nystromformer,
        )

        tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")
        if args.hidden_size is not None:
            n_heads = args.num_attention_heads or max(4, args.hidden_size // 64)
            model = load_small_nystromformer(
                hidden_size=args.hidden_size,
                num_attention_heads=n_heads,
                num_hidden_layers=args.num_hidden_layers or 4,
                num_landmarks=args.num_landmarks,
                seq_len=MAX_SEQ_LEN,
            )
        else:
            model = load_nystromformer_with_landmarks_override(
                model_id="uw-madison/nystromformer-512",
                num_landmarks=args.num_landmarks,
                seq_len=MAX_SEQ_LEN,
            )
        model.to(device).eval()
        n_swapped = replace_activation_with_sigmoid_gelu(model, "nystromformer")
        print(f"[Nystromformer] GELU swapped: {n_swapped} layers")
        return model, tokenizer


# ── Batch Creation ────────────────────────────────────────────────────────
def make_batches(tokenizer):
    total_needed = args.num_batch * args.batch_size
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


# ── Witness Loading & Dequantization ─────────────────────────────────────
def find_witness_files(batch_idx):
    """Find all layer attention witness NPZs for a given batch, sorted by layer index."""
    pattern = os.path.join(WITNESS_DIR, f"*-batch{batch_idx}-*.witness.npz")
    files = sorted(glob.glob(pattern))
    # Filter: only attention witnesses (exclude .norm.witness.npz)
    files = [f for f in files if not f.endswith(".norm.witness.npz")]
    # Extract layer index and sort
    layer_files = []
    for f in files:
        m = re.search(r'layer[._](\d+)', f)
        if m:
            layer_files.append((int(m.group(1)), f))
    layer_files.sort(key=lambda x: x[0])
    return layer_files


def dequantize_attention_output(witness_path):
    """Load witness NPZ and dequantize O-projection output to float tensor.

    Returns: torch.Tensor [B, S, HD] (float32)
    """
    w = np.load(witness_path, allow_pickle=True)
    Q_o = w["Q_raw_proj_o"].astype(np.float64)   # [B, H, S, D]
    z_raw_o = float(w["z_raw_o"])
    s_raw_o = float(w["s_raw_o"])

    # Dequantize: o_approx = (Q_o - z_raw_o) * s_raw_o
    o_approx = (Q_o - z_raw_o) * s_raw_o   # [B, H, S, D]

    B, H, S, D = o_approx.shape
    # Reshape [B,H,S,D] → [B,S,H,D] → [B,S,HD]
    o_flat = o_approx.transpose(0, 2, 1, 3).reshape(B, S, H * D)

    return torch.tensor(o_flat, dtype=torch.float32, device=device)


# ── Attention Output Replacement Hook ─────────────────────────────────────
def make_replacement_hook(replacement_tensor):
    """Create a forward hook that replaces the attention module's output."""
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            return (replacement_tensor,) + output[1:]
        return replacement_tensor
    return hook_fn


def get_attention_output_dense_modules(model):
    """Get the O-projection dense (nn.Linear) modules sorted by layer index.

    Q_raw_proj_o is the output AFTER O-projection, so we must hook the dense
    layer itself (attention.output.dense) to avoid applying W_o twice.
    """
    modules = []
    for name, m in model.named_modules():
        # Pattern: *.attention.output.dense (both BERT and Nystromformer)
        if name.endswith(".attention.output.dense") and isinstance(m, nn.Linear):
            layer_m = re.search(r'layer\.(\d+)', name)
            if layer_m:
                modules.append((int(layer_m.group(1)), name, m))
    modules.sort(key=lambda x: x[0])
    return modules


# ── Metrics Computation ───────────────────────────────────────────────────
def compute_topk_agreement(float_logits, quant_logits, mask_positions=None):
    """Compute top-1 and top-5 prediction agreement.

    Args:
        float_logits: [B, S, V] float model logits
        quant_logits: [B, S, V] quantized model logits
        mask_positions: optional boolean tensor [B, S] for specific positions

    Returns: dict with top1_agreement, top5_agreement, total_positions
    """
    if mask_positions is not None:
        # Flatten to selected positions
        fl = float_logits[mask_positions]   # [N, V]
        ql = quant_logits[mask_positions]   # [N, V]
    else:
        B, S, V = float_logits.shape
        fl = float_logits.reshape(-1, V)
        ql = quant_logits.reshape(-1, V)

    if fl.shape[0] == 0:
        return {"top1_agreement": 0.0, "top5_agreement": 0.0, "total_positions": 0}

    # Float model's top-1 prediction
    float_top1 = fl.argmax(dim=-1)  # [N]

    # Quantized model's top-1 and top-5 predictions
    quant_top1 = ql.argmax(dim=-1)  # [N]
    quant_top5 = ql.topk(5, dim=-1).indices  # [N, 5]

    N = float_top1.shape[0]

    # Top-1 agreement: float top-1 == quant top-1
    top1_agree = (float_top1 == quant_top1).sum().item()

    # Top-5 agreement: float top-1 is in quant top-5
    float_top1_expanded = float_top1.unsqueeze(-1)  # [N, 1]
    top5_agree = (quant_top5 == float_top1_expanded).any(dim=-1).sum().item()

    return {
        "top1_agreement": top1_agree / N,
        "top5_agreement": top5_agree / N,
        "total_positions": N,
    }


def compute_quantization_error(float_logits, quant_logits, mask_positions=None):
    """Compute logit-level quantization error metrics."""
    if mask_positions is not None:
        fl = float_logits[mask_positions].float()
        ql = quant_logits[mask_positions].float()
    else:
        fl = float_logits.reshape(-1, float_logits.shape[-1]).float()
        ql = quant_logits.reshape(-1, quant_logits.shape[-1]).float()

    if fl.shape[0] == 0:
        return {}

    diff = fl - ql
    mse = diff.pow(2).mean().item()
    rmse = math.sqrt(mse)
    max_abs = diff.abs().max().item()

    # Cosine similarity (per position, then average)
    cos_sim = nn.functional.cosine_similarity(fl, ql, dim=-1).mean().item()

    return {
        "logit_mse": mse,
        "logit_rmse": rmse,
        "logit_max_abs_error": max_abs,
        "logit_cosine_similarity": cos_sim,
    }


# ── Main Evaluation ──────────────────────────────────────────────────────
def evaluate_batch(model, tokenizer, batch, batch_idx):
    """Evaluate a single batch: float vs quantized predictions."""

    # 1. Float model forward → logits
    with torch.no_grad():
        float_outputs = model(**batch)
    float_logits = float_outputs.logits  # [B, S, V]

    # 2. Load witness files for this batch
    layer_files = find_witness_files(batch_idx)
    if not layer_files:
        print(f"  [WARNING] No witness files found for batch {batch_idx}")
        return None

    attn_modules = get_attention_output_dense_modules(model)
    num_layers = len(attn_modules)

    # 3. Dequantize all layers
    dequantized = {}
    for layer_idx, witness_path in layer_files:
        if layer_idx < num_layers:
            dequantized[layer_idx] = dequantize_attention_output(witness_path)

    if len(dequantized) != num_layers:
        print(f"  [WARNING] Found witnesses for {len(dequantized)}/{num_layers} layers")

    # 4. Hook all attention layers with dequantized outputs
    handles = []
    for layer_idx, name, module in attn_modules:
        if layer_idx in dequantized:
            hook = make_replacement_hook(dequantized[layer_idx])
            handles.append(module.register_forward_hook(hook))

    # 5. Quantized model forward → logits
    with torch.no_grad():
        quant_outputs = model(**batch)
    quant_logits = quant_outputs.logits  # [B, S, V]

    # 6. Remove hooks
    for h in handles:
        h.remove()

    # 7. Find [MASK] positions
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (batch["input_ids"] == mask_token_id) if mask_token_id else None

    # 8. Find non-padding positions
    attn_mask = batch["attention_mask"].bool()

    # 9. Compute metrics
    results = {"batch_idx": batch_idx}

    # All non-padding positions
    results["all_positions"] = compute_topk_agreement(float_logits, quant_logits, attn_mask)
    results["all_positions_error"] = compute_quantization_error(float_logits, quant_logits, attn_mask)

    # [MASK] positions only
    if mask_positions is not None and mask_positions.any():
        results["mask_positions"] = compute_topk_agreement(float_logits, quant_logits, mask_positions)
        results["mask_positions_error"] = compute_quantization_error(float_logits, quant_logits, mask_positions)

        # Print float model's top predictions at [MASK] positions
        mask_logits = float_logits[mask_positions]  # [N_mask, V]
        quant_mask_logits = quant_logits[mask_positions]  # [N_mask, V]
        for i in range(mask_logits.shape[0]):
            float_top5_ids = mask_logits[i].topk(5).indices.tolist()
            quant_top5_ids = quant_mask_logits[i].topk(5).indices.tolist()
            float_top5_tokens = tokenizer.convert_ids_to_tokens(float_top5_ids)
            quant_top5_tokens = tokenizer.convert_ids_to_tokens(quant_top5_ids)
            results.setdefault("mask_predictions", []).append({
                "float_top5": float_top5_tokens,
                "quant_top5": quant_top5_tokens,
            })

    return results


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {args.model} model...")
    model, tokenizer = load_model_and_tokenizer()
    batches = make_batches(tokenizer)

    all_results = []
    agg_top1_all = []
    agg_top5_all = []
    agg_top1_mask = []
    agg_top5_mask = []

    batch_indices = [args.batch_idx] if args.batch_idx is not None else range(len(batches))

    for bi in batch_indices:
        if bi >= len(batches):
            print(f"[WARNING] batch_idx {bi} out of range (max {len(batches)-1})")
            continue

        print(f"\n{'='*60}")
        print(f"Batch {bi}")
        print(f"{'='*60}")

        result = evaluate_batch(model, tokenizer, batches[bi], bi)
        if result is None:
            continue

        all_results.append(result)

        # Print results
        ap = result["all_positions"]
        ae = result.get("all_positions_error", {})
        print(f"\n  [All Positions] (n={ap['total_positions']})")
        print(f"    Top-1 Agreement: {ap['top1_agreement']:.4f} ({ap['top1_agreement']*100:.2f}%)")
        print(f"    Top-5 Agreement: {ap['top5_agreement']:.4f} ({ap['top5_agreement']*100:.2f}%)")
        if ae:
            print(f"    Logit CosSim:    {ae['logit_cosine_similarity']:.8f}")
            print(f"    Logit RMSE:      {ae['logit_rmse']:.6e}")

        agg_top1_all.append(ap["top1_agreement"])
        agg_top5_all.append(ap["top5_agreement"])

        if "mask_positions" in result:
            mp = result["mask_positions"]
            me = result.get("mask_positions_error", {})
            print(f"\n  [[MASK] Positions] (n={mp['total_positions']})")
            print(f"    Top-1 Agreement: {mp['top1_agreement']:.4f} ({mp['top1_agreement']*100:.2f}%)")
            print(f"    Top-5 Agreement: {mp['top5_agreement']:.4f} ({mp['top5_agreement']*100:.2f}%)")
            if me:
                print(f"    Logit CosSim:    {me['logit_cosine_similarity']:.8f}")

            agg_top1_mask.append(mp["top1_agreement"])
            agg_top5_mask.append(mp["top5_agreement"])

        if "mask_predictions" in result:
            print(f"\n  [MASK] Predictions:")
            for i, pred in enumerate(result["mask_predictions"]):
                print(f"    [MASK #{i}]  Float: {pred['float_top5']}  |  Quant: {pred['quant_top5']}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS ({len(all_results)} batches)")
        print(f"{'='*60}")
        print(f"  All Positions:")
        print(f"    Top-1 Agreement: {np.mean(agg_top1_all):.4f} ({np.mean(agg_top1_all)*100:.2f}%)")
        print(f"    Top-5 Agreement: {np.mean(agg_top5_all):.4f} ({np.mean(agg_top5_all)*100:.2f}%)")
        if agg_top1_mask:
            print(f"  [MASK] Positions:")
            print(f"    Top-1 Agreement: {np.mean(agg_top1_mask):.4f} ({np.mean(agg_top1_mask)*100:.2f}%)")
            print(f"    Top-5 Agreement: {np.mean(agg_top5_mask):.4f} ({np.mean(agg_top5_mask)*100:.2f}%)")

        # Save results to JSON
        output_path = os.path.join(WITNESS_DIR, "quantization_eval.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")
