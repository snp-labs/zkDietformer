"""
Linformer witness generation entry point (A2 + A9).

Loads BERT model, replaces BertSelfAttention with LinformerSelfAttention,
applies monkey-patch capture, runs forward, generates quantized witness NPZ.

Usage:
    python run_linformer_validation.py --hidden_size 128 --batch_size 1
    python run_linformer_validation.py  # pretrained bert-base-uncased
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
from gelu_swap import replace_activation_with_sigmoid_gelu
from bert_attention_pipeline import load_small_bert
from linformer_attention import LinformerSelfAttention
from linformer_attention_capture import apply_linformer_monkey_patch
from linformer_attention_pipeline import process_linformer_attn_debug_npz
from normalization_pipeline import process_standard_normalization_debug_npz
from layernorm_table import EPS_BERT


# ── Seed ──────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── argparse ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_batch", type=int, default=1)
parser.add_argument("--save_attn_npz", type=str, default=None)
parser.add_argument("--hidden_size", type=int, default=None)
parser.add_argument("--num_attention_heads", type=int, default=None)
parser.add_argument("--num_hidden_layers", type=int, default=None)
parser.add_argument(
    "--proj_dim", type=int, default=32,
    help="Linformer projection dim k (default 32).",
)
args = parser.parse_args()

assert_fixture_invariants()

total_needed = args.num_batch * args.batch_size
if total_needed > NUM_SAMPLES:
    raise ValueError(
        f"requested {total_needed} samples but fixture has {NUM_SAMPLES}."
    )


# ── Replace BertSelfAttention with LinformerSelfAttention ─────────────────
def replace_bert_attn_with_linformer(model, seq_len, proj_dim=32):
    """Swap all BertSelfAttention modules with LinformerSelfAttention."""
    layers = model.bert.encoder.layer
    n_replaced = 0
    for layer_module in layers:
        old_attn = layer_module.attention.self
        hidden_size = old_attn.query.in_features
        num_heads = old_attn.num_attention_heads

        new_attn = LinformerSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            seq_len=seq_len,
            proj_dim=proj_dim,
        )
        # Copy Q/K/V weights from BERT
        new_attn.query.weight.data.copy_(old_attn.query.weight.data)
        new_attn.key.weight.data.copy_(old_attn.key.weight.data)
        new_attn.value.weight.data.copy_(old_attn.value.weight.data)
        if old_attn.query.bias is not None:
            new_attn.query.bias.data.copy_(old_attn.query.bias.data)
        if old_attn.key.bias is not None:
            new_attn.key.bias.data.copy_(old_attn.key.bias.data)
        if old_attn.value.bias is not None:
            new_attn.value.bias.data.copy_(old_attn.value.bias.data)

        layer_module.attention.self = new_attn
        n_replaced += 1

    expected = len(layers)
    assert n_replaced == expected, (
        f"replaced {n_replaced} layers, expected {expected}"
    )
    return n_replaced


# ── Output dir ────────────────────────────────────────────────────────────
if args.save_attn_npz:
    output_dir_name = args.save_attn_npz
elif args.hidden_size is not None:
    n_layers = args.num_hidden_layers or 4
    output_dir_name = (
        f"outputs_n{MAX_SEQ_LEN}_k{args.proj_dim}"
        f"_L{n_layers}_h{args.hidden_size}_linformer"
    )
else:
    output_dir_name = f"outputs_n{MAX_SEQ_LEN}_k{args.proj_dim}_linformer"
save_prefix = output_dir_name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, output_dir_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tokenizer + Model ────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased", model_max_length=MAX_SEQ_LEN,
)
device = torch.device("cpu")

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
    print(f"[Linformer] RANDOM INIT — hidden={args.hidden_size}, heads={n_heads}")
else:
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

model.to(device).eval()

# GELU swap
n_swapped = replace_activation_with_sigmoid_gelu(model, "bert")
expected_layers = model.config.num_hidden_layers
assert n_swapped == expected_layers, (
    f"GELU swap incomplete: {n_swapped}/{expected_layers}"
)

# Replace BERT attention with Linformer
n_replaced = replace_bert_attn_with_linformer(
    model, MAX_SEQ_LEN, proj_dim=args.proj_dim,
)
print(
    f"[Linformer] replaced {n_replaced} layers, "
    f"proj_dim={args.proj_dim}, seq_len={MAX_SEQ_LEN}"
)

# Apply monkey-patch capture
registry = apply_linformer_monkey_patch(model)
print(f"[Linformer] monkey-patch applied, registry ready")


# ── Batch generation ─────────────────────────────────────────────────────
def _make_batches():
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


# ── Normalization hooks (BERT-identical pattern) ─────────────────────────
def _register_norm_hooks(model, saved_norm):
    handles = []
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        layer = model.bert.encoder.layer[i]
        layer_key = f"layer_{i}"

        def _make_hooks(lk):
            def h_aln(module, inputs, output):
                x = inputs[0].detach().cpu()
                saved_norm.setdefault(lk, {}).update({
                    "ln_input": x.numpy(),
                    "mean": x.mean(dim=-1, keepdim=True).numpy(),
                    "var": x.var(dim=-1, unbiased=False, keepdim=True).numpy(),
                    "ln_weight": module.weight.detach().cpu().numpy(),
                    "ln_bias": module.bias.detach().cpu().numpy(),
                    "ln_output": output.detach().cpu().numpy(),
                })

            def h_ff1(module, _inputs, output):
                entry = {
                    "linear_weight": module.weight.detach().cpu().numpy(),
                    "linear_output": output.detach().cpu().numpy(),
                }
                if module.bias is not None:
                    entry["linear_bias"] = module.bias.detach().cpu().numpy()
                saved_norm.setdefault(lk, {}).update(entry)

            def h_int(_module, _inputs, output):
                saved_norm.setdefault(lk, {})["gelu_output"] = (
                    output.detach().cpu().numpy()
                )

            def h_ff2(module, _inputs, output):
                entry = {
                    "linear_weight_ff2": module.weight.detach().cpu().numpy(),
                    "linear_output_ff2": output.detach().cpu().numpy(),
                }
                if module.bias is not None:
                    entry["linear_bias_ff2"] = module.bias.detach().cpu().numpy()
                saved_norm.setdefault(lk, {}).update(entry)

            def h_fln(module, inputs, output):
                x = inputs[0].detach().cpu()
                saved_norm.setdefault(lk, {}).update({
                    "ln_input_2": x.numpy(),
                    "mean_2": x.mean(dim=-1, keepdim=True).numpy(),
                    "var_2": x.var(dim=-1, unbiased=False, keepdim=True).numpy(),
                    "ln_weight_2": module.weight.detach().cpu().numpy(),
                    "ln_bias_2": module.bias.detach().cpu().numpy(),
                    "ln_output_2": output.detach().cpu().numpy(),
                })
            return h_aln, h_ff1, h_int, h_ff2, h_fln

        h_aln, h_ff1, h_int, h_ff2, h_fln = _make_hooks(layer_key)
        handles.extend([
            layer.attention.output.LayerNorm.register_forward_hook(h_aln),
            layer.intermediate.dense.register_forward_hook(h_ff1),
            layer.intermediate.register_forward_hook(h_int),
            layer.output.dense.register_forward_hook(h_ff2),
            layer.output.LayerNorm.register_forward_hook(h_fln),
        ])
    return handles


# ── Main loop ─────────────────────────────────────────────────────────────
print(
    f"Starting Linformer Validation: {args.num_batch} batches x "
    f"{args.batch_size} samples (seq_len={MAX_SEQ_LEN}, k={args.proj_dim})"
)

batches = _make_batches()
log_path = os.path.join(OUTPUT_DIR, "linformer_validation_output.log")
log_f = open(log_path, "a+")

for batch_idx, batch in enumerate(batches):
    saved_norm = {}
    norm_handles = _register_norm_hooks(model, saved_norm)

    # Forward with capture
    registry.reset()
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)

    for h in norm_handles:
        h.remove()

    # Mask
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        batch_mask_np = batch["attention_mask"].cpu().numpy().astype(np.int64)
    else:
        batch_mask_np = np.ones(
            (batch["input_ids"].shape[0], MAX_SEQ_LEN), dtype=np.int64,
        )

    # Process each captured layer
    captured_attn = registry.all()
    for layer_name, attn_data in captured_attn.items():
        attn_data["mask"] = batch_mask_np
        safe_name = layer_name.replace(".", "_")
        debug_path = os.path.join(
            OUTPUT_DIR,
            f"{save_prefix}-batch{batch_idx}-{safe_name}.npz",
        )
        np.savez_compressed(debug_path, **attn_data)

        # Attention witness
        witness_path = debug_path.replace(".npz", ".witness.npz")
        result = process_linformer_attn_debug_npz(
            debug_path, witness_path, k_bits=26,
        )
        print(
            f"[Linformer Batch {batch_idx}] {layer_name}: "
            f"MaxAbsErr={result['max_abs_error']:.6e} "
            f"CosSim={result['cosine_similarity']:.8f}"
        )

        # Normalization witness
        m = re.search(r'layer\.(\d+)', layer_name)
        if m:
            layer_key = f"layer_{m.group(1)}"
            if layer_key in saved_norm:
                norm_debug_path = debug_path.replace(
                    ".npz", ".norm_debug.npz",
                )
                norm_data = {
                    k: np.asarray(v)
                    for k, v in saved_norm[layer_key].items()
                }
                np.savez_compressed(norm_debug_path, **norm_data)

                norm_witness_path = norm_debug_path.replace(
                    ".norm_debug.npz", ".norm.witness.npz",
                )
                process_standard_normalization_debug_npz(
                    norm_debug_path, norm_witness_path,
                    k_bits=26, eps=EPS_BERT, table_dir=OUTPUT_DIR,
                )
                print(
                    f"[Linformer Batch {batch_idx}] "
                    f"Norm witness: {norm_witness_path}"
                )

    # Performance measurement
    t0 = time.time()
    with torch.no_grad():
        out2 = model(**batch)
        loss = (
            out2.loss.item()
            if hasattr(out2, "loss") and out2.loss is not None
            else 0.0
        )
    t1 = time.time()

    summary = {
        "batch_idx": batch_idx,
        "time": round(t1 - t0, 4),
        "loss": loss,
    }
    log_f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(f"Batch {batch_idx}: Time {summary['time']}s, Loss {summary['loss']:.4f}")
    log_f.flush()

log_f.close()
print(f"Linformer Validation Completed. Output: {OUTPUT_DIR}/")
