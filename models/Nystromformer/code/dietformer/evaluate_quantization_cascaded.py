#!/usr/bin/env python3
"""
Cascaded quantization evaluation — layer-by-layer error propagation.

기존 evaluate_quantization_full.py는 각 레이어의 witness를 독립적으로 로드하여
MLP witness가 float attention 출력에서 생성되어 cascade 효과가 없었음.
본 스크립트는 MLP를 corrupted 입력에 대해 재양자화하여 진정한 cascade 측정.

Evaluation Modes:
  1. attn_cascade:  attention witness dequant only, MLP float (cascade through layers)
  2. full_cascade:  attention witness dequant + MLP on-the-fly re-quantization
  3. independent:   기존 방식 (pre-computed witness, 비교용)

Usage:
  python evaluate_quantization_cascaded.py \\
    --model bert --witness_dir outputs_n128_L2_h128_bert --num_batch 10

  python evaluate_quantization_cascaded.py \\
    --model nystrom --witness_dir outputs_n128_m32_L2_h128_nystrom --num_batch 10
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


class TeeWriter:
    """stdout을 콘솔 + 파일에 동시 출력."""

    def __init__(self, log_path, original_stdout):
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = original_stdout

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

from evaluation_fixtures import SAMPLE_TEXTS, NUM_SAMPLES, assert_fixture_invariants
from gelu_swap import replace_activation_with_sigmoid_gelu
from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul,
    get_gelu_index_multiplier,
)
from normalization_pipeline import _process_layernorm_block
from layernorm_table import (
    generate_layernorm_tables,
    FRAC_BITS, VAR_SCALE, EPS_BERT, EPS_NYSTROMFORMER,
)
from gelu_table import generate_gelu_sigmoid_tables, extract_gelu_sigmoid_single

# ── Seed ─────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Argparse ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Cascaded quantization evaluation")
parser.add_argument("--model", type=str, required=True, choices=["bert", "nystrom"])
parser.add_argument("--witness_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_batch", type=int, default=1)
parser.add_argument("--batch_idx", type=int, default=None)
parser.add_argument("--hidden_size", type=int, default=None)
parser.add_argument("--num_attention_heads", type=int, default=None)
parser.add_argument("--num_hidden_layers", type=int, default=None)
parser.add_argument("--num_landmarks", type=int, default=64)
parser.add_argument("--seq_len", type=int, default=None,
                    help="Sequence length (auto-detected from witness_dir name)")
parser.add_argument("--log_file", type=str, default=None,
                    help="Log file path. 미지정 시 witness_dir 내 자동 생성.")
args = parser.parse_args()

assert_fixture_invariants()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WITNESS_DIR = os.path.join(SCRIPT_DIR, args.witness_dir)

# ── Auto-detect params from witness_dir name ─────────────────────────────
_dir_match = re.match(
    r'outputs_n(\d+)(?:_m(\d+))?(?:_L(\d+))?(?:_h(\d+))?_',
    os.path.basename(args.witness_dir),
)
if _dir_match:
    if args.seq_len is None and _dir_match.group(1):
        args.seq_len = int(_dir_match.group(1))
    if _dir_match.group(2) and args.num_landmarks == 64:
        args.num_landmarks = int(_dir_match.group(2))
    if _dir_match.group(3) and args.num_hidden_layers is None:
        args.num_hidden_layers = int(_dir_match.group(3))
    if _dir_match.group(4) and args.hidden_size is None:
        args.hidden_size = int(_dir_match.group(4))

SEQ_LEN = args.seq_len or 128
K_BITS = 26
EPS = EPS_NYSTROMFORMER if args.model == "nystrom" else EPS_BERT

print(f"[Config] model={args.model}, seq_len={SEQ_LEN}, "
      f"hidden_size={args.hidden_size}, num_hidden_layers={args.num_hidden_layers}, "
      f"num_landmarks={args.num_landmarks}, eps={EPS}")

device = torch.device("cpu")


# ── Model & Tokenizer Loading ────────────────────────────────────────────
def load_model_and_tokenizer():
    if args.model == "bert":
        from transformers import BertForMaskedLM, BertTokenizerFast
        from bert_attention_pipeline import load_small_bert

        tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased", model_max_length=SEQ_LEN,
        )
        if args.hidden_size is not None:
            n_heads = args.num_attention_heads or max(4, args.hidden_size // 64)
            model = load_small_bert(
                hidden_size=args.hidden_size,
                num_attention_heads=n_heads,
                num_hidden_layers=args.num_hidden_layers or 4,
                seq_len=SEQ_LEN,
            )
        else:
            model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        model.to(device).eval()
        n_swapped = replace_activation_with_sigmoid_gelu(model, "bert")
        print(f"[BERT] GELU swapped: {n_swapped} layers")
        return model, tokenizer
    else:
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
                seq_len=SEQ_LEN,
            )
        else:
            model = load_nystromformer_with_landmarks_override(
                model_id="uw-madison/nystromformer-512",
                num_landmarks=args.num_landmarks,
                seq_len=SEQ_LEN,
            )
        model.to(device).eval()
        n_swapped = replace_activation_with_sigmoid_gelu(model, "nystromformer")
        print(f"[Nystromformer] GELU swapped: {n_swapped} layers")
        return model, tokenizer


# ── Batch Creation ───────────────────────────────────────────────────────
def make_batches(tokenizer):
    total_needed = args.num_batch * args.batch_size
    batches = []
    texts = SAMPLE_TEXTS[:total_needed]
    for i in range(0, len(texts), args.batch_size):
        chunk = list(texts[i : i + args.batch_size])
        enc = tokenizer(
            chunk, padding="max_length", max_length=SEQ_LEN,
            truncation=True, return_tensors="pt",
        )
        batches.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        })
    return batches


# ── Witness File Discovery ───────────────────────────────────────────────
def find_attn_witness_files(batch_idx):
    pattern = os.path.join(WITNESS_DIR, f"*-batch{batch_idx}-*.witness.npz")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not f.endswith(".norm.witness.npz")]
    layer_files = []
    for f in files:
        m = re.search(r'layer[._](\d+)', f)
        if m:
            layer_files.append((int(m.group(1)), f))
    layer_files.sort(key=lambda x: x[0])
    return layer_files


def find_mlp_files(batch_idx):
    pattern = os.path.join(WITNESS_DIR, f"*-batch{batch_idx}-*.norm.witness.npz")
    witness_files = sorted(glob.glob(pattern))
    result = []
    for wf in witness_files:
        m = re.search(r'layer[._](\d+)', wf)
        if not m:
            continue
        layer_idx = int(m.group(1))
        debug_path = wf.replace(".norm.witness.npz", ".norm_debug.npz")
        if os.path.exists(debug_path):
            result.append((layer_idx, debug_path, wf))
    result.sort(key=lambda x: x[0])
    return result


# ── Dequantization ───────────────────────────────────────────────────────
def dequantize_attention_output(witness_path):
    w = np.load(witness_path, allow_pickle=True)
    Q_o = w["Q_raw_proj_o"].astype(np.float64)
    z_raw_o = float(w["z_raw_o"])
    s_raw_o = float(w["s_raw_o"])
    o_approx = (Q_o - z_raw_o) * s_raw_o
    B, H, S, D = o_approx.shape
    o_flat = o_approx.transpose(0, 2, 1, 3).reshape(B, S, H * D)
    return torch.tensor(o_flat, dtype=torch.float32, device=device)


def dequantize_mlp_output(debug_path, witness_path):
    debug = np.load(debug_path, allow_pickle=True)
    witness = np.load(witness_path, allow_pickle=True)
    Q_out = witness["Q_ln_output_sim_ln2"].astype(np.float64)
    ln_output_2_ref = debug["ln_output_2"]
    s_out, z_out = get_quantization_params(ln_output_2_ref)
    out_approx = (Q_out - z_out) * s_out
    return torch.tensor(out_approx, dtype=torch.float32, device=device)


# ── SigmoidGELU in NumPy ────────────────────────────────────────────────
def sigmoid_gelu_numpy(x):
    """SigmoidGELU: sigma(1.702 * x) * x"""
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


# ── Lookup Tables ────────────────────────────────────────────────────────
def load_tables():
    table_dir = WITNESS_DIR
    ln_path = generate_layernorm_tables(
        table_dir, k_bits=K_BITS, frac_bits=FRAC_BITS, eps=EPS,
    )
    ln_table = np.load(ln_path)['table']
    gelu_path = generate_gelu_sigmoid_tables(
        table_dir, k_bits=K_BITS, frac_bits=FRAC_BITS,
    )
    gelu_table = np.load(gelu_path)['table']
    return ln_table, gelu_table


# ── MLP Quantization Pipeline ───────────────────────────────────────────
def quantize_mlp_block(
    ln1_input_np,     # [B, S, H] float — corrupted attention + residual
    ln1_output_np,    # [B, S, H] float — corrupted post-attention LN output
    ln1_weight_np,    # [H] float — LayerNorm gamma
    ln1_bias_np,      # [H] float — LayerNorm beta
    ff1_weight_np,    # [4H, H] float — FFN up-projection weight
    ff1_bias_np,      # [4H] float or None
    ff2_weight_np,    # [H, 4H] float — FFN down-projection weight
    ff2_bias_np,      # [H] float or None
    ln2_weight_np,    # [H] float — post-FFN LayerNorm gamma
    ln2_bias_np,      # [H] float — post-FFN LayerNorm beta
    ln_table, gelu_table,
    k_bits=26, eps=EPS_BERT,
):
    """Run full MLP quantization pipeline on corrupted input.

    Returns: dequantized post-FFN LayerNorm output [B, S, H] (float64 numpy)
    """
    # ── 1. Post-attention LayerNorm (LN1) ────────────────────────────────
    var_ref1 = np.var(ln1_input_np, axis=-1, keepdims=True).astype(np.float64)
    block1 = _process_layernorm_block(
        x_real=ln1_input_np,
        var_ref=var_ref1,
        gamma_real=ln1_weight_np,
        beta_real=ln1_bias_np,
        out_real=ln1_output_np,
        k_bits=k_bits, frac_bits=FRAC_BITS, eps=eps, ln_table=ln_table,
    )

    # LN1 output quantization params (for Linear1 input)
    s_ln1_out, z_ln1_out = get_quantization_params(ln1_output_np)

    # ── 2. Float references (from corrupted LN1 output) ─────────────────
    ff1_ref = np.matmul(ln1_output_np, ff1_weight_np.T)
    if ff1_bias_np is not None:
        ff1_ref = ff1_ref + ff1_bias_np
    gelu_ref = sigmoid_gelu_numpy(ff1_ref)
    ff2_ref = np.matmul(gelu_ref, ff2_weight_np.T)
    if ff2_bias_np is not None:
        ff2_ref = ff2_ref + ff2_bias_np

    # ── 3. Linear1 (FFN up-projection) ──────────────────────────────────
    s_w1, z_w1 = get_quantization_params(ff1_weight_np)
    s_out1, z_out1 = get_quantization_params(ff1_ref)
    Q_w1 = quantize_tensor(ff1_weight_np, s_w1, z_w1)

    Q_in_diff1 = (block1['Q_ln_output_sim'] - z_ln1_out).astype(np.int64)
    Q_w_diff1 = (Q_w1 - z_w1).astype(np.int64)

    G1 = np.matmul(Q_in_diff1, Q_w_diff1.T)
    if ff1_bias_np is not None:
        s_b1 = s_w1 * s_ln1_out
        Q_b_diff1 = np.round(ff1_bias_np / s_b1).astype(np.int64)
        acc1 = G1 + Q_b_diff1
    else:
        acc1 = G1

    M1 = get_m_constant(s_w1, s_ln1_out, s_out1, k=k_bits)
    M_prime1 = int(np.floor((z_out1 * (1 << k_bits)) / M1))
    num1 = M1 * (acc1 + M_prime1)
    Q_lin1 = (num1 >> k_bits).astype(np.int64)
    R_lin1 = (num1 % (1 << k_bits)).astype(np.int64)

    # ── 4. GELU (sigmoid lookup) ────────────────────────────────────────
    M_gelu_idx = get_gelu_index_multiplier(s_out1, k_bits=k_bits)
    sigmoid_part = extract_gelu_sigmoid_single(
        Q_lin1, R_lin1, z_out1, gelu_table, M_gelu_idx, k_bits,
    )
    acc_gelu = (Q_lin1 - z_out1).astype(object) * sigmoid_part.astype(object)
    s_gelu, z_gelu = get_quantization_params(gelu_ref)

    M_gelu = get_m_constant(s_out1, 1.0, s_gelu * (1 << FRAC_BITS), k=k_bits)
    M_gelu = max(1, M_gelu)
    M_prime_gelu = int(np.floor((z_gelu * (1 << k_bits)) / M_gelu))
    num_gelu = M_gelu * (acc_gelu.astype(object) + M_prime_gelu)
    Q_gelu = (num_gelu >> k_bits).astype(np.int64)

    # ── 5. Linear2 (FFN down-projection) ────────────────────────────────
    s_w2, z_w2 = get_quantization_params(ff2_weight_np)
    s_out2, z_out2 = get_quantization_params(ff2_ref)
    Q_w2 = quantize_tensor(ff2_weight_np, s_w2, z_w2)

    Q_in_diff2 = (Q_gelu - z_gelu).astype(np.int64)
    Q_w_diff2 = (Q_w2 - z_w2).astype(np.int64)

    G2 = np.matmul(Q_in_diff2, Q_w_diff2.T)
    if ff2_bias_np is not None:
        s_b2 = s_w2 * s_gelu
        Q_b_diff2 = np.round(ff2_bias_np / s_b2).astype(np.int64)
        acc2 = G2 + Q_b_diff2
    else:
        acc2 = G2

    M2 = get_m_constant(s_w2, s_gelu, s_out2, k=k_bits)
    M_prime2 = int(np.floor((z_out2 * (1 << k_bits)) / M2))
    num2 = M2 * (acc2 + M_prime2)
    Q_lin2 = (num2 >> k_bits).astype(np.int64)

    # ── 6. Dequant Linear2 + residual → LN2 input ──────────────────────
    dequant_lin2 = (Q_lin2.astype(np.float64) - z_out2) * s_out2
    # Residual: ln2_input = ff2_dequant + post-attention LN output
    ln2_x = dequant_lin2 + ln1_output_np

    # ── 7. Post-FFN LayerNorm (LN2) ─────────────────────────────────────
    var_ref2 = np.var(ln2_x, axis=-1, keepdims=True).astype(np.float64)
    ln2_mean = ln2_x.mean(axis=-1, keepdims=True)
    ln2_out_ref = (
        ln2_weight_np * (ln2_x - ln2_mean) / np.sqrt(var_ref2 + eps)
        + ln2_bias_np
    )

    block2 = _process_layernorm_block(
        x_real=ln2_x,
        var_ref=var_ref2,
        gamma_real=ln2_weight_np,
        beta_real=ln2_bias_np,
        out_real=ln2_out_ref,
        k_bits=k_bits, frac_bits=FRAC_BITS, eps=eps, ln_table=ln_table,
    )

    # ── 8. Final dequant ────────────────────────────────────────────────
    s_ln2, z_ln2 = get_quantization_params(ln2_out_ref)
    dequant_ln2 = (block2['Q_ln_output_sim'].astype(np.float64) - z_ln2) * s_ln2

    return dequant_ln2


# ── Hook Helpers ─────────────────────────────────────────────────────────
def make_replacement_hook(replacement_tensor):
    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            return (replacement_tensor,) + output[1:]
        return replacement_tensor
    return hook_fn


def get_encoder_layers(model):
    """Return list of transformer layer modules sorted by index."""
    if args.model == "bert":
        return list(model.bert.encoder.layer)
    else:
        return list(model.nystromformer.encoder.layer)


# ── Cascade Hook Manager ────────────────────────────────────────────────
class CascadeHookManager:
    """Manages per-layer hooks for cascaded quantization evaluation."""

    def __init__(self, model, attn_dequant, mlp_dequant_independent,
                 ln_table, gelu_table, mode="full_cascade"):
        """
        Args:
            model: PyTorch transformer model
            attn_dequant: {layer_idx: tensor} — attention witness dequant
            mlp_dequant_independent: {layer_idx: tensor} — pre-computed MLP dequant (for independent mode)
            ln_table, gelu_table: lookup tables
            mode: 'attn_cascade' | 'full_cascade' | 'independent'
        """
        self.model = model
        self.attn_dequant = attn_dequant
        self.mlp_dequant_independent = mlp_dequant_independent
        self.ln_table = ln_table
        self.gelu_table = gelu_table
        self.mode = mode
        self.handles = []
        # Per-layer capture storage
        self.captured = {}

    def setup_hooks(self):
        encoder_layers = get_encoder_layers(self.model)

        for layer_idx, layer in enumerate(encoder_layers):
            # Hook 1: attention.output.dense — replace with witness dequant
            if layer_idx in self.attn_dequant:
                if args.model == "bert":
                    attn_dense = layer.attention.output.dense
                else:
                    attn_dense = layer.attention.output.dense
                h = attn_dense.register_forward_hook(
                    make_replacement_hook(self.attn_dequant[layer_idx])
                )
                self.handles.append(h)

            if self.mode == "full_cascade":
                self._setup_cascade_hooks(layer_idx, layer)
            elif self.mode == "independent":
                self._setup_independent_hooks(layer_idx, layer)
            # attn_cascade: no MLP hooks needed

    def _setup_cascade_hooks(self, layer_idx, layer):
        """Set up hooks for on-the-fly MLP re-quantization."""
        # Hook 2: attention.output.LayerNorm — capture corrupted input/output
        if args.model == "bert":
            attn_ln = layer.attention.output.LayerNorm
        else:
            attn_ln = layer.attention.output.LayerNorm
        self.captured[layer_idx] = {}

        def make_capture_hook(lidx):
            def hook_fn(module, inputs, output):
                self.captured[lidx]['ln1_input'] = inputs[0].detach().cpu().numpy().astype(np.float64)
                self.captured[lidx]['ln1_output'] = output.detach().cpu().numpy().astype(np.float64)
                self.captured[lidx]['ln1_weight'] = module.weight.detach().cpu().numpy().astype(np.float64)
                self.captured[lidx]['ln1_bias'] = module.bias.detach().cpu().numpy().astype(np.float64)
            return hook_fn

        h = attn_ln.register_forward_hook(make_capture_hook(layer_idx))
        self.handles.append(h)

        # Hook 3: output.LayerNorm — replace with re-quantized MLP output
        if args.model == "bert":
            out_ln = layer.output.LayerNorm
            ff1_dense = layer.intermediate.dense
            ff2_dense = layer.output.dense
        else:
            out_ln = layer.output.LayerNorm
            ff1_dense = layer.intermediate.dense
            ff2_dense = layer.output.dense

        def make_mlp_hook(lidx, ff1_mod, ff2_mod, out_ln_mod):
            def hook_fn(module, inputs, output):
                cap = self.captured.get(lidx)
                if cap is None or 'ln1_input' not in cap:
                    return output

                # Extract model weights
                ff1_w = ff1_mod.weight.detach().cpu().numpy().astype(np.float64)
                ff1_b = ff1_mod.bias.detach().cpu().numpy().astype(np.float64) if ff1_mod.bias is not None else None
                ff2_w = ff2_mod.weight.detach().cpu().numpy().astype(np.float64)
                ff2_b = ff2_mod.bias.detach().cpu().numpy().astype(np.float64) if ff2_mod.bias is not None else None
                ln2_w = out_ln_mod.weight.detach().cpu().numpy().astype(np.float64)
                ln2_b = out_ln_mod.bias.detach().cpu().numpy().astype(np.float64)

                # Run MLP quantization pipeline
                dequant_output = quantize_mlp_block(
                    ln1_input_np=cap['ln1_input'],
                    ln1_output_np=cap['ln1_output'],
                    ln1_weight_np=cap['ln1_weight'],
                    ln1_bias_np=cap['ln1_bias'],
                    ff1_weight_np=ff1_w,
                    ff1_bias_np=ff1_b,
                    ff2_weight_np=ff2_w,
                    ff2_bias_np=ff2_b,
                    ln2_weight_np=ln2_w,
                    ln2_bias_np=ln2_b,
                    ln_table=self.ln_table,
                    gelu_table=self.gelu_table,
                    k_bits=K_BITS,
                    eps=EPS,
                )

                result = torch.tensor(
                    dequant_output, dtype=torch.float32, device=device,
                )
                if isinstance(output, tuple):
                    return (result,) + output[1:]
                return result
            return hook_fn

        h = out_ln.register_forward_hook(
            make_mlp_hook(layer_idx, ff1_dense, ff2_dense, out_ln)
        )
        self.handles.append(h)

    def _setup_independent_hooks(self, layer_idx, layer):
        """Set up hooks using pre-computed MLP witnesses (non-cascaded)."""
        if layer_idx not in self.mlp_dequant_independent:
            return
        if args.model == "bert":
            out_ln = layer.output.LayerNorm
        else:
            out_ln = layer.output.LayerNorm
        h = out_ln.register_forward_hook(
            make_replacement_hook(self.mlp_dequant_independent[layer_idx])
        )
        self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.captured.clear()


# ── Metrics ──────────────────────────────────────────────────────────────
def compute_topk_agreement(float_logits, quant_logits, mask_positions=None):
    if mask_positions is not None:
        fl = float_logits[mask_positions]
        ql = quant_logits[mask_positions]
    else:
        B, S, V = float_logits.shape
        fl = float_logits.reshape(-1, V)
        ql = quant_logits.reshape(-1, V)

    if fl.shape[0] == 0:
        return {"top1_agreement": 0.0, "top5_agreement": 0.0, "total_positions": 0}

    float_top1 = fl.argmax(dim=-1)
    quant_top1 = ql.argmax(dim=-1)
    quant_top5 = ql.topk(5, dim=-1).indices

    N = float_top1.shape[0]
    top1_agree = (float_top1 == quant_top1).sum().item()
    top5_agree = (quant_top5 == float_top1.unsqueeze(-1)).any(dim=-1).sum().item()

    return {
        "top1_agreement": top1_agree / N,
        "top5_agreement": top5_agree / N,
        "total_positions": N,
    }


def compute_quantization_error(float_logits, quant_logits, mask_positions=None):
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
    cos_sim = nn.functional.cosine_similarity(fl, ql, dim=-1).mean().item()

    return {
        "logit_mse": mse,
        "logit_rmse": rmse,
        "logit_max_abs_error": max_abs,
        "logit_cosine_similarity": cos_sim,
    }


# ── Batch Evaluation ────────────────────────────────────────────────────
def evaluate_batch(model, tokenizer, batch, batch_idx, ln_table, gelu_table):
    # 1. Float baseline
    with torch.no_grad():
        float_logits = model(**batch).logits

    # 2. Load attention witnesses
    attn_layer_files = find_attn_witness_files(batch_idx)
    attn_dequant = {}
    for layer_idx, witness_path in attn_layer_files:
        attn_dequant[layer_idx] = dequantize_attention_output(witness_path)

    if not attn_dequant:
        print(f"  [WARNING] No attention witness files for batch {batch_idx}")
        return None

    # 3. Load MLP witnesses (for independent mode)
    mlp_layer_files = find_mlp_files(batch_idx)
    mlp_dequant_independent = {}
    for layer_idx, debug_path, witness_path in mlp_layer_files:
        mlp_dequant_independent[layer_idx] = dequantize_mlp_output(
            debug_path, witness_path,
        )

    print(f"  Loaded: {len(attn_dequant)} attn layers, "
          f"{len(mlp_dequant_independent)} MLP layers")

    # 4. Run each evaluation mode
    results = {"batch_idx": batch_idx}
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (batch["input_ids"] == mask_token_id) if mask_token_id else None
    attn_mask = batch["attention_mask"].bool()

    for mode_name in ["attn_cascade", "full_cascade", "independent"]:
        manager = CascadeHookManager(
            model, attn_dequant, mlp_dequant_independent,
            ln_table, gelu_table, mode=mode_name,
        )
        manager.setup_hooks()

        with torch.no_grad():
            logits = model(**batch).logits

        manager.remove_hooks()

        r = {}
        r["all_positions"] = compute_topk_agreement(float_logits, logits, attn_mask)
        r["all_positions_error"] = compute_quantization_error(float_logits, logits, attn_mask)

        if mask_positions is not None and mask_positions.any():
            r["mask_positions"] = compute_topk_agreement(float_logits, logits, mask_positions)
            r["mask_positions_error"] = compute_quantization_error(float_logits, logits, mask_positions)

            mask_fl = float_logits[mask_positions]
            mask_ql = logits[mask_positions]
            preds = []
            for i in range(mask_fl.shape[0]):
                preds.append({
                    "float_top5": tokenizer.convert_ids_to_tokens(mask_fl[i].topk(5).indices.tolist()),
                    "quant_top5": tokenizer.convert_ids_to_tokens(mask_ql[i].topk(5).indices.tolist()),
                })
            r["mask_predictions"] = preds

        results[mode_name] = r

    return results


def print_mode_results(mode_name, r):
    ap = r["all_positions"]
    ae = r.get("all_positions_error", {})
    print(f"\n  [{mode_name}] All Positions (n={ap['total_positions']})")
    print(f"    Top-1 Agreement: {ap['top1_agreement']:.4f} ({ap['top1_agreement']*100:.2f}%)")
    print(f"    Top-5 Agreement: {ap['top5_agreement']:.4f} ({ap['top5_agreement']*100:.2f}%)")
    if ae:
        print(f"    Logit CosSim:    {ae['logit_cosine_similarity']:.8f}")
        print(f"    Logit RMSE:      {ae['logit_rmse']:.6e}")
        print(f"    Logit MaxAbs:    {ae['logit_max_abs_error']:.6e}")

    if "mask_positions" in r:
        mp = r["mask_positions"]
        me = r.get("mask_positions_error", {})
        print(f"  [{mode_name}] [MASK] Positions (n={mp['total_positions']})")
        print(f"    Top-1 Agreement: {mp['top1_agreement']:.4f} ({mp['top1_agreement']*100:.2f}%)")
        print(f"    Top-5 Agreement: {mp['top5_agreement']:.4f} ({mp['top5_agreement']*100:.2f}%)")
        if me:
            print(f"    Logit CosSim:    {me['logit_cosine_similarity']:.8f}")

    if "mask_predictions" in r:
        for i, pred in enumerate(r["mask_predictions"]):
            print(f"    [MASK #{i}]  Float: {pred['float_top5']}  |  Quant: {pred['quant_top5']}")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Log file setup ────────────────────────────────────────────────
    if args.log_file:
        log_path = args.log_file
    else:
        os.makedirs(WITNESS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(WITNESS_DIR, f"cascaded_eval_{ts}.log")

    tee = TeeWriter(log_path, sys.stdout)
    sys.stdout = tee
    print(f"[Log] Saving output to: {log_path}")

    print(f"Loading {args.model} model...")
    model, tokenizer = load_model_and_tokenizer()
    batches = make_batches(tokenizer)

    print("Loading lookup tables...")
    ln_table, gelu_table = load_tables()

    all_results = []
    agg = {
        mode: {
            "top1_all": [], "top5_all": [], "top1_mask": [], "top5_mask": [],
            "cossim_all": [], "rmse_all": [], "maxabs_all": [],
            "cossim_mask": [], "rmse_mask": [], "maxabs_mask": [],
        }
        for mode in ["attn_cascade", "full_cascade", "independent"]
    }

    batch_indices = (
        [args.batch_idx] if args.batch_idx is not None
        else range(len(batches))
    )

    for bi in batch_indices:
        if bi >= len(batches):
            print(f"[WARNING] batch_idx {bi} out of range (max {len(batches)-1})")
            continue

        print(f"\n{'='*60}")
        print(f"Batch {bi}")
        print(f"{'='*60}")

        result = evaluate_batch(model, tokenizer, batches[bi], bi, ln_table, gelu_table)
        if result is None:
            continue

        all_results.append(result)

        for mode_name in ["attn_cascade", "full_cascade", "independent"]:
            if mode_name not in result:
                continue
            r = result[mode_name]
            print_mode_results(mode_name, r)

            ap = r["all_positions"]
            ap_err = r.get("all_positions_error", {})
            agg[mode_name]["top1_all"].append(ap["top1_agreement"])
            agg[mode_name]["top5_all"].append(ap["top5_agreement"])
            if ap_err:
                agg[mode_name]["cossim_all"].append(ap_err["logit_cosine_similarity"])
                agg[mode_name]["rmse_all"].append(ap_err["logit_rmse"])
                agg[mode_name]["maxabs_all"].append(ap_err["logit_max_abs_error"])
            if "mask_positions" in r:
                mp = r["mask_positions"]
                mp_err = r.get("mask_positions_error", {})
                agg[mode_name]["top1_mask"].append(mp["top1_agreement"])
                agg[mode_name]["top5_mask"].append(mp["top5_agreement"])
                if mp_err:
                    agg[mode_name]["cossim_mask"].append(mp_err["logit_cosine_similarity"])
                    agg[mode_name]["rmse_mask"].append(mp_err["logit_rmse"])
                    agg[mode_name]["maxabs_mask"].append(mp_err["logit_max_abs_error"])

    # ── Aggregate ────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS ({len(all_results)} batches, seq_len={SEQ_LEN})")
        print(f"{'='*60}")

        for mode_name in ["attn_cascade", "full_cascade", "independent"]:
            a = agg[mode_name]
            if not a["top1_all"]:
                continue
            print(f"\n  [{mode_name}]")
            print(f"    All Positions:")
            print(f"      Top-1:   {np.mean(a['top1_all'])*100:.2f}% "
                  f"\u00b1 {np.std(a['top1_all'])*100:.2f}%")
            print(f"      Top-5:   {np.mean(a['top5_all'])*100:.2f}% "
                  f"\u00b1 {np.std(a['top5_all'])*100:.2f}%")
            print(f"      CosSim:  {np.mean(a['cossim_all']):.8f} "
                  f"\u00b1 {np.std(a['cossim_all']):.8f}")
            print(f"      RMSE:    {np.mean(a['rmse_all']):.6e} "
                  f"\u00b1 {np.std(a['rmse_all']):.6e}")
            print(f"      MaxAbs:  {np.mean(a['maxabs_all']):.6e} "
                  f"\u00b1 {np.std(a['maxabs_all']):.6e}")
            if a["top1_mask"]:
                print(f"    [MASK] Positions:")
                print(f"      Top-1:   {np.mean(a['top1_mask'])*100:.2f}% "
                      f"\u00b1 {np.std(a['top1_mask'])*100:.2f}%")
                print(f"      Top-5:   {np.mean(a['top5_mask'])*100:.2f}% "
                      f"\u00b1 {np.std(a['top5_mask'])*100:.2f}%")
                print(f"      CosSim:  {np.mean(a['cossim_mask']):.8f} "
                      f"\u00b1 {np.std(a['cossim_mask']):.8f}")
                print(f"      RMSE:    {np.mean(a['rmse_mask']):.6e} "
                      f"\u00b1 {np.std(a['rmse_mask']):.6e}")
                print(f"      MaxAbs:  {np.mean(a['maxabs_mask']):.6e} "
                      f"\u00b1 {np.std(a['maxabs_mask']):.6e}")

        # Save JSON
        save_data = {
            "meta": {
                "model": args.model,
                "seq_len": SEQ_LEN,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "num_landmarks": args.num_landmarks,
                "num_batches": len(all_results),
                "witness_dir": args.witness_dir,
                "k_bits": K_BITS,
                "eps": EPS,
            },
            "aggregate": {},
            "per_batch": all_results,
        }
        for mode_name in ["attn_cascade", "full_cascade", "independent"]:
            a = agg[mode_name]
            if not a["top1_all"]:
                continue
            save_data["aggregate"][mode_name] = {
                "top1_all_mean": float(np.mean(a["top1_all"])),
                "top1_all_std": float(np.std(a["top1_all"])),
                "top5_all_mean": float(np.mean(a["top5_all"])),
                "top5_all_std": float(np.std(a["top5_all"])),
                "cossim_all_mean": float(np.mean(a["cossim_all"])),
                "cossim_all_std": float(np.std(a["cossim_all"])),
                "rmse_all_mean": float(np.mean(a["rmse_all"])),
                "rmse_all_std": float(np.std(a["rmse_all"])),
                "maxabs_all_mean": float(np.mean(a["maxabs_all"])),
                "maxabs_all_std": float(np.std(a["maxabs_all"])),
                "top1_mask_mean": float(np.mean(a["top1_mask"])) if a["top1_mask"] else None,
                "top1_mask_std": float(np.std(a["top1_mask"])) if a["top1_mask"] else None,
                "top5_mask_mean": float(np.mean(a["top5_mask"])) if a["top5_mask"] else None,
                "top5_mask_std": float(np.std(a["top5_mask"])) if a["top5_mask"] else None,
                "cossim_mask_mean": float(np.mean(a["cossim_mask"])) if a["cossim_mask"] else None,
                "cossim_mask_std": float(np.std(a["cossim_mask"])) if a["cossim_mask"] else None,
                "rmse_mask_mean": float(np.mean(a["rmse_mask"])) if a["rmse_mask"] else None,
                "rmse_mask_std": float(np.std(a["rmse_mask"])) if a["rmse_mask"] else None,
                "maxabs_mask_mean": float(np.mean(a["maxabs_mask"])) if a["maxabs_mask"] else None,
                "maxabs_mask_std": float(np.std(a["maxabs_mask"])) if a["maxabs_mask"] else None,
            }

        output_path = os.path.join(
            WITNESS_DIR, f"quantization_eval_cascaded_n{SEQ_LEN}.json",
        )
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    # ── Close log ─────────────────────────────────────────────────────
    sys.stdout = tee._stdout
    tee.close()
    print(f"[Log] Log saved to: {log_path}")
