"""
Generate a synthetic small-scale RoBERTa normalization+FFN witness NPZ for testing.
Uses B=1, S=4, hidden_dim=8, ffn_dim=16 so the circuit runs in seconds.

Output goes to outputs_bert_small/ as
  small_bert_norm-batch0-layer0.norm.witness.npz
which the bert_mlp_test discovers via the .norm.witness.npz suffix.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from bert_normalization_pipeline import process_standard_normalization_debug_npz

# ── Dimensions ────────────────────────────────────────────────────────────────
B, S, H, F_DIM = 1, 4, 8, 16   # batch, seq_len, hidden_dim, ffn_dim
K_BITS = 22
OUT_DIR = "outputs_bert_small"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# ── Synthetic float data ──────────────────────────────────────────────────────
rms_input = rng.normal(0, 0.5, (B, S, H)).astype(np.float32)
rms_weight = rng.uniform(0.8, 1.2, (H,)).astype(np.float32)

# RMSNorm 1: output = weight * x / rms(x)
x_t = torch.from_numpy(rms_input)
w_t = torch.from_numpy(rms_weight)
rms_output = (w_t * x_t / (x_t.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)).numpy()
ms = x_t.pow(2).mean(dim=-1, keepdim=True).numpy()

# Linear 1 (up-projection)
linear_weight = rng.normal(0, 0.1, (F_DIM, H)).astype(np.float32)
linear_bias   = rng.normal(0, 0.01, (F_DIM,)).astype(np.float32)
linear_output = rms_output.reshape(B * S, H) @ linear_weight.T + linear_bias
linear_output = linear_output.reshape(B, S, F_DIM)

# GeLU
gelu_output = F.gelu(torch.from_numpy(linear_output)).numpy()

# Linear 2 (down-projection)
linear_weight_ff2 = rng.normal(0, 0.1, (H, F_DIM)).astype(np.float32)
linear_bias_ff2   = rng.normal(0, 0.01, (H,)).astype(np.float32)
linear_output_ff2 = gelu_output.reshape(B * S, F_DIM) @ linear_weight_ff2.T + linear_bias_ff2
linear_output_ff2 = linear_output_ff2.reshape(B, S, H)

# RMSNorm 2: residual add then norm
rms_input_rms2 = (linear_output_ff2 + rms_output).astype(np.float32)
rms_weight_rms2 = rng.uniform(0.8, 1.2, (H,)).astype(np.float32)

x2_t = torch.from_numpy(rms_input_rms2)
w2_t = torch.from_numpy(rms_weight_rms2)
rms_output_rms2 = (w2_t * x2_t / (x2_t.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)).numpy()
ms_rms2 = x2_t.pow(2).mean(dim=-1, keepdim=True).numpy()

# ── Build debug NPZ and run the pipeline ────────────────────────────────────
debug_path   = os.path.join(OUT_DIR, "small_bert_norm-batch0-layer0.norm_debug.npz")
witness_path = os.path.join(OUT_DIR, "small_bert_norm-batch0-layer0.norm.witness.npz")

np.savez_compressed(
    debug_path,
    rms_input=rms_input,
    ms=ms,
    rms_weight=rms_weight,
    rms_output=rms_output,
    linear_weight=linear_weight,
    linear_output=linear_output,
    linear_bias=linear_bias,
    gelu_output=gelu_output,
    linear_weight_ff2=linear_weight_ff2,
    linear_output_ff2=linear_output_ff2,
    linear_bias_ff2=linear_bias_ff2,
    rms_input_rms2=rms_input_rms2,
    ms_rms2=ms_rms2,
    rms_weight_rms2=rms_weight_rms2,
    rms_output_rms2=rms_output_rms2,
)
print(f"Saved debug NPZ: {debug_path}")

process_standard_normalization_debug_npz(debug_path, witness_path, k_bits=K_BITS)
print(f"Saved witness NPZ: {witness_path}")
print(f"Dims: B={B}, S={S}, H={H}, F={F_DIM}")
