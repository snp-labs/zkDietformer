"""
Generate a synthetic small-sequence RoBERTa attention witness NPZ for testing.
Uses B=1, H=2, S=8, head_dim=8 so the circuit runs in seconds.
"""
import os
import sys
import math
import numpy as np

# ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul, get_softmax_index_multiplier,
)
from softmax_table import (
    generate_softmax_tables, extract_exp_from_segments, combine_exp_limbs,
    FRAC_BITS, LIMB_BITS, K_SEGMENTS,
)

# ── Dimensions ───────────────────────────────────────────────────────────────
B, H, S, D = 1, 2, 8, 8   # batch, heads, seq_len, head_dim
K_BITS = 22
OUT_DIR = "outputs_bert_small"
OUT_NAME = "outputs_bert_small-batch0-roberta.encoder.layer.0.attention.self.witness.npz"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# ── Synthetic Q/K/V (scaled like real attention, already scaled by 1/sqrt(sqrt(D))) ─
scale = 1.0 / math.sqrt(math.sqrt(D))
Q_real = rng.normal(0, 0.2, (B, H, S, D)).astype(np.float32) * scale
K_real = rng.normal(0, 0.2, (B, H, S, D)).astype(np.float32) * scale
V_real = rng.normal(0, 0.2, (B, H, S, D)).astype(np.float32)
mask   = np.ones((B, S), dtype=np.int64)   # all positions valid

# ── Quantise Q, K, V ─────────────────────────────────────────────────────────
s_q, z_q = get_quantization_params(Q_real)
Q_q = quantize_tensor(Q_real, s_q, z_q).astype(np.int64)

s_k, z_k = get_quantization_params(K_real)
Q_k = quantize_tensor(K_real, s_k, z_k).astype(np.int64)

s_v, z_v = get_quantization_params(V_real)
Q_v = quantize_tensor(V_real, s_v, z_v).astype(np.int64)

# ── Attention scores Q*K^T via sign_bit_grouping_matmul ──────────────────────
K_T = np.swapaxes(Q_k, -1, -2)

Y_int_prelim, _, _ = sign_bit_grouping_matmul(
    Q_q, z_q, K_T, z_k, z_y=0, M=(1 << K_BITS), k_bits=K_BITS
)
Y_real = Y_int_prelim.astype(np.float64) * s_q * s_k

s_y, z_y = get_quantization_params(Y_real)
z_y += 10
M_attn_scores = get_m_constant(s_q, s_k, s_y, k=K_BITS)

Q_score_sim, R_matmul_score, M_prime_score = sign_bit_grouping_matmul(
    Q_q, z_q, K_T, z_k, z_y, M_attn_scores, K_BITS
)

# ── Softmax tables ───────────────────────────────────────────────────────────
table_path = generate_softmax_tables(OUT_DIR, K_BITS, FRAC_BITS)
sm_tables = np.load(table_path)['tables']

# ── Softmax ──────────────────────────────────────────────────────────────────
mask_4d = mask[:, None, None, :].astype(np.int64)
Q_score_sim_masked = Q_score_sim * mask_4d

M_sm = get_softmax_index_multiplier(s_q, s_k, K_BITS)
limbs, exp_limbs, r_sm = extract_exp_from_segments(
    Q_score_sim_masked, R_matmul_score, sm_tables, M_sm, K_BITS
)
exp_total, Q_exp, R_exp = combine_exp_limbs(exp_limbs, FRAC_BITS)
exp_sum = np.sum(exp_total, axis=-1, keepdims=True)

P_attn     = (exp_total << FRAC_BITS) // exp_sum
R_div_attn = (exp_total << FRAC_BITS) % exp_sum

# ── Output X = P * V ─────────────────────────────────────────────────────────
# Compute expected X from float path for quantisation
P_float = P_attn.astype(np.float64) / (1 << FRAC_BITS)
X_real = np.matmul(P_float, V_real)

s_x, z_sim_x = get_quantization_params(X_real)
z_p = 0
s_p = 1.0 / (1 << FRAC_BITS)
M_final = get_m_constant(s_p, s_v, s_x, k=K_BITS)

Q_sim_x, R_matmul_x, M_prime_x = sign_bit_grouping_matmul(
    P_attn.astype(np.int64), z_p, Q_v, z_v, z_sim_x, M_final, K_BITS
)

# ── Save witness NPZ ─────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, OUT_NAME)
np.savez_compressed(
    out_path,
    Q_q=Q_q, z_q=np.uint64(z_q),
    Q_k=Q_k, z_k=np.uint64(z_k),
    Q_v=Q_v, z_v=np.uint64(z_v),

    Q_sim_attn=Q_score_sim.astype(np.int64),
    R_matmul_attn=R_matmul_score.astype(np.int64),
    M_attn=np.uint64(M_attn_scores),
    M_prime_attn=np.uint64(M_prime_score),
    z_sim_attn=np.uint64(z_y),

    M_sm=np.uint64(M_sm),
    limbs=limbs.astype(np.int64),
    exp_limbs=exp_limbs.astype(np.uint64),
    r_sm=r_sm.astype(np.int64),
    Q_exp=np.stack(Q_exp, axis=0).astype(np.uint64),
    R_exp=np.stack(R_exp, axis=0).astype(np.uint64),
    exp_total=exp_total.astype(np.uint64),
    P_attn=P_attn.astype(np.uint64),
    R_div_attn=R_div_attn.astype(np.uint64),
    exp_sum=exp_sum.astype(np.uint64),

    Q_sim_x=Q_sim_x.astype(np.int64),
    R_matmul_x=R_matmul_x.astype(np.int64),
    M_x=np.uint64(M_final),
    M_prime_x=np.uint64(M_prime_x),
    z_sim_x=np.uint64(z_sim_x),

    sm_tables=sm_tables,
    frac_bits=np.uint32(FRAC_BITS),
    limb_bits=np.uint32(LIMB_BITS),
    k_segments=np.uint32(K_SEGMENTS),
    k_bits=np.uint32(K_BITS),
    mask=mask.astype(np.int64),
)
print(f"Saved: {out_path}")
print(f"Shapes: Q_q={Q_q.shape}, P_attn={P_attn.shape}, R_exp={np.stack(R_exp).shape}")
