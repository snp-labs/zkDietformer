import numpy as np
import os
import math
import argparse

# ZEN Matmul의 결과 정수 X = (Q_sim << k_bits) + R_mat을 분해
K_SEGMENTS = 5    
LIMB_BITS = 16
LIMB_SIZE = 1 << LIMB_BITS
UINT64_MAX = (1 << 64) - 1
FRAC_BITS = 16

def float_to_fixed(x, frac_bits):
    x_safe = np.nan_to_num(x, nan=0.0, posinf=float(UINT64_MAX), neginf=0.0)
    scaled = x_safe * float(1 << frac_bits)
    
    scaled = np.clip(scaled, 0, float(UINT64_MAX))
    return scaled.astype(np.uint64)

def generate_softmax_tables(out_dir, k_bits=26, frac_bits=16):
    os.makedirs(out_dir, exist_ok=True)
    
    tables = []
    for k in range(K_SEGMENTS):
        weight = float(1 << (LIMB_BITS * k))
        limb_vals = np.arange(LIMB_SIZE, dtype=np.float64)
        real_x = limb_vals * weight
        
        real_x = np.clip(real_x, None, 20.0)
        exps = np.exp(real_x)
        
        fixed_exps = float_to_fixed(exps, frac_bits)
        tables.append(fixed_exps)
        
        print(f"Segment {k}: weight=2^{LIMB_BITS*k}")

    stacked = np.stack(tables, axis=0)
    out_path = os.path.join(out_dir, "softmax_tables.npz")
    np.savez_compressed(
        out_path,
        tables=stacked,
        frac_bits=np.int32(frac_bits),
        k_bits=np.int32(k_bits)
    )
    print(f"✅ ZEN Softmax tables saved to {out_path}")
    return out_path


def extract_exp_from_segments(Q_sim, R_mat, sm_tables, M_sm, k_bits):

    X = (Q_sim.astype(object) << k_bits) + R_mat.astype(object)
    scaled_product = X * M_sm # 테이블 인덱스와 맞추기 위하여 곱하긴 곱해야 함.
    scaled_x = scaled_product >> k_bits
    r_sm = scaled_product % (1 << k_bits)

    limbs = []
    mask = (1 << LIMB_BITS) - 1
    for i in range(K_SEGMENTS):
        limb = (scaled_x >> (LIMB_BITS * i)) & mask
        limbs.append(limb.astype(np.int64))
    
    limbs_stacked = np.stack(limbs, axis=0)
    
    exp_limbs = []
    for k in range(K_SEGMENTS):
        val_k = sm_tables[k][limbs_stacked[k].astype(int)]
        exp_limbs.append(val_k.astype(np.uint64))
        
    exp_limbs_stacked = np.stack(exp_limbs, axis=0)
    
    return limbs_stacked, exp_limbs_stacked, r_sm


def combine_exp_limbs(exp_limbs_list, frac_bits=16):
    exp_total = exp_limbs_list[0].astype(object) 
    
    inter_quotients = [] 
    inter_remainders = [] 

    for k in range(1, len(exp_limbs_list)):
        product = exp_total * exp_limbs_list[k].astype(object)
        quotient = product >> frac_bits 
        remainder = product % (1 << frac_bits)
        
        inter_remainders.append(remainder)
        inter_quotients.append(quotient)
        
        exp_total = quotient

    return exp_total, inter_quotients, inter_remainders


# ── V3: Single fractional exp table ──────────────────────────────────────
TABLE_FRAC_BITS = 10  # 1024 steps per unit

def generate_fractional_exp_table(table_frac_bits=TABLE_FRAC_BITS, frac_bits=FRAC_BITS):
    """Single exp table: table[i] = fixed_point(exp(i / 2^table_frac_bits))

    table_frac_bits=10 → 1024 steps/unit, covers [0, 64), ~20480 of 65536 entries
    """
    TABLE_SIZE = 1 << 16  # 65536 entries
    scale = 1 << table_frac_bits
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    real_x = indices / scale
    real_x = np.clip(real_x, None, 20.0)
    exps = np.exp(real_x)
    fixed_exps = (exps * float(1 << frac_bits))
    fixed_exps = np.clip(fixed_exps, 0, float(UINT64_MAX))
    return fixed_exps.astype(np.uint64)


def save_softmax_tables_v3(out_dir, table_frac_bits=TABLE_FRAC_BITS, frac_bits=FRAC_BITS):
    """Save v3 single exp table as softmax_tables.npz (shape 1×65536) for Rust Lasso."""
    os.makedirs(out_dir, exist_ok=True)
    exp_table = generate_fractional_exp_table(table_frac_bits, frac_bits)
    tables_2d = exp_table.reshape(1, -1)  # (1, 65536)
    out_path = os.path.join(out_dir, "softmax_tables.npz")
    np.savez_compressed(out_path, tables=tables_2d,
                        frac_bits=np.int32(frac_bits),
                        table_frac_bits=np.int32(table_frac_bits))
    print(f"✅ v3 softmax table saved to {out_path} — shape {tables_2d.shape}")
    return out_path


def softmax_v3_lookup(Q_sim, R_mat, s_y, k_bits, frac_bits=FRAC_BITS,
                       table_frac_bits=TABLE_FRAC_BITS, exp_table=None, mask=None):
    """v3 single-table softmax lookup.

    Returns: P, R_div, exp_total, exp_sum, M_sm, table_idx, r_sm
    """
    if exp_table is None:
        exp_table = generate_fractional_exp_table(table_frac_bits, frac_bits)
    TABLE_SIZE = len(exp_table)

    M_sm = int(np.floor(s_y * (1 << k_bits) * (1 << table_frac_bits)))
    if M_sm <= 0:
        M_sm = 1

    X = (Q_sim.astype(object) << k_bits) + R_mat.astype(object)
    scaled_product = X * M_sm
    table_idx = (scaled_product >> (2 * k_bits)).astype(np.int64)
    r_sm = scaled_product % (1 << (2 * k_bits))

    table_idx = np.clip(table_idx, 0, TABLE_SIZE - 1)

    exp_total = exp_table[table_idx.astype(int)].astype(object)

    if mask is not None:
        exp_total = exp_total * mask

    exp_sum = np.sum(exp_total, axis=-1, keepdims=True)
    P = (exp_total << frac_bits) // exp_sum
    R_div = (exp_total << frac_bits) % exp_sum

    return P, R_div, exp_total, exp_sum, M_sm, table_idx, r_sm