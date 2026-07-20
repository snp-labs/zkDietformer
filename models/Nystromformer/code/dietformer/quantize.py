#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import hashlib
import argparse
from gelu_table import GELU_INDEX_SCALE

def get_quantization_params(data, bits=12):
    data_min = np.min(data)
    data_max = np.max(data)
    # nudge
    rmin = min(0.0, data_min)
    rmax = max(0.0, data_max)
    
    qmin = 0
    qmax = (1 << bits) - 1
    
    scale = (rmax - rmin) / (qmax - qmin)
    if scale == 0:
        scale = 1e-9
        
    initial_zero_point = qmin - rmin / scale
    nudged_zero_point = np.round(initial_zero_point)
    
    if nudged_zero_point < qmin:
        nudged_zero_point = qmin
    elif nudged_zero_point > qmax:
        nudged_zero_point = qmax
        
    # positive check
    assert scale > 0, f"Scale must be positive, got {scale}"
    assert nudged_zero_point >= 0, f"Zero-point must be non-negative, got {nudged_zero_point}"

    return float(scale), int(nudged_zero_point)


def quantize_tensor(data, scale, zero_point, bits=12):
    # positive check
    assert scale > 0, f"Scale must be positive, got {scale}"
    assert zero_point >= 0, f"Zero-point must be non-negative, got {zero_point}"

    qmin = 0
    qmax = (1 << bits) - 1
    q_data = np.floor(data / scale + zero_point)
    return np.clip(q_data, qmin, qmax).astype(np.int32)


def get_m_constant(s_w, s_x, s_y, k=22):
    # positive check
    assert s_w > 0 and s_x > 0 and s_y > 0, f"All scales must be positive: s_w={s_w}, s_x={s_x}, s_y={s_y}"
    
    M = int(np.floor((1 << k) * (s_w * s_x) / s_y))
    assert M >= 0, f"Multiplier M must be non-negative, got {M}"
    return M


def sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits=26):
    # positive check
    assert M > 0, f"Multiplier M must be positive for matmul rescaling, got {M}"
    assert z_w >= 0 and z_x >= 0 and z_y >= 0, f"Zero-points must be non-negative: z_w={z_w}, z_x={z_x}, z_y={z_y}"
    
    n = Q_w.shape[-1] 
    
    G1 = np.matmul(Q_w.astype(np.int64), Q_x.astype(np.int64))
    G2 = z_x * np.sum(Q_w.astype(np.int64), axis=-1, keepdims=True)
    G3 = z_w * np.sum(Q_x.astype(np.int64), axis=-2, keepdims=True)
    
    M_prime = int(np.floor((z_y * (1 << k_bits)) / M))
    
    constant_term = n * z_w * z_x
    inner_val = (G1 + constant_term + M_prime - G2 - G3)
    numerator = M * inner_val
    
    Q_y_sim = (numerator >> k_bits)
    R = numerator % (1 << k_bits)

    # Overflow guard: detect values that would have been truncated by int32
    max_abs = np.max(np.abs(Q_y_sim))
    if max_abs > np.iinfo(np.int32).max:
        print(f"[WARN] Q_y_sim exceeds int32 range: max_abs={max_abs}, shape={Q_y_sim.shape}")

    return Q_y_sim.astype(np.int64), R.astype(np.int64), M_prime


def get_softmax_index_multiplier(s_q, s_k, k_bits=26):
    # positive check
    assert s_q > 0 and s_k > 0, f"Scales must be positive: s_q={s_q}, s_k={s_k}"
    
    real_unit_scale = (s_q * s_k) / (1 << k_bits)
    multiplier = int(np.floor(real_unit_scale * (1 << k_bits)))
    
    assert multiplier >= 0, f"Softmax multiplier must be non-negative, got {multiplier}"
    return multiplier

def get_rms_index_multiplier(s_ms, k_bits=26):
    # positive check
    assert s_ms > 0, f"Scales must be positive: s_ri={s_ms}"

    multiplier = int(np.floor(s_ms * (1 << k_bits)))
    assert multiplier >= 0, f"Softmax multiplier must be non-negative, got {multiplier}"
    return multiplier

def get_gelu_index_multiplier(s_in, k_bits=26):
    # (s_in / GELU_INDEX_SCALE)를 고정소수점 multiplier로 변환.
    return int(np.floor((s_in / GELU_INDEX_SCALE) * (1 << k_bits)))

def ensure_nonneg_matmul(Q_w, z_w, Q_x, z_x, s_w, s_x, s_y, z_y, k_bits=26):
    """ZEN matmul with dynamic z_y adjustment to ensure Q_y_sim >= 0.
    Required for inputs to softmax, where negative Q_sim breaks the
    scaling identity (np.clip on table_idx violates the constraint).
    """
    M = get_m_constant(s_w, s_x, s_y, k=k_bits)
    Q_y_sim, R, M_prime = sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits)
    min_val = int(np.min(Q_y_sim))
    if min_val < 0:
        z_y += (-min_val + 1)
        Q_y_sim, R, M_prime = sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits)
        while np.min(Q_y_sim) < 0:
            z_y += 1
            Q_y_sim, R, M_prime = sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits)
        print(f"  [INFO] z_y adjusted to {z_y} to ensure Q_sim >= 0 (was min={min_val})")
    return Q_y_sim, R, M_prime, M, z_y


def check_neg_torch(tensor, label="Tensor"):
    neg_mask = tensor < 0
    count = neg_mask.sum().item()
    
    print(f"[{label}] 음수 개수: {count}")