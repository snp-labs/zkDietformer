#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import hashlib
import argparse

def get_quantization_params(data, bits=8):
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


def quantize_tensor(data, scale, zero_point, bits=8):
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


def sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits=22):
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
    
    return Q_y_sim.astype(np.int32), R.astype(np.int64), M_prime


def get_softmax_index_multiplier(s_q, s_k, k_bits=22):
    # positive check
    assert s_q > 0 and s_k > 0, f"Scales must be positive: s_q={s_q}, s_k={s_k}"
    
    real_unit_scale = (s_q * s_k) / (1 << k_bits)
    multiplier = int(np.floor(real_unit_scale * (1 << k_bits)))
    
    assert multiplier >= 0, f"Softmax multiplier must be non-negative, got {multiplier}"
    return multiplier

def get_rms_index_multiplier(s_ms, k_bits=22):
    # positive check
    assert s_ms > 0, f"Scales must be positive: s_ri={s_ms}"

    multiplier = int(np.floor(s_ms * (1 << k_bits)))
    assert multiplier >= 0, f"Softmax multiplier must be non-negative, got {multiplier}"
    return multiplier

def get_gelu_index_multiplier(s_in, k_bits=22):
    # GELU_INDEX_SCALE은 gelu_table.py와 동일한 1/512 사용
    GELU_INDEX_SCALE = 1.0 / (1 << 9)
    # M = (s_in / GELU_INDEX_SCALE) * 2^k
    return int(np.floor((s_in / GELU_INDEX_SCALE) * (1 << k_bits)))