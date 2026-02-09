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
    
    qmin = 0
    qmax = (1 << bits) - 1
    
    scale = (data_max - data_min) / (qmax - qmin)
    if scale == 0:
        scale = 1e-9
        
    initial_zero_point = qmin - data_min / scale
    zero_point = np.clip(np.round(initial_zero_point), qmin, qmax).astype(np.int32)
    
    return float(scale), int(zero_point)


def quantize_tensor(data, scale, zero_point, bits=8):
    qmin = 0
    qmax = (1 << bits) - 1
    q_data = np.floor(data / scale + zero_point)
    return np.clip(q_data, qmin, qmax).astype(np.int32)


def get_m_constant(s_w, s_x, s_y, k=22):
    return int(np.floor((1 << k) * (s_w * s_x) / s_y))


def sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits=22):
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

