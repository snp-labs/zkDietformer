#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import hashlib
import argparse

def get_quantization_params(data, bits=8):
    """
    [JKC+18] 방식: 데이터의 범위를 bits(보통 8bit) 정수 영역으로 매핑
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    # 8bit 기준 (0 ~ 255)
    qmin = 0
    qmax = (1 << bits) - 1
    
    # Scale 계산: s = (max - min) / (qmax - qmin)
    scale = (data_max - data_min) / (qmax - qmin)
    if scale == 0:
        scale = 1e-9
        
    # Zero-point 계산: z = qmin - min / s
    initial_zero_point = qmin - data_min / scale
    zero_point = np.clip(np.round(initial_zero_point), qmin, qmax).astype(np.int32)
    
    return float(scale), int(zero_point)


def quantize_tensor(data, scale, zero_point, bits=8):
    """실수 데이터를 정수(Q)로 변환: Q = floor(Y/s + z)"""
    qmin = 0
    qmax = (1 << bits) - 1
    q_data = np.floor(data / scale + zero_point)
    return np.clip(q_data, qmin, qmax).astype(np.int32)


def get_m_constant(s_w, s_x, s_y, k=22):
    """ZEN 서킷에서 사용할 고정 정수 M 계산: M = floor(2^k * (s_w * s_x / s_y))"""
    return int(np.floor((1 << k) * (s_w * s_x) / s_y))


def sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits=22):
    """
    ZEN의 Sign-bit grouping 및 Remainder-based verification
    """
    n = Q_w.shape[-1] # 내적 차원
    
    # G1, G2, G3 계산 (Association 활용으로 음수 제거) 
    G1 = np.matmul(Q_w.astype(np.int64), Q_x.astype(np.int64))
    G2 = z_x * np.sum(Q_w.astype(np.int64), axis=-1, keepdims=True)
    G3 = z_w * np.sum(Q_x.astype(np.int64), axis=-2, keepdims=True)
    
    # M' 계산 
    M_prime = int(np.floor((z_y * (1 << k_bits)) / M))
    
    # ZEN : QY * 2^k + R = M * (G1 + n*zw*zx + M' - G2 - G3) 
    constant_term = n * z_w * z_x
    inner_val = (G1 + constant_term + M_prime - G2 - G3)
    numerator = M * inner_val
    
    Q_y_sim = (numerator >> k_bits)
    R = numerator % (1 << k_bits) # Remainder Matrix 
    
    return Q_y_sim.astype(np.int32), R.astype(np.int64), M_prime

def _to_u64(arr):
    return arr.astype(np.uint64)