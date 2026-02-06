import numpy as np
import os
import math
import argparse

# ZEN Matmul의 결과 정수 X = (Q_sim << k_bits) + R_mat을 분해
K_SEGMENTS = 5    
LIMB_BITS = 16
LIMB_SIZE = 1 << LIMB_BITS
UINT64_MAX = (1 << 64) - 1
FRAC_BITS = 32

def float_to_fixed(x, frac_bits):
    # 1. NaN/Inf 사전 방어
    x_safe = np.nan_to_num(x, nan=0.0, posinf=float(UINT64_MAX), neginf=0.0)
    
    # 2. 스케일링 (float64 범위 내 연산)
    scaled = x_safe * float(1 << frac_bits)
    
    # 3. UINT64 범위 클리핑
    scaled = np.clip(scaled, 0, float(UINT64_MAX))
    return scaled.astype(np.uint64)

def generate_softmax_tables(out_dir, s_q, s_k, k_bits=22, frac_bits=32):
    os.makedirs(out_dir, exist_ok=True)
    
    unit_scale = (float(s_q) * float(s_k)) / float(1 << k_bits)
    if unit_scale == 0:
        unit_scale = 1e-15
    
    tables = []
    for k in range(K_SEGMENTS):
        weight = float(1 << (LIMB_BITS * k))
        limb_vals = np.arange(LIMB_SIZE, dtype=np.float64)
        real_x = limb_vals * weight * unit_scale
        
        real_x = np.clip(real_x, None, 20.0)
        exps = np.exp(real_x)
        
        fixed_exps = float_to_fixed(exps, frac_bits)
        tables.append(fixed_exps)
        
        print(f"Segment {k}: weight=2^{LIMB_BITS*k}, unit_scale={unit_scale:.20f}")

    stacked = np.stack(tables, axis=0)
    out_path = os.path.join(out_dir, "softmax_tables.npz")
    np.savez_compressed(
        out_path,
        tables=stacked,
        frac_bits=np.int32(frac_bits),
        s_q=np.float64(s_q),
        s_k=np.float64(s_k),
        k_bits=np.int32(k_bits)
    )
    print(f"✅ ZEN Softmax tables saved to {out_path}")
    return out_path


def extract_exp_from_segments(Q_sim, R_mat, sm_tables, k_bits):

    X = (Q_sim.astype(object) << k_bits) + R_mat.astype(object)
    
    limbs = []
    mask = (1 << LIMB_BITS) - 1
    for i in range(K_SEGMENTS):
        limb = (X >> (LIMB_BITS * i)) & mask
        limbs.append(limb.astype(np.int64))
    
    limbs_stacked = np.stack(limbs, axis=0)
    
    exp_limbs = []
    for k in range(K_SEGMENTS):
        val_k = sm_tables[k][limbs_stacked[k].astype(int)]
        exp_limbs.append(val_k.astype(np.uint64))
        
    exp_limbs_stacked = np.stack(exp_limbs, axis=0)
    
    return limbs_stacked, exp_limbs_stacked


def combine_exp_limbs(exp_limbs_list, frac_bits=32):
    exp_total = exp_limbs_list[0].astype(object)
    
    inter_quotients = [] # 중간 몫 (다음 곱셈의 입력)
    inter_remainders = [] # 중간 나머지 (서킷 검증용)

    # 2. 순차적 곱셈: exp_total = exp_total * e^xk / 2^frac_bits
    for k in range(1, len(exp_limbs_list)):
        # (A * B) 실행 -> 소수점 비트가 2 * frac_bits가 됨
        product = exp_total * exp_limbs_list[k].astype(object)
        
        # 소수점 위치를 원래대로(frac_bits) 돌리기 위한 연산
        # 서킷에서는 나누기 대신 이 몫(Q)과 나머지(R)의 관계를 검증함
        quotient = product >> frac_bits
        remainder = product % (1 << frac_bits)
        
        inter_remainders.append(remainder)
        if k < len(exp_limbs_list) - 1:
            inter_quotients.append(quotient)
        
        exp_total = quotient

    return exp_total, inter_quotients, inter_remainders