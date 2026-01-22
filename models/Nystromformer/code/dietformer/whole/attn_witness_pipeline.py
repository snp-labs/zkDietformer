#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import hashlib
import argparse

from attention_nystrom_quantize import quantize_with_auto_zero, decompose_targets
from softmax_table import generate_tables_quantized

MASK22 = (1 << 22) - 1
UINT64_MAX = (1 << 64) - 1

# 4차원 배열 targets_obj (B,H,R,C)를 k_segments 개수의 limb로 분해함.
def decompose_to_limbs_array(targets_obj, limb_bits, k_segments):
    shape = targets_obj.shape
    batch, num_head, rows, cols = shape[0], shape[1], shape[2], shape[3]

    limb_mask = (1 << limb_bits) - 1
    limbs = np.zeros((batch, num_head, rows, cols, k_segments), dtype=np.uint64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(rows):
                for j in range(cols):
                    t = int(targets_obj[n,h,i,j])
                    if t < 0:
                        raise ValueError(f"음수값 ({n},{h},{i},{j}): {t}")
                    unsigned = int(t)
                    for k in range(k_segments):
                        limb = (unsigned >> (k * limb_bits)) & limb_mask
                        limbs[n,h,i,j,k] = np.uint64(limb)
    return limbs

# 양자화 시키는 함수
def quantize_matrix_fixed_point(arr_float, scale):
    return np.round(arr_float * float(scale)).astype(np.int64)

# 정수로 저장되어 있는targets_obj로부터 float 소프트맥스 계산
def compute_float_softmax_from_targets_int_rect(targets_obj):
    shape = targets_obj.shape
    batch, num_head, rows, cols = shape[0], shape[1], shape[2], shape[3]

    attn = np.zeros((batch, num_head, rows, cols), dtype=np.float64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(rows):
                row = np.array([int(targets_obj[n,h,i,j]) for j in range(cols)], dtype=np.float64)
                row_shift = row - np.max(row)
                exps = np.exp(row_shift)
                denom = np.sum(exps)
                if denom == 0:
                    s = np.ones_like(exps) / len(exps)
                else:
                    s = exps / denom
                attn[n,h,i,:] = s
    return attn

# 위 함수와 동일, 그러나 정사각형 메트릭스만 허용
def compute_float_softmax_from_targets_int(targets_obj):
    shape = targets_obj.shape
    batch, num_head, rows, cols = shape[0], shape[1], shape[2], shape[3]

    assert rows == cols
    attn = np.zeros((batch, num_head, rows, cols), dtype=np.float64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(rows):
                row = np.array([int(targets_obj[n,h,i,j]) for j in range(cols)], dtype=np.float64)
                row_shift = row - np.max(row)
                exps = np.exp(row_shift)
                denom = np.sum(exps)
                if denom == 0:
                    s = np.ones_like(exps) / len(exps)
                else:
                    s = exps / denom
                attn[n,h,i,:] = s
    return attn

def reconstruct_targets_from_c_div_rem(c, div, rem):
    shape = c.shape
    batch, num_head, rows, cols = shape[0], shape[1], shape[2], shape[3]

    targets = np.zeros((batch, num_head, rows, cols), dtype=object)
    for n in range(batch):
        for h in range(num_head):
            for i in range(rows):
                for j in range(cols):
                    c_val = int(c[n,h,i,j])
                    div_val = int(div[n,h,i,j])
                    rem_val = int(rem[n,h,i,j])
                    div22 = (div_val << 8) + c_val
                    target = (div22 << 22) + rem_val
                    targets[n,h,i,j] = int(target)
    return targets

def process_attn_debug_npz(
    debug_npz_path,
    out_npz_path,
    scale_input = 1<<8,
    frac_bits = 16,
    k_segments = 2,
    limb_bits = 16,
    v_scale = 1,
    inv_scale = 1,
    tables_outdir = "tmp_tables",
    multiplier = 1,
    final_auto_scale = False,   # new flag: allow auto-scaling for final X_q decomposition
):
    debug = np.load(debug_npz_path, allow_pickle=True)
    # 뽑아오려는 데이터의 키값이 없으면 에러처리
    if 'Q_proc' not in debug or 'K_proc' not in debug or 'V' not in debug or 'mask' not in debug:
        raise RuntimeError("debug npz missing required tensors (Q_proc,K_proc,V,mask)")

    Q_proc = debug['Q_proc']  # shape (B,H,S,D)
    K_proc = debug['K_proc']
    mask = debug['mask']  # expect shape (B,S,1) or (B,S)

    # normalize mask shape to (B,S)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask_np = mask.reshape(mask.shape[0], mask.shape[1])
    else:
        mask_np = mask.reshape(mask.shape[0], mask.shape[1])

    B, H, S, D = Q_proc.shape

    # Determine number of landmarks L.
    if 'kernel_2' in debug:
        kernel_2 = debug['kernel_2']  # shape (B,H,L,L)
        L = kernel_2.shape[-1]
    elif 'Q_landmarks' in debug:
        Q_landmarks = debug['Q_landmarks']
        L = Q_landmarks.shape[2]
    else:
        # fallback: assume full-attention (no landmarks) — treat as L = S
        L = S

    # 랜드마크가 존재하지 않으면 랜드마크 계산
    if 'Q_landmarks' in debug:
        Q_landmarks_np = debug['Q_landmarks']
        K_landmarks_np = debug['K_landmarks'] if 'K_landmarks' in debug else None
    else:
        seg = S // L
        Q_landmarks_np = Q_proc.reshape(B, H, L, seg, D).mean(axis=3)
        K_landmarks_np = K_proc.reshape(B, H, L, seg, D).mean(axis=3)

    # 양자화 시키기
    # kernel_1
    a_q_k1, b_q_k1, input_0_k1, a_raw_k1, b_raw_k1 = quantize_with_auto_zero(
        Q_proc, K_landmarks_np, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_k1_t = b_raw_k1.transpose(0,1,3,2).copy()
    c_k1, div_k1, rem_k1, c0_k1 = decompose_targets(a_q_k1, b_q_k1, a_raw_k1, b_raw_k1_t, input_0_k1, c0=0, multiplier=multiplier)
    matmul_targets_k1 = reconstruct_targets_from_c_div_rem(c_k1, div_k1, rem_k1)

    # check and store matmul_targets as uint64 if possible
    flat_vals_k1 = [int(x) for x in np.array(matmul_targets_k1).ravel()]
    flat_max = max(flat_vals_k1)
    flat_min = min(flat_vals_k1)
    if flat_min < 0:
        raise RuntimeError(f"Unexpected negative target values: min={flat_min}")
    if flat_max >= (1 << 64):
        raise RuntimeError(f"matmul_targets too large for u64: max={flat_max}; reduce frac_bits/k_segments/scale")
    
    # kernel_2
    a_q_k2, b_q_k2, input_0_k2, a_raw_k2, b_raw_k2 = quantize_with_auto_zero(
        Q_landmarks_np, K_landmarks_np, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_k2_t = b_raw_k2.transpose(0,1,3,2).copy()
    c_k2, div_k2, rem_k2, c0_k2 = decompose_targets(a_q_k2, b_q_k2, a_raw_k2, b_raw_k2_t, input_0_k2, c0=0, multiplier=multiplier)
    matmul_targets_k2 = reconstruct_targets_from_c_div_rem(c_k2, div_k2, rem_k2)

    # check and store matmul_targets as uint64 if possible
    flat_vals_k2 = [int(x) for x in np.array(matmul_targets_k2).ravel()]
    flat_max = max(flat_vals_k2)
    flat_min = min(flat_vals_k2)
    if flat_min < 0:
        raise RuntimeError(f"Unexpected negative target values: min={flat_min}")
    if flat_max >= (1 << 64):
        raise RuntimeError(f"matmul_targets too large for u64: max={flat_max}; reduce frac_bits/k_segments/scale")

    # kernel_3
    a_q_k3, b_q_k3, input_0_k3, a_raw_k3, b_raw_k3 = quantize_with_auto_zero(
        Q_landmarks_np, K_proc, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_k3_t = b_raw_k3.transpose(0,1,3,2).copy()
    c_k3, div_k3, rem_k3, c0_k3 = decompose_targets(a_q_k3, b_q_k3, a_raw_k3, b_raw_k3_t, input_0_k3, c0=0, multiplier=multiplier)
    matmul_targets_k3 = reconstruct_targets_from_c_div_rem(c_k3, div_k3, rem_k3)

    print("Q_proc : ", Q_proc.shape, "K_landmarks_np : ", K_landmarks_np.shape)
    print("Q_landmarks_np : ", Q_landmarks_np.shape, "K_proc : ", K_proc.shape)

    # check and store matmul_targets as uint64 if possible
    flat_vals_k3 = [int(x) for x in np.array(matmul_targets_k3).ravel()]
    flat_max = max(flat_vals_k3)
    flat_min = min(flat_vals_k3)
    if flat_min < 0:
        raise RuntimeError(f"Unexpected negative target values: min={flat_min}")
    if flat_max >= (1 << 64):
        raise RuntimeError(f"matmul_targets too large for u64: max={flat_max}; reduce frac_bits/k_segments/scale")

    # 테이블도 준비 generate/locate softmax tables
    os.makedirs(tables_outdir, exist_ok=True)
    tables_path = generate_tables_quantized(
        tables_outdir, 
        scale_input=scale_input, 
        multiplier=multiplier, 
        frac_bits=frac_bits, 
        k_segments=k_segments, 
        limb_bits=limb_bits
    )
    tables_npz = np.load(tables_path)

    # validate loaded tables match expected limb_bits/k_segments
    softmax_tables_arr = tables_npz['tables'].astype(np.uint64)
    #softmax_prime_tables = tables_npz['prime_tables'].astype(np.uint64)
    #softmax_z_tables = tables_npz['z_tables'].astype(np.uint64)
    softmax_frac_bits = int(tables_npz.get('frac_bits', frac_bits))
    softmax_scale_input = int(tables_npz.get('scale_input', scale_input))
    softmax_multiplier = int(tables_npz.get('multiplier', multiplier))
    softmax_base_scale_exp = int(tables_npz.get('base_scale_exp', 16))
    softmax_limb_bits = int(tables_npz.get('limb_bits', limb_bits))
    softmax_k_segments = int(tables_npz.get('k_segments', k_segments))
    

    # defensive check: ensure table width is at least 2**limb_bits
    expected_width = 1 << softmax_limb_bits
    actual_width = softmax_tables_arr.shape[1]
    if actual_width < expected_width:
        raise RuntimeError(
            f"Softmax table width mismatch: table width={actual_width}, "
            f"expected at least 2**limb_bits={expected_width} (limb_bits={softmax_limb_bits}). "
            "Regenerate tables with the correct --limb_bits / --k_segments."
        )

    # Convert matmul_targets -> uint64 checks
    matmul_targets_k1_save = np.array(matmul_targets_k1, dtype=np.uint64)
    matmul_targets_k2_save = np.array(matmul_targets_k2, dtype=np.uint64)
    matmul_targets_k3_save = np.array(matmul_targets_k3, dtype=np.uint64)

    # decompose into limbs
    limb_indices_k1 = decompose_to_limbs_array(matmul_targets_k1, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)
    limb_indices_k2 = decompose_to_limbs_array(matmul_targets_k2, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)
    limb_indices_k3 = decompose_to_limbs_array(matmul_targets_k3, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)

    # optional float softmax inspection
    attn_k1_float = compute_float_softmax_from_targets_int_rect(matmul_targets_k1)
    attn_k2_float = compute_float_softmax_from_targets_int(matmul_targets_k2)
    attn_k3_float = compute_float_softmax_from_targets_int_rect(matmul_targets_k3)

    # iterative inverse: try to obtain inv from debug if present, else compute inv from attn_k2_float
    if 'inv_kernel_2' in debug:
        inv_np = debug['inv_kernel_2']  # shape (B,H,L,L)

    # A and P as float arrays (from attn_k2_float and inv_np)
    A = attn_k2_float.astype(np.float64)   # shape (B,H,L,L)
    P = inv_np.astype(np.float64)          # shape (B,H,L,L)

    # 1) AAplus = A @ P  (B,H,L,L)
    a_q_A_AP, b_q_A_AP, input_0_A_AP, a_raw_A_AP, b_raw_A_AP = quantize_with_auto_zero(
        A, P, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_A_AP_t = b_raw_A_AP.transpose(0,1,3,2).copy()
    c_A_AP, div_A_AP, rem_A_AP, c0_A_AP = decompose_targets(a_q_A_AP, b_q_A_AP, a_raw_A_AP, b_raw_A_AP_t, input_0_A_AP, c0=0, multiplier=multiplier)
    matmul_targets_A_AP = reconstruct_targets_from_c_div_rem(c_A_AP, div_A_AP, rem_A_AP)

    # 2) AplusA = P @ A  (B,H,L,L)
    a_q_P_A, b_q_P_A, input_0_P_A, a_raw_P_A, b_raw_P_A = quantize_with_auto_zero(
        P, A, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_P_A_t = b_raw_P_A.transpose(0,1,3,2).copy()
    c_P_A, div_P_A, rem_P_A, c0_P_A = decompose_targets(a_q_P_A, b_q_P_A, a_raw_P_A, b_raw_P_A_t, input_0_P_A, c0=0, multiplier=multiplier)
    matmul_targets_P_A = reconstruct_targets_from_c_div_rem(c_P_A, div_P_A, rem_P_A)

    # For triple products compute intermediate floats then quantize those with partner:
    AAplus_float = np.zeros((B, H, L, L), dtype=np.float64)
    PAm_float = np.zeros((B, H, L, L), dtype=np.float64)  # P @ A
    for n in range(B):
        for h in range(H):
            AAplus_float[n,h,:,:] = A[n,h,:,:].dot(P[n,h,:,:])
            PAm_float[n,h,:,:] = P[n,h,:,:].dot(A[n,h,:,:])

    # 3) AAplusA = (A @ P) @ A  -> quantize (AAplus_float, A)
    a_q_AAplus_A, b_q_AAplus_A, input_0_AAplus_A, a_raw_AAplus_A, b_raw_AAplus_A = quantize_with_auto_zero(
        AAplus_float, A, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_AAplus_A_t = b_raw_AAplus_A.transpose(0,1,3,2).copy()
    c_AAplus_A, div_AAplus_A, rem_AAplus_A, c0_AAplus_A = decompose_targets(a_q_AAplus_A, b_q_AAplus_A, a_raw_AAplus_A, b_raw_AAplus_A_t, input_0_AAplus_A, c0=0, multiplier=multiplier)
    matmul_targets_AAplus_A = reconstruct_targets_from_c_div_rem(c_AAplus_A, div_AAplus_A, rem_AAplus_A)

    # 4) AplusAAplus = (P @ A) @ P  -> quantize (PAm_float, P)
    a_q_PAm_P, b_q_PAm_P, input_0_PAm_P, a_raw_PAm_P, b_raw_PAm_P = quantize_with_auto_zero(
        PAm_float, P, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_PAm_P_t = b_raw_PAm_P.transpose(0,1,3,2).copy()
    c_PAm_P, div_PAm_P, rem_PAm_P, c0_PAm_P = decompose_targets(a_q_PAm_P, b_q_PAm_P, a_raw_PAm_P, b_raw_PAm_P_t, input_0_PAm_P, c0=0, multiplier=multiplier)
    matmul_targets_PAm_P = reconstruct_targets_from_c_div_rem(c_PAm_P, div_PAm_P, rem_PAm_P)

    # Sanity checks for MP matmul targets (non-negative and within u64)
    mp_targets = [
        ("A_AP", matmul_targets_A_AP),
        ("P_A", matmul_targets_P_A),
        ("AAplus_A", matmul_targets_AAplus_A),
        ("PAm_P", matmul_targets_PAm_P),
    ]
    for name, arr in mp_targets:
        flat = [int(x) for x in np.array(arr).ravel()]
        if min(flat) < 0:
            raise RuntimeError(f"MP matmul targets {name} contain negative values.")
        if max(flat) >= (1 << 64):
            raise RuntimeError(f"MP matmul targets {name} exceed u64 range; max={max(flat)}")

    # 정확히 등식을 만들 수는 없으니까 차이를 구하기
    # compute AAplusA_float and AplusAAplus_float from earlier intermediates
    AAplusA_float = np.zeros((B, H, L, L), dtype=np.float64)
    AplusAAplus_float = np.zeros((B, H, L, L), dtype=np.float64)
    for n in range(B):
        for h in range(H):
            AAplusA_float[n,h,:,:] = AAplus_float[n,h,:,:].dot(A[n,h,:,:])
            AplusAAplus_float[n,h,:,:] = PAm_float[n,h,:,:].dot(P[n,h,:,:])

    # compute per (n,h) aggregates
    sum_abs_AAplusA = np.zeros((B, H), dtype=np.float64)
    sum_signed_AAplusA = np.zeros((B, H), dtype=np.float64)
    sum_abs_AplusAAplus = np.zeros((B, H), dtype=np.float64)
    sum_signed_AplusAAplus = np.zeros((B, H), dtype=np.float64)

    for n in range(B):
        for h in range(H):
            diff1 = AAplusA_float[n,h,:,:] - A[n,h,:,:]
            sum_abs_AAplusA[n,h] = float(np.sum(np.abs(diff1)))
            sum_signed_AAplusA[n,h] = float(np.sum(diff1))

            diff2 = AplusAAplus_float[n,h,:,:] - P[n,h,:,:]
            sum_abs_AplusAAplus[n,h] = float(np.sum(np.abs(diff2)))
            sum_signed_AplusAAplus[n,h] = float(np.sum(diff2))

    # quantize aggregates using frac_bits and store as integers
    scale_res = 1 << frac_bits
    # absolute sums are non-negative -> store as uint64 (check overflow)
    sum_abs_AAplusA_q = np.round(sum_abs_AAplusA * scale_res).astype(np.uint64)
    sum_abs_AplusAAplus_q = np.round(sum_abs_AplusAAplus * scale_res).astype(np.uint64)
    # signed sums may be negative -> store as int64
    sum_signed_AAplusA_q = np.round(sum_signed_AAplusA * scale_res).astype(np.int64)
    sum_signed_AplusAAplus_q = np.round(sum_signed_AplusAAplus * scale_res).astype(np.int64)

    # 최종곱 검증
    # temp = kernel_3 @ V  -> (B,H,L,D)
    kernel_3 = debug['kernel_3']
    V = debug['V']

    # sanity check: contraction dimension must match (kernel_3 last dim == V second dim)
    if kernel_3.shape[-1] != V.shape[2]:
        raise RuntimeError(f"Shape mismatch for kernel3@V: kernel_3.shape={kernel_3.shape}, V.shape={V.shape}")
    
    # pass V transposed so quantize_with_auto_zero sees matching head_dim
    # b_float passed is V.transpose(0,1,3,2) => shape (B,H,D,S) so its last dim == S (head_dim)
    a_q_kernel3, b_q_V, input_0_k3V, a_raw_kernel3, b_raw_V = quantize_with_auto_zero(
        kernel_3, V.transpose(0,1,3,2), scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    # b_raw_V returned has shape (B,H,rows_b,head_dim) = (B,H,D,S)
    # we need b_raw_V_t of shape (B,H,head_dim,rows_b) = (B,H,S,D)
    b_raw_V_t = b_raw_V.transpose(0,1,3,2).copy()
    c_k3V, div_k3V, rem_k3V, c0_k3V = decompose_targets(a_q_kernel3, b_q_V, a_raw_kernel3, b_raw_V_t, input_0_k3V, c0=0, multiplier=multiplier)
    matmul_targets_k3V = reconstruct_targets_from_c_div_rem(c_k3V, div_k3V, rem_k3V)

    # mid = kernel_1 @ inv_kernel_2 -> (B,H,S,L)
    kernel_1 = debug['kernel_1']
    inv_kernel_2 = debug['inv_kernel_2']

    # sanity check
    if kernel_1.shape[-1] != inv_kernel_2.shape[2]:
        raise RuntimeError(f"Shape mismatch for kernel1@inv_kernel_2: kernel_1.shape={kernel_1.shape}, inv_kernel_2.shape={inv_kernel_2.shape}")

    a_q_kernel_1, b_q_inv_k2, input_0_k1_inv_k2, a_raw_kernel1, b_raw_inv_k2 = quantize_with_auto_zero(
        kernel_1, inv_kernel_2, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_inv_k2_t = b_raw_inv_k2.transpose(0,1,3,2).copy()
    c_k1_inv_k2, div_k1_inv_k2, rem_k1_inv_k2, c0_k1_inv_k2 = decompose_targets(
        a_q_kernel_1, b_q_inv_k2, a_raw_kernel1, b_raw_inv_k2_t,
        input_0_k1_inv_k2, c0=0, multiplier=multiplier
    )
    matmul_targets_k1_inv_k2 = reconstruct_targets_from_c_div_rem(c_k1_inv_k2, div_k1_inv_k2, rem_k1_inv_k2)

    # X = mid @ temp -> (B,H,S,D)
    mid = debug['mid']    # (B,H,S,L)
    temp = debug['temp']  # (B,H,L,D)

    # sanity check
    if mid.shape[-1] != temp.shape[2]:
        raise RuntimeError(f"Shape mismatch for mid@temp: mid.shape={mid.shape}, temp.shape={temp.shape}")

    # pass temp transposed so quantize_with_auto_zero sees last-dim == head_dim (L)
    a_q_mid, b_q_temp, input_0_mt, a_raw_mid, b_raw_temp = quantize_with_auto_zero(
        mid, temp.transpose(0,1,3,2), scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    # b_raw_temp has shape (B,H,rows_b,head_dim) == (B,H,D,L)
    # decompose_targets expects b_raw_t shape (B,H,head_dim,rows_b) == (B,H,L,D)
    b_raw_temp_t = b_raw_temp.transpose(0,1,3,2).copy()
    c_mt, div_mt, rem_mt, c0_mt = decompose_targets(
        a_q_mid, b_q_temp, a_raw_mid, b_raw_temp_t,
        input_0_mt, c0=0, multiplier=multiplier
    )
    matmul_targets_mt = reconstruct_targets_from_c_div_rem(c_mt, div_mt, rem_mt)
    # fiat-shamir challenge 계산 (unchanged)
    h = hashlib.sha256()
    try:
        h.update(matmul_targets_k1_save.tobytes())
    except Exception:
        h.update(str(matmul_targets_k1_save).encode('utf8'))
    try:
        h.update(matmul_targets_k2_save.tobytes())
    except Exception:
        h.update(str(matmul_targets_k2_save).encode('utf8'))
    try:
        h.update(matmul_targets_k3_save.tobytes())
    except Exception:
        h.update(str(matmul_targets_k3_save).encode('utf8'))

    digest = h.digest()
    r_bytes = digest[:8]
    r_int = int.from_bytes(r_bytes, "big")
    r_int = r_int % (1 << 63)
    r_challenge = np.uint64(r_int)

    # Save integrated npz similar to real_whole_pipeline
    out_dir = os.path.dirname(out_npz_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    save_kwargs = dict(
        # matmul target pieces (kernels) c_k1, div_k1, rem_k1, c0_k1 
        a_k1=a_q_k1.astype(np.uint64), # transpose하지 않은 상태로 들어감
        b_k1=b_q_k1.astype(np.uint64),
        c_k1=c_k1.astype(np.uint64),
        div_k1=div_k1.astype(np.uint64),
        rem_k1=rem_k1.astype(np.uint64),
        input_0_k1=np.uint64(input_0_k1),
        c_0_converted_k1=np.uint64(c0_k1),

        a_k2=a_q_k2.astype(np.uint64),
        b_k2=b_q_k2.astype(np.uint64),
        c_k2=c_k2.astype(np.uint64),
        div_k2=div_k2.astype(np.uint64),
        rem_k2=rem_k2.astype(np.uint64),
        input_0_k2=np.uint64(input_0_k2),
        c_0_converted_k2=np.uint64(c0_k2),

        a_k3=a_q_k3.astype(np.uint64),
        b_k3=b_q_k3.astype(np.uint64),
        c_k3=c_k3.astype(np.uint64),
        div_k3=div_k3.astype(np.uint64),
        rem_k3=rem_k3.astype(np.uint64),
        input_0_k3=np.uint64(input_0_k3),
        c_0_converted_k3=np.uint64(c0_k3),

        # 일단 공용
        scale_input=np.int64(scale_input),
        multiplier=np.uint64(multiplier),
        frac_bits=np.int32(frac_bits),

        # 소프트맥스
        matmul_targets_k1=matmul_targets_k1_save,
        limb_indices_k1=limb_indices_k1.astype(np.uint64),

        matmul_targets_k2=matmul_targets_k2_save,
        limb_indices_k2=limb_indices_k2.astype(np.uint64),

        matmul_targets_k3=matmul_targets_k3_save,
        limb_indices_k3=limb_indices_k3.astype(np.uint64),

        # softmax tables
        softmax_tables=softmax_tables_arr,
        #softmax_prime_tables=softmax_prime_tables,
        #softmax_z_tables=softmax_z_tables,
        softmax_tables_frac_bits=np.int32(softmax_frac_bits),
        softmax_scale_input=np.int64(softmax_scale_input),
        softmax_multiplier=np.int64(softmax_multiplier),
        softmax_base_scale_exp=np.int32(softmax_base_scale_exp),
        softmax_limb_bits=np.int32(softmax_limb_bits),
        softmax_k_segments=np.int32(softmax_k_segments),
        softmax_tables_path=np.array([tables_path]),

        # float softmax inspection
        attn_k1_float=attn_k1_float.astype(np.float64),
        attn_k2_float=attn_k2_float.astype(np.float64),
        attn_k3_float=attn_k3_float.astype(np.float64),

        # 무어 -펜로즈 관련 행렬 곱셈
        # AAplus = A @ P
        a_A_AP=a_q_A_AP.astype(np.uint64),
        b_A_AP=b_q_A_AP.astype(np.uint64),
        c_A_AP=c_A_AP.astype(np.uint64),
        div_A_AP=div_A_AP.astype(np.uint64),
        rem_A_AP=rem_A_AP.astype(np.uint64),
        input_0_A_AP=np.uint64(input_0_A_AP),
        c_0_converted_A_AP=np.uint64(c0_A_AP),
        matmul_targets_A_AP=np.array(matmul_targets_A_AP, dtype=np.uint64),

        # P @ A
        a_P_A=a_q_P_A.astype(np.uint64),
        b_P_A=b_q_P_A.astype(np.uint64),
        c_P_A=c_P_A.astype(np.uint64),
        div_P_A=div_P_A.astype(np.uint64),
        rem_P_A=rem_P_A.astype(np.uint64),
        input_0_P_A=np.uint64(input_0_P_A),
        c_0_converted_P_A=np.uint64(c0_P_A),
        matmul_targets_P_A=np.array(matmul_targets_P_A, dtype=np.uint64),

        # (A @ P) @ A
        a_AAplus_A=a_q_AAplus_A.astype(np.uint64),
        b_AAplus_A=b_q_AAplus_A.astype(np.uint64),
        c_AAplus_A=c_AAplus_A.astype(np.uint64),
        div_AAplus_A=div_AAplus_A.astype(np.uint64),
        rem_AAplus_A=rem_AAplus_A.astype(np.uint64),
        input_0_AAplus_A=np.uint64(input_0_AAplus_A),
        c_0_converted_AAplus_A=np.uint64(c0_AAplus_A),
        matmul_targets_AAplus_A=np.array(matmul_targets_AAplus_A, dtype=np.uint64),

        # (P @ A) @ P
        a_PAm_P=a_q_PAm_P.astype(np.uint64),
        b_PAm_P=b_q_PAm_P.astype(np.uint64),
        c_PAm_P=c_PAm_P.astype(np.uint64),
        div_PAm_P=div_PAm_P.astype(np.uint64),
        rem_PAm_P=rem_PAm_P.astype(np.uint64),
        input_0_PAm_P=np.uint64(input_0_PAm_P),
        c_0_converted_PAm_P=np.uint64(c0_PAm_P),
        matmul_targets_PAm_P=np.array(matmul_targets_PAm_P, dtype=np.uint64),

        # sum of absolute differences between AAplusA and A, per (batch,head)
        mp_res_AAplusA_sum_abs_q=np.array(sum_abs_AAplusA_q, dtype=np.uint64),
        # signed sum of differences between AAplusA and A, per (batch,head)
        mp_res_AAplusA_sum_signed_q=np.array(sum_signed_AAplusA_q, dtype=np.int64),

        # sum of absolute differences between AplusAAplus and P, per (batch,head)
        mp_res_AplusAAplus_sum_abs_q=np.array(sum_abs_AplusAAplus_q, dtype=np.uint64),
        # signed sum of differences between AplusAAplus and P, per (batch,head)
        mp_res_AplusAAplus_sum_signed_q=np.array(sum_signed_AplusAAplus_q, dtype=np.int64),

        # 최종곱
        a_k3V=a_q_kernel3.astype(np.uint64),
        b_k3V=b_q_V.astype(np.uint64),
        c_k3V=c_k3V.astype(np.uint64),
        div_k3V=div_k3V.astype(np.uint64),
        rem_k3V=rem_k3V.astype(np.uint64),
        input_0_k3V=np.uint64(input_0_k3V),
        c_0_converted_k3V=np.uint64(c0_k3V),
        matmul_targets_k3V=np.array(matmul_targets_k3V, dtype=np.uint64),

        a_k1_inv_k2=a_q_kernel_1.astype(np.uint64),
        b_k1_inv_k2=b_q_inv_k2.astype(np.uint64),
        c_k1_inv_k2=c_k1_inv_k2.astype(np.uint64),
        div_k1_inv_k2=div_k1_inv_k2.astype(np.uint64),
        rem_k1_inv_k2=rem_k1_inv_k2.astype(np.uint64),
        input_0_k1_inv_k2=np.uint64(input_0_k1_inv_k2),
        c_0_converted_k1_inv_k2=np.uint64(c0_k1_inv_k2),
        matmul_targets_k1_inv_k2=np.array(matmul_targets_k1_inv_k2, dtype=np.uint64),

        a_mt=a_q_mid.astype(np.uint64),
        b_mt=b_q_temp.astype(np.uint64),
        c_mt=c_mt.astype(np.uint64),
        div_mt=div_mt.astype(np.uint64),
        rem_mt=rem_mt.astype(np.uint64),
        input_0_mt=np.uint64(input_0_mt),
        c_0_converted_mt=np.uint64(c0_mt),
        matmul_targets_mt=np.array(matmul_targets_mt, dtype=np.uint64),
        
        # keep compatibility: top-level r and r_challenge
        r=np.uint64(r_challenge),
        r_challenge=np.uint64(r_challenge),

        # meta
        v_scale=np.int64(v_scale),
        frac_bits_softmax=np.int32(frac_bits),
        r_challenge_meta=np.uint64(r_challenge),
        meta=np.array([{"batch": B, "num_head": H, "seq_len": S, "head_dim": D, "num_landmarks": L}])
    )

    np.savez_compressed(out_npz_path, **save_kwargs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("debug_npz", help="input debug npz saved by attention collect")
    ap.add_argument("out_npz", help="output witness npz")
    ap.add_argument("--scale_input", type=int, default=(1<<8))
    ap.add_argument("--frac_bits", type=int, default=16)
    ap.add_argument("--k_segments", type=int, default=2)
    ap.add_argument("--limb_bits", type=int, default=16)
    ap.add_argument("--v_scale", type=int, default=1)
    ap.add_argument("--inv_scale", type=int, default=1)
    ap.add_argument("--tables_outdir", type=str, default="tmp_tables")
    ap.add_argument("--multiplier", type=int, default=1)
    ap.add_argument("--final-auto-scale", action="store_true", help="auto-scale final X_q decomposition to fit uint64")
    args = ap.parse_args()

    process_attn_debug_npz(
        args.debug_npz,
        args.out_npz,
        scale_input = args.scale_input,
        frac_bits = args.frac_bits,
        k_segments = args.k_segments,
        limb_bits = args.limb_bits,
        v_scale = args.v_scale,
        inv_scale = args.inv_scale,
        tables_outdir = args.tables_outdir,
        multiplier = args.multiplier,
        final_auto_scale = args.final_auto_scale,
    )