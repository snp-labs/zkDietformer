import os
import math
import numpy as np
import torch

from attention_nystrom_quantize import quantize_with_auto_zero, decompose_targets
from softmax_table import generate_tables_quantized

def decompose_to_limbs_array(targets_obj, limb_bits, k_segments):
    batch, num_head, rows, cols = targets_obj.shape
    limb_mask = (1 << limb_bits) - 1
    limbs = np.zeros((batch, num_head, rows, cols, k_segments), dtype=np.uint64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(rows):
                for j in range(cols):
                    t = int(targets_obj[n,h,i,j])
                    if t < 0:
                        raise ValueError(f"Negative matmul target encountered at ({n},{h},{i},{j}): {t}")
                    unsigned = int(t)
                    for k in range(k_segments):
                        limb = (unsigned >> (k * limb_bits)) & limb_mask
                        limbs[n,h,i,j,k] = np.uint64(limb)
    return limbs

def quantize_matrix_fixed_point(arr_float, scale):
    return np.round(arr_float * float(scale)).astype(np.int64)

def compute_float_softmax_from_targets_int_rect(targets_obj):
    batch, num_head, rows, cols = targets_obj.shape
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

def compute_float_softmax_from_targets_int(targets_obj):
    batch, num_head, rows, cols = targets_obj.shape
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

def _reconstruct_targets_from_c_div_rem(c, div, rem):
    # helper: reconstruct integer target from c/div/rem (supports non-square shapes)
    batch, num_head, rows, cols = c.shape
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
    # parameters for quantization and tables (defaults chosen to mirror real_whole_pipeline.py)
    scale_input = 1<<8,
    frac_bits = 16,
    k_segments = 2,
    limb_bits = 2,
    v_scale = 1,
    inv_scale = 1,
    tables_outdir = "tmp_tables",
    multiplier = 1,
    r = 3,
):
    """
    Load an attention debug npz produced by the attention module (collect_intermediates),
    perform quantization + softmax-table based exp_prod computations, quantize iterative inverse,
    compute integer numerators left/right and final X_q, and compute Moore-Penrose residuals.
    Save an integrated npz to out_npz_path.

    debug_npz_path: path to npz produced by attention (must contain Q_proc, K_proc, V, mask,
                    and optionally kernel_2 / inv_kernel_2)
    out_npz_path: output integrated npz path
    """
    debug = np.load(debug_npz_path, allow_pickle=True)
    # Expect these keys (from modified attention forward): Q_proc, K_proc, V, mask
    if 'Q_proc' not in debug or 'K_proc' not in debug or 'V' not in debug or 'mask' not in debug:
        raise RuntimeError("debug npz missing required tensors (Q_proc,K_proc,V,mask)")

    Q_proc = debug['Q_proc']  # shape (B,H,S,D)
    K_proc = debug['K_proc']
    V = debug['V']
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

    # compute landmarks if not present
    if 'Q_landmarks' in debug:
        Q_landmarks_np = debug['Q_landmarks']
        K_landmarks_np = debug['K_landmarks'] if 'K_landmarks' in debug else None
    else:
        seg = S // L
        Q_landmarks_np = Q_proc.reshape(B, H, L, seg, D).mean(axis=3)
        K_landmarks_np = K_proc.reshape(B, H, L, seg, D).mean(axis=3)

    # Quantize/decompose each matmul needed using attention_nystrom_quantize helpers
    # kernel_1
    a_q_k1, b_q_k1, input_0_k1, a_raw_k1, b_raw_k1 = quantize_with_auto_zero(
        Q_proc, K_landmarks_np, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_k1_t = b_raw_k1.transpose(0,1,3,2).copy()
    c_k1, div_k1, rem_k1, _ = decompose_targets(a_q_k1, b_q_k1, a_raw_k1, b_raw_k1_t, input_0_k1, c0=0, multiplier=multiplier)
    matmul_targets_k1 = _reconstruct_targets_from_c_div_rem(c_k1, div_k1, rem_k1)

    # kernel_2
    a_q_k2, b_q_k2, input_0_k2, a_raw_k2, b_raw_k2 = quantize_with_auto_zero(
        Q_landmarks_np, K_landmarks_np, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_k2_t = b_raw_k2.transpose(0,1,3,2).copy()
    c_k2, div_k2, rem_k2, _ = decompose_targets(a_q_k2, b_q_k2, a_raw_k2, b_raw_k2_t, input_0_k2, c0=0, multiplier=multiplier)
    matmul_targets_k2 = _reconstruct_targets_from_c_div_rem(c_k2, div_k2, rem_k2)

    # kernel_3
    a_q_k3, b_q_k3, input_0_k3, a_raw_k3, b_raw_k3 = quantize_with_auto_zero(
        Q_landmarks_np, K_proc, scale_input=scale_input, input_0_default=0, check_uint64=False
    )
    b_raw_k3_t = b_raw_k3.transpose(0,1,3,2).copy()
    c_k3, div_k3, rem_k3, _ = decompose_targets(a_q_k3, b_q_k3, a_raw_k3, b_raw_k3_t, input_0_k3, c0=0, multiplier=multiplier)
    matmul_targets_k3 = _reconstruct_targets_from_c_div_rem(c_k3, div_k3, rem_k3)

    # generate/locate softmax tables
    os.makedirs(tables_outdir, exist_ok=True)
    tables_path = generate_tables_quantized(
        tables_outdir, scale_input=scale_input, multiplier=multiplier, frac_bits=frac_bits, k_segments=k_segments, limb_bits=limb_bits
    )
    tables_npz = np.load(tables_path)
    if 'tables' not in tables_npz:
        raise RuntimeError(f"Generated softmax tables file {tables_path} does not contain 'tables' array.")
    softmax_tables_arr = tables_npz['tables'].astype(np.uint64)
    softmax_frac_bits = int(tables_npz.get('frac_bits', frac_bits))
    softmax_scale_input = int(tables_npz.get('scale_input', scale_input))
    softmax_multiplier = int(tables_npz.get('multiplier', multiplier))
    softmax_base_scale_exp = int(tables_npz.get('base_scale_exp', 16))
    softmax_limb_bits = int(tables_npz.get('limb_bits', limb_bits))
    softmax_k_segments = int(tables_npz.get('k_segments', k_segments))

    # Convert matmul_targets -> uint64 checks
    # kernel1
    flat_vals_k1 = [int(x) for x in np.array(matmul_targets_k1).ravel()]
    max_k1 = max(flat_vals_k1); min_k1 = min(flat_vals_k1)
    if min_k1 < 0:
        raise RuntimeError("Negative targets in kernel1")
    if max_k1 >= (1 << 64):
        raise RuntimeError("kernel1 targets too large for u64")
    matmul_targets_k1_save = np.array(matmul_targets_k1, dtype=np.uint64)

    # kernel2
    flat_vals_k2 = [int(x) for x in np.array(matmul_targets_k2).ravel()]
    max_k2 = max(flat_vals_k2); min_k2 = min(flat_vals_k2)
    if min_k2 < 0:
        raise RuntimeError("Negative targets in kernel2")
    if max_k2 >= (1 << 64):
        raise RuntimeError("kernel2 targets too large for u64")
    matmul_targets_k2_save = np.array(matmul_targets_k2, dtype=np.uint64)

    # kernel3
    flat_vals_k3 = [int(x) for x in np.array(matmul_targets_k3).ravel()]
    max_k3 = max(flat_vals_k3); min_k3 = min(flat_vals_k3)
    if min_k3 < 0:
        raise RuntimeError("Negative targets in kernel3")
    if max_k3 >= (1 << 64):
        raise RuntimeError("kernel3 targets too large for u64")
    matmul_targets_k3_save = np.array(matmul_targets_k3, dtype=np.uint64)

    # decompose into limbs
    limb_indices_k1 = decompose_to_limbs_array(matmul_targets_k1, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)
    limb_indices_k2 = decompose_to_limbs_array(matmul_targets_k2, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)
    limb_indices_k3 = decompose_to_limbs_array(matmul_targets_k3, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)

    # compute exp_prod arrays using tables, applying mask for kernel3 (mask zeros -> exp_prod = 0)
    Kt = softmax_tables_arr.shape[0]
    W = softmax_tables_arr.shape[1]

    exp_prod_k1 = np.zeros((B, H, S, L), dtype=object)
    for n in range(B):
        for h in range(H):
            for i in range(S):
                for j in range(L):
                    prod = 1
                    for k in range(Kt):
                        idx = int(limb_indices_k1[n,h,i,j,k])
                        if idx < 0 or idx >= W:
                            raise RuntimeError(f"limb index out of bounds for k1: k={k} idx={idx} width={W}")
                        val = int(softmax_tables_arr[k, idx])
                        prod = prod * val
                    exp_prod_k1[n,h,i,j] = prod

    exp_prod_k2 = np.zeros((B, H, L, L), dtype=object)
    for n in range(B):
        for h in range(H):
            for i in range(L):
                for j in range(L):
                    prod = 1
                    for k in range(Kt):
                        idx = int(limb_indices_k2[n,h,i,j,k])
                        if idx < 0 or idx >= W:
                            raise RuntimeError(f"limb index out of bounds for k2: k={k} idx={idx} width={W}")
                        val = int(softmax_tables_arr[k, idx])
                        prod = prod * val
                    exp_prod_k2[n,h,i,j] = prod

    exp_prod_k3 = np.zeros((B, H, L, S), dtype=object)
    for n in range(B):
        for h in range(H):
            for i in range(L):
                for j in range(S):
                    # mask check
                    if int(mask_np[n, j]) == 0:
                        exp_prod_k3[n,h,i,j] = 0
                        continue
                    prod = 1
                    for k in range(Kt):
                        idx = int(limb_indices_k3[n,h,i,j,k])
                        if idx < 0 or idx >= W:
                            raise RuntimeError(f"limb index out of bounds for k3: k={k} idx={idx} width={W}")
                        val = int(softmax_tables_arr[k, idx])
                        prod = prod * val
                    exp_prod_k3[n,h,i,j] = prod

    # optional float softmax inspection
    attn_k1_float = compute_float_softmax_from_targets_int_rect(matmul_targets_k1)
    attn_k2_float = compute_float_softmax_from_targets_int(matmul_targets_k2)
    attn_k3_float = compute_float_softmax_from_targets_int_rect(matmul_targets_k3)

    # quantize V
    V_q = quantize_matrix_fixed_point(V, v_scale).astype(np.int64)

    # iterative inverse: try to obtain inv from debug if present, else compute inv from attn_k2_float
    if 'inv_kernel_2' in debug:
        inv_np = np.array(debug['inv_kernel_2']).astype(np.float64)
    else:
        # compute per-batch/head pseudoinverse (pinv) as fallback
        inv_np = np.zeros_like(attn_k2_float)
        for n in range(B):
            for h in range(H):
                inv_np[n,h,:,:] = np.linalg.pinv(attn_k2_float[n,h,:,:])

    # quantize inv_np
    inv_q = quantize_matrix_fixed_point(inv_np, inv_scale).astype(np.int64)

    # compute right_num = kernel3_exp_prod @ V_q  -> shape (B,H,L,D)
    right_num = np.zeros((B, H, L, D), dtype=object)
    max_abs_right = 0
    for n in range(B):
        for h in range(H):
            for i in range(L):
                for t in range(D):
                    acc = 0
                    for j in range(S):
                        ep = int(exp_prod_k3[n,h,i,j])
                        vq = int(V_q[n,h,j,t])
                        acc += ep * vq
                    right_num[n,h,i,t] = int(acc)
                    if abs(acc) > max_abs_right:
                        max_abs_right = abs(acc)

    # compute left_num = kernel1_exp_prod @ inv_q -> shape (B,H,S,L)
    left_num = np.zeros((B, H, S, L), dtype=object)
    max_abs_left = 0
    for n in range(B):
        for h in range(H):
            for i in range(S):
                for k in range(L):
                    acc = 0
                    for j in range(L):
                        ep = int(exp_prod_k1[n,h,i,j])
                        invq = int(inv_q[n,h,j,k])
                        acc += ep * invq
                    left_num[n,h,i,k] = int(acc)
                    if abs(acc) > max_abs_left:
                        max_abs_left = abs(acc)

    # final X_num = left_num @ right_num -> (B,H,S,D)
    X_num = np.zeros((B, H, S, D), dtype=object)
    max_abs_final = 0
    for n in range(B):
        for h in range(H):
            for i in range(S):
                for t in range(D):
                    acc = 0
                    for m in range(L):
                        ln = int(left_num[n,h,i,m])
                        rn = int(right_num[n,h,m,t])
                        acc += ln * rn
                    X_num[n,h,i,t] = int(acc)
                    if abs(acc) > max_abs_final:
                        max_abs_final = abs(acc)

    MAX_I64 = (1 << 63) - 1
    if max_abs_final > MAX_I64:
        scale_needed = math.ceil(max_abs_final / float(MAX_I64))
        # scale down V_q and inv_q, recompute
        V_q = (V_q // scale_needed).astype(np.int64)
        inv_q = (inv_q // scale_needed).astype(np.int64)
        # recompute right_num, left_num, X_num
        right_num = np.zeros((B, H, L, D), dtype=object)
        for n in range(B):
            for h in range(H):
                for i in range(L):
                    for t in range(D):
                        acc = 0
                        for j in range(S):
                            ep = int(exp_prod_k3[n,h,i,j])
                            vq = int(V_q[n,h,j,t])
                            acc += ep * vq
                        right_num[n,h,i,t] = int(acc)
        left_num = np.zeros((B, H, S, L), dtype=object)
        for n in range(B):
            for h in range(H):
                for i in range(S):
                    for k in range(L):
                        acc = 0
                        for j in range(L):
                            ep = int(exp_prod_k1[n,h,i,j])
                            invq = int(inv_q[n,h,j,k])
                            acc += ep * invq
                        left_num[n,h,i,k] = int(acc)
        X_num = np.zeros((B, H, S, D), dtype=object)
        max_abs_final = 0
        for n in range(B):
            for h in range(H):
                for i in range(S):
                    for t in range(D):
                        acc = 0
                        for m in range(L):
                            ln = int(left_num[n,h,i,m])
                            rn = int(right_num[n,h,m,t])
                            acc += ln * rn
                        X_num[n,h,i,t] = int(acc)
                        if abs(acc) > max_abs_final:
                            max_abs_final = abs(acc)
        if max_abs_final > MAX_I64:
            raise RuntimeError(f"Even after scaling V_q and inv_q by {scale_needed}, X_num max_abs={max_abs_final} still exceeds i64 range. Reduce frac_bits/k_segments or use smaller v_scale/inv_scale.")

    # convert X_q to int64 ndarray
    X_q = np.zeros((B, H, S, D), dtype=np.int64)
    for n in range(B):
        for h in range(H):
            for i in range(S):
                for t in range(D):
                    X_q[n,h,i,t] = int(X_num[n,h,i,t])

    # Moore-Penrose checks (float)
    mp_residuals = {}
    AAplusA_diff_max = 0.0
    AplusAAplus_diff_max = 0.0
    sym1_diff_max = 0.0
    sym2_diff_max = 0.0

    for n in range(B):
        for h in range(H):
            A = attn_k2_float[n,h,:,:]        # (L,L)
            P = inv_np[n,h,:,:]              # (L,L)
            AAplusA = A.dot(P).dot(A)
            diff1 = np.max(np.abs(AAplusA - A))
            if diff1 > AAplusA_diff_max: AAplusA_diff_max = diff1

            AplusAAplus = P.dot(A).dot(P)
            diff2 = np.max(np.abs(AplusAAplus - P))
            if diff2 > AplusAAplus_diff_max: AplusAAplus_diff_max = diff2

            AAplus = A.dot(P)
            sym1 = np.max(np.abs(AAplus.T - AAplus))
            if sym1 > sym1_diff_max: sym1_diff_max = sym1

            AplusA = P.dot(A)
            sym2 = np.max(np.abs(AplusA.T - AplusA))
            if sym2 > sym2_diff_max: sym2_diff_max = sym2

    mp_residuals['AAplusA_diff_max'] = AAplusA_diff_max
    mp_residuals['AplusAAplus_diff_max'] = AplusAAplus_diff_max
    mp_residuals['sym_AAplus_diff_max'] = sym1_diff_max
    mp_residuals['sym_AplusA_diff_max'] = sym2_diff_max

    # Integer approx MP check (quantized A_q = round(attn_k2_float * inv_scale))
    A_q = quantize_matrix_fixed_point(attn_k2_float, inv_scale)  # int64
    AAplusA_q = np.zeros((B, H, L, L), dtype=object)
    max_abs_int_residual = 0
    for n in range(B):
        for h in range(H):
            for i in range(L):
                for j in range(L):
                    acc = 0
                    for p in range(L):
                        for q in range(L):
                            acc += int(A_q[n,h,i,p]) * int(inv_q[n,h,p,q]) * int(A_q[n,h,q,j])
                    AAplusA_q[n,h,i,j] = int(acc)
            for i in range(L):
                for j in range(L):
                    target = int(A_q[n,h,i,j]) * (inv_scale**2)
                    residual = abs(int(AAplusA_q[n,h,i,j]) - target)
                    if residual > max_abs_int_residual:
                        max_abs_int_residual = residual

    # Save integrated npz similar to real_whole_pipeline
    out_dir = os.path.dirname(out_npz_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    save_kwargs = dict(
        # matmul target pieces
        matmul_targets_k1 = matmul_targets_k1_save,
        limb_indices_k1 = limb_indices_k1.astype(np.uint64),
        c_k1 = c_k1.astype(np.uint64),
        div_k1 = div_k1.astype(np.uint64),
        rem_k1 = rem_k1.astype(np.uint64),

        matmul_targets_k2 = matmul_targets_k2_save,
        limb_indices_k2 = limb_indices_k2.astype(np.uint64),
        c_k2 = c_k2.astype(np.uint64),
        div_k2 = div_k2.astype(np.uint64),
        rem_k2 = rem_k2.astype(np.uint64),

        matmul_targets_k3 = matmul_targets_k3_save,
        limb_indices_k3 = limb_indices_k3.astype(np.uint64),
        c_k3 = c_k3.astype(np.uint64),
        div_k3 = div_k3.astype(np.uint64),
        rem_k3 = rem_k3.astype(np.uint64),

        # softmax tables
        softmax_tables = softmax_tables_arr,
        softmax_tables_frac_bits = np.int32(softmax_frac_bits),
        softmax_scale_input = np.int64(softmax_scale_input),
        softmax_multiplier = np.int64(softmax_multiplier),
        softmax_base_scale_exp = np.int32(softmax_base_scale_exp),
        softmax_limb_bits = np.int32(softmax_limb_bits),
        softmax_k_segments = np.int32(softmax_k_segments),
        softmax_tables_path = np.array([tables_path]),

        # float softmax inspection
        attn_k1_float = attn_k1_float.astype(np.float64),
        attn_k2_float = attn_k2_float.astype(np.float64),
        attn_k3_float = attn_k3_float.astype(np.float64),

        # iterative inverse
        inv_float = inv_np.astype(np.float64),
        inv_q = inv_q.astype(np.int64),
        inv_scale = np.int64(inv_scale),

        # integer intermediates
        exp_prod_k1 = np.array(exp_prod_k1, dtype=object),
        exp_prod_k2 = np.array(exp_prod_k2, dtype=object),
        exp_prod_k3 = np.array(exp_prod_k3, dtype=object),

        V_q = V_q.astype(np.int64),
        right_num = np.array(right_num, dtype=object),
        left_num = np.array(left_num, dtype=object),
        X_q = X_q.astype(np.int64),

        # MP residuals
        mp_residuals = np.array([mp_residuals]),
        mp_int_max_abs_residual = np.array([max_abs_int_residual]),

        # meta
        v_scale = np.int64(v_scale),
        frac_bits = np.int32(frac_bits),
        r = np.uint64(r),
        meta = np.array([{"batch": B, "num_head": H, "seq_len": S, "head_dim": D, "num_landmarks": L}])
    )

    np.savez_compressed(out_npz_path, **save_kwargs)
    print("Wrote attention witness npz to", out_npz_path)
    print("MP float residuals:", mp_residuals)
    print("MP integer approx max abs residual:", max_abs_int_residual)