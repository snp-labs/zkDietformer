import argparse
import numpy as np
import os
import math
import torch
from attention_nystrom_quantize import quantize_with_auto_zero, decompose_targets, random_inputs_torch
from softmax_table import generate_tables_quantized
from attention_nystrom import NystromAttention

# helper: reconstruct integer target from c/div/rem
def reconstruct_targets_from_c_div_rem(c, div, rem):
    batch, num_head, seq_len, _ = c.shape
    targets = np.zeros((batch, num_head, seq_len, seq_len), dtype=object)
    for n in range(batch):
        for h in range(num_head):
            for i in range(seq_len):
                for j in range(seq_len):
                    c_val = int(c[n,h,i,j])
                    div_val = int(div[n,h,i,j])
                    rem_val = int(rem[n,h,i,j])
                    div22 = (div_val << 8) + c_val
                    target = (div22 << 22) + rem_val
                    targets[n,h,i,j] = int(target)
    return targets

def decompose_to_limbs_array(targets_obj, limb_bits, k_segments):
    batch, num_head, seq_len, _ = targets_obj.shape
    limb_mask = (1 << limb_bits) - 1
    limbs = np.zeros((batch, num_head, seq_len, seq_len, k_segments), dtype=np.uint64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(seq_len):
                for j in range(seq_len):
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

def compute_float_softmax_from_targets_int(targets_obj):
    batch, num_head, seq_len, _ = targets_obj.shape
    attn = np.zeros((batch, num_head, seq_len, seq_len), dtype=np.float64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(seq_len):
                row = np.array([int(targets_obj[n,h,i,j]) for j in range(seq_len)], dtype=np.float64)
                row_shift = row - np.max(row)
                exps = np.exp(row_shift)
                denom = np.sum(exps)
                if denom == 0:
                    s = np.ones_like(exps) / len(exps)
                else:
                    s = exps / denom
                attn[n,h,i,:] = s
    return attn

# (추가)
def compute_float_softmax_from_targets_int_rect(targets_obj):
    # For non-square softmaxs (e.g., seq_len x num_landmarks), compute softmax along last dim
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # attention args
    parser.add_argument("out_npz", type=str)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=1<<8)
    parser.add_argument("--input_0", type=int, default=0)
    parser.add_argument("--r", type=int, default=3)
    # IMPORTANT: choose frac_bits small enough for K small to avoid huge exp_prod integers
    parser.add_argument("--frac_bits", type=int, default=16, help="fractional bits used for exp fixed-point in lookup tables")
    parser.add_argument("--device", type=str, default="cpu")
    # table / second-matmul args
    parser.add_argument("--v_scale", type=int, default=1, help="scale to quantize V and X for second matmul verification (keep small for integer safety)")
    parser.add_argument("--inv_scale", type=int, default=1, help="scale to quantize iterative inverse matrix for integer matmuls")
    parser.add_argument("--tables_outdir", type=str, default="tmp_tables")
    parser.add_argument("--k_segments", type=int, default=2, help="K segments used in softmax tables")
    parser.add_argument("--limb_bits", type=int, default=2, help="bits per limb used in softmax tables")
    # (추가)
    parser.add_argument("--num_landmarks", type=int, default=None, help="number of landmarks for Nystrom; default=seq_len")
    args = parser.parse_args()
    # (추가)
    if args.num_landmarks is None:
        args.num_landmarks = args.seq_len

    # initialize Nystrom config (kept for compatibility)
    config = {"head_dim": args.head_dim, "num_head": args.num_head, "num_landmarks": args.num_landmarks, "seq_len": args.seq_len}
    attn_module = NystromAttention(config)

    # 1) create random Q,K
    torch.manual_seed(args.seed)
    Q_t, K_t, mask = random_inputs_torch(args.batch, args.num_head, args.seq_len, args.head_dim, seed=args.seed, device=args.device)
    scale_div = math.sqrt(math.sqrt(args.head_dim))
    mask_expand = mask.view(args.batch, 1, args.seq_len, 1)
    Q_proc = (Q_t * mask_expand) / scale_div
    K_proc = (K_t * mask_expand) / scale_div

    Q_np = Q_proc.cpu().numpy()
    K_np = K_proc.cpu().numpy()

    # 2) quantize Q,K and decompose targets (first matmul) -- keep for the full-attention case
    a_q_full, b_q_full, input_0_fin, a_raw_full, b_raw_full = quantize_with_auto_zero(Q_np, K_np, scale_input=args.scale, input_0_default=args.input_0, check_uint64=False)
    b_raw_full_t = b_raw_full.transpose(0,1,3,2).copy()

    # For both full-attention and Nystrom branches we generate the softmax tables once
    os.makedirs(args.tables_outdir, exist_ok=True)
    tables_path = generate_tables_quantized(
        args.tables_outdir,
        scale_input=args.scale,
        multiplier=1,
        frac_bits=args.frac_bits,
        k_segments=args.k_segments,
        limb_bits=args.limb_bits,
    )

    tables_npz = np.load(tables_path)
    if 'tables' not in tables_npz:
        raise RuntimeError(f"Generated softmax tables file {tables_path} does not contain 'tables' array.")
    softmax_tables_arr = tables_npz['tables'].astype(np.uint64)
    softmax_frac_bits = int(tables_npz['frac_bits']) if 'frac_bits' in tables_npz else int(args.frac_bits)
    softmax_scale_input = int(tables_npz['scale_input']) if 'scale_input' in tables_npz else int(args.scale)
    softmax_multiplier = int(tables_npz['multiplier']) if 'multiplier' in tables_npz else 1
    softmax_base_scale_exp = int(tables_npz['base_scale_exp']) if 'base_scale_exp' in tables_npz else 16
    softmax_limb_bits = int(tables_npz['limb_bits']) if 'limb_bits' in tables_npz else int(args.limb_bits)
    softmax_k_segments = int(tables_npz['k_segments']) if 'k_segments' in tables_npz else int(args.k_segments)

    # If using full-attention (num_landmarks == seq_len), keep original pipeline behavior:
    if args.num_landmarks == args.seq_len:
        # Existing flow for full softmax attention
        max_val = max(int(a_q_full.max()), int(b_q_full.max()))
        if max_val >= (1 << 63):
            print("Warning: quantized values are large; consider lowering scale to avoid overflow when casting to u64.")

        c, div, rem, c0_used = decompose_targets(a_q_full, b_q_full, a_raw_full, b_raw_full_t, input_0_fin, c0=0, multiplier=1)

        # 4) reconstruct integer targets to be used as softmax inputs
        matmul_targets_obj = reconstruct_targets_from_c_div_rem(c, div, rem)  # dtype object

        # check and store matmul_targets as uint64 if possible
        flat_vals = [int(x) for x in np.array(matmul_targets_obj).ravel()]
        flat_max = max(flat_vals)
        flat_min = min(flat_vals)
        if flat_min < 0:
            raise RuntimeError(f"Unexpected negative target values: min={flat_min}")
        if flat_max >= (1 << 64):
            raise RuntimeError(f"matmul_targets too large for u64: max={flat_max}; reduce frac_bits/k_segments/scale")

        matmul_targets_to_save = np.array(matmul_targets_obj, dtype=np.uint64)

        # 5) decompose each target into limbs (indices) for lookup
        limb_indices = decompose_to_limbs_array(matmul_targets_obj, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)

        # 6) compute per-element exp_prod integers using embedded tables
        Kt = softmax_tables_arr.shape[0]
        W = softmax_tables_arr.shape[1]
        batch, num_head, seq_len, _ = matmul_targets_to_save.shape

        exp_prod_int = np.zeros((batch, num_head, seq_len, seq_len), dtype=object)
        for n in range(batch):
            for h in range(num_head):
                for i in range(seq_len):
                    for j in range(seq_len):
                        prod = 1
                        for k in range(Kt):
                            idx = int(limb_indices[n,h,i,j,k])
                            if idx < 0 or idx >= W:
                                raise RuntimeError(f"limb index out of bounds: k={k} idx={idx} width={W}")
                            val = int(softmax_tables_arr[k, idx])
                            prod = prod * val
                        exp_prod_int[n,h,i,j] = prod

        # Optional: compute floating softmax for inspection (not used for X_q now)
        attn_float = compute_float_softmax_from_targets_int(matmul_targets_obj)

        # 7) generate random V and quantize to integers V_q
        rng = np.random.RandomState(args.seed + 1)
        V = rng.randn(args.batch, args.num_head, args.seq_len, args.head_dim).astype(np.float64)

        # Quantize V to integer matrix V_q (i64)
        V_q = quantize_matrix_fixed_point(V, args.v_scale).astype(np.int64)

        # 8) compute X_num (integer numerator) for second matmul: X_num[i,t] = sum_j exp_prod_int[i,j] * V_q[j,t]
        # Produces X_num shape (batch,num_head,seq_len,head_dim)
        X_num = np.zeros((args.batch, args.num_head, args.seq_len, args.head_dim), dtype=object)
        max_abs = 0
        for n in range(args.batch):
            for h in range(args.num_head):
                for i in range(args.seq_len):
                    for t in range(args.head_dim):
                        acc = 0
                        for j in range(args.seq_len):
                            ep = int(exp_prod_int[n,h,i,j])
                            vq = int(V_q[n,h,j,t])
                            acc += ep * vq
                        X_num[n,h,i,t] = int(acc)
                        if abs(acc) > max_abs:
                            max_abs = abs(acc)

        # 9) Ensure X_num fits in signed 64-bit for saving into npz as i64.
        MAX_I64 = (1 << 63) - 1
        if max_abs > MAX_I64:
            # Attempt to scale down V_q uniformly to make X_num fit.
            scale_needed = math.ceil(max_abs / float(MAX_I64))
            print(f"Warning: X_num max_abs={max_abs} exceeds i64 max. Scaling down V_q by factor {scale_needed} and recomputing.")
            # integer division scaling (may lose precision)
            V_q = (V_q // scale_needed).astype(np.int64)
            # recompute X_num
            X_num = np.zeros((args.batch, args.num_head, args.seq_len, args.head_dim), dtype=object)
            max_abs2 = 0
            for n in range(args.batch):
                for h in range(args.num_head):
                    for i in range(args.seq_len):
                        for t in range(args.head_dim):
                            acc = 0
                            for j in range(args.seq_len):
                                ep = int(exp_prod_int[n,h,i,j])
                                vq = int(V_q[n,h,j,t])
                                acc += ep * vq
                            X_num[n,h,i,t] = int(acc)
                            if abs(acc) > max_abs2:
                                max_abs2 = abs(acc)
            max_abs = max_abs2
            if max_abs > MAX_I64:
                raise RuntimeError(f"Even after scaling V_q by {scale_needed}, X_num max_abs={max_abs} still exceeds i64 range. Reduce frac_bits/k_segments or use smaller v_scale.")

        # Convert X_num to int64 ndarray
        X_q = np.zeros((args.batch, args.num_head, args.seq_len, args.head_dim), dtype=np.int64)
        for n in range(args.batch):
            for h in range(args.num_head):
                for i in range(args.seq_len):
                    for t in range(args.head_dim):
                        X_q[n,h,i,t] = int(X_num[n,h,i,t])

        # 10) Save everything into integrated npz
        out_dir = os.path.dirname(args.out_npz) or "."
        os.makedirs(out_dir, exist_ok=True)

        save_kwargs = dict(
            a=a_q_full.astype(np.uint64),
            b=b_q_full.astype(np.uint64),
            c=c.astype(np.uint64),
            div=div.astype(np.uint64),
            rem=rem.astype(np.uint64),
            r=np.uint64(args.r),
            input_0=np.uint64(input_0_fin),
            c_0_converted=np.uint64(0),
            multiplier=np.uint64(1),
            frac_bits=np.int32(args.frac_bits),
            # softmax pieces (targets + limbs)
            matmul_targets=matmul_targets_to_save,
            limb_indices=limb_indices.astype(np.uint64),
            # embed softmax tables directly
            softmax_tables=softmax_tables_arr,
            softmax_tables_frac_bits=np.int32(softmax_frac_bits),
            softmax_scale_input=np.int64(softmax_scale_input),
            softmax_multiplier=np.int64(softmax_multiplier),
            softmax_base_scale_exp=np.int32(softmax_base_scale_exp),
            softmax_limb_bits=np.int32(softmax_limb_bits),
            softmax_k_segments=np.int32(softmax_k_segments),
            softmax_tables_path=np.array([tables_path]),
            attn_float=attn_float.astype(np.float64),
            # second-matmul quantized: V_q and X_q (integer numerator)
            V_q=V_q.astype(np.int64),
            X_q=X_q.astype(np.int64),
            v_scale=np.int64(args.v_scale),
            meta=np.array([{"batch": args.batch, "num_head": args.num_head, "seq_len": args.seq_len, "head_dim": args.head_dim}])
        )

        np.savez_compressed(args.out_npz, **save_kwargs)
        print("Wrote integrated npz to", args.out_npz)
        print("Embedded softmax tables from", tables_path)
        print("matmul_targets max =", flat_max, "min =", flat_min)
        print("X_q max_abs =", max_abs)
        if flat_max >= (1 << 64):
            print("Note: matmul_targets stored as object array because of u64 overflow risk.")

    else:
        # Nystrom branch: prepare all matmuls needed for forward:
        # - kernel_1: softmax(Q @ K_landmarks^T)       -> mat shape (batch,num_head,seq_len,num_landmarks)
        # - kernel_2: softmax(Q_landmarks @ K_landmarks^T) -> (batch,num_head,num_landmarks,num_landmarks)
        # - kernel_3: softmax(Q_landmarks @ K^T (with mask)) -> (batch,num_head,num_landmarks,seq_len)
        # We'll quantize and decompose targets for each of these matmuls and then build integer pipelines
        B = args.batch
        H = args.num_head
        S = args.seq_len
        L = args.num_landmarks
        D = args.head_dim

        if S % L != 0:
            raise ValueError("seq_len must be divisible by num_landmarks for the simple landmark averaging used here.")

        # compute landmarks (numpy)
        seg = S // L
        Q_landmarks_np = Q_np.reshape(B, H, L, seg, D).mean(axis=3)
        K_landmarks_np = K_np.reshape(B, H, L, seg, D).mean(axis=3)

        # Quantize/decompose each matmul needed
        # kernel_1: Q_np @ K_landmarks_np^T -> shape (B,H,S,L)
        a_q_k1, b_q_k1, input_0_k1, a_raw_k1, b_raw_k1 = quantize_with_auto_zero(Q_np, K_landmarks_np, scale_input=args.scale, input_0_default=args.input_0, check_uint64=False)
        b_raw_k1_t = b_raw_k1.transpose(0,1,3,2).copy()
        c_k1, div_k1, rem_k1, c0_used_k1 = decompose_targets(a_q_k1, b_q_k1, a_raw_k1, b_raw_k1_t, input_0_k1, c0=0, multiplier=1)
        matmul_targets_k1 = reconstruct_targets_from_c_div_rem(c_k1, div_k1, rem_k1)  # shape (B,H,S,L) as object

        # kernel_2: Q_landmarks_np @ K_landmarks_np^T -> shape (B,H,L,L)
        a_q_k2, b_q_k2, input_0_k2, a_raw_k2, b_raw_k2 = quantize_with_auto_zero(Q_landmarks_np, K_landmarks_np, scale_input=args.scale, input_0_default=args.input_0, check_uint64=False)
        b_raw_k2_t = b_raw_k2.transpose(0,1,3,2).copy()
        c_k2, div_k2, rem_k2, c0_used_k2 = decompose_targets(a_q_k2, b_q_k2, a_raw_k2, b_raw_k2_t, input_0_k2, c0=0, multiplier=1)
        matmul_targets_k2 = reconstruct_targets_from_c_div_rem(c_k2, div_k2, rem_k2)  # shape (B,H,L,L)

        # kernel_3: Q_landmarks_np @ K_np^T -> shape (B,H,L,S)
        a_q_k3, b_q_k3, input_0_k3, a_raw_k3, b_raw_k3 = quantize_with_auto_zero(Q_landmarks_np, K_np, scale_input=args.scale, input_0_default=args.input_0, check_uint64=False)
        b_raw_k3_t = b_raw_k3.transpose(0,1,3,2).copy()
        c_k3, div_k3, rem_k3, c0_used_k3 = decompose_targets(a_q_k3, b_q_k3, a_raw_k3, b_raw_k3_t, input_0_k3, c0=0, multiplier=1)
        matmul_targets_k3 = reconstruct_targets_from_c_div_rem(c_k3, div_k3, rem_k3)  # shape (B,H,L,S)

        # Convert to uint64 checks
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

        # 5) decompose each target into limbs (indices) for lookup for each kernel
        limb_indices_k1 = decompose_to_limbs_array(matmul_targets_k1, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)
        limb_indices_k2 = decompose_to_limbs_array(matmul_targets_k2, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)
        limb_indices_k3 = decompose_to_limbs_array(matmul_targets_k3, limb_bits=softmax_limb_bits, k_segments=softmax_k_segments)

        # 6) compute per-element exp_prod integers using embedded tables for each kernel
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
                        prod = 1
                        for k in range(Kt):
                            idx = int(limb_indices_k3[n,h,i,j,k])
                            if idx < 0 or idx >= W:
                                raise RuntimeError(f"limb index out of bounds for k3: k={k} idx={idx} width={W}")
                            val = int(softmax_tables_arr[k, idx])
                            prod = prod * val
                        exp_prod_k3[n,h,i,j] = prod

        # Optional: compute floating softmax for inspection for each kernel
        attn_k1_float = compute_float_softmax_from_targets_int_rect(matmul_targets_k1)  # (B,H,S,L)
        attn_k2_float = compute_float_softmax_from_targets_int(matmul_targets_k2)       # (B,H,L,L)
        attn_k3_float = compute_float_softmax_from_targets_int_rect(matmul_targets_k3)  # (B,H,L,S)

        # 7) generate random V and quantize to integers V_q (for right-most multiplication kernel3 @ V)
        rng = np.random.RandomState(args.seed + 1)
        V = rng.randn(B, H, S, D).astype(np.float64)
        V_q = quantize_matrix_fixed_point(V, args.v_scale).astype(np.int64)

        # 8) compute iterative inverse of kernel_2 float softmax using attn_module.iterative_inv
        # convert attn_k2_float to torch tensor of shape (B,H,L,L)
        K2_torch = torch.from_numpy(attn_k2_float.astype(np.float32))
        inv_torch = attn_module.iterative_inv(K2_torch, n_iter=6)  # we'll take this as the pseudoinverse witness
        inv_np = inv_torch.cpu().numpy()  # shape (B,H,L,L)

        # Quantize inv_np to integers for matmuls (scale specified by inv_scale)
        inv_q = quantize_matrix_fixed_point(inv_np, args.inv_scale).astype(np.int64)

        # 9) compute right part: kernel3 (exp_prod_k3) @ V_q -> right_num shape (B,H,L,D)
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

        # 10) compute left part: kernel1 (exp_prod_k1) @ inv_q -> left_num shape (B,H,S,L)
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

        # 11) final X_num = left_num @ right_num -> shape (B,H,S,D)
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

        # 12) Ensure final X_num fits in signed 64-bit for saving into npz as i64.
        MAX_I64 = (1 << 63) - 1
        if max_abs_final > MAX_I64:
            # Try scaling down V_q and inv_q uniformly to attempt to fit into i64.
            # Compute scale needed (conservative)
            scale_needed = math.ceil(max_abs_final / float(MAX_I64))
            print(f"Warning: X_num max_abs={max_abs_final} exceeds i64 max. Scaling down V_q and inv_q by factor {scale_needed} and recomputing.")
            V_q = (V_q // scale_needed).astype(np.int64)
            inv_q = (inv_q // scale_needed).astype(np.int64)

            # recompute right_num
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

            # recompute left_num
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

            # recompute final X_num
            X_num = np.zeros((B, H, S, D), dtype=object)
            max_abs_final2 = 0
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
                            if abs(acc) > max_abs_final2:
                                max_abs_final2 = abs(acc)
            max_abs_final = max_abs_final2
            if max_abs_final > MAX_I64:
                raise RuntimeError(f"Even after scaling V_q and inv_q by {scale_needed}, X_num max_abs={max_abs_final} still exceeds i64 range. Reduce frac_bits/k_segments or use smaller v_scale/inv_scale.")

        # 13) Convert X_num to int64 ndarray
        X_q = np.zeros((B, H, S, D), dtype=np.int64)
        for n in range(B):
            for h in range(H):
                for i in range(S):
                    for t in range(D):
                        X_q[n,h,i,t] = int(X_num[n,h,i,t])

        # 14) Save everything into integrated npz (include per-kernel targets, limb indices, iterative inverse, etc.)
        out_dir = os.path.dirname(args.out_npz) or "."
        os.makedirs(out_dir, exist_ok=True)

        save_kwargs = dict(
            # original quantization of Q,K for reference
            a_full=a_q_full.astype(np.uint64),
            b_full=b_q_full.astype(np.uint64),
            # kernel1 pieces
            matmul_targets_k1=matmul_targets_k1_save,
            limb_indices_k1=limb_indices_k1.astype(np.uint64),
            c_k1=c_k1.astype(np.uint64),
            div_k1=div_k1.astype(np.uint64),
            rem_k1=rem_k1.astype(np.uint64),
            # kernel2 pieces
            matmul_targets_k2=matmul_targets_k2_save,
            limb_indices_k2=limb_indices_k2.astype(np.uint64),
            c_k2=c_k2.astype(np.uint64),
            div_k2=div_k2.astype(np.uint64),
            rem_k2=rem_k2.astype(np.uint64),
            # kernel3 pieces
            matmul_targets_k3=matmul_targets_k3_save,
            limb_indices_k3=limb_indices_k3.astype(np.uint64),
            c_k3=c_k3.astype(np.uint64),
            div_k3=div_k3.astype(np.uint64),
            rem_k3=rem_k3.astype(np.uint64),

            # common softmax tables
            softmax_tables=softmax_tables_arr,
            softmax_tables_frac_bits=np.int32(softmax_frac_bits),
            softmax_scale_input=np.int64(softmax_scale_input),
            softmax_multiplier=np.int64(softmax_multiplier),
            softmax_base_scale_exp=np.int32(softmax_base_scale_exp),
            softmax_limb_bits=np.int32(softmax_limb_bits),
            softmax_k_segments=np.int32(softmax_k_segments),
            softmax_tables_path=np.array([tables_path]),

            # floating softmax inspection arrays
            attn_k1_float=attn_k1_float.astype(np.float64),
            attn_k2_float=attn_k2_float.astype(np.float64),
            attn_k3_float=attn_k3_float.astype(np.float64),

            # iterative inverse (float and quantized)
            inv_float=inv_np.astype(np.float64),
            inv_q=inv_q.astype(np.int64),
            inv_scale=np.int64(args.inv_scale),

            # right/left intermediate numerators and final X_q
            right_num=np.array(right_num, dtype=object),
            left_num=np.array(left_num, dtype=object),
            X_q=X_q.astype(np.int64),
            V_q=V_q.astype(np.int64),

            # metadata
            v_scale=np.int64(args.v_scale),
            r=np.uint64(args.r),
            input_0=np.uint64(input_0_k1),
            frac_bits=np.int32(args.frac_bits),
            meta=np.array([{"batch": B, "num_head": H, "seq_len": S, "head_dim": D, "num_landmarks": L}])
        )

        np.savez_compressed(args.out_npz, **save_kwargs)
        print("Wrote integrated npz to", args.out_npz)
        print("Embedded softmax tables from", tables_path)
        print("kernel1 targets max =", max_k1, "min =", min_k1)
        print("kernel2 targets max =", max_k2, "min =", min_k2)
        print("kernel3 targets max =", max_k3, "min =", min_k3)
        print("X_q max_abs =", max_abs_final)