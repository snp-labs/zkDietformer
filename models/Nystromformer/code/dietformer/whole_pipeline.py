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
    parser.add_argument("--tables_outdir", type=str, default="tmp_tables")
    parser.add_argument("--k_segments", type=int, default=2, help="K segments used in softmax tables")
    parser.add_argument("--limb_bits", type=int, default=2, help="bits per limb used in softmax tables")
    args = parser.parse_args()

    # initialize Nystrom config (kept for compatibility)
    config = {"head_dim": args.head_dim, "num_head": args.num_head, "num_landmarks": args.seq_len, "seq_len": args.seq_len}
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

    # 2) quantize Q,K and decompose targets (first matmul)
    a_q, b_q, input_0_fin, a_raw, b_raw = quantize_with_auto_zero(Q_np, K_np, scale_input=args.scale, input_0_default=args.input_0, check_uint64=False)
    b_raw_t = b_raw.transpose(0,1,3,2).copy()

    max_val = max(int(a_q.max()), int(b_q.max()))
    if max_val >= (1 << 63):
        print("Warning: quantized values are large; consider lowering scale to avoid overflow when casting to u64.")

    c, div, rem, c0_used = decompose_targets(a_q, b_q, a_raw, b_raw_t, input_0_fin, c0=0, multiplier=1)

    # 3) generate softmax tables (quantized) and load them so we can embed the arrays
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
    # softmax_tables_arr has shape (K, table_width)
    K = softmax_tables_arr.shape[0]
    W = softmax_tables_arr.shape[1]
    batch, num_head, seq_len, _ = matmul_targets_to_save.shape

    # Build exp_prod_int array shape (batch,num_head,seq_len,seq_len) as Python int (use object dtype to be safe)
    exp_prod_int = np.zeros((batch, num_head, seq_len, seq_len), dtype=object)
    for n in range(batch):
        for h in range(num_head):
            for i in range(seq_len):
                for j in range(seq_len):
                    prod = 1
                    for k in range(K):
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
    X_num = np.zeros((batch, num_head, args.seq_len, args.head_dim), dtype=object)
    max_abs = 0
    for n in range(batch):
        for h in range(num_head):
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
        X_num = np.zeros((batch, num_head, args.seq_len, args.head_dim), dtype=object)
        max_abs2 = 0
        for n in range(batch):
            for h in range(num_head):
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
    X_q = np.zeros((batch, num_head, args.seq_len, args.head_dim), dtype=np.int64)
    for n in range(batch):
        for h in range(num_head):
            for i in range(args.seq_len):
                for t in range(args.head_dim):
                    X_q[n,h,i,t] = int(X_num[n,h,i,t])

    # 10) Save everything into integrated npz
    out_dir = os.path.dirname(args.out_npz) or "."
    os.makedirs(out_dir, exist_ok=True)

    save_kwargs = dict(
        a=a_q.astype(np.uint64),
        b=b_q.astype(np.uint64),
        c=c.astype(np.uint64),
        div=div.astype(np.uint64),
        rem=rem.astype(np.uint64),
        r=np.uint64(args.r),
        input_0=np.uint64(input_0_fin),
        c_0_converted=np.uint64(c0_used),
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