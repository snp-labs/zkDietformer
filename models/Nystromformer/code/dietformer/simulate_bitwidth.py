#!/usr/bin/env python3
"""
Simulate the full 14-step Nystrom attention quantization chain
at different bit widths to measure accumulated error.

Usage:
  python simulate_bitwidth.py --debug_npz <path> --bits 8 12
  python simulate_bitwidth.py --debug_npz <path> --bits 8 12 --table_frac_bits 10
"""
import math
import os
import argparse
import numpy as np

from softmax_table import (
    generate_softmax_tables,
    extract_exp_from_segments, combine_exp_limbs,
    FRAC_BITS, LIMB_BITS, K_SEGMENTS
)


# ── Quantization primitives ─────────────────────────────────────────────────

def get_quantization_params(data, bits=8):
    rmin = min(0.0, float(np.min(data)))
    rmax = max(0.0, float(np.max(data)))
    qmin, qmax = 0, (1 << bits) - 1
    scale = (rmax - rmin) / (qmax - qmin)
    if scale == 0:
        scale = 1e-9
    zp = int(np.round(np.clip(qmin - rmin / scale, qmin, qmax)))
    return float(scale), zp


def quantize_tensor(data, scale, zero_point, bits=8):
    qmin, qmax = 0, (1 << bits) - 1
    q = np.floor(data / scale + zero_point)
    return np.clip(q, qmin, qmax).astype(np.int64)


def get_m_constant(s_w, s_x, s_y, k=22):
    M = int(np.floor((1 << k) * (s_w * s_x) / s_y))
    if M <= 0:
        M = 1
    return M


def sign_bit_grouping_matmul(Q_w, z_w, Q_x, z_x, z_y, M, k_bits=22):
    n = Q_w.shape[-1]
    # Use Python arbitrary precision (object) for large k_bits to avoid int64 overflow
    use_bigint = k_bits >= 28
    if use_bigint:
        G1 = np.matmul(Q_w.astype(object), Q_x.astype(object))
        G2 = z_x * np.sum(Q_w.astype(object), axis=-1, keepdims=True)
        G3 = z_w * np.sum(Q_x.astype(object), axis=-2, keepdims=True)
    else:
        G1 = np.matmul(Q_w.astype(np.int64), Q_x.astype(np.int64))
        G2 = z_x * np.sum(Q_w.astype(np.int64), axis=-1, keepdims=True)
        G3 = z_w * np.sum(Q_x.astype(np.int64), axis=-2, keepdims=True)
    M_prime = int(np.floor((z_y * (1 << k_bits)) / M)) if M > 0 else 0
    constant_term = n * z_w * z_x
    inner_val = (G1 + constant_term + M_prime - G2 - G3)
    numerator = M * inner_val
    Q_y = (numerator >> k_bits)
    R = numerator % (1 << k_bits)
    if use_bigint:
        return Q_y.astype(np.int64), R.astype(np.int64), M_prime
    return Q_y.astype(np.int64), R.astype(np.int64), M_prime


def _load_optional_array(debug, key):
    if key not in debug:
        return None
    val = debug[key]
    if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 0:
        return None if val.item() is None else val
    return val


# ── Softmax implementations ─────────────────────────────────────────────────

# --- V1: Original (broken) --- M_sm = floor(s_q*s_k) = 0
def softmax_v1_old(Q_sim, R_mat, s_q, s_k, sm_tables, k_bits, frac_bits):
    M_sm = int(np.floor(s_q * s_k))
    M_sm = max(M_sm, 1)
    limbs, exp_limbs, _ = extract_exp_from_segments(Q_sim, R_mat, sm_tables, M_sm, k_bits)
    exp_total, _, _ = combine_exp_limbs(exp_limbs, frac_bits)
    exp_sum = np.sum(exp_total, axis=-1, keepdims=True)
    P = (exp_total << frac_bits) // exp_sum
    return P, M_sm, "OLD(broken)"


# --- V2: Fixed M_sm + multi-segment --- M_sm = floor(s_y * 2^k), double shift
def softmax_v2_fixed_msm(Q_sim, R_mat, s_y, sm_tables, k_bits, frac_bits):
    M_sm = int(np.floor(s_y * (1 << k_bits)))
    if M_sm <= 0:
        M_sm = 1
    X = (Q_sim.astype(object) << k_bits) + R_mat.astype(object)
    scaled_product = X * M_sm
    scaled_x = scaled_product >> (2 * k_bits)

    mask = (1 << LIMB_BITS) - 1
    exp_limbs = []
    for i in range(K_SEGMENTS):
        limb = (scaled_x >> (LIMB_BITS * i)) & mask
        val = sm_tables[i][limb.astype(np.int64).astype(int)]
        exp_limbs.append(val.astype(np.uint64))

    exp_total, _, _ = combine_exp_limbs(exp_limbs, frac_bits)
    exp_sum = np.sum(exp_total, axis=-1, keepdims=True)
    P = (exp_total << frac_bits) // exp_sum
    return P, M_sm, "FIXED_MSM"


# --- V3: Single table with fractional precision ---
def generate_fractional_exp_table(table_frac_bits=10, frac_bits=16):
    """
    Single exp table: table[i] = fixed_point(exp(i / 2^table_frac_bits))

    table_frac_bits=10 → 1024 steps/unit, covers [0, 64), uses 20480 of 65536 entries
    """
    TABLE_SIZE = 1 << 16  # 65536 entries
    scale = 1 << table_frac_bits
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    real_x = indices / scale
    real_x = np.clip(real_x, None, 20.0)
    exps = np.exp(real_x)
    fixed_exps = (exps * float(1 << frac_bits))
    fixed_exps = np.clip(fixed_exps, 0, float((1 << 64) - 1))
    return fixed_exps.astype(np.uint64)


def softmax_v3_single_table(Q_sim, R_mat, s_y, k_bits, frac_bits, table_frac_bits=10,
                             mask=None):
    """
    Single-table fractional softmax.

    table[i] = exp(i / 2^table_frac_bits)
    table_index = X * M_sm >> (2*k_bits) ≈ Q_sim * s_y * 2^table_frac_bits

    The z_y*s_y constant offset cancels in softmax normalization.
    If mask is provided, masked positions (mask==0) get exp=0.
    """
    table = generate_fractional_exp_table(table_frac_bits, frac_bits)
    TABLE_SIZE = len(table)

    # M_sm encodes: s_y * 2^table_frac_bits, lifted into fixed-point by 2^k_bits
    M_sm = int(np.floor(s_y * (1 << k_bits) * (1 << table_frac_bits)))
    if M_sm <= 0:
        M_sm = 1

    X = (Q_sim.astype(object) << k_bits) + R_mat.astype(object)
    scaled_product = X * M_sm
    table_index = scaled_product >> (2 * k_bits)

    # Clip to valid range
    table_index = np.clip(table_index.astype(np.int64), 0, TABLE_SIZE - 1)

    exp_vals = table[table_index.astype(int)].astype(object)

    # Zero out masked positions (simulate exp(-inf) = 0)
    if mask is not None:
        exp_vals = exp_vals * mask

    exp_sum = np.sum(exp_vals, axis=-1, keepdims=True)
    P = (exp_vals << frac_bits) // exp_sum

    return P, M_sm, f"SINGLE_TBL(tfb={table_frac_bits})"


# ── Metrics ──────────────────────────────────────────────────────────────────

def cos_sim(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dequant(Q, z, s):
    return (Q.astype(np.float64) - z) * s


def report(step_name, q_val, z, s, ref, step_num):
    approx = dequant(q_val, z, s)
    cs = cos_sim(approx, ref)
    rmse = float(np.sqrt(np.mean((ref.astype(np.float64) - approx) ** 2)))
    print(f"  Step {step_num:2d} [{step_name:20s}] CosSim={cs:.8f}  RMSE={rmse:.6e}")
    return cs


# ── Full 14-step chain simulation ───────────────────────────────────────────

def simulate_chain(debug_npz_path, bits=8, k_bits=22, frac_bits=FRAC_BITS,
                   softmax_version="v3", table_frac_bits=10):
    debug = np.load(debug_npz_path, allow_pickle=True)

    hidden_states = debug['hidden_states']
    W_q, W_k, W_v = debug['W_q'], debug['W_k'], debug['W_v']
    b_q = _load_optional_array(debug, 'b_q')
    b_k = _load_optional_array(debug, 'b_k')
    b_v = _load_optional_array(debug, 'b_v')
    W_o = debug['W_o']
    b_o = _load_optional_array(debug, 'b_o')
    attn_output_for_proj = debug['attn_output_for_proj']

    Q_proc_real = debug['Q_proc']
    B, H, S, D = Q_proc_real.shape
    HD = H * D
    scale_factor = math.sqrt(math.sqrt(D))

    results = {}
    step = 0

    # ── Steps 1-3: Q/K/V projections ─────────────────────────────────────
    s_x_input, z_x_input = get_quantization_params(hidden_states, bits)
    Q_x_input = quantize_tensor(hidden_states, s_x_input, z_x_input, bits)

    for name, W, b, apply_scale, ref_name in [
        ("q", W_q, b_q, True, 'Q_proc'), ("k", W_k, b_k, True, 'K_proc'),
        ("v", W_v, b_v, False, 'V')
    ]:
        step += 1
        s_w, z_w = get_quantization_params(W, bits)
        Q_w = quantize_tensor(W, s_w, z_w, bits)

        raw_real = np.matmul(hidden_states, W.T)
        if b is not None:
            raw_real = raw_real + b
        raw_real_4d = raw_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        s_raw, z_raw = get_quantization_params(raw_real_4d, bits)
        M_proj = get_m_constant(s_x_input, s_w, s_raw, k=k_bits)

        Q_raw_flat, R_flat, _ = sign_bit_grouping_matmul(
            Q_x_input, z_x_input, Q_w.T.astype(np.int64), z_w,
            z_raw, M_proj, k_bits)

        if b is not None:
            s_b = s_w * s_x_input
            Q_b_diff = np.round(b / s_b).astype(np.int64)
            numerator = Q_raw_flat * (1 << k_bits) + R_flat
            numerator = numerator + M_proj * Q_b_diff
            Q_raw_flat = (numerator >> k_bits)
            R_flat = numerator % (1 << k_bits)

        Q_raw_4d = Q_raw_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

        if apply_scale:
            ref_data = debug[ref_name]
            s_proc, z_proc = get_quantization_params(ref_data, bits)
            M_scale = int(np.floor((1 << k_bits) * s_raw / (scale_factor * s_proc)))
            if M_scale == 0:
                M_scale = 1
            M_prime_scale = int(np.floor((z_proc * (1 << k_bits)) / M_scale))
            inner_scale = (Q_raw_4d - z_raw).astype(np.int64)
            numerator_scale = M_scale * (inner_scale + M_prime_scale)
            Q_scaled = (numerator_scale >> k_bits)
            cs = report(f"proj_{name}+scale", Q_scaled, z_proc, s_proc, ref_data, step)
            results[f"proj_{name}"] = cs
            if name == "q":
                Q_q, s_q, z_q = Q_scaled, s_proc, z_proc
            else:
                Q_k, s_k, z_k = Q_scaled, s_proc, z_proc
        else:
            ref_data = debug[ref_name]
            cs = report(f"proj_{name}", Q_raw_4d, z_raw, s_raw, ref_data, step)
            results[f"proj_{name}"] = cs
            Q_v, s_v, z_v = Q_raw_4d, s_raw, z_raw

    # ── Steps 4-6: Kernel matmuls ────────────────────────────────────────
    L = debug['Q_landmarks'].shape[2]
    seg = S // L

    K_sum = Q_k.reshape(B, H, L, seg, D).sum(axis=-2)
    K_landmarks_q = K_sum // seg
    Q_sum = Q_q.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_q = Q_sum // seg

    kernels_info = []
    for ki, (Y_key, Q_left, Q_right, k_name) in enumerate([
        ('kernel_1_scores', Q_q, np.swapaxes(K_landmarks_q, -1, -2), "kernel_1"),
        ('kernel_2_scores', Q_landmarks_q, np.swapaxes(K_landmarks_q, -1, -2), "kernel_2"),
        ('kernel_3_scores', Q_landmarks_q, np.swapaxes(Q_k, -1, -2), "kernel_3"),
    ]):
        step += 1
        Y_real = debug[Y_key]
        s_y, z_y = get_quantization_params(Y_real, bits)
        z_y += 10
        M_k = get_m_constant(s_q, s_k, s_y, k=k_bits)
        Q_sim, R_mat, _ = sign_bit_grouping_matmul(Q_left, z_q, Q_right, z_k, z_y, M_k, k_bits)
        cs = report(k_name, Q_sim, z_y, s_y, Y_real, step)
        results[k_name] = cs
        kernels_info.append((Q_sim, R_mat, s_y, z_y, k_name))

    # ── Steps 7-9: Softmax ───────────────────────────────────────────────
    # Prepare multi-segment tables (for v1/v2)
    table_dir = os.path.dirname(debug_npz_path) or "."
    if softmax_version in ("v1", "v2"):
        table_path = generate_softmax_tables(table_dir, k_bits, frac_bits)
        sm_tables = np.load(table_path)['tables']

    softmax_refs = ['kernel_1', 'kernel_2', 'kernel_3']
    P_list = []
    for ki, (Q_sim, R_mat, s_y, z_y, k_name) in enumerate(kernels_info):
        step += 1
        sm_name = f"softmax_{k_name}"
        ref_sm = debug[softmax_refs[ki]].astype(np.float64)

        # Apply mask for kernel_3
        Q_sim_in, R_mat_in = Q_sim, R_mat
        sm_mask = None
        if ki == 2:
            mask = debug['mask']
            mask_4d = mask[:, None, None, :].astype(np.int64)
            sm_mask = mask_4d

        if softmax_version == "v1":
            P, M_sm, tag = softmax_v1_old(Q_sim_in, R_mat_in, s_q, s_k,
                                           sm_tables, k_bits, frac_bits)
        elif softmax_version == "v2":
            P, M_sm, tag = softmax_v2_fixed_msm(Q_sim_in, R_mat_in, s_y,
                                                  sm_tables, k_bits, frac_bits)
        else:  # v3
            P, M_sm, tag = softmax_v3_single_table(Q_sim_in, R_mat_in, s_y,
                                                     k_bits, frac_bits, table_frac_bits,
                                                     mask=sm_mask)

        approx_sm = P.astype(np.float64) / (1 << frac_bits)
        cs_sm = cos_sim(approx_sm, ref_sm)
        print(f"  Step {step:2d} [{sm_name:20s}] CosSim={cs_sm:.8f}  M_sm={M_sm}  ({tag})")
        results[sm_name] = cs_sm
        P_list.append(P)

    P_k1, P_k2, P_k3 = P_list

    # ── Pseudoinverse ────────────────────────────────────────────────────
    inv_k2_real = debug['inv_kernel_2']
    s_inv, z_inv = get_quantization_params(inv_k2_real, bits)
    Q_inv_k2 = quantize_tensor(inv_k2_real, s_inv, z_inv, bits)

    A_q = P_k2.astype(np.int64)
    z_p2, s_p2 = 0, 1.0 / (1 << frac_bits)

    step += 1
    M_inv1 = get_m_constant(s_p2, s_inv, s_p2, k=k_bits)
    Q_y1_inv, _, _ = sign_bit_grouping_matmul(A_q, z_p2, Q_inv_k2, z_inv, z_p2, M_inv1, k_bits)
    M_inv2 = get_m_constant(s_p2, s_p2, s_p2, k=k_bits)
    Q_y2_inv, _, _ = sign_bit_grouping_matmul(Q_y1_inv, z_p2, A_q, z_p2, z_p2, M_inv2, k_bits)
    print(f"  Step {step:2d} [{'pseudoinv (A*A+*A)':20s}] (verification)")

    step += 1
    M_inv3 = get_m_constant(s_inv, s_p2, s_inv, k=k_bits)
    Q_y3_inv, _, _ = sign_bit_grouping_matmul(Q_inv_k2, z_inv, A_q, z_p2, z_inv, M_inv3, k_bits)
    M_inv4 = get_m_constant(s_inv, s_inv, s_inv, k=k_bits)
    Q_y4_inv, _, _ = sign_bit_grouping_matmul(Q_y3_inv, z_inv, Q_inv_k2, z_inv, z_inv, M_inv4, k_bits)
    print(f"  Step {step:2d} [{'pseudoinv (A+*A*A+)':20s}] (verification)")

    # ── temp = P_k3 @ V ──────────────────────────────────────────────────
    step += 1
    temp_real = debug['temp']
    s_sim_temp, z_sim_temp = get_quantization_params(temp_real, bits)
    z_pk3, s_pk3 = 0, 1.0 / (1 << frac_bits)
    M_temp = get_m_constant(s_pk3, s_v, s_sim_temp, k=k_bits)
    Q_sim_temp, _, _ = sign_bit_grouping_matmul(
        P_k3.astype(np.int64), z_pk3, Q_v.astype(np.int64), z_v, z_sim_temp, M_temp, k_bits)
    cs = report("temp (P_k3 @ V)", Q_sim_temp, z_sim_temp, s_sim_temp, temp_real, step)
    results["temp"] = cs

    # ── mid = P_k1 @ inv_k2 ─────────────────────────────────────────────
    step += 1
    mid_real = debug['mid']
    s_mid, z_sim_mid = get_quantization_params(mid_real, bits)
    z_pk1, s_pk1 = 0, 1.0 / (1 << frac_bits)
    M_mid = get_m_constant(s_pk1, s_inv, s_mid, k=k_bits)
    Q_sim_mid, _, _ = sign_bit_grouping_matmul(
        P_k1.astype(np.int64), z_pk1, Q_inv_k2.astype(np.int64), z_inv, z_sim_mid, M_mid, k_bits)
    cs = report("mid (P_k1 @ inv)", Q_sim_mid, z_sim_mid, s_mid, mid_real, step)
    results["mid"] = cs

    # ── X = mid @ temp ───────────────────────────────────────────────────
    step += 1
    X_real = debug['X']
    s_x, z_sim_x = get_quantization_params(X_real, bits)
    M_x = get_m_constant(s_mid, s_sim_temp, s_x, k=k_bits)
    Q_sim_x, _, _ = sign_bit_grouping_matmul(
        Q_sim_mid.astype(np.int64), z_sim_mid, Q_sim_temp.astype(np.int64), z_sim_temp, z_sim_x, M_x, k_bits)
    cs = report("X (mid @ temp)", Q_sim_x, z_sim_x, s_x, X_real, step)
    results["X"] = cs

    # ── X_final = X + conv_term (if conv exists) ─────────────────────────
    conv_term = _load_optional_array(debug, 'conv_term')
    if conv_term is not None:
        step += 1
        X_final_real = debug['X_final']   # X + conv in float
        # Requantize X to X_real scale for addition
        X_deq = dequant(Q_sim_x, z_sim_x, s_x)
        X_final_approx = X_deq + conv_term  # conv is float (from model weights)
        s_xf, z_xf = get_quantization_params(X_final_real, bits)
        Q_xf = quantize_tensor(X_final_approx, s_xf, z_xf, bits)
        cs = report("X_final (X+conv)", Q_xf, z_xf, s_xf, X_final_real, step)
        results["X_final"] = cs
        # Use X_final for O-projection
        Q_sim_x_for_proj = Q_xf
        s_x_proj, z_x_proj = s_xf, z_xf
    else:
        Q_sim_x_for_proj = Q_sim_x
        s_x_proj, z_x_proj = s_x, z_sim_x

    # ── O-projection = X_final @ W_o^T + b_o ─────────────────────────────
    step += 1
    Q_sim_x_flat = Q_sim_x_for_proj.transpose(0, 2, 1, 3).reshape(B, S, HD)
    s_w_o, z_w_o = get_quantization_params(W_o, bits)
    Q_w_o = quantize_tensor(W_o, s_w_o, z_w_o, bits)

    o_real = np.matmul(attn_output_for_proj, W_o.T)
    if b_o is not None:
        o_real = o_real + b_o
    o_real_4d = o_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    s_raw_o, z_raw_o = get_quantization_params(o_real_4d, bits)

    M_proj_o = get_m_constant(s_x_proj, s_w_o, s_raw_o, k=k_bits)
    Q_o_flat, R_o_flat, _ = sign_bit_grouping_matmul(
        Q_sim_x_flat.astype(np.int64), z_x_proj,
        Q_w_o.T.astype(np.int64), z_w_o,
        z_raw_o, M_proj_o, k_bits)

    if b_o is not None:
        s_b_o = s_w_o * s_x_proj
        Q_b_diff_o = np.round(b_o / s_b_o).astype(np.int64)
        numerator_o = Q_o_flat * (1 << k_bits) + R_o_flat
        numerator_o = numerator_o + M_proj_o * Q_b_diff_o
        Q_o_flat = (numerator_o >> k_bits)
        R_o_flat = numerator_o % (1 << k_bits)

    Q_o_4d = Q_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    cs = report("O-projection", Q_o_4d, z_raw_o, s_raw_o, o_real_4d, step)
    results["O_projection"] = cs

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simulate Nystrom chain at various bit widths")
    parser.add_argument("--debug_npz", type=str, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--k_bits_base", type=int, default=22)
    parser.add_argument("--frac_bits", type=int, default=None,
                        help="Softmax output fixed-point bits (default: 16)")
    parser.add_argument("--softmax", choices=["v1", "v2", "v3"], default="v3",
                        help="v1=old(broken), v2=fixed M_sm+segments, v3=single fractional table")
    parser.add_argument("--table_frac_bits", type=int, default=10,
                        help="Table index fractional bits for v3 (default: 10 = 1024 steps/unit)")
    args = parser.parse_args()

    fb = args.frac_bits if args.frac_bits else FRAC_BITS

    all_results = {}
    for b in args.bits:
        k_bits = args.k_bits_base + (b - 8)
        print(f"\n{'='*72}")
        if args.softmax == "v3":
            print(f"  {b}-bit | k={k_bits} | frac={fb} | softmax=v3(tfb={args.table_frac_bits})")
        else:
            print(f"  {b}-bit | k={k_bits} | frac={fb} | softmax={args.softmax}")
        print(f"{'='*72}")
        try:
            results = simulate_chain(
                args.debug_npz, bits=b, k_bits=k_bits, frac_bits=fb,
                softmax_version=args.softmax, table_frac_bits=args.table_frac_bits)
            all_results[b] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[b] = None

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY  (softmax={args.softmax})")
    print(f"{'='*72}")
    steps = ["proj_q", "proj_k", "proj_v",
             "kernel_1", "kernel_2", "kernel_3",
             "softmax_kernel_1", "softmax_kernel_2", "softmax_kernel_3",
             "temp", "mid", "X", "O_projection"]
    header = f"{'Step':22s}"
    for b in args.bits:
        header += f" {b:2d}-bit   "
    print(header)
    print("-" * len(header))
    for s in steps:
        row = f"{s:22s}"
        for b in args.bits:
            if all_results.get(b) and s in all_results[b]:
                row += f" {all_results[b][s]:8.6f} "
            else:
                row += f" {'N/A':>8s} "
        print(row)


if __name__ == "__main__":
    main()
