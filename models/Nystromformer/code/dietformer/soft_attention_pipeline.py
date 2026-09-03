"""SOFT attention quantization witness pipeline.

Generates quantized witness NPZ from captured float intermediates.
Covers: projection, scaling, squared norms, cross-term matmul,
Gaussian lookup, MP pseudoinverse, chain matmul, O-projection.

No softmax, no row-sum normalization (SOFT design).
"""
import math
import os
import numpy as np
from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul,
)
from gaussian_table import (
    GAUSS_FRAC_BITS, GAUSS_TABLE_FRAC_BITS,
    generate_gaussian_table, gaussian_lookup, save_gaussian_tables,
)


def _load_optional_array(debug, key):
    if key not in debug:
        return None
    val = debug[key]
    if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 0:
        item = val.item()
        return None if item is None else val
    return val


def _cos_sim(a, b):
    """Cosine similarity between two float arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _max_abs_err(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def process_soft_attn_debug_npz(debug_npz_path, out_npz_path,
                                k_bits=26, n_iter=20):
    """Main pipeline: debug NPZ -> witness NPZ + A-gate report."""
    save_gaussian_tables(os.path.dirname(out_npz_path))

    debug = np.load(debug_npz_path, allow_pickle=True)

    # ── Phase 0: Projection witness (Q, V only; K = Q) ──
    hidden_states = debug['hidden_states']
    W_q = debug['W_q']
    b_q = _load_optional_array(debug, 'b_q')
    W_v = debug['W_v']
    b_v = _load_optional_array(debug, 'b_v')
    W_o = _load_optional_array(debug, 'W_o')
    b_o = _load_optional_array(debug, 'b_o')
    attn_output_for_proj = debug['attn_output_for_proj']
    mask = debug['mask']

    Q_proc_real = debug['Q_proc']
    B, H, S, D = Q_proc_real.shape
    HD = H * D
    m = debug['Q_landmarks'].shape[2]
    seg = S // m
    scale_factor = math.sqrt(math.sqrt(D))
    denom = 2.0 * math.sqrt(D)

    s_x_input, z_x_input = get_quantization_params(hidden_states)
    Q_x_input = quantize_tensor(hidden_states, s_x_input, z_x_input)

    proj_witness = {
        "Q_x_input": Q_x_input.astype(np.int64),
        "z_x_input": np.uint64(z_x_input),
    }

    # Q and V projections (K projection skipped: W_k = W_q)
    for name, W, b, apply_scale in [("q", W_q, b_q, True),
                                     ("v", W_v, b_v, False)]:
        s_w, z_w = get_quantization_params(W)
        Q_w = quantize_tensor(W, s_w, z_w)

        raw_real = np.matmul(hidden_states, W.T)
        if b is not None:
            raw_real = raw_real + b
        raw_real_4d = raw_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        s_raw, z_raw = get_quantization_params(raw_real_4d)

        M_proj = get_m_constant(s_x_input, s_w, s_raw, k=k_bits)

        Q_raw_flat, R_flat, M_prime_proj = sign_bit_grouping_matmul(
            Q_x_input, z_x_input, Q_w.T.astype(np.int64), z_w,
            z_raw, M_proj, k_bits
        )

        if b is not None:
            s_b = s_w * s_x_input
            Q_b_diff = np.round(b / s_b).astype(np.int64)
            numerator = Q_raw_flat.astype(np.int64) * (1 << k_bits) + R_flat.astype(np.int64)
            numerator = numerator + M_proj * Q_b_diff
            Q_raw_flat = (numerator >> k_bits).astype(np.int64)
            R_flat = (numerator % (1 << k_bits)).astype(np.int64)
        else:
            Q_b_diff = np.zeros(HD, dtype=np.int64)

        Q_raw_4d = Q_raw_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        R_proj_4d = R_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

        Q_w_diff = np.array([
            (Q_w[h*D:(h+1)*D, :] - z_w).astype(np.int64) for h in range(H)
        ])
        Q_b_diff_per_head = Q_b_diff.reshape(H, D)

        proj_witness[f"Q_raw_proj_{name}"] = Q_raw_4d.astype(np.int64)
        proj_witness[f"R_proj_{name}"] = R_proj_4d.astype(np.int64)
        proj_witness[f"Q_w_diff_{name}"] = Q_w_diff.astype(np.int64)
        proj_witness[f"Q_b_diff_{name}"] = Q_b_diff_per_head.astype(np.int64)
        proj_witness[f"z_w_{name}"] = np.uint64(z_w)
        proj_witness[f"M_proj_{name}"] = np.uint64(M_proj)
        proj_witness[f"M_prime_proj_{name}"] = np.uint64(M_prime_proj)
        proj_witness[f"z_raw_{name}"] = np.uint64(z_raw)

        if apply_scale:
            s_proc, z_proc = get_quantization_params(debug['Q_proc'])
            M_scale = int(np.floor((1 << k_bits) * s_raw / (scale_factor * s_proc)))
            if M_scale == 0:
                M_scale = 1
            M_prime_scale = int(np.floor((z_proc * (1 << k_bits)) / M_scale))

            inner_scale = (Q_raw_4d - z_raw).astype(np.int64)
            numerator_scale = M_scale * (inner_scale + M_prime_scale)
            Q_scaled = (numerator_scale >> k_bits).astype(np.int64)
            R_scale = (numerator_scale % (1 << k_bits)).astype(np.int64)

            proj_witness[f"Q_scaled_{name}"] = Q_scaled.astype(np.int64)
            proj_witness[f"R_scale_{name}"] = R_scale.astype(np.int64)
            proj_witness[f"M_scale_{name}"] = np.uint64(M_scale)
            proj_witness[f"M_prime_scale_{name}"] = np.uint64(M_prime_scale)

            Q_q = Q_scaled.astype(np.int64)
            s_q, z_q = s_proc, z_proc
        else:
            Q_v = Q_raw_4d.astype(np.int64)
            s_v, z_v = s_raw, z_raw

    # ── Phase 1: Landmarks (segment averaging from chain Q_q) ──
    Q_sum = Q_q.reshape(B, H, m, seg, D).sum(axis=-2)
    Q_landmarks_q = Q_sum // seg
    R_mean_q = Q_sum % seg

    # ── Phase 2: Squared norms (integer domain) ──
    # norm_q[b][h][i] = sum_d (Q_q[b][h][i][d] - z_q)^2
    Q_q_centered = (Q_q - z_q).astype(np.int64)
    norm_q_int = np.sum(Q_q_centered ** 2, axis=-1)  # [B,H,S]

    Q_lm_centered = (Q_landmarks_q - z_q).astype(np.int64)
    norm_qtilde_int = np.sum(Q_lm_centered ** 2, axis=-1)  # [B,H,m]

    # ── Phase 3: Cross-term matmul (ZEN) ──
    # C_B = Q_q @ Q~^T : [B,H,S,D] @ [B,H,D,m] -> [B,H,S,m]
    Q_lm_T = np.swapaxes(Q_landmarks_q, -1, -2)  # [B,H,D,m]

    cross_B_real = debug['cross_B']
    s_cb, z_cb = get_quantization_params(cross_B_real)

    M_cb = get_m_constant(s_q, s_q, s_cb, k=k_bits)
    Q_sim_cross_B, R_cross_B, M_prime_cb = sign_bit_grouping_matmul(
        Q_q, z_q, Q_lm_T, z_q, z_cb, M_cb, k_bits
    )

    # C_A = Q~ @ Q~^T : [B,H,m,D] @ [B,H,D,m] -> [B,H,m,m]
    cross_A_real = debug['cross_A']
    s_ca, z_ca = get_quantization_params(cross_A_real)

    M_ca = get_m_constant(s_q, s_q, s_ca, k=k_bits)
    Q_sim_cross_A, R_cross_A, M_prime_ca = sign_bit_grouping_matmul(
        Q_landmarks_q, z_q, Q_lm_T, z_q, z_ca, M_ca, k_bits
    )

    # ── Phase 4: Gaussian distance and lookup ──
    # dist_B[i][j] = norm_q[i] + norm_qtilde[j] - 2 * cross_B[i][j]
    # The distance is in the "integer squared norm" scale, not the cross-term scale.
    # We need to combine: norm uses (Q-z)^2 directly (scale = s_q^2),
    # cross uses ZEN output (scale = s_cb).
    # To combine, we compute dist in a unified quantized domain.

    # Strategy: compute dist as float, then quantize together.
    # This follows chain-output principle: use chain cross-term Q_sim values.
    # Dequantize cross to float for distance combination, then quantize dist.
    cross_B_deq = (Q_sim_cross_B.astype(np.float64) - z_cb) * s_cb
    norm_q_deq = norm_q_int.astype(np.float64) * (s_q ** 2)
    norm_qt_deq = norm_qtilde_int.astype(np.float64) * (s_q ** 2)

    dist_B_float = (norm_q_deq[:, :, :, np.newaxis]
                    + norm_qt_deq[:, :, np.newaxis, :]
                    - 2.0 * cross_B_deq)
    # Divide by denom = 2*sqrt(D) for Gaussian kernel argument
    dist_B_scaled = dist_B_float / denom
    # Clamp to non-negative (numerical noise can make tiny negatives)
    dist_B_scaled = np.maximum(dist_B_scaled, 0.0)

    s_dist_B, z_dist_B = get_quantization_params(dist_B_scaled)
    Q_dist_B = quantize_tensor(dist_B_scaled, s_dist_B, z_dist_B)

    # For Gaussian lookup, we need (Q_dist, R_dist) pair.
    # Since dist is freshly quantized (not from ZEN matmul), R_dist = 0.
    R_dist_B = np.zeros_like(Q_dist_B, dtype=np.int64)

    gauss_table = generate_gaussian_table(GAUSS_TABLE_FRAC_BITS, GAUSS_FRAC_BITS)
    kernel_B_val, table_idx_B, r_gauss_B, M_gauss_B, clamp_lo_B, clamp_hi_B = gaussian_lookup(
        Q_dist_B.astype(np.int64), R_dist_B, s_dist_B, k_bits,
        GAUSS_FRAC_BITS, GAUSS_TABLE_FRAC_BITS, gauss_table
    )

    # A kernel: dist_A
    cross_A_deq = (Q_sim_cross_A.astype(np.float64) - z_ca) * s_ca

    dist_A_float = (norm_qt_deq[:, :, :, np.newaxis]
                    + norm_qt_deq[:, :, np.newaxis, :]
                    - 2.0 * cross_A_deq)
    dist_A_scaled = dist_A_float / denom
    dist_A_scaled = np.maximum(dist_A_scaled, 0.0)

    s_dist_A, z_dist_A = get_quantization_params(dist_A_scaled)
    Q_dist_A = quantize_tensor(dist_A_scaled, s_dist_A, z_dist_A)
    R_dist_A = np.zeros_like(Q_dist_A, dtype=np.int64)

    kernel_A_val, table_idx_A, r_gauss_A, M_gauss_A, clamp_lo_A, clamp_hi_A = gaussian_lookup(
        Q_dist_A.astype(np.int64), R_dist_A, s_dist_A, k_bits,
        GAUSS_FRAC_BITS, GAUSS_TABLE_FRAC_BITS, gauss_table
    )

    # ── A-gate 5: Clamping check ──
    # Clamping = np.clip actually changed a value (pre-clip != post-clip).
    # For Gaussian distances (non-negative), low clamping (table_idx < 0) is
    # impossible. table_idx = 0 for dist = 0 is a natural value, not clamping.
    TABLE_SIZE = len(gauss_table)
    n_clamp_B = clamp_lo_B + clamp_hi_B
    n_clamp_A = clamp_lo_A + clamp_hi_A
    print(f"  [Clamping] B: lo={clamp_lo_B} hi={clamp_hi_B}, "
          f"A: lo={clamp_lo_A} hi={clamp_hi_A}")
    print(f"  table_idx_B: min={int(np.min(table_idx_B))}, "
          f"max={int(np.max(table_idx_B))}, TABLE_SIZE={TABLE_SIZE}")
    print(f"  table_idx_A: min={int(np.min(table_idx_A))}, "
          f"max={int(np.max(table_idx_A))}, TABLE_SIZE={TABLE_SIZE}")
    print(f"  Margin: B high={TABLE_SIZE - 1 - int(np.max(table_idx_B))}, "
          f"A high={TABLE_SIZE - 1 - int(np.max(table_idx_A))}")
    if n_clamp_B > 0 or n_clamp_A > 0:
        print("  STOP: Clamping detected. Recovery injectivity broken.")

    # ── Phase 5: MP pseudoinverse ──
    inv_A_real = debug['inv_A']
    s_inv, z_inv = get_quantization_params(inv_A_real)
    Q_inv_A = quantize_tensor(inv_A_real, s_inv, z_inv).astype(np.int64)

    # Gaussian kernel zero-point and scale
    z_gauss = 0  # Gaussian output is in [0, 2^FRAC_BITS], zero-point = 0
    s_gauss = 1.0 / (1 << GAUSS_FRAC_BITS)

    A_q = kernel_A_val.astype(np.int64)

    # Step 1: A @ A^{+} = Q_y1
    M_inv1 = get_m_constant(s_gauss, s_inv, s_gauss, k=k_bits)
    Q_y1, R_inv1, M_prime_inv1 = sign_bit_grouping_matmul(
        A_q, z_gauss, Q_inv_A, z_inv, z_gauss, M_inv1, k_bits
    )
    # Step 2: Q_y1 @ A = Q_y2 (condition 1: Q_y2 ~ A)
    M_inv2 = get_m_constant(s_gauss, s_gauss, s_gauss, k=k_bits)
    Q_y2, R_inv2, M_prime_inv2 = sign_bit_grouping_matmul(
        Q_y1, z_gauss, A_q, z_gauss, z_gauss, M_inv2, k_bits
    )
    # Step 2-1: A^{+} @ A = Q_y3
    M_inv3 = get_m_constant(s_inv, s_gauss, s_inv, k=k_bits)
    Q_y3, R_inv3, M_prime_inv3 = sign_bit_grouping_matmul(
        Q_inv_A, z_inv, A_q, z_gauss, z_inv, M_inv3, k_bits
    )
    # Step 2-2: Q_y3 @ A^{+} = Q_y4 (condition 2)
    M_inv4 = get_m_constant(s_inv, s_inv, s_inv, k=k_bits)
    Q_y4, R_inv4, M_prime_inv4 = sign_bit_grouping_matmul(
        Q_y3, z_inv, Q_inv_A, z_inv, z_inv, M_inv4, k_bits
    )

    # ── A-gate 6: MP condition residuals (all four) ──
    # Condition 1: |Q_y2 - A_q|
    cond1_diff = np.abs(Q_y2.astype(np.int64) - A_q.astype(np.int64))
    cond1_max = int(np.max(cond1_diff))
    # Condition 2: |Q_y4 - Q_inv_A|
    cond2_diff = np.abs(Q_y4.astype(np.int64) - Q_inv_A.astype(np.int64))
    cond2_max = int(np.max(cond2_diff))
    # Condition 3: |Q_y1[i][j] - Q_y1[j][i]| (symmetry)
    Q_y1_T = np.swapaxes(Q_y1, -1, -2)
    cond3_diff = np.abs(Q_y1.astype(np.int64) - Q_y1_T.astype(np.int64))
    cond3_max = int(np.max(cond3_diff))
    # Condition 4: |Q_y3[i][j] - Q_y3[j][i]| (symmetry)
    Q_y3_T = np.swapaxes(Q_y3, -1, -2)
    cond4_diff = np.abs(Q_y3.astype(np.int64) - Q_y3_T.astype(np.int64))
    cond4_max = int(np.max(cond4_diff))

    quant_range = 4095  # 12-bit [0, 4095]
    print(f"\n  [MP Conditions]")
    print(f"  {'Cond':>5} | {'max|diff|':>10} | {'% of 4095':>10} | "
          f"{'3x':>8} | {'5x':>8} | {'Nystrom':>8} | {'Paper':>8}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for cname, cmax, nval, pval in [
        ("1", cond1_max, 4096, 512),
        ("2", cond2_max, None, None),
        ("3", cond3_max, 512, 64),
        ("4", cond4_max, 32, 4),
    ]:
        pct = f"{cmax/quant_range*100:.2f}%"
        x3 = cmax * 3
        x5 = cmax * 5
        nstr = str(nval) if nval is not None else "N/A"
        pstr = str(pval) if pval is not None else "N/A"
        print(f"  {cname:>5} | {cmax:>10} | {pct:>10} | "
              f"{x3:>8} | {x5:>8} | {nstr:>8} | {pstr:>8}")

    # ── Phase 6: Chain matmul ──
    # temp = B^T @ V : [B,H,m,S] @ [B,H,S,D] -> [B,H,m,D]
    temp_real = debug['temp']
    s_temp, z_temp = get_quantization_params(temp_real)

    z_B = z_gauss
    s_B = s_gauss
    B_kernel_T = np.swapaxes(kernel_B_val, -1, -2).astype(np.int64)  # [B,H,m,S]

    M_temp = get_m_constant(s_B, s_v, s_temp, k=k_bits)
    Q_sim_temp, R_matmul_temp, M_prime_temp = sign_bit_grouping_matmul(
        B_kernel_T, z_B, Q_v.astype(np.int64), z_v, z_temp, M_temp, k_bits
    )

    # mid = B @ A^{+} : [B,H,S,m] @ [B,H,m,m] -> [B,H,S,m]
    mid_real = debug['mid']
    s_mid, z_mid = get_quantization_params(mid_real)

    M_mid = get_m_constant(s_B, s_inv, s_mid, k=k_bits)
    Q_sim_mid, R_matmul_mid, M_prime_mid = sign_bit_grouping_matmul(
        kernel_B_val.astype(np.int64), z_B,
        Q_inv_A.astype(np.int64), z_inv,
        z_mid, M_mid, k_bits
    )

    # Y = mid @ temp : [B,H,S,m] @ [B,H,m,D] -> [B,H,S,D]
    Y_real = debug['Y']
    s_Y, z_Y = get_quantization_params(Y_real)

    M_Y = get_m_constant(s_mid, s_temp, s_Y, k=k_bits)
    Q_sim_Y, R_matmul_Y, M_prime_Y = sign_bit_grouping_matmul(
        Q_sim_mid.astype(np.int64), z_mid,
        Q_sim_temp.astype(np.int64), z_temp,
        z_Y, M_Y, k_bits
    )

    # ── Phase 7: O-projection ──
    Q_sim_Y_flat = Q_sim_Y.transpose(0, 2, 1, 3).reshape(B, S, HD)
    s_x_out, z_x_out = s_Y, z_Y

    s_w_o, z_w_o = get_quantization_params(W_o)
    Q_w_o = quantize_tensor(W_o, s_w_o, z_w_o)

    o_real = np.matmul(attn_output_for_proj, W_o.T)
    if b_o is not None:
        o_real = o_real + b_o
    o_real_4d = o_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    s_raw_o, z_raw_o = get_quantization_params(o_real_4d)

    M_proj_o = get_m_constant(s_x_out, s_w_o, s_raw_o, k=k_bits)

    Q_o_flat, R_o_flat, M_prime_proj_o = sign_bit_grouping_matmul(
        Q_sim_Y_flat.astype(np.int64), z_x_out,
        Q_w_o.T.astype(np.int64), z_w_o,
        z_raw_o, M_proj_o, k_bits
    )

    if b_o is not None:
        s_b_o = s_w_o * s_x_out
        Q_b_diff_o = np.round(b_o / s_b_o).astype(np.int64)
        numerator_o = Q_o_flat.astype(np.int64) * (1 << k_bits) + R_o_flat.astype(np.int64)
        numerator_o = numerator_o + M_proj_o * Q_b_diff_o
        Q_o_flat = (numerator_o >> k_bits).astype(np.int64)
        R_o_flat = (numerator_o % (1 << k_bits)).astype(np.int64)
    else:
        Q_b_diff_o = np.zeros(HD, dtype=np.int64)

    Q_o_4d = Q_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    R_o_4d = R_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

    Q_w_diff_o = np.array([
        (Q_w_o[h*D:(h+1)*D, :] - z_w_o).astype(np.int64) for h in range(H)
    ])
    Q_b_diff_o_per_head = Q_b_diff_o.reshape(H, D)

    proj_witness.update({
        "Q_attn_output": Q_sim_Y_flat.astype(np.int64),
        "z_attn_output": np.uint64(z_x_out),
        "Q_raw_proj_o": Q_o_4d.astype(np.int64),
        "R_proj_o": R_o_4d.astype(np.int64),
        "Q_w_diff_o": Q_w_diff_o.astype(np.int64),
        "Q_b_diff_o": Q_b_diff_o_per_head.astype(np.int64),
        "z_w_o": np.uint64(z_w_o),
        "M_proj_o": np.uint64(M_proj_o),
        "M_prime_proj_o": np.uint64(M_prime_proj_o),
        "z_raw_o": np.uint64(z_raw_o),
        "s_raw_o": np.float64(s_raw_o),
    })

    # ── A-gate 7: Newton-Raphson convergence ──
    _run_newton_convergence_report(debug, n_iter)

    # ── A-gate 8: Fidelity ──
    fidelity = {}
    stages = [
        ("landmarks", debug['Q_landmarks'],
         (Q_landmarks_q.astype(np.float64) - z_q) * s_q),
        ("cross_B", debug['cross_B'],
         (Q_sim_cross_B.astype(np.float64) - z_cb) * s_cb),
        ("cross_A", debug['cross_A'],
         (Q_sim_cross_A.astype(np.float64) - z_ca) * s_ca),
        ("dist_B", debug['dist_B_raw'] / denom, dist_B_scaled),
        ("dist_A", debug['dist_A_raw'] / denom, dist_A_scaled),
        ("kernel_B", debug['kernel_B'],
         kernel_B_val.astype(np.float64) / (1 << GAUSS_FRAC_BITS)),
        ("kernel_A", debug['kernel_A'],
         kernel_A_val.astype(np.float64) / (1 << GAUSS_FRAC_BITS)),
        ("inv_A", debug['inv_A'],
         (Q_inv_A.astype(np.float64) - z_inv) * s_inv),
        ("temp", debug['temp'],
         (Q_sim_temp.astype(np.float64) - z_temp) * s_temp),
        ("mid", debug['mid'],
         (Q_sim_mid.astype(np.float64) - z_mid) * s_mid),
        ("Y", debug['Y'],
         (Q_sim_Y.astype(np.float64) - z_Y) * s_Y),
        ("O-proj", o_real_4d,
         (Q_o_4d.astype(np.float64) - z_raw_o) * s_raw_o),
    ]
    print(f"\n  [Fidelity]")
    print(f"  {'Stage':>12} | {'CosSim':>10} | {'MaxAbsErr':>12}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}")
    for stage_name, ref, approx in stages:
        cs = _cos_sim(ref, approx)
        mae = _max_abs_err(ref, approx)
        fidelity[stage_name] = {"cos_sim": cs, "max_abs_err": mae}
        print(f"  {stage_name:>12} | {cs:>10.8f} | {mae:>12.6e}")

    o_proj_cos = fidelity["O-proj"]["cos_sim"]
    if o_proj_cos < 0.9:
        print(f"\n  STOP: O-proj CosSim = {o_proj_cos:.8f} < 0.9")

    # ── A-gate 9: dist statistics ──
    print(f"\n  [dist statistics]")
    for dname, darr in [("dist_B", dist_B_scaled), ("dist_A", dist_A_scaled)]:
        print(f"  {dname}: min={np.min(darr):.6f}, max={np.max(darr):.6f}, "
              f"mean={np.mean(darr):.6f}")

    # ── A-gate 3: Gaussian kernel consistency ──
    _kernel_B_from_table = gauss_table[table_idx_B.astype(int)]
    n_mismatch_B = int(np.sum(kernel_B_val != _kernel_B_from_table))
    _kernel_A_from_table = gauss_table[table_idx_A.astype(int)]
    n_mismatch_A = int(np.sum(kernel_A_val != _kernel_A_from_table))
    print(f"\n  [Gaussian consistency] B: {n_mismatch_B}/{kernel_B_val.size}, "
          f"A: {n_mismatch_A}/{kernel_A_val.size}")

    # ── A-gate 2: ZEN equation verification ──
    _verify_zen_equations(proj_witness, Q_q, z_q, Q_v, z_v,
                          Q_sim_cross_B, R_cross_B, M_cb, M_prime_cb, z_cb,
                          Q_sim_cross_A, R_cross_A, M_ca, M_prime_ca, z_ca,
                          Q_landmarks_q, Q_lm_T,
                          A_q, z_gauss, Q_inv_A, z_inv,
                          Q_y1, R_inv1, M_inv1, M_prime_inv1,
                          Q_y2, R_inv2, M_inv2, M_prime_inv2,
                          Q_y3, R_inv3, M_inv3, M_prime_inv3,
                          Q_y4, R_inv4, M_inv4, M_prime_inv4,
                          B_kernel_T, z_B, Q_sim_temp, R_matmul_temp,
                          M_temp, M_prime_temp, z_temp,
                          kernel_B_val, Q_sim_mid, R_matmul_mid,
                          M_mid, M_prime_mid, z_mid,
                          Q_sim_Y, R_matmul_Y, M_Y, M_prime_Y, z_Y,
                          k_bits)

    # ── A-gate 4: Recovery injectivity ──
    _verify_recovery(norm_q_int, norm_qtilde_int, s_q,
                     Q_sim_cross_B, R_cross_B, s_cb, z_cb, M_cb, M_prime_cb,
                     Q_sim_cross_A, R_cross_A, s_ca, z_ca, M_ca, M_prime_ca,
                     table_idx_B, r_gauss_B, M_gauss_B,
                     table_idx_A, r_gauss_A, M_gauss_A,
                     Q_dist_B, Q_dist_A, s_dist_B, s_dist_A,
                     z_dist_B, z_dist_A, denom, k_bits)

    # ── A-gate 10: Misc ──
    import torch
    kernel_A_tensor = torch.tensor(debug['kernel_A'].astype(np.float64))
    # Min singular value per (b,h) head
    svd_mins = []
    for bi in range(B):
        for hi in range(H):
            sv = torch.linalg.svdvals(kernel_A_tensor[bi, hi])
            svd_mins.append(float(sv[-1]))
    print(f"\n  [Misc]")
    print(f"  Sequence length: S={S}, landmarks: m={m}, seg={seg}")
    print(f"  A min singular values (per head): {svd_mins}")
    print(f"  Masking: Q_proc and V pre-masked at z_q={z_q}, z_v={z_v}")

    # ── NPZ save ──
    quant_metrics = {
        "cosine_similarity": fidelity.get("O-proj", {}).get("cos_sim", 0.0),
        "cond1_max": cond1_max,
        "cond2_max": cond2_max,
        "cond3_max": cond3_max,
        "cond4_max": cond4_max,
    }

    np.savez_compressed(
        out_npz_path,
        **proj_witness,
        # Chain inputs
        Q_q=Q_q.astype(np.int64),
        z_q=np.uint64(z_q),
        Q_v=Q_v.astype(np.int64),
        z_v=np.uint64(z_v),
        # Landmarks
        Q_landmarks_q=Q_landmarks_q.astype(np.int64),
        R_mean_q=R_mean_q.astype(np.int64),
        seg_size=np.uint64(seg),
        # Squared norms (integer)
        norm_q_int=norm_q_int.astype(np.int64),
        norm_qtilde_int=norm_qtilde_int.astype(np.int64),
        # Cross-term B
        Q_sim_cross_B=Q_sim_cross_B.astype(np.int64),
        R_cross_B=R_cross_B.astype(np.int64),
        M_cross_B=np.uint64(M_cb),
        M_prime_cross_B=np.uint64(M_prime_cb),
        z_sim_cross_B=np.uint64(z_cb),
        # Cross-term A
        Q_sim_cross_A=Q_sim_cross_A.astype(np.int64),
        R_cross_A=R_cross_A.astype(np.int64),
        M_cross_A=np.uint64(M_ca),
        M_prime_cross_A=np.uint64(M_prime_ca),
        z_sim_cross_A=np.uint64(z_ca),
        # Gaussian kernel B
        Q_dist_B=Q_dist_B.astype(np.int64),
        s_dist_B=np.float64(s_dist_B),
        z_dist_B=np.uint64(z_dist_B),
        kernel_B=kernel_B_val.astype(np.uint64),
        table_idx_B=table_idx_B.astype(np.int64),
        r_gauss_B=np.array(r_gauss_B, dtype=object),
        M_gauss_B=np.uint64(M_gauss_B),
        # Gaussian kernel A
        Q_dist_A=Q_dist_A.astype(np.int64),
        s_dist_A=np.float64(s_dist_A),
        z_dist_A=np.uint64(z_dist_A),
        kernel_A=kernel_A_val.astype(np.uint64),
        table_idx_A=table_idx_A.astype(np.int64),
        r_gauss_A=np.array(r_gauss_A, dtype=object),
        M_gauss_A=np.uint64(M_gauss_A),
        # Gaussian table
        gauss_table=gauss_table,
        gauss_frac_bits=np.uint32(GAUSS_FRAC_BITS),
        gauss_table_frac_bits=np.uint32(GAUSS_TABLE_FRAC_BITS),
        # Pseudoinverse
        Q_inv_A=Q_inv_A.astype(np.int64),
        z_inv=np.uint64(z_inv),
        M_inv_all=np.array([M_inv1, M_inv2, M_inv3, M_inv4], dtype=np.uint64),
        M_prime_inv_all=np.array([M_prime_inv1, M_prime_inv2,
                                   M_prime_inv3, M_prime_inv4], dtype=np.uint64),
        Q_y_inv_intermediates=np.stack([Q_y1, Q_y2, Q_y3, Q_y4]).astype(np.int64),
        R_inv_remainders=np.stack([R_inv1, R_inv2, R_inv3, R_inv4]).astype(np.int64),
        # Chain: temp = B^T @ V
        Q_sim_temp=Q_sim_temp.astype(np.int64),
        R_matmul_temp=R_matmul_temp.astype(np.int64),
        M_temp=np.uint64(M_temp),
        M_prime_temp=np.uint64(M_prime_temp),
        z_sim_temp=np.uint64(z_temp),
        # Chain: mid = B @ A^{+}
        Q_sim_mid=Q_sim_mid.astype(np.int64),
        R_matmul_mid=R_matmul_mid.astype(np.int64),
        M_mid=np.uint64(M_mid),
        M_prime_mid=np.uint64(M_prime_mid),
        z_sim_mid=np.uint64(z_mid),
        # Chain: Y = mid @ temp
        Q_sim_Y=Q_sim_Y.astype(np.int64),
        R_matmul_Y=R_matmul_Y.astype(np.int64),
        M_Y=np.uint64(M_Y),
        M_prime_Y=np.uint64(M_prime_Y),
        z_sim_Y=np.uint64(z_Y),
        # Metadata
        k_bits=np.uint32(k_bits),
        mask=mask.astype(np.int64),
        n_iter=np.uint32(n_iter),
    )

    return quant_metrics


def _run_newton_convergence_report(debug, n_iter):
    """A-gate 7: Newton-Raphson convergence per iteration."""
    import torch
    kernel_A = torch.tensor(debug['kernel_A'].astype(np.float64))
    B, H, mm, _ = kernel_A.shape

    norm_A = torch.norm(kernel_A, p=1, dim=(-2, -1), keepdim=True)
    alpha = 2.0 / (norm_A ** 2 + 1e-12)
    A_k = alpha * kernel_A

    print(f"\n  [Newton-Raphson convergence] T={n_iter}")
    print(f"  {'t':>4} | {'||A A_k A - A||/||A||':>22} | {'||A_k A A_k - A_k||/||A_k||':>28}")
    print(f"  {'-'*4}-+-{'-'*22}-+-{'-'*28}")
    for t in range(n_iter):
        A_k_next = 2 * A_k - A_k @ kernel_A @ A_k

        # Condition 1 residual: ||A A_k A - A|| / ||A||
        r1 = kernel_A @ A_k @ kernel_A - kernel_A
        nr1 = float(torch.norm(r1) / (torch.norm(kernel_A) + 1e-12))
        # Condition 2 residual: ||A_k A A_k - A_k|| / ||A_k||
        r2 = A_k @ kernel_A @ A_k - A_k
        nr2 = float(torch.norm(r2) / (torch.norm(A_k) + 1e-12))

        A_k = A_k_next
        if t < 5 or t >= n_iter - 3 or t % 5 == 0:
            print(f"  {t:>4} | {nr1:>22.12e} | {nr2:>28.12e}")


def _verify_zen_equations(proj_witness, Q_q, z_q, Q_v, z_v,
                          Q_sim_cross_B, R_cross_B, M_cb, M_prime_cb, z_cb,
                          Q_sim_cross_A, R_cross_A, M_ca, M_prime_ca, z_ca,
                          Q_landmarks_q, Q_lm_T,
                          A_q, z_gauss, Q_inv_A, z_inv,
                          Q_y1, R_inv1, M_inv1, M_prime_inv1,
                          Q_y2, R_inv2, M_inv2, M_prime_inv2,
                          Q_y3, R_inv3, M_inv3, M_prime_inv3,
                          Q_y4, R_inv4, M_inv4, M_prime_inv4,
                          B_kernel_T, z_B, Q_sim_temp, R_matmul_temp,
                          M_temp, M_prime_temp, z_temp,
                          kernel_B_val, Q_sim_mid, R_matmul_mid,
                          M_mid, M_prime_mid, z_mid,
                          Q_sim_Y, R_matmul_Y, M_Y, M_prime_Y, z_Y,
                          k_bits):
    """A-gate 2: Verify ZEN identity Q_sim * 2^k + R = M * inner for each matmul."""
    print(f"\n  [ZEN equations]")
    print(f"  {'Matmul':>15} | {'Total':>8} | {'Mismatch':>8} | {'MaxDev':>8}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    def check_zen(name, Q_w, z_w, Q_x, z_x, z_y, M, M_prime, Q_sim, R_mat, kb):
        n = Q_w.shape[-1]
        G1 = np.matmul(Q_w.astype(object), Q_x.astype(object))
        G2 = int(z_x) * np.sum(Q_w.astype(object), axis=-1, keepdims=True)
        G3 = int(z_w) * np.sum(Q_x.astype(object), axis=-2, keepdims=True)
        constant = n * int(z_w) * int(z_x)
        inner = G1 + constant + int(M_prime) - G2 - G3
        numerator = int(M) * inner
        lhs = Q_sim.astype(object) * (1 << kb) + R_mat.astype(object)
        diff = lhs - numerator
        total = int(np.prod(Q_sim.shape))
        n_ne = int(np.sum(diff != 0))
        max_dev = int(np.max(np.abs(diff))) if n_ne > 0 else 0
        print(f"  {name:>15} | {total:>8} | {n_ne:>8} | {max_dev:>8}")

    # Cross-term B: Q_q @ Q_lm_T
    check_zen("cross_B", Q_q, z_q, Q_lm_T, z_q, z_cb, M_cb, M_prime_cb,
              Q_sim_cross_B, R_cross_B, k_bits)
    # Cross-term A: Q_lm @ Q_lm_T
    check_zen("cross_A", Q_landmarks_q, z_q, Q_lm_T, z_q, z_ca, M_ca, M_prime_ca,
              Q_sim_cross_A, R_cross_A, k_bits)
    # MP step 1: A @ A^{+}
    check_zen("MP_step1", A_q, z_gauss, Q_inv_A, z_inv, z_gauss,
              M_inv1, M_prime_inv1, Q_y1, R_inv1, k_bits)
    # MP step 2: Q_y1 @ A
    check_zen("MP_step2", Q_y1, z_gauss, A_q, z_gauss, z_gauss,
              M_inv2, M_prime_inv2, Q_y2, R_inv2, k_bits)
    # MP step 2-1: A^{+} @ A
    check_zen("MP_step2-1", Q_inv_A, z_inv, A_q, z_gauss, z_inv,
              M_inv3, M_prime_inv3, Q_y3, R_inv3, k_bits)
    # MP step 2-2: Q_y3 @ A^{+}
    check_zen("MP_step2-2", Q_y3, z_inv, Q_inv_A, z_inv, z_inv,
              M_inv4, M_prime_inv4, Q_y4, R_inv4, k_bits)
    # temp: B^T @ V
    check_zen("temp", B_kernel_T, z_B, Q_v.astype(np.int64), z_v, z_temp,
              M_temp, M_prime_temp, Q_sim_temp, R_matmul_temp, k_bits)
    # mid: B @ A^{+}
    check_zen("mid", kernel_B_val.astype(np.int64), z_B,
              Q_inv_A.astype(np.int64), z_inv, z_mid,
              M_mid, M_prime_mid, Q_sim_mid, R_matmul_mid, k_bits)
    # Y: mid @ temp
    check_zen("Y", Q_sim_mid.astype(np.int64), z_mid,
              Q_sim_temp.astype(np.int64), z_temp, z_Y,
              M_Y, M_prime_Y, Q_sim_Y, R_matmul_Y, k_bits)


def _verify_recovery(norm_q_int, norm_qtilde_int, s_q,
                     Q_sim_cross_B, R_cross_B, s_cb, z_cb, M_cb, M_prime_cb,
                     Q_sim_cross_A, R_cross_A, s_ca, z_ca, M_ca, M_prime_ca,
                     table_idx_B, r_gauss_B, M_gauss_B,
                     table_idx_A, r_gauss_A, M_gauss_A,
                     Q_dist_B, Q_dist_A, s_dist_B, s_dist_A,
                     z_dist_B, z_dist_A, denom, k_bits):
    """A-gate 4: Recovery injectivity (reverse direction).

    Recover dist from committed values (table_idx, r_gauss) only.
    Then verify cross = (norm_q + norm_qtilde - dist*denom) / 2 matches
    the ZEN cross-term Q_sim.
    Forbidden inputs: C_B, C_A, Q_proc, Q~.
    """
    print(f"\n  [Recovery injectivity]")

    for kname, tidx, rg, Mg, Qd, sd, zd in [
        ("B", table_idx_B, r_gauss_B, M_gauss_B, Q_dist_B, s_dist_B, z_dist_B),
        ("A", table_idx_A, r_gauss_A, M_gauss_A, Q_dist_A, s_dist_A, z_dist_A),
    ]:
        # (a) Recover dist from table_idx and r_gauss
        # scaled_product = table_idx * 2^(2*k_bits) + r_gauss
        # X = scaled_product / M_gauss
        # dist = X >> k_bits, R_dist = X % 2^k_bits
        # But since R_dist = 0 in our pipeline, we can recover Q_dist directly:
        scaled_product_recovered = (tidx.astype(object) * (1 << (2 * k_bits))
                                    + rg.astype(object) if isinstance(rg, np.ndarray) else rg)
        # Actually we need: scaled_product = X * M_gauss where X = (Q_dist << k_bits) + R_dist
        # So X_recovered = scaled_product_recovered / M_gauss (integer division)
        X_recovered = scaled_product_recovered // Mg
        Q_dist_recovered = (X_recovered >> k_bits).astype(np.int64)

        total = int(np.prod(Qd.shape))
        match = int(np.sum(Q_dist_recovered == Qd.astype(np.int64)))
        mismatch = total - match
        print(f"  kernel_{kname} dist recovery: {match}/{total} match, {mismatch} mismatch")

    # (b)(c) Cross recovery from dist and norms
    # dist_scaled = (Q_dist - z_dist) * s_dist
    # dist_raw = dist_scaled * denom
    # cross = (norm_q_deq + norm_qtilde_deq - dist_raw) / 2
    # Then check cross matches dequantized Q_sim_cross
    for kname, Qd, sd, zd, Qsim, scb, zcb, nq, nqt in [
        ("B", Q_dist_B, s_dist_B, z_dist_B, Q_sim_cross_B, s_cb, z_cb,
         norm_q_int, norm_qtilde_int),
        ("A", Q_dist_A, s_dist_A, z_dist_A, Q_sim_cross_A, s_ca, z_ca,
         norm_qtilde_int, norm_qtilde_int),
    ]:
        dist_deq = (Qd.astype(np.float64) - zd) * sd
        dist_raw = dist_deq * denom
        nq_deq = nq.astype(np.float64) * (s_q ** 2)
        nqt_deq = nqt.astype(np.float64) * (s_q ** 2)
        if kname == "B":
            cross_recovered = (nq_deq[:, :, :, np.newaxis]
                               + nqt_deq[:, :, np.newaxis, :]
                               - dist_raw) / 2.0
        else:
            cross_recovered = (nq_deq[:, :, :, np.newaxis]
                               + nqt_deq[:, :, np.newaxis, :]
                               - dist_raw) / 2.0
        cross_from_zen = (Qsim.astype(np.float64) - zcb) * scb
        diff = np.abs(cross_recovered - cross_from_zen)
        total = int(np.prod(Qsim.shape))
        # Allow small floating point tolerance
        n_close = int(np.sum(diff < 1e-4))
        print(f"  kernel_{kname} cross recovery: {n_close}/{total} within 1e-4, "
              f"max_diff={float(np.max(diff)):.6e}")
