"""A-2r: Integer-domain cross recovery check.

Verifies: from committed (table_idx, r_gauss) -> Q_dist -> cross,
can cross be recovered exactly in integer domain (not float)?
"""
import math
import numpy as np

out_dir = "outputs_n128_m32_L2_h128_soft_dense"
H = 2
D = 64
S = 128
m = 32
denom = 2.0 * math.sqrt(D)

layers = [
    "bert_encoder_layer_0_attention_self",
    "bert_encoder_layer_1_attention_self",
]

for li, lname in enumerate(layers):
    debug_path = f"{out_dir}/{out_dir}-batch0-{lname}.npz"
    witness_path = debug_path.replace(".npz", ".witness.npz")
    w = np.load(witness_path, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"LAYER {li}: {lname}")
    print(f"{'='*60}")

    for kname, prefix in [("B", "B"), ("A", "A")]:
        tidx = w[f'table_idx_{prefix}'].astype(np.int64)
        Q_dist = w[f'Q_dist_{prefix}'].astype(np.int64)
        z_dist = int(w[f'z_dist_{prefix}'])
        s_dist = float(w[f's_dist_{prefix}'])

        if kname == "B":
            Q_cross = w['Q_sim_cross_B'].astype(np.int64)
            z_cross = int(w['z_sim_cross_B'])
            norm_1 = w['norm_q_int'].astype(np.int64)    # [B,H,S]
            norm_2 = w['norm_qtilde_int'].astype(np.int64)  # [B,H,m]
            s_q = float(w['z_q'])  # need actual scale
        else:
            Q_cross = w['Q_sim_cross_A'].astype(np.int64)
            z_cross = int(w['z_sim_cross_A'])
            norm_1 = w['norm_qtilde_int'].astype(np.int64)
            norm_2 = w['norm_qtilde_int'].astype(np.int64)

        total = int(np.prod(Q_cross.shape))

        # --- Step 1: table_idx -> Q_dist (integer exact) ---
        # Already verified 100% match in A-gate. Reconfirm.
        M_gauss = int(w[f'M_gauss_{prefix}'])
        r_gauss = w[f'r_gauss_{prefix}']
        k_bits = int(w['k_bits'])

        # Recover: scaled_product = tidx * 2^(2k) + r_gauss
        # X = scaled_product // M_gauss
        # Q_dist_recovered = X >> k_bits
        sp = tidx.astype(object) * (1 << (2 * k_bits))
        if isinstance(r_gauss, np.ndarray) and r_gauss.dtype == object:
            sp = sp + r_gauss
        else:
            sp = sp + r_gauss.astype(object)
        X_rec = sp // M_gauss
        Q_dist_rec = (X_rec >> k_bits).astype(np.int64)

        match_dist = int(np.sum(Q_dist_rec == Q_dist))
        print(f"\n  [{kname}] Step 1: table_idx -> Q_dist")
        print(f"    match={match_dist}/{total}, "
              f"mismatch={total - match_dist}")

        # --- Step 2: Q_dist + norms -> cross (integer domain) ---
        # dist_float = (Q_dist - z_dist) * s_dist
        # dist_raw = dist_float * denom
        # cross_float = (norm_1_deq + norm_2_deq - dist_raw) / 2
        # cross_int_expected = round(cross_float / s_cross + z_cross)
        #
        # This involves float: s_dist, denom, s_q^2, s_cross
        # NOT exact in integer domain.

        # First show the float-domain recovery (what we had before)
        from quantize import get_quantization_params
        d = np.load(debug_path, allow_pickle=True)
        Q_proc_float = d['Q_proc']
        s_q_actual, z_q_actual = get_quantization_params(Q_proc_float)

        dist_deq = (Q_dist.astype(np.float64) - z_dist) * s_dist
        dist_raw = dist_deq * denom

        if kname == "B":
            norm1_deq = norm_1.astype(np.float64) * (s_q_actual ** 2)
            norm2_deq = norm_2.astype(np.float64) * (s_q_actual ** 2)
            cross_recovered_float = (
                norm1_deq[:, :, :, np.newaxis]
                + norm2_deq[:, :, np.newaxis, :]
                - dist_raw
            ) / 2.0
        else:
            norm1_deq = norm_1.astype(np.float64) * (s_q_actual ** 2)
            norm2_deq = norm_2.astype(np.float64) * (s_q_actual ** 2)
            cross_recovered_float = (
                norm1_deq[:, :, :, np.newaxis]
                + norm2_deq[:, :, np.newaxis, :]
                - dist_raw
            ) / 2.0

        if kname == "B":
            s_cross, _ = get_quantization_params(d['cross_B'])
        else:
            s_cross, _ = get_quantization_params(d['cross_A'])

        # Recover Q_cross from float
        Q_cross_from_float = np.floor(
            cross_recovered_float / s_cross + z_cross
        ).astype(np.int64)

        diff_float = np.abs(Q_cross_from_float - Q_cross)
        match_float = int(np.sum(diff_float == 0))
        max_diff_float = int(np.max(diff_float))

        print(f"\n  [{kname}] Step 2: Q_dist + norms -> Q_cross (via float)")
        print(f"    match={match_float}/{total}, "
              f"mismatch={total - match_float}")
        print(f"    max|diff| = {max_diff_float}")

        # Can we do this in pure integer domain?
        # dist_equation: (Q_dist - z_dist) * s_dist * denom
        #   = norm_1 * s_q^2 + norm_2 * s_q^2 - 2 * (Q_cross - z_cross) * s_cross
        #
        # Rearranging for Q_cross:
        #   2 * (Q_cross - z_cross) * s_cross
        #     = norm_1 * s_q^2 + norm_2 * s_q^2 - (Q_dist - z_dist) * s_dist * denom
        #   Q_cross = z_cross + [norm_1*s_q^2 + norm_2*s_q^2 - (Q_dist-z_dist)*s_dist*denom] / (2*s_cross)
        #
        # For exact integer: need s_q^2 / s_cross and s_dist*denom / s_cross to be rational
        # with small denominators. In practice they are irrational floats.
        #
        # Alternative: ZEN-style integer encoding
        #   M_dist * (norm_q + norm_qtilde) - 2 * M_cross * (Q_cross - z_cross) + ...
        #   = Q_dist * 2^k + R_dist
        # This requires designing a combined ZEN identity for the dist equation.

        print(f"\n  [{kname}] Scale ratios (exact integer impossible):")
        print(f"    s_q^2 / (2*s_cross) = {s_q_actual**2 / (2*s_cross):.12e}")
        print(f"    s_dist*denom / (2*s_cross) = {s_dist*denom / (2*s_cross):.12e}")
