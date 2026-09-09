"""Part 2-2: Dist constraint path analysis.

Q1: Scale alignment for norm + cross -> dist.
    Can the dist equation be computed exactly in integer domain?
    Does rescale produce off-by-1?

Q2: Clamping max(dist, 0) needed in quantized domain?
    Check if quantized dist (not float) can ever be negative.
"""
import math
import numpy as np
from quantize import get_quantization_params

out_dir = "outputs_n128_m32_L2_h128_soft_dense"
H = 2
D = 64
S = 128
m = 32
denom = 2.0 * math.sqrt(D)
k_bits = 26

layers = [
    "bert_encoder_layer_0_attention_self",
    "bert_encoder_layer_1_attention_self",
]

for li, lname in enumerate(layers):
    debug_path = f"{out_dir}/{out_dir}-batch0-{lname}.npz"
    witness_masked = debug_path.replace(".npz", ".witness_masked.npz")
    witness_orig = debug_path.replace(".npz", ".witness.npz")
    import os
    witness_path = witness_masked if os.path.exists(witness_masked) else witness_orig
    w = np.load(witness_path, allow_pickle=True)
    d = np.load(debug_path, allow_pickle=True)
    print(f"  witness: {os.path.basename(witness_path)}")

    print(f"\n{'='*60}")
    print(f"LAYER {li}")
    print(f"{'='*60}")

    s_q_actual, z_q_actual = get_quantization_params(d['Q_proc'])

    for kname in ["B", "A"]:
        Q_cross = w[f'Q_sim_cross_{kname}'].astype(np.int64)
        z_cross = int(w[f'z_sim_cross_{kname}'])
        norm_q_int = w['norm_q_int'].astype(np.int64)
        norm_qt_int = w['norm_qtilde_int'].astype(np.int64)
        Q_dist = w[f'Q_dist_{kname}'].astype(np.int64)
        z_dist = int(w[f'z_dist_{kname}'])
        s_dist = float(w[f's_dist_{kname}'])

        if kname == "B":
            s_cross, _ = get_quantization_params(d['cross_B'])
            n1 = norm_q_int   # [B,H,S]
            n2 = norm_qt_int  # [B,H,m]
        else:
            s_cross, _ = get_quantization_params(d['cross_A'])
            n1 = norm_qt_int  # [B,H,m]
            n2 = norm_qt_int  # [B,H,m]

        print(f"\n--- {kname} kernel ---")
        print(f"  s_q^2 = {s_q_actual**2:.12e}")
        print(f"  s_cross = {s_cross:.12e}")
        print(f"  s_dist = {s_dist:.12e}")
        print(f"  denom = {denom:.6f}")

        # Q1: Scale alignment
        # dist_float = [n1*s_q^2 + n2*s_q^2 - 2*(Q_cross-z_cross)*s_cross] / denom
        # Q_dist = floor(dist_float / s_dist + z_dist)
        #
        # In-circuit: we have committed C_B (Q_cross), computed n1/n2 from Q_q_masked.
        # To compute dist in circuit:
        #   Step 1: dequant cross: cross_deq = (Q_cross - z_cross) * s_cross
        #   Step 2: dequant norms: norm_deq = n * s_q^2
        #   Step 3: combine: dist = (norm1_deq + norm2_deq - 2*cross_deq) / denom
        #   Step 4: quantize: Q_dist = floor(dist / s_dist + z_dist)
        #
        # Problem: Steps 1-4 involve float. Circuit needs integer.
        #
        # Integer equivalent using ZEN:
        #   M_dist * [A * n1 + A * n2 - 2 * B * (Q_cross - z_cross) + shift]
        #     = Q_dist * 2^k + R_dist
        #   where A = s_q^2 / (s_dist * denom), B = s_cross / (s_dist * denom)
        #   M_dist includes floor() to make A,B integers.
        #
        # This is the ZEN rescale for dist. Let's see if floor() causes off-by-1.

        A_ratio = s_q_actual**2 / (s_dist * denom)
        B_ratio = s_cross / (s_dist * denom)

        # ZEN multiplier approach:
        # Choose M so that M * A_ratio and M * B_ratio are "close" to integers.
        M_trial = 1 << k_bits
        MA = int(np.floor(M_trial * A_ratio))
        MB = int(np.floor(M_trial * B_ratio))

        print(f"\n  Q1: ZEN-style rescale")
        print(f"    A_ratio = s_q^2/(s_dist*denom) = {A_ratio:.12e}")
        print(f"    B_ratio = s_cross/(s_dist*denom) = {B_ratio:.12e}")
        print(f"    MA = floor(2^{k_bits} * A_ratio) = {MA}")
        print(f"    MB = floor(2^{k_bits} * B_ratio) = {MB}")
        print(f"    MA fractional loss = {M_trial * A_ratio - MA:.6e}")
        print(f"    MB fractional loss = {M_trial * B_ratio - MB:.6e}")

        # Simulate the ZEN dist computation in integer domain
        if kname == "B":
            n1_exp = n1[:, :, :, np.newaxis]  # [B,H,S,1]
            n2_exp = n2[:, :, np.newaxis, :]  # [B,H,1,m]
        else:
            n1_exp = n1[:, :, :, np.newaxis]
            n2_exp = n2[:, :, np.newaxis, :]

        cross_centered = (Q_cross - z_cross).astype(object)

        # ZEN: numerator = MA*(n1+n2) - 2*MB*cross_centered + shift
        # shift chosen so Q_dist = numerator >> k_bits maps to z_dist when dist=0
        # Actually, let's just compute Q_dist_zen and compare.
        numerator = (MA * (n1_exp.astype(object) + n2_exp.astype(object))
                     - 2 * MB * cross_centered)

        # Need to add z_dist * 2^k to shift output to correct range
        # Q_dist_zen = (numerator + z_dist * 2^k) >> k_bits
        # But this isn't quite right. Let me think more carefully.
        #
        # The float equation: Q_dist = floor(dist_float / s_dist + z_dist)
        # dist_float = [n1*s_q^2 + n2*s_q^2 - 2*(Q_cross-z_cross)*s_cross] / denom
        # Q_dist = floor([n1*s_q^2 + n2*s_q^2 - 2*(Q_cross-z_cross)*s_cross] / (s_dist*denom) + z_dist)
        #
        # Integer approximation:
        # Q_dist_approx = (MA*(n1+n2) - 2*MB*(Q_cross-z_cross) + z_dist*(1<<k_bits)) >> k_bits
        shifted_num = numerator + z_dist * (1 << k_bits)
        Q_dist_zen = (shifted_num >> k_bits).astype(np.int64)
        R_dist_zen = (shifted_num % (1 << k_bits)).astype(np.int64)

        diff = Q_dist_zen - Q_dist.astype(np.int64)
        n_exact = int(np.sum(diff == 0))
        n_off1 = int(np.sum(np.abs(diff) == 1))
        n_off2plus = int(np.sum(np.abs(diff) >= 2))
        total = int(np.prod(Q_dist.shape))

        print(f"\n    ZEN dist vs pipeline dist:")
        print(f"    exact: {n_exact}/{total}")
        print(f"    off-by-1: {n_off1}/{total}")
        print(f"    off-by-2+: {n_off2plus}/{total}")
        print(f"    max|diff| = {int(np.max(np.abs(diff)))}")
        if n_off1 > 0 or n_off2plus > 0:
            print(f"    -> ZEN rescale has floor-induced error")

        # Q2: Can quantized dist be negative?
        # Q_dist is quantize_tensor(max(dist_float, 0), s_dist, z_dist)
        # quantize_tensor: floor(x/s + z), clipped to [0, 4095]
        # If dist_float >= 0, then x/s + z >= z >= 0, so Q_dist >= 0.
        # But what about in the ZEN integer path?
        # Q_dist_zen could be negative if numerator < 0.

        # Check if ZEN numerator can go negative (before z_dist shift)
        raw_num = numerator  # without z_dist shift
        n_neg_raw = int(np.sum(raw_num < 0))
        min_raw = int(np.min(raw_num))

        print(f"\n  Q2: Clamping in quantized domain")
        print(f"    Pipeline Q_dist min = {int(np.min(Q_dist))}, z_dist = {z_dist}")
        print(f"    Pipeline Q_dist - z_dist min = {int(np.min(Q_dist)) - z_dist}")
        print(f"    ZEN raw numerator (before z_dist shift) negatives: "
              f"{n_neg_raw}/{total}")
        if n_neg_raw > 0:
            print(f"    ZEN raw numerator min = {min_raw}")
        # After z_dist shift
        n_neg_shifted = int(np.sum(shifted_num < 0))
        min_shifted = int(np.min(shifted_num)) if total > 0 else 0
        print(f"    ZEN shifted numerator negatives: {n_neg_shifted}/{total}")
        if n_neg_shifted > 0:
            print(f"    ZEN shifted numerator min = {min_shifted}")
            print(f"    -> Q_dist_zen could be negative. Clamping needed.")
        else:
            print(f"    -> Q_dist_zen always non-negative. No clamping needed.")

        # Check if diagonal elements of dist_A cause issues
        if kname == "A":
            diag_dist = np.array([Q_dist[0, h, i, i]
                                  for h in range(H) for i in range(m)])
            print(f"    dist_A diagonal: min={int(np.min(diag_dist))}, "
                  f"max={int(np.max(diag_dist))}, z_dist={z_dist}")
            diag_zen = np.array([int(Q_dist_zen[0, h, i, i])
                                 for h in range(H) for i in range(m)])
            print(f"    dist_A_zen diagonal: min={int(np.min(diag_zen))}, "
                  f"max={int(np.max(diag_zen))}")
