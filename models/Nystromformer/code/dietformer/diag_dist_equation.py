"""Part 2-2 Q1 re-confirmation: dist equation integer identity.

Verifies that the quotient-remainder equation holds EXACTLY:
  Q_dist * 2^k + R_dist = M_A * (n1 + n2) - 2 * M_B * (Q_cross - z_cross) + z_dist * 2^k

where Q_dist = LHS // 2^k, R_dist = LHS % 2^k (Euclidean division).
"""
import os
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
TWO_K = 1 << k_bits

layers = [
    "bert_encoder_layer_0_attention_self",
    "bert_encoder_layer_1_attention_self",
]

for li, lname in enumerate(layers):
    debug_path = f"{out_dir}/{out_dir}-batch0-{lname}.npz"
    witness_masked = debug_path.replace(".npz", ".witness_masked.npz")
    witness_orig = debug_path.replace(".npz", ".witness.npz")
    witness_path = witness_masked if os.path.exists(witness_masked) else witness_orig
    w = np.load(witness_path, allow_pickle=True)
    d = np.load(debug_path, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"LAYER {li}")
    print(f"{'='*60}")

    s_q_actual, z_q_actual = get_quantization_params(d['Q_proc'])

    for kname in ["B", "A"]:
        Q_cross = w[f'Q_sim_cross_{kname}'].astype(np.int64)
        z_cross = int(w[f'z_sim_cross_{kname}'])
        norm_q_int = w['norm_q_int'].astype(np.int64)
        norm_qt_int = w['norm_qtilde_int'].astype(np.int64)
        Q_dist_pipeline = w[f'Q_dist_{kname}'].astype(np.int64)
        z_dist = int(w[f'z_dist_{kname}'])
        s_dist = float(w[f's_dist_{kname}'])

        if kname == "B":
            s_cross, _ = get_quantization_params(d['cross_B'])
            n1 = norm_q_int    # [B,H,S]
            n2 = norm_qt_int   # [B,H,m]
        else:
            s_cross, _ = get_quantization_params(d['cross_A'])
            n1 = norm_qt_int   # [B,H,m]
            n2 = norm_qt_int   # [B,H,m]

        # Compute M_A, M_B (ZEN multipliers)
        A_ratio = s_q_actual**2 / (s_dist * denom)
        B_ratio = s_cross / (s_dist * denom)
        MA = int(np.floor(TWO_K * A_ratio))
        MB = int(np.floor(TWO_K * B_ratio))

        print(f"\n--- {kname} kernel ---")
        print(f"  MA = {MA}, MB = {MB}")
        print(f"  z_cross = {z_cross}, z_dist = {z_dist}")

        # Expand norms for broadcasting
        if kname == "B":
            n1_exp = n1[:, :, :, np.newaxis].astype(object)  # [B,H,S,1]
            n2_exp = n2[:, :, np.newaxis, :].astype(object)  # [B,H,1,m]
        else:
            n1_exp = n1[:, :, :, np.newaxis].astype(object)  # [B,H,m,1]
            n2_exp = n2[:, :, np.newaxis, :].astype(object)  # [B,H,1,m]

        cross_centered = (Q_cross.astype(object) - z_cross)

        # Compute LHS of the equation
        LHS = (MA * (n1_exp + n2_exp)
               - 2 * MB * cross_centered
               + z_dist * TWO_K)

        # Euclidean division
        Q_dist_circuit = (LHS // TWO_K).astype(np.int64)
        R_dist = (LHS % TWO_K).astype(np.int64)

        # === Assert 1: LHS >= 0 (non-negative for valid Euclidean division) ===
        n_neg = int(np.sum(LHS < 0))
        total = int(np.prod(Q_cross.shape))
        print(f"\n  Assert 1: LHS >= 0")
        print(f"    negatives: {n_neg}/{total}")
        if n_neg > 0:
            print(f"    FAIL: LHS has negative values. min = {int(np.min(LHS))}")
            print(f"    STOP: equation form invalid for unsigned Euclidean division")
            continue

        # === Assert 2: identity Q_dist * 2^k + R_dist == LHS ===
        recon = Q_dist_circuit.astype(object) * TWO_K + R_dist.astype(object)
        identity_fail = int(np.sum(recon != LHS))
        print(f"  Assert 2: Q_dist * 2^k + R_dist == LHS")
        print(f"    failures: {identity_fail}/{total}")

        # === Assert 3: R_dist in [0, 2^k) ===
        r_min = int(np.min(R_dist))
        r_max = int(np.max(R_dist))
        r_oob = int(np.sum((R_dist < 0) | (R_dist >= TWO_K)))
        r_bits_needed = int(np.ceil(np.log2(max(r_max, 1) + 1))) if r_max > 0 else 0
        print(f"  Assert 3: 0 <= R_dist < 2^{k_bits}")
        print(f"    range: [{r_min}, {r_max}]")
        print(f"    out-of-bounds: {r_oob}/{total}")
        print(f"    bits needed: {r_bits_needed}")

        # === Assert 4: Q_dist_circuit in valid range ===
        qd_min = int(np.min(Q_dist_circuit))
        qd_max = int(np.max(Q_dist_circuit))
        print(f"  Assert 4: Q_dist_circuit range")
        print(f"    range: [{qd_min}, {qd_max}]")
        print(f"    12-bit valid: {qd_min >= 0 and qd_max <= 4095}")

        # === Compare with pipeline Q_dist ===
        diff = Q_dist_circuit - Q_dist_pipeline
        n_exact = int(np.sum(diff == 0))
        n_off1 = int(np.sum(np.abs(diff) == 1))
        n_off2plus = int(np.sum(np.abs(diff) >= 2))
        print(f"\n  Pipeline Q_dist vs circuit Q_dist:")
        print(f"    exact: {n_exact}/{total}")
        print(f"    off-by-1: {n_off1}/{total}")
        print(f"    off-by-2+: {n_off2plus}/{total}")
        if n_off2plus > 0:
            print(f"    FAIL: off-by-2+ exists. max|diff| = {int(np.max(np.abs(diff)))}")

        # === Constraint cost ===
        if kname == "B":
            n_elements = S * m  # per batch per head
        else:
            n_elements = m * m
        # 1 linear constraint (enforce_equal) per element
        # 1 split_remainder_and_push per element (2 vars: low, high)
        n_constraints_eq = n_elements  # linear equality
        n_constraints_rng = n_elements  # remainder split
        print(f"\n  Constraint cost (per b*h):")
        print(f"    dist equation: {n_elements} linear constraints")
        print(f"    R_dist range check: {n_elements} split_remainder calls")
        print(f"    R_dist d1 commitment: {2 * n_elements} limb vars (low+high)")

    # === Commit classification summary ===
    print(f"\n--- Commit classification (Layer {li}) ---")
    print(f"  C_B, C_A: aux (Freivalds verification)")
    print(f"  norm_q, norm_qtilde: omega (R1CS square constraints)")
    print(f"  Q_dist_B, Q_dist_A: omega (uniquely determined by dist equation)")
    print(f"  R_dist_B, R_dist_A: d1 (Lasso range check)")
    print(f"  table_idx, r_gauss: d1 (Lasso table lookup)")
    print(f"  kernel_B, kernel_A: omega (table lookup recovery)")
