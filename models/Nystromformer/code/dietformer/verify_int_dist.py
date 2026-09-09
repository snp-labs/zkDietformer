"""Verify integer-path Q_dist: re-run pipeline, check dist equation identity."""
import os
import math
import numpy as np
from soft_attention_pipeline import process_soft_attn_debug_npz

out_dir = "outputs_n128_m32_L2_h128_soft_dense"
k_bits = 26
TWO_K = 1 << k_bits

layers = [
    "bert_encoder_layer_0_attention_self",
    "bert_encoder_layer_1_attention_self",
]

for li, lname in enumerate(layers):
    debug_path = f"{out_dir}/{out_dir}-batch0-{lname}.npz"
    witness_path = debug_path.replace(".npz", ".witness_intdist.npz")

    print(f"\n{'='*60}")
    print(f"LAYER {li}: {lname}")
    print(f"{'='*60}")

    result = process_soft_attn_debug_npz(debug_path, witness_path,
                                          k_bits=k_bits, n_iter=20)

    # Load and verify the dist equation identity
    w = np.load(witness_path, allow_pickle=True)
    d = np.load(debug_path, allow_pickle=True)

    from quantize import get_quantization_params
    s_q, z_q = get_quantization_params(d['Q_proc'])
    denom = 2.0 * math.sqrt(64)

    for kname in ["B", "A"]:
        Q_dist = w[f'Q_dist_{kname}'].astype(np.int64)
        R_dist = w[f'R_dist_{kname}'].astype(np.int64)
        z_dist = int(w[f'z_dist_{kname}'])
        M_A = int(w[f'M_A_{kname}'])
        M_B = int(w[f'M_B_{kname}'])
        Q_cross = w[f'Q_sim_cross_{kname}'].astype(np.int64)
        z_cross = int(w[f'z_sim_cross_{kname}'])

        if kname == "B":
            n1 = w['norm_q_int'].astype(np.int64)
            n2 = w['norm_qtilde_int'].astype(np.int64)
            n1_exp = n1[:, :, :, np.newaxis].astype(object)
            n2_exp = n2[:, :, np.newaxis, :].astype(object)
        else:
            n1 = w['norm_qtilde_int'].astype(np.int64)
            n2 = w['norm_qtilde_int'].astype(np.int64)
            n1_exp = n1[:, :, :, np.newaxis].astype(object)
            n2_exp = n2[:, :, np.newaxis, :].astype(object)

        # Verify: Q_dist * 2^k + R_dist == M_A*(n1+n2) - 2*M_B*(Q_cross-z_cross) + z_dist*2^k
        LHS = Q_dist.astype(object) * TWO_K + R_dist.astype(object)
        RHS = (M_A * (n1_exp + n2_exp)
               - 2 * M_B * (Q_cross.astype(object) - z_cross)
               + z_dist * TWO_K)

        total = int(np.prod(Q_dist.shape))
        n_mismatch = int(np.sum(LHS != RHS))

        print(f"\n  [{kname}] dist equation identity:")
        print(f"    total: {total}, mismatch: {n_mismatch}")
        print(f"    R_dist range: [{int(np.min(R_dist))}, {int(np.max(R_dist))}]")
        print(f"    Q_dist range: [{int(np.min(Q_dist))}, {int(np.max(Q_dist))}]")
        if n_mismatch > 0:
            print(f"    FAIL: {n_mismatch} elements do not satisfy the equation")

    # Gaussian lookup consistency
    n_mis_B = int(np.sum(w['kernel_B'] != w['gauss_table'][w['table_idx_B'].astype(int)]))
    n_mis_A = int(np.sum(w['kernel_A'] != w['gauss_table'][w['table_idx_A'].astype(int)]))
    print(f"\n  Gaussian consistency: B={n_mis_B}, A={n_mis_A}")

    # Masking check
    mask = w['mask']
    pad_pos = np.where(mask[0] == 0)[0]
    q_masked = w['Q_q_masked']
    v_masked = w['Q_v_masked']
    z_q_val = int(w['z_q'])
    z_v_val = int(w['z_v'])
    q_match = int(np.sum(q_masked[0, :, pad_pos, :] == z_q_val))
    q_total = q_masked[0, :, pad_pos, :].size
    v_match = int(np.sum(v_masked[0, :, pad_pos, :] == z_v_val))
    v_total = v_masked[0, :, pad_pos, :].size
    print(f"  Masking: Q_q_masked padding={q_match}/{q_total}, "
          f"Q_v_masked padding={v_match}/{v_total}")
