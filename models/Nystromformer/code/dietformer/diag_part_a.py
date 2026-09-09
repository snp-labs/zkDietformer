"""Part A diagnostics: masking mismatch, negative dist, table usage."""
import os
import math
import numpy as np

out_dir = "outputs_n128_m32_L2_h128_soft_dense"

H = 2
D = 64
S = 128
m = 32
seg = S // m
denom = 2.0 * math.sqrt(D)

layers = [
    "bert_encoder_layer_0_attention_self",
    "bert_encoder_layer_1_attention_self",
]

for li, lname in enumerate(layers):
    debug_path = os.path.join(out_dir, f"{out_dir}-batch0-{lname}.npz")
    witness_path = debug_path.replace(".npz", ".witness.npz")
    d = np.load(debug_path, allow_pickle=True)
    w = np.load(witness_path, allow_pickle=True)

    mask = d['mask']  # [B, S]
    pad_pos = np.where(mask[0] == 0)[0]
    n_pad = len(pad_pos)

    print(f"\n{'='*60}")
    print(f"LAYER {li}: {lname}")
    print(f"{'='*60}")

    # === A-1: Masking quantization mismatch ===
    print(f"\n--- A-1: Masking quantization mismatch ---")
    Q_q = w['Q_q']       # [B, H, S, D]
    z_q = int(w['z_q'])
    Q_v = w['Q_v']       # [B, H, S, D]
    z_v = int(w['z_v'])

    # Q_q at padding
    q_pad = Q_q[0, :, pad_pos, :]  # [H, n_pad, D]
    q_at_zq = int(np.sum(q_pad == z_q))
    q_total = q_pad.size
    q_unique = np.unique(q_pad)
    print(f"  Q_q at padding: {q_at_zq}/{q_total} == z_q({z_q})")
    print(f"  Q_q padding unique values: {len(q_unique)} distinct")
    print(f"  Q_q padding min={int(np.min(q_pad))}, "
          f"max={int(np.max(q_pad))}, mean={np.mean(q_pad):.1f}")

    # V at padding
    v_pad = Q_v[0, :, pad_pos, :]
    v_at_zv = int(np.sum(v_pad == z_v))
    v_total = v_pad.size
    v_unique = np.unique(v_pad)
    print(f"  Q_v at padding: {v_at_zv}/{v_total} == z_v({z_v})")
    print(f"  Q_v padding unique values: {len(v_unique)} distinct")
    print(f"  Q_v padding min={int(np.min(v_pad))}, "
          f"max={int(np.max(v_pad))}, mean={np.mean(v_pad):.1f}")

    # Compare with what quantize_tensor(0.0, s, z) gives
    from quantize import get_quantization_params, quantize_tensor
    Q_proc_float = d['Q_proc']  # [B, H, S, D]
    s_proc, z_proc = get_quantization_params(Q_proc_float)
    q_of_zero = quantize_tensor(np.array([0.0]), s_proc, z_proc)
    print(f"  quantize_tensor(0.0, s_proc={s_proc:.6e}, z_proc={z_proc}) = {int(q_of_zero[0])}")
    print(f"  z_q stored in witness = {z_q}")
    print(f"  Match: {'YES' if int(q_of_zero[0]) == z_q else 'NO'}")

    # Check if Q_proc float at padding is really zero
    q_proc_pad = Q_proc_float[0, :, pad_pos, :]
    print(f"  Q_proc float at padding: max|val|={float(np.max(np.abs(q_proc_pad))):.6e}")

    # Root cause: pipeline computes Q_q from projection chain, not from Q_proc
    # Show a sample padding element
    if n_pad > 0:
        p0 = pad_pos[0]
        print(f"\n  Sample padding position {p0}, head 0, dim 0:")
        print(f"    Q_proc float = {float(Q_proc_float[0, 0, p0, 0]):.6e}")
        print(f"    Q_q value    = {int(Q_q[0, 0, p0, 0])}")
        print(f"    Expected z_q = {z_q}")
        print(f"    Diff         = {int(Q_q[0, 0, p0, 0]) - z_q}")

    # === A-2: Negative distance guard ===
    print(f"\n--- A-2: Negative distance guard ---")

    # Float domain: pre-clamp dist
    cross_B_deq = (w['Q_sim_cross_B'].astype(np.float64) - int(w['z_sim_cross_B'])) * float(w.get('s_cross_B', d.get('s_cross_B', 0)))
    # Actually, we need to recompute. Let me use the raw float dist from capture.
    dist_B_raw = d['dist_B_raw'].astype(np.float64)  # pre-denom
    dist_A_raw = d['dist_A_raw'].astype(np.float64)

    dist_B_scaled = dist_B_raw / denom
    dist_A_scaled = dist_A_raw / denom

    # Float domain negatives
    n_neg_B = int(np.sum(dist_B_scaled < 0))
    n_neg_A = int(np.sum(dist_A_scaled < 0))
    min_B = float(np.min(dist_B_scaled))
    min_A = float(np.min(dist_A_scaled))

    print(f"  Float domain (pre-clamp):")
    print(f"    dist_B: min={min_B:.12e}, negatives={n_neg_B}/{dist_B_scaled.size}")
    print(f"    dist_A: min={min_A:.12e}, negatives={n_neg_A}/{dist_A_scaled.size}")

    # Where are the negatives?
    if n_neg_A > 0:
        neg_mask_A = dist_A_scaled[0] < 0  # [H, m, m]
        for hi in range(H):
            neg_indices = np.argwhere(neg_mask_A[hi])
            if len(neg_indices) > 0:
                print(f"    dist_A head {hi}: {len(neg_indices)} negative elements")
                for idx in neg_indices[:5]:
                    i, j = idx
                    is_diag = "DIAG" if i == j else f"off-diag"
                    val = float(dist_A_scaled[0, hi, i, j])
                    cross_val = float(d['cross_A'][0, hi, i, j])
                    norm_i = float(d['norm_qtilde'][0, hi, i])
                    norm_j = float(d['norm_qtilde'][0, hi, j])
                    print(f"      [{i},{j}] {is_diag}: dist={val:.12e}, "
                          f"norm_i={norm_i:.6f}, norm_j={norm_j:.6f}, "
                          f"cross={cross_val:.6f}")

    if n_neg_B > 0:
        neg_mask_B = dist_B_scaled[0] < 0
        for hi in range(H):
            neg_indices = np.argwhere(neg_mask_B[hi])
            if len(neg_indices) > 0:
                print(f"    dist_B head {hi}: {len(neg_indices)} negative elements")
                for idx in neg_indices[:5]:
                    i, j = idx
                    val = float(dist_B_scaled[0, hi, i, j])
                    print(f"      [{i},{j}]: dist={val:.12e}")

    # Quantized integer domain dist (post-clamp, so always >= 0)
    Q_dist_B = w['Q_dist_B'].astype(np.int64)
    Q_dist_A = w['Q_dist_A'].astype(np.int64)
    z_dist_B = int(w['z_dist_B'])
    z_dist_A = int(w['z_dist_A'])
    print(f"\n  Quantized domain (post-clamp):")
    print(f"    Q_dist_B: min={int(np.min(Q_dist_B))}, z_dist_B={z_dist_B}, "
          f"min-z={int(np.min(Q_dist_B))-z_dist_B}")
    print(f"    Q_dist_A: min={int(np.min(Q_dist_A))}, z_dist_A={z_dist_A}, "
          f"min-z={int(np.min(Q_dist_A))-z_dist_A}")

    # What would the integer-domain dist look like WITHOUT float dequant?
    # norm_q_int * s_q^2 + norm_qt_int * s_q^2 - 2 * (Q_cross - z_cross) * s_cross
    # = s_q^2 * (norm_q + norm_qt) - 2 * s_cross * (Q_cross - z_cross)
    # These have different scales, so direct integer comparison is not meaningful.
    # The circuit must handle the scale mismatch.
    norm_qt_int = w['norm_qtilde_int'].astype(np.int64)  # [B, H, m]
    Q_cross_A = w['Q_sim_cross_A'].astype(np.int64)  # [B, H, m, m]
    z_ca = int(w['z_sim_cross_A'])

    # Show that diagonal of cross_A in integer domain is close to norm_qtilde
    # cross_A[i][i] should approximate norm_qtilde[i] (since Q~ @ Q~^T diagonal = ||q~_i||^2)
    print(f"\n  Cross-A diagonal vs norm_qtilde (integer domain):")
    for hi in range(H):
        cross_diag = np.diag(Q_cross_A[0, hi])  # [m]
        norms = norm_qt_int[0, hi]  # [m]
        # cross is in ZEN-quantized domain, norm is sum of (Q-z)^2
        # These are NOT directly comparable (different scales)
        # But the float-domain relationship: cross_A[i][i] = ||q~_i||^2
        cross_A_diag_float = np.diag(d['cross_A'][0, hi])
        norm_float = d['norm_qtilde'][0, hi]
        diff = np.abs(cross_A_diag_float - norm_float)
        max_diff = float(np.max(diff))
        print(f"    Head {hi}: float max|cross_diag - norm| = {max_diff:.6e}")

    # === A-3: Table usage ===
    print(f"\n--- A-3: Table usage ---")
    tidx_B = w['table_idx_B'].astype(np.int64)
    tidx_A = w['table_idx_A'].astype(np.int64)
    unique_B = len(np.unique(tidx_B))
    unique_A = len(np.unique(tidx_A))
    print(f"  table_idx_B: unique={unique_B}/65536 "
          f"({unique_B/65536*100:.4f}%)")
    print(f"  table_idx_A: unique={unique_A}/65536 "
          f"({unique_A/65536*100:.4f}%)")
    print(f"  table_idx_B range: [{int(np.min(tidx_B))}, {int(np.max(tidx_B))}]")
    print(f"  table_idx_A range: [{int(np.min(tidx_A))}, {int(np.max(tidx_A))}]")
