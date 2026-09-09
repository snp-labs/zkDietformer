"""
Linformer attention witness pipeline (Stage A4-A8).

Generates quantized witness NPZ from raw debug captures.
Follows bert_attention_pipeline.py pattern with Linformer-specific matmuls.

NPZ schema: docs/linformer_dev_plan.md Section A-3.
"""
import math
import os
import numpy as np

from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul,
    ensure_nonneg_matmul,
)
from softmax_table import (
    FRAC_BITS, TABLE_FRAC_BITS,
    generate_fractional_exp_table, softmax_v3_lookup, save_softmax_tables_v3,
)


def _load_optional_array(debug, key):
    """Safely load optional array from NPZ (may be None)."""
    if key not in debug:
        return None
    val = debug[key]
    if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 0:
        item = val.item()
        return None if item is None else val
    return val


def _projection_witness(hidden_states, W, b, s_x_input, z_x_input,
                         Q_x_input, B, H, S, D, HD, k_bits,
                         apply_scale, ref_proc, name):
    """Quantize a single Q/K/V projection (BERT-identical logic)."""
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
        z_raw, M_proj, k_bits,
    )

    if b is not None:
        s_b = s_w * s_x_input
        Q_b_diff = np.round(b / s_b).astype(np.int64)
        num = Q_raw_flat.astype(np.int64) * (1 << k_bits) + R_flat.astype(np.int64)
        num = num + M_proj * Q_b_diff
        Q_raw_flat = (num >> k_bits).astype(np.int64)
        R_flat = (num % (1 << k_bits)).astype(np.int64)
    else:
        Q_b_diff = np.zeros(HD, dtype=np.int64)

    Q_raw_4d = Q_raw_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    R_proj_4d = R_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    Q_w_diff = np.array([
        (Q_w[h * D:(h + 1) * D, :] - z_w).astype(np.int64) for h in range(H)
    ])
    Q_b_diff_per_head = Q_b_diff.reshape(H, D)

    wit = {
        f"Q_raw_proj_{name}": Q_raw_4d.astype(np.int64),
        f"R_proj_{name}": R_proj_4d.astype(np.int64),
        f"Q_w_diff_{name}": Q_w_diff.astype(np.int64),
        f"Q_b_diff_{name}": Q_b_diff_per_head.astype(np.int64),
        f"z_w_{name}": np.uint64(z_w),
        f"M_proj_{name}": np.uint64(M_proj),
        f"M_prime_proj_{name}": np.uint64(M_prime_proj),
        f"z_raw_{name}": np.uint64(z_raw),
    }

    scale_info = None
    if apply_scale:
        scale_factor = math.sqrt(math.sqrt(D))
        s_proc, z_proc = get_quantization_params(ref_proc)
        M_scale = int(np.floor((1 << k_bits) * s_raw / (scale_factor * s_proc)))
        if M_scale == 0:
            M_scale = 1
        M_prime_scale = int(np.floor((z_proc * (1 << k_bits)) / M_scale))

        inner = (Q_raw_4d - z_raw).astype(np.int64)
        num_scale = M_scale * (inner + M_prime_scale)
        Q_scaled = (num_scale >> k_bits).astype(np.int64)
        R_scale = (num_scale % (1 << k_bits)).astype(np.int64)

        wit[f"Q_scaled_{name}"] = Q_scaled.astype(np.int64)
        wit[f"R_scale_{name}"] = R_scale.astype(np.int64)
        wit[f"M_scale_{name}"] = np.uint64(M_scale)
        wit[f"M_prime_scale_{name}"] = np.uint64(M_prime_scale)
        scale_info = (Q_scaled, s_proc, z_proc, s_raw, z_raw)
    else:
        scale_info = (Q_raw_4d, s_raw, z_raw, s_raw, z_raw)

    return wit, scale_info


def _linformer_matmul(Q_w, z_w, Q_x, z_x, s_w, s_x, ref_real, k_bits):
    """ZEN matmul for E_i@K or F_i@V: derive output scale, then compute."""
    s_y, z_y = get_quantization_params(ref_real)
    M = get_m_constant(s_w, s_x, s_y, k=k_bits)
    Q_sim, R, M_prime = sign_bit_grouping_matmul(
        Q_w, z_w, Q_x, z_x, z_y, M, k_bits,
    )
    return Q_sim, R, M_prime, M, s_y, z_y


def process_linformer_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=26):
    """Generate quantized witness NPZ from raw Linformer debug capture."""
    save_softmax_tables_v3(os.path.dirname(out_npz_path))

    debug = np.load(debug_npz_path, allow_pickle=True)

    hidden_states = debug['hidden_states']
    W_q, W_k, W_v = debug['W_q'], debug['W_k'], debug['W_v']
    b_q = _load_optional_array(debug, 'b_q')
    b_k = _load_optional_array(debug, 'b_k')
    b_v = _load_optional_array(debug, 'b_v')
    W_o = debug['W_o']
    b_o = _load_optional_array(debug, 'b_o')
    E_raw = debug['E']           # [H, k, n] float
    F_raw = debug['F_proj']      # [H, k, n] float
    mask = debug['mask']          # [B, S]

    Q_proc_real = debug['Q_proc']
    B, H, S, D = Q_proc_real.shape
    HD = H * D
    k_proj = E_raw.shape[1]  # projection dim k

    # == Phase 0: Linear Projection Witness (BERT-identical) ==
    s_x_input, z_x_input = get_quantization_params(hidden_states)
    Q_x_input = quantize_tensor(hidden_states, s_x_input, z_x_input)

    proj_witness = {
        "Q_x_input": Q_x_input.astype(np.int64),
        "z_x_input": np.uint64(z_x_input),
    }

    # Collect chain outputs for Linformer matmuls.
    # The circuit uses scale_k_out/scale_q_out/proj_v_out as matmul inputs,
    # so the pipeline must use the same values (not fresh quantization).
    chain_outputs = {}
    for name, W, b, apply_scale in [("q", W_q, b_q, True),
                                     ("k", W_k, b_k, True),
                                     ("v", W_v, b_v, False)]:
        ref = debug['Q_proc'] if name == "q" else (
            debug['K_proc'] if name == "k" else None
        )
        wit, scale_info = _projection_witness(
            hidden_states, W, b, s_x_input, z_x_input, Q_x_input,
            B, H, S, D, HD, k_bits, apply_scale, ref, name,
        )
        proj_witness.update(wit)
        # scale_info: (Q_out, s_out, z_out, s_raw, z_raw)
        chain_outputs[name] = scale_info

    # Use projection chain outputs as Linformer matmul inputs.
    # This matches the circuit which uses scale_k_out, scale_q_out, proj_v_out.
    Q_k = chain_outputs["k"][0]   # Q_scaled_k [B,H,S,D]
    s_k, z_k = chain_outputs["k"][1], chain_outputs["k"][2]
    Q_q = chain_outputs["q"][0]   # Q_scaled_q [B,H,S,D]
    s_q, z_q = chain_outputs["q"][1], chain_outputs["q"][2]
    Q_v = chain_outputs["v"][0]   # Q_raw_proj_v [B,H,S,D]
    s_v, z_v = chain_outputs["v"][1], chain_outputs["v"][2]

    # Apply mask to quantized K, Q, V at padding positions (set to zero-point).
    # Linformer F@V mixes all positions without softmax gating, so padding V
    # must be masked. K and Q are also masked for consistency.
    # Circuit constrains this via conditional_select after enforce_scaling.
    mask_bhsd = mask[:, np.newaxis, :, np.newaxis]  # [B, 1, S, 1] broadcast
    Q_k = np.where(mask_bhsd == 0, z_k, Q_k)
    Q_q = np.where(mask_bhsd == 0, z_q, Q_q)
    Q_v = np.where(mask_bhsd == 0, z_v, Q_v)

    # == Phase 1: Linformer Projections (E_i @ K, F_i @ V) ==
    # Quantize E_i and F_i (model weights, 12-bit asymmetric)
    s_e, z_e = get_quantization_params(E_raw)
    Q_e = quantize_tensor(E_raw, s_e, z_e)
    s_f, z_f = get_quantization_params(F_raw)
    Q_f = quantize_tensor(F_raw, s_f, z_f)

    # Q_w_diff: [H, k, n] - z_w
    Q_w_diff_e = (Q_e - z_e).astype(np.int64)  # [H, k, n]
    Q_w_diff_f = (Q_f - z_f).astype(np.int64)  # [H, k, n]

    # E_i @ K_proc per head: E_i [H, k, n] @ K_proc [B, H, n, D] -> [B, H, k, D]
    EK_real = debug['EK']  # float reference [B, H, k, D]

    Q_sim_ek_all = np.zeros((B, H, k_proj, D), dtype=np.int64)
    R_matmul_ek_all = np.zeros((B, H, k_proj, D), dtype=np.int64)

    # Per-head matmul: E_i[h] [k, n] @ K_proc[b, h] [n, D]
    s_ek, z_ek = get_quantization_params(EK_real)
    M_ek = get_m_constant(s_e, s_k, s_ek, k=k_bits)

    for b_idx in range(B):
        for h_idx in range(H):
            Q_sim_h, R_h, M_prime_ek = sign_bit_grouping_matmul(
                Q_e[h_idx].astype(np.int64), z_e,
                Q_k[b_idx, h_idx].astype(np.int64), z_k,
                z_ek, M_ek, k_bits,
            )
            # Q_sim_h: [k, D], R_h: [k, D]
            Q_sim_ek_all[b_idx, h_idx] = Q_sim_h.astype(np.int64)
            R_matmul_ek_all[b_idx, h_idx] = R_h.astype(np.int64)

    # F_i @ V per head: F_i [H, k, n] @ V [B, H, n, D] -> [B, H, k, D]
    FV_real = debug['FV']  # float reference [B, H, k, D]
    s_fv, z_fv = get_quantization_params(FV_real)
    M_fv = get_m_constant(s_f, s_v, s_fv, k=k_bits)

    Q_sim_fv_all = np.zeros((B, H, k_proj, D), dtype=np.int64)
    R_matmul_fv_all = np.zeros((B, H, k_proj, D), dtype=np.int64)

    for b_idx in range(B):
        for h_idx in range(H):
            Q_sim_h, R_h, M_prime_fv = sign_bit_grouping_matmul(
                Q_f[h_idx].astype(np.int64), z_f,
                Q_v[b_idx, h_idx].astype(np.int64), z_v,
                z_fv, M_fv, k_bits,
            )
            Q_sim_fv_all[b_idx, h_idx] = Q_sim_h.astype(np.int64)
            R_matmul_fv_all[b_idx, h_idx] = R_h.astype(np.int64)

    # == Phase 2: Attention Logit (Q_proc @ EK^T) ==
    # Q_proc [B, H, S, D] @ EK^T [B, H, D, k] -> [B, H, S, k]
    logit_real = debug['logit']  # float reference
    s_logit, z_logit = get_quantization_params(logit_real)

    # ensure_nonneg_matmul: logit feeds softmax, so Q_sim must be >= 0
    # Need per-(b,h) matmul: Q_proc[b,h] [S,D] @ EK[b,h]^T [D,k] -> [S,k]
    # Use global quantization params, but ensure non-negative globally.
    M_logit_val = get_m_constant(s_q, s_ek, s_logit, k=k_bits)

    # First pass: check if any Q_sim < 0
    Q_sim_logit_all = np.zeros((B, H, S, k_proj), dtype=np.int64)
    R_matmul_logit_all = np.zeros((B, H, S, k_proj), dtype=np.int64)

    for b_idx in range(B):
        for h_idx in range(H):
            ek_t = Q_sim_ek_all[b_idx, h_idx].T.astype(np.int64)  # [D, k]
            Q_sim_h, R_h, M_prime_logit = sign_bit_grouping_matmul(
                Q_q[b_idx, h_idx].astype(np.int64), z_q,
                ek_t, z_ek,
                z_logit, M_logit_val, k_bits,
            )
            Q_sim_logit_all[b_idx, h_idx] = Q_sim_h.astype(np.int64)
            R_matmul_logit_all[b_idx, h_idx] = R_h.astype(np.int64)

    # Adjust z_logit if any Q_sim < 0 (required for softmax table indexing)
    min_logit = int(np.min(Q_sim_logit_all))
    if min_logit < 0:
        z_logit += (-min_logit + 1)
        for b_idx in range(B):
            for h_idx in range(H):
                ek_t = Q_sim_ek_all[b_idx, h_idx].T.astype(np.int64)
                Q_sim_h, R_h, M_prime_logit = sign_bit_grouping_matmul(
                    Q_q[b_idx, h_idx].astype(np.int64), z_q,
                    ek_t, z_ek,
                    z_logit, M_logit_val, k_bits,
                )
                Q_sim_logit_all[b_idx, h_idx] = Q_sim_h.astype(np.int64)
                R_matmul_logit_all[b_idx, h_idx] = R_h.astype(np.int64)

        # Safety: check again
        while np.min(Q_sim_logit_all) < 0:
            z_logit += 1
            for b_idx in range(B):
                for h_idx in range(H):
                    ek_t = Q_sim_ek_all[b_idx, h_idx].T.astype(np.int64)
                    Q_sim_h, R_h, M_prime_logit = sign_bit_grouping_matmul(
                        Q_q[b_idx, h_idx].astype(np.int64), z_q,
                        ek_t, z_ek,
                        z_logit, M_logit_val, k_bits,
                    )
                    Q_sim_logit_all[b_idx, h_idx] = Q_sim_h.astype(np.int64)
                    R_matmul_logit_all[b_idx, h_idx] = R_h.astype(np.int64)

        print(f"  [INFO] z_logit adjusted to {z_logit} for non-negative Q_sim_logit")

    # == Phase 3: Softmax (single table, v3) ==
    exp_table = generate_fractional_exp_table(TABLE_FRAC_BITS, FRAC_BITS)

    # No mask for softmax: all k columns are valid (K_proc is pre-masked)
    P_lin, R_div_lin, exp_total, exp_sum, M_sm, table_idx, r_sm = \
        softmax_v3_lookup(
            Q_sim_logit_all, R_matmul_logit_all, s_logit, k_bits,
            FRAC_BITS, TABLE_FRAC_BITS, exp_table, mask=None,
        )

    # == Phase 4: Output Matmul (P @ FV) ==
    X_real = debug['X']  # float reference [B, H, S, D]
    s_x, z_sim_x = get_quantization_params(X_real)

    z_p = 0  # softmax output zero-point
    s_p = 1.0 / (1 << FRAC_BITS)

    M_final = get_m_constant(s_p, s_fv, s_x, k=k_bits)

    Q_sim_x_all = np.zeros((B, H, S, D), dtype=np.int64)
    R_matmul_x_all = np.zeros((B, H, S, D), dtype=np.int64)

    for b_idx in range(B):
        for h_idx in range(H):
            Q_sim_h, R_h, M_prime_x = sign_bit_grouping_matmul(
                P_lin[b_idx, h_idx].astype(np.int64), z_p,
                Q_sim_fv_all[b_idx, h_idx].astype(np.int64), z_fv,
                z_sim_x, M_final, k_bits,
            )
            Q_sim_x_all[b_idx, h_idx] = Q_sim_h.astype(np.int64)
            R_matmul_x_all[b_idx, h_idx] = R_h.astype(np.int64)

    # == Phase 5: Output Projection (BERT-identical) ==
    Q_sim_x_flat = Q_sim_x_all.transpose(0, 2, 1, 3).reshape(B, S, HD)

    s_w_o, z_w_o = get_quantization_params(W_o)
    Q_w_o = quantize_tensor(W_o, s_w_o, z_w_o)

    attn_output_for_proj = debug['attn_output_for_proj']
    o_real = np.matmul(attn_output_for_proj, W_o.T)
    if b_o is not None:
        o_real = o_real + b_o
    o_real_4d = o_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    s_raw_o, z_raw_o = get_quantization_params(o_real_4d)

    M_proj_o = get_m_constant(s_x, s_w_o, s_raw_o, k=k_bits)

    Q_o_flat, R_o_flat, M_prime_proj_o = sign_bit_grouping_matmul(
        Q_sim_x_flat.astype(np.int64), z_sim_x,
        Q_w_o.T.astype(np.int64), z_w_o,
        z_raw_o, M_proj_o, k_bits,
    )

    if b_o is not None:
        s_b_o = s_w_o * s_x
        Q_b_diff_o = np.round(b_o / s_b_o).astype(np.int64)
        num_o = Q_o_flat.astype(np.int64) * (1 << k_bits) + R_o_flat.astype(np.int64)
        num_o = num_o + M_proj_o * Q_b_diff_o
        Q_o_flat = (num_o >> k_bits).astype(np.int64)
        R_o_flat = (num_o % (1 << k_bits)).astype(np.int64)
    else:
        Q_b_diff_o = np.zeros(HD, dtype=np.int64)

    Q_o_4d = Q_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    R_o_4d = R_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

    # Quantization error measurement
    o_approx = (Q_o_4d.astype(np.float64) - z_raw_o) * s_raw_o
    o_ref = o_real_4d.astype(np.float64)
    _diff = o_ref - o_approx
    _max_abs = float(np.max(np.abs(_diff)))
    _cos_sim = float(np.sum(o_ref * o_approx) /
                     (np.linalg.norm(o_ref) * np.linalg.norm(o_approx) + 1e-12))
    print(f"  [Linformer O-proj] MaxAbsErr={_max_abs:.6e}  CosSim={_cos_sim:.8f}")

    Q_w_diff_o_per_head = np.array([
        (Q_w_o[h * D:(h + 1) * D, :] - z_w_o).astype(np.int64) for h in range(H)
    ])
    Q_b_diff_o_per_head = Q_b_diff_o.reshape(H, D)

    proj_witness.update({
        "Q_attn_output": Q_sim_x_flat.astype(np.int64),
        "z_attn_output": np.uint64(z_sim_x),
        "Q_raw_proj_o": Q_o_4d.astype(np.int64),
        "R_proj_o": R_o_4d.astype(np.int64),
        "Q_w_diff_o": Q_w_diff_o_per_head.astype(np.int64),
        "Q_b_diff_o": Q_b_diff_o_per_head.astype(np.int64),
        "z_w_o": np.uint64(z_w_o),
        "M_proj_o": np.uint64(M_proj_o),
        "M_prime_proj_o": np.uint64(M_prime_proj_o),
        "z_raw_o": np.uint64(z_raw_o),
        "s_raw_o": np.float64(s_raw_o),
    })

    # == NPZ Save (A-3 schema) ==
    np.savez_compressed(
        out_npz_path,
        # Projection witness (BERT-common)
        **proj_witness,
        Q_q=Q_q, z_q=np.uint64(z_q),
        Q_k=Q_k, z_k=np.uint64(z_k),
        Q_v=Q_v, z_v=np.uint64(z_v),
        # Linformer projection matrices
        Q_w_diff_e=Q_w_diff_e.astype(np.int64),
        z_w_e=np.uint64(z_e),
        Q_w_diff_f=Q_w_diff_f.astype(np.int64),
        z_w_f=np.uint64(z_f),
        # E_i @ K -> EK
        Q_sim_ek=Q_sim_ek_all.astype(np.int64),
        R_matmul_ek=R_matmul_ek_all.astype(np.int64),
        M_ek=np.uint64(M_ek),
        M_prime_ek=np.uint64(M_prime_ek),
        z_sim_ek=np.uint64(z_ek),
        # F_i @ V -> FV
        Q_sim_fv=Q_sim_fv_all.astype(np.int64),
        R_matmul_fv=R_matmul_fv_all.astype(np.int64),
        M_fv=np.uint64(M_fv),
        M_prime_fv=np.uint64(M_prime_fv),
        z_sim_fv=np.uint64(z_fv),
        # Q @ EK^T -> logit
        Q_sim_logit=Q_sim_logit_all.astype(np.int64),
        R_matmul_logit=R_matmul_logit_all.astype(np.int64),
        M_logit=np.uint64(M_logit_val),
        M_prime_logit=np.uint64(M_prime_logit),
        z_sim_logit=np.uint64(z_logit),
        # Softmax
        M_sm=np.uint64(M_sm),
        table_idx=table_idx.astype(np.int64),
        r_sm=r_sm.astype(np.int64),
        exp_total=exp_total.astype(np.uint64),
        P_lin=P_lin.astype(np.uint64),
        R_div_lin=R_div_lin.astype(np.uint64),
        exp_sum=exp_sum.astype(np.uint64),
        # P @ FV -> x
        Q_sim_x=Q_sim_x_all.astype(np.int64),
        R_matmul_x=R_matmul_x_all.astype(np.int64),
        M_x=np.uint64(M_final),
        M_prime_x=np.uint64(M_prime_x),
        z_sim_x=np.uint64(z_sim_x),
        # Metadata
        exp_table=exp_table,
        table_frac_bits=np.uint32(TABLE_FRAC_BITS),
        frac_bits=np.uint32(FRAC_BITS),
        k_bits=np.uint32(k_bits),
        mask=mask.astype(np.int64),
    )

    return {
        "max_abs_error": _max_abs,
        "cosine_similarity": _cos_sim,
        "s_raw_o": s_raw_o,
    }
