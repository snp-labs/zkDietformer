import hashlib
import math
import os
import numpy as np
from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul, check_neg_torch,
    ensure_nonneg_matmul,
)
from softmax_table import (
    FRAC_BITS, TABLE_FRAC_BITS,
    generate_fractional_exp_table, softmax_v3_lookup, save_softmax_tables_v3,
)


def _load_optional_array(debug, key):
    """NPZ에서 None으로 저장된 optional 배열을 안전하게 로드."""
    if key not in debug:
        return None
    val = debug[key]
    if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 0:
        item = val.item()
        return None if item is None else val
    return val

    
def process_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=26):
    # Save v3 softmax table for Rust Lasso lookup
    save_softmax_tables_v3(os.path.dirname(out_npz_path))

    debug = np.load(debug_npz_path, allow_pickle=True)

    ### ── Phase 0: Linear Projection Witness ──
    # Q/K/V projection: Y = X @ W^T + b, then Q/K scaling by 1/sqrt⁴(D)
    # O projection: O = attn_output @ W_o^T + b_o

    hidden_states = debug['hidden_states']       # [B, S, HD] float 입력값 X
    W_q, W_k, W_v = debug['W_q'], debug['W_k'], debug['W_v']
    b_q = _load_optional_array(debug, 'b_q')
    b_k = _load_optional_array(debug, 'b_k')
    b_v = _load_optional_array(debug, 'b_v')
    W_o = debug['W_o']
    b_o = _load_optional_array(debug, 'b_o')
    attn_output_for_proj = debug['attn_output_for_proj']  # [B, S, HD]

    Q_proc_real = debug['Q_proc']  # [B, H, S, D] 그냥 D만 알아내려고 뽑아옴. 그 외 쓰는 곳 없음
    B_proj, H, S_proj, D = Q_proc_real.shape
    HD = H * D
    scale_factor = math.sqrt(math.sqrt(D))

    # 1. Quantize hidden_states (projection 공통 입력)
    s_x_input, z_x_input = get_quantization_params(hidden_states)
    Q_x_input = quantize_tensor(hidden_states, s_x_input, z_x_input)

    proj_witness = {
        "Q_x_input": Q_x_input.astype(np.int64),     # [B, S, HD] 양자화한 입력값 X
        "z_x_input": np.uint64(z_x_input),
    }

    # 2. Q/K/V projections (per-head)
    for name, W, b, apply_scale in [("q", W_q, b_q, True),
                                     ("k", W_k, b_k, True),
                                     ("v", W_v, b_v, False)]:
        # Weight quantization (single scale for full weight matrix)
        s_w, z_w = get_quantization_params(W)
        Q_w = quantize_tensor(W, s_w, z_w)

        # Raw projection output in float (for output quantization params)
        raw_real = np.matmul(hidden_states, W.T)
        if b is not None:
            raw_real = raw_real + b
        raw_real_4d = raw_real.reshape(B_proj, S_proj, H, D).transpose(0, 2, 1, 3)
        s_raw, z_raw = get_quantization_params(raw_real_4d)

        M_proj = get_m_constant(s_x_input, s_w, s_raw, k=k_bits)

        # Full ZEN matmul: Q_x_input [B,S,HD] @ Q_w.T [HD,HD] → [B,S,HD]
        Q_raw_flat, R_flat, M_prime_proj = sign_bit_grouping_matmul(
            Q_x_input, z_x_input, Q_w.T.astype(np.int64), z_w,
            z_raw, M_proj, k_bits
        )

        # Bias: Q_b_diff = round(b / (s_w * s_x))
        if b is not None:
            s_b = s_w * s_x_input
            Q_b_diff = np.round(b / s_b).astype(np.int64)  # [HD]
            numerator = Q_raw_flat.astype(np.int64) * (1 << k_bits) + R_flat.astype(np.int64)
            numerator = numerator + M_proj * Q_b_diff
            Q_raw_flat = (numerator >> k_bits).astype(np.int64)
            R_flat = (numerator % (1 << k_bits)).astype(np.int64)
        else:
            Q_b_diff = np.zeros(HD, dtype=np.int64)

        # Reshape [B,S,HD] → [B,H,S,D]
        Q_raw_4d = Q_raw_flat.reshape(B_proj, S_proj, H, D).transpose(0, 2, 1, 3)
        R_proj_4d = R_flat.reshape(B_proj, S_proj, H, D).transpose(0, 2, 1, 3)

        # Per-head weight diffs [H, D, HD] and bias diffs [H, D]
        Q_w_diff = np.array([(Q_w[h*D:(h+1)*D, :] - z_w).astype(np.int64) for h in range(H)])
        Q_b_diff_per_head = Q_b_diff.reshape(H, D)

        proj_witness[f"Q_raw_proj_{name}"] = Q_raw_4d.astype(np.int64)
        proj_witness[f"R_proj_{name}"] = R_proj_4d.astype(np.int64)
        proj_witness[f"Q_w_diff_{name}"] = Q_w_diff.astype(np.int64) # 가중치와 가중치에 대한 제로포인트의 차
        proj_witness[f"Q_b_diff_{name}"] = Q_b_diff_per_head.astype(np.int64)
        proj_witness[f"z_w_{name}"] = np.uint64(z_w)
        proj_witness[f"M_proj_{name}"] = np.uint64(M_proj)
        proj_witness[f"M_prime_proj_{name}"] = np.uint64(M_prime_proj)
        proj_witness[f"z_raw_{name}"] = np.uint64(z_raw)

        # Q/K scaling: Q_proc = Q_raw / scale_factor
        if apply_scale:
            if name == "q":
                s_proc, z_proc = get_quantization_params(debug['Q_proc'])
            else:
                s_proc, z_proc = get_quantization_params(debug['K_proc'])

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

            # Connect scaling output → kernel chain (Q_q/Q_k notation 유지)
            if name == "q":
                Q_q = Q_scaled.astype(np.int64)
                s_q, z_q = s_proc, z_proc
            else:  # name == "k"
                Q_k = Q_scaled.astype(np.int64)
                s_k, z_k = s_proc, z_proc
        else:
            # V: scaling 없음 → raw projection 출력을 kernel chain에 연결
            Q_v = Q_raw_4d.astype(np.int64)
            s_v, z_v = s_raw, z_raw

    ### ── Phase 1: Attention Computation ──
    ### kernel 1
    B, H, S, D = Q_q.shape
    L = debug['Q_landmarks'].shape[2] 
    seg = S // L
    
    K_sum = Q_k.reshape(B, H, L, seg, D).sum(axis=-2)
    K_landmarks_q = K_sum // seg
    R_mean_k = K_sum % seg 

    Y_real = debug['kernel_1_scores']
    s_y1, z_y1 = get_quantization_params(Y_real)

    K_landmarks_T = np.swapaxes(K_landmarks_q, -1, -2)

    Q_y1_sim, R_matmul_k1, M_prime_k1, M_k1, z_y1 = ensure_nonneg_matmul(
        Q_q, z_q, K_landmarks_T, z_k, s_q, s_k, s_y1, z_y1, k_bits
    )

    ### kernel 2
    Q_sum = Q_q.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_q = Q_sum // seg
    R_mean_q = Q_sum % seg

    Y2_real = debug['kernel_2_scores']
    s_y2, z_y2 = get_quantization_params(Y2_real)

    K_landmarks_T = np.swapaxes(K_landmarks_q, -1, -2)

    Q_y2_sim, R_matmul_k2, M_prime_k2, M_k2, z_y2 = ensure_nonneg_matmul(
        Q_landmarks_q, z_q, K_landmarks_T, z_k, s_q, s_k, s_y2, z_y2, k_bits
    )

    ### kernel 3
    Y3_real = debug['kernel_3_scores']
    s_y3, z_y3 = get_quantization_params(Y3_real)

    K_T = np.swapaxes(Q_k, -1, -2)

    Q_sim_k3, R_matmul_k3, M_prime_k3, M_k3, z_y3 = ensure_nonneg_matmul(
        Q_landmarks_q, z_q, K_T, z_k, s_q, s_k, s_y3, z_y3, k_bits
    )

    ### softmax v3 (single table, table_frac_bits=10)
    exp_table = generate_fractional_exp_table(TABLE_FRAC_BITS, FRAC_BITS)

    ### softmax(kernel1)
    P_k1, R_div_k1, exp_total_k1, exp_sum_k1, M_sm_k1, table_idx_k1, r_sm_k1 = \
        softmax_v3_lookup(Q_y1_sim, R_matmul_k1, s_y1, k_bits, FRAC_BITS,
                          TABLE_FRAC_BITS, exp_table)

    ### softmax(kernel2)
    P_k2, R_div_k2, exp_total_k2, exp_sum_k2, M_sm_k2, table_idx_k2, r_sm_k2 = \
        softmax_v3_lookup(Q_y2_sim, R_matmul_k2, s_y2, k_bits, FRAC_BITS,
                          TABLE_FRAC_BITS, exp_table)

    ### softmax(kernel3) — mask 적용
    mask = debug['mask']
    mask_4d = mask[:, None, None, :].astype(object)
    P_k3, R_div_k3, exp_total_k3, exp_sum_k3, M_sm_k3, table_idx_k3, r_sm_k3 = \
        softmax_v3_lookup(Q_sim_k3, R_matmul_k3, s_y3, k_bits, FRAC_BITS,
                          TABLE_FRAC_BITS, exp_table, mask=mask_4d)

    check_neg_torch(P_k3, "P_k3")
    check_neg_torch(R_div_k3, "R_div_k3")

    ### Pseudoinverse
    inv_k2_real = debug['inv_kernel_2']
    s_inv, z_inv = get_quantization_params(inv_k2_real)
    Q_inv_k2 = quantize_tensor(inv_k2_real, s_inv, z_inv).astype(np.int64)

    A_q = P_k2.astype(np.int64) # A는 Softmax 결과인 P_k2를 사용
    z_p2 = 0 # softmax범위 0~1
    s_p2 = 1.0 / (1 << FRAC_BITS) 


    ### A * A+ * A = A
    M_inv1 = get_m_constant(s_p2, s_inv, s_p2, k=k_bits) # 스케일을 맞추기 위함
    Q_y1, R_inv1, M_prime_inv1 = sign_bit_grouping_matmul(
        A_q, z_p2, Q_inv_k2, z_inv, z_p2, M_inv1, k_bits
    )
    
    M_inv2 = get_m_constant(s_p2, s_p2, s_p2, k=k_bits)
    Q_y2, R_inv2, M_prime_inv2 = sign_bit_grouping_matmul(
        Q_y1, z_p2, A_q, z_p2, z_p2, M_inv2, k_bits 
    )

    ### A+ * A * A+ = A+
    M_inv3 = get_m_constant(s_inv, s_p2, s_inv, k=k_bits)
    Q_y3, R_inv3, M_prime_inv3 = sign_bit_grouping_matmul(
        Q_inv_k2, z_inv, A_q, z_p2, z_inv, M_inv3, k_bits
    )
    
    M_inv4 = get_m_constant(s_inv, s_inv, s_inv, k=k_bits)
    Q_y4, R_inv4, M_prime_inv4 = sign_bit_grouping_matmul(
        Q_y3, z_inv, Q_inv_k2, z_inv, z_inv, M_inv4, k_bits
    )

    temp_real = debug['temp']
    s_sim_temp, z_sim_temp = get_quantization_params(temp_real)
    
    z_pk3 = 0 # softmax범위 0~1
    s_pk3 = 1.0 / (1 << FRAC_BITS)

    M_temp = get_m_constant(s_pk3, s_v, s_sim_temp, k=k_bits)
    
    Q_sim_temp, R_matmul_temp, M_prime_temp = sign_bit_grouping_matmul(
        P_k3.astype(np.int64), z_pk3, Q_v.astype(np.int64), z_v, z_sim_temp, M_temp, k_bits
    )

    check_neg_torch(Q_sim_temp, "Q_sim_temp")
    check_neg_torch(R_matmul_temp, "R_matmul_temp")

    ### mid
    mid_real = debug['mid']
    s_mid, z_sim_mid = get_quantization_params(mid_real)
    
    z_pk1 = 0 # softmax범위 0~1
    s_pk1 = 1.0 / (1 << FRAC_BITS)

    M_mid = get_m_constant(s_pk1, s_inv, s_mid, k=k_bits)
    
    Q_sim_mid, R_matmul_mid, M_prime_mid = sign_bit_grouping_matmul(
        P_k1.astype(np.int64), z_pk1, Q_inv_k2.astype(np.int64), z_inv, z_sim_mid, M_mid, k_bits
    )

    check_neg_torch(Q_sim_mid, "Q_sim_mid")
    check_neg_torch(R_matmul_mid, "R_matmul_mid")

    ### X
    X_real = debug['X']
    s_x, z_sim_x = get_quantization_params(X_real)
    
    M_x = get_m_constant(s_mid, s_sim_temp, s_x, k=k_bits)
    
    Q_sim_x, R_matmul_x, M_prime_x = sign_bit_grouping_matmul(
        Q_sim_mid.astype(np.int64), z_sim_mid, Q_sim_temp.astype(np.int64), z_sim_temp, z_sim_x, M_x, k_bits
    )

    check_neg_torch(Q_sim_x, "Q_sim_x")
    check_neg_torch(R_matmul_x, "R_matmul_x")

    # Save original matmul_x results before potential conv overwrite
    Q_sim_x_matmul = Q_sim_x.copy()  # pre-conv: for matmul_x constraint
    z_sim_x_matmul = z_sim_x          # pre-conv: for matmul_x constraint

    ### ── Phase 2: Conv: Toeplitz(W_conv) @ V ──
    W_conv = _load_optional_array(debug, 'W_conv')  # [H, 1, K_conv, 1]
    conv_term_real = _load_optional_array(debug, 'conv_term')  # [B, H, S, D]

    if W_conv is not None and conv_term_real is not None:
        K_conv = W_conv.shape[2]  # 65
        pad_conv = K_conv // 2     # 32

        # Toeplitz matrix: T[h, s, j] = kernel[h, s-j+pad]
        T_toeplitz = np.zeros((H, S, S), dtype=np.float64)
        for h in range(H):
            kernel = W_conv[h, 0, :, 0]
            for s_idx in range(S):
                for k_idx in range(K_conv):
                    j = s_idx + k_idx - pad_conv
                    if 0 <= j < S:
                        T_toeplitz[h, s_idx, j] = kernel[k_idx]

        # Toeplitz quantization
        s_t_conv, z_t_conv = get_quantization_params(T_toeplitz)
        Q_t_conv = quantize_tensor(T_toeplitz, s_t_conv, z_t_conv)

        # Conv output quantization params (float reference)
        s_conv_out, z_conv_out = get_quantization_params(conv_term_real)

        # ZEN matmul per head: T_h(S,S) @ V_h(S,D)
        M_conv = get_m_constant(s_t_conv, s_v, s_conv_out, k=k_bits)
        Q_conv_list, R_conv_list = [], []
        M_prime_conv = None
        for h in range(H):
            Q_ch, R_ch, M_prime_conv = sign_bit_grouping_matmul(
                Q_t_conv[h], z_t_conv,
                Q_v[:, h, :, :].astype(np.int64), z_v,
                z_conv_out, M_conv, k_bits
            )
            Q_conv_list.append(Q_ch)
            R_conv_list.append(R_ch)
        Q_conv = np.stack(Q_conv_list, axis=1)  # [B, H, S, D]
        R_conv = np.stack(R_conv_list, axis=1)

        # X_final = X + conv (integer rescaled addition)
        X_final_real = debug['X_final']
        s_xf, z_xf = get_quantization_params(X_final_real)

        M_add_x = int(np.floor((1 << k_bits) * s_x / s_xf))
        M_add_conv = int(np.floor((1 << k_bits) * s_conv_out / s_xf))
        if M_add_x == 0: M_add_x = 1
        if M_add_conv == 0: M_add_conv = 1

        numerator_add = (M_add_x * (Q_sim_x - z_sim_x).astype(np.int64)
                       + M_add_conv * (Q_conv - z_conv_out).astype(np.int64)
                       + z_xf * (1 << k_bits))
        Q_xf = (numerator_add >> k_bits).astype(np.int64)
        R_xf = (numerator_add % (1 << k_bits)).astype(np.int64)

        # Update variables so O-projection uses X_final
        Q_sim_x = Q_xf.astype(np.int64)
        s_x = s_xf
        z_sim_x = z_xf

        conv_witness = {
            "Q_t_conv": Q_t_conv.astype(np.int64),
            "z_t_conv": np.uint64(z_t_conv),
            "M_conv": np.uint64(M_conv),
            "M_prime_conv": np.uint64(M_prime_conv),
            "z_conv_out": np.uint64(z_conv_out),
            "Q_conv": Q_conv.astype(np.int64),
            "R_conv": R_conv.astype(np.int64),
            "Q_xf": Q_xf.astype(np.int64),
            "R_xf": R_xf.astype(np.int64),
            "M_add_x": np.uint64(M_add_x),
            "M_add_conv": np.uint64(M_add_conv),
            "z_xf": np.uint64(z_xf),
        }

        print(f"  [Conv Toeplitz] K_conv={K_conv}, M_conv={M_conv}, "
              f"M_add_x={M_add_x}, M_add_conv={M_add_conv}")
    else:
        conv_witness = {}

    # ── Phase 3. Output Projection: O = Q_sim_x @ W_O^T + b_O ───
    # Q_sim_x [B,H,S,D] → [B,S,HD] reshape 후 sign_bit_grouping_matmul
    Q_sim_x_flat = Q_sim_x.transpose(0, 2, 1, 3).reshape(B, S, HD)

    s_w_o, z_w_o = get_quantization_params(W_o)
    Q_w_o = quantize_tensor(W_o, s_w_o, z_w_o)

    # Float reference → output quantization params
    o_real = np.matmul(attn_output_for_proj, W_o.T)
    if b_o is not None:
        o_real = o_real + b_o
    o_real_4d = o_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    s_raw_o, z_raw_o = get_quantization_params(o_real_4d)

    M_proj_o = get_m_constant(s_x, s_w_o, s_raw_o, k=k_bits)

    # Full ZEN matmul: Q_sim_x_flat [B,S,HD] @ Q_w_o.T [HD,HD]
    Q_o_flat, R_o_flat, M_prime_proj_o = sign_bit_grouping_matmul(
        Q_sim_x_flat.astype(np.int64), z_sim_x,
        Q_w_o.T.astype(np.int64), z_w_o,
        z_raw_o, M_proj_o, k_bits
    )

    # Bias: Q_b_diff = round(b / (s_w * s_x))
    if b_o is not None:
        s_b_o = s_w_o * s_x
        Q_b_diff_o = np.round(b_o / s_b_o).astype(np.int64)     # [HD]
        numerator_o = Q_o_flat.astype(np.int64) * (1 << k_bits) + R_o_flat.astype(np.int64)
        numerator_o = numerator_o + M_proj_o * Q_b_diff_o
        Q_o_flat = (numerator_o >> k_bits).astype(np.int64)
        R_o_flat = (numerator_o % (1 << k_bits)).astype(np.int64)
    else:
        Q_b_diff_o = np.zeros(HD, dtype=np.int64)

    # Reshape [B,S,HD] → [B,H,S,D]
    Q_o_4d = Q_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    R_o_4d = R_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

    # ── Quantization Error Measurement ───
    o_approx = (Q_o_4d.astype(np.float64) - z_raw_o) * s_raw_o
    o_ref = o_real_4d.astype(np.float64)
    _diff = o_ref - o_approx
    _mse = float(np.mean(_diff ** 2))
    _rmse = float(np.sqrt(_mse))
    _max_abs = float(np.max(np.abs(_diff)))
    _norm_ref = np.linalg.norm(o_ref)
    _norm_approx = np.linalg.norm(o_approx)
    _cos_sim = float(np.sum(o_ref * o_approx) / (_norm_ref * _norm_approx + 1e-12))
    _rel_err = float(np.mean(np.abs(_diff) / (np.abs(o_ref) + 1e-12)))
    quant_metrics = {
        "mse": _mse, "rmse": _rmse, "max_abs_error": _max_abs,
        "cosine_similarity": _cos_sim, "mean_relative_error": _rel_err,
    }
    print(f"  [Nystrom O-proj] MSE={_mse:.6e}  RMSE={_rmse:.6e}  "
          f"MaxAbs={_max_abs:.6e}  CosSim={_cos_sim:.8f}  RelErr={_rel_err:.6e}")

    Q_w_diff_o_per_head = np.array([(Q_w_o[h*D:(h+1)*D, :] - z_w_o).astype(np.int64) for h in range(H)])
    Q_b_diff_o_per_head = Q_b_diff_o.reshape(H, D)

    proj_witness.update({
        "Q_attn_output": Q_sim_x_flat.astype(np.int64),   # connected: Q_sim_x
        "z_attn_output": np.uint64(z_sim_x),               # connected: z_sim_x
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

    np.savez_compressed(
        out_npz_path,
        ### Projection witness (Phase 0 + Output Projection)
        **proj_witness,
        ### 공통 입력 (projection chain에서 연결됨)
        Q_q=Q_q.astype(np.int64),
        z_q=np.uint64(z_q),
        Q_k=Q_k.astype(np.int64),
        z_k=np.uint64(z_k),
        ### kernel 1
        R_mean_k1=R_mean_k.astype(np.int64),
        Q_sim_k1=Q_y1_sim.astype(np.int64),
        R_matmul_k1=R_matmul_k1.astype(np.int64),
        M_k1=np.uint64(M_k1),  
        M_prime_k1=np.uint64(M_prime_k1), 
        z_sim_k1=np.uint64(z_y1),  
        ### kernel 2
        R_mean_k2=R_mean_q.astype(np.int64),
        Q_sim_k2=Q_y2_sim.astype(np.int64),
        R_matmul_k2=R_matmul_k2.astype(np.int64),
        M_k2=np.uint64(M_k2),  
        M_prime_k2=np.uint64(M_prime_k2),  
        z_sim_k2=np.uint64(z_y2),  
        ### kernel 3
        Q_sim_k3=Q_sim_k3.astype(np.int64),
        R_matmul_k3=R_matmul_k3.astype(np.int64),
        M_k3=np.uint64(M_k3),  
        M_prime_k3=np.uint64(M_prime_k3),  
        z_sim_k3=np.uint64(z_y3),  
        ### softmax(kernel1) — v3
        M_sm_k1=np.uint64(M_sm_k1),
        table_idx_k1=table_idx_k1.astype(np.int64),
        r_sm_k1=r_sm_k1.astype(np.int64),
        exp_total_k1=exp_total_k1.astype(np.uint64),
        exp_sum_k1=exp_sum_k1.astype(np.uint64),
        P_k1=P_k1.astype(np.uint64),
        R_div_k1=R_div_k1.astype(np.uint64),
        ### softmax(kernel2) — v3
        M_sm_k2=np.uint64(M_sm_k2),
        table_idx_k2=table_idx_k2.astype(np.int64),
        r_sm_k2=r_sm_k2.astype(np.int64),
        exp_total_k2=exp_total_k2.astype(np.uint64),
        exp_sum_k2=exp_sum_k2.astype(np.uint64),
        P_k2=P_k2.astype(np.uint64),
        R_div_k2=R_div_k2.astype(np.uint64),
        ### softmax(kernel3) — v3
        M_sm_k3=np.uint64(M_sm_k3),
        table_idx_k3=table_idx_k3.astype(np.int64),
        r_sm_k3=r_sm_k3.astype(np.int64),
        exp_total_k3=exp_total_k3.astype(np.uint64),
        exp_sum_k3=exp_sum_k3.astype(np.uint64),
        P_k3=P_k3.astype(np.uint64),
        R_div_k3=R_div_k3.astype(np.uint64),
        ### moore-penrose (Pseudoinverse)
        Q_inv_k2=Q_inv_k2.astype(np.int64),
        z_inv_k2=np.uint64(z_inv), 
        M_inv_all=np.array([M_inv1, M_inv2, M_inv3, M_inv4], dtype=np.uint64),  
        M_prime_inv_all=np.array([M_prime_inv1, M_prime_inv2, M_prime_inv3, M_prime_inv4], dtype=np.uint64),  
        Q_y_inv_intermediates=np.stack([Q_y1, Q_y2, Q_y3, Q_y4], axis=0).astype(np.int64),
        R_inv_remainders=np.stack([R_inv1, R_inv2, R_inv3, R_inv4], axis=0).astype(np.int64),
        ### temp
        Q_v=Q_v.astype(np.int64),
        z_v=np.uint64(z_v),  
        Q_sim_temp=Q_sim_temp.astype(np.int64),
        R_matmul_temp=R_matmul_temp.astype(np.int64),
        M_temp=np.uint64(M_temp),  
        M_prime_temp=np.uint64(M_prime_temp),  
        z_sim_temp=np.uint64(z_sim_temp),  
        ### mid = P_k1 @ Q_inv_k2
        Q_sim_mid=Q_sim_mid.astype(np.int64),
        R_matmul_mid=R_matmul_mid.astype(np.int64),
        M_mid=np.uint64(M_mid),  
        M_prime_mid=np.uint64(M_prime_mid),  
        z_sim_mid=np.uint64(z_sim_mid),  
        ### X = mid @ temp (pre-conv values for matmul constraint)
        Q_sim_x=Q_sim_x_matmul.astype(np.int64),
        R_matmul_x=R_matmul_x.astype(np.int64),
        M_x=np.uint64(M_x),
        M_prime_x=np.uint64(M_prime_x),
        z_sim_x=np.uint64(z_sim_x_matmul),
        ### meta data
        exp_table=exp_table,
        table_frac_bits=np.uint32(TABLE_FRAC_BITS),
        frac_bits=np.uint32(FRAC_BITS),
        seg_size=np.uint64(seg),  
        k_bits=np.uint32(k_bits),
        mask=mask.astype(np.int64),
        ### conv (Toeplitz) — W_conv 존재 시에만
        **conv_witness,
    )
    return quant_metrics