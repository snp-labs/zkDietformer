import math
import numpy as np
import os
from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul, check_neg_torch
)
from softmax_table import (
    FRAC_BITS, TABLE_FRAC_BITS,
    generate_fractional_exp_table, softmax_v3_lookup, save_softmax_tables_v3,
)
from transformers import BertForMaskedLM, BertConfig


def _load_optional_array(debug, key):
    """NPZ에서 None으로 저장된 optional 배열을 안전하게 로드."""
    if key not in debug:
        return None
    val = debug[key]
    if isinstance(val, np.ndarray) and val.dtype == object and val.ndim == 0:
        item = val.item()
        return None if item is None else val
    return val


def load_small_bert(
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    intermediate_size: int = None,
    seq_len: int = 128,
    vocab_size: int = 30522,
) -> BertForMaskedLM:
    """
    Random init BERT — 회로 비용 측정 전용.

    pretrained 가중치 없이 config 만으로 모델 생성.
    회로 constraint 수 / prove time 은 architecture 만 의존하므로
    가중치가 random 이어도 측정 결과 동일.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})."
        )

    config = BertConfig.from_pretrained("bert-base-uncased")
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size or (hidden_size * 4)
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.vocab_size = vocab_size
    config.max_position_embeddings = max(seq_len, 512)

    model = BertForMaskedLM(config)
    return model

def process_standard_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=26):
    # Save v3 softmax table for Rust Lasso lookup
    save_softmax_tables_v3(os.path.dirname(out_npz_path))

    debug = np.load(debug_npz_path, allow_pickle=True)

    # ── Phase 0: Linear Projection Witness ──────────────────────────────────
    # Q/K/V projection: Y = X @ W^T + b, then Q/K scaling by 1/sqrt⁴(D)
    # O projection: O = attn_output @ W_o^T + b_o

    hidden_states = debug['hidden_states']       # [B, S, HD] float
    W_q, W_k, W_v = debug['W_q'], debug['W_k'], debug['W_v']
    b_q = _load_optional_array(debug, 'b_q')
    b_k = _load_optional_array(debug, 'b_k')
    b_v = _load_optional_array(debug, 'b_v')
    W_o = debug['W_o']
    b_o = _load_optional_array(debug, 'b_o')
    attn_output_for_proj = debug['attn_output_for_proj']  # [B, S, HD]

    Q_proc_real = debug['Q_proc']  # [B, H, S, D]
    B, H, S, D = Q_proc_real.shape
    HD = H * D
    scale_factor = math.sqrt(math.sqrt(D))

    # 1. Quantize hidden_states (projection 공통 입력)
    s_x_input, z_x_input = get_quantization_params(hidden_states)
    Q_x_input = quantize_tensor(hidden_states, s_x_input, z_x_input)

    proj_witness = {
        "Q_x_input": Q_x_input.astype(np.int64),     # [B, S, HD]
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
        raw_real_4d = raw_real.reshape(B, S, H, D).transpose(0, 2, 1, 3)
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
            Q_raw_flat = (numerator >> k_bits).astype(np.int32)
            R_flat = (numerator % (1 << k_bits)).astype(np.int64)
        else:
            Q_b_diff = np.zeros(HD, dtype=np.int64)

        # Reshape [B,S,HD] → [B,H,S,D]
        Q_raw_4d = Q_raw_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        R_proj_4d = R_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

        # Per-head weight diffs [H, D, HD] and bias diffs [H, D]
        Q_w_diff = np.array([(Q_w[h*D:(h+1)*D, :] - z_w).astype(np.int64) for h in range(H)])
        Q_b_diff_per_head = Q_b_diff.reshape(H, D)

        proj_witness[f"Q_raw_proj_{name}"] = Q_raw_4d.astype(np.int64)
        proj_witness[f"R_proj_{name}"] = R_proj_4d.astype(np.int64)
        proj_witness[f"Q_w_diff_{name}"] = Q_w_diff.astype(np.int64)
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
            Q_scaled = (numerator_scale >> k_bits).astype(np.int32)
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

    # ── Phase 0 완료 (Output Projection은 kernel chain 이후 계산) ──────────

    # 2. Score 행렬곱 (Q * K^T) - O(S^2)
    # sign_bit_grouping_matmul 전용으로 처리. np.matmul 미사용.
    K_T = np.swapaxes(Q_k, -1, -2)

    # Pass 1: M=2^k_bits, z_y=0 으로 호출하면 출력 = (Q_q - z_q) @ (K_T - z_k) (정수)
    # → float 변환 후 s_y, z_y 도출 (순환 의존성 해소)
    Y_int_prelim, _, _ = sign_bit_grouping_matmul(
        Q_q, z_q, K_T, z_k, z_y=0, M=(1 << k_bits), k_bits=k_bits
    )
    Y_real = Y_int_prelim.astype(np.float64) * s_q * s_k

    s_y, z_y = get_quantization_params(Y_real)
    z_y += 10

    M_attn_scores = get_m_constant(s_q, s_k, s_y, k=k_bits)

    # Pass 2: 올바른 M, z_y 로 ZEN Matmul 적용 (Sign-bit grouping)
    Q_score_sim, R_matmul_score, M_prime_score = sign_bit_grouping_matmul(
        Q_q, z_q, K_T, z_k, z_y, M_attn_scores, k_bits
    )
    assert Q_score_sim.shape == Y_real.shape, \
        f"Q_score_sim shape {Q_score_sim.shape} != Y_real shape {Y_real.shape}"

    # 3. Softmax v3 (single table, table_frac_bits=10)
    exp_table = generate_fractional_exp_table(TABLE_FRAC_BITS, FRAC_BITS)

    # 4. Softmax 처리 — mask 적용
    mask = debug['mask']
    mask_4d = mask[:, None, None, :].astype(object)

    P_attn, R_div_attn, exp_total, exp_sum, M_sm, table_idx, r_sm = \
        softmax_v3_lookup(Q_score_sim, R_matmul_score, s_y, k_bits, FRAC_BITS,
                          TABLE_FRAC_BITS, exp_table, mask=mask_4d)

    # 5. 최종 출력 행렬곱 (P * V)
    X_real = debug['X']
    s_x, z_sim_x = get_quantization_params(X_real)
    
    z_p = 0 # Softmax 결과 영점
    s_p = 1.0 / (1 << FRAC_BITS) #

    M_final = get_m_constant(s_p, s_v, s_x, k=k_bits)
    
    Q_sim_x, R_matmul_x, M_prime_x = sign_bit_grouping_matmul(
        P_attn.astype(np.int64), z_p, Q_v, z_v, z_sim_x, M_final, k_bits
    )

    # ── Output Projection: O = Q_sim_x @ W_O^T + b_O ─────────────────────
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
        Q_o_flat = (numerator_o >> k_bits).astype(np.int32)
        R_o_flat = (numerator_o % (1 << k_bits)).astype(np.int64)
    else:
        Q_b_diff_o = np.zeros(HD, dtype=np.int64)

    # Reshape [B,S,HD] → [B,H,S,D]
    Q_o_4d = Q_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)
    R_o_4d = R_o_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)

    # ── Quantization Error Measurement ──────────────────────────────────────
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
    print(f"  [BERT O-proj] MSE={_mse:.6e}  RMSE={_rmse:.6e}  "
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

    # 6. NPZ 저장
    np.savez_compressed(
        out_npz_path,
        ### Projection witness (Phase 0 + Output Projection)
        **proj_witness,
        ### 공통 입력 (projection chain에서 연결됨)
        Q_q=Q_q, z_q=np.uint64(z_q),
        Q_k=Q_k, z_k=np.uint64(z_k),
        Q_v=Q_v, z_v=np.uint64(z_v),
        
        # Attention Scores (Full Matrix)
        Q_sim_attn=Q_score_sim.astype(np.int64),
        R_matmul_attn=R_matmul_score.astype(np.int64),
        M_attn=np.uint64(M_attn_scores),
        M_prime_attn=np.uint64(M_prime_score),
        z_sim_attn=np.uint64(z_y),
        
        # Softmax 결과 — v3
        M_sm=np.uint64(M_sm),
        table_idx=table_idx.astype(np.int64),
        r_sm=r_sm.astype(np.int64),
        exp_total=exp_total.astype(np.uint64),
        P_attn=P_attn.astype(np.uint64),
        R_div_attn=R_div_attn.astype(np.uint64),
        exp_sum=exp_sum.astype(np.uint64),
        
        # 최종 출력 X
        Q_sim_x=Q_sim_x.astype(np.int64),
        R_matmul_x=R_matmul_x.astype(np.int64),
        M_x=np.uint64(M_final),
        M_prime_x=np.uint64(M_prime_x),
        z_sim_x=np.uint64(z_sim_x),
        
        # 메타데이터
        exp_table=exp_table,
        table_frac_bits=np.uint32(TABLE_FRAC_BITS),
        frac_bits=np.uint32(FRAC_BITS),
        k_bits=np.uint32(k_bits),
        mask=mask.astype(np.int64)
    )
    return quant_metrics