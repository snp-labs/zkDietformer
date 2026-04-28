import os
import numpy as np
from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, get_gelu_index_multiplier, check_neg_torch,
)
from layernorm_table import (
    generate_layernorm_tables, extract_inv_std_single,
    get_layernorm_index_multiplier,
    FRAC_BITS, EPS_BERT, VAR_SCALE,
)
from gelu_table import generate_gelu_sigmoid_tables, extract_gelu_sigmoid_single


def _process_layernorm_block(
    x_real, var_ref, gamma_real, beta_real, out_real,
    k_bits, frac_bits, eps, ln_table,
):
    dim = x_real.shape[-1]

    # 1. 입력 양자화
    s_ri, z_ri = get_quantization_params(x_real)
    Q_ri = quantize_tensor(x_real, s_ri, z_ri).astype(np.int64)

    # 2. 평균 (정수 도메인). z_ri는 (Q_ri − Q_mean) 형성 시 자동 상쇄.
    sum_x = np.sum(Q_ri, axis=-1, keepdims=True)
    Q_mean = sum_x // dim
    R_mean = sum_x % dim

    # 3. 평균 차감된 정수 차이
    Q_diff = (Q_ri - Q_mean).astype(np.int64)

    # 4. 분산: sum_sq = Σ Q_diff². var-domain으로 rescale.
    s_var, z_var = get_quantization_params(var_ref)
    M_var = get_m_constant(s_ri, s_ri, s_var, k=k_bits)
    M_prime_var = int(np.floor((z_var * (1 << k_bits)) / M_var))

    sum_sq = np.sum(Q_diff ** 2, axis=-1, keepdims=True)

    numerator_var = M_var * (sum_sq + M_prime_var)
    shifted_val_var = numerator_var >> k_bits
    R_shift_var = numerator_var % (1 << k_bits)

    Q_var_sim = shifted_val_var // dim # 1/N
    R_var = shifted_val_var % dim 

    # 5. inv_std lookup (1/√(var+ε)) — Phase 4: VAR_SCALE 적용한 fixed-point 인덱스
    M_ln = get_layernorm_index_multiplier(s_var, k_bits=k_bits, var_scale=VAR_SCALE)
    inv_std_total = extract_inv_std_single(Q_var_sim, R_var, dim, ln_table, M_ln, k_bits)

    # 6. γ 양자화
    s_gamma, z_gamma = get_quantization_params(gamma_real)
    Q_gamma = quantize_tensor(gamma_real, s_gamma, z_gamma).astype(np.int64)

    # 7. β 양자화 (별도 scale)
    s_beta, z_beta = get_quantization_params(beta_real)
    Q_beta = quantize_tensor(beta_real, s_beta, z_beta).astype(np.int64)

    # 8. 출력 rescale 상수
    s_out_norm, z_out_norm = get_quantization_params(out_real)
    M_norm = get_m_constant(s_ri, s_gamma, s_out_norm * (1 << frac_bits), k=k_bits)
    M_prime_norm = int(np.floor((z_out_norm * (1 << k_bits)) / M_norm))

    # 9. β를 inner-product scale로 lifting.
    #    inner-product scale: s_inner = s_ri · s_gamma / 2^frac_bits
    #    Q_beta_lifted ≈ (Q_beta − z_beta) · s_beta / s_inner
    #    M_beta_lift = floor(2^k_bits · s_beta / s_inner)
    s_inner = (s_ri * s_gamma) / (1 << frac_bits)
    M_beta_lift = max(1, int(np.floor(((1 << k_bits) * s_beta) / s_inner)))
    Q_beta_diff = (Q_beta - z_beta).astype(np.int64)
    numerator_beta = M_beta_lift * Q_beta_diff.astype(object)
    Q_beta_lifted = (numerator_beta >> k_bits).astype(np.int64)
    R_beta_lift = (numerator_beta % (1 << k_bits)).astype(np.int64)

    # 10. inner = Q_diff · inv_std · (Q_gamma − z_gamma)
    inner = (
        Q_diff.astype(object)
        * inv_std_total.astype(object)
        * (Q_gamma - z_gamma).astype(object)
    )

    # β는 차원 [H] 이지만 broadcast로 inner [B,S,H]에 더해진다.
    numerator_norm = M_norm * (inner + Q_beta_lifted.astype(object) + M_prime_norm)
    Q_ln_output_sim = (numerator_norm >> k_bits).astype(np.int64)
    R_ln_output = (numerator_norm % (1 << k_bits)).astype(np.int64)

    check_neg_torch(Q_ln_output_sim, "Q_ln_output_sim")
    check_neg_torch(R_ln_output, "R_ln_output")

    return {
        # 입력
        "Q_ri": Q_ri,
        "z_ri": np.uint64(z_ri),
        # 평균
        "sum_x": sum_x.astype(np.int64),
        "Q_mean": Q_mean.astype(np.int64),
        "R_mean": R_mean.astype(np.int64),
        # 차이 (zero-mean)
        "Q_diff": Q_diff,
        # 분산
        "sum_sq": sum_sq.astype(np.int64),
        "M_var": np.uint64(M_var),
        "M_prime_var": np.uint64(M_prime_var),
        "shifted_val_var": shifted_val_var.astype(np.int64),
        "R_shift_var": R_shift_var.astype(np.int64),
        "Q_var_sim": Q_var_sim.astype(np.int64),
        "R_var": R_var.astype(np.int64),
        "inv_std_total": inv_std_total.astype(np.uint64),
        "M_ln": np.uint64(M_ln),
        # γ
        "Q_gamma": Q_gamma,
        "z_gamma": np.uint64(z_gamma),
        # β + lifting
        "Q_beta": Q_beta,
        "z_beta": np.uint64(z_beta),
        "M_beta_lift": np.uint64(M_beta_lift),
        "Q_beta_lifted": Q_beta_lifted,
        "R_beta_lift": R_beta_lift,
        # 출력 rescale
        "M_norm": np.uint64(M_norm),
        "M_prime_norm": np.uint64(M_prime_norm),
        "Q_ln_output_sim": Q_ln_output_sim,
        "R_ln_output": R_ln_output,
    }


def _flatten_block(suffix, block_dict):
    """{'Q_ri': ...} → {'Q_ri_<suffix>': ...} 로 prefix 부착."""
    return {f"{k}_{suffix}": v for k, v in block_dict.items()}


def process_standard_normalization_debug_npz(
    debug_npz_path, out_npz_path, k_bits=22, eps=EPS_BERT,
):
    debug = np.load(debug_npz_path, allow_pickle=True)

    table_dir = "outputs_bert"

    # 표 생성/로드 (1회).
    ln_table_path = generate_layernorm_tables(
        table_dir, k_bits=k_bits, frac_bits=FRAC_BITS, eps=eps,
    )
    ln_table = np.load(ln_table_path)['table']
    gelu_table_path = generate_gelu_sigmoid_tables(
        table_dir, k_bits=k_bits, frac_bits=FRAC_BITS,
    )
    gelu_table = np.load(gelu_table_path)['table']

    # ── Block 1: post-attention LayerNorm ────────────────────────────────
    block1 = _process_layernorm_block(
        x_real=debug['ln_input'],
        var_ref=debug['var'],
        gamma_real=debug['ln_weight'],
        beta_real=debug['ln_bias'],
        out_real=debug['ln_output'],
        k_bits=k_bits, frac_bits=FRAC_BITS, eps=eps, ln_table=ln_table,
    )

    # ── Linear 1 (FFN up-projection) ─────────────────────────────────────
    w_lin1 = debug['linear_weight']
    out_lin1 = debug['linear_output']
    b_lin1 = debug.get('linear_bias', None)

    s_in_lin1, z_in_lin1 = get_quantization_params(debug['ln_output'])
    s_w_lin1, z_w_lin1 = get_quantization_params(w_lin1)
    s_out_lin1, z_out_lin1 = get_quantization_params(out_lin1)

    Q_w_diff1 = (quantize_tensor(w_lin1, s_w_lin1, z_w_lin1) - z_w_lin1).astype(np.int64)
    Q_in_diff1 = (block1['Q_ln_output_sim'] - z_in_lin1).astype(np.int64)

    G1 = np.matmul(Q_in_diff1, Q_w_diff1.T)
    if b_lin1 is not None:
        s_b1 = s_w_lin1 * s_in_lin1
        Q_b_diff1 = np.round(b_lin1 / s_b1).astype(np.int64)
        acc1 = G1 + Q_b_diff1
    else:
        Q_b_diff1 = np.zeros_like(out_lin1[..., 0:1])
        acc1 = G1

    M_lin1 = get_m_constant(s_w_lin1, s_in_lin1, s_out_lin1, k=k_bits)
    M_prime_lin1 = int(np.floor((z_out_lin1 * (1 << k_bits)) / M_lin1))
    numerator_lin1 = M_lin1 * (acc1 + M_prime_lin1)
    Q_out_sim_lin1 = (numerator_lin1 >> k_bits).astype(np.int32)
    R_out_lin1 = (numerator_lin1 % (1 << k_bits)).astype(np.int64)

    check_neg_torch(numerator_lin1, "numerator_lin1")
    check_neg_torch(Q_out_sim_lin1, "Q_out_sim_lin1")
    check_neg_torch(R_out_lin1, "R_out_lin1")

    # ── GELU ─────────────────────────────────────────────────────────────
    M_gelu_idx = get_gelu_index_multiplier(s_out_lin1, k_bits=k_bits)
    sigmoid_part = extract_gelu_sigmoid_single(
        Q_out_sim_lin1, R_out_lin1, z_out_lin1, gelu_table, M_gelu_idx, k_bits,
    )
    acc_gelu = (Q_out_sim_lin1 - z_out_lin1).astype(object) * sigmoid_part.astype(object)
    s_gelu_out, z_gelu_out = get_quantization_params(debug['gelu_output'])

    M_gelu = get_m_constant(s_out_lin1, 1.0, s_gelu_out * (1 << FRAC_BITS), k=k_bits)
    M_gelu = max(1, M_gelu)
    M_prime_gelu = int(np.floor((z_gelu_out * (1 << k_bits)) / M_gelu))

    numerator_gelu = M_gelu * (acc_gelu.astype(object) + M_prime_gelu)
    Q_gelu_out_sim = (numerator_gelu >> k_bits).astype(np.int64)
    R_gelu_out = (numerator_gelu % (1 << k_bits)).astype(np.int64)

    check_neg_torch(Q_gelu_out_sim, "Q_gelu_out_sim")
    check_neg_torch(R_gelu_out, "R_gelu_out")

    # ── Linear 2 (FFN down-projection) ───────────────────────────────────
    w_lin2 = debug['linear_weight_ff2']
    s_w_lin2, z_w_lin2 = get_quantization_params(w_lin2)
    s_out_lin2, z_out_lin2 = get_quantization_params(debug['linear_output_ff2'])

    Q_w_diff2 = (quantize_tensor(w_lin2, s_w_lin2, z_w_lin2) - z_w_lin2).astype(np.int64)
    Q_in_diff2 = (Q_gelu_out_sim - z_gelu_out).astype(np.int64)

    G2 = np.matmul(Q_in_diff2, Q_w_diff2.T)
    b_lin2 = debug.get('linear_bias_ff2', None)
    if b_lin2 is not None:
        s_b2 = s_w_lin2 * s_gelu_out
        Q_b_diff2 = np.round(b_lin2 / s_b2).astype(np.int64)
        acc2 = G2 + Q_b_diff2
    else:
        Q_b_diff2 = np.zeros_like(debug['linear_output_ff2'][..., 0:1])
        acc2 = G2

    M_lin2 = get_m_constant(s_w_lin2, s_gelu_out, s_out_lin2, k=k_bits)
    M_prime_lin2 = int(np.floor((z_out_lin2 * (1 << k_bits)) / M_lin2))
    numerator_lin2 = M_lin2 * (acc2 + M_prime_lin2)
    Q_out_sim_lin2 = (numerator_lin2 >> k_bits).astype(np.int32)
    R_out_lin2 = (numerator_lin2 % (1 << k_bits)).astype(np.int64)

    check_neg_torch(numerator_lin2, "numerator_lin2")
    check_neg_torch(Q_out_sim_lin2, "Q_out_sim_lin2")
    check_neg_torch(R_out_lin2, "R_out_lin2")

    # ── Block 2: post-FFN LayerNorm ──────────────────────────────────────
    block2 = _process_layernorm_block(
        x_real=debug['ln_input_2'],
        var_ref=debug['var_2'],
        gamma_real=debug['ln_weight_2'],
        beta_real=debug['ln_bias_2'],
        out_real=debug['ln_output_2'],
        k_bits=k_bits, frac_bits=FRAC_BITS, eps=eps, ln_table=ln_table,
    )

    # ── 최종 witness 저장 ────────────────────────────────────────────────
    out_dict = {
        "k_bits": np.uint32(k_bits),
        "frac_bits": np.uint32(FRAC_BITS),
        "eps": np.float64(eps),
        "var_scale": np.uint32(VAR_SCALE),  # Phase 4: 회로 측이 동일 scaling 사용 검증용
    }
    out_dict.update(_flatten_block("ln1", block1))
    out_dict.update(_flatten_block("ln2", block2))
    out_dict.update({
        # LIN1
        "z_in_lin1": np.uint64(z_in_lin1),
        "Q_in_diff_lin1": Q_in_diff1.astype(np.int32),
        "Q_w_diff_lin1": Q_w_diff1.astype(np.int32),
        "Q_b_diff_lin1": Q_b_diff1,
        "M_lin1": np.uint64(M_lin1),
        "M_prime_lin1": np.uint64(M_prime_lin1),
        "Q_out_sim_lin1": Q_out_sim_lin1,
        "R_out_lin1": R_out_lin1,
        # GELU
        "z_in_gelu": np.uint64(z_out_lin1),
        "sigmoid_part_gelu": sigmoid_part.astype(np.uint64),
        "acc_gelu": acc_gelu.astype(np.int64),
        "M_gelu": np.uint64(M_gelu),
        "M_prime_gelu": np.uint64(M_prime_gelu),
        "Q_out_sim_gelu": Q_gelu_out_sim,
        "R_out_gelu": R_gelu_out,
        "M_gelu_idx": np.uint64(M_gelu_idx),
        # LIN2
        "z_in_lin2": np.uint64(z_gelu_out),
        "Q_in_diff_lin2": Q_in_diff2.astype(np.int32),
        "Q_w_diff_lin2": Q_w_diff2.astype(np.int32),
        "Q_b_diff_lin2": Q_b_diff2,
        "M_lin2": np.uint64(M_lin2),
        "M_prime_lin2": np.uint64(M_prime_lin2),
        "Q_out_sim_lin2": Q_out_sim_lin2,
        "R_out_lin2": R_out_lin2,
        # 임베드된 표
        "layernorm_table": ln_table,
        "gelu_table": gelu_table,
    })

    np.savez_compressed(out_npz_path, **out_dict)
