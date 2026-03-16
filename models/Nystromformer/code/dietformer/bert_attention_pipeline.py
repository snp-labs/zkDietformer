import numpy as np
import os
from quantize import (
    get_quantization_params, quantize_tensor, 
    get_m_constant, sign_bit_grouping_matmul, get_softmax_index_multiplier, check_neg_torch
)
from softmax_table import (
    generate_softmax_tables, 
    extract_exp_from_segments, combine_exp_limbs,
    FRAC_BITS, LIMB_BITS, K_SEGMENTS
)

def process_standard_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=22):
    """
    BERT의 표준 Full-Attention 중간값을 읽어와 
    Nystromformer와 동일한 규격의 양자화 및 서킷 입력 데이터로 변환합니다.
    """
    debug = np.load(debug_npz_path, allow_pickle=True)
    
    # 1. Q, K, V 양자화 (Nystromformer와 동일한 함수 사용)
    Q_real = debug['Q_proc']
    s_q, z_q = get_quantization_params(Q_real)
    Q_q = quantize_tensor(Q_real, s_q, z_q).astype(np.int64)
    
    K_real = debug['K_proc']
    s_k, z_k = get_quantization_params(K_real)
    Q_k = quantize_tensor(K_real, s_k, z_k).astype(np.int64)

    V_real = debug['V']
    s_v, z_v = get_quantization_params(V_real)
    Q_v = quantize_tensor(V_real, s_v, z_v).astype(np.int64)

    # 2. Score 행렬곱 (Q * K^T) - O(S^2)
    # Nystromformer의 kernel_1_scores 대신 Full Score를 처리합니다.
    Y_real = np.matmul(Q_real, np.swapaxes(K_real, -1, -2))
    
    s_y, z_y = get_quantization_params(Y_real)
    z_y += 10

    K_T = np.swapaxes(Q_k, -1, -2)
    M_attn_scores = get_m_constant(s_q, s_k, s_y, k=k_bits) #
    
    # ZEN Matmul 적용 (Sign-bit grouping)
    Q_score_sim, R_matmul_score, M_prime_score = sign_bit_grouping_matmul(
        Q_q, z_q, K_T, z_k, z_y, M_attn_scores, k_bits
    )

    # 3. Softmax Tables 로드 (Nystromformer와 공유)
    table_path = generate_softmax_tables("outputs_bert", k_bits, FRAC_BITS)
    sm_tables = np.load(table_path)['tables'] 

    # 4. Softmax 처리
    # 마스크 적용
    mask = debug['mask']
    mask_4d = mask[:, None, None, :].astype(np.int64)
    Q_score_sim_masked = Q_score_sim * mask_4d # 0인 부분은 마스킹 처리

    M_sm = get_softmax_index_multiplier(s_q, s_k, k_bits) #
    
    # 5-Segment 분해 및 테이블 룩업
    limbs, exp_limbs, r_sm = extract_exp_from_segments(Q_score_sim_masked, R_matmul_score, sm_tables, M_sm, k_bits)
    
    # 지수값 결합 및 정규화 (P = exp / sum(exp))
    exp_total, Q_exp, R_exp = combine_exp_limbs(exp_limbs, FRAC_BITS)
    exp_sum = np.sum(exp_total, axis=-1, keepdims=True)
    
    P_attn = (exp_total << FRAC_BITS) // exp_sum
    R_div_attn = (exp_total << FRAC_BITS) % exp_sum

    # 5. 최종 출력 행렬곱 (P * V)
    X_real = debug['X']
    s_x, z_sim_x = get_quantization_params(X_real)
    
    z_p = 0 # Softmax 결과 영점
    s_p = 1.0 / (1 << FRAC_BITS) #

    M_final = get_m_constant(s_p, s_v, s_x, k=k_bits)
    
    Q_sim_x, R_matmul_x, M_prime_x = sign_bit_grouping_matmul(
        P_attn.astype(np.int64), z_p, Q_v, z_v, z_sim_x, M_final, k_bits
    )

    # 6. NPZ 저장 (Rust 회로에서 읽기 위한 포맷)
    np.savez_compressed(
        out_npz_path,
        # 공통 입력
        Q_q=Q_q, z_q=np.uint64(z_q),
        Q_k=Q_k, z_k=np.uint64(z_k),
        Q_v=Q_v, z_v=np.uint64(z_v),
        
        # Attention Scores (Full Matrix)
        Q_sim_attn=Q_score_sim.astype(np.int64),
        R_matmul_attn=R_matmul_score.astype(np.int64),
        M_attn=np.uint64(M_attn_scores),
        M_prime_attn=np.uint64(M_prime_score),
        z_sim_attn=np.uint64(z_y),
        
        # Softmax 결과
        M_sm=np.uint64(M_sm),
        limbs=limbs.astype(np.int64),
        exp_limbs=exp_limbs.astype(np.uint64),
        r_sm=r_sm.astype(np.int64),
        Q_exp=np.stack(Q_exp, axis=0).astype(np.uint64),
        R_exp=np.stack(R_exp, axis=0).astype(np.uint64),
        exp_total=exp_total.astype(np.uint64),
        P_attn=P_attn.astype(np.uint64),
        R_div_attn=R_div_attn.astype(np.uint64),
        
        # 최종 출력 X
        Q_sim_x=Q_sim_x.astype(np.int64),
        R_matmul_x=R_matmul_x.astype(np.int64),
        M_x=np.uint64(M_final),
        M_prime_x=np.uint64(M_prime_x),
        z_sim_x=np.uint64(z_sim_x),
        
        # 메타데이터 (Nystrom과 동일)
        sm_tables=sm_tables,
        frac_bits=np.uint32(FRAC_BITS),
        limb_bits=np.uint32(LIMB_BITS),
        k_segments=np.uint32(K_SEGMENTS),
        k_bits=np.uint32(k_bits),
        mask=mask.astype(np.int64)
    )
    print(f"✅ BERT Standard Attention Witness saved to {out_npz_path}")