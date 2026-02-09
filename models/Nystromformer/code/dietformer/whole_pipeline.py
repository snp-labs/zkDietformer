import hashlib
import numpy as np
from quantize import (
    get_quantization_params, quantize_tensor, 
    get_m_constant, sign_bit_grouping_matmul
)
from softmax_table import (
    generate_softmax_tables, 
    extract_exp_from_segments, combine_exp_limbs,
    FRAC_BITS, LIMB_BITS, K_SEGMENTS
)


def process_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=22):
    debug = np.load(debug_npz_path, allow_pickle=True)
    
    ### kernel 1
    Q_proc_real = debug['Q_proc']
    s_q, z_q = get_quantization_params(Q_proc_real)
    Q_q = quantize_tensor(Q_proc_real, s_q, z_q)
    
    K_proc_real = debug['K_proc']
    s_k, z_k = get_quantization_params(K_proc_real)
    Q_k = quantize_tensor(K_proc_real, s_k, z_k)

    B, H, S, D = Q_q.shape
    L = debug['Q_landmarks'].shape[2] 
    seg = S // L
    
    K_sum = Q_k.reshape(B, H, L, seg, D).sum(axis=-2)
    K_landmarks_q = K_sum // seg
    R_mean_k = K_sum % seg 

    Y_real = debug['kernel_1_scores']
    s_y1, z_y1 = get_quantization_params(Y_real)
    z_y1 += 10  # 안전 마진

    K_landmarks_T = np.swapaxes(K_landmarks_q, -1, -2)
    
    M_k1 = get_m_constant(s_q, s_k, s_y1, k=k_bits)
    
    Q_y1_sim, R_matmul_k1, M_prime_k1= sign_bit_grouping_matmul(
        Q_q, z_q, K_landmarks_T, z_k, z_y1, M_k1, k_bits
    )

    ### kernel 2
    Q_sum = Q_q.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_q = Q_sum // seg
    R_mean_q = Q_sum % seg

    Y2_real = debug['kernel_2_scores']
    s_y2, z_y2 = get_quantization_params(Y2_real)
    z_y2 += 10  # 안전 마진
    
    K_landmarks_T = np.swapaxes(K_landmarks_q, -1, -2)
    M_k2 = get_m_constant(s_q, s_k, s_y2, k=k_bits)
    
    Q_y2_sim, R_matmul_k2, M_prime_k2 = sign_bit_grouping_matmul(
        Q_landmarks_q, z_q, K_landmarks_T, z_k, z_y2, M_k2, k_bits
    )

    ### kernel 3
    Y3_real = debug['kernel_3_scores']
    s_y3, z_y3 = get_quantization_params(Y3_real)
    z_y3 += 10  # 안전 마진

    K_T = np.swapaxes(Q_k, -1, -2)
    
    M_k3 = get_m_constant(s_q, s_k, s_y3, k=k_bits)
    
    Q_sim_k3, R_matmul_k3, M_prime_k3 = sign_bit_grouping_matmul(
        Q_landmarks_q, z_q, K_T, z_k, z_y3, M_k3, k_bits
    )

    ### softmax tables 생성
    table_path = generate_softmax_tables("outputs", s_q, s_k, k_bits, FRAC_BITS)
    sm_tables = np.load(table_path)['tables'] 

    ### softmax(kernel)
    limbs_k1, exp_limbs_k1 = extract_exp_from_segments(Q_y1_sim, R_matmul_k1, sm_tables, k_bits)
    exp_total_k1, Q_exp_k1, R_exp_k1 = combine_exp_limbs(exp_limbs_k1, FRAC_BITS)
    exp_sum_k1 = np.sum(exp_total_k1, axis=-1, keepdims=True)
    P_k1 = (exp_total_k1 << FRAC_BITS) // exp_sum_k1
    R_div_k1 = (exp_total_k1 << FRAC_BITS) % exp_sum_k1

    ### softmax(kernel2) 
    limbs_k2, exp_limbs_k2 = extract_exp_from_segments(Q_y2_sim, R_matmul_k2, sm_tables, k_bits)
    exp_total_k2, Q_exp_k2, R_exp_k2 = combine_exp_limbs(exp_limbs_k2, FRAC_BITS)
    exp_sum_k2 = np.sum(exp_total_k2, axis=-1, keepdims=True)
    P_k2 = (exp_total_k2 << FRAC_BITS) // exp_sum_k2
    R_div_k2 = (exp_total_k2 << FRAC_BITS) % exp_sum_k2

    ### softmax(kernel3) 
    mask = debug['mask'] # [B, S]
    mask_4d = mask[:, None, None, :].astype(np.int64) # [B, 1, 1, S]
    Q_sim_k3_masked = Q_sim_k3 * mask_4d

    limbs_k3, exp_limbs_k3 = extract_exp_from_segments(Q_sim_k3_masked, R_matmul_k3, sm_tables, k_bits)
    exp_total_k3, Q_exp_k3, R_exp_k3 = combine_exp_limbs(exp_limbs_k3, FRAC_BITS)
    exp_sum_k3 = np.sum(exp_total_k3, axis=-1, keepdims=True)
    P_k3 = (exp_total_k3 << FRAC_BITS) // exp_sum_k3
    R_div_k3 = (exp_total_k3 << FRAC_BITS) % exp_sum_k3

    ### Pseudoinverse
    inv_k2_real = debug['inv_kernel_2']
    s_inv, z_inv = get_quantization_params(inv_k2_real)
    Q_inv_k2 = quantize_tensor(inv_k2_real, s_inv, z_inv).astype(np.int64)

    # A는 Softmax 결과인 P_k2를 사용
    A_q = P_k2.astype(np.int64)
    z_p2 = 0 
    s_p2 = 1.0 / (1 << FRAC_BITS) 

    ### [조건 1] A * A+ * A = A
    # 1-1. Y1 = A @ A+ 
    M_inv1 = get_m_constant(s_p2, s_inv, s_p2, k=k_bits)
    Q_y1, R_inv1, M_prime_inv1 = sign_bit_grouping_matmul(
        A_q, z_p2, Q_inv_k2, z_inv, z_p2, M_inv1, k_bits
    )
    
    # 1-2. Q_y2 = Y1 @ A (최종적으로 A_q와 값이 비슷해야 함)
    M_inv2 = get_m_constant(s_p2, s_p2, s_p2, k=k_bits)
    Q_y2, R_inv2, M_prime_inv2 = sign_bit_grouping_matmul(
        Q_y1, z_p2, A_q, z_p2, z_p2, M_inv2, k_bits 
    )

    ### [조건 2] A+ * A * A+ = A+
    # 2-1. Y3 = A+ @ A
    M_inv3 = get_m_constant(s_inv, s_p2, s_inv, k=k_bits)
    Q_y3, R_inv3, M_prime_inv3 = sign_bit_grouping_matmul(
        Q_inv_k2, z_inv, A_q, z_p2, z_inv, M_inv3, k_bits
    )
    
    # 2-2. Q_y4 = Y3 @ A+ 
    M_inv4 = get_m_constant(s_inv, s_inv, s_inv, k=k_bits)
    Q_y4, R_inv4, M_prime_inv4 = sign_bit_grouping_matmul(
        Q_y3, z_inv, Q_inv_k2, z_inv, z_inv, M_inv4, k_bits
    )

    ### temp
    V_real = debug['V']
    s_v, z_v = get_quantization_params(V_real)
    Q_v = quantize_tensor(V_real, s_v, z_v)

    temp_real = debug['temp']
    s_sim_temp, z_sim_temp = get_quantization_params(temp_real)
    
    z_pk3 = 0
    s_pk3 = 1.0 / (1 << FRAC_BITS)

    M_temp = get_m_constant(s_pk3, s_v, s_sim_temp, k=k_bits)
    
    Q_sim_temp, R_matmul_temp, M_prime_temp = sign_bit_grouping_matmul(
        P_k3.astype(np.int64), z_pk3, Q_v.astype(np.int64), z_v, z_sim_temp, M_temp, k_bits
    )

    ### mid
    mid_real = debug['mid']
    s_mid, z_sim_mid = get_quantization_params(mid_real)
    
    z_pk1 = 0
    s_pk1 = 1.0 / (1 << FRAC_BITS)

    M_mid = get_m_constant(s_pk1, s_inv, s_mid, k=k_bits)
    
    Q_sim_mid, R_matmul_mid, M_prime_mid = sign_bit_grouping_matmul(
        P_k1.astype(np.int64), z_pk1, Q_inv_k2.astype(np.int64), z_inv, z_sim_mid, M_mid, k_bits
    )

    ### X
    X_real = debug['X']
    s_x, z_sim_x = get_quantization_params(X_real)
    
    M_x = get_m_constant(s_mid, s_sim_temp, s_x, k=k_bits)
    
    Q_sim_x, R_matmul_x, M_prime_x = sign_bit_grouping_matmul(
        Q_sim_mid.astype(np.int64), z_sim_mid, Q_sim_temp.astype(np.int64), z_sim_temp, z_sim_x, M_x, k_bits
    )

    np.savez_compressed(
        out_npz_path,
        ### 공통 입력
        Q_q=Q_q.astype(np.int64), 
        z_q=np.int64(z_q),
        Q_k=Q_k.astype(np.int64), 
        z_k=np.int64(z_k),
        ### kernel 1
        R_mean_k1=R_mean_k.astype(np.int64),
        Q_sim_k1=Q_y1_sim.astype(np.int64),
        R_matmul_k1=R_matmul_k1.astype(np.int64),
        M_k1=np.int64(M_k1),
        M_prime_k1=np.int64(M_prime_k1), 
        z_sim_k1=np.int64(z_y1),
        ### kernel 2
        R_mean_k2=R_mean_q.astype(np.int64),
        Q_sim_k2=Q_y2_sim.astype(np.int64),
        R_matmul_k2=R_matmul_k2.astype(np.int64),
        M_k2=np.int64(M_k2),
        M_prime_k2=np.int64(M_prime_k2),
        z_sim_k2=np.int64(z_y2),
        ### kernel 3
        Q_sim_k3=Q_sim_k3.astype(np.int64),
        R_matmul_k3=R_matmul_k3.astype(np.int64),
        M_k3=np.int64(M_k3),
        M_prime_k3=np.int64(M_prime_k3),
        z_sim_k3=np.int64(z_y3),
        ### softmax(kernel1)
        limbs_k1=limbs_k1.astype(np.int64), 
        exp_limbs_k1=exp_limbs_k1.astype(np.uint64),
        Q_exp_k1=np.stack(Q_exp_k1, axis=0).astype(np.uint64),
        R_exp_k1=np.stack(R_exp_k1, axis=0).astype(np.uint64),
        exp_total_k1=exp_total_k1.astype(np.uint64),
        exp_sum_k1=exp_sum_k1.astype(np.uint64),
        P_k1=P_k1.astype(np.uint64),
        R_div_k1=R_div_k1.astype(np.uint64),
        ### softmax(kernel2)
        limbs_k2=limbs_k2.astype(np.int64), 
        exp_limbs_k2=exp_limbs_k2.astype(np.uint64),
        Q_exp_k2=np.stack(Q_exp_k2, axis=0).astype(np.uint64),
        R_exp_k2=np.stack(R_exp_k2, axis=0).astype(np.uint64),
        exp_total_k2=exp_total_k2.astype(np.uint64),
        exp_sum_k2=exp_sum_k2.astype(np.uint64),
        P_k2=P_k2.astype(np.uint64),
        R_div_k2=R_div_k2.astype(np.uint64),
        ### softmax(kernel3)
        limbs_k3=limbs_k3.astype(np.int64), 
        exp_limbs_k3=exp_limbs_k3.astype(np.uint64),
        Q_exp_k3=np.stack(Q_exp_k3, axis=0).astype(np.uint64),
        R_exp_k3=np.stack(R_exp_k3, axis=0).astype(np.uint64), 
        exp_total_k3=exp_total_k3.astype(np.uint64),
        P_k3=P_k3.astype(np.uint64), 
        R_div_k3=R_div_k3.astype(np.uint64),
        ### moore-penrose (Pseudoinverse)
        Q_inv_k2=Q_inv_k2.astype(np.int64),
        z_inv_k2=np.int64(z_inv),
        M_inv_all=np.array([M_inv1, M_inv2, M_inv3, M_inv4], dtype=np.int64),
        M_prime_inv_all=np.array([M_prime_inv1, M_prime_inv2, M_prime_inv3, M_prime_inv4], dtype=np.int64),
        Q_y_inv_intermediates=np.stack([Q_y1, Q_y2, Q_y3, Q_y4], axis=0).astype(np.int64),
        R_inv_remainders=np.stack([R_inv1, R_inv2, R_inv3, R_inv4], axis=0).astype(np.int64),
        ### temp
        Q_v=Q_v.astype(np.int64),
        z_v=np.int64(z_v),
        Q_sim_temp=Q_sim_temp.astype(np.int64),
        R_matmul_temp=R_matmul_temp.astype(np.int64),
        M_temp=np.int64(M_temp),
        M_prime_temp=np.int64(M_prime_temp),
        z_sim_temp=np.int64(z_sim_temp),
        ### mid = P_k1 @ Q_inv_k2
        Q_sim_mid=Q_sim_mid.astype(np.int64),
        R_matmul_mid=R_matmul_mid.astype(np.int64),
        M_mid=np.int64(M_mid),
        M_prime_mid=np.int64(M_prime_mid),
        z_sim_mid=np.int64(z_sim_mid),
        ### X = mid @ temp
        Q_sim_x=Q_sim_x.astype(np.int64),
        R_matmul_x=R_matmul_x.astype(np.int64),
        M_x=np.int64(M_x),
        M_prime_x=np.int64(M_prime_x),
        z_sim_x=np.int64(z_sim_x),
        ### 추가 정보
        sm_tables=sm_tables,
        frac_bits=np.int32(FRAC_BITS),
        limb_bits=np.int32(LIMB_BITS),
        k_segments=np.int32(K_SEGMENTS),
        seg_size=np.int64(seg),
        k_bits=np.int32(k_bits),
        mask=mask.astype(np.int64)
    )
    