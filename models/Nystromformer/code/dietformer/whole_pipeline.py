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

    limbs_k1, exp_limbs_k1 = extract_exp_from_segments(Q_y1_sim, R_matmul_k1, sm_tables, k_bits)
    exp_total_k1, Q_exp_k1, R_exp_k1 = combine_exp_limbs(exp_limbs_k1, FRAC_BITS)
    exp_sum_k1 = np.sum(exp_total_k1, axis=-1, keepdims=True)
    P_k1 = (exp_total_k1 << FRAC_BITS) // exp_sum_k1
    R_div_k1 = (exp_total_k1 << FRAC_BITS) % exp_sum_k1

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
        ### 추가 정보
        sm_tables=sm_tables,
        frac_bits=np.int32(FRAC_BITS),
        limb_bits=np.int32(LIMB_BITS),
        k_segments=np.int32(K_SEGMENTS),
        seg_size=np.int64(seg),
        k_bits=np.int32(k_bits)
    )
    