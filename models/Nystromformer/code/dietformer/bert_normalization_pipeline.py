import os
import numpy as np
from quantize import (
    get_quantization_params, quantize_tensor, 
    get_m_constant, get_rms_index_multiplier, get_gelu_index_multiplier, check_neg_torch
)
from rms_table import generate_rmsnorm_tables, extract_inv_rms_single, FRAC_BITS
from gelu_table import generate_gelu_sigmoid_tables, extract_gelu_sigmoid_single

def process_standard_normalization_debug_npz(debug_npz_path, out_npz_path, k_bits=22):
    """
    BERT의 Feed-Forward Network 및 RMSNorm 중간값을 읽어와 
    Nystromformer와 동일한 규격의 Witness 데이터로 변환합니다.
    """
    debug = np.load(debug_npz_path, allow_pickle=True)
    
    ### 1. RMSNorm 1 (Attention 출력 이후)
    x_ri1 = debug['rms_input']
    s_ri1, z_ri1 = get_quantization_params(x_ri1)
    Q_ri1 = quantize_tensor(x_ri1, s_ri1, z_ri1).astype(np.int64)
    
    ms_real1 = debug['ms']
    dim1 = x_ri1.shape[-1]
    s_ms1, z_ms1 = get_quantization_params(ms_real1)

    M_ms1 = get_m_constant(s_ri1, s_ri1, s_ms1, k=k_bits)
    M_prime_ms1 = int(np.floor((z_ms1 * (1 << k_bits)) / M_ms1))
    
    Q_diff1 = Q_ri1 - z_ri1
    sum_sq1 = np.sum(Q_diff1**2, axis=-1, keepdims=True)
    
    numerator1 = M_ms1 * (sum_sq1 + M_prime_ms1)
    shifted_val1 = numerator1 >> k_bits
    R_shift1 = numerator1 % (1 << k_bits) 
    
    Q_ms_sim1 = shifted_val1 // dim1 
    R_ms1 = shifted_val1 % dim1  

    # RMSNorm Tables 로드 (Nystrom과 공유)
    table_dir = "outputs_bert"
    table_path = generate_rmsnorm_tables(table_dir, k_bits=k_bits, frac_bits=FRAC_BITS)
    rms_table = np.load(table_path)['table'] 

    M_rms1 = get_rms_index_multiplier(s_ms1, k_bits=k_bits)
    inv_rms_total1 = extract_inv_rms_single(Q_ms_sim1, R_ms1, dim1, rms_table, M_rms1, k_bits)
    
    gamma1 = debug['rms_weight']
    s_gamma1, z_gamma1 = get_quantization_params(gamma1)
    Q_gamma1 = quantize_tensor(gamma1, s_gamma1, z_gamma1).astype(np.int64)

    s_out_norm1, z_out_norm1 = get_quantization_params(debug['rms_output']) 
    M_norm1 = get_m_constant(s_ri1, s_gamma1, s_out_norm1 * (1 << FRAC_BITS), k=k_bits)
    M_prime_norm1 = int(np.floor((z_out_norm1 * (1 << k_bits)) / M_norm1))

    norm_no_gamma1 = Q_diff1.astype(object) * inv_rms_total1.astype(object)
    final_int_product1 = norm_no_gamma1 * (Q_gamma1 - z_gamma1).astype(object)

    numerator_norm1 = M_norm1 * (final_int_product1 + M_prime_norm1)
    Q_rms_output_sim1 = (numerator_norm1 >> k_bits).astype(np.int64)
    R_rms_output1 = (numerator_norm1 % (1 << k_bits)).astype(np.int64)

    ### 2. Linear 1 (FFN Up-projection)
    w_lin1 = debug['linear_weight']
    out_lin1 = debug['linear_output'] 
    b_lin1 = debug.get('linear_bias', None)

    s_in_lin1, z_in_lin1 = get_quantization_params(debug['rms_output']) 
    s_w_lin1, z_w_lin1 = get_quantization_params(w_lin1)
    s_out_lin1, z_out_lin1 = get_quantization_params(out_lin1)

    Q_w_diff1 = (quantize_tensor(w_lin1, s_w_lin1, z_w_lin1) - z_w_lin1).astype(np.int64)
    Q_in_diff1 = (Q_rms_output_sim1 - z_in_lin1).astype(np.int64)

    acc1 = np.matmul(Q_in_diff1, Q_w_diff1.T)
    if b_lin1 is not None:
        s_b1 = s_w_lin1 * s_in_lin1
        Q_b_diff1 = np.round(b_lin1 / s_b1).astype(np.int64)
        acc1 += Q_b_diff1
    else:
        Q_b_diff1 = np.zeros_like(out_lin1[..., 0:1])

    M_lin1 = get_m_constant(s_w_lin1, s_out_norm1, s_out_lin1, k=k_bits)
    M_prime_lin1 = int(np.floor((z_out_lin1 * (1 << k_bits)) / M_lin1))
    
    numerator_lin1 = M_lin1 * (acc1 + M_prime_lin1)
    Q_out_sim_lin1 = (numerator_lin1 >> k_bits).astype(np.int32)
    R_out_lin1 = (numerator_lin1 % (1 << k_bits)).astype(np.int64)

    ### 3. GeLU Activation
    gelu_table_path = generate_gelu_sigmoid_tables(table_dir, k_bits=k_bits, frac_bits=FRAC_BITS)
    gelu_table = np.load(gelu_table_path)['table']
    M_gelu_idx = get_gelu_index_multiplier(s_out_lin1, k_bits=k_bits)

    sigmoid_part = extract_gelu_sigmoid_single(
        Q_out_sim_lin1, R_out_lin1, z_out_lin1, gelu_table, M_gelu_idx, k_bits
    )
    acc_gelu = (Q_out_sim_lin1 - z_out_lin1).astype(object) * sigmoid_part.astype(object)
    s_gelu_out, z_gelu_out = get_quantization_params(debug['gelu_output'])

    M_gelu = get_m_constant(s_out_lin1, 1.0, s_gelu_out * (1 << FRAC_BITS), k=k_bits)
    M_prime_gelu = int(np.floor((z_gelu_out * (1 << k_bits)) / M_gelu))

    numerator_gelu = M_gelu * (acc_gelu.astype(object) + M_prime_gelu)
    Q_gelu_out_sim = (numerator_gelu >> k_bits).astype(np.int64)
    R_gelu_out = (numerator_gelu % (1 << k_bits)).astype(np.int64)

    ### 4. Linear 2 (FFN Down-projection)
    w_lin2 = debug['linear_weight_ff2']
    s_w_lin2, z_w_lin2 = get_quantization_params(w_lin2)
    s_out_lin2, z_out_lin2 = get_quantization_params(debug['linear_output_ff2'])

    Q_w_diff2 = (quantize_tensor(w_lin2, s_w_lin2, z_w_lin2) - z_w_lin2).astype(np.int64)
    Q_in_diff2 = (Q_gelu_out_sim - z_gelu_out).astype(np.int64)

    acc2 = np.matmul(Q_in_diff2, Q_w_diff2.T)
    b_lin2 = debug.get('linear_bias_ff2', None)
    if b_lin2 is not None:
        s_b2 = s_w_lin2 * s_gelu_out
        Q_b_diff2 = np.round(b_lin2 / s_b2).astype(np.int64)
        acc2 += Q_b_diff2
    else:
        Q_b_diff2 = np.zeros_like(debug['linear_output_ff2'][..., 0:1])

    M_lin2 = get_m_constant(s_w_lin2, s_gelu_out, s_out_lin2, k=k_bits)
    M_prime_lin2 = int(np.floor((z_out_lin2 * (1 << k_bits)) / M_lin2))
    
    numerator_lin2 = M_lin2 * (acc2 + M_prime_lin2)
    Q_out_sim_lin2 = (numerator_lin2 >> k_bits).astype(np.int32)
    R_out_lin2 = (numerator_lin2 % (1 << k_bits)).astype(np.int64)

    ### 5. RMSNorm 2 (FFN 출력 이후)
    x_ri2 = debug['rms_input_rms2']
    s_ri2, z_ri2 = get_quantization_params(x_ri2)
    Q_ri2 = quantize_tensor(x_ri2, s_ri2, z_ri2).astype(np.int64)
    
    ms_real2 = debug['ms_rms2']
    dim2 = x_ri2.shape[-1]
    s_ms2, z_ms2 = get_quantization_params(ms_real2)

    M_ms2 = get_m_constant(s_ri2, s_ri2, s_ms2, k=k_bits)
    M_prime_ms2 = int(np.floor((z_ms2 * (1 << k_bits)) / M_ms2))
    
    Q_diff2 = Q_ri2 - z_ri2
    sum_sq2 = np.sum(Q_diff2**2, axis=-1, keepdims=True)
    
    numerator2 = M_ms2 * (sum_sq2 + M_prime_ms2)
    shifted_val2 = numerator2 >> k_bits
    R_shift2 = numerator2 % (1 << k_bits)
    
    Q_ms_sim2 = shifted_val2 // dim2
    R_ms2 = shifted_val2 % dim2  

    M_rms2 = get_rms_index_multiplier(s_ms2, k_bits=k_bits)
    inv_rms_total2 = extract_inv_rms_single(Q_ms_sim2, R_ms2, dim2, rms_table, M_rms2, k_bits)
    
    gamma2 = debug['rms_weight_rms2']
    s_gamma2, z_gamma2 = get_quantization_params(gamma2)
    Q_gamma2 = quantize_tensor(gamma2, s_gamma2, z_gamma2).astype(np.int64)

    s_out_norm2, z_out_norm2 = get_quantization_params(debug['rms_output_rms2']) 
    M_norm2 = get_m_constant(s_ri2, s_gamma2, s_out_norm2 * (1 << FRAC_BITS), k=k_bits)
    M_prime_norm2 = int(np.floor((z_out_norm2 * (1 << k_bits)) / M_norm2))

    final_int_product2 = (Q_diff2.astype(object) * inv_rms_total2.astype(object)) * (Q_gamma2 - z_gamma2).astype(object)
    numerator_norm2 = M_norm2 * (final_int_product2 + M_prime_norm2)
    Q_rms_output_sim2 = (numerator_norm2 >> k_bits).astype(np.int64)
    R_rms_output2 = (numerator_norm2 % (1 << k_bits)).astype(np.int64)

    # 6. 최종 NPZ 저장 (Nystrom과 동일한 Key 규격 유지)
    np.savez_compressed(
        out_npz_path,
        k_bits=np.uint32(k_bits),
        frac_bits=np.uint32(FRAC_BITS),
        
        # RMS1 관련
        Q_ri_rms1=Q_ri1, z_ri_rms1=np.uint64(z_ri1),
        M_ms_rms1=np.uint64(M_ms1), M_prime_ms_rms1=np.uint64(M_prime_ms1),
        Q_ms_sim_rms1=Q_ms_sim1.astype(np.int64), R_ms_rms1=R_ms1.astype(np.int64),
        inv_rms_total_rms1=inv_rms_total1.astype(np.uint64),
        Q_gamma_rms1=Q_gamma1, z_gamma_rms1=np.uint64(z_gamma1),
        M_norm_rms1=np.uint64(M_norm1), M_prime_norm_rms1=np.uint64(M_prime_norm1),
        Q_rms_output_sim_rms1=Q_rms_output_sim1, R_rms_output_rms1=R_rms_output1,
        M_rms1=np.uint64(M_rms1),

        # LIN1 관련
        z_in_lin1=np.uint64(z_out_norm1), Q_in_diff_lin1=Q_in_diff1.astype(np.int32),
        Q_w_diff_lin1=Q_w_diff1.astype(np.int32), Q_b_diff_lin1=Q_b_diff1,
        M_lin1=np.uint64(M_lin1), M_prime_lin1=np.uint64(M_prime_lin1),
        Q_out_sim_lin1=Q_out_sim_lin1, R_out_lin1=R_out_lin1,

        # GELU 관련
        z_in_gelu=np.uint64(z_out_lin1), sigmoid_part_gelu=sigmoid_part.astype(np.uint64),
        M_gelu=np.uint64(M_gelu), M_prime_gelu=np.uint64(M_prime_gelu),
        Q_out_sim_gelu=Q_gelu_out_sim, R_out_gelu=R_gelu_out,
        M_gelu_idx=np.uint64(M_gelu_idx),

        # LIN2 관련
        z_in_lin2=np.uint64(z_gelu_out), Q_in_diff_lin2=Q_in_diff2.astype(np.int32),
        Q_w_diff_lin2=Q_w_diff2.astype(np.int32), Q_b_diff_lin2=Q_b_diff2,
        M_lin2=np.uint64(M_lin2), M_prime_lin2=np.uint64(M_prime_lin2),
        Q_out_sim_lin2=Q_out_sim_lin2, R_out_lin2=R_out_lin2,

        # RMS2 관련
        Q_ri_rms2=Q_ri2, z_ri_rms2=np.uint64(z_ri2),
        M_ms_rms2=np.uint64(M_ms2), M_prime_ms_rms2=np.uint64(M_prime_ms2),
        Q_ms_sim_rms2=Q_ms_sim2.astype(np.int64), R_ms_rms2=R_ms2.astype(np.int64),
        inv_rms_total_rms2=inv_rms_total2.astype(np.uint64),
        Q_gamma_rms2=Q_gamma2, z_gamma_rms2=np.uint64(z_gamma2),
        M_norm_rms2=np.uint64(M_norm2), M_prime_norm_rms2=np.uint64(M_prime_norm2),
        Q_rms_output_sim_rms2=Q_rms_output_sim2, R_rms_output_rms2=R_rms_output2,
        M_rms2=np.uint64(M_rms2),
        
        rms_table=rms_table,
        gelu_table=gelu_table
    )
    print(f"✅ BERT Normalization Witness saved to {out_npz_path}")