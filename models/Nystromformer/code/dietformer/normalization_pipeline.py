import os
import numpy as np
from quantize import get_quantization_params, quantize_tensor, get_m_constant
from rms_table import generate_rmsnorm_tables, extract_inv_rms_single, FRAC_BITS

def process_normalization_debug_npz(debug_npz_path, out_npz_path, k_bits=22):
    debug = np.load(debug_npz_path, allow_pickle=True)
    
    x_ri = debug['rms_input']
    s_ri, z_ri = get_quantization_params(x_ri)
    Q_ri = quantize_tensor(x_ri, s_ri, z_ri).astype(np.int64)
    
    ms_real = debug['ms']
    dim = x_ri.shape[-1]
    s_ms, z_ms = get_quantization_params(ms_real) # 1/N까지 포함한 값

    M_ms = get_m_constant(s_ri, s_ri, s_ms, k=k_bits)
    M_prime_ms = int(np.floor((z_ms * (1 << k_bits)) / M_ms))
    
    Q_diff = Q_ri - z_ri
    sum_sq = np.sum(Q_diff**2, axis=-1, keepdims=True)
    
    numerator = M_ms * (sum_sq + M_prime_ms)
    shifted_val = numerator >> k_bits
    R_shift = numerator % (1 << k_bits) 
    
    Q_ms_sim = shifted_val // dim
    R_ms = shifted_val % dim  

    table_dir = "outputs"
    table_path = generate_rmsnorm_tables(table_dir, s_ri, k_bits=k_bits, frac_bits=FRAC_BITS)
    rms_table = np.load(table_path)['table'] 
    inv_rms_total = extract_inv_rms_single(Q_ms_sim, R_ms, rms_table, k_bits)
    
    gamma = debug['rms_weight']
    s_gamma, z_gamma = get_quantization_params(gamma)
    Q_gamma = quantize_tensor(gamma, s_gamma, z_gamma).astype(np.int64)


    norm_no_gamma = (Q_diff * inv_rms_total.astype(np.int64))
    final_product_full = norm_no_gamma * Q_gamma

    W_BITS = 8
    total_shift = FRAC_BITS + W_BITS
    Q_rms_output_sim = final_product_full >> total_shift
    R_rms_output = final_product_full % (1 << total_shift)

    np.savez_compressed(
        out_npz_path,
        Q_ri_rms=Q_ri.astype(np.int64),
        z_ri_rms=np.int64(z_ri),

        sum_sq_rms=sum_sq.astype(np.int64),
        M_ms_rms=np.int64(M_ms),
        M_prime_ms_rms=np.int64(M_prime_ms),
        
        shifted_val_rms=shifted_val.astype(np.int64),
        R_shift_rms=R_shift.astype(np.int64), 
        Q_ms_sim_rms=Q_ms_sim.astype(np.int64),
        R_ms_rms=R_ms.astype(np.int64),       
        

        Q_gamma_rms=Q_gamma.astype(np.int64),      
        z_gamma_rms=np.int64(z_gamma),             
        weight_bits=np.int32(W_BITS),   

        inv_rms_total_rms=inv_rms_total.astype(np.uint64),
        Q_rms_output_sim_rms=Q_rms_output_sim.astype(np.int64),
        R_rms_output_rms=R_rms_output.astype(np.int64),
        # 해당 값은 softmax쪽 재사용
        rms_table=rms_table,
        k_bits=np.int32(k_bits),
        frac_bits=np.int32(FRAC_BITS)
    )