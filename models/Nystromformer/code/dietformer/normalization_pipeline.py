import os
import numpy as np
from quantize import get_quantization_params, quantize_tensor, get_m_constant, sign_bit_grouping_matmul
from rms_table import generate_rmsnorm_tables, extract_inv_rms_single, FRAC_BITS
from gelu_table import generate_gelu_sigmoid_tables, extract_gelu_sigmoid_single

def process_normalization_debug_npz(debug_npz_path, out_npz_path, k_bits=22):
    debug = np.load(debug_npz_path, allow_pickle=True)
    ### RMSNorm1
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
    #final_product_full = norm_no_gamma * Q_gamma
    final_product_full = norm_no_gamma * (Q_gamma - z_gamma)

    W_BITS = 8
    total_shift = FRAC_BITS + W_BITS
    Q_rms_output_sim = (final_product_full >> total_shift).astype(np.int64)
    R_rms_output = final_product_full % (1 << total_shift)

    ### Linear1
    w_lin = debug['linear_weight']
    out_lin = debug['linear_output'] 
    b_lin = debug.get('linear_bias', None)

    # 파라미터 추출
    s_in_lin, z_in_lin = get_quantization_params(debug['rms_output']) 
    s_w_lin, z_w_lin = get_quantization_params(w_lin)
    s_out_lin, z_out_lin = get_quantization_params(out_lin)

    # 1. Zero-centering (수식의 Q_w - z_w, Q_in - z_in 부분)
    Q_w_diff = (quantize_tensor(w_lin, s_w_lin, z_w_lin) - z_w_lin).astype(np.int64)
    Q_in_diff = (Q_rms_output_sim - z_in_lin).astype(np.int64)

    # 2. Pure MatMul (수식의 (Q_w - z_w)(Q_in - z_in) 부분)
    # G는 s_w * s_in 스케일을 가짐
    G = np.matmul(Q_in_diff, Q_w_diff.T)

    # 3. Bias 합산 (수식의 + (Q_b - z_b) 부분)
    if b_lin is not None:
        s_b = s_w_lin * s_in_lin
        # z_b는 0으로 가정 (Bias는 보통 영점이 없음)
        Q_b_diff = np.round(b_lin / s_b).astype(np.int64)
        acc = G + Q_b_diff
    else:
        acc = G

    # 4. Rescaling & Offset (수식의 M * [...] + z_out 부분)
    M_lin = get_m_constant(s_w_lin, s_in_lin, s_out_lin, k=k_bits)
    # M_prime은 z_out을 M의 분자 영역으로 끌어올린 보정치
    M_prime_lin = int(np.floor((z_out_lin * (1 << k_bits)) / M_lin))

    # 최종 분자 계산 (M * [acc + M_prime])
    numerator_lin = M_lin * (acc + M_prime_lin)
    
    # 시프트 및 나머지 추출 (enforce_equal용)
    Q_out_sim_lin = (numerator_lin >> k_bits).astype(np.int32)
    R_out_lin = (numerator_lin % (1 << k_bits)).astype(np.int64)

    # Gelu
    gelu_table_path = generate_gelu_sigmoid_tables("outputs", s_out_lin, k_bits=k_bits, frac_bits=FRAC_BITS)
    gelu_table = np.load(gelu_table_path)['table']

    s_gelu_in = s_out_lin   # Linear1의 출력이 GeLU의 입력
    z_gelu_in = z_out_lin
    s_gelu_out, z_gelu_out = get_quantization_params(debug['gelu_output'])

    sigmoid_part = extract_gelu_sigmoid_single(Q_out_sim_lin, R_out_lin, z_gelu_in, gelu_table, k_bits)


    # 3. 알맹이 계산 (acc = (Q_in - z_in) * Sigmoid_Table_Val)
    acc_gelu = (Q_out_sim_lin - z_out_lin).astype(object) * sigmoid_part.astype(object)

    # 4. GeLU 전용 Multiplier 및 Offset 계산 (Linear와 동일 로직)
    # M = (s_in / (s_out * 2^FRAC_BITS)) * 2^k
    s_gelu_in = s_out_lin
    M_gelu = get_m_constant(
        s_gelu_in, 
        1.0, 
        s_gelu_out * (1 << FRAC_BITS), 
        k=k_bits
    )
    M_prime_gelu = int(np.floor((z_gelu_out * (1 << k_bits)) / M_gelu))

    # 5. 최종 분자 계산 및 시프트
    numerator_gelu = M_gelu * (acc_gelu + M_prime_gelu)
    Q_gelu_out_sim = (numerator_gelu >> k_bits).astype(np.int32)
    R_gelu_out = (numerator_gelu % (1 << k_bits)).astype(np.int64)


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
        Q_rms_output_sim_rms=Q_rms_output_sim,
        R_rms_output_rms=R_rms_output.astype(np.int64),

        # Linear
        # 1. 입력 및 파라미터 Witness
        z_in_lin = np.int64(z_in_lin),
        Q_in_diff_lin = Q_in_diff.astype(np.int32),   # (Q_in - z_in)
        Q_w_diff_lin  = Q_w_diff.astype(np.int32),    # (Q_w - z_w)
        Q_b_diff_lin  = Q_b_diff.astype(np.int64) if b_lin is not None else np.zeros(1), # (Q_b - z_b)
        
        # 2. 중간 연산 결과 (서킷의 단계별 검증 대상)
        G_lin         = G.astype(np.int64),           # MatMul 결과: (Q_in_diff * Q_w_diff)
        acc_lin       = acc.astype(np.int64),         # 대괄호 안의 합계: G + Q_b_diff
        
        # 3. 재스케일링 및 오프셋 Witness
        M_lin         = np.int64(M_lin),              # 재스케일링 상수
        M_prime_lin   = np.int64(M_prime_lin),        # z_out 보정 상수
        numerator_lin = numerator_lin.astype(object), # M * (acc + M_prime) -> 매우 큰 값 대비용
        
        # 4. 최종 결과 및 나눗셈 증명 파편 (enforce_equal의 핵심)
        Q_out_sim_lin = Q_out_sim_lin.astype(np.int32), # 최종 몫 (Quotient)
        R_out_lin     = R_out_lin.astype(np.int64),     # 시프트 나머지 (Remainder)
        
        # 해당 값은 softmax쪽 재사용
        rms_table=rms_table,
        k_bits=np.int32(k_bits),
        frac_bits=np.int32(FRAC_BITS),

        #gelu
        sigmoid_part_gelu = sigmoid_part.astype(np.uint64),
        Q_out_sim_gelu    = Q_gelu_out_sim,
        R_out_gelu        = R_gelu_out,
        gelu_table        = gelu_table,
        z_in_gelu=np.int64(z_gelu_in),
        acc_gelu=acc_gelu.astype(np.int64),
        M_gelu=np.int64(M_gelu),
        M_prime_gelu=np.int64(M_prime_gelu),
    )