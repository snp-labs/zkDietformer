import os
import numpy as np
from quantize import get_quantization_params, quantize_tensor, get_m_constant, get_rms_index_multiplier, get_gelu_index_multiplier
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
    
    Q_ms_sim = shifted_val // dim # 1/N
    R_ms = shifted_val % dim  

    table_dir = "outputs"
    table_path = generate_rmsnorm_tables(table_dir, k_bits=k_bits, frac_bits=FRAC_BITS)
    rms_table = np.load(table_path)['table'] 

    M_rms = get_rms_index_multiplier(s_ms, k_bits=k_bits)
    inv_rms_total = extract_inv_rms_single(Q_ms_sim, R_ms, dim, rms_table, M_rms, k_bits)
    
    gamma = debug['rms_weight']
    s_gamma, z_gamma = get_quantization_params(gamma)
    Q_gamma = quantize_tensor(gamma, s_gamma, z_gamma).astype(np.int64)

    s_out_norm, z_out_norm = get_quantization_params(debug['rms_output']) 
    M_norm = get_m_constant(s_ri, s_gamma, s_out_norm * (1 << FRAC_BITS), k=k_bits)
    M_prime_norm = int(np.floor((z_out_norm * (1 << k_bits)) / M_norm))

    norm_no_gamma = Q_diff.astype(object) * inv_rms_total.astype(object)
    final_int_product = norm_no_gamma * (Q_gamma - z_gamma).astype(object)

    numerator_norm = M_norm * (final_int_product + M_prime_norm)
    Q_rms_output_sim = (numerator_norm >> k_bits).astype(np.int64)
    R_rms_output = (numerator_norm % (1 << k_bits)).astype(np.int64)

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
    gelu_table_path = generate_gelu_sigmoid_tables("outputs", k_bits=k_bits, frac_bits=FRAC_BITS)
    gelu_table = np.load(gelu_table_path)['table']
    M_gelu_idx = get_gelu_index_multiplier(s_out_lin, k_bits=k_bits)

    # 2. 테이블 조회 (시그모이드 파트 추출)
    sigmoid_part = extract_gelu_sigmoid_single(
        Q_out_sim_lin, R_out_lin, z_out_lin, gelu_table, M_gelu_idx, k_bits
    )

    # 3. x * sigmoid(x) 계산
    # (Q_out_sim_lin - z_out_lin)이 수식의 x 역할을 수행
    acc_gelu = (Q_out_sim_lin - z_out_lin).astype(object) * sigmoid_part.astype(object)

    # 4. 최종 재스케일링 및 오프셋 (total_shift 적용)
    s_gelu_out, z_gelu_out = get_quantization_params(debug['gelu_output'])

    # M_gelu는 순수하게 입력/출력 스케일 비율만 계산 (분모에 FRAC_BITS 포함)
    M_gelu = get_m_constant(s_out_lin, 1.0, s_gelu_out * (1 << FRAC_BITS), k=k_bits)
    M_gelu = max(1, M_gelu)

    # 오프셋 계산 시 total_shift가 아닌 k_bits만 사용해야 M_gelu 내부의 FRAC_BITS와 정렬됨
    # 혹은 M_gelu에서 FRAC_BITS를 빼고 total_shift를 사용하는 방식으로 통일 필요
    # 여기서는 M_gelu에 포함시켰으므로 k_bits만 시프트합니다.
    M_prime_gelu = int(np.floor((z_gelu_out * (1 << k_bits)) / M_gelu))

    # [중요] 계산 시 반드시 .astype(object)를 사용하여 오버플로 방지
    numerator_gelu = M_gelu * (acc_gelu.astype(object) + M_prime_gelu)

    # 최종 시프트는 M_gelu가 이미 분모에서 16비트를 갉아먹었으므로 k_bits만 수행
    Q_gelu_out_sim = (numerator_gelu >> k_bits).astype(np.int64)
    R_gelu_out = (numerator_gelu % (1 << k_bits)).astype(np.int64)

    np.savez_compressed(
        out_npz_path,
        # --- [1] 입력 및 기본 정보 ---
        Q_ri_rms=Q_ri.astype(np.int64),
        z_ri_rms=np.uint64(z_ri),
        sum_sq_rms=sum_sq.astype(np.int64),
        
        # --- [2] Mean Square 계산 및 시프트 검증 (Verification 1) ---
        M_ms_rms=np.uint64(M_ms),
        M_prime_ms_rms=np.uint64(M_prime_ms),
        shifted_val_rms=shifted_val.astype(np.int64),
        R_shift_rms=R_shift.astype(np.int64), 
        
        # --- [3] 평균(1/N) 나눗셈 검증 (Verification 2) ---
        Q_ms_sim_rms=Q_ms_sim.astype(np.int64),
        R_ms_rms=R_ms.astype(np.int64),
        
        # --- [4] 고정 테이블 룩업 및 인덱스 배수 ---
        M_rms=np.uint64(M_rms),            # 테이블 인덱스 변환용 배수
        inv_rms_total_rms=inv_rms_total.astype(np.uint64), # 룩업 결과값
        
        # --- [5] 최종 출력 및 재스케일링 (Verification 3) ---
        Q_gamma_rms=Q_gamma.astype(np.int64),
        z_gamma_rms=np.uint64(z_gamma),
        M_norm_rms=np.uint64(M_norm),          # 최종 출력을 위한 멀티플라이어
        M_prime_norm_rms=np.uint64(M_prime_norm), # 최종 출력을 위한 오프셋
        Q_rms_output_sim_rms=Q_rms_output_sim.astype(np.int64), # 최종 몫
        R_rms_output_rms=R_rms_output.astype(np.int64),         # 최종 나머지

        # 해당 값은 softmax쪽 재사용
        rms_table=rms_table,
        k_bits=np.uint32(k_bits),
        frac_bits=np.uint32(FRAC_BITS),

        # Linear
        # 1. 입력 및 파라미터 Witness
        z_in_lin = np.uint64(z_in_lin),
        Q_in_diff_lin = Q_in_diff.astype(np.int32),   # (Q_in - z_in)
        Q_w_diff_lin  = Q_w_diff.astype(np.int32),    # (Q_w - z_w)
        Q_b_diff_lin  = Q_b_diff.astype(np.int64) if b_lin is not None else np.zeros(1), # (Q_b - z_b)
        
        # 2. 중간 연산 결과 (서킷의 단계별 검증 대상)
        G_lin         = G.astype(np.int64),           # MatMul 결과: (Q_in_diff * Q_w_diff)
        acc_lin       = acc.astype(np.int64),         # 대괄호 안의 합계: G + Q_b_diff
        
        # 3. 재스케일링 및 오프셋 Witness
        M_lin         = np.uint64(M_lin),              # 재스케일링 상수
        M_prime_lin   = np.uint64(M_prime_lin),        # z_out 보정 상수
        numerator_lin = numerator_lin.astype(object), # M * (acc + M_prime) -> 매우 큰 값 대비용
        
        # 4. 최종 결과 및 나눗셈 증명 파편 (enforce_equal의 핵심)
        Q_out_sim_lin = Q_out_sim_lin.astype(np.int32), # 최종 몫 (Quotient)
        R_out_lin     = R_out_lin.astype(np.int64),     # 시프트 나머지 (Remainder)

        # gelu
        sigmoid_part_gelu = sigmoid_part.astype(np.uint64),
        M_gelu_idx        = np.uint64(M_gelu_idx),      # 인덱스 변환 배수
        Q_out_sim_gelu    = Q_gelu_out_sim.astype(np.int64),
        R_out_gelu        = R_gelu_out.astype(np.int64),
        z_in_gelu         = np.uint64(z_out_lin),        # Linear1의 z_out이 GeLU의 z_in
        acc_gelu          = acc_gelu.astype(np.int64),
        M_gelu            = np.uint64(M_gelu),          # 최종 출력 멀티플라이어
        M_prime_gelu      = np.uint64(M_prime_gelu),    # 최종 출력 오프셋
        gelu_table        = gelu_table
    )