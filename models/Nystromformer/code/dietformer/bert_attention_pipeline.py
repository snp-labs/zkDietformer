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
from transformers import BertForMaskedLM, BertConfig


def load_small_bert(
    hidden_size: int = 512,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 8,
    intermediate_size: int = None,
    seq_len: int = 512,
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

def process_standard_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=22):
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

    # 3. Softmax Tables 로드 — 출력 디렉터리에 자동 생성 (caller 의 out_npz_path 기준)
    # 예: out_npz_path="outputs_bert/...witness.npz" → table_dir="outputs_bert"
    # 다른 디렉터리 (예: --save_attn_npz 가변) 사용 시에도 자동 정합.
    table_dir = os.path.dirname(out_npz_path) or "."
    table_path = generate_softmax_tables(table_dir, k_bits, FRAC_BITS)
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
        exp_sum=exp_sum.astype(np.uint64),
        
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