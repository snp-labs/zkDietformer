import numpy as np
import os

TABLE_SIZE = 1 << 16  
FRAC_BITS = 8 

def generate_rmsnorm_tables(out_dir, k_bits=22, frac_bits=8, eps=1e-6):
    os.makedirs(out_dir, exist_ok=True)
    
    real_ms = np.arange(TABLE_SIZE, dtype=np.float64)

    inv_rms = 1.0 / np.sqrt(real_ms + eps)
    fixed_inv_rms = (inv_rms * (1 << frac_bits)).astype(np.uint64)

    out_path = os.path.join(out_dir, "rmsnorm_tables.npz")
    np.savez_compressed(
        out_path,
        table=fixed_inv_rms,
        frac_bits=np.int32(frac_bits),
        k_bits=np.int32(k_bits)
    )
    return out_path

def extract_inv_rms_single(Q_ms_sim, R_ms, dim, rms_table, M_rms, k_bits):
    # 1. 몫과 나머지를 이용해 합산값(shifted_val) 복원
    # shifted_val = Q_ms_sim * dim + R_ms
    shifted_val = (Q_ms_sim.astype(object) * dim) + R_ms.astype(object)
    
    # 2. 멀티플라이어를 곱해 실제 수치 도메인으로 확장 (Scaling)
    # numerator = shifted_val * (s_ms * 2^k)
    numerator = shifted_val * M_rms
    
    # 3. 2^k 시프트(소수점 제거) 후 dim으로 나누어 평균(Mean) 인덱스 도출
    # Index = ( (Sum * s_ms) / 1.0 ) / dim
    scaled_val = numerator >> k_bits
    index = (scaled_val // dim).astype(np.uint64)
    
    # 4. 테이블 범위 클리핑
    index_clipped = np.clip(index, 0, TABLE_SIZE - 1)
    
    # 5. 테이블에서 1/sqrt(MS) 값 추출
    inv_rms_total = rms_table[index_clipped.astype(int)]  
    return inv_rms_total.astype(np.uint64)