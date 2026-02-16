import numpy as np
import os

TABLE_SIZE = 1 << 16  # 65536
FRAC_BITS = 16 

# 고정 인덱스 해상도: 인덱스 1 단위가 실제 값 1/512 (약 0.00195)를 의미함
# TABLE_SIZE // 2 (32768)가 실제 값 0.0 지점
GELU_INDEX_SCALE = 1.0 / (1 << 9)

def generate_gelu_sigmoid_tables(out_dir, k_bits=22, frac_bits=16):
    os.makedirs(out_dir, exist_ok=True)
    
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    # 인덱스를 실제 수치 도메인으로 매핑 (중간값을 0으로 설정)
    real_x = (indices - (TABLE_SIZE // 2)) * GELU_INDEX_SCALE
    
    # GeLU sigmoid 근사식: sigmoid(1.702 * x)
    sigmoid = 1.0 / (1.0 + np.exp(-1.702 * real_x))
    
    # 고정 소수점으로 변환하여 저장
    fixed_sigmoid = (sigmoid * (1 << frac_bits)).astype(np.uint64)

    out_path = os.path.join(out_dir, "gelu_tables.npz")
    np.savez_compressed(
        out_path,
        table=fixed_sigmoid,
        frac_bits=np.int32(frac_bits),
        k_bits=np.int32(k_bits),
        gelu_index_scale=np.float64(GELU_INDEX_SCALE)
    )
    return out_path

def extract_gelu_sigmoid_single(Q_sim, R_mat, z_in, gelu_table, M_gelu_idx, k_bits):
    # 1. 고정밀 분자값 계산: (Q - z) * 2^k + R
    X_diff = (Q_sim.astype(object) - int(z_in)) * (1 << k_bits) + R_mat.astype(object)
    
    # 2. 멀티플라이어를 곱해 테이블 인덱스 스케일로 변환 후 2^k 시프트
    # Index_offset = (X_diff * M_gelu_idx) >> k_bits
    scaled_diff = (X_diff * M_gelu_idx) >> k_bits
    
    # 3. 중간값(Offset)을 더해 최종 인덱스 산출
    index = scaled_diff + (TABLE_SIZE // 2)
    
    # 4. 클리핑 및 조회
    index_clipped = np.clip(index, 0, TABLE_SIZE - 1)
    return gelu_table[index_clipped.astype(int)].astype(np.uint64)