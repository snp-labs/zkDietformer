import numpy as np
import os

TABLE_SIZE = 1 << 16  # 65536
FRAC_BITS = 16 

def generate_gelu_sigmoid_tables(out_dir, s_in, k_bits=22, frac_bits=16):
    os.makedirs(out_dir, exist_ok=True)
    
    shift = k_bits - 8 # 22 - 8 = 14
    # [수정] 인덱스 1단위가 실제 값에서 가지는 크기 (Q의 1/256 정밀도)
    unit_scale = s_in / float(1 << 8) 
    
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    # 인덱스 중간(32768)이 x=0이 되도록 설정하여 음수 영역 커버
    real_x = (indices - (TABLE_SIZE // 2)) * unit_scale
    
    def sigmoid_1702(val):
        return 1.0 / (1.0 + np.exp(-1.702 * val))
    
    table_vals = sigmoid_1702(real_x)
    fixed_table = (table_vals * (1 << frac_bits)).astype(np.uint64)

    out_path = os.path.join(out_dir, "gelu_tables.npz")
    np.savez_compressed(
        out_path,
        table=fixed_table,
        frac_bits=np.int32(frac_bits),
        s_in=np.float64(s_in),
        k_bits=np.int32(k_bits)
    )
    return out_path

def extract_gelu_sigmoid_single(Q_sim, R_mat, z_in, gelu_table, k_bits):
    # 1. (Q - z) * 2^k + R 을 통해 영점이 조정된 고정밀 차이값 계산
    # int(z_in)으로 명시적 변환하여 연산 오류 방지
    X_diff = (Q_sim.astype(object) - int(z_in)) * (1 << k_bits) + R_mat.astype(object)
    
    # 2. 16비트 테이블 범위로 다운시프트 (shift = 22 - 8 = 14)
    shift = k_bits - 8
    index = (X_diff >> shift) + (TABLE_SIZE // 2)
    
    # 3. 안전하게 클리핑 후 조회
    index_clipped = np.clip(index, 0, TABLE_SIZE - 1)
    return gelu_table[index_clipped.astype(int)].astype(np.uint64)