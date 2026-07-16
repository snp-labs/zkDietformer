import numpy as np
import os

TABLE_SIZE = 1 << 16  # 65536
FRAC_BITS = 8
GELU_INDEX_SCALE = 1.0 / (1 << 9)

def generate_gelu_sigmoid_tables(out_dir, k_bits=26, frac_bits=16):
    os.makedirs(out_dir, exist_ok=True)
    
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    real_x = (indices - (TABLE_SIZE // 2)) * GELU_INDEX_SCALE
    
    sigmoid = 1.0 / (1.0 + np.exp(-1.702 * real_x))
    
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
    X_diff = (Q_sim.astype(object) - int(z_in)) * (1 << k_bits) + R_mat.astype(object)
    
    scaled_diff = (X_diff * M_gelu_idx) >> k_bits
    index = scaled_diff + (TABLE_SIZE // 2)
    # cliping
    index_clipped = np.clip(index, 0, TABLE_SIZE - 1)
    return gelu_table[index_clipped.astype(int)].astype(np.uint64)