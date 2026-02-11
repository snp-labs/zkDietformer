import numpy as np
import os

TABLE_SIZE = 1 << 16  
FRAC_BITS = 16 

def generate_rmsnorm_tables(out_dir, s_x, k_bits=22, frac_bits=16, eps=1e-6):
    os.makedirs(out_dir, exist_ok=True)
    
    unit_scale = s_x / float(1 << k_bits)
    indices = np.arange(TABLE_SIZE, dtype=np.float64)

    real_ms = indices * unit_scale
    
    inv_rms = 1.0 / np.sqrt(real_ms + eps)
    fixed_inv_rms = (inv_rms * (1 << frac_bits)).astype(np.uint64)

    out_path = os.path.join(out_dir, "rmsnorm_tables.npz")
    np.savez_compressed(
        out_path,
        table=fixed_inv_rms,
        frac_bits=np.int32(frac_bits),
        s_x=np.float64(s_x),
        k_bits=np.int32(k_bits)
    )
    return out_path

def extract_inv_rms_single(Q_sim, R_mat, rms_table, k_bits):
    X = (Q_sim.astype(object) << k_bits) + R_mat.astype(object)

    index = (X).astype(np.uint64)
    index = np.clip(index, 0, TABLE_SIZE - 1)
    
    inv_rms_total = rms_table[index.astype(int)]  
    return inv_rms_total.astype(np.uint64)