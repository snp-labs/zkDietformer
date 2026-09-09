"""Gaussian kernel lookup table for SOFT attention.

Table function: exp(-x), x >= 0 (unsigned distance input).
Differs from softmax table (exp(+x), signed logit input).

Table parameters (initial values, may be adjusted after A-gate measurement):
  TABLE_FRAC_BITS = 10  (input resolution: 1/1024 per unit)
  FRAC_BITS = 16        (output precision: 1/65536)
  TABLE_SIZE = 2^16     (65536 entries, Lasso infrastructure compatible)

Input range: [0, 64) since 2^16 / 2^10 = 64.
Effective range: [0, ~12) since exp(-12) ~ 6e-6, negligible in 16-bit fixed point.
"""
import os
import numpy as np

GAUSS_FRAC_BITS = 16
GAUSS_TABLE_FRAC_BITS = 10
UINT64_MAX = (1 << 64) - 1


def generate_gaussian_table(table_frac_bits=GAUSS_TABLE_FRAC_BITS,
                            frac_bits=GAUSS_FRAC_BITS,
                            z_dist=0):
    """Generate exp(-x) lookup table for Gaussian kernel.

    With z_dist offset (absorbs floor-truncation negative guard):
      table[j] = 1.0 (fixed point)                    for j <= z_dist
      table[j] = exp(-(j - z_dist) / 2^table_frac_bits) for j > z_dist

    Returns:
        np.ndarray of shape (65536,), dtype uint64.
    """
    TABLE_SIZE = 1 << 16  # 65536
    scale = 1 << table_frac_bits
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    shifted = np.maximum(indices - z_dist, 0.0)
    real_x = shifted / scale
    exps = np.exp(-real_x)
    fixed_exps = exps * float(1 << frac_bits)
    fixed_exps = np.clip(fixed_exps, 0, float(UINT64_MAX))
    return fixed_exps.astype(np.uint64)


def save_gaussian_tables(out_dir, table_frac_bits=GAUSS_TABLE_FRAC_BITS,
                         frac_bits=GAUSS_FRAC_BITS, z_dist=0):
    """Save Gaussian exp(-x) table as gaussian_tables.npz for Rust Lasso."""
    os.makedirs(out_dir, exist_ok=True)
    gauss_table = generate_gaussian_table(table_frac_bits, frac_bits, z_dist)
    tables_2d = gauss_table.reshape(1, -1)  # (1, 65536)
    out_path = os.path.join(out_dir, "gaussian_tables.npz")
    np.savez_compressed(out_path, tables=tables_2d,
                        frac_bits=np.int32(frac_bits),
                        table_frac_bits=np.int32(table_frac_bits))
    print(f"  Gaussian table saved to {out_path} -- shape {tables_2d.shape}")
    return out_path


def gaussian_lookup(dist, R_matmul_dist, s_dist, k_bits,
                    frac_bits=GAUSS_FRAC_BITS,
                    table_frac_bits=GAUSS_TABLE_FRAC_BITS,
                    gauss_table=None):
    """Gaussian kernel lookup: kernel_val = gauss_table[table_idx].

    No row-sum normalization (unlike softmax). No R_div, no exp_sum.

    Args:
        dist: int64 array, quantized squared distance.
        R_matmul_dist: int64 array, matmul remainder for dist.
        s_dist: float, scale of dist quantization.
        k_bits: int, fixed-point shift bits.
        gauss_table: uint64 array of shape (TABLE_SIZE,).

    Returns:
        kernel_val: uint64 array, fixed-point exp(-x) values.
        table_idx: int64 array, table indices.
        r_gauss: object array, Gaussian remainder (2*k_bits wide).
        M_gauss: int, Gaussian multiplier.
    """
    if gauss_table is None:
        gauss_table = generate_gaussian_table(table_frac_bits, frac_bits)
    TABLE_SIZE = len(gauss_table)

    # M_gauss: maps quantized dist to table index
    # table_idx = (dist * s_dist * 2^table_frac_bits) / (2 * sqrt(D))
    # But the denominator 2*sqrt(D) is already baked into dist by the caller.
    # So: M_gauss = floor(s_dist * 2^k_bits * 2^table_frac_bits)
    M_gauss = int(np.floor(s_dist * (1 << k_bits) * (1 << table_frac_bits)))
    if M_gauss <= 0:
        M_gauss = 1

    X = (dist.astype(object) << k_bits) + R_matmul_dist.astype(object)
    scaled_product = X * M_gauss
    table_idx_raw = (scaled_product >> (2 * k_bits)).astype(np.int64)
    r_gauss = scaled_product % (1 << (2 * k_bits))

    # Detect actual clamping (values changed by clip)
    n_clamp_low = int(np.sum(table_idx_raw < 0))
    n_clamp_high = int(np.sum(table_idx_raw >= TABLE_SIZE))
    table_idx = np.clip(table_idx_raw, 0, TABLE_SIZE - 1)

    kernel_val = gauss_table[table_idx.astype(int)].astype(np.uint64)

    return kernel_val, table_idx, r_gauss, M_gauss, n_clamp_low, n_clamp_high
