#!/usr/bin/env python3
"""C-1: GELU sigmoid lookup shift fix verification.

Validates that extract_gelu_sigmoid_single produces sigmoid values
matching the float reference sigmoid(1.702 * x) within table resolution.

Error budget:
  - Table output quantization: 1/(2^FRAC_BITS) = 1/256 ≈ 0.004
  - Input quantization step:   s_in * 1.702 * 0.25 (sigmoid max slope)
  - Table index resolution:    GELU_INDEX_SCALE * 1.702 * 0.25 ≈ 0.0008
"""
import sys
import numpy as np

from gelu_table import (
    generate_gelu_sigmoid_tables, extract_gelu_sigmoid_single,
    TABLE_SIZE, FRAC_BITS, GELU_INDEX_SCALE,
)
from quantize import get_gelu_index_multiplier

K_BITS = 26
TABLE_DIR = "/tmp/test_gelu_tables"

# Generate table with correct frac_bits
generate_gelu_sigmoid_tables(TABLE_DIR, k_bits=K_BITS, frac_bits=FRAC_BITS)
gelu_table = np.load(f"{TABLE_DIR}/gelu_tables.npz")["table"]

# Table valid range: index 0..65535 maps to real x in [-64, 64) via GELU_INDEX_SCALE=1/512
TABLE_RANGE_LO = -(TABLE_SIZE // 2) * GELU_INDEX_SCALE  # -64.0
TABLE_RANGE_HI = (TABLE_SIZE // 2 - 1) * GELU_INDEX_SCALE  # ~+63.998

x_real = np.linspace(-70, 70, 10001)
s_in_values = [0.01, 0.05, 0.2]

all_pass = True

for s_in in s_in_values:
    # Simulate quantization: Q = round(x / s_in), R = 0 (no remainder for clean test)
    Q_sim = np.round(x_real / s_in).astype(np.int64)
    z_in = 0  # zero-point = 0 for simplicity
    R_mat = np.zeros_like(Q_sim, dtype=np.int64)

    M_gelu_idx = get_gelu_index_multiplier(s_in, k_bits=K_BITS)

    # Reshape to 3D as expected by the function (batch=1, seq=1, dim=N)
    Q_3d = Q_sim.reshape(1, 1, -1)
    R_3d = R_mat.reshape(1, 1, -1)

    sigmoid_part = extract_gelu_sigmoid_single(
        Q_3d, R_3d, z_in, gelu_table, M_gelu_idx, K_BITS,
    )
    sigmoid_fp = sigmoid_part.reshape(-1).astype(np.float64) / (1 << FRAC_BITS)

    # Float reference
    ref_sigmoid = 1.0 / (1.0 + np.exp(-1.702 * x_real))

    # Compute index to check clipping ratio (KEY validation — buggy shift clips nearly all)
    X_diff = (Q_sim.astype(object) - int(z_in)) * (1 << K_BITS) + R_mat.astype(object)
    raw_index = ((X_diff * M_gelu_idx) >> (2 * K_BITS)) + (TABLE_SIZE // 2)
    n_clipped = np.sum((raw_index <= 0) | (raw_index >= TABLE_SIZE - 1))
    # Only endpoints beyond table range should clip
    n_out_of_range = np.sum((x_real < TABLE_RANGE_LO) | (x_real > TABLE_RANGE_HI))
    clip_ratio = n_clipped / len(x_real)

    # Split into in-range and out-of-range
    in_range = (x_real >= TABLE_RANGE_LO) & (x_real <= TABLE_RANGE_HI)
    out_range_lo = x_real < TABLE_RANGE_LO
    out_range_hi = x_real > TABLE_RANGE_HI

    abs_err = np.abs(sigmoid_fp - ref_sigmoid)
    max_err_in = abs_err[in_range].max() if in_range.any() else 0.0
    mean_err_in = abs_err[in_range].mean() if in_range.any() else 0.0

    # Error threshold: table quantization + input quantization
    # sigmoid max slope ≈ 1.702 * 0.25 = 0.4255
    table_quant_err = 1.0 / (1 << FRAC_BITS)   # 1/256
    input_quant_err = s_in * 0.4255             # proportional to s_in
    threshold = table_quant_err + input_quant_err + 0.001  # small margin

    # Out of range: should saturate to 0 (low) or ~1 (high)
    sat_lo_ok = True
    sat_hi_ok = True
    if out_range_lo.any():
        lo_vals = sigmoid_fp[out_range_lo]
        if not np.all(lo_vals < 0.01):
            sat_lo_ok = False
    if out_range_hi.any():
        hi_vals = sigmoid_fp[out_range_hi]
        if not np.all(hi_vals > 0.99):
            sat_hi_ok = False

    pass_in = max_err_in <= threshold
    # Clipping should only happen for out-of-range values (± small margin)
    # Quantization can push boundary values ±(s_in * 512) entries past table edge
    clip_margin = max(10, int(s_in * 512) + 5)
    clip_ok = n_clipped <= n_out_of_range + clip_margin
    pass_str = "PASS" if (pass_in and sat_lo_ok and sat_hi_ok and clip_ok) else "FAIL"
    if pass_str == "FAIL":
        all_pass = False

    print(f"s_in={s_in:.2f}: max_err={max_err_in:.6f} (threshold={threshold:.4f}), "
          f"mean_err={mean_err_in:.6f}, clip_ratio={clip_ratio:.4f} "
          f"(n_clip={n_clipped}, n_oor={n_out_of_range}), "
          f"sat_lo={'OK' if sat_lo_ok else 'FAIL'}, sat_hi={'OK' if sat_hi_ok else 'FAIL'} → {pass_str}")

print()
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
