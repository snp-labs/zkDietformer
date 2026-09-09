"""B-0: Check chain vs fresh quantization error and logit table bounds."""
import numpy as np
from quantize import get_quantization_params, quantize_tensor

wp = ('outputs_n128_k32_L2_h128_linformer/'
      'outputs_n128_k32_L2_h128_linformer-batch0-'
      'bert_encoder_layer_0_attention_self.witness.npz')
w = np.load(wp, allow_pickle=True)
debug_path = wp.replace('.witness.npz', '.npz')
d = np.load(debug_path, allow_pickle=True)

K_proc_float = d['K_proc']
mask = w['mask']
val_idx = np.where(mask[0] == 1)[0]
z_k = int(w['z_k'])

# Chain output (unmasked scaling result)
Q_scaled_k = w['Q_scaled_k']

# Fresh quantization of float K_proc
s_fresh, z_fresh = get_quantization_params(K_proc_float)
Q_k_fresh = quantize_tensor(K_proc_float, s_fresh, z_fresh)

print(f"z_chain={z_k}, z_fresh={z_fresh}")

# Compare at valid positions only
chain_valid = Q_scaled_k[0, :, val_idx, :].astype(np.int64)
fresh_valid = Q_k_fresh[0, :, val_idx, :].astype(np.int64)
diff = chain_valid - fresh_valid
n_ne = int(np.sum(diff != 0))
print(f"At valid: chain vs fresh: nonzero={n_ne}/{diff.size}, "
      f"max|diff|={np.max(np.abs(diff))}")

# ---- B-0-2: logit table boundary ----
print("\n=== Logit table boundary ===")
for li in [0, 1]:
    name = f"bert_encoder_layer_{li}_attention_self"
    wpath = (f"outputs_n128_k32_L2_h128_linformer/"
             f"outputs_n128_k32_L2_h128_linformer-batch0-{name}.witness.npz")
    ww = np.load(wpath, allow_pickle=True)
    Q_sim_logit = ww['Q_sim_logit'].astype(np.int64)
    table_idx = ww['table_idx'].astype(np.int64)
    k_bits = int(ww['k_bits'])

    logit_min = int(np.min(Q_sim_logit))
    logit_max = int(np.max(Q_sim_logit))
    tidx_min = int(np.min(table_idx))
    tidx_max = int(np.max(table_idx))

    # Table size
    table_frac_bits = int(ww['table_frac_bits'])
    table_size = 1 << table_frac_bits

    print(f"\nLayer {li}:")
    print(f"  Q_sim_logit: min={logit_min}, max={logit_max}")
    print(f"  table_idx:   min={tidx_min}, max={tidx_max}")
    print(f"  table_size:  {table_size} (2^{table_frac_bits})")
    print(f"  margin_low:  {tidx_min} (distance from 0)")
    print(f"  margin_high: {table_size - 1 - tidx_max} (distance from max)")

    # Check for clamping
    n_clamped_low = int(np.sum(table_idx <= 0))
    n_clamped_high = int(np.sum(table_idx >= table_size - 1))
    if n_clamped_low > 0 or n_clamped_high > 0:
        print(f"  WARNING: clamped! low={n_clamped_low}, high={n_clamped_high}")
    else:
        print(f"  No clamping detected.")
