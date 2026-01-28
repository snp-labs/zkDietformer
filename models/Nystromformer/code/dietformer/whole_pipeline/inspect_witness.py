import numpy as np
import sys

debug_npz = "./attn_debug_outputs/attn_debug-model.transformer_1.mha.attn.npz"
tables_npz = "tmp_tables/softmax_tables_quantized.npz"

d = np.load(debug_npz, allow_pickle=True)
print("Debug keys:", list(d.keys()))
for k in ['kernel_3', 'V', 'kernel_1', 'inv_kernel_2']:
    if k in d:
        print(f"{k}.shape =", np.array(d[k]).shape, "dtype=", np.array(d[k]).dtype)
    else:
        print(f"{k} not in debug npz")

t = np.load(tables_npz, allow_pickle=True)
print("Tables keys:", list(t.keys()))
tables = t['tables']
print("tables.shape =", tables.shape, "tables.dtype=", tables.dtype)
print("tables min/max finite check: isnan:", np.isnan(tables).any(), "isinf:", np.isinf(tables).any())

# some details
try:
    limb_bits = int(t.get('limb_bits', -1))
    k_segments = int(t.get('k_segments', -1))
    print("limb_bits=", limb_bits, "k_segments=", k_segments)
    expected_W = 1 << limb_bits if limb_bits>0 else None
    print("expected table width (2**limb_bits) ->", expected_W)
except Exception as e:
    print("Could not read limb_bits:", e)

# load matmul_targets_k3 saved by pipeline if exists
if 'matmul_targets_k3' in d:
    mt3 = np.array(d['matmul_targets_k3'])
    print("matmul_targets_k3.shape:", mt3.shape, " sample values:", mt3.ravel()[:10])

# If attn_witness_pipeline already saved limb_indices (maybe in witness npz), try to load
witness_path = debug_npz.replace(".npz", ".witness.npz")
try:
    w = np.load(witness_path, allow_pickle=True)
    print("Witness keys:", list(w.keys()))
    if 'limb_indices_k3' in w:
        li = np.array(w['limb_indices_k3'])
        print("limb_indices_k3.shape", li.shape, "min/max:", li.min(), li.max())
        print("sample limb indices [0,0,0,0,:10]:", li[0,0,0,0,:10])
except Exception as e:
    print("No witness npz or failed to load:", e)