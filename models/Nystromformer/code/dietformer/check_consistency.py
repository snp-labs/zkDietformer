"""Part 1-2: Verify circuit input == pipeline input after fix."""
import numpy as np

wp = 'outputs_n128_k32_L2_h128_linformer/outputs_n128_k32_L2_h128_linformer-batch0-bert_encoder_layer_0_attention_self.witness.npz'
w = np.load(wp, allow_pickle=True)

# After the fix, Q_k in NPZ should be Q_scaled_k (chain output)
Q_k_npz = w['Q_k']
Q_scaled_k = w['Q_scaled_k']
diff_k = Q_k_npz.astype(np.int64) - Q_scaled_k.astype(np.int64)
n_mm_k = int(np.sum(diff_k != 0))
print(f"Q_k (NPZ) vs Q_scaled_k: {n_mm_k}/{diff_k.size} mismatch, max|diff|={np.max(np.abs(diff_k))}")

Q_q_npz = w['Q_q']
Q_scaled_q = w['Q_scaled_q']
diff_q = Q_q_npz.astype(np.int64) - Q_scaled_q.astype(np.int64)
n_mm_q = int(np.sum(diff_q != 0))
print(f"Q_q (NPZ) vs Q_scaled_q: {n_mm_q}/{diff_q.size} mismatch, max|diff|={np.max(np.abs(diff_q))}")

Q_v_npz = w['Q_v']
Q_raw_v = w['Q_raw_proj_v']
diff_v = Q_v_npz.astype(np.int64) - Q_raw_v.astype(np.int64)
n_mm_v = int(np.sum(diff_v != 0))
print(f"Q_v (NPZ) vs Q_raw_proj_v: {n_mm_v}/{diff_v.size} mismatch, max|diff|={np.max(np.abs(diff_v))}")

# Zero-point consistency
print(f"\nz_k = {int(w['z_k'])}, z_raw_k = {int(w['z_raw_k'])}")
print(f"z_q = {int(w['z_q'])}, z_raw_q = {int(w['z_raw_q'])}")
print(f"z_v = {int(w['z_v'])}, z_raw_v = {int(w['z_raw_v'])}")
