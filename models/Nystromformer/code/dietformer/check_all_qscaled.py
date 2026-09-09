"""Check consistency between circuit variables and pipeline witness."""
import numpy as np

wp = 'outputs_n128_k32_L2_h128_linformer/outputs_n128_k32_L2_h128_linformer-batch0-bert_encoder_layer_0_attention_self.witness.npz'
w = np.load(wp, allow_pickle=True)

# Q_scaled_q vs Q_q
Q_scaled_q = w['Q_scaled_q']
Q_q = w['Q_q']
diff_q = Q_scaled_q.astype(np.int64) - Q_q.astype(np.int64)
n_mm_q = int(np.sum(diff_q != 0))
print(f"Q_scaled_q vs Q_q: {n_mm_q}/{diff_q.size} mismatch, max|diff|={np.max(np.abs(diff_q))}")

# Q_scaled_k vs Q_k
Q_scaled_k = w['Q_scaled_k']
Q_k = w['Q_k']
diff_k = Q_scaled_k.astype(np.int64) - Q_k.astype(np.int64)
n_mm_k = int(np.sum(diff_k != 0))
print(f"Q_scaled_k vs Q_k: {n_mm_k}/{diff_k.size} mismatch, max|diff|={np.max(np.abs(diff_k))}")

# Q_raw_proj_v vs Q_v (V has no scaling)
Q_raw_v = w['Q_raw_proj_v']
Q_v = w['Q_v']
diff_v = Q_raw_v.astype(np.int64) - Q_v.astype(np.int64)
n_mm_v = int(np.sum(diff_v != 0))
print(f"Q_raw_proj_v vs Q_v: {n_mm_v}/{diff_v.size} mismatch, max|diff|={np.max(np.abs(diff_v))}")

# Check what the circuit uses for logit matmul:
# logit = scale_q_out @ EK^T (circuit)
# logit = Q_q @ Q_sim_ek^T (pipeline)
# So scale_q_out is used for logit matmul in circuit
print(f"\nLogit input (Q side): circuit uses scale_q_out = Q_scaled_q")
print(f"  Pipeline uses Q_q. Mismatch: {n_mm_q}/{diff_q.size}")

# Check what the circuit uses for FV matmul:
# FV = f_vars @ proj_v_out (circuit)
# FV = Q_f @ Q_v (pipeline)
# proj_v_out = Q_raw_proj_v
print(f"\nFV input (V side): circuit uses proj_v_out = Q_raw_proj_v")
print(f"  Pipeline uses Q_v. Mismatch: {n_mm_v}/{diff_v.size}")

# Summary: which zero-points match?
print(f"\nz_k (fresh quant) = {int(w['z_k'])}")
print(f"z_q (fresh quant) = {int(w['z_q'])}")
print(f"z_v (fresh quant) = {int(w['z_v'])}")
print(f"z_raw_q (proj chain) = {int(w['z_raw_q'])}")
print(f"z_raw_k (proj chain) = {int(w['z_raw_k'])}")
print(f"z_raw_v (proj chain) = {int(w['z_raw_v'])}")
