"""Check if Q_scaled_k (projection chain) matches Q_k (fresh quantization)."""
import numpy as np

wp = 'outputs_n128_k32_L2_h128_linformer/outputs_n128_k32_L2_h128_linformer-batch0-bert_encoder_layer_0_attention_self.witness.npz'
w = np.load(wp, allow_pickle=True)

Q_scaled_k = w['Q_scaled_k']
Q_k = w['Q_k']

print(f"Q_scaled_k shape: {Q_scaled_k.shape}, Q_k shape: {Q_k.shape}")
diff = Q_scaled_k.astype(np.int64) - Q_k.astype(np.int64)
n_mismatch = int(np.sum(diff != 0))
print(f"max|diff|: {np.max(np.abs(diff))}")
print(f"n_mismatch: {n_mismatch} / {diff.size}")

if n_mismatch > 0:
    idx = np.argwhere(diff != 0)
    print(f"\nFirst 10 mismatches:")
    for i in range(min(10, len(idx))):
        loc = tuple(idx[i])
        print(f"  {loc}: Q_scaled_k={Q_scaled_k[loc]}, Q_k={Q_k[loc]}, diff={diff[loc]}")
else:
    print("Q_scaled_k == Q_k: EXACT MATCH")

# Also check at padding positions
mask = w['mask']  # [B, S]
B, H, S, D = Q_k.shape
z_k = int(w['z_k'])
print(f"\nz_k = {z_k}")
print(f"mask shape: {mask.shape}")
n_pad = int(np.sum(mask == 0))
print(f"padding positions: {n_pad} / {S}")

if n_pad > 0:
    for b in range(B):
        pad_pos = np.where(mask[b] == 0)[0]
        if len(pad_pos) > 0:
            qk_pad = Q_k[b, :, pad_pos, :]
            qs_pad = Q_scaled_k[b, :, pad_pos, :]
            print(f"\nBatch {b}, padding positions: {pad_pos[:10]}...")
            print(f"  Q_k at pad: min={qk_pad.min()}, max={qk_pad.max()}, all==z_k? {np.all(qk_pad == z_k)}")
            print(f"  Q_scaled_k at pad: min={qs_pad.min()}, max={qs_pad.max()}, all==z_k? {np.all(qs_pad == z_k)}")
