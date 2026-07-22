"""Diagnose Moore-Penrose 4 conditions in quantized domain.

Runs one forward pass, generates witness via the actual pipeline,
then checks how far off the 4 MP conditions are.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import random
import tempfile

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

from transformers import AutoTokenizer
from nystrom_attention_capture import load_small_nystromformer, apply_monkey_patch
from gelu_swap import replace_activation_with_sigmoid_gelu
from evaluation_fixtures import SAMPLE_TEXTS, MAX_SEQ_LEN
from attention_pipeline import process_attn_debug_npz

K_BITS = 26

# ── Setup model ──
tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")
model = load_small_nystromformer(
    hidden_size=128, num_attention_heads=2,
    num_hidden_layers=2, num_landmarks=32, seq_len=MAX_SEQ_LEN,
)
model.eval()
replace_activation_with_sigmoid_gelu(model, "nystromformer")
registry = apply_monkey_patch(model)

# ── Run one forward pass ──
enc = tokenizer(
    [SAMPLE_TEXTS[0]], padding="max_length",
    max_length=MAX_SEQ_LEN, truncation=True, return_tensors="pt",
)
with torch.no_grad():
    _ = model(**enc)

# ── Save debug NPZ and generate witness ──
layer_name = list(registry.all().keys())[0]
layer_data = registry.get(layer_name)

with tempfile.TemporaryDirectory() as tmpdir:
    debug_path = os.path.join(tmpdir, "debug.npz")
    out_path = os.path.join(tmpdir, "witness.npz")

    # Save debug data
    save_dict = {}
    for k, v in layer_data.items():
        save_dict[k] = v
    save_dict['mask'] = enc['attention_mask'].numpy()
    np.savez(debug_path, **save_dict)

    # Run actual pipeline
    process_attn_debug_npz(debug_path, out_path, k_bits=K_BITS)

    # Load witness
    w = np.load(out_path, allow_pickle=True)

    P_k2 = w['P_k2'].astype(np.int64)
    Q_inv = w['Q_inv_k2'].astype(np.int64)
    Q_y = w['Q_y_inv_intermediates'].astype(np.int64)  # [4, B, H, M, M]
    R_inv = w['R_inv_remainders'].astype(np.int64)
    M_inv = w['M_inv_all']
    z_inv = int(w['z_inv_k2'])

    Q_y1, Q_y2, Q_y3, Q_y4 = Q_y[0], Q_y[1], Q_y[2], Q_y[3]
    R1, R2, R3, R4 = R_inv[0], R_inv[1], R_inv[2], R_inv[3]

    B, H, M_dim, _ = P_k2.shape
    FRAC = 1 << K_BITS

    print(f"Config: B={B}, H={H}, M={M_dim}, K_BITS={K_BITS}")
    print(f"P_k2 range: [{P_k2.min()}, {P_k2.max()}]")
    print(f"Q_inv range: [{Q_inv.min()}, {Q_inv.max()}], z_inv={z_inv}")
    print(f"M_inv: {M_inv}")

    # ── Condition 1: Q_y2 == P_k2 ──
    diff1 = Q_y2 - P_k2
    print(f"\n{'='*60}")
    print(f"CONDITION 1: A·A⁺·A = A  →  Q_y2 == P_k2")
    print(f"  Q_y2 range: [{Q_y2.min()}, {Q_y2.max()}]")
    print(f"  diff range: [{diff1.min()}, {diff1.max()}]")
    print(f"  |diff| max: {np.abs(diff1).max()}")
    print(f"  |diff| mean: {np.abs(diff1).mean():.4f}")
    print(f"  exact: {(diff1==0).sum()}/{diff1.size} ({100*(diff1==0).sum()/diff1.size:.1f}%)")
    for d in sorted(set(diff1.flatten())):
        if abs(d) <= 10:
            print(f"    diff={d:+d}: {(diff1==d).sum()}")

    # ── Condition 2: Q_y4 == Q_inv ──
    diff2 = Q_y4 - Q_inv
    print(f"\n{'='*60}")
    print(f"CONDITION 2: A⁺·A·A⁺ = A⁺  →  Q_y4 == Q_inv")
    print(f"  Q_y4 range: [{Q_y4.min()}, {Q_y4.max()}]")
    print(f"  diff range: [{diff2.min()}, {diff2.max()}]")
    print(f"  |diff| max: {np.abs(diff2).max()}")
    print(f"  |diff| mean: {np.abs(diff2).mean():.4f}")
    print(f"  exact: {(diff2==0).sum()}/{diff2.size} ({100*(diff2==0).sum()/diff2.size:.1f}%)")
    for d in sorted(set(diff2.flatten())):
        if abs(d) <= 10:
            print(f"    diff={d:+d}: {(diff2==d).sum()}")

    # ── Condition 3: Q_y1 symmetric ──
    asym3 = Q_y1 - np.swapaxes(Q_y1, -2, -1)
    print(f"\n{'='*60}")
    print(f"CONDITION 3: (A·A⁺)^T = A·A⁺  →  Q_y1 symmetric")
    print(f"  Q_y1 range: [{Q_y1.min()}, {Q_y1.max()}]")
    upper = []
    for b in range(B):
        for h in range(H):
            for i in range(M_dim):
                for j in range(i+1, M_dim):
                    upper.append(asym3[b,h,i,j])
    upper = np.array(upper)
    print(f"  upper-tri asym range: [{upper.min()}, {upper.max()}]")
    print(f"  |asym| max: {np.abs(upper).max()}")
    print(f"  |asym| mean: {np.abs(upper).mean():.4f}")
    print(f"  exact sym: {(upper==0).sum()}/{upper.size} ({100*(upper==0).sum()/upper.size:.1f}%)")
    for d in sorted(set(upper)):
        if abs(d) <= 10:
            print(f"    asym={d:+d}: {(upper==d).sum()}")

    # ── Condition 4: Q_y3 symmetric ──
    asym4 = Q_y3 - np.swapaxes(Q_y3, -2, -1)
    print(f"\n{'='*60}")
    print(f"CONDITION 4: (A⁺·A)^T = A⁺·A  →  Q_y3 symmetric")
    print(f"  Q_y3 range: [{Q_y3.min()}, {Q_y3.max()}]")
    upper4 = []
    for b in range(B):
        for h in range(H):
            for i in range(M_dim):
                for j in range(i+1, M_dim):
                    upper4.append(asym4[b,h,i,j])
    upper4 = np.array(upper4)
    print(f"  upper-tri asym range: [{upper4.min()}, {upper4.max()}]")
    print(f"  |asym| max: {np.abs(upper4).max()}")
    print(f"  |asym| mean: {np.abs(upper4).mean():.4f}")
    print(f"  exact sym: {(upper4==0).sum()}/{upper4.size} ({100*(upper4==0).sum()/upper4.size:.1f}%)")
    for d in sorted(set(upper4)):
        if abs(d) <= 10:
            print(f"    asym={d:+d}: {(upper4==d).sum()}")

    # ── Remainder validity if forced ──
    print(f"\n{'='*60}")
    print(f"REMAINDER VALIDITY (if we force conditions)")

    # Cond 1: force Q_y2 = P_k2 → R2_new = numerator2 - P_k2 * 2^k
    # numerator2 = Q_y2 * 2^k + R2, so: numerator2 = Q_y2 * FRAC + R2
    numerator2 = Q_y2 * FRAC + R2
    R2_forced = numerator2 - P_k2 * FRAC
    n_neg = (R2_forced < 0).sum()
    n_over = (R2_forced >= FRAC).sum()
    print(f"\n  Cond1 forced R range: [{R2_forced.min()}, {R2_forced.max()}]")
    print(f"  Invalid (R<0): {n_neg}, Invalid (R>=2^k): {n_over}, Total invalid: {n_neg+n_over}/{R2_forced.size}")

    # Cond 2: force Q_y4 = Q_inv → R4_new = numerator4 - Q_inv * 2^k
    numerator4 = Q_y4 * FRAC + R4
    R4_forced = numerator4 - Q_inv * FRAC
    n_neg = (R4_forced < 0).sum()
    n_over = (R4_forced >= FRAC).sum()
    print(f"\n  Cond2 forced R range: [{R4_forced.min()}, {R4_forced.max()}]")
    print(f"  Invalid (R<0): {n_neg}, Invalid (R>=2^k): {n_over}, Total invalid: {n_neg+n_over}/{R4_forced.size}")

    # Cond 3: force Q_y1 symmetric
    numerator1 = Q_y1 * FRAC + R1
    violations3 = 0
    total3 = 0
    for b in range(B):
        for h in range(H):
            for i in range(M_dim):
                for j in range(i+1, M_dim):
                    total3 += 1
                    if Q_y1[b,h,i,j] == Q_y1[b,h,j,i]:
                        continue
                    n_ij = int(numerator1[b,h,i,j])
                    n_ji = int(numerator1[b,h,j,i])
                    ok = False
                    for v in [Q_y1[b,h,i,j], Q_y1[b,h,j,i], (Q_y1[b,h,i,j]+Q_y1[b,h,j,i])//2]:
                        v = int(v)
                        r1 = n_ij - v * FRAC
                        r2 = n_ji - v * FRAC
                        if 0 <= r1 < FRAC and 0 <= r2 < FRAC:
                            ok = True
                            break
                    if not ok:
                        violations3 += 1
    print(f"\n  Cond3 unfixable symmetric pairs: {violations3}/{total3}")

    # Cond 4: force Q_y3 symmetric
    numerator3_full = Q_y3 * FRAC + R3
    violations4 = 0
    total4 = 0
    for b in range(B):
        for h in range(H):
            for i in range(M_dim):
                for j in range(i+1, M_dim):
                    total4 += 1
                    if Q_y3[b,h,i,j] == Q_y3[b,h,j,i]:
                        continue
                    n_ij = int(numerator3_full[b,h,i,j])
                    n_ji = int(numerator3_full[b,h,j,i])
                    ok = False
                    for v in [Q_y3[b,h,i,j], Q_y3[b,h,j,i], (Q_y3[b,h,i,j]+Q_y3[b,h,j,i])//2]:
                        v = int(v)
                        r1 = n_ij - v * FRAC
                        r2 = n_ji - v * FRAC
                        if 0 <= r1 < FRAC and 0 <= r2 < FRAC:
                            ok = True
                            break
                    if not ok:
                        violations4 += 1
    print(f"  Cond4 unfixable symmetric pairs: {violations4}/{total4}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Cond1 max |diff|={np.abs(diff1).max()}, Cond2 max |diff|={np.abs(diff2).max()}")
    print(f"  Cond3 max |asym|={np.abs(upper).max()}, Cond4 max |asym|={np.abs(upper4).max()}")
    print(f"  All exact: {(diff1==0).all() and (diff2==0).all() and (upper==0).all() and (upper4==0).all()}")
