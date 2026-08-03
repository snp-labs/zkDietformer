"""Multi-sample MP4 diagnostic to calibrate bounds.

Usage:
  python diagnose_mp4_multi.py                          # default: tiny (h=128)
  python diagnose_mp4_multi.py --hidden_size 512        # small (h=512)
  python diagnose_mp4_multi.py --hidden_size 256 --num_hidden_layers 4
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import random
import tempfile

from transformers import AutoTokenizer
from nystrom_attention_capture import load_small_nystromformer, apply_monkey_patch
from gelu_swap import replace_activation_with_sigmoid_gelu
from evaluation_fixtures import SAMPLE_TEXTS, MAX_SEQ_LEN
from attention_pipeline import process_attn_debug_npz

parser = argparse.ArgumentParser(description="MP4 bounded-difference diagnostic")
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--num_attention_heads", type=int, default=None,
                    help="Default: max(4, hidden_size // 64)")
parser.add_argument("--num_hidden_layers", type=int, default=2)
parser.add_argument("--num_landmarks", type=int, default=32)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--seq_len", type=int, default=None,
                    help="Default: MAX_SEQ_LEN from evaluation_fixtures")
args = parser.parse_args()

HIDDEN_SIZE = args.hidden_size
NUM_HEADS = args.num_attention_heads or max(4, HIDDEN_SIZE // 64)
NUM_LAYERS = args.num_hidden_layers
NUM_LANDMARKS = args.num_landmarks
NUM_SAMPLES = args.num_samples
SEQ_LEN = args.seq_len or MAX_SEQ_LEN
K_BITS = 26

print(f"=== MP4 Diagnostic ===")
print(f"  hidden_size={HIDDEN_SIZE}, heads={NUM_HEADS}, layers={NUM_LAYERS}, "
      f"landmarks={NUM_LANDMARKS}, seq_len={SEQ_LEN}, samples={NUM_SAMPLES}")
print(f"  Current circuit deltas: Cond1=4096, Cond3=512, Cond4=32")
print()

max_diff1 = 0; max_diff2 = 0; max_asym3 = 0; max_asym4 = 0

for seed in range(NUM_SAMPLES):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")
    model = load_small_nystromformer(
        hidden_size=HIDDEN_SIZE, num_attention_heads=NUM_HEADS,
        num_hidden_layers=NUM_LAYERS, num_landmarks=NUM_LANDMARKS,
        seq_len=SEQ_LEN,
    )
    model.eval()
    replace_activation_with_sigmoid_gelu(model, "nystromformer")
    registry = apply_monkey_patch(model)

    enc = tokenizer(
        [SAMPLE_TEXTS[seed % len(SAMPLE_TEXTS)]],
        padding="max_length", max_length=SEQ_LEN,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        _ = model(**enc)

    for layer_name in registry.all().keys():
        layer_data = registry.get(layer_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_path = os.path.join(tmpdir, "debug.npz")
            out_path = os.path.join(tmpdir, "witness.npz")
            save_dict = {k: v for k, v in layer_data.items()}
            save_dict['mask'] = enc['attention_mask'].numpy()
            np.savez(debug_path, **save_dict)
            process_attn_debug_npz(debug_path, out_path, k_bits=K_BITS)
            w = np.load(out_path, allow_pickle=True)

            P_k2 = w['P_k2'].astype(np.int64)
            Q_inv = w['Q_inv_k2'].astype(np.int64)
            Q_y = w['Q_y_inv_intermediates'].astype(np.int64)
            Q_y1, Q_y2, Q_y3, Q_y4 = Q_y[0], Q_y[1], Q_y[2], Q_y[3]

            d1 = int(np.abs(Q_y2 - P_k2).max())
            d2 = int(np.abs(Q_y4 - Q_inv).max())
            a3 = int(np.abs(Q_y1 - np.swapaxes(Q_y1, -2, -1)).max())
            a4 = int(np.abs(Q_y3 - np.swapaxes(Q_y3, -2, -1)).max())

            max_diff1 = max(max_diff1, d1)
            max_diff2 = max(max_diff2, d2)
            max_asym3 = max(max_asym3, a3)
            max_asym4 = max(max_asym4, a4)

            ok1 = "OK" if d1 < 4096 else "EXCEED"
            ok3 = "OK" if a3 < 512 else "EXCEED"
            ok4 = "OK" if a4 < 32 else "EXCEED"
            print(f"  seed={seed} {layer_name}: d1={d1}[{ok1}], d2={d2}, a3={a3}[{ok3}], a4={a4}[{ok4}]")

print(f"\nMax across all {NUM_SAMPLES} samples (h={HIDDEN_SIZE}):")
print(f"  Cond1 max|Q_y2-P_k2|     = {max_diff1:>6}  (delta=4096, limit=8192)  {'OK' if max_diff1 < 4096 else 'EXCEED!'}")
print(f"  Cond2 max|Q_y4-Q_inv|    = {max_diff2:>6}")
print(f"  Cond3 max|Q_y1-Q_y1^T|   = {max_asym3:>6}  (delta=512,  limit=1024)  {'OK' if max_asym3 < 512 else 'EXCEED!'}")
print(f"  Cond4 max|Q_y3-Q_y3^T|   = {max_asym4:>6}  (delta=32,   limit=64)    {'OK' if max_asym4 < 32 else 'EXCEED!'}")

if max_diff1 >= 4096 or max_asym3 >= 512 or max_asym4 >= 32:
    print(f"\nWARNING: Some conditions exceed current circuit deltas!")
    print(f"  Suggested deltas (next power of 2 with 2x margin):")
    import math
    if max_diff1 >= 4096:
        sugg = 1 << math.ceil(math.log2(max_diff1 * 2))
        print(f"    Cond1: delta={sugg} ({math.ceil(math.log2(sugg*2))}-bit range)")
    if max_asym3 >= 512:
        sugg = 1 << math.ceil(math.log2(max_asym3 * 2))
        print(f"    Cond3: delta={sugg} ({math.ceil(math.log2(sugg*2))}-bit range)")
    if max_asym4 >= 32:
        sugg = 1 << math.ceil(math.log2(max_asym4 * 2))
        print(f"    Cond4: delta={sugg} ({math.ceil(math.log2(sugg*2))}-bit range)")
else:
    print(f"\nAll conditions within circuit deltas.")
