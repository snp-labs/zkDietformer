"""Multi-sample MP4 diagnostic to calibrate bounds."""
import os, sys
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

K_BITS = 26
NUM_SAMPLES = 5

max_diff1 = 0; max_diff2 = 0; max_asym3 = 0; max_asym4 = 0

for seed in range(NUM_SAMPLES):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")
    model = load_small_nystromformer(
        hidden_size=128, num_attention_heads=2,
        num_hidden_layers=2, num_landmarks=32, seq_len=MAX_SEQ_LEN,
    )
    model.eval()
    replace_activation_with_sigmoid_gelu(model, "nystromformer")
    registry = apply_monkey_patch(model)

    enc = tokenizer(
        [SAMPLE_TEXTS[seed % len(SAMPLE_TEXTS)]],
        padding="max_length", max_length=MAX_SEQ_LEN,
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

            print(f"  seed={seed} {layer_name}: d1={d1}, d2={d2}, a3={a3}, a4={a4}")

print(f"\nMax across all samples:")
print(f"  Cond1 max|diff|={max_diff1}")
print(f"  Cond2 max|diff|={max_diff2}")
print(f"  Cond3 max|asym|={max_asym3}")
print(f"  Cond4 max|asym|={max_asym4}")
