"""B-gate fixture generator for SOFT attention circuit.

Fixture parameters:
  S=8, m=4, H=2, D=4, L=1, B=1, padding=1, seg_size=2, n_iter=20.
  7 valid tokens + 1 padding token.
  No purely-padding segment, so A is full rank and MP conditions are meaningful.
"""
import os
import sys
import numpy as np
import torch

from bert_attention_pipeline import load_small_bert
from gelu_swap import replace_activation_with_sigmoid_gelu
from soft_attention import SOFTSelfAttention
from soft_attention_capture import apply_soft_monkey_patch
from soft_attention_pipeline import process_soft_attn_debug_npz
from gaussian_table import save_gaussian_tables
from transformers import BertTokenizerFast

S = 8
M_LANDMARKS = 4
H = 2
D = 4
HIDDEN = H * D  # 8
L = 1
N_ITER = 20

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

out_dir = f"outputs_n{S}_m{M_LANDMARKS}_L{L}_h{HIDDEN}_soft_bgate"
os.makedirs(out_dir, exist_ok=True)

# Build small model
model = load_small_bert(
    hidden_size=HIDDEN, num_attention_heads=H,
    num_hidden_layers=L, seq_len=S,
)
model.eval()
replace_activation_with_sigmoid_gelu(model, "bert")

# Replace attention with SOFT
for i, layer in enumerate(model.bert.encoder.layer):
    old_self = layer.attention.self
    new_self = SOFTSelfAttention(
        hidden_size=HIDDEN,
        num_attention_heads=H,
        seq_len=S,
        num_landmarks=M_LANDMARKS,
        n_iter=N_ITER,
    )
    new_self.query = old_self.query
    new_self.key = new_self.query  # W_k = W_q enforced
    new_self.value = old_self.value
    layer.attention.self = new_self

# Apply capture monkey-patch
registry = apply_soft_monkey_patch(model)

# Save Gaussian table
save_gaussian_tables(out_dir)

# Tokenize: need exactly 7 valid tokens (1 padding).
# [CLS] + 5 wordpiece tokens + [SEP] = 7 valid, 1 [PAD].
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased", model_max_length=S,
)
text = "I like my red hat"
enc = tokenizer(
    [text], padding="max_length", max_length=S,
    truncation=True, return_tensors="pt",
)
mask_np = enc["attention_mask"].numpy()
n_valid = int(np.sum(mask_np == 1))
n_pad = int(np.sum(mask_np == 0))
print(f"S={S}, m={M_LANDMARKS}, seg={S // M_LANDMARKS}, "
      f"valid={n_valid}, pad={n_pad}")
print(f"tokens={enc['input_ids'][0].tolist()}")
print(f"mask={mask_np[0].tolist()}")

if n_pad != 1:
    print(f"ERROR: expected 1 padding token, got {n_pad}.")
    print("Change the text to produce exactly 5 wordpiece tokens.")
    sys.exit(1)

# Forward
with torch.no_grad():
    outputs = model(**enc)

# Process captured attention
captured = registry.all()
for layer_name, attn_data in captured.items():
    safe_name = layer_name.replace(".", "_")
    debug_path = os.path.join(
        out_dir, f"{os.path.basename(out_dir)}-batch0-{safe_name}.npz",
    )
    np.savez_compressed(debug_path, mask=mask_np, **attn_data)

    witness_path = debug_path.replace(".npz", ".witness.npz")
    result = process_soft_attn_debug_npz(
        debug_path, witness_path, k_bits=26, n_iter=N_ITER,
    )
    print(f"\n  {safe_name}: CosSim={result['cosine_similarity']:.8f}")
    print(f"  MP cond1_max={result['cond1_max']}, "
          f"cond2_max={result['cond2_max']}, "
          f"cond3_max={result['cond3_max']}, "
          f"cond4_max={result['cond4_max']}")

    # Append fixture epsilon (3x margin) to witness NPZ.
    # These are FIXTURE-ONLY values, not production values.
    # Ensure LOG_R has headroom: eps = max(3*raw, 2^(min_log_r - 1)).
    eps1 = max(int(result["cond1_max"]) * 3, 1)
    eps3 = max(int(result["cond3_max"]) * 3, 1 << 12)  # min LOG_R=13
    eps4 = max(int(result["cond4_max"]) * 3, 1)
    print(f"  Fixture eps: eps1={eps1}, eps3={eps3}, eps4={eps4}")
    existing = dict(np.load(witness_path, allow_pickle=True))
    existing["eps1"] = np.array(eps1, dtype=np.uint64)
    existing["eps3"] = np.array(eps3, dtype=np.uint64)
    existing["eps4"] = np.array(eps4, dtype=np.uint64)
    np.savez_compressed(witness_path, **existing)

print(f"\nOutput: {out_dir}/")
