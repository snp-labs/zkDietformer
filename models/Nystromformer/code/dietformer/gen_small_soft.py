"""Generate a small SOFT witness for A-gate testing.

Fixture: S=8, m=4, H=2, D=4, L=1, B=1, n_iter=20.
"""
import os
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

out_dir = f"outputs_n{S}_m{M_LANDMARKS}_L{L}_h{HIDDEN}_soft"
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

# Tokenize a short input
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased", model_max_length=S,
)
text = "Hi"
enc = tokenizer(
    [text], padding="max_length", max_length=S,
    truncation=True, return_tensors="pt",
)
mask_np = enc["attention_mask"].numpy()
n_valid = int(np.sum(mask_np == 1))
n_pad = int(np.sum(mask_np == 0))
print(f"S={S}, m={M_LANDMARKS}, valid={n_valid}, pad={n_pad}, "
      f"tokens={enc['input_ids'][0].tolist()}")

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
    print(f"  {safe_name}: CosSim={result['cosine_similarity']:.8f}")

print(f"\nOutput: {out_dir}/")
