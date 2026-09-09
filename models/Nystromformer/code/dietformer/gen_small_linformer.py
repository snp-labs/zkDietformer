"""Generate a small Linformer witness for B-gate testing.

Fixture: S=8, k=4, H=2, D=4, L=1, B=1.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from transformers import BertForMaskedLM, BertTokenizerFast
from gelu_swap import replace_activation_with_sigmoid_gelu
from bert_attention_pipeline import load_small_bert
from linformer_attention import LinformerSelfAttention
from linformer_attention_capture import apply_linformer_monkey_patch
from linformer_attention_pipeline import process_linformer_attn_debug_npz
from softmax_table import save_softmax_tables_v3

S = 8
K_PROJ = 4
H = 2
D = 4
HIDDEN = H * D  # 8
L = 1

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

out_dir = f"outputs_n{S}_k{K_PROJ}_L{L}_h{HIDDEN}_linformer"
os.makedirs(out_dir, exist_ok=True)

# Build small model
model = load_small_bert(
    hidden_size=HIDDEN, num_attention_heads=H,
    num_hidden_layers=L, seq_len=S,
)
model.eval()
replace_activation_with_sigmoid_gelu(model, "bert")

# Replace attention with Linformer
for i, layer in enumerate(model.bert.encoder.layer):
    old_self = layer.attention.self
    new_self = LinformerSelfAttention(
        hidden_size=HIDDEN,
        num_attention_heads=H,
        seq_len=S,
        proj_dim=K_PROJ,
    )
    new_self.query = old_self.query
    new_self.key = old_self.key
    new_self.value = old_self.value
    layer.attention.self = new_self

# Apply capture monkey-patch
registry = apply_linformer_monkey_patch(model)

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
print(f"S={S}, valid={n_valid}, pad={n_pad}, tokens={enc['input_ids'][0].tolist()}")

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
    np.savez_compressed(debug_path, **attn_data)

    witness_path = debug_path.replace(".npz", ".witness.npz")
    result = process_linformer_attn_debug_npz(
        debug_path, witness_path, k_bits=26,
    )
    print(f"  {safe_name}: CosSim={result['cosine_similarity']:.8f}")

print(f"\nOutput: {out_dir}/")
