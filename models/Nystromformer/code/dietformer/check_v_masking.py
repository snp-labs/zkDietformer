"""Check V masking effect on Linformer output."""
import math
import numpy as np
import torch
import torch.nn as nn
import random

from evaluation_fixtures import SAMPLE_TEXTS, MAX_SEQ_LEN
from transformers import BertForMaskedLM, BertTokenizerFast
from gelu_swap import replace_activation_with_sigmoid_gelu
from bert_attention_pipeline import load_small_bert
from linformer_attention import LinformerSelfAttention

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased", model_max_length=MAX_SEQ_LEN)

model = load_small_bert(hidden_size=128, num_attention_heads=4,
                        num_hidden_layers=2, seq_len=MAX_SEQ_LEN)
model.eval()
replace_activation_with_sigmoid_gelu(model, "bert")

# Replace with Linformer
from run_linformer_validation import replace_bert_attn_with_linformer
replace_bert_attn_with_linformer(model, MAX_SEQ_LEN, proj_dim=32)

enc = tokenizer(SAMPLE_TEXTS[:1], padding="max_length",
                max_length=MAX_SEQ_LEN, truncation=True,
                return_tensors="pt")
mask_np = enc["attention_mask"].numpy().astype(np.int64)
n_pad = int(np.sum(mask_np == 0))
n_valid = int(np.sum(mask_np == 1))
print(f"Padding positions: {n_pad}, Valid: {n_valid}")

# Get V, K_proc, Q_proc for layer 0
layer = model.bert.encoder.layer[0].attention.self
hidden = model.bert.embeddings(enc["input_ids"])
with torch.no_grad():
    q = layer.query(hidden).view(1, MAX_SEQ_LEN, 4, 32).transpose(1, 2)
    k = layer.key(hidden).view(1, MAX_SEQ_LEN, 4, 32).transpose(1, 2)
    v = layer.value(hidden).view(1, MAX_SEQ_LEN, 4, 32).transpose(1, 2)

    scale = 1.0 / math.sqrt(math.sqrt(32))
    mask_t = enc["attention_mask"].float().view(1, 1, MAX_SEQ_LEN, 1)
    k_proc = k * scale * mask_t
    q_proc = q * scale * mask_t

    # Case 1: V unmasked (current)
    fv_unmasked = torch.matmul(layer.F.unsqueeze(0), v)
    ek = torch.matmul(layer.E.unsqueeze(0), k_proc)
    logit = torch.matmul(q_proc, ek.transpose(-2, -1))
    p = torch.softmax(logit, dim=-1)
    x_unmasked = torch.matmul(p, fv_unmasked)

    # Case 2: V masked
    v_masked = v * mask_t
    fv_masked = torch.matmul(layer.F.unsqueeze(0), v_masked)
    x_masked = torch.matmul(p, fv_masked)

    # Compare
    diff = (x_unmasked - x_masked).numpy()
    x_u = x_unmasked.numpy()
    x_m = x_masked.numpy()

    cos = float(np.sum(x_u * x_m) /
                (np.linalg.norm(x_u) * np.linalg.norm(x_m) + 1e-30))
    mae = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    print(f"\nFV masked vs unmasked:")
    fv_d = (fv_unmasked - fv_masked).numpy()
    print(f"  FV max|diff|: {np.max(np.abs(fv_d)):.6e}")
    print(f"  FV CosSim: {float(np.sum(fv_unmasked.numpy()*fv_masked.numpy())/(np.linalg.norm(fv_unmasked.numpy())*np.linalg.norm(fv_masked.numpy())+1e-30)):.8f}")

    print(f"\nOutput x (P@FV):")
    print(f"  CosSim: {cos:.8f}")
    print(f"  MaxAbsErr: {mae:.6e}")
    print(f"  RMSE: {rmse:.6e}")

    # V padding values
    pad_idx = np.where(mask_np[0] == 0)[0]
    if len(pad_idx) > 0:
        v_pad = v[0, :, pad_idx, :].numpy()
        print(f"\nV at padding: max|V|={np.max(np.abs(v_pad)):.6e}, mean|V|={np.mean(np.abs(v_pad)):.6e}")
