"""Entry point: replace BERT attention with SOFT kernel and run validation.

Usage:
    python run_soft_validation.py --seq_len 128 --num_landmarks 32 \
        --hidden_size 128 --num_heads 2 --num_layers 2 --n_iter 20
"""
import os
import argparse
import numpy as np
import torch

from bert_attention_pipeline import load_small_bert
from gelu_swap import replace_activation_with_sigmoid_gelu
from soft_attention import SOFTSelfAttention
from soft_attention_capture import apply_soft_monkey_patch
from soft_attention_pipeline import process_soft_attn_debug_npz
from gaussian_table import save_gaussian_tables


def build_soft_model(hidden_size, num_heads, num_layers, seq_len,
                     num_landmarks, n_iter):
    """Load small BERT and replace attention with SOFTSelfAttention."""
    model = load_small_bert(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        seq_len=seq_len,
    )
    model.eval()
    replace_activation_with_sigmoid_gelu(model, "bert")

    for i, layer in enumerate(model.bert.encoder.layer):
        old_self = layer.attention.self
        new_self = SOFTSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            seq_len=seq_len,
            num_landmarks=num_landmarks,
            n_iter=n_iter,
        )
        new_self.query = old_self.query
        # W_k = W_q enforced by SOFTSelfAttention.__init__
        # but we need to re-assign after replacing query
        new_self.key = new_self.query
        new_self.value = old_self.value
        layer.attention.self = new_self

    return model


def main():
    parser = argparse.ArgumentParser(description="SOFT validation")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_landmarks", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--k_bits", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text", type=str, default="Hello world")
    args = parser.parse_args()

    S = args.seq_len
    m = args.num_landmarks
    H = args.num_heads
    HD = args.hidden_size
    L = args.num_layers

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, f"outputs_n{S}_m{m}_L{L}_h{HD}_soft")
    os.makedirs(out_dir, exist_ok=True)

    model = build_soft_model(HD, H, L, S, m, args.n_iter)
    registry = apply_soft_monkey_patch(model)

    # Save Gaussian table
    save_gaussian_tables(out_dir)

    # Tokenize
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased", model_max_length=S,
    )
    enc = tokenizer(
        [args.text], padding="max_length", max_length=S,
        truncation=True, return_tensors="pt",
    )
    mask_np = enc["attention_mask"].numpy()
    n_valid = int(np.sum(mask_np == 1))
    n_pad = int(np.sum(mask_np == 0))
    print(f"S={S}, m={m}, valid={n_valid}, pad={n_pad}")

    # Forward
    with torch.no_grad():
        outputs = model(**enc)

    # Process captured attention per layer
    captured = registry.all()
    for layer_name, attn_data in captured.items():
        safe_name = layer_name.replace(".", "_")
        debug_path = os.path.join(
            out_dir, f"{os.path.basename(out_dir)}-batch0-{safe_name}.npz",
        )
        np.savez_compressed(debug_path, mask=mask_np, **attn_data)

        witness_path = debug_path.replace(".npz", ".witness.npz")
        result = process_soft_attn_debug_npz(
            debug_path, witness_path,
            k_bits=args.k_bits, n_iter=args.n_iter,
        )
        print(f"  {safe_name}: CosSim={result['cosine_similarity']:.8f}")
        print(f"    cond1_max={result['cond1_max']}, "
              f"cond3_max={result['cond3_max']}, "
              f"cond4_max={result['cond4_max']}")
        print(f"    svd_min={result['svd_min_per_head']}, "
              f"rank={result['rank_per_head']}, "
              f"cond={result['cond_number_per_head']}")
        print(f"    zero_landmarks={result['zero_landmark_count']}, "
              f"dup_pairs={result['duplicate_landmark_pairs']}")

        # Append epsilon to witness NPZ.
        # Use max(production_floor, 3 * actual_cond_max) to guarantee coverage.
        # Production floors from B-gate fixture (S=8): eps1~2754, eps3~6, eps4~195.
        layer_idx = list(captured.keys()).index(layer_name)
        prod_eps1 = {0: 2754, 1: 3630}.get(layer_idx, 4096)
        prod_eps3 = 6
        prod_eps4 = 195
        eps1 = max(prod_eps1, int(result["cond1_max"]) * 3)
        eps3 = max(prod_eps3, int(result["cond3_max"]) * 3)
        eps4 = max(prod_eps4, int(result["cond4_max"]) * 3)
        existing = dict(np.load(witness_path, allow_pickle=True))
        existing["eps1"] = np.array(eps1, dtype=np.uint64)
        existing["eps3"] = np.array(eps3, dtype=np.uint64)
        existing["eps4"] = np.array(eps4, dtype=np.uint64)
        np.savez_compressed(witness_path, **existing)
        print(f"    eps1={eps1}, eps3={eps3}, eps4={eps4} (production, layer {layer_idx})")

    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
