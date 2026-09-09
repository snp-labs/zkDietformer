"""Part 1-2: Re-generate witness with per-layer optimal T.

L0: T=12, L1: T=14 (from sweep_T_mp.py).
Verifies: fidelity, dist equation, Gaussian consistency, masking.
"""
import os
import math
import numpy as np
import torch

from bert_attention_pipeline import load_small_bert
from gelu_swap import replace_activation_with_sigmoid_gelu
from soft_attention import SOFTSelfAttention
from soft_attention_capture import apply_soft_monkey_patch
from soft_attention_pipeline import process_soft_attn_debug_npz
from gaussian_table import save_gaussian_tables, GAUSS_FRAC_BITS
from transformers import BertTokenizerFast
from quantize import get_quantization_params

BENCHMARK_TEXT = (
    'The Transformer architecture has revolutionized natural language processing '
    'since its introduction in the landmark Attention Is All You Need paper published '
    'by researchers at Google Brain. Unlike recurrent neural networks that process '
    'sequences step by step in a sequential manner, transformers leverage powerful '
    'self-attention mechanisms to efficiently capture long-range dependencies between '
    'all positions in a given input sequence simultaneously. This massively parallel '
    'processing capability enables significantly faster training on modern GPU and TPU '
    'hardware accelerators. Recent model variants such as BERT and GPT have achieved '
    'remarkable state of the art results across a wide range of challenging benchmarks. '
    'Models scale ever larger now.'
)

S = 128
M_LANDMARKS = 32
H = 2
D = 64
HIDDEN = H * D
L = 2
K_BITS = 26
SEED = 42
GAUSS_Z_DIST = 1

# Per-layer optimal T from sweep
N_ITER_PER_LAYER = [12, 14]

torch.manual_seed(SEED)
np.random.seed(SEED)

out_dir = f"outputs_n{S}_m{M_LANDMARKS}_L{L}_h{HIDDEN}_soft_optT"
os.makedirs(out_dir, exist_ok=True)

model = load_small_bert(
    hidden_size=HIDDEN, num_attention_heads=H,
    num_hidden_layers=L, seq_len=S,
)
model.eval()
replace_activation_with_sigmoid_gelu(model, "bert")

for i, layer in enumerate(model.bert.encoder.layer):
    old_self = layer.attention.self
    new_self = SOFTSelfAttention(
        hidden_size=HIDDEN, num_attention_heads=H,
        seq_len=S, num_landmarks=M_LANDMARKS,
        n_iter=N_ITER_PER_LAYER[i],
    )
    new_self.query = old_self.query
    new_self.key = new_self.query
    new_self.value = old_self.value
    layer.attention.self = new_self

registry = apply_soft_monkey_patch(model)
save_gaussian_tables(out_dir, z_dist=GAUSS_Z_DIST)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",
                                               model_max_length=S)
enc = tokenizer([BENCHMARK_TEXT], padding="max_length", max_length=S,
                truncation=True, return_tensors="pt")
mask_np = enc["attention_mask"].numpy()
n_valid = int(np.sum(mask_np == 1))
n_pad = int(np.sum(mask_np == 0))
print(f"S={S}, m={M_LANDMARKS}, valid={n_valid}, pad={n_pad}")
assert n_valid == 125 and n_pad == 3

with torch.no_grad():
    outputs = model(**enc)

captured = registry.all()
denom = 2.0 * math.sqrt(D)


def _cos(a, b):
    a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / (na * nb))


def _mae(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


layer_items = list(captured.items())
for li, (layer_name, attn_data) in enumerate(layer_items):
    safe_name = layer_name.replace(".", "_")
    n_iter_layer = N_ITER_PER_LAYER[li]
    debug_path = os.path.join(
        out_dir, f"{os.path.basename(out_dir)}-batch0-{safe_name}.npz")
    np.savez_compressed(debug_path, mask=mask_np, **attn_data)
    witness_path = debug_path.replace(".npz", ".witness.npz")
    result = process_soft_attn_debug_npz(
        debug_path, witness_path, k_bits=K_BITS, n_iter=n_iter_layer)

    d = np.load(debug_path, allow_pickle=True)
    w = np.load(witness_path, allow_pickle=True)

    print(f"\n{'='*70}")
    print(f"LAYER {li}: T={n_iter_layer}")
    print(f"{'='*70}")

    # SVD
    kernel_A = d['kernel_A']
    B_s, H_s, m_s, D_s = d['Q_landmarks'].shape
    print(f"\n--- SVD ---")
    for hi in range(H_s):
        sv = torch.linalg.svdvals(
            torch.tensor(kernel_A[0, hi].astype(np.float64))).numpy()
        rank = m_s - int(np.sum(sv < 1e-10))
        print(f"  H{hi}: rank={rank}/{m_s}, min_sv={float(sv[-1]):.6e}, "
              f"cond={float(sv[0]/sv[-1]) if sv[-1]>1e-30 else float('inf'):.2e}")

    # MP conditions
    print(f"\n--- MP conditions (quantized integer) ---")
    Q_inv_A = w['Q_inv_A']
    Q_y_inv = w['Q_y_inv_intermediates']
    kernel_A_q = w['kernel_A']

    for hi in range(H_s):
        A_qi = kernel_A_q[0, hi].astype(np.int64)
        inv_qi = Q_inv_A[0, hi].astype(np.int64)
        y1 = Q_y_inv[0, 0, hi].astype(np.int64)
        y2 = Q_y_inv[1, 0, hi].astype(np.int64)
        y3 = Q_y_inv[2, 0, hi].astype(np.int64)
        y4 = Q_y_inv[3, 0, hi].astype(np.int64)

        c1 = int(np.max(np.abs(y2 - A_qi)))
        c2 = int(np.max(np.abs(y4 - inv_qi)))
        c3 = int(np.max(np.abs(y1 - y1.T)))
        c4 = int(np.max(np.abs(y3 - y3.T)))
        cmax = max(c1, c2, c3, c4)
        print(f"  H{hi}: c1={c1} c2={c2} c3={c3} c4={c4} max={cmax} "
              f"({cmax/4095*100:.1f}%)")

    # Fidelity
    print(f"\n--- Fidelity ---")
    z_inv = int(w['z_inv'])
    inv_float = d['inv_A']
    s_inv_f, _ = get_quantization_params(inv_float)
    inv_deq = (Q_inv_A.astype(np.float64) - z_inv) * s_inv_f
    cs_inv = _cos(inv_float, inv_deq)
    mae_inv = _mae(inv_float, inv_deq)

    kb_deq = w['kernel_B'].astype(np.float64) / (1 << GAUSS_FRAC_BITS)
    cs_kb = _cos(d['kernel_B'], kb_deq)
    ka_deq = w['kernel_A'].astype(np.float64) / (1 << GAUSS_FRAC_BITS)
    cs_ka = _cos(d['kernel_A'], ka_deq)

    W_o = d['W_o']
    b_o_arr = d.get('b_o', None)
    if isinstance(b_o_arr, np.ndarray) and b_o_arr.dtype == object and b_o_arr.ndim == 0:
        b_o_arr = b_o_arr.item()
    attn_out = d['attn_output_for_proj']
    o_real = np.matmul(attn_out, W_o.T)
    if b_o_arr is not None:
        o_real = o_real + b_o_arr
    o_real_4d = o_real.reshape(B_s, S, H_s, D_s).transpose(0, 2, 1, 3)
    s_ro, _ = get_quantization_params(o_real_4d)
    z_raw_o = int(w['z_raw_o'])
    o_deq = (w['Q_raw_proj_o'].astype(np.float64) - z_raw_o) * s_ro
    cs_o = _cos(o_real_4d, o_deq)
    mae_o = _mae(o_real_4d, o_deq)

    print(f"  {'Stage':>12} | {'CosSim':>10} | {'MaxAbsErr':>12}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}")
    for sn, cs, mae in [("inv_A", cs_inv, mae_inv),
                         ("kernel_B", cs_kb, _mae(d['kernel_B'], kb_deq)),
                         ("kernel_A", cs_ka, _mae(d['kernel_A'], ka_deq)),
                         ("O-proj", cs_o, mae_o)]:
        print(f"  {sn:>12} | {cs:>10.8f} | {mae:>12.6e}")

    if cs_o < 0.99:
        print(f"  WARNING: O-proj CosSim={cs_o:.8f} < 0.99")

    # Dist equation identity
    print(f"\n--- Dist equation identity ---")
    TWO_K = 1 << K_BITS
    for kname in ["B", "A"]:
        Q_dist = w[f'Q_dist_{kname}'].astype(np.int64)
        R_dist = w[f'R_dist_{kname}'].astype(np.int64)
        z_dist = int(w[f'z_dist_{kname}'])
        M_A = int(w[f'M_A_{kname}'])
        M_B = int(w[f'M_B_{kname}'])
        Q_cross = w[f'Q_sim_cross_{kname}'].astype(np.int64)
        z_cross = int(w[f'z_sim_cross_{kname}'])

        if kname == "B":
            n1 = w['norm_q_int'][:, :, :, np.newaxis].astype(object)
            n2 = w['norm_qtilde_int'][:, :, np.newaxis, :].astype(object)
        else:
            n1 = w['norm_qtilde_int'][:, :, :, np.newaxis].astype(object)
            n2 = w['norm_qtilde_int'][:, :, np.newaxis, :].astype(object)

        LHS = Q_dist.astype(object) * TWO_K + R_dist.astype(object)
        RHS = (M_A * (n1 + n2)
               - 2 * M_B * (Q_cross.astype(object) - z_cross)
               + z_dist * TWO_K)
        total = int(np.prod(Q_dist.shape))
        n_mis = int(np.sum(LHS != RHS))
        print(f"  [{kname}] total={total}, mismatch={n_mis}, "
              f"Q_dist=[{int(np.min(Q_dist))},{int(np.max(Q_dist))}]")

    # Gaussian consistency
    gauss_table = w['gauss_table']
    n_mis_B = int(np.sum(
        w['kernel_B'] != gauss_table[w['table_idx_B'].astype(int)]))
    n_mis_A = int(np.sum(
        w['kernel_A'] != gauss_table[w['table_idx_A'].astype(int)]))
    print(f"\n  Gaussian consistency: B={n_mis_B}, A={n_mis_A}")

    # Masking
    pad_pos = np.where(mask_np[0] == 0)[0]
    z_q_val = int(w['z_q'])
    z_v_val = int(w['z_v'])
    q_m = w['Q_q_masked'][0, :, pad_pos, :]
    v_m = w['Q_v_masked'][0, :, pad_pos, :]
    print(f"  Masking: Q_q={int(np.sum(q_m==z_q_val))}/{q_m.size}, "
          f"Q_v={int(np.sum(v_m==z_v_val))}/{v_m.size}")

    # ZEN equations
    print(f"\n--- ZEN equations ---")
    matmul_keys = ['cross_B', 'cross_A', 'MP_step1', 'MP_step2',
                   'MP_step2-1', 'MP_step2-2', 'temp', 'mid', 'Y']
    for mk in matmul_keys:
        q_key = f'Q_sim_{mk}'
        r_key = f'R_matmul_{mk}'
        if q_key not in w.files or r_key not in w.files:
            continue
        Q_sim = w[q_key].astype(np.int64)
        print(f"  {mk}: {int(np.prod(Q_sim.shape))} elements")

    # Newton-Raphson convergence
    print(f"\n--- Newton-Raphson (T={n_iter_layer}) ---")
    A_t = torch.tensor(kernel_A.astype(np.float64))
    for hi in range(H_s):
        At = A_t[0:1, hi:hi+1]
        norm_A = torch.norm(At, p=1, dim=(-2, -1), keepdim=True)
        alpha = 2.0 / (norm_A ** 2 + 1e-12)
        Ak = alpha * At
        for t in range(n_iter_layer + 1):
            r1 = float(torch.norm(At @ Ak @ At - At) /
                       (torch.norm(At) + 1e-12))
            r2 = float(torch.norm(Ak @ At @ Ak - Ak) /
                       (torch.norm(Ak) + 1e-12))
            if t == n_iter_layer:
                print(f"  H{hi} t={t}: ||AAkA-A||/||A||={r1:.6e}, "
                      f"||AkAAk-Ak||/||Ak||={r2:.6e}")
            if t < n_iter_layer:
                Ak = 2 * Ak - Ak @ At @ Ak
