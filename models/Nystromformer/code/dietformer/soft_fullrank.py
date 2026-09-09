"""SOFT A-gate full-rank measurement (125 valid, 3 padding).

Key question: does condition 2 divergence disappear with full-rank A?
If divergence vanishes -> cause was rank deficiency.
If divergence persists -> intrinsic property of 2nd-order Newton-Raphson.
"""
import os
import math
import numpy as np
import torch

from bert_attention_pipeline import load_small_bert
from gelu_swap import replace_activation_with_sigmoid_gelu
from soft_attention import SOFTSelfAttention, newton_raphson_pseudoinverse
from soft_attention_capture import apply_soft_monkey_patch
from soft_attention_pipeline import process_soft_attn_debug_npz
from gaussian_table import save_gaussian_tables
from transformers import BertTokenizerFast

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
HIDDEN = H * D  # 128
L = 2
N_ITER = 20
K_BITS = 26
SEED = 42
GAUSS_Z_DIST = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

out_dir = f"outputs_n{S}_m{M_LANDMARKS}_L{L}_h{HIDDEN}_soft_fullrank"
os.makedirs(out_dir, exist_ok=True)

# Build model
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
        seq_len=S, num_landmarks=M_LANDMARKS, n_iter=N_ITER,
    )
    new_self.query = old_self.query
    new_self.key = new_self.query
    new_self.value = old_self.value
    layer.attention.self = new_self

registry = apply_soft_monkey_patch(model)
save_gaussian_tables(out_dir, z_dist=GAUSS_Z_DIST)

# Tokenize
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=S)
enc = tokenizer(
    [BENCHMARK_TEXT], padding="max_length", max_length=S,
    truncation=True, return_tensors="pt",
)
mask_np = enc["attention_mask"].numpy()
n_valid = int(np.sum(mask_np == 1))
n_pad = int(np.sum(mask_np == 0))
print(f"S={S}, m={M_LANDMARKS}, valid={n_valid}, pad={n_pad}")
assert n_valid == 125 and n_pad == 3, f"Expected valid=125 pad=3, got valid={n_valid} pad={n_pad}"

# Forward
with torch.no_grad():
    outputs = model(**enc)

# Process layers
captured = registry.all()
layer_results = {}
for layer_name, attn_data in captured.items():
    safe_name = layer_name.replace(".", "_")
    debug_path = os.path.join(
        out_dir, f"{os.path.basename(out_dir)}-batch0-{safe_name}.npz",
    )
    np.savez_compressed(debug_path, mask=mask_np, **attn_data)
    witness_path = debug_path.replace(".npz", ".witness.npz")
    result = process_soft_attn_debug_npz(debug_path, witness_path, k_bits=K_BITS, n_iter=N_ITER)
    layer_results[safe_name] = (debug_path, witness_path, result)

# ======================================================================
# MEASUREMENTS
# ======================================================================
denom = 2.0 * math.sqrt(D)


def _cos(a, b):
    a_f = a.flatten().astype(np.float64)
    b_f = b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / (na * nb))


def _mae(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


for li, (layer_name, (debug_path, witness_path, result)) in enumerate(layer_results.items()):
    d = np.load(debug_path, allow_pickle=True)
    w = np.load(witness_path, allow_pickle=True)

    print(f"\n{'='*70}")
    print(f"LAYER {li}: {layer_name}")
    print(f"{'='*70}")

    # ── SVD and rank ──
    Q_landmarks = d['Q_landmarks']
    B_size, H_size, m_size, D_size = Q_landmarks.shape
    kernel_A = d['kernel_A']

    print(f"\n--- SVD / rank ---")
    for hi in range(H_size):
        lm = Q_landmarks[0, hi]
        norms = np.linalg.norm(lm, axis=-1)
        n_zero_lm = int(np.sum(norms < 1e-10))

        A_h = torch.tensor(kernel_A[0, hi].astype(np.float64))
        sv = torch.linalg.svdvals(A_h).numpy()
        n_zero_sv = int(np.sum(sv < 1e-10))
        rank = m_size - n_zero_sv
        min_sv = float(sv[-1])
        max_sv = float(sv[0])
        cond = max_sv / min_sv if min_sv > 1e-30 else float('inf')

        print(f"  Head {hi}: zero_lm={n_zero_lm}/{m_size}, "
              f"rank={rank}/{m_size}, min_sv={min_sv:.6e}, cond={cond:.2e}")
        print(f"    SV: {' '.join(f'{v:.4e}' for v in sv)}")

    # ── MP condition residuals ──
    print(f"\n--- MP condition residuals ---")
    Q_inv_A = w['Q_inv_A']
    Q_y_inv = w['Q_y_inv_intermediates']
    kernel_A_q = w['kernel_A']

    for hi in range(H_size):
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

        print(f"  Head {hi}:")
        print(f"    {'Cond':>5} | {'max|diff|':>10} | {'%4095':>8} | "
              f"{'epsilon':>8} | {'verdict':>10}")
        for cn, cv, eps in [("1", c1, 4096), ("2", c2, None),
                            ("3", c3, 512), ("4", c4, 32)]:
            pct = f"{cv/4095*100:.2f}%"
            if eps is not None:
                verdict = "PASS" if cv <= eps else f"FAIL(>{eps})"
            else:
                verdict = f"{'diverge' if cv > 4095 else 'ok'}"
            print(f"    {cn:>5} | {cv:>10} | {pct:>8} | "
                  f"{str(eps) if eps else 'N/A':>8} | {verdict:>10}")

    # ── Newton-Raphson convergence ──
    print(f"\n--- Newton-Raphson convergence ---")
    kernel_A_t = torch.tensor(kernel_A.astype(np.float64))
    for hi in range(H_size):
        A_t = kernel_A_t[0:1, hi:hi+1]
        norm_A = torch.norm(A_t, p=1, dim=(-2, -1), keepdim=True)
        alpha = 2.0 / (norm_A ** 2 + 1e-12)
        A_k = alpha * A_t

        print(f"  Head {hi}:")
        print(f"    {'t':>4} | {'||AAkA-A||/||A||':>20} | {'||AkAAk-Ak||/||Ak||':>20}")
        for t in range(N_ITER + 1):
            r1_mat = A_t @ A_k @ A_t - A_t
            nr1 = float(torch.norm(r1_mat) / (torch.norm(A_t) + 1e-12))
            r2_mat = A_k @ A_t @ A_k - A_k
            nr2 = float(torch.norm(r2_mat) / (torch.norm(A_k) + 1e-12))
            print(f"    {t:>4} | {nr1:>20.12e} | {nr2:>20.12e}")
            if t < N_ITER:
                A_k = 2 * A_k - A_k @ A_t @ A_k

    # ── dist distribution ──
    print(f"\n--- dist distribution ---")
    for dname, tidx_key in [("dist_B", "table_idx_B"), ("dist_A", "table_idx_A")]:
        dist_raw = d[f'{dname}_raw'].astype(np.float64)
        dist_scaled = dist_raw / denom
        tidx = w[tidx_key].astype(np.int64)
        print(f"  {dname}: min={np.min(dist_scaled):.6f}, max={np.max(dist_scaled):.6f}, "
              f"mean={np.mean(dist_scaled):.6f}")
        print(f"    table_idx: min={int(np.min(tidx))}, max={int(np.max(tidx))}")

    # ── Fidelity ──
    print(f"\n--- Fidelity ---")
    from quantize import get_quantization_params
    from gaussian_table import GAUSS_FRAC_BITS

    z_q = int(w['z_q'])

    # O-proj
    W_o = d['W_o']
    b_o_arr = d.get('b_o', None)
    if isinstance(b_o_arr, np.ndarray) and b_o_arr.dtype == object and b_o_arr.ndim == 0:
        b_o_arr = b_o_arr.item()
    attn_out = d['attn_output_for_proj']
    o_real = np.matmul(attn_out, W_o.T)
    if b_o_arr is not None:
        o_real = o_real + b_o_arr
    o_real_4d = o_real.reshape(B_size, S, H_size, D_size).transpose(0, 2, 1, 3)
    s_ro, z_ro = get_quantization_params(o_real_4d)
    z_raw_o = int(w['z_raw_o'])
    o_deq = (w['Q_raw_proj_o'].astype(np.float64) - z_raw_o) * s_ro
    cs_o = _cos(o_real_4d, o_deq)
    mae_o = _mae(o_real_4d, o_deq)

    # kernel fidelity
    kb_float = d['kernel_B']
    kb_deq = w['kernel_B'].astype(np.float64) / (1 << GAUSS_FRAC_BITS)
    cs_kb = _cos(kb_float, kb_deq)
    mae_kb = _mae(kb_float, kb_deq)

    ka_float = d['kernel_A']
    ka_deq = w['kernel_A'].astype(np.float64) / (1 << GAUSS_FRAC_BITS)
    cs_ka = _cos(ka_float, ka_deq)
    mae_ka = _mae(ka_float, ka_deq)

    print(f"  {'Stage':>12} | {'CosSim':>10} | {'MaxAbsErr':>12}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}")
    for sname, cs, mae_val in [
        ("kernel_B", cs_kb, mae_kb), ("kernel_A", cs_ka, mae_ka),
        ("O-proj", cs_o, mae_o),
    ]:
        print(f"  {sname:>12} | {cs:>10.6f} | {mae_val:>12.6e}")

    # ── ZEN equations ──
    print(f"\n--- ZEN equations ---")
    matmul_keys = ['cross_B', 'cross_A', 'MP_step1', 'MP_step2',
                   'MP_step2-1', 'MP_step2-2', 'temp', 'mid', 'Y']
    for mk in matmul_keys:
        q_key = f'Q_sim_{mk}' if mk not in ['cross_B', 'cross_A'] else f'Q_sim_{mk}'
        r_key = f'R_matmul_{mk}'
        if q_key not in w.files or r_key not in w.files:
            continue
        Q_sim = w[q_key].astype(np.int64)
        total = int(np.prod(Q_sim.shape))
        # Quick mismatch check from pipeline result
        print(f"  {mk}: {total} elements")

    # ── Gaussian consistency ──
    gauss_table = w['gauss_table']
    n_mis_B = int(np.sum(w['kernel_B'] != gauss_table[w['table_idx_B'].astype(int)]))
    n_mis_A = int(np.sum(w['kernel_A'] != gauss_table[w['table_idx_A'].astype(int)]))
    print(f"\n  Gaussian consistency: B={n_mis_B}, A={n_mis_A}")

    # ── Dist equation identity (with z_dist=1) ──
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
            n1 = w['norm_q_int'].astype(np.int64)
            n2 = w['norm_qtilde_int'].astype(np.int64)
            n1_exp = n1[:, :, :, np.newaxis].astype(object)
            n2_exp = n2[:, :, np.newaxis, :].astype(object)
        else:
            n1 = w['norm_qtilde_int'].astype(np.int64)
            n2 = w['norm_qtilde_int'].astype(np.int64)
            n1_exp = n1[:, :, :, np.newaxis].astype(object)
            n2_exp = n2[:, :, np.newaxis, :].astype(object)

        LHS = Q_dist.astype(object) * TWO_K + R_dist.astype(object)
        RHS = (M_A * (n1_exp + n2_exp)
               - 2 * M_B * (Q_cross.astype(object) - z_cross)
               + z_dist * TWO_K)

        total = int(np.prod(Q_dist.shape))
        n_mismatch = int(np.sum(LHS != RHS))
        n_neg = int(np.sum(LHS < 0))

        print(f"  [{kname}] total: {total}, mismatch: {n_mismatch}, negatives: {n_neg}")
        print(f"    Q_dist range: [{int(np.min(Q_dist))}, {int(np.max(Q_dist))}]")
        print(f"    z_dist: {z_dist}")

    # ── Masking ──
    print(f"\n--- Masking ---")
    pad_pos = np.where(mask_np[0] == 0)[0]
    z_q_val = int(w['z_q'])
    z_v_val = int(w['z_v'])
    q_masked = w['Q_q_masked']
    v_masked = w['Q_v_masked']
    q_match = int(np.sum(q_masked[0, :, pad_pos, :] == z_q_val))
    q_total = q_masked[0, :, pad_pos, :].size
    v_match = int(np.sum(v_masked[0, :, pad_pos, :] == z_v_val))
    v_total = v_masked[0, :, pad_pos, :].size
    print(f"  Q_q_masked padding={q_match}/{q_total}, Q_v_masked padding={v_match}/{v_total}")

print("\n" + "=" * 70)
print("CONDITION 2 JUDGMENT")
print("=" * 70)
print("Compare condition 2 max|diff| above with dense input (119 valid):")
print("  Dense: Head 0 c2=3834, Head 1 c2=3798 (both >4095 = diverged)")
print("If full-rank c2 << 4095 -> rank deficiency was the cause.")
print("If full-rank c2 >> 4095 -> 2nd-order NR intrinsic divergence.")
