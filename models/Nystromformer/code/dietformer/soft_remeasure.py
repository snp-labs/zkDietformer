"""SOFT A-gate remeasurement with dense input (119 valid, 9 padding).

Measures: A condition number, MP residuals, Newton convergence,
dist distribution, fidelity, masking effectiveness.
Does NOT re-measure integrity items (ZEN, Gaussian consistency, recovery, clamping).
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
    'remarkable state of the art results across a wide range of challenging benchmarks.'
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

torch.manual_seed(SEED)
np.random.seed(SEED)

out_dir = f"outputs_n{S}_m{M_LANDMARKS}_L{L}_h{HIDDEN}_soft_dense"
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
save_gaussian_tables(out_dir)

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
print(f"Tokens: {tokenizer.convert_ids_to_tokens(enc['input_ids'][0].tolist()[:5])} ... "
      f"{tokenizer.convert_ids_to_tokens(enc['input_ids'][0].tolist()[n_valid-3:n_valid+2])}")

# B-3: Position embedding check
max_pos = model.config.max_position_embeddings
print(f"\n=== B-3: Position embedding ===")
print(f"max_position_embeddings={max_pos}")
print(f"Current S={S} <= {max_pos}: OK")
print(f"For S=1024: {'OK' if max_pos >= 1024 else 'NEED RESIZE'}")
print(f"For S=2048: {'OK' if max_pos >= 2048 else 'NEED RESIZE'}")

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
# DETAILED MEASUREMENTS
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

    # ── C-1: A condition number ──
    Q_landmarks = d['Q_landmarks']  # [B, H, m, D]
    B_size, H_size, m_size, D_size = Q_landmarks.shape
    kernel_A = d['kernel_A']

    print(f"\n--- C-1: A condition number ---")
    for hi in range(H_size):
        lm = Q_landmarks[0, hi]
        norms = np.linalg.norm(lm, axis=-1)
        n_zero_lm = int(np.sum(norms < 1e-10))

        n_dup = 0
        for i in range(m_size):
            for j in range(i+1, m_size):
                if np.linalg.norm(lm[i] - lm[j]) < 1e-10:
                    n_dup += 1

        A_h = torch.tensor(kernel_A[0, hi].astype(np.float64))
        sv = torch.linalg.svdvals(A_h).numpy()
        n_zero_sv = int(np.sum(sv < 1e-10))
        rank = m_size - n_zero_sv
        min_sv = float(sv[-1])
        max_sv = float(sv[0])
        cond = max_sv / min_sv if min_sv > 1e-30 else float('inf')

        print(f"  Head {hi}: zero_lm={n_zero_lm}/{m_size}, dup_pairs={n_dup}, "
              f"rank={rank}/{m_size}, min_sv={min_sv:.6e}, cond={cond:.2e}")
        print(f"    SV: {' '.join(f'{v:.4e}' for v in sv)}")

        if min_sv < 1e-6:
            print(f"  STOP: min_sv={min_sv:.6e} < 1e-6")

    # ── C-2: MP condition residuals ──
    print(f"\n--- C-2: MP condition residuals ---")
    Q_inv_A = w['Q_inv_A']
    Q_y_inv = w['Q_y_inv_intermediates']  # [4, B, H, m, m]
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
              f"{'3x':>8} | {'5x':>8} | {'prev':>8}")
        for cn, cv, pv in [("1", c1, "1684/1428"),
                           ("2", c2, "3834/3798"),
                           ("3", c3, "74/137"),
                           ("4", c4, "89/117")]:
            pct = f"{cv/4095*100:.2f}%"
            x3 = cv * 3
            x5 = cv * 5
            over = " [>4095]" if x3 > 4095 else ""
            print(f"    {cn:>5} | {cv:>10} | {pct:>8} | "
                  f"{x3:>8}{over} | {x5:>8} | {pv:>8}")

    # ── C-3: Newton-Raphson convergence ──
    print(f"\n--- C-3: Newton-Raphson convergence ---")
    kernel_A_t = torch.tensor(kernel_A.astype(np.float64))
    for hi in range(H_size):
        A_t = kernel_A_t[0:1, hi:hi+1]  # [1, 1, m, m]
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

    # ── C-4: dist distribution ──
    print(f"\n--- C-4: dist distribution ---")
    for dname, tidx_key in [("dist_B", "table_idx_B"), ("dist_A", "table_idx_A")]:
        dist_raw = d[f'{dname}_raw'].astype(np.float64)
        dist_scaled = dist_raw / denom
        tidx = w[tidx_key].astype(np.int64)
        n_unique = len(np.unique(tidx))
        print(f"  {dname}: min={np.min(dist_scaled):.6f}, max={np.max(dist_scaled):.6f}, "
              f"mean={np.mean(dist_scaled):.6f}")
        print(f"    table_idx: min={int(np.min(tidx))}, max={int(np.max(tidx))}, "
              f"unique={n_unique}/65536 ({n_unique/65536*100:.3f}%)")

    # ── C-5: Fidelity ──
    print(f"\n--- C-5: Fidelity ---")
    s_q = float(w['z_q'])  # need scale, not zero-point
    # Recompute scales from witness
    z_q = int(w['z_q'])
    z_cb = int(w['z_sim_cross_B'])
    z_ca = int(w['z_sim_cross_A'])
    z_inv = int(w['z_inv'])
    z_temp = int(w['z_sim_temp'])
    z_mid = int(w['z_sim_mid'])
    z_Y = int(w['z_sim_Y'])
    z_raw_o = int(w['z_raw_o'])

    # For fidelity we compare float reference with dequantized witness.
    # The pipeline already printed fidelity; we just need to format comparison.
    # Let's re-derive from the data.
    from quantize import get_quantization_params

    # landmarks
    lm_float = d['Q_landmarks']
    Q_lm_q = w['Q_landmarks_q']
    s_proc, z_proc = get_quantization_params(d['Q_proc'])
    lm_deq = (Q_lm_q.astype(np.float64) - z_proc) * s_proc
    cs_lm = _cos(lm_float, lm_deq)
    mae_lm = _mae(lm_float, lm_deq)

    # cross_B
    cb_float = d['cross_B']
    s_cb_f, z_cb_f = get_quantization_params(cb_float)
    cb_deq = (w['Q_sim_cross_B'].astype(np.float64) - z_cb) * s_cb_f
    cs_cb = _cos(cb_float, cb_deq)
    mae_cb = _mae(cb_float, cb_deq)

    # cross_A
    ca_float = d['cross_A']
    s_ca_f, z_ca_f = get_quantization_params(ca_float)
    ca_deq = (w['Q_sim_cross_A'].astype(np.float64) - z_ca) * s_ca_f
    cs_ca = _cos(ca_float, ca_deq)
    mae_ca = _mae(ca_float, ca_deq)

    # dist_B, dist_A
    dist_B_float = d['dist_B_raw'].astype(np.float64) / denom
    s_db, z_db = get_quantization_params(np.maximum(dist_B_float, 0.0))
    db_deq = (w['Q_dist_B'].astype(np.float64) - z_db) * s_db
    cs_db = _cos(dist_B_float, db_deq)
    mae_db = _mae(dist_B_float, db_deq)

    dist_A_float = d['dist_A_raw'].astype(np.float64) / denom
    s_da, z_da = get_quantization_params(np.maximum(dist_A_float, 0.0))
    da_deq = (w['Q_dist_A'].astype(np.float64) - z_da) * s_da
    cs_da = _cos(dist_A_float, da_deq)
    mae_da = _mae(dist_A_float, da_deq)

    # kernel_B, kernel_A
    from gaussian_table import GAUSS_FRAC_BITS
    kb_float = d['kernel_B']
    kb_deq = w['kernel_B'].astype(np.float64) / (1 << GAUSS_FRAC_BITS)
    cs_kb = _cos(kb_float, kb_deq)
    mae_kb = _mae(kb_float, kb_deq)

    ka_float = d['kernel_A']
    ka_deq = w['kernel_A'].astype(np.float64) / (1 << GAUSS_FRAC_BITS)
    cs_ka = _cos(ka_float, ka_deq)
    mae_ka = _mae(ka_float, ka_deq)

    # inv_A
    inv_float = d['inv_A']
    s_inv_f, z_inv_f = get_quantization_params(inv_float)
    inv_deq = (w['Q_inv_A'].astype(np.float64) - z_inv) * s_inv_f
    cs_inv = _cos(inv_float, inv_deq)
    mae_inv = _mae(inv_float, inv_deq)

    # temp
    temp_float = d['temp']
    s_temp_f, z_temp_f = get_quantization_params(temp_float)
    temp_deq = (w['Q_sim_temp'].astype(np.float64) - z_temp) * s_temp_f
    cs_temp = _cos(temp_float, temp_deq)
    mae_temp = _mae(temp_float, temp_deq)

    # mid
    mid_float = d['mid']
    s_mid_f, z_mid_f = get_quantization_params(mid_float)
    mid_deq = (w['Q_sim_mid'].astype(np.float64) - z_mid) * s_mid_f
    cs_mid = _cos(mid_float, mid_deq)
    mae_mid = _mae(mid_float, mid_deq)

    # Y
    Y_float = d['Y']
    s_Y_f, z_Y_f = get_quantization_params(Y_float)
    Y_deq = (w['Q_sim_Y'].astype(np.float64) - z_Y) * s_Y_f
    cs_Y = _cos(Y_float, Y_deq)
    mae_Y = _mae(Y_float, Y_deq)

    # O-proj
    W_o = d['W_o']
    b_o_arr = d.get('b_o', None)
    if isinstance(b_o_arr, np.ndarray) and b_o_arr.dtype == object and b_o_arr.ndim == 0:
        b_o_arr = b_o_arr.item()
    attn_out = d['attn_output_for_proj']
    o_real = np.matmul(attn_out, W_o.T)
    if b_o_arr is not None:
        o_real = o_real + b_o_arr
    B_s = o_real.shape[0]
    S_s = o_real.shape[1]
    H_s = H_size
    D_s = D_size
    o_real_4d = o_real.reshape(B_s, S_s, H_s, D_s).transpose(0, 2, 1, 3)
    s_ro, z_ro = get_quantization_params(o_real_4d)
    o_deq = (w['Q_raw_proj_o'].astype(np.float64) - z_raw_o) * s_ro
    cs_o = _cos(o_real_4d, o_deq)
    mae_o = _mae(o_real_4d, o_deq)

    # Previous measurements (from first run with 11 valid tokens)
    prev = {
        "landmarks": 0.287, "cross_B": 0.080, "cross_A": 0.084,
        "dist_B": 0.454, "dist_A": 0.496,
        "kernel_B": 0.99998, "kernel_A": 0.99999,
        "inv_A": 1.00000, "temp": 0.916, "mid": 0.99995,
        "Y": 0.916, "O-proj": 0.916,
    }

    print(f"  {'Stage':>12} | {'CosSim':>10} | {'MaxAbsErr':>12} | {'prev CosSim':>12}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    for sname, cs, mae_val in [
        ("landmarks", cs_lm, mae_lm), ("cross_B", cs_cb, mae_cb),
        ("cross_A", cs_ca, mae_ca), ("dist_B", cs_db, mae_db),
        ("dist_A", cs_da, mae_da), ("kernel_B", cs_kb, mae_kb),
        ("kernel_A", cs_ka, mae_ka), ("inv_A", cs_inv, mae_inv),
        ("temp", cs_temp, mae_temp), ("mid", cs_mid, mae_mid),
        ("Y", cs_Y, mae_Y), ("O-proj", cs_o, mae_o),
    ]:
        pv = prev.get(sname, "N/A")
        pvs = f"{pv:.3f}" if isinstance(pv, float) else pv
        print(f"  {sname:>12} | {cs:>10.6f} | {mae_val:>12.6e} | {pvs:>12}")

    if cs_o < 0.9:
        print(f"  STOP: O-proj CosSim = {cs_o:.6f} < 0.9")

    # ── C-6: Masking constraint effectiveness ──
    print(f"\n--- C-6: Masking effectiveness ---")
    Q_proc = d['Q_proc']  # [B, H, S, D] float
    V_raw = d['V']  # [B, H, S, D] float (before masking)
    # Check padding positions
    pad_positions = np.where(mask_np[0] == 0)[0]
    print(f"  Padding positions: {len(pad_positions)} (first: {pad_positions[0] if len(pad_positions)>0 else 'N/A'}, "
          f"last: {pad_positions[-1] if len(pad_positions)>0 else 'N/A'})")

    # Q_proc at padding should be 0 (float domain)
    q_proc_pad = Q_proc[0, :, pad_positions, :]  # [H, n_pad, D]
    q_proc_pad_max = float(np.max(np.abs(q_proc_pad)))
    print(f"  Q_proc at padding: max|val|={q_proc_pad_max:.6e} (should be 0)")

    # Check if masking changed values: compare Q before and after masking
    # Q_proc = Q_raw * scale * mask, so at padding positions it should be 0
    # V is pre-masked, so V at padding should also be 0
    # Since capture stores post-masking V as 'V', let me check Q_proc non-pad vs pad
    q_proc_valid = Q_proc[0, :, :n_valid, :]
    q_proc_valid_max = float(np.max(np.abs(q_proc_valid)))
    print(f"  Q_proc at valid: max|val|={q_proc_valid_max:.6e}")
    print(f"  Masking ratio: valid max / pad max = "
          f"{'inf' if q_proc_pad_max < 1e-30 else f'{q_proc_valid_max/q_proc_pad_max:.0f}'}")

    # Quantized domain: check Q_q at padding = z_q
    Q_q_w = w['Q_q']  # [B, H, S, D]
    z_q_val = int(w['z_q'])
    q_q_pad = Q_q_w[0, :, pad_positions, :]
    n_at_zq = int(np.sum(q_q_pad == z_q_val))
    n_pad_total = q_q_pad.size
    print(f"  Q_q at padding == z_q({z_q_val}): {n_at_zq}/{n_pad_total}")

    Q_v_w = w['Q_v']
    z_v_val = int(w['z_v'])
    q_v_pad = Q_v_w[0, :, pad_positions, :]
    n_at_zv = int(np.sum(q_v_pad == z_v_val))
    n_pad_v_total = q_v_pad.size
    print(f"  Q_v at padding == z_v({z_v_val}): {n_at_zv}/{n_pad_v_total}")


if __name__ == "__main__":
    pass  # All code runs at module level
