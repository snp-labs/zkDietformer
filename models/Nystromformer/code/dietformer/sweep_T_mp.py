"""Sweep T=8..20: measure MP conditions in quantized integer domain.

Uses integer kernel_A from witness (table lookup output, fixed across T)
and recomputes pseudoinverse at each t in float, then quantizes.
"""
import numpy as np
import torch
from quantize import (
    get_quantization_params, quantize_tensor,
    get_m_constant, sign_bit_grouping_matmul,
)
from gaussian_table import GAUSS_FRAC_BITS

K_BITS = 26
H = 2


def mp_conditions_at_t(kernel_A_float, A_q_int, t_iter):
    """Compute MP conditions 1-4 for pseudoinverse at iteration t.

    Args:
        kernel_A_float: float kernel_A [B,H,m,m] for NR iteration
        A_q_int: integer kernel_A [B,H,m,m] from table lookup (fixed)
        t_iter: number of Newton-Raphson iterations
    """
    H_size = kernel_A_float.shape[1]

    # Newton-Raphson in float64
    A_t = torch.tensor(kernel_A_float.astype(np.float64))
    norm_A = torch.norm(A_t, p=1, dim=(-2, -1), keepdim=True)
    alpha = 2.0 / (norm_A ** 2 + 1e-12)
    A_k = alpha * A_t
    for _ in range(t_iter):
        A_k = 2 * A_k - A_k @ A_t @ A_k
    inv_A_float = A_k.numpy()

    # Quantize pseudoinverse
    s_inv, z_inv = get_quantization_params(inv_A_float)
    Q_inv_A = quantize_tensor(inv_A_float, s_inv, z_inv).astype(np.int64)

    z_gauss = 0
    s_gauss = 1.0 / (1 << GAUSS_FRAC_BITS)

    # MP chain matmuls using fixed integer A_q
    M1 = get_m_constant(s_gauss, s_inv, s_gauss, k=K_BITS)
    Q_y1, _, _ = sign_bit_grouping_matmul(
        A_q_int, z_gauss, Q_inv_A, z_inv, z_gauss, M1, K_BITS)

    M2 = get_m_constant(s_gauss, s_gauss, s_gauss, k=K_BITS)
    Q_y2, _, _ = sign_bit_grouping_matmul(
        Q_y1, z_gauss, A_q_int, z_gauss, z_gauss, M2, K_BITS)

    M3 = get_m_constant(s_inv, s_gauss, s_inv, k=K_BITS)
    Q_y3, _, _ = sign_bit_grouping_matmul(
        Q_inv_A, z_inv, A_q_int, z_gauss, z_inv, M3, K_BITS)

    M4 = get_m_constant(s_inv, s_inv, s_inv, k=K_BITS)
    Q_y4, _, _ = sign_bit_grouping_matmul(
        Q_y3, z_inv, Q_inv_A, z_inv, z_inv, M4, K_BITS)

    results = []
    for hi in range(H_size):
        a_h = A_q_int[0, hi].astype(np.int64)
        inv_h = Q_inv_A[0, hi].astype(np.int64)
        y2_h = Q_y2[0, hi].astype(np.int64)
        y4_h = Q_y4[0, hi].astype(np.int64)
        y1_h = Q_y1[0, hi].astype(np.int64)
        y3_h = Q_y3[0, hi].astype(np.int64)

        c1 = int(np.max(np.abs(y2_h - a_h)))
        c2 = int(np.max(np.abs(y4_h - inv_h)))
        c3 = int(np.max(np.abs(y1_h - y1_h.T)))
        c4 = int(np.max(np.abs(y3_h - y3_h.T)))
        results.append((c1, c2, c3, c4))
    return results


out_dir = "outputs_n128_m32_L2_h128_soft_fullrank"
layers = [
    "bert_encoder_layer_0_attention_self",
    "bert_encoder_layer_1_attention_self",
]

for li, lname in enumerate(layers):
    debug_path = f"{out_dir}/{out_dir}-batch0-{lname}.npz"
    witness_path = debug_path.replace(".npz", ".witness.npz")
    d = np.load(debug_path, allow_pickle=True)
    w = np.load(witness_path, allow_pickle=True)

    kernel_A_float = d['kernel_A']       # float reference
    A_q_int = w['kernel_A'].astype(np.int64)  # integer table output (fixed)

    print(f"\n{'='*80}")
    print(f"LAYER {li}")
    print(f"{'='*80}")

    print(f"\n  {'T':>3} | ", end="")
    for hi in range(H):
        print(f"  H{hi}: c1   c2   c3   c4   max | ", end="")
    print()
    print(f"  {'-'*3}-+-", end="")
    for hi in range(H):
        print(f"{'-'*31}-+-", end="")
    print()

    best_t = None
    best_max_all = 99999
    all_results = {}

    for t in range(8, 21):
        results = mp_conditions_at_t(kernel_A_float, A_q_int, t)
        all_results[t] = results
        max_all = 0
        print(f"  {t:>3} | ", end="")
        for hi, (c1, c2, c3, c4) in enumerate(results):
            cmax = max(c1, c2, c3, c4)
            max_all = max(max_all, cmax)
            print(f"  {c1:>4} {c2:>4} {c3:>4} {c4:>4} {cmax:>4} | ", end="")
        print()

        if max_all < best_max_all:
            best_max_all = max_all
            best_t = t

    print(f"\n  >>> Best T={best_t}, max(all conditions, all heads)={best_max_all}")

# Detailed percentage table
print(f"\n{'='*80}")
print(f"DETAILED (% of 4095)")
print(f"{'='*80}")

for li, lname in enumerate(layers):
    debug_path = f"{out_dir}/{out_dir}-batch0-{lname}.npz"
    witness_path = debug_path.replace(".npz", ".witness.npz")
    d = np.load(debug_path, allow_pickle=True)
    w = np.load(witness_path, allow_pickle=True)
    kernel_A_float = d['kernel_A']
    A_q_int = w['kernel_A'].astype(np.int64)

    print(f"\nLAYER {li}")
    print(f"  {'T':>3} | {'H':>2} | {'c1':>6} {'%':>5} | "
          f"{'c2':>6} {'%':>5} | {'c3':>6} {'%':>5} | "
          f"{'c4':>6} {'%':>5} | {'max':>6} {'%':>5}")
    print(f"  {'-'*3}-+-{'-'*2}-+-"
          f"{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for t in range(8, 21):
        results = mp_conditions_at_t(kernel_A_float, A_q_int, t)
        for hi, (c1, c2, c3, c4) in enumerate(results):
            cmax = max(c1, c2, c3, c4)
            print(f"  {t:>3} | {hi:>2} | "
                  f"{c1:>6} {c1/40.95:>4.1f}% | "
                  f"{c2:>6} {c2/40.95:>4.1f}% | "
                  f"{c3:>6} {c3/40.95:>4.1f}% | "
                  f"{c4:>6} {c4/40.95:>4.1f}% | "
                  f"{cmax:>6} {cmax/40.95:>4.1f}%")
