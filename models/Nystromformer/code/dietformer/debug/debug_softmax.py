import numpy as np
import os


def _debug_softmax_kernel_v3(data, keys, label, k_bits):
    print(f"\n--- Softmax Kernel (v3): {label} ---")
    frac_bits = int(data['frac_bits'])

    # [Step 1] Scaling: (Q_sim << k_bits + R_matmul) * M_sm == table_idx * 2^(2*k_bits) + r_sm
    Q_sim = data[keys['Q_sim']].astype(object)
    R_matmul = data[keys['R_matmul']].astype(object)
    M_sm = int(data[keys['M_sm']])
    table_idx = data[keys['table_idx']].astype(object)
    r_sm = data[keys['r_sm']].astype(object)

    X = (Q_sim << k_bits) + R_matmul
    lhs = X * M_sm
    rhs = (table_idx << (2 * k_bits)) + r_sm
    ok_scale = np.array_equal(lhs, rhs)
    if not ok_scale:
        mismatch = np.argwhere(lhs != rhs)
        print(f"  [1] Scaling:    FAIL  ({len(mismatch)} mismatches, first: {mismatch[0].tolist()})")
        idx = tuple(mismatch[0])
        print(f"      Q_sim={Q_sim[idx]}, table_idx={table_idx[idx]}, r_sm={r_sm[idx]}")
        print(f"      LHS={lhs[idx]}, RHS={rhs[idx]}")
    else:
        print(f"  [1] Scaling:    PASS")

    # [Step 2] Non-negative checks
    ok_nonneg_qsim = np.all(Q_sim >= 0)
    ok_nonneg_tidx = np.all(table_idx >= 0)
    ok_nonneg_rsm = np.all(r_sm >= 0)
    print(f"  [2] Q_sim >= 0: {'PASS' if ok_nonneg_qsim else 'FAIL (min=' + str(np.min(Q_sim)) + ')'}")
    print(f"      tidx >= 0:  {'PASS' if ok_nonneg_tidx else 'FAIL (min=' + str(np.min(table_idx)) + ')'}")
    print(f"      r_sm >= 0:  {'PASS' if ok_nonneg_rsm else 'FAIL (min=' + str(np.min(r_sm)) + ')'}")

    # [Step 3] Division: exp_total << frac_bits == P * exp_sum + R_div
    exp_total = data[keys['exp_total']].astype(object)
    exp_sum = data[keys['exp_sum']].astype(object)
    P = data[keys['P']].astype(object)
    R_div = data[keys['R_div']].astype(object)

    numerator = exp_total << frac_bits
    div_check = P * exp_sum + R_div
    ok_div = np.array_equal(numerator, div_check)
    if not ok_div:
        mismatch = np.argwhere(numerator != div_check)
        print(f"  [3] Division:   FAIL  ({len(mismatch)} mismatches, first: {mismatch[0].tolist()})")
    else:
        print(f"  [3] Division:   PASS")

    return ok_scale and ok_nonneg_qsim and ok_nonneg_tidx and ok_nonneg_rsm and ok_div


def debug_softmax_witness_v3(npz_path):
    print(f"=== Softmax v3 Debugging: {os.path.basename(npz_path)} ===")
    if not os.path.exists(npz_path):
        print("File not found.")
        return

    data = np.load(npz_path)
    k_bits = int(data['k_bits'])

    is_bert = 'Q_sim_attn' in data
    is_nystrom = 'Q_sim_k1' in data

    if not is_bert and not is_nystrom:
        print("Unknown witness format: neither Q_sim_attn nor Q_sim_k1 found.")
        return

    all_ok = True

    if is_bert:
        keys = {
            'Q_sim': 'Q_sim_attn', 'R_matmul': 'R_matmul_attn',
            'M_sm': 'M_sm', 'table_idx': 'table_idx', 'r_sm': 'r_sm',
            'exp_total': 'exp_total', 'exp_sum': 'exp_sum',
            'P': 'P_attn', 'R_div': 'R_div_attn',
        }
        ok = _debug_softmax_kernel_v3(data, keys, "BERT full-attention", k_bits)
        all_ok = all_ok and ok

    if is_nystrom:
        for ki in range(1, 4):
            keys = {
                'Q_sim': f'Q_sim_k{ki}', 'R_matmul': f'R_matmul_k{ki}',
                'M_sm': f'M_sm_k{ki}', 'table_idx': f'table_idx_k{ki}',
                'r_sm': f'r_sm_k{ki}',
                'exp_total': f'exp_total_k{ki}', 'exp_sum': f'exp_sum_k{ki}',
                'P': f'P_k{ki}', 'R_div': f'R_div_k{ki}',
            }
            ok = _debug_softmax_kernel_v3(data, keys, f"Nystrom kernel {ki}", k_bits)
            all_ok = all_ok and ok

    print(f"\n{'=== ALL SOFTMAX V3 CHECKS PASSED ===' if all_ok else '=== SOME CHECKS FAILED ==='}")


if __name__ == "__main__":
    import glob, sys

    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            debug_softmax_witness_v3(path)
    else:
        # Auto-detect witness files
        for pattern in [
            "../outputs_bert/*attention.self.witness.npz",
            "../outputs_n*_m*_*nystrom/*attention.self.witness.npz",
            "../outputs_m*/*attention.self.witness.npz",
        ]:
            for f in sorted(glob.glob(pattern)):
                debug_softmax_witness_v3(f)
                print()
