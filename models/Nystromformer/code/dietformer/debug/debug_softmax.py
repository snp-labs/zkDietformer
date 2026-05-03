import numpy as np
import os


def _debug_softmax_kernel(data, keys, label, k_bits, frac_bits, k_segments):
    """Single softmax kernel verification.

    Args:
        data: loaded npz dict.
        keys: dict mapping canonical names to actual npz key names.
              Required: Q_sim, R_matmul, r_sm, limbs, exp_limbs,
                        Q_exp, R_exp, exp_total, exp_sum, P, R_div.
        label: display label (e.g. "BERT attn", "Nystrom k1").
    """
    print(f"\n--- Softmax Kernel: {label} ---")

    # [Step 1] Scaling & limb reconstruction
    X_recovered = (data[keys['Q_sim']].astype(object) << k_bits) + data[keys['R_matmul']].astype(object)
    M_sm = data['M_sm'].astype(object)
    r_sm = data[keys['r_sm']].astype(object)

    limbs = data[keys['limbs']]
    scaled_x_recovered = np.zeros_like(limbs[0], dtype=object)
    limb_bits = int(data['limb_bits'])
    for i in range(k_segments):
        scaled_x_recovered += limbs[i].astype(object) << (limb_bits * i)

    lhs_scale = X_recovered * M_sm
    rhs_scale = (scaled_x_recovered << k_bits) + r_sm
    ok_scale = np.array_equal(lhs_scale, rhs_scale)
    print(f"  [1] Scaling & Remainder:              {'PASS' if ok_scale else 'FAIL'}")

    # [Step 2] Exponential chain multiplication
    exp_limbs = data[keys['exp_limbs']]
    Q_exp = data[keys['Q_exp']]
    R_exp = data[keys['R_exp']]
    exp_total = data[keys['exp_total']].astype(object)

    current_acc = exp_limbs[0].astype(object)
    all_step_ok = True
    for k in range(k_segments - 1):
        multiplier = exp_limbs[k + 1].astype(object)
        lhs_mult = current_acc * multiplier

        if k == k_segments - 2:
            target_q = exp_total
        else:
            target_q = Q_exp[k].astype(object)

        rhs_mult = (target_q << frac_bits) + R_exp[k].astype(object)
        if not np.array_equal(lhs_mult, rhs_mult):
            all_step_ok = False
            print(f"  [2] Exp Chain Step {k}:                FAIL")
            break
        current_acc = target_q

    if all_step_ok:
        print(f"  [2] Exp Chain Multiplication:         PASS")

    # [Step 3] Softmax division: P * exp_sum + R_div == exp_total << frac_bits
    exp_sum = data[keys['exp_sum']].astype(object)
    P = data[keys['P']].astype(object)
    R_div = data[keys['R_div']].astype(object)

    numerator = exp_total << frac_bits
    div_check = P * exp_sum + R_div
    ok_div = np.array_equal(numerator, div_check)
    print(f"  [3] Softmax Division (P*sum + R):     {'PASS' if ok_div else 'FAIL'}")

    return ok_scale and all_step_ok and ok_div


def debug_softmax_witness(npz_path):
    print(f"=== Softmax Debugging: {os.path.basename(npz_path)} ===")
    if not os.path.exists(npz_path):
        print("File not found.")
        return

    data = np.load(npz_path)
    k_bits = int(data['k_bits'])
    frac_bits = int(data['frac_bits'])
    k_segments = int(data['k_segments'])

    # Auto-detect: BERT (Q_sim_attn) vs Nystrom (Q_sim_k1)
    is_bert = 'Q_sim_attn' in data
    is_nystrom = 'Q_sim_k1' in data

    if not is_bert and not is_nystrom:
        print("Unknown witness format: neither Q_sim_attn nor Q_sim_k1 found.")
        return

    all_ok = True

    if is_bert:
        keys = {
            'Q_sim': 'Q_sim_attn', 'R_matmul': 'R_matmul_attn',
            'r_sm': 'r_sm', 'limbs': 'limbs', 'exp_limbs': 'exp_limbs',
            'Q_exp': 'Q_exp', 'R_exp': 'R_exp', 'exp_total': 'exp_total',
            'exp_sum': 'exp_sum', 'P': 'P_attn', 'R_div': 'R_div_attn',
        }
        ok = _debug_softmax_kernel(data, keys, "BERT full-attention", k_bits, frac_bits, k_segments)
        all_ok = all_ok and ok

    if is_nystrom:
        for ki in range(1, 4):
            suffix = f"_k{ki}"
            keys = {
                'Q_sim': f'Q_sim_k{ki}', 'R_matmul': f'R_matmul_k{ki}',
                'r_sm': f'r_sm_k{ki}', 'limbs': f'limbs_k{ki}',
                'exp_limbs': f'exp_limbs_k{ki}',
                'Q_exp': f'Q_exp_k{ki}', 'R_exp': f'R_exp_k{ki}',
                'exp_total': f'exp_total_k{ki}', 'exp_sum': f'exp_sum_k{ki}',
                'P': f'P_k{ki}', 'R_div': f'R_div_k{ki}',
            }
            ok = _debug_softmax_kernel(data, keys, f"Nystrom kernel {ki}", k_bits, frac_bits, k_segments)
            all_ok = all_ok and ok

    print(f"\n{'=== ALL SOFTMAX CHECKS PASSED ===' if all_ok else '=== SOME CHECKS FAILED ==='}")


if __name__ == "__main__":
    # BERT
    bert_witness = "../outputs_bert/outputs_bert-batch0-bert.encoder.layer.0.attention.self.witness.npz"
    if os.path.exists(bert_witness):
        debug_softmax_witness(bert_witness)

    # Nystrom
    nystrom_witness = "../outputs_m64/outputs_m64-batch0-nystromformer.encoder.layer.0.attention.self.witness.npz"
    if os.path.exists(nystrom_witness):
        print("\n")
        debug_softmax_witness(nystrom_witness)
