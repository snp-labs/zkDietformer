import numpy as np
import os
import glob

def debug_bert_normalization(witness_path, k_bits=22, frac_bits=16):
    print(f"=== BERT Normalization Circuit Debugging: {os.path.basename(witness_path)} ===")
    sim = np.load(witness_path, allow_pickle=True)

    # [1] LayerNorm 검증 (ln1: post-attention, ln2: post-FFN)
    for suffix in ['ln1', 'ln2']:
        if f'Q_ln_output_sim_{suffix}' not in sim: continue
        print(f"\n[Step: {suffix}] LayerNorm Verification:")

        Q_ri = sim[f'Q_ri_{suffix}'].astype(object)
        z_ri = int(sim[f'z_ri_{suffix}'])
        dim = Q_ri.shape[-1]

        # Eq 1: Mean — sum_x = Q_mean * dim + R_mean
        sum_x = sim[f'sum_x_{suffix}'].astype(object)
        Q_mean = sim[f'Q_mean_{suffix}'].astype(object)
        R_mean = sim[f'R_mean_{suffix}'].astype(object)
        ok_mean = np.array_equal(sum_x, Q_mean * dim + R_mean)
        print(f"  - Mean Constraint (sum = Q*d + R):    {'PASS' if ok_mean else 'FAIL'}")

        # Eq 2: Q_diff = Q_ri - Q_mean
        Q_diff = sim[f'Q_diff_{suffix}'].astype(object)
        ok_diff = np.array_equal(Q_diff, Q_ri - Q_mean)
        print(f"  - Q_diff = Q_ri - Q_mean:             {'PASS' if ok_diff else 'FAIL'}")

        # Eq 3: Variance rescale — M_var * (sum_sq + M_prime_var) = (shifted_val_var << k) + R_shift_var
        M_var = int(sim[f'M_var_{suffix}'])
        M_prime_var = int(sim[f'M_prime_var_{suffix}'])
        sum_sq = sim[f'sum_sq_{suffix}'].astype(object)
        shifted_val_var = sim[f'shifted_val_var_{suffix}'].astype(object)
        R_shift_var = sim[f'R_shift_var_{suffix}'].astype(object)

        lhs_var = M_var * (sum_sq + M_prime_var)
        rhs_var = (shifted_val_var << k_bits) + R_shift_var
        ok_var = np.array_equal(lhs_var, rhs_var)
        print(f"  - Variance Rescale Constraint:        {'PASS' if ok_var else 'FAIL'}")

        # Eq 4: Var division — shifted_val_var = Q_var_sim * dim + R_var
        Q_var_sim = sim[f'Q_var_sim_{suffix}'].astype(object)
        R_var = sim[f'R_var_{suffix}'].astype(object)
        ok_var_div = np.array_equal(shifted_val_var, Q_var_sim * dim + R_var)
        print(f"  - Var Division (shifted = Q*d + R):   {'PASS' if ok_var_div else 'FAIL'}")

        # Eq 5: Beta lifting — M_beta_lift * (Q_beta - z_beta) = (Q_beta_lifted << k) + R_beta_lift
        M_beta_lift = int(sim[f'M_beta_lift_{suffix}'])
        Q_beta = sim[f'Q_beta_{suffix}'].astype(object)
        z_beta = int(sim[f'z_beta_{suffix}'])
        Q_beta_lifted = sim[f'Q_beta_lifted_{suffix}'].astype(object)
        R_beta_lift = sim[f'R_beta_lift_{suffix}'].astype(object)

        lhs_beta = M_beta_lift * (Q_beta - z_beta)
        rhs_beta = (Q_beta_lifted << k_bits) + R_beta_lift
        ok_beta = np.array_equal(lhs_beta, rhs_beta)
        print(f"  - Beta Lifting Constraint:            {'PASS' if ok_beta else 'FAIL'}")

        # Eq 6: Output norm — M_norm * (inner + Q_beta_lifted + M_prime_norm) = (Q_ln_out << k) + R_ln_out
        #   where inner = Q_diff * inv_std * (Q_gamma - z_gamma)
        Q_gamma = sim[f'Q_gamma_{suffix}'].astype(object)
        z_gamma = int(sim[f'z_gamma_{suffix}'])
        inv_std = sim[f'inv_std_total_{suffix}'].astype(object)
        M_norm = int(sim[f'M_norm_{suffix}'])
        M_prime_norm = int(sim[f'M_prime_norm_{suffix}'])

        inner = Q_diff * inv_std * (Q_gamma - z_gamma)
        lhs_norm = M_norm * (inner + Q_beta_lifted + M_prime_norm)
        Q_ln_out = sim[f'Q_ln_output_sim_{suffix}'].astype(object)
        R_ln_out = sim[f'R_ln_output_{suffix}'].astype(object)
        rhs_norm = (Q_ln_out << k_bits) + R_ln_out
        ok_norm = np.array_equal(lhs_norm, rhs_norm)
        print(f"  - Norm Output Constraint:             {'PASS' if ok_norm else 'FAIL'}")

    # [2] Linear Layers 검증 (lin1: FFN up, lin2: FFN down)
    for suffix in ['lin1', 'lin2']:
        if f'Q_out_sim_{suffix}' not in sim: continue
        print(f"\n[Step: {suffix}] Linear (FFN) Verification:")

        M_lin = sim[f'M_{suffix}'].astype(object)
        M_prime_lin = sim[f'M_prime_{suffix}'].astype(object)
        Q_in_diff = sim[f'Q_in_diff_{suffix}'].astype(np.int64)
        Q_w_diff = sim[f'Q_w_diff_{suffix}'].astype(np.int64)
        Q_b_diff = sim[f'Q_b_diff_{suffix}'].astype(np.int64)

        # acc = (Q_in - z_in) @ (Q_w - z_w)^T + (Q_b - z_b)
        acc = np.matmul(Q_in_diff.astype(object), Q_w_diff.astype(object).T) + Q_b_diff.astype(object)

        lhs_lin = M_lin * (acc + M_prime_lin)
        rhs_lin = (sim[f'Q_out_sim_{suffix}'].astype(object) << k_bits) + sim[f'R_out_{suffix}'].astype(object)

        ok_lin = np.array_equal(lhs_lin, rhs_lin)
        print(f"  - Matmul & Bias Constraint:           {'PASS' if ok_lin else 'FAIL'}")

    # [3] GeLU Activation 검증
    if 'Q_out_sim_gelu' in sim:
        print("\n[Step: GeLU] Activation Verification:")
        M_gelu = sim['M_gelu'].astype(object)
        M_prime_gelu = sim['M_prime_gelu'].astype(object)

        z_in_gelu = sim['z_in_gelu'].astype(object)
        in_gelu_raw = sim['Q_out_sim_lin1'].astype(object)

        in_gelu_adjusted = in_gelu_raw - z_in_gelu
        sig_part = sim['sigmoid_part_gelu'].astype(object)

        lhs_gelu = M_gelu * (in_gelu_adjusted * sig_part + M_prime_gelu)
        rhs_gelu = (sim['Q_out_sim_gelu'].astype(object) << k_bits) + sim['R_out_gelu'].astype(object)

        ok_gelu = np.array_equal(lhs_gelu, rhs_gelu)
        print(f"  - GeLU Lookup & Rescale:              {'PASS' if ok_gelu else 'FAIL'}")

if __name__ == "__main__":
    witness_file = "../outputs_bert/outputs_bert-batch0-bert.encoder.layer.1.attention.self.norm.witness.npz"
    if os.path.exists(witness_file):
        debug_bert_normalization(witness_file)
    else:
        print(f"File not found: {witness_file}")
