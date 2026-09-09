"""
A-Gate re-verification: 0-1 (ZEN equation), 0-2 (softmax consistency),
0-3 (reverse injectivity).

Run from the dietformer directory after run_linformer_validation.py has generated
witness NPZ files.
"""
import os
import sys
import numpy as np

from quantize import get_quantization_params, quantize_tensor, get_m_constant
from softmax_table import FRAC_BITS, TABLE_FRAC_BITS, generate_fractional_exp_table

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def verify_zen_equation(witness_path, debug_path):
    """0-1: Verify Q_sim * 2^k + R = M * G_total for 4 matmul types."""
    w = np.load(witness_path, allow_pickle=True)
    d = np.load(debug_path, allow_pickle=True)

    k_bits = int(w['k_bits'])
    B, H, S, D = w['Q_q'].shape
    k_proj = w['Q_sim_ek'].shape[2]

    results = {}

    # ---- EK: E_i[h] @ K_proc[b,h] ----
    Q_e_raw = quantize_tensor(d['E'], *get_quantization_params(d['E']))
    s_e, z_e = get_quantization_params(d['E'])
    Q_k = w['Q_k']  # [B, H, S, D] - scaled K
    s_k, z_k = get_quantization_params(
        d['K_proc'] if 'K_proc' in d else d['Q_proc']  # K_proc for k
    )
    # Reconstruct Q_k from witness (use the stored scaled value)
    z_k_val = int(w['z_k'])
    z_e_val = int(w['z_w_e'])
    M_ek = int(w['M_ek'])
    M_prime_ek = int(w['M_prime_ek'])
    z_ek = int(w['z_sim_ek'])

    total_ek = 0
    mismatch_ek = 0
    max_diff_ek = 0

    for b in range(B):
        for h in range(H):
            Qw = Q_e_raw[h].astype(np.int64)  # [k_proj, S]
            Qx = Q_k[b, h].astype(np.int64)    # [S, D]
            n = Qw.shape[-1]  # S

            G1 = np.matmul(Qw, Qx)
            G2 = z_k_val * np.sum(Qw, axis=-1, keepdims=True)
            G3 = z_e_val * np.sum(Qx, axis=-2, keepdims=True)
            G_total = G1 + n * z_e_val * z_k_val + M_prime_ek - G2 - G3

            lhs = w['Q_sim_ek'][b, h].astype(object) * (1 << k_bits) + w['R_matmul_ek'][b, h].astype(object)
            rhs = M_ek * G_total.astype(object)

            diff = np.abs(lhs - rhs)
            n_mismatch = int(np.sum(diff != 0))
            total_ek += diff.size
            mismatch_ek += n_mismatch
            if diff.size > 0:
                max_diff_ek = max(max_diff_ek, int(np.max(diff)))

    results['EK'] = (total_ek, mismatch_ek, max_diff_ek)

    # ---- FV: F_i[h] @ V[b,h] ----
    Q_f_raw = quantize_tensor(d['F_proj'], *get_quantization_params(d['F_proj']))
    s_f, z_f = get_quantization_params(d['F_proj'])
    # V is NOT scaled (no apply_scale for v)
    Q_v = w['Q_v']  # [B, H, S, D]
    z_f_val = int(w['z_w_f'])
    z_v_val = int(w['z_v'])
    M_fv = int(w['M_fv'])
    M_prime_fv = int(w['M_prime_fv'])
    z_fv = int(w['z_sim_fv'])

    total_fv = 0
    mismatch_fv = 0
    max_diff_fv = 0

    for b in range(B):
        for h in range(H):
            Qw = Q_f_raw[h].astype(np.int64)  # [k_proj, S]
            Qx = Q_v[b, h].astype(np.int64)    # [S, D]
            n = Qw.shape[-1]

            G1 = np.matmul(Qw, Qx)
            G2 = z_v_val * np.sum(Qw, axis=-1, keepdims=True)
            G3 = z_f_val * np.sum(Qx, axis=-2, keepdims=True)
            G_total = G1 + n * z_f_val * z_v_val + M_prime_fv - G2 - G3

            lhs = w['Q_sim_fv'][b, h].astype(object) * (1 << k_bits) + w['R_matmul_fv'][b, h].astype(object)
            rhs = M_fv * G_total.astype(object)

            diff = np.abs(lhs - rhs)
            n_mismatch = int(np.sum(diff != 0))
            total_fv += diff.size
            mismatch_fv += n_mismatch
            if diff.size > 0:
                max_diff_fv = max(max_diff_fv, int(np.max(diff)))

    results['FV'] = (total_fv, mismatch_fv, max_diff_fv)

    # ---- Logit: Q_q[b,h] @ EK[b,h]^T ----
    Q_q = w['Q_q']  # [B, H, S, D]
    z_q_val = int(w['z_q'])
    z_ek_val = int(w['z_sim_ek'])
    M_logit = int(w['M_logit'])
    M_prime_logit = int(w['M_prime_logit'])
    z_logit = int(w['z_sim_logit'])

    total_logit = 0
    mismatch_logit = 0
    max_diff_logit = 0

    for b in range(B):
        for h in range(H):
            Qw = Q_q[b, h].astype(np.int64)  # [S, D]
            Qx = w['Q_sim_ek'][b, h].T.astype(np.int64)  # [D, k_proj]
            n = Qw.shape[-1]  # D

            G1 = np.matmul(Qw, Qx)
            G2 = z_ek_val * np.sum(Qw, axis=-1, keepdims=True)
            G3 = z_q_val * np.sum(Qx, axis=-2, keepdims=True)
            G_total = G1 + n * z_q_val * z_ek_val + M_prime_logit - G2 - G3

            lhs = w['Q_sim_logit'][b, h].astype(object) * (1 << k_bits) + w['R_matmul_logit'][b, h].astype(object)
            rhs = M_logit * G_total.astype(object)

            diff = np.abs(lhs - rhs)
            n_mismatch = int(np.sum(diff != 0))
            total_logit += diff.size
            mismatch_logit += n_mismatch
            if diff.size > 0:
                max_diff_logit = max(max_diff_logit, int(np.max(diff)))

    results['logit'] = (total_logit, mismatch_logit, max_diff_logit)

    # ---- x: P[b,h] @ FV[b,h] ----
    P_lin = w['P_lin']
    z_p = 0
    z_fv_val = int(w['z_sim_fv'])
    M_x = int(w['M_x'])
    M_prime_x = int(w['M_prime_x'])
    z_x = int(w['z_sim_x'])

    total_x = 0
    mismatch_x = 0
    max_diff_x = 0

    for b in range(B):
        for h in range(H):
            Qw = P_lin[b, h].astype(np.int64)  # [S, k_proj]
            Qx = w['Q_sim_fv'][b, h].astype(np.int64)  # [k_proj, D]
            n = Qw.shape[-1]  # k_proj

            G1 = np.matmul(Qw, Qx)
            G2 = z_fv_val * np.sum(Qw, axis=-1, keepdims=True)
            G3 = z_p * np.sum(Qx, axis=-2, keepdims=True)
            G_total = G1 + n * z_p * z_fv_val + M_prime_x - G2 - G3

            lhs = w['Q_sim_x'][b, h].astype(object) * (1 << k_bits) + w['R_matmul_x'][b, h].astype(object)
            rhs = M_x * G_total.astype(object)

            diff = np.abs(lhs - rhs)
            n_mismatch = int(np.sum(diff != 0))
            total_x += diff.size
            mismatch_x += n_mismatch
            if diff.size > 0:
                max_diff_x = max(max_diff_x, int(np.max(diff)))

    results['x'] = (total_x, mismatch_x, max_diff_x)

    return results


def verify_softmax_consistency(witness_path):
    """0-2: Verify softmax equations exactly."""
    w = np.load(witness_path, allow_pickle=True)
    frac_bits = int(w['frac_bits'])
    table_frac_bits = int(w['table_frac_bits'])

    exp_table = w['exp_table']
    table_idx = w['table_idx']
    exp_total = w['exp_total']
    P_lin = w['P_lin']
    R_div = w['R_div_lin']
    exp_sum = w['exp_sum']

    results = {}

    # Equation 1: exp_total == exp_table[table_idx]
    recon_exp = exp_table[table_idx.astype(int)]
    diff1 = np.abs(recon_exp.astype(object) - exp_total.astype(object))
    n_total1 = diff1.size
    n_mismatch1 = int(np.sum(diff1 != 0))
    results['exp_table_lookup'] = (n_total1, n_mismatch1)

    # Equation 2: P * exp_sum + R_div == exp_total << frac_bits
    lhs2 = P_lin.astype(object) * exp_sum.astype(object) + R_div.astype(object)
    rhs2 = exp_total.astype(object) << frac_bits
    diff2 = np.abs(lhs2 - rhs2)
    n_total2 = diff2.size
    n_mismatch2 = int(np.sum(diff2 != 0))
    results['div_equation'] = (n_total2, n_mismatch2)

    return results


def verify_reverse_injectivity(witness_path):
    """0-3: Recover Q_sim_logit from (table_idx, r_sm, R_matmul_logit) only.

    From softmax_v3_lookup:
        scaled_product = ((Q_sim << k) + R_mat) * M_sm
        table_idx = scaled_product >> (2*k)
        r_sm = scaled_product % (1 << (2*k))

    Recovery (no Q_sim_logit reference):
        scaled_product = table_idx * (1 << (2*k)) + r_sm
        X = scaled_product // M_sm
        Q_sim_recovered = (X - R_matmul_logit) >> k_bits
        (since X = Q_sim * 2^k + R_mat, so Q_sim = (X - R_mat) >> k)
    """
    w = np.load(witness_path, allow_pickle=True)
    k_bits = int(w['k_bits'])
    M_sm = int(w['M_sm'])

    table_idx = w['table_idx'].astype(object)
    r_sm = w['r_sm'].astype(object)
    R_matmul_logit = w['R_matmul_logit'].astype(object)
    Q_sim_logit_ref = w['Q_sim_logit'].astype(np.int64)  # for comparison only

    # Step 1: Reconstruct scaled_product
    scaled_product = table_idx * (1 << (2 * k_bits)) + r_sm

    # Step 2: Recover X = scaled_product // M_sm
    X = scaled_product // M_sm

    # Step 3: Recover Q_sim = (X - R_mat) >> k_bits
    Q_sim_recovered = ((X - R_matmul_logit) >> k_bits).astype(np.int64)

    # Compare
    diff = np.abs(Q_sim_recovered - Q_sim_logit_ref)
    n_total = diff.size
    n_mismatch = int(np.sum(diff != 0))
    max_diff = int(np.max(diff)) if n_mismatch > 0 else 0

    # If mismatch, investigate which table_idx values have collisions
    collision_info = None
    if n_mismatch > 0:
        mismatch_mask = (diff != 0)
        mismatch_tidx = w['table_idx'][mismatch_mask]
        mismatch_qsim_ref = Q_sim_logit_ref[mismatch_mask]
        mismatch_qsim_rec = Q_sim_recovered[mismatch_mask]
        collision_info = {
            'n_mismatch': n_mismatch,
            'max_diff': max_diff,
            'sample_tidx': mismatch_tidx[:10].tolist(),
            'sample_ref': mismatch_qsim_ref[:10].tolist(),
            'sample_rec': mismatch_qsim_rec[:10].tolist(),
        }

    # Also verify the forward direction is exact (sanity)
    X_forward = (Q_sim_logit_ref.astype(object) << k_bits) + R_matmul_logit
    sp_forward = X_forward * M_sm
    tidx_forward = (sp_forward >> (2 * k_bits)).astype(np.int64)
    rsm_forward = sp_forward % (1 << (2 * k_bits))

    fwd_tidx_ok = int(np.sum(np.abs(tidx_forward - w['table_idx'].astype(np.int64)) != 0))
    fwd_rsm_ok = int(np.sum(np.abs((rsm_forward - w['r_sm'].astype(object)).astype(np.int64)) != 0))

    return {
        'total': n_total,
        'mismatch': n_mismatch,
        'max_diff': max_diff,
        'collision_info': collision_info,
        'forward_tidx_mismatch': fwd_tidx_ok,
        'forward_rsm_mismatch': fwd_rsm_ok,
    }


def main():
    output_dir = os.path.join(SCRIPT_DIR, 'outputs_n128_k32_L2_h128_linformer')
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output dir not found: {output_dir}")
        sys.exit(1)

    witness_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith('.witness.npz') and 'norm' not in f
    ])

    all_passed = True

    for wf in witness_files:
        wp = os.path.join(output_dir, wf)
        # Derive debug path from witness path
        dp = wp.replace('.witness.npz', '.npz')

        layer_name = wf.split('-batch0-')[1].replace('.witness.npz', '')
        print(f'\n{"=" * 70}')
        print(f'Layer: {layer_name}')
        print(f'{"=" * 70}')

        # ---- 0-1: ZEN equation ----
        print('\n--- 0-1: ZEN equation verification ---')
        zen_results = verify_zen_equation(wp, dp)
        for matmul_name, (total, mismatch, max_diff) in zen_results.items():
            status = 'PASS' if mismatch == 0 else 'FAIL'
            if status == 'FAIL':
                all_passed = False
            print(f'  {matmul_name}: {status} | total={total}, mismatch={mismatch}, max_diff={max_diff}')

        # ---- 0-2: Softmax consistency ----
        print('\n--- 0-2: Softmax consistency ---')
        sm_results = verify_softmax_consistency(wp)
        for eq_name, (total, mismatch) in sm_results.items():
            status = 'PASS' if mismatch == 0 else 'FAIL'
            if status == 'FAIL':
                all_passed = False
            print(f'  {eq_name}: {status} | total={total}, mismatch={mismatch}')

        # ---- 0-3: Reverse injectivity ----
        print('\n--- 0-3: Reverse injectivity ---')
        inj_results = verify_reverse_injectivity(wp)
        status = 'PASS' if inj_results['mismatch'] == 0 else 'FAIL'
        if status == 'FAIL':
            all_passed = False
        print(f'  Recovery: {status} | total={inj_results["total"]}, '
              f'mismatch={inj_results["mismatch"]}, max_diff={inj_results["max_diff"]}')
        print(f'  Forward check: tidx_mismatch={inj_results["forward_tidx_mismatch"]}, '
              f'rsm_mismatch={inj_results["forward_rsm_mismatch"]}')

        if inj_results['collision_info']:
            ci = inj_results['collision_info']
            print(f'  COLLISION DETAILS:')
            print(f'    sample table_idx: {ci["sample_tidx"]}')
            print(f'    sample Q_sim_ref: {ci["sample_ref"]}')
            print(f'    sample Q_sim_rec: {ci["sample_rec"]}')

    print(f'\n{"=" * 70}')
    if all_passed:
        print('PART 0 (0-1, 0-2, 0-3): ALL PASSED')
    else:
        print('PART 0 (0-1, 0-2, 0-3): SOME CHECKS FAILED')
    print(f'{"=" * 70}')

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
