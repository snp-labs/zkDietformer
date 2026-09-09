"""
A-4: Per-stage fidelity analysis for Linformer attention.

Reports CosSim and MaxAbsErr at each stage:
  EK, FV, logit, P, x, O-proj
"""
import os
import sys
import numpy as np


def cos_sim(a, b):
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    dot = np.sum(a_f * b_f)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    return float(dot / (na * nb + 1e-30))


def max_abs_err(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def analyze_layer(witness_path, debug_path):
    w = np.load(witness_path, allow_pickle=True)
    d = np.load(debug_path, allow_pickle=True)

    results = []

    # EK: dequant Q_sim_ek vs float EK
    s_ek = float(d['EK'].max() - d['EK'].min()) / 255.0 if True else 1.0
    # Actually use the quantization params from witness
    z_ek = int(w['z_sim_ek'])
    ek_real = d['EK'].astype(np.float64)
    # We need s_ek. Reconstruct from quantization:
    # Q_sim = round(real / s - z)? No, get_quantization_params does min-max.
    # Simpler: just compute from the debug data
    from quantize import get_quantization_params
    s_ek_val, z_ek_val = get_quantization_params(d['EK'])
    ek_deq = (w['Q_sim_ek'].astype(np.float64) - z_ek_val) * s_ek_val
    results.append(('EK', cos_sim(ek_real, ek_deq), max_abs_err(ek_real, ek_deq)))

    # FV
    s_fv_val, z_fv_val = get_quantization_params(d['FV'])
    fv_real = d['FV'].astype(np.float64)
    fv_deq = (w['Q_sim_fv'].astype(np.float64) - z_fv_val) * s_fv_val
    results.append(('FV', cos_sim(fv_real, fv_deq), max_abs_err(fv_real, fv_deq)))

    # logit
    s_lo, z_lo = get_quantization_params(d['logit'])
    logit_real = d['logit'].astype(np.float64)
    logit_deq = (w['Q_sim_logit'].astype(np.float64) - z_lo) * s_lo
    results.append(('logit', cos_sim(logit_real, logit_deq), max_abs_err(logit_real, logit_deq)))

    # P (softmax output) - compare with float softmax
    # P_lin is fixed-point: P_real = P_lin / 2^frac_bits
    frac_bits = int(w['frac_bits'])
    P_deq = w['P_lin'].astype(np.float64) / (1 << frac_bits)
    P_real = d['P'].astype(np.float64)
    results.append(('P', cos_sim(P_real, P_deq), max_abs_err(P_real, P_deq)))

    # x (P@FV output)
    s_x, z_x = get_quantization_params(d['X'])
    x_real = d['X'].astype(np.float64)
    x_deq = (w['Q_sim_x'].astype(np.float64) - z_x) * s_x
    results.append(('x', cos_sim(x_real, x_deq), max_abs_err(x_real, x_deq)))

    # O-proj
    s_o = float(w.get('s_raw_o', 1.0))
    z_o = int(w['z_raw_o'])
    o_deq = (w['Q_raw_proj_o'].astype(np.float64) - z_o) * s_o
    # float ref: recompute from debug
    W_o = d['W_o']
    attn_out = d['attn_output_for_proj']
    b_o = d.get('b_o', None)
    if b_o is not None and hasattr(b_o, 'shape'):
        pass
    else:
        b_o = None
    o_real = np.matmul(attn_out, W_o.T)
    if b_o is not None:
        o_real = o_real + b_o
    B, H = w['Q_q'].shape[:2]
    D = w['Q_q'].shape[3]
    o_real_4d = o_real.reshape(B, -1, H, D).transpose(0, 2, 1, 3)
    results.append(('O-proj', cos_sim(o_real_4d, o_deq), max_abs_err(o_real_4d, o_deq)))

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'outputs_n128_k32_L2_h128_linformer')
    if not os.path.isdir(output_dir):
        print(f"ERROR: {output_dir} not found")
        sys.exit(1)

    witness_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith('.witness.npz') and 'norm' not in f
    ])

    print(f"{'Stage':<10} {'CosSim':>12} {'MaxAbsErr':>14}")
    print('-' * 38)

    for wf in witness_files:
        wp = os.path.join(output_dir, wf)
        dp = wp.replace('.witness.npz', '.npz')
        layer_name = wf.split('-batch0-')[1].replace('.witness.npz', '')
        print(f'\n=== Layer: {layer_name} ===')
        results = analyze_layer(wp, dp)
        for stage, cs, mae in results:
            print(f"  {stage:<10} {cs:>12.8f} {mae:>14.6e}")


if __name__ == '__main__':
    main()
