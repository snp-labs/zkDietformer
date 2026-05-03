import numpy as np
import os
import glob

def debug_linear_simulation(output_dir="outputs"):
    witness_files = glob.glob(os.path.join(output_dir, "*.norm.witness.npz"))
    if not witness_files:
        print(f"❌ '{output_dir}' 폴더에서 witness.npz 파일을 찾을 수 없습니다.")
        return

    for witness_path in witness_files:
        debug_path = witness_path.replace(".norm.witness.npz", ".norm_debug.npz")
        if not os.path.exists(debug_path): continue

        print(f"\n" + "="*75)
        print(f"🔍 Linear Witness Debug: {os.path.basename(debug_path)}")
        print("="*75)

        ref = np.load(debug_path, allow_pickle=True)
        sim = np.load(witness_path, allow_pickle=True)
        k = int(sim['k_bits'])

        for suffix in ['lin1', 'lin2']:
            if f'Q_out_sim_{suffix}' not in sim: continue

            print(f"\n--- Testing Layer: {suffix} ---")
            
            M_lin = sim[f'M_{suffix}'].astype(object)
            M_prime_lin = sim[f'M_prime_{suffix}'].astype(object)

            Q_in_diff = sim[f'Q_in_diff_{suffix}'].astype(np.int64)
            Q_w_diff = sim[f'Q_w_diff_{suffix}'].astype(np.int64)
            Q_b_diff = sim[f'Q_b_diff_{suffix}'].astype(np.int64)
            
            # acc = (Q_in - z_in) * (Q_w - z_w)^T + (Q_b - z_b)
            acc_recalc = np.matmul(Q_in_diff, Q_w_diff.T) + Q_b_diff
            
            # numerator = M * (acc + M')
            numerator_recalc = M_lin * (acc_recalc.astype(object) + M_prime_lin)
            
            # numerator = (Q_out << k) + R_out
            Q_out = sim[f'Q_out_sim_{suffix}'].astype(object)
            R_out = sim[f'R_out_{suffix}'].astype(object)
            shifted_val = (Q_out << k) + R_out
            
            check_eq = np.array_equal(numerator_recalc, shifted_val)

            if suffix == 'lin1':
                out_ref = ref['linear_output']
            else:
                out_ref = ref['linear_output_ff2'] 

            final_corr = np.corrcoef(out_ref.flatten(), Q_out.flatten().astype(float))[0, 1]

            print(f"   - Integer Equality ({suffix}): {'✅ PASS' if check_eq else '❌ FAIL'}")
            print(f"   - Correlation (Ref vs Sim): {final_corr:.8f}")

if __name__ == "__main__":
    debug_linear_simulation("../outputs_bert")