import numpy as np
import os
import glob

def debug_rmsnorm_simulation(output_dir="outputs"):
    witness_files = glob.glob(os.path.join(output_dir, "*.norm*.witness.npz"))
    if not witness_files:
        print(f"❌ '{output_dir}' 폴더에서 witness 파일을 찾을 수 없습니다.")
        return

    for witness_path in witness_files:
        debug_path = witness_path.replace(".witness.npz", ".npz")
        if not os.path.exists(debug_path): continue

        print(f"\n🔍 Circuit Witness Debug: {os.path.basename(debug_path)}")
        ref = np.load(debug_path, allow_pickle=True)
        sim = np.load(witness_path, allow_pickle=True)
        k_bits = int(sim['k_bits'])

        for suffix in ['rms1', 'rms2']:
            if f'Q_rms_output_sim_{suffix}' not in sim: continue
            
            print(f"\n--- Testing Layer: {suffix} ---")
            
            lhs1 = sim[f'M_ms_{suffix}'].astype(object) * (sim[f'sum_sq_{suffix}'].astype(object) + sim[f'M_prime_ms_{suffix}'].astype(object))
            rhs1 = (sim[f'shifted_val_{suffix}'].astype(object) << k_bits) + sim[f'R_shift_{suffix}'].astype(object)
            check1 = np.array_equal(lhs1, rhs1)

            Q_diff = (sim[f'Q_ri_{suffix}'] - sim[f'z_ri_{suffix}']).astype(object)
            Q_gamma_diff = (sim[f'Q_gamma_{suffix}'].astype(object) - sim[f'z_gamma_{suffix}'].astype(object))
            inv_rms = sim[f'inv_rms_total_{suffix}'].astype(object)
            
            inner_product = Q_diff * inv_rms * Q_gamma_diff
            
            lhs3 = sim[f'M_norm_{suffix}'].astype(object) * (inner_product + sim[f'M_prime_norm_{suffix}'].astype(object))
            rhs3 = (sim[f'Q_rms_output_sim_{suffix}'].astype(object) << k_bits) + sim[f'R_rms_output_{suffix}'].astype(object)
            check3 = np.array_equal(lhs3, rhs3)

            if suffix == 'rms1':
                ref_out = ref['rms_output']
            else:
                ref_out = ref['rms_output_rms2'] 

            final_corr = np.corrcoef(ref_out.flatten(), sim[f'Q_rms_output_sim_{suffix}'].flatten().astype(float))[0, 1]

            print(f"   - MS Equality:       {'✅ PASS' if check1 else '❌ FAIL'}")
            print(f"   - Fused Rescaling:   {'✅ PASS' if check3 else '❌ FAIL'}")
            print(f"   - Correlation (Ref): {final_corr:.8f}")

if __name__ == "__main__":
    debug_rmsnorm_simulation()