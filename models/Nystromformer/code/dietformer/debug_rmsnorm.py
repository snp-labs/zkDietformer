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
        ref = np.load(debug_path)
        sim = np.load(witness_path)

        k_bits = int(sim['k_bits'])
        frac_bits = int(sim['frac_bits'])

        # --- [검증 1] Mean Square 계산 및 평균 나눗셈 ---
        # Eq 1: M_ms * (sum_sq + M_prime_ms) == (shifted_val << k) + R_shift
        lhs1 = sim['M_ms_rms'].astype(object) * (sim['sum_sq_rms'].astype(object) + sim['M_prime_ms_rms'].astype(object))
        rhs1 = (sim['shifted_val_rms'].astype(object) << k_bits) + sim['R_shift_rms'].astype(object)
        check1 = np.array_equal(lhs1, rhs1)

        # Eq 2: shifted_val == (Q_ms_sim * dim) + R_ms
        dim = ref['rms_input'].shape[-1]
        rhs2 = (sim['Q_ms_sim_rms'].astype(object) * dim) + sim['R_ms_rms'].astype(object)
        check2 = np.array_equal(sim['shifted_val_rms'], rhs2)

        # --- [검증 2] 고정 테이블 인덱스 추출 로직 ---
        # Index = ((shifted_val * M_rms_idx) >> k) // dim
        numerator_idx = sim['shifted_val_rms'].astype(object) * sim['M_rms'].astype(object)
        scaled_idx = numerator_idx >> k_bits
        calculated_idx = (scaled_idx // dim).astype(np.uint64)
        calculated_idx = np.clip(calculated_idx, 0, (1 << 16) - 1)
        
        # 실제 테이블 조회값과 계산된 인덱스의 테이블값이 일치하는지 확인
        table_val_check = np.array_equal(sim['rms_table'][calculated_idx.astype(int)], sim['inv_rms_total_rms'])

        # --- [검증 3] 통합 재스케일링 및 최종 출력 (Fused Equation) ---
        # Eq 3: M_norm * ( (Q_diff * inv_rms * Q_gamma_diff) + M_prime_norm ) == (Q_out << k) + R_out
        Q_diff = (sim['Q_ri_rms'] - sim['z_ri_rms']).astype(object)
        Q_gamma_diff = (sim['Q_gamma_rms'].astype(object) - sim['z_gamma_rms'].astype(object))
        inv_rms = sim['inv_rms_total_rms'].astype(object)
        
        inner_product = Q_diff * inv_rms * Q_gamma_diff
        
        lhs3 = sim['M_norm_rms'].astype(object) * (inner_product + sim['M_prime_norm_rms'].astype(object))
        rhs3 = (sim['Q_rms_output_sim_rms'].astype(object) << k_bits) + sim['R_rms_output_rms'].astype(object)
        check3 = np.array_equal(lhs3, rhs3)

        print(f"1. MS & Division Equality:   {'✅ PASS' if (check1 and check2) else '❌ FAIL'}")
        print(f"2. Fixed Table Index Match: {'✅ PASS' if table_val_check else '❌ FAIL'}")
        print(f"3. Fused Rescaling (Output): {'✅ PASS' if check3 else '❌ FAIL'}")

        # 정밀도 분석
        final_corr = np.corrcoef(ref['rms_output'].flatten(), sim['Q_rms_output_sim_rms'].flatten())[0, 1]
        print(f"   - Correlation (Ref vs Sim): {final_corr:.8f}")

if __name__ == "__main__":
    debug_rmsnorm_simulation()