import numpy as np
import os
import glob

def debug_rmsnorm_simulation(output_dir="outputs"):
    # 1. Witness 파일 찾기 (*.norm*.witness.npz)
    witness_files = glob.glob(os.path.join(output_dir, "*.norm*.witness.npz"))
    if not witness_files:
        print(f"❌ '{output_dir}' 폴더에서 .witness.npz 파일을 찾을 수 없습니다.")
        return

    for witness_path in witness_files:
        debug_path = witness_path.replace(".witness.npz", ".npz")
        if not os.path.exists(debug_path):
            continue

        print(f"\n" + "="*70)
        print(f"🔍 Circuit Witness Debug: {os.path.basename(debug_path)}")
        print("="*70)

        # 데이터 로드
        ref = np.load(debug_path)      # 원본 부동소수점 데이터
        sim = np.load(witness_path)    # 정수 시뮬레이션 데이터 (Witness)

        k_bits = int(sim['k_bits'])
        frac_bits = int(sim['frac_bits'])
        w_bits = int(sim.get('weight_bits', 8)) 
        total_shift = frac_bits + w_bits

        # --- [검증 1] MS 계산 및 시프트/나머지 등식 (enforce_equal 대상) ---
        # 1) numerator = (shifted_val << k_bits) + R_shift
        numerator = sim['M_ms_rms'] * (sim['sum_sq_rms'] + sim['M_prime_ms_rms'])
        eq1_val = (sim['shifted_val_rms'].astype(object) << k_bits) + sim['R_shift_rms'].astype(object)
        check1 = np.array_equal(numerator, eq1_val)

        # 2) shifted_val = (Q_ms_sim * dim) + R_ms
        dim = ref['rms_input'].shape[-1]
        eq2_val = (sim['Q_ms_sim_rms'].astype(object) * dim) + sim['R_ms_rms'].astype(object)
        check2 = np.array_equal(sim['shifted_val_rms'], eq2_val)

        print(f"1. Integer Equality (Circuit Constraints) Check:")
        print(f"   - Eq 1 (Shift/Remainder): {'✅ PASS' if check1 else '❌ FAIL'}")
        print(f"   - Eq 2 (Division/Remainder): {'✅ PASS' if check2 else '❌ FAIL'}")

        # --- [검증 2] 1/RMS 테이블 조회 정밀도 ---
        inv_rms_ref = (1.0 / np.sqrt(ref['ms'] + ref.get('eps', 1e-6))).flatten()
        inv_rms_sim_float = sim['inv_rms_total_rms'].flatten().astype(float) / (2**frac_bits)
        
        mse_inv = np.mean((inv_rms_ref - inv_rms_sim_float)**2)
        print(f"2. Table Precision Check:")
        print(f"   - Ref vs Sim MSE: {mse_inv:.2e}")

        # --- [검증 3] 최종 출력 등식 (Weight 포함) ---
        # 3) (Q_diff * inv_rms * Q_gamma) = (Q_out << (frac + weight_bits)) + R_out
        Q_diff = (sim['Q_ri_rms'] - sim['z_ri_rms']).astype(object)
        inv_rms = sim['inv_rms_total_rms'].astype(object)
        Q_gamma = sim['Q_gamma_rms'].astype(object)

        # 파이프라인의 결합 연산 재현
        # (B, S, D) * (B, S, 1) * (D,) -> 브로드캐스팅 적용
        full_product = Q_diff * inv_rms * Q_gamma
        
        # Witness에 저장된 결과로 등식 성립 확인
        lhs = (sim['Q_rms_output_sim_rms'].astype(object) << total_shift) + sim['R_rms_output_rms'].astype(object)
        
        check3 = np.array_equal(full_product, lhs)

        # 상관관계 분석
        out_ref = ref['rms_output'].flatten()
        out_sim = sim['Q_rms_output_sim_rms'].flatten()
        final_corr = np.corrcoef(out_ref, out_sim)[0, 1]

        print(f"3. Weight & Output Consistency:")
        print(f"   - Total Shift Bits: {total_shift} ({frac_bits} + {w_bits})")
        print(f"   - Eq 3 (Fused Multiplication): {'✅ PASS' if check3 else '❌ FAIL'}")
        print(f"   - Correlation (Float vs Int): {final_corr:.8f}")

        if check3 and final_corr > 0.98:
            print("✨ FINAL RESULT: [SUCCESS] Weight inclusion verified.")
        else:
            print("🚨 FINAL RESULT: [RECHECK] Output inconsistency detected.")

if __name__ == "__main__":
    debug_rmsnorm_simulation("outputs")