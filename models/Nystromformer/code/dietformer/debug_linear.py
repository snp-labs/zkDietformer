import numpy as np
import os
import glob

def debug_linear_simulation(output_dir="outputs"):
    # 1. Witness 파일 찾기 (*.norm*.witness.npz - Linear 데이터가 포함된 파일)
    witness_files = glob.glob(os.path.join(output_dir, "*.norm*.witness.npz"))
    if not witness_files:
        print(f"❌ '{output_dir}' 폴더에서 witness.npz 파일을 찾을 수 없습니다.")
        return

    for witness_path in witness_files:
        debug_path = witness_path.replace(".witness.npz", ".npz")
        if not os.path.exists(debug_path):
            continue

        print(f"\n" + "="*75)
        print(f"🔍 Linear Witness Debug: {os.path.basename(debug_path)}")
        print("="*75)

        # 데이터 로드
        ref = np.load(debug_path, allow_pickle=True)
        sim = np.load(witness_path, allow_pickle=True)

        # 필수 키 존재 확인 (Linear 데이터가 없는 파일은 스킵)
        if 'Q_out_sim_lin' not in sim:
            print("⏭️ No Linear data found in this witness file. Skipping...")
            continue

        k = int(sim['k_bits'])

        # --- [검증 1] 선형 연산 정수 등식 (enforce_equal 대상) ---
        # 수식: numerator_lin = M_lin * (acc_lin + M_prime_lin)
        # acc_lin은 이미 MatMul(G) + Bias(Q_b)가 완료된 상태여야 함
        
        # 큰 정수 연산을 위해 object 타입으로 변환 (오버플로 방지)
        M_lin = sim['M_lin'].astype(object)
        acc_lin = sim['acc_lin'].astype(object)
        M_prime_lin = sim['M_prime_lin'].astype(object)
        numerator_lin_sim = sim['numerator_lin'].astype(object)
        
        # 파이썬에서 직접 다시 계산해봄
        numerator_recalc = M_lin * (acc_lin + M_prime_lin)
        
        # 1) 분자 구성 확인
        check_numerator = np.array_equal(numerator_recalc, numerator_lin_sim)
        
        # 2) 시프트 및 나머지 등식 확인: numerator = (Q_out << k) + R_out
        shifted_val = (sim['Q_out_sim_lin'].astype(object) << k) + sim['R_out_lin'].astype(object)
        check_shift = np.array_equal(numerator_lin_sim, shifted_val)

        print(f"1. Integer Equality (Circuit Constraints):")
        print(f"   - Eq 1 (Numerator Construction): {'✅ PASS' if check_numerator else '❌ FAIL'}")
        print(f"   - Eq 2 (Shift & Remainder):     {'✅ PASS' if check_shift else '❌ FAIL'}")

        # --- [검증 2] 실제 부동소수점 결과와의 통계적 일치성 ---
        # linear_output (B, S, Out_Dim) vs Q_out_sim_lin
        out_ref = ref['linear_output'].flatten()
        out_sim = sim['Q_out_sim_lin'].flatten()
        
        # 상관관계 측정
        final_corr = np.corrcoef(out_ref, out_sim)[0, 1]
        
        # 오차 분석
        # 정수 시뮬레이션을 다시 부동소수점으로 복원하여 MSE 측정 (선택 사항)
        # s_out, z_out = get_quantization_params(ref['linear_output']) 
        # out_sim_float = s_out * (out_sim - z_out)
        
        print(f"2. Statistical Accuracy:")
        print(f"   - Correlation (Float vs Int Sim): {final_corr:.8f}")

        # --- [검증 3] 데이터 범위 확인 (Range Check) ---
        print(f"3. Value Ranges (Witness Health):")
        print(f"   - Q_out_sim_lin: {np.min(sim['Q_out_sim_lin'])} ~ {np.max(sim['Q_out_sim_lin'])}")
        print(f"   - R_out_lin (Must be < 2^{k}): {'✅ OK' if np.all(sim['R_out_lin'] < (1<<k)) else '❌ OVER'}")

        # 종합 판정
        if check_numerator and check_shift and final_corr > 0.99:
            print("\n✨ LINEAR DEBUG RESULT: [SUCCESS]")
        else:
            print("\n🚨 LINEAR DEBUG RESULT: [FAILED] Check scaling or bias addition.")

if __name__ == "__main__":
    debug_linear_simulation("outputs")