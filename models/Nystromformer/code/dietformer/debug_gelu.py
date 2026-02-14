import numpy as np
import os
import glob

def debug_gelu_constraints(output_dir="outputs"):
    witness_files = glob.glob(os.path.join(output_dir, "*.norm*.witness.npz"))
    if not witness_files:
        print(f"❌ '{output_dir}' 폴더에서 witness 파일을 찾을 수 없습니다.")
        return

    for witness_path in witness_files:
        debug_path = witness_path.replace(".witness.npz", ".npz")
        if not os.path.exists(debug_path): continue

        print(f"\n" + "="*70)
        print(f"🔍 GeLU Constraint Debug: {os.path.basename(witness_path)}")
        print("="*70)

        # 데이터 로드 (allow_pickle=True 필수)
        ref = np.load(debug_path, allow_pickle=True)
        sim = np.load(witness_path, allow_pickle=True)

        if 'Q_gelu_out_sim' not in sim:
            print("⏭️ GeLU 데이터가 없습니다. 스킵합니다.")
            continue

        k = int(sim['k_bits'])
        frac_bits = int(sim['frac_bits'])

        # --- [검증 1] GeLU 알맹이(acc) 등식 검증 ---
        # acc_gelu = (Q_in - z_in) * sigmoid_part
        # Q_in은 Linear1의 출력(Q_out_sim_lin)
        q_in_diff = (sim['Q_out_sim_lin'].astype(object) - sim['z_in_gelu'].astype(object))
        acc_recalc = q_in_diff * sim['sigmoid_part_gelu'].astype(object)
        
        check_acc = np.array_equal(acc_recalc, sim['acc_gelu'].astype(object))

        # --- [검증 2] Rescaling & Offset 등식 검증 ---
        # --- [검증 2] Rescaling & Offset 등식 검증 ---
        # numerator_gelu = M_gelu * (acc_gelu + M_prime_gelu)
        M_gelu = sim['M_gelu'].astype(object)
        M_prime_gelu = sim['M_prime_gelu'].astype(object)
        acc_gelu = sim['acc_gelu'].astype(object)
        
        # [수정] 괄호 안의 합을 먼저 object로 계산하여 오버플로 방지
        inner_sum = acc_gelu + M_prime_gelu
        numerator_recalc = M_gelu * inner_sum

        # --- [검증 3] 시프트 및 나머지 등식 ---
        # [수정] Q_out_sim_gelu 뿐만 아니라 R_out_gelu도 object로 변환하여 더함
        q_out_part = (sim['Q_out_sim_gelu'].astype(object) << k)
        r_out_part = sim['R_out_gelu'].astype(object)
        shifted_val = q_out_part + r_out_part
        
        check_equality = np.array_equal(numerator_recalc, shifted_val)

        print(f"1. Integer Equality (Circuit Constraints):")
        print(f"   - Eq 1 (Acc Construction):     {'✅ PASS' if check_acc else '❌ FAIL'}")
        print(f"   - Eq 2 (Rescale & Remainder):  {'✅ PASS' if check_equality else '❌ FAIL'}")

        # --- [검증 4] 실제 부동소수점 대비 정확도 ---
        out_ref = ref['gelu_output'].flatten()
        out_sim = sim['Q_out_sim_gelu'].flatten()
        correlation = np.corrcoef(out_ref, out_sim)[0, 1]

        print(f"2. Statistical Accuracy:")
        print(f"   - Correlation (Float vs Sim): {correlation:.8f}")

        if check_acc and check_equality and correlation > 0.97:
            print("\n✨ GELU DEBUG RESULT: [SUCCESS] Circuit implementation is safe.")
        else:
            print("\n🚨 GELU DEBUG RESULT: [FAILED] Check scale factors or table range.")

if __name__ == "__main__":
    debug_gelu_constraints("outputs")