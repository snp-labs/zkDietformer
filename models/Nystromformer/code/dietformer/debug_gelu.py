import numpy as np
import os
import glob

# gelu_table.py의 설정과 동기화
TABLE_SIZE = 1 << 16  # 65536
FRAC_BITS = 16

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

        ref = np.load(debug_path, allow_pickle=True)
        sim = np.load(witness_path, allow_pickle=True)

        if 'Q_out_sim_gelu' not in sim:
            print("⏭️ GeLU 데이터가 없습니다. 스킵합니다.")
            continue

        k = int(sim['k_bits'])
        total_shift = k + FRAC_BITS 

        # --- [검증 1] 테이블 인덱스 및 시그모이드 조회 검증 ---
        # Eq 1: Index = (( (Q_in - z_in) * 2^k + R_in ) * M_gelu_idx >> k) + Offset
        q_in_diff = (sim['Q_out_sim_lin'].astype(object) - sim['z_in_gelu'].astype(object))
        x_diff_full = (q_in_diff << k) + sim['R_out_lin'].astype(object)
        
        scaled_diff = (x_diff_full * sim['M_gelu_idx'].astype(object)) >> k
        calculated_idx = (scaled_diff + (TABLE_SIZE // 2)).astype(np.int64)
        clipped_idx = np.clip(calculated_idx, 0, TABLE_SIZE - 1)
        
        # 실제 테이블 값과 위트니스에 저장된 sigmoid_part_gelu 비교
        table_val_check = np.array_equal(sim['gelu_table'][clipped_idx.astype(int)], sim['sigmoid_part_gelu'])

        # --- [검증 2] GeLU 알맹이(acc) 등식 검증 ---
        # Eq 2: acc_gelu = (Q_in - z_in) * sigmoid_part_gelu
        acc_recalc = q_in_diff * sim['sigmoid_part_gelu'].astype(object)
        check_acc = np.array_equal(acc_recalc, sim['acc_gelu'].astype(object))

        # --- [검증 3] 최종 통합 재스케일링 및 시프트 검증 ---
        # Eq 3: M_gelu * (acc_gelu + M_prime_gelu) == (Q_out << total_shift) + R_out
        M_gelu = sim['M_gelu'].astype(object)
        M_prime_gelu = sim['M_prime_gelu'].astype(object)
        acc_gelu = sim['acc_gelu'].astype(object)
        
        check_shift = k 

        lhs_rescale = M_gelu * (acc_gelu.astype(object) + M_prime_gelu)
        rhs_rescale = (sim['Q_out_sim_gelu'].astype(object) << check_shift) + sim['R_out_gelu'].astype(object)

        check_equality = np.array_equal(lhs_rescale, rhs_rescale)

        print(f"1. Integer Equality (Circuit Constraints):")
        print(f"   - Eq 1 (Table Lookup Index):   {'✅ PASS' if table_val_check else '❌ FAIL'}")
        print(f"   - Eq 2 (Acc Construction):     {'✅ PASS' if check_acc else '❌ FAIL'}")
        print(f"   - Eq 3 (Fused Rescale & Rem):  {'✅ PASS' if check_equality else '❌ FAIL'}")

        # --- [검증 4] 실제 부동소수점 대비 정확도 ---
        out_ref = ref['gelu_output'].flatten()
        out_sim = sim['Q_out_sim_gelu'].flatten()
        correlation = np.corrcoef(out_ref, out_sim)[0, 1]

        print(f"2. Statistical Accuracy:")
        print(f"   - Correlation (Float vs Sim): {correlation:.8f}")

        if table_val_check and check_acc and check_equality and correlation > 0.97:
            print(f"\n✨ GELU DEBUG RESULT: [SUCCESS] Verified with total_shift={total_shift}")
        else:
            print("\n🚨 GELU DEBUG RESULT: [FAILED] Check multiplier logic or total_shift.")

if __name__ == "__main__":
    debug_gelu_constraints("outputs")