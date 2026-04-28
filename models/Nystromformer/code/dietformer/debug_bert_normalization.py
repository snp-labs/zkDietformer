import numpy as np
import os
import glob

def debug_bert_normalization(witness_path, k_bits=22, frac_bits=16):
    print(f"=== 🔍 BERT Normalization Circuit Debugging: {os.path.basename(witness_path)} ===")
    sim = np.load(witness_path, allow_pickle=True)
    
    # [1] RMSNorm 검증 (rms1, rms2)
    for suffix in ['rms1', 'rms2']:
        if f'Q_rms_output_sim_{suffix}' not in sim: continue
        print(f"\n[Step: {suffix}] RMSNorm Verification:")
        
        # Eq 1: MS(Mean Square) 계산 검증
        lhs1 = sim[f'M_ms_{suffix}'].astype(object) * (sim[f'sum_sq_{suffix}'].astype(object) + sim[f'M_prime_ms_{suffix}'].astype(object))
        rhs1 = (sim[f'shifted_val_{suffix}'].astype(object) << k_bits) + sim[f'R_shift_{suffix}'].astype(object)
        ok1 = np.array_equal(lhs1, rhs1)
        
        # Eq 2: Normalization & Gamma 곱셈 검증
        Q_diff = (sim[f'Q_ri_{suffix}'] - sim[f'z_ri_{suffix}']).astype(object)
        Q_gamma_diff = (sim[f'Q_gamma_{suffix}'].astype(object) - sim[f'z_gamma_{suffix}'].astype(object))
        inv_rms = sim[f'inv_rms_total_{suffix}'].astype(object)
        
        inner_product = Q_diff * inv_rms * Q_gamma_diff
        lhs2 = sim[f'M_norm_{suffix}'].astype(object) * (inner_product + sim[f'M_prime_norm_{suffix}'].astype(object))
        rhs2 = (sim[f'Q_rms_output_sim_{suffix}'].astype(object) << k_bits) + sim[f'R_rms_output_{suffix}'].astype(object)
        ok2 = np.array_equal(lhs2, rhs2)
        
        print(f"  - Mean Square Constraint:  {'✅ PASS' if ok1 else '❌ FAIL'}")
        print(f"  - Norm & Gamma Constraint: {'✅ PASS' if ok2 else '❌ FAIL'}")

    # [2] Linear Layers 검증 (lin1, lin2)
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
        print(f"  - Matmul & Bias Constraint: {'✅ PASS' if ok_lin else '❌ FAIL'}")

    # [3] GeLU Activation 검증
    if 'Q_out_sim_gelu' in sim:
        print("\n[Step: GeLU] Activation Verification:")
        M_gelu = sim['M_gelu'].astype(object)
        M_prime_gelu = sim['M_prime_gelu'].astype(object)
        
        z_in_gelu = sim['z_in_gelu'].astype(object) # lin1의 zero_point
        in_gelu_raw = sim['Q_out_sim_lin1'].astype(object)
        
        # 반드시 Zero-point를 빼서 '실제 값'을 복원한 뒤 계산해야 함
        in_gelu_adjusted = in_gelu_raw - z_in_gelu 
        sig_part = sim['sigmoid_part_gelu'].astype(object)
        
        # 서킷 제약 조건 식
        lhs_gelu = M_gelu * (in_gelu_adjusted * sig_part + M_prime_gelu)
        rhs_gelu = (sim['Q_out_sim_gelu'].astype(object) << k_bits) + sim['R_out_gelu'].astype(object)
        
        ok_gelu = np.array_equal(lhs_gelu, rhs_gelu)
        print(f"  - GeLU Lookup & Rescale:   {'✅ PASS' if ok_gelu else '❌ FAIL'}")

if __name__ == "__main__":
    witness_file = "outputs_bert/outputs_bert-batch0-bert.encoder.layer.1.attention.self.norm.witness.npz"
    if os.path.exists(witness_file):
        debug_bert_normalization(witness_file)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {witness_file}")