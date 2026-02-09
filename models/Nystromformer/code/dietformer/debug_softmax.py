import numpy as np
import os

def debug_final_witness(npz_path):
    print(f"=== [ZEN Final Witness Debugging] Target: {npz_path} ===")
    if not os.path.exists(npz_path):
        print("❌ File not found.")
        return

    data = np.load(npz_path)
    k_bits = int(data['k_bits'])
    frac_bits = int(data['frac_bits'])
    k_segments = int(data['k_segments'])
    
    print("\n[Check 1] ZEN Matmul Logic (Kernel 1)")
    min_q = np.min(data['Q_sim_k1'])
    if min_q < 0:
        print(f"  ❌ Negative value in Q_sim_k1: {min_q} (Safety Margin Failure)")
    else:
        print(f"  ✅ All Matmul results are non-negative. Min: {min_q}")

    print("\n[Check 2] Softmax Bit-Decomposition (Limb Extraction)")
    X_recovered = (data['Q_sim_k1'].astype(object) << k_bits) + data['R_matmul_k1'].astype(object)
    limbs = data['limbs_k1'] 
    X_from_limbs = np.zeros_like(X_recovered, dtype=object)
    for k in range(k_segments):
        X_from_limbs += limbs[k].astype(object) << (16 * k)
    
    if np.array_equal(X_recovered, X_from_limbs):
        print("  ✅ X_recovered == Sum(limbs * 2^16k): Passed")
    else:
        print("  ❌ X Decomposition Failed!")

    print("\n[Check 3] Fixed-point Exponential Combination")
    exp_limbs = data['exp_limbs_k1'] 
    Q_exp = data['Q_exp_k1']         
    R_exp = data['R_exp_k1']         
    final_exp_total = data['exp_total_k1'].astype(object)
    
    current_val = exp_limbs[0].astype(object)
    all_step_ok = True
    for k in range(k_segments - 1):
        lhs = current_val * exp_limbs[k+1].astype(object)
        target_q = final_exp_total if k == k_segments - 2 else Q_exp[k].astype(object)
        rhs = (target_q << frac_bits) + R_exp[k].astype(object)
        
        if not np.array_equal(lhs, rhs):
            all_step_ok = False
            print(f"  ❌ Step {k} Multiplication mismatch!")
            break
        current_val = target_q

    if all_step_ok:
        print("  ✅ Remainder-based Exp multiplication: Passed")

    print("\n[Check 4] Final Softmax Division (P * Sum + R)")
    numerator = data['exp_total_k1'].astype(object) << frac_bits
    denominator = data['exp_sum_k1'].astype(object)
    P = data['P_k1'].astype(object)
    R_div = data['R_div_k1'].astype(object)
    
    div_check = (P * denominator) + R_div
    if np.array_equal(numerator, div_check):
        print("  ✅ Softmax Probability Division Logic: Passed")
    else:
        print("  ❌ Division Equality Failed!")

if __name__ == "__main__":
    debug_final_witness("outputs/outputs-model.transformer_0.mha.attn.witness.npz")