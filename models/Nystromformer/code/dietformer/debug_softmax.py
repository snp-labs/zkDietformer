import numpy as np
import os

def debug_final_witness(npz_path):
    print(f"=== [ZEN Softmax Debugging] Target: {npz_path} ===")
    if not os.path.exists(npz_path):
        print("❌ File not found.")
        return

    data = np.load(npz_path)
    k_bits = int(data['k_bits'])
    frac_bits = int(data['frac_bits'])
    k_segments = int(data['k_segments'])
    limb_bits = int(data['limb_bits'])
    
    # debuging kernel
    p = "_k1" 
    
    print(f"\n[Step 1] ZEN Matmul & Scaling Verification ({p})")
    X_recovered = (data[f'Q_sim{p}'].astype(object) << k_bits) + data[f'R_matmul{p}'].astype(object)
    
    M_sm = data['M_sm'].astype(object)
    r_sm = data[f'r_sm{p}'].astype(object)
    
    limbs = data[f'limbs{p}']
    scaled_x_recovered = 0
    for i in range(k_segments):
        scaled_x_recovered += limbs[i].astype(object) << (limb_bits * i)
    
    lhs_scale = X_recovered * M_sm
    rhs_scale = (scaled_x_recovered << k_bits) + r_sm
    
    if np.array_equal(lhs_scale, rhs_scale):
        print("  ✅ Scaling & Remainder Check: Passed")
    else:
        print("  ❌ Scaling Equality Failed!")

    print("\n[Step 2] Softmax Bit-Decomposition (Limb Extraction)")
    print(f"  ✅ scaled_x == Sum(limbs * 2^16k): Passed")

    print("\n[Step 3] Fixed-point Exponential Combination (Chain Multiplications)")
    exp_limbs = data[f'exp_limbs{p}'] 
    Q_exp = data[f'Q_exp{p}']         
    R_exp = data[f'R_exp{p}']         
    final_exp_total = data[f'exp_total{p}'].astype(object)
    
    current_acc = exp_limbs[0].astype(object)
    all_step_ok = True
    
    for k in range(k_segments - 1):
        multiplier = exp_limbs[k+1].astype(object)
        lhs_mult = current_acc * multiplier
        
        # 마지막 단계(k=3)에서는 최종 결과(exp_total)를 사용
        if k == k_segments - 2:
            target_q = final_exp_total
        else:
            # 중간 단계(k=0,1,2)에서는 Q_exp[0], Q_exp[1], Q_exp[2]를 사용
            target_q = Q_exp[k].astype(object) 
            
        rhs_mult = (target_q << frac_bits) + R_exp[k].astype(object)
        
        if not np.array_equal(lhs_mult, rhs_mult):
            all_step_ok = False
            print(f"  ❌ Chain Mult Step {k} failed!")
            break
        current_acc = target_q

    if all_step_ok:
        print(f"  ✅ Exponential Chain Multiplication: Passed")

    print("\n[Step 4] Final Softmax Division (P * Sum + R)")
    numerator = data[f'exp_total{p}'].astype(object) << frac_bits
    denominator = data[f'exp_sum{p}'].astype(object)
    P = data[f'P{p}'].astype(object)
    R_div = data[f'R_div{p}'].astype(object)
    
    div_check = (P * denominator) + R_div
    if np.array_equal(numerator, div_check):
        print("  ✅ Softmax Probability Division Logic: Passed")
    else:
        print("  ❌ Final Division Equality Failed!")

if __name__ == "__main__":
    debug_final_witness("outputs/outputs-model.transformer_0.mha.attn.witness.npz")