import numpy as np
import os

BN254_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def verify_matmul_zen(W, X, Y_sim, R_mat, M, M_prime, z_w, z_x, k_bits):
    two_k = 1 << k_bits
    B, H, S_row, S_col = Y_sim.shape # 결과 행렬의 차원을 기준으로 잡음
    
    # 랜덤 벡터 r의 길이를 결과 행렬의 열 개수(S_col, 즉 512)에 맞춤
    r = np.array([(42**i) % BN254_ORDER for i in range(S_col)], dtype=object)
    
    for h in range(H):
        w_head = W[0, h].astype(object)     # (S, D)
        x_head = X[0, h].astype(object)     # (D, S)
        y_sim_head = Y_sim[0, h].astype(object) # (S, S)
        r_mat_head = R_mat[0, h].astype(object) # (S, S)

        # LHS: (Y_sim * 2^k + R) @ r -> (S, S) @ (S, 1) = (S, 1)
        lhs = ((y_sim_head * two_k + r_mat_head) @ r) % BN254_ORDER
        
        # RHS 계산 (동일)
        n = W.shape[3] # D_inner (64)
        w_sum = np.sum(w_head, axis=1) # (S,)
        x_sum_r = (x_head @ r) % BN254_ORDER # (D,)
        
        term1 = (w_head @ x_sum_r) % BN254_ORDER
        term2 = (w_sum * (z_x * np.sum(r))) % BN254_ORDER
        
        # x_head @ r의 모든 요소의 합
        x_head_sum_all_r = np.sum(x_sum_r) % BN254_ORDER
        term3 = (z_w * x_head_sum_all_r) % BN254_ORDER
        
        term4 = (n * z_w * z_x * np.sum(r)) % BN254_ORDER
        
        rhs = (M * (term1 - term2 - term3 + term4 + M_prime * np.sum(r))) % BN254_ORDER
        
        if not np.all(lhs == rhs): # 행렬 비교 시 all 사용
            return False
    return True

def debug_bert_witness(npz_path, k_bits=22, frac_bits=16):
    print(f"=== 🔍 BERT Attention Circuit Debugging: {os.path.basename(npz_path)} ===")
    data = np.load(npz_path)
    
    # [1] Attention Score Matmul 검증: (Q-zq) @ (K-zk)^T
    print("\n[Step 1] Attention Score Matmul (Q * K^T):")
    Q_q = data['Q_q']
    Q_k = data['Q_k']
    K_T = np.swapaxes(Q_k, -1, -2)
    ok_matmul = verify_matmul_zen(
        Q_q, K_T, data['Q_sim_attn'], data['R_matmul_attn'], 
        int(data['M_attn']), int(data['M_prime_attn']), 
        int(data['z_q']), int(data['z_k']), k_bits
    )
    print(f"  - Matmul Consistency: {'✅ PASS' if ok_matmul else '❌ FAIL'}")

    # [2] Softmax Scaling 검증: Q_sim * M_sm = Index_Limbs
    print("\n[Step 2] Softmax Index Scaling:")
    Q_sim = data['Q_sim_attn'].astype(object)
    M_sm = int(data['M_sm'])
    r_sm = data['r_sm'].astype(object)
    limbs = data['limbs'].astype(object) # (5, B, H, S, S)
    
    # 분해된 limb들을 다시 합침
    limb_bits = 16 # 고정값
    recovered_idx = sum(limbs[i] << (limb_bits * i) for i in range(5))
    
    lhs_sm = Q_sim * M_sm
    rhs_sm = (recovered_idx << k_bits) + r_sm
    
    if np.array_equal(lhs_sm, rhs_sm):
        print("  - Scaling & Limb Extraction: ✅ PASS")
    else:
        print("  - Scaling & Limb Extraction: ❌ FAIL")

    # [3] Softmax Exponential Chain 검증
    print("\n[Step 3] Softmax Exponential Chain (exp_total):")
    exp_limbs = data['exp_limbs'].astype(object) # (5, B, H, S, S)
    Q_exp = data['Q_exp'].astype(object)
    R_exp = data['R_exp'].astype(object)
    exp_total = data['exp_total'].astype(object)
    
    chain_ok = True
    current_acc = exp_limbs[0]
    for i in range(4): # 5-segment 이므로 4번의 곱셈
        target = exp_total if i == 3 else Q_exp[i]
        lhs_chain = current_acc * exp_limbs[i+1]
        rhs_chain = (target << frac_bits) + R_exp[i]
        if not np.array_equal(lhs_chain, rhs_chain):
            chain_ok = False
            break
        current_acc = target
    print(f"  - Exp Chain Multiplication: {'✅ PASS' if chain_ok else '❌ FAIL'}")

    # [4] Softmax Division (Normalization) 검증
    print("\n[Step 4] Softmax Normalization (P = exp / sum):")
    exp_sum = data['exp_sum'].astype(object)
    P = data['P_attn'].astype(object)
    R_div = data['R_div_attn'].astype(object)
    
    lhs_div = exp_total << frac_bits
    rhs_div = P * exp_sum + R_div
    
    if np.array_equal(lhs_div, rhs_div):
        print("  - Division Consistency: ✅ PASS")
    else:
        print("  - Division Consistency: ❌ FAIL")

if __name__ == "__main__":
    # 생성된 BERT NPZ 파일 경로를 넣으세요
    debug_bert_witness("outputs_bert/outputs_bert-batch0-bert.encoder.layer.0.attention.self.witness.npz")