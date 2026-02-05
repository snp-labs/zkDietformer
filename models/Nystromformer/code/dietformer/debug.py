import numpy as np
import os

BN254_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def debug_circuit_consistency(npz_path, k_bits=22):
    print(f"=== Debugging Circuit Logic for: {npz_path} ===")
    data = np.load(npz_path)
    two_k = 1 << k_bits
    
    # 1. 입력 데이터 로드
    Q_q = data['Q_q'].astype(object)
    Q_k = data['Q_k'].astype(object)
    z_q = int(data['z_q'])
    z_k = int(data['z_k'])
    
    B, H, S, D = Q_q.shape
    R_mean_k = data['R_mean_k1'].astype(object)
    L = R_mean_k.shape[2]
    seg = S // L

    # 2. Q_k로부터 Q_landmarks_k 복구
    K_sum = Q_k.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_k = (K_sum // seg).astype(object)

    # 3. Q_q로부터 Q_landmarks_q 복구 (npz에서 가져오지 않고 직접 계산)
    Q_sum = Q_q.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_q = (Q_sum // seg).astype(object)
    R_mean_q_stored = data['R_mean_k2'].astype(object)

    # [Step 1] Landmark 생성 검증 (Q와 K 모두 확인)
    print("\n[Step 1] Verifying Landmark Pool (Avg Pool)...")
    for h in range(H):
        k_ok = np.array_equal(K_sum[0,h], Q_landmarks_k[0,h] * seg + R_mean_k[0,h])
        q_ok = np.array_equal(Q_sum[0,h], Q_landmarks_q[0,h] * seg + R_mean_q_stored[0,h])
        print(f"  ✅ Head {h}: K-Landmark {'Passed' if k_ok else 'FAILED'}, Q-Landmark {'Passed' if q_ok else 'FAILED'}")

    # [Step 2] Kernel 1 검증 (Q * K_L^T)
    print("\n[Step 2] Verifying Kernel 1 Equality (Q * K_L^T)...")
    verify_matmul_zen(Q_q, Q_landmarks_k, data['Q_sim_k1'], data['R_matmul_k1'], 
                      int(data['M_k1']), int(data['M_prime_k1']), z_q, z_k, two_k)

    # [Step 3] Kernel 2 검증 (Q_L * K_L^T)
    print("\n[Step 3] Verifying Kernel 2 Equality (Q_L * K_L^T)...")
    verify_matmul_zen(Q_landmarks_q, Q_landmarks_k, data['Q_sim_k2'], data['R_matmul_k2'], 
                      int(data['M_k2']), int(data['M_prime_k2']), z_q, z_k, two_k)

def verify_matmul_zen(W, X, Y_sim, R_mat, M, M_prime, z_w, z_x, two_k):
    """Freivalds 알고리즘 기반 ZEN 수식 검증"""
    B, H, Row, D = W.shape
    Col = X.shape[2]
    r = np.array([(42**i) % BN254_ORDER for i in range(Col)], dtype=object)
    r_sum = sum(r) % BN254_ORDER

    for h in range(H):
        head_W = W[0,h].astype(object)
        head_XT = X[0,h].T.astype(object)
        
        # LHS: (Y * 2^k + R) * r
        lhs = np.matmul((Y_sim[0,h].astype(object) * two_k + R_mat[0,h].astype(object)), r) % BN254_ORDER
        
        # RHS 연산 (ZEN 최적화 수식)
        xt_r = np.matmul(head_XT, r) % BN254_ORDER
        g1_r = np.matmul(head_W, xt_r) % BN254_ORDER
        g2_r = (head_W.sum(axis=1) * z_x * r_sum) % BN254_ORDER
        g3_val = (z_w * np.matmul(np.ones((1, D), dtype=object), xt_r)[0]) % BN254_ORDER
        const_vec = ((D * z_w * z_x + M_prime) * r_sum) % BN254_ORDER
        
        rhs = [(M * (g1_r[i] - g2_r[i] - g3_val + const_vec)) % BN254_ORDER for i in range(Row)]
        
        if np.array_equal(lhs, rhs):
            print(f"  ✅ Head {h}: Passed")
        else:
            print(f"  ❌ Head {h}: FAILED")

if __name__ == "__main__":
    debug_circuit_consistency("outputs/outputs-model.transformer_0.mha.attn.witness.npz")