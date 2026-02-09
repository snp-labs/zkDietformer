import numpy as np
import os

BN254_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def debug_circuit_consistency(npz_path, k_bits=22):
    print(f"=== Debugging Circuit Logic for: {npz_path} ===")
    data = np.load(npz_path)
    two_k = 1 << k_bits
    
    Q_q = data['Q_q'].astype(object)
    Q_k = data['Q_k'].astype(object)
    z_q = int(data['z_q'])
    z_k = int(data['z_k'])
    
    B, H, S, D = Q_q.shape
    R_mean_k = data['R_mean_k1'].astype(object)
    L = R_mean_k.shape[2]
    seg = S // L

    K_sum = Q_k.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_k = (K_sum // seg).astype(object)

    Q_sum = Q_q.reshape(B, H, L, seg, D).sum(axis=-2)
    Q_landmarks_q = (Q_sum // seg).astype(object)
    R_mean_q_stored = data['R_mean_k2'].astype(object)

    print("\n[Step 1] Verifying Landmark Pool (Avg Pool)...")
    for h in range(H):
        k_ok = np.array_equal(K_sum[0,h], Q_landmarks_k[0,h] * seg + R_mean_k[0,h])
        q_ok = np.array_equal(Q_sum[0,h], Q_landmarks_q[0,h] * seg + R_mean_q_stored[0,h])
        print(f"  ✅ Head {h}: K-Landmark {'Passed' if k_ok else 'FAILED'}, Q-Landmark {'Passed' if q_ok else 'FAILED'}")

    kernels = [
        ("Kernel 1", data['Q_sim_k1'], data['R_matmul_k1'], data['M_k1'], data['M_prime_k1']),
        ("Kernel 2", data['Q_sim_k2'], data['R_matmul_k2'], data['M_k2'], data['M_prime_k2']),
        ("Kernel 3", data['Q_sim_k3'], data['R_matmul_k3'], data['M_k3'], data['M_prime_k3']),
        ("Temp (P_k3 @ V)", data['Q_sim_temp'], data['R_matmul_temp'], data['M_temp'], data['M_prime_temp']),
        ("Mid (P_k1 @ Q_inv)", data['Q_sim_mid'], data['R_matmul_mid'], data['M_mid'], data['M_prime_mid']),
        ("Final X (Mid @ Temp)", data['Q_sim_x'], data['R_matmul_x'], data['M_x'], data['M_prime_x'])
    ]

    for name, q_sim, r_mat, m_val, m_prime in kernels:
        print(f"\n[Step] Verifying {name}...")
        
        min_val = np.min(q_sim)
        if min_val < 0:
            print(f"  ❌ {name} Negative Value Detected! Min: {min_val}")
        else:
            print(f"  ✅ {name} Non-negative check passed.")

        if name == "Kernel 1":
            verify_matmul_zen(Q_q, Q_landmarks_k, q_sim, r_mat, int(m_val), int(m_prime), z_q, z_k, two_k)
        elif name == "Kernel 2":
            verify_matmul_zen(Q_landmarks_q, Q_landmarks_k, q_sim, r_mat, int(m_val), int(m_prime), z_q, z_k, two_k)
        elif name == "Kernel 3":
            verify_matmul_zen(Q_landmarks_q, Q_k, q_sim, r_mat, int(m_val), int(m_prime), z_q, z_k, two_k)
        elif name == "Temp (P_k3 @ V)":
            Q_v = data['Q_v'].astype(object)
            z_v = int(data['z_v'])
            P_k3 = data['P_k3'].astype(object)
            z_pk3 = 0
            verify_matmul_zen(P_k3, np.transpose(Q_v, (0, 1, 3, 2)), q_sim, r_mat, int(m_val), int(m_prime), z_pk3, z_v, two_k)
        elif name == "Mid (P_k1 @ Q_inv)":
            P_k1 = data['P_k1'].astype(object)
            Q_inv_k2 = data['Q_inv_k2'].astype(object)
            z_inv = int(data['z_inv_k2'])
            z_pk1 = 0
            verify_matmul_zen(P_k1, np.transpose(Q_inv_k2, (0, 1, 3, 2)), q_sim, r_mat, int(m_val), int(m_prime), z_pk1, z_inv, two_k)
        elif name == "Final X (Mid @ Temp)":
            Q_sim_mid = data['Q_sim_mid'].astype(object)
            Q_sim_temp = data['Q_sim_temp'].astype(object)
            z_sim_mid = int(data['z_sim_mid'])
            z_sim_temp = int(data['z_sim_temp'])
            verify_matmul_zen(Q_sim_mid, np.transpose(Q_sim_temp, (0, 1, 3, 2)), q_sim, r_mat, int(m_val), int(m_prime), z_sim_mid, z_sim_temp, two_k)

def verify_matmul_zen(W, X, Y_sim, R_mat, M, M_prime, z_w, z_x, two_k):
    B, H, Row, D_inner = W.shape
    Col = X.shape[2] 
    
    r = np.array([(42**i) % BN254_ORDER for i in range(Col)], dtype=object)
    r_sum = sum(r) % BN254_ORDER

    for h in range(H):
        head_W = W[0,h].astype(object)
        head_XT = X[0,h].T.astype(object) 
        
        # LHS: (Y * 2^k + R) * r
        lhs = np.matmul((Y_sim[0,h].astype(object) * two_k + R_mat[0,h].astype(object)), r) % BN254_ORDER
        
        # RHS: (M * (G1 - G2 - G3 + Const) * r)
        xt_r = np.matmul(head_XT, r) % BN254_ORDER
        g1_r = np.matmul(head_W, xt_r) % BN254_ORDER
        g2_r = (head_W.sum(axis=1) * z_x * r_sum) % BN254_ORDER
        g3_val = (z_w * np.sum(xt_r)) % BN254_ORDER
        
        n = D_inner
        const_vec = ((n * z_w * z_x + M_prime) * r_sum) % BN254_ORDER
        
        rhs = [(M * (g1_r[i] - g2_r[i] - g3_val + const_vec)) % BN254_ORDER for i in range(Row)]
        
        if np.array_equal(lhs, rhs):
            print(f"  ✅ Head {h}: Passed")
        else:
            print(f"  ❌ Head {h}: FAILED")
            print(f"     LHS[0]: {lhs[0]}")
            print(f"     RHS[0]: {rhs[0]}")

if __name__ == "__main__":
    debug_circuit_consistency("outputs/outputs-model.transformer_0.mha.attn.witness.npz")