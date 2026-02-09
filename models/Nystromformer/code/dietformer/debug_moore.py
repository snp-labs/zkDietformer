import numpy as np
import os

# 유한체 BN254 Order (zk-SNARK 환경 모사)
BN254_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def verify_matmul_zen_direct(W, X, Y_sim, R_mat, M, M_prime, z_w, z_x, z_y, two_k):
    B, H, Row, D_inner = W.shape
    Col = X.shape[-1]
    
    r = np.array([(42**i) % BN254_ORDER for i in range(Col)], dtype=object)

    for h in range(H):
        head_W = W[0,h].astype(object)
        head_X = X[0,h].astype(object) 
        head_Y = Y_sim[0,h].astype(object)
        head_R = R_mat[0,h].astype(object)
        
        # LHS: (Y * 2^k + R) @ r
        # 실제 회로의 결과값 검증 대상
        lhs = (np.matmul(head_Y * two_k + head_R, r)) % BN254_ORDER
        
        # RHS: M * (G1 + Constant + M_prime - G2 - G3) @ r
        # 1. G1_vec = (W @ X) @ r
        g1_vec = np.matmul(np.matmul(head_W, head_X) % BN254_ORDER, r) % BN254_ORDER
        
        # 2. G2_vec = z_x * (W_row_sums) @ r 
        # (W @ ones) * z_x 가 각 행의 보정치이며, 여기에 r이 곱해짐
        # W_row_sums: (Row,)
        w_row_sums = head_W.sum(axis=1) % BN254_ORDER
        # g2_vec[i] = w_row_sums[i] * z_x * sum(r) 이 아니라, 
        # 원본 행렬곱에서 G2는 (W @ Z_x_mat) 이므로 r과의 결합은 아래와 같음
        r_sum = sum(r) % BN254_ORDER
        g2_vec = (w_row_sums * z_x % BN254_ORDER * r_sum % BN254_ORDER) % BN254_ORDER
        
        # 3. G3_vec = z_w * (X_col_sums) @ r
        # z_w * ones @ X @ r
        x_col_sums = head_X.sum(axis=0) % BN254_ORDER # (Col,)
        g3_val_linear = np.dot(x_col_sums, r) % BN254_ORDER
        g3_vec = (z_w * g3_val_linear) % BN254_ORDER
        
        # 4. Constant & M_prime (모든 행에 공통으로 더해지는 r_sum 배수)
        constant_val = (D_inner * z_w * z_x) % BN254_ORDER
        common_term = (constant_val + M_prime) % BN254_ORDER
        common_vec = (common_term * r_sum) % BN254_ORDER

        rhs = []
        for i in range(Row):
            # 수식: M * (G1_r + Common_r - G2_r - G3_r)
            val = (g1_vec[i] + common_vec - g2_vec[i] - g3_vec) % BN254_ORDER
            rhs.append((M * val) % BN254_ORDER)
        
        rhs = np.array(rhs, dtype=object)

        if not np.array_equal(lhs, rhs):
            return False, h, lhs, rhs
    return True, None, None, None

def debug_pseudoinverse_consistency(npz_path, k_bits=22):
    print(f"=== Debugging Pseudoinverse Circuit Logic: {npz_path} ===")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return

    data = np.load(npz_path)
    two_k = 1 << k_bits
    
    # ===== 데이터 로드 및 shape 검증 =====
    A_q = data['P_k2'].astype(object)
    Q_inv = data['Q_inv_k2'].astype(object)
    
    print(f"[DEBUG] A_q shape: {A_q.shape}, dtype: {A_q.dtype}")
    print(f"[DEBUG] Q_inv shape: {Q_inv.shape}, dtype: {Q_inv.dtype}")
    
    Q_y_inv = data['Q_y_inv_intermediates'].astype(object)
    R_inv = data['R_inv_remainders'].astype(object)
    
    print(f"[DEBUG] Q_y_inv shape: {Q_y_inv.shape}")  # 꼭 [2, B, H, Row, Col]인가?
    print(f"[DEBUG] R_inv shape: {R_inv.shape}")      # 꼭 [4, B, H, Row, Col]인가?
    
    # shape 문제가 있으면 여기서 즉시 드러남
    
    M_all = data['M_inv_all']
    Mp_all = data['M_prime_inv_all']
    
    # zero-point 로드 (whole_pipeline.py의 설정과 일치시켜야 함)
    # debug_moore.py 수정 부분
    
    # 1. zero-point 명시적 로드
    z_inv = int(data['z_inv_k2'])
    z_p2 = 0
    
    # 2. 각 단계의 결과물(Y)이 가질 zero-point (whole_pipeline.py의 설정과 일치)
    z_y1 = z_p2  # Step 1-1 결과인 Q_y1의 zero-point
    z_y3 = z_inv # Step 2-1 결과인 Q_y3의 zero-point

    # Q_y_inv 구조: [Q_y1, Q_y2, Q_y3, Q_y4]
    steps = [
        # (Name, W, X, Target_Y, R, M, Mp, z_w, z_x, z_y)
        ("Step 1-1: A * A+ = Y1", A_q, Q_inv, Q_y_inv[0], R_inv[0], M_all[0], Mp_all[0], z_p2, z_inv, z_y1),
        ("Step 1-2: Y1 * A = A",  Q_y_inv[0], A_q, Q_y_inv[1], R_inv[1], M_all[1], Mp_all[1], z_y1, z_p2, z_p2),
        ("Step 2-1: A+ * A = Y3", Q_inv, A_q, Q_y_inv[2], R_inv[2], M_all[2], Mp_all[2], z_inv, z_p2, z_y3),
        ("Step 2-2: Y3 * A+ = A+", Q_y_inv[2], Q_inv, Q_y_inv[3], R_inv[3], M_all[3], Mp_all[3], z_y3, z_inv, z_inv),
    ]

    for name, W, X, Y, R, M, Mp, zw, zx, zy in steps: # zy 추가
        print(f"\n[Verify] {name}")
        print(f"  W shape: {W.shape}, X shape: {X.shape}, Y shape: {Y.shape}")
        ok, head, lhs, rhs = verify_matmul_zen_direct(W, X, Y, R, int(M), int(Mp), zw, zx, zy, two_k)
        if ok:
            print(f"  ✅ Circuit Consistency Passed")
        else:
            print(f"  ❌ FAILED at Head {head}")
            print(f"     LHS shape: {np.array(lhs).shape}, RHS shape: {np.array(rhs).shape}")
            print(f"     LHS[0]: {lhs[0]}")
            print(f"     RHS[0]: {rhs[0]}")
            print(f"     Difference: {(lhs[0] - rhs[0]) % BN254_ORDER}")
if __name__ == "__main__":
    # 생성된 npz 파일 경로를 입력하세요.
    debug_pseudoinverse_consistency("outputs/outputs-model.transformer_0.mha.attn.witness.npz")