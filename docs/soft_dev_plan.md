# SOFT 커널 개발 계획

확정 설계 변수: landmark 구간 평균 풀링(m=32), Gaussian kernel,
정규화 제거 (NeurIPS 판본), W_q = W_k (가중치 공유).
Newton-Raphson 2차 반복 (SOFT Eq. 10), T=20 초기값.

Masking convention:
- Q_proc (= K_proc): attention_mask (binary [B,S]) multiplicative pre-masking 후 1/sqrt(sqrt(D)) scaling.
  양자화 도메인에서 padding 위치의 값은 z_q (zero-point)이다.
  K_proc = Q_proc이므로 별도 K 마스킹 불필요.
- V: pre-masking 적용. padding 위치의 값은 z_v. chain matmul에서 (V - z_v) = 0이므로
  padding V가 temp 결과를 오염하지 않는다.
- B (Gaussian kernel): explicit post-kernel masking 불필요.
  근거: temp = B^T @ V에서 V가 pre-masked이므로 padding 위치의 기여는 ZEN matmul에서 0.
  Y[padding] 행은 O-projection 이후 discarded되므로 B[padding][:] 값은 무관하다.
- A (landmark kernel): masking 불필요. landmark는 Q_proc에서 구간 평균으로 생성되며,
  Q_proc이 pre-masked이므로 landmark 값에 padding 영향이 반영된다.
  참조 모델과 witness 파이프라인이 동일한 masked Q_proc에서 landmark를 생성하면 일치한다.

---

## 0. 선행 조사 요약

docs/soft_vs_nystrom.md에서 확정된 사항. 본 절은 참조 요약이다.

### 0-1. 구조 차이 핵심

| 항목 | SOFT | Nystrom |
|------|------|---------|
| 커널 함수 | Gaussian: exp(-\|\|q-k\|\|^2 / (2 sqrt(d))) | softmax(qk^T / sqrt(d)) |
| 정규화 | 없음 | 행 합 나눗셈 |
| W_q, W_k | 공유 (Q = K) | 별개 |
| 커널 행렬 수 | 2 (B, A) | 3 (K1, K2, K3) |
| Newton 반복 | 2차 (Eq. 10), T=20 | 4차 (HF iterative_inv), T=6 |
| softmax 횟수 | 0 | 3 |
| Gaussian lookup 횟수 | 2 (B, A) | 0 |
| 행 합 나눗셈 세트 | 0 | 3 |
| 사영 Freivalds | 2 (proj_q, proj_v) | 3 (proj_q, proj_k, proj_v) |
| MP 조건 검증 | 1, 3, 4 (조건 2 생략) | 동일 |

### 0-2. 양자화 상수

Nystrom 및 Linformer와 동일. 변경 없음.

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| 양자화 비트 | 12 (unsigned, [0,4095]) | quantize.py:10 |
| k_bits | 26 | config.rs KBITS_C=2 (16+10) |
| Softmax FRAC_BITS | 16 | 해당 없음 (SOFT는 softmax 미사용) |
| Gaussian FRAC_BITS | 16 (초기값, 실측 후 조정) | 신규 |
| Gaussian TABLE_FRAC_BITS | 10 (초기값, 실측 후 조정) | 신규 |
| Gaussian 테이블 크기 | 2^16 = 65536 (초기값) | Lasso 인프라 호환 |
| r_sm | 해당 없음 (정규화 없음) | 제거 |
| R_div | 해당 없음 (정규화 없음) | 제거 |

### 0-3. 커밋 분류 요약 (soft_vs_nystrom.md 1-5)

| 범주 | 항목 |
|------|------|
| d2 | W_Q, W_V, W_O (3종) |
| aux | proj_q, proj_v, Z* (A^{+}), temp, mid, Y, C_B, C_A (8종) |
| omega | Q_proc, Q~, norm_q, norm_qtilde, Q_dist_B, Q_dist_A, B, A |
| d1 | Gaussian table_idx, R_gauss, R_dist_B, R_dist_A, R_matmul (cross/chain/proj), MP remainder, MP bounded-diff |

**C_B/C_A aux 이동 근거 (Option 2)**: C_B/C_A를 omega로 유지하려면
커밋된 값(table_idx, r_gauss, norm)에서 정수 도메인으로 역복원이 가능해야 한다.
실측 결과 정수 경로 역복원에서 676/8192 원소가 off-by-1 이상 불일치.
floor 절삭이 비가역적이므로 C_B/C_A는 omega 자격을 상실한다.
aux로 이동하고 Freivalds 항등식으로 건전성을 보장한다.

---

## Stage A: Python witness 추출

### A-1. SOFT 어텐션 커널 구현

BERT 골격(BertForMaskedLM)에 어텐션 레이어만 SOFT 커널로 교체한다.
HuggingFace에 SOFTForMaskedLM 클래스가 없으므로 Linformer와 동일한 monkey-patch 방식.

**랜덤 가중치 사용의 근거**: 회로 비용은 가중치 값에 독립이다 (soft_vs_nystrom.md 1-4).

**구현 위치**:

| 파일 | 역할 |
|------|------|
| `soft_attention.py` (신규) | SOFTSelfAttention 클래스. W_q = W_k 가중치 공유 강제. |
| `run_soft_validation.py` (신규) | 진입점. BertForMaskedLM 로드 후 어텐션을 SOFT로 교체. |
| `soft_attention_capture.py` (신규) | monkey-patch로 중간값 캡처. |
| `soft_attention_pipeline.py` (신규) | 양자화 witness 생성 + NPZ 저장. |
| `gaussian_table.py` (신규) | Gaussian kernel lookup 테이블 생성. |

기존 파일 수정 없음. gelu_swap.py, quantize.py, normalization_pipeline.py 재사용.
softmax_table.py는 사용하지 않음 (Gaussian 테이블로 대체).

**SOFTSelfAttention 클래스 핵심**:

```python
class SOFTSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, seq_len, num_landmarks=32, n_iter=20):
        # W_q = W_k 공유: self.query = nn.Linear(hidden_size, hidden_size)
        # self.key = self.query  (동일 참조)
        # self.value = nn.Linear(hidden_size, hidden_size)
        # self.num_landmarks = num_landmarks
        # self.n_iter = n_iter (Newton-Raphson 반복 횟수)
```

### A-2. Newton-Raphson 2차 반복 구현

SOFT 논문 Algorithm 2 (Eq. 10)를 구현한다.
HuggingFace의 iterative_inv(4차)를 사용하지 않는다.

```python
def newton_raphson_pseudoinverse(A, n_iter=20):
    """SOFT Eq. 10: 2nd order Newton-Raphson for Moore-Penrose pseudoinverse.
    A: (B, H, m, m) Gaussian kernel matrix (landmark-to-landmark)
    """
    norm_A_1 = torch.norm(A, p=1, dim=(-2, -1), keepdim=True)
    alpha = 2.0 / (norm_A_1 ** 2)
    A_k = alpha * A
    for _ in range(n_iter):
        A_k = 2 * A_k - A_k @ A @ A_k
    return A_k
```

초기화: alpha = 2 / ||A||_1^2 (원논문 기본값).
실용적 변형(beta search)은 수렴 실패 시 도입한다.

### A-3. 추출 대상 목록

n = seq_len, m = num_landmarks, D = head_dim, H = num_heads, HD = H*D.

#### 사영 + 스케일링 (BERT 동일, 2종으로 축소)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q_x_input | [B, S, HD] | witness (Poseidon hash 입력) |
| Q_raw_proj_q (= Q_raw_proj_k) | [B, H, S, D] | aux |
| R_proj_q | [B, H, S, D] | d1 (26-bit Lasso) |
| Q_w_diff_q (= Q_w_diff_k) | [H, D, HD] | d2 |
| Q_scaled_q (= Q_scaled_k = Q_proc = K_proc) | [B, H, S, D] | omega |
| R_scale_q | [B, H, S, D] | d1 (26-bit Lasso) |
| Q_raw_proj_v | [B, H, S, D] | aux |
| R_proj_v | [B, H, S, D] | d1 (26-bit Lasso) |
| Q_w_diff_v | [H, D, HD] | d2 |

W_K 사영, K 스케일링 없음 (W_q = W_k이므로 Q_proc = K_proc).

#### Landmark 생성 (Nystrom 동일, 1종으로 축소)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q~ (= K~) | [H, m, D] | omega (Q_proc에서 segment averaging 결정) |
| R_mean_q | [H, m, D] | d1 (26-bit Lasso) |
| seg_size | scalar | 상수 |

K~ landmark 별도 생성 없음 (Q~ = K~).

#### 제곱 노름 (SOFT 전용)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| norm_q | [B, H, n] | omega (Q_proc 원소별 제곱 합) |
| norm_qtilde | [H, m] | omega (Q~ 원소별 제곱 합) |

#### Cross-term matmul (SOFT 전용)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q_sim_cross_B (Q_proc @ Q~^T) | [B, H, n, m] | omega |
| R_matmul_cross_B | [B, H, n, m] | d1 (26-bit Lasso) |
| Q_sim_cross_A (Q~ @ Q~^T) | [B, H, m, m] | omega |
| R_matmul_cross_A | [B, H, m, m] | d1 (26-bit Lasso) |

ZEN 파라미터: M_cross_B, M_prime_cross_B, z_sim_cross_B 등.

**cross-term 커밋 분류**: C_B, C_A는 aux (Freivalds-only). omega가 아닌 이유는
0-3절 Option 2 근거 참조.

#### Gaussian kernel distance + lookup (SOFT 전용)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q_dist_B | [B, H, n, m] | omega (dist 등식 Euclidean 몫, 유일 결정) |
| R_dist_B | [B, H, n, m] | d1 (dist 등식 Euclidean 나머지, 26-bit, C=2) |
| table_idx_B | [B, H, n, m] | d1 (Gaussian Lasso) |
| r_gauss_B | [B, H, n, m] | d1 (Gaussian remainder) |
| kernel_B | [B, H, n, m] | omega (table[table_idx]로 복원) |
| Q_dist_A | [B, H, m, m] | omega (dist 등식 Euclidean 몫, 유일 결정) |
| R_dist_A | [B, H, m, m] | d1 (dist 등식 Euclidean 나머지, 26-bit, C=2) |
| table_idx_A | [B, H, m, m] | d1 (Gaussian Lasso) |
| r_gauss_A | [B, H, m, m] | d1 (Gaussian remainder) |
| kernel_A | [B, H, m, m] | omega |
| M_A_B, M_B_B | scalar | ZEN 파라미터 (dist_B 등식 두 multiplier) |
| M_A_A, M_B_A | scalar | ZEN 파라미터 (dist_A 등식 두 multiplier) |
| z_dist | scalar (= 1) | ZEN 파라미터 (floor 절삭 guard offset) |

#### Pseudoinverse (Nystrom 동일)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q_inv (A^{+}) | [B, H, m, m] | aux (자유 증인) |
| z_inv | scalar | ZEN 파라미터 |
| M_inv_all[4] | 4 scalars | ZEN 파라미터 |
| Q_y_intermediates[4] | [4, B, H, m, m] | Freivalds 중간곱 |
| R_inv_remainders[4] | [4, B, H, m, m] | d1 (26-bit Lasso) |

#### Chain matmul (Nystrom 패턴, 입력만 변경)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q_sim_temp (B^T @ V) | [B, H, m, D] | aux |
| R_matmul_temp | [B, H, m, D] | d1 (26-bit Lasso) |
| Q_sim_mid (B @ A^{+}) | [B, H, n, m] | aux |
| R_matmul_mid | [B, H, n, m] | d1 (26-bit Lasso) |
| Q_sim_Y (mid @ temp) | [B, H, n, D] | aux |
| R_matmul_Y | [B, H, n, D] | d1 (26-bit Lasso) |

#### O-projection (BERT 동일)

| 중간값 | Shape | 분류 |
|--------|-------|------|
| Q_raw_proj_o | [B, H, S, D] | Freivalds 출력 |
| R_proj_o | [B, H, S, D] | d1 (26-bit Lasso) |
| Q_w_diff_o | [H, D, HD] | d2 |

### A-4. Gaussian kernel 계산 절차

Python witness 파이프라인의 Gaussian kernel 계산 순서:

```
1. Q_proc = quantize(Q_float)                    # 12-bit 양자화
2. Q~ = segment_average(Q_proc)                   # landmark 생성
3. norm_q[i] = sum_d (Q_proc[i][d] - z_q)^2      # 행별 제곱 노름 (정수)
4. norm_qtilde[j] = sum_d (Q~[j][d] - z_qtilde)^2 # landmark 제곱 노름
5. C_B = sign_bit_grouping_matmul(Q_proc, z_q, Q~^T, z_qtilde, ...)  # cross-term
6. LHS_B = M_A*(n_q+n_qt) - 2*M_B*(C_B-z_cross) + z_dist*2^k       # dist 등식 (정수)
   Q_dist_B = LHS_B // 2^k,  R_dist_B = LHS_B % 2^k                 # Euclidean division
7. (kernel_B, table_idx_B, r_gauss_B) = gaussian_lookup(Q_dist_B, R_dist_B, ...)  # 테이블 lookup
```

C_A 및 kernel_A도 동일 절차 (Q~ 자기 자신 간 거리).

**chain output 원칙**: chain 계산은 양자화 도메인에서 수행하되,
참조 모델 float forward의 중간값과 1:1 대응이 유지되어야 한다.
각 단계의 양자화 파라미터(M, M', z_w, z_x, z_y)는
sign_bit_grouping_matmul이 자동 산출한다.

**A-gate 측정을 위한 보고 항목**:
- dist_B, dist_A의 min/max/mean 통계 (테이블 범위 결정용)
- table_idx 클램핑 비율 (table boundary margin)
- MP 조건 1, 3, 4 각각의 max|diff| (epsilon 결정용)
- chain output Y의 역양자화 vs float Y의 CosSim

### A-5. Gaussian lookup 테이블

**테이블 생성** (gaussian_table.py):

```python
def generate_gaussian_table(table_frac_bits=10, frac_bits=16, z_dist=0):
    TABLE_SIZE = 1 << 16  # 65536
    scale = 1 << table_frac_bits
    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    shifted = np.maximum(indices - z_dist, 0.0)  # z_dist offset 흡수
    real_x = shifted / scale  # x in [0, 64)
    exps = np.exp(-real_x)    # exp(-x), decreasing from 1 to ~0
    fixed_exps = (exps * float(1 << frac_bits))
    fixed_exps = np.clip(fixed_exps, 0, float((1 << 64) - 1))
    return fixed_exps.astype(np.uint64)
# z_dist=1일 때: table[0]=table[1]=65536 (=1.0), table[2]=exp(-1/1024)*65536, ...
```

**softmax 테이블과의 차이**:
- 함수: exp(x) -> exp(-x) (부호 반전)
- 입력: signed logit -> unsigned distance (항상 비음수)
- 정규화: 행 합 나눗셈 있음 -> 없음 (R_div, r_sm 제거)
- 테이블 크기: 동일 (2^16), Lasso 인프라 호환

**softmax 테이블 재사용 불가 판정**: 함수 형태와 입력 부호가 다르므로 새 테이블 필요.
그러나 **lookup 회로 구조**(table_idx 할당, remainder, ZEN 등식, Lasso range check)는
softmax와 동일한 패턴을 따른다.

**테이블 파라미터 초기값과 조정 절차**:

| 파라미터 | 초기값 | 조정 기준 |
|----------|--------|-----------|
| FRAC_BITS | 16 | chain output CosSim < 0.9이면 증가 검토 |
| TABLE_FRAC_BITS | 10 | 테이블 boundary margin < 10%이면 증가 검토 |
| 테이블 크기 | 2^16 | Lasso 인프라 호환, 변경 불필요 예상 |
| M_gauss | 실측 산출 | s_y * 2^k_bits * 2^TABLE_FRAC_BITS |

Gaussian kernel 출력이 항상 [0, 1] 범위이므로 FRAC_BITS=16에서 출력 정밀도는
1/65536 ~ 1.5e-5이며, 이는 softmax 출력 정밀도와 동일하다.
입력이 비음수이므로 테이블의 "음수 절반"은 사용되지 않고 자연스럽게 0으로 수렴하는
구간만 유효하다 (exp(-12) < 1e-5, 16-bit fixed point에서 ~0).

---

## Stage B: Rust 증명 회로

### B-1. 모듈 배치

```
circuits/src/circuits/
  freivalds_matmul/
    soft_attn_circuit.rs    (신규) -- SOFTAttentionCircuit + load_soft_circuit()
  naive_matmul/
    soft_attn_circuit.rs    (신규) -- SOFTAttentionCircuitNot (naive wrapper)
  util/
    soft_attn_lookup.rs     (신규) -- run_soft_lasso_bundle, run_all_soft_remainders_with_link
```

기존 파일 수정 사항:
- `freivalds_matmul/mod.rs`: `pub mod soft_attn_circuit;` 추가
- `naive_matmul/mod.rs`: `pub mod soft_attn_circuit;` 추가
- `util/mod.rs`: `pub mod soft_attn_lookup;` 추가

공통 유틸 변경 불필요. 재사용 가젯 목록:

| 가젯 | 파일 | 용도 |
|------|------|------|
| `enforce_freivalds` | gadgets/matmul/quantized.rs | 모든 matmul 검증 |
| `enforce_scaling` | 동일 | Q 스케일링 (K 스케일링 불필요) |
| `gen_challenge_vector` | 동일 | Freivalds 챌린지 |
| `transpose_matrix` | 동일 | Q~^T, B^T 생성 |
| `compute_circuit_landmarks` | 동일 | segment averaging |
| `split_remainder_and_push` | gadgets/activation/gelu.rs | 26-bit 분할 |
| `ProjectionBundle`, `ScalingBundle`, `MatmulBundle`, `PseudoinverseBundle` | 동일 | 데이터 구조체 |
| `hash_commitments_to_field_v3` | util/utils.rs | Fiat-Shamir |

**신규 가젯**: 없음. 제곱 노름은 inline R1CS 제약으로 구현한다 (B-4 참조).
Gaussian lookup 등식은 enforce_softmax의 구조를 참고하되 정규화 부분을 제거한 별도 함수로 구현한다.

### B-2. Freivalds 항등식 목록 (전치 방향 확정)

SOFT 어텐션 커널의 matmul 검증. Q/V 사영 2종 + O-proj 1종은 BERT와 동일.

#### 사영 + 스케일링 (BERT와 동일, 2종 + 스케일링 1종)

| # | 항등식 | w_matrix | x_matrix_t | 전치 | r_pows 길이 | 결과 Shape |
|---|--------|----------|------------|------|-------------|------------|
| 1 | X @ W_Q^T = Q_raw | X [S x HD] | W_Q^T [HD x D] | 불필요 (NPZ가 전치 저장) | D | [S x D] |
| 2 | X @ W_V^T = V_raw | X [S x HD] | W_V^T [HD x D] | 동일 | D | [S x D] |

스케일링 1종 (enforce_scaling): Q_raw -> Q_proc. K 스케일링 불필요 (K_proc = Q_proc).

#### Cross-term (SOFT 전용, 2종)

| # | 항등식 | w_matrix | x_matrix_t | 전치 | r_pows 길이 | 결과 Shape |
|---|--------|----------|------------|------|-------------|------------|
| 3 | Q_proc @ Q~^T = C_B | Q_proc [n x D] | **transpose_matrix**(Q~) [D x m] | **필요** | m | [n x m] |
| 4 | Q~ @ Q~^T = C_A | Q~ [m x D] | **transpose_matrix**(Q~) [D x m] | **필요** (동일 전치 재사용) | m | [m x m] |

전치 근거: enforce_freivalds는 Y = W @ X_T를 검증한다 (quantized.rs:183-185).
Q_proc @ Q~는 차원 불일치. Q_proc [n x D] @ Q~ [m x D]는 D != m일 때 불가.
Q~^T [D x m]을 x_matrix_t로 전달하면 Y = Q_proc [n x D] @ Q~^T [D x m] = [n x m]. 정합.
Nystrom의 K1 = Q_proc @ K~^T에서 transpose_matrix(K~)와 동일한 패턴
(nystrom_attn_circuit.rs:402, 448).

#### MP pseudoinverse 검증 (Nystrom 동일, 3-4종)

| # | 항등식 | w_matrix | x_matrix_t | 전치 | r_pows 길이 | 결과 Shape | 용도 |
|---|--------|----------|------------|------|-------------|------------|------|
| 5 | A @ A^{+} = Q_y1 | kernel_A [m x m] | A^{+} [m x m] | 불필요 | m | [m x m] | 조건 3 대칭 |
| 6 | Q_y1 @ A = Q_y2 | Q_y1 [m x m] | kernel_A [m x m] | 불필요 | m | [m x m] | 조건 1: Q_y2 ~= A |
| 7 | A^{+} @ A = Q_y3 | A^{+} [m x m] | kernel_A [m x m] | 불필요 | m | [m x m] | 조건 4 대칭 |
| 8 | Q_y3 @ A^{+} = Q_y4 | Q_y3 [m x m] | A^{+} [m x m] | 불필요 | m | [m x m] | 조건 2 (생략) |

조건 2 생략 근거: 실측 오차가 range의 ~63%이며 bounded-difference 검증이 무의미하다
(nystrom_attn_circuit.rs:667 주석). SOFT Proposition 1도 ||A_k A A_k - A_k||의
단조 감소를 증명할 수 없다고 명시 (Appendix A.2.2).
단, Q_y4의 Freivalds matmul 자체는 유지할 수 있다 (constraint 비용이 적으므로).
생략 여부는 B-gate에서 결정.

#### Chain matmul (Nystrom 패턴, 3종)

| # | 항등식 | w_matrix | x_matrix_t | 전치 | r_pows 길이 | 결과 Shape |
|---|--------|----------|------------|------|-------------|------------|
| 9 | B^T @ V = temp | **transpose_matrix**(B) [m x n] | V [n x D] | **필요**: B 전치 | D | [m x D] |
| 10 | B @ A^{+} = mid | B [n x m] | A^{+} [m x m] | 불필요 | m | [n x m] |
| 11 | mid @ temp = Y | mid [n x m] | temp [m x D] | 불필요 | D | [n x D] |

전치 근거 (항등식 9): Nystrom의 temp = P_k3 @ V에서 P_k3 [m x S]은 이미 [m x n] 형태이므로
전치 불필요 (nystrom_attn_circuit.rs:718-720). SOFT의 B는 [n x m]이므로 B^T [m x n]을
W로 사용하려면 transpose_matrix(B)가 필요하다.

#### O-projection (BERT 동일, 1종)

| # | 항등식 | w_matrix | x_matrix_t | 전치 | r_pows 길이 | 결과 Shape |
|---|--------|----------|------------|------|-------------|------------|
| 12 | attn_out @ W_O^T = O | attn_out [S x HD] | W_O^T [HD x D] | 불필요 | D | [S x D] |

**합계**: 12종 (사영 2 + cross-term 2 + MP 4 + chain 3 + O-proj 1).
전치가 필요한 것: #3, #4 (Q~ 전치), #9 (B 전치). 총 2회 transpose_matrix 호출
(Q~ 전치는 1회 계산 후 #3, #4에서 재사용).

### B-3. ZenParams 구성

| matmul | z_w | z_x | 근거 |
|--------|-----|-----|------|
| X @ W_Q^T = Q_raw | z_input | F::zero() | W_Q는 diff 형태. z_x=0 (이중 차감 방지) |
| X @ W_V^T = V_raw | z_input | F::zero() | 동일 |
| Q_proc @ Q~^T = C_B | z_q | z_qtilde | 양자화된 raw 값. ZEN이 (A-z_w)(B-z_x) 계산 |
| Q~ @ Q~^T = C_A | z_qtilde | z_qtilde | 동일 |
| A @ A^{+} (MP) | z_A (Gaussian zero-point) | z_inv | Nystrom 동일 패턴. z_A != 0 (softmax는 z=0) |
| Q_y1 @ A (MP) | F::zero() | z_A | Q_y1은 MP 중간곱 |
| A^{+} @ A (MP) | z_inv | z_A | Nystrom 동일 |
| Q_y3 @ A^{+} (MP) | z_inv | z_inv | Nystrom 동일 |
| B^T @ V = temp | z_B | z_v | B는 Gaussian kernel 출력. z_B = Gaussian zero-point |
| B @ A^{+} = mid | z_B | z_inv | 동일 |
| mid @ temp = Y | z_mid | z_temp | chain 중간곱 |
| attn_out @ W_O^T = O | z_attn_out | F::zero() | W_O는 diff 형태 |

**Nystrom 대비 변경**: softmax 출력의 z=0이 Gaussian kernel 출력의 z=z_B로 변경.
MP 단계의 z_w도 softmax z=0에서 Gaussian z=z_A로 변경.

### B-4. 제곱 노름 gadget (SOFT 전용)

Gaussian kernel 거리 계산에 필요한 제곱 노름을 inline R1CS 제약으로 구현한다.

```rust
// norm_q[i] = sum_d (Q_proc[i][d] - z_q)^2
for i in 0..n {
    let mut norm_sum = FpVar::zero();
    for d in 0..D {
        let diff = &q_proc_vars[b][h][i][d] - &z_q_var;
        let sq = FpVar::new_witness(cs.clone(), || {
            diff.value().map(|v| v * v)
        })?;
        // R1CS: sq = diff * diff
        cs.enforce_constraint(
            lc!() + diff.variable(),
            lc!() + diff.variable(),
            lc!() + sq.variable(),
        )?;
        norm_sum = norm_sum + &sq;
    }
    norm_q_vars[i] = norm_sum;
}
```

제약 수: (n + m) * D per head (곱셈). 선형 합은 추가 제약 없음.
축소 fixture (n=8, m=4, D=4): 48 곱셈 제약.

**커밋 불필요**: norm_q, norm_qtilde는 omega (Q_proc와 Q~에서 결정론적으로 유일 결정).

### B-4b. Dist 제약 명세 (SOFT 전용)

Gaussian kernel 거리 Q_dist를 정수 도메인에서 제약하는 등식.
enforce_scaling (단일 multiplier)과 달리 **두 개의 multiplier** (M_A, M_B)를 사용한다.

**등식 형태** (two-multiplier ZEN):

```
Q_dist * 2^k + R_dist = M_A * (n1 + n2) - 2 * M_B * (Q_cross - z_cross) + z_dist * 2^k
```

- k = k_bits = 26
- n1, n2: 제곱 노름 (omega, B-4 R1CS로 결정)
- Q_cross: cross-term matmul 결과 (aux, Freivalds 검증)
- z_cross: Q_cross zero-point
- M_A = floor(2^k × s_q² / (s_dist × 2√D))
- M_B = floor(2^k × s_cross / (s_dist × 2√D))
- z_dist = 1 (floor 절삭 guard offset)

**z_dist 오프셋 근거**:

M_A와 M_B가 독립적으로 floor 절삭되므로 대각선 원소 (i==j, true distance≈0)에서
LHS가 음수가 될 수 있다.

| z_dist | Layer 1 kernel A 음수 원소 | LHS 최소값 |
|--------|--------------------------|-----------|
| 0 | 6/2048 | -6 |
| 1 | 0/2048 | ≥ 0 |

z_dist=1은 LHS에 2^26 = 67,108,864를 더한다. floor 절삭 오차(최대 ≈6)를 충분히 흡수한다.
Gaussian 테이블이 z_dist offset을 흡수: table[j]=1.0 for j ≤ z_dist.
clamping이나 부호 검사 게이트를 사용하지 않는다.

**Euclidean division 유일성**:

LHS ≥ 0이 보장되므로 (Q_dist, R_dist)는 유일하다.
Q_dist는 omega (등식에 의해 결정론적으로 유일 결정, 커밋 불필요).
R_dist는 d1 (26-bit Lasso range check, KBITS_C=2).

**실측 결과** (n=128, m=32, L=2, H=2, dense input, z_dist=1):

| Layer | Kernel | Total | Mismatch | Q_dist range | R_dist range |
|-------|--------|-------|----------|-------------|--------------|
| 0 | B | 8,192 | 0 | [1, 4094] | [8560, 67107056] |
| 0 | A | 2,048 | 0 | [1, 4094] | [0, 67100345] |
| 1 | B | 8,192 | 0 | [1, 4094] | [0, 67104832] |
| 1 | A | 2,048 | 0 | [1, 4094] | [0, 67075768] |

R_dist bit width: 26 (= k_bits). 12-bit Q_dist: min=1 (z_dist=1), max=4094.

**회로 제약 비용** (per batch per head):

| 커널 | 원소 수 | 선형 등식 (enforce_equal) | R_dist d1 limb |
|------|---------|------------------------|----------------|
| B | S×m = 4096 | 4096 | 8192 (16+10 split) |
| A | m×m = 1024 | 1024 | 2048 (16+10 split) |

### B-4c. 커밋 분류 원칙 (Stage B 가이드)

1. **omega**: 선행 커밋/공개 상수에서 결정론적으로 유일 결정되는 값.
   자유 증인이 아니라 등식 제약이 유일한 값을 강제한다.
   예: Q_dist (dist 등식), norm_q (R1CS 제곱), kernel (table lookup).

2. **aux**: Freivalds 항등식의 일부로만 참여하는 중간값.
   Freivalds random projection이 건전성을 보장하므로 값 자체의
   결정론적 복원이 불필요하다.
   예: proj_q, proj_v, temp, mid, Y, C_B, C_A, A^{+}.

3. **d1**: Lasso range check 대상. 값 범위를 증명해야 하는 나머지/인덱스.
   예: R_matmul, R_dist, table_idx, r_gauss, MP bounded-diff.

4. **d2**: 모델 가중치. 신뢰 모델과 동일함을 보장해야 한다.
   예: W_Q, W_V, W_O.

**omega vs aux 판정 기준**:
- 정수 도메인에서 커밋된 값만으로 역복원이 가능한가?
- 역복원에 floor, clip 등 비가역 연산이 포함되면 off-by-1 불일치 위험.
- 불일치가 존재하면 omega 자격 상실 → aux로 이동.
- 예: C_B/C_A 역복원(676/8192 off-by-1) → aux. Q_dist 순방향 등식(0 불일치) → omega.

### B-5. 커밋 그룹 구성

#### d2 (가중치)

W_Q, W_V, W_O (전 헤드 평탄화). 3종.
Nystrom 대비 W_K 제거.

#### aux (Freivalds-only 중간값)

| 순서 | 항목 | 근거 |
|------|------|------|
| 1 | proj_q_out [B][H][S][D] | 사영 Freivalds-only |
| 2 | proj_v_out [B][H][S][D] | 사영 Freivalds-only |
| 3 | Q_inv (A^{+}) [B][H][m][m] | 자유 증인 |
| 4 | Q_sim_temp [B][H][m][D] | chain Freivalds-only |
| 5 | Q_sim_mid [B][H][n][m] | chain Freivalds-only |
| 6 | Q_sim_Y [B][H][n][D] | chain Freivalds-only |
| 7 | C_B (Q_sim_cross_B) [B][H][n][m] | cross-term Freivalds-only (Option 2) |
| 8 | C_A (Q_sim_cross_A) [B][H][m][m] | cross-term Freivalds-only (Option 2) |

8종. Nystrom 대비 proj_k_out 제거, C_B/C_A 추가.

#### d1 (Lasso lookup 대상)

Push 순서 (Lasso target 순서와 일치 필수):

```
1. Gaussian table_idx (B 커널) [C=1]
2. Gaussian table_idx (A 커널) [C=1]
3. 26-bit remainders (C=2: 16+10):
   - R_dist_B, R_dist_A (dist 등식 Euclidean 나머지)
   - R_cross_B, R_cross_A
   - R_temp, R_mid, R_Y
   - R_proj_q, R_proj_v, R_proj_o
   - R_scale_q
   - R_inv[0..3] (MP 4단계)
4. Gaussian r_gauss_B, r_gauss_A (C 결정 필요, 초기 C=2)
5. MP bounded-diff:
   - cond1 [13-bit, C=1]
   - cond3 [10-bit, C=1]
   - cond4 [7-bit, C=1]
6. R_mean_q (segment averaging 나머지)
```

Nystrom 대비 제거: softmax table_idx 3세트, r_sm 3세트 (52-bit, C=4), R_div 3세트.
Nystrom 대비 추가: Gaussian table_idx 2세트, r_gauss 2세트.

---

## Masking 결정 (미결 항목 #3 해결)

### 강제 지점

| 시점 | 대상 텐서 | 강제 값 (양자화 도메인) | 근거 |
|------|-----------|----------------------|------|
| pre-kernel | Q_proc | padding -> z_q | Q_proc에서 파생되는 모든 값(Q~, norm, dist, kernel)이 consistent |
| pre-kernel | V | padding -> z_v | chain temp = B^T @ V에서 (V - z_v) = 0이므로 padding 기여 제거 |
| post-kernel | B, A | 불필요 | V pre-masking이 chain에서 padding 기여를 제거 |
| post-chain | Y | 선택적 | Y[padding] 행은 O-projection 이후 discarded. 필요시 z_Y 강제 |

**참조 모델 forward와 동기화**: Python SOFTSelfAttention.forward에서
Q_proc과 V에 attention_mask를 곱해 padding을 0으로 만든다 (float 도메인).
양자화 파이프라인에서 이 masked Q_proc을 양자화하면 padding 위치에 z_q가 할당된다.
회로에서도 동일한 masked 양자화 값을 사용하므로 동기화가 보장된다.

**literal 0 vs zero-point z**: 양자화 도메인에서 "값이 0"인 것은 zero-point z이다.
예: Q_proc[padding][d] = z_q (not 0). ZEN matmul에서 (Q_proc[i][d] - z_q) = 0이 되어
실질적으로 기여가 0이 된다. 이는 Nystrom의 K_proc masking과 동일한 원리이다.

---

## Gaussian lookup 테이블 파라미터 (미결 항목 #1 해결)

### 파라미터 결정

| 파라미터 | 결정값 | 근거 |
|----------|--------|------|
| 테이블 함수 | exp(-x) | softmax의 exp(x)와 다름. 새 테이블 필요 |
| 테이블 크기 | 2^16 = 65536 | 기존 Lasso 인프라(SOFTMAX_M=65536) 호환. 변경 불필요 |
| GAUSS_FRAC_BITS | 16 (초기값) | 출력 정밀도 1/65536 ~ 1.5e-5. softmax와 동일 |
| GAUSS_TABLE_FRAC_BITS | 10 (초기값) | 입력 해상도 1/1024. softmax와 동일 |
| 입력 범위 | [0, 64) (2^16 / 2^10) | exp(-64) ~ 1.6e-28, 16-bit fixed point에서 0 |
| 유효 범위 | [0, ~12) | exp(-12) ~ 6e-6, 16-bit에서 ~0. 실제 유효 entries ~ 12288개 |
| r_gauss bit width | 2 * k_bits = 52 (초기값) | softmax의 r_sm과 동일 산출 |
| r_gauss C | 4 (16+16+16+4) | softmax의 SM_KBITS_C=4와 동일 |

**입력 성질 활용**: Gaussian kernel 인자 -||q-k||^2 / (2*sqrt(d))는 항상 비양수이므로
테이블 입력 x = ||q-k||^2 / (2*sqrt(d))는 항상 비음수이다. 따라서:
1. 부호 처리 불필요 (softmax는 logit이 signed)
2. 테이블의 "음수 입력" 영역은 사용되지 않음
3. 출력이 항상 [0, 2^FRAC_BITS] 범위로 bounded, overflow 위험 없음

**softmax 테이블 재사용 불가**: exp(x) vs exp(-x)로 함수 자체가 다르다.
그러나 **lookup 회로 구조**(Lasso range check, ZEN 등식)는 동일 패턴이다.
enforce_softmax에서 정규화(R_div, r_sm, exp_sum, P) 부분을 제거하고
kernel_val = gauss_table[table_idx] 등식만 유지하면 된다.

### 실측 후 조정 절차

1. Python witness 생성 시 dist_B, dist_A의 min/max 통계를 보고한다.
2. table_idx의 boundary margin (min_idx, max_idx vs table_size-1)을 확인한다.
3. Margin이 10% 미만이면 TABLE_FRAC_BITS를 8로 축소하거나 table_size를 증가시킨다.
4. Chain output CosSim이 0.9 미만이면 FRAC_BITS를 20으로 증가시킨다.

---

## Freivalds 항등식 목록과 전치 방향 (미결 항목 #6 해결)

B-2에서 확정. 요약:

| # | W @ X_T | 전치 필요 | 근거 코드 참조 |
|---|---------|-----------|---------------|
| 1 | X @ W_Q^T | 불필요 | NPZ가 전치 저장 |
| 2 | X @ W_V^T | 불필요 | 동일 |
| 3 | Q_proc @ Q~^T | **필요** (Q~ 전치) | nystrom_attn_circuit.rs:402 패턴 |
| 4 | Q~ @ Q~^T | **필요** (동일 전치 재사용) | nystrom_attn_circuit.rs:431 패턴 |
| 5-8 | MP 4단계 (m x m) | 불필요 | 정방행렬, 전치 불필요 |
| 9 | B^T @ V | **필요** (B 전치) | Nystrom의 P_k3은 이미 [m x n]이므로 전치 불필요했으나, SOFT의 B는 [n x m]이므로 전치 필요 |
| 10 | B @ A^{+} | 불필요 | [n x m] @ [m x m] |
| 11 | mid @ temp | 불필요 | [n x m] @ [m x D] |
| 12 | attn_out @ W_O^T | 불필요 | NPZ가 전치 저장 |

---

## 양자화 고정

기존 양자화 파라미터는 Nystrom/Linformer와 완전히 동일하게 유지한다.

| 파라미터 | 값 | 변경 여부 |
|----------|-----|-----------|
| 양자화 비트 | 12 | 변경 없음 |
| k_bits | 26 | 변경 없음 |
| KBITS_C | 2 (16+10) | 변경 없음 |
| LayerNorm FRAC_BITS | 8 | 변경 없음 |
| GELU 대체 | SigmoidGELU | 변경 없음 |

신규 Gaussian 테이블 파라미터는 상기 "Gaussian lookup 테이블 파라미터" 참조.

---

## 검증 방법

### Stage A 완료 판정 (A-gate)

1. **회로 입력 vs witness 입력 일치**: Poseidon hash 입력 Q_x_input이
   참조 모델의 양자화된 입력과 일치.

2. **ZEN 등식 검증**: 모든 matmul에 대해
   `Q_sim * 2^k + R = M * (A-z_w)(B-z_x) + M'`이 Python에서 정확히 성립.

3. **Gaussian kernel 일관성**: kernel_B[i][j] == gauss_table[table_idx_B[i][j]]이
   모든 원소에서 성립. kernel_A 동일.

4. **Recovery injectivity (역방향 검증)**: committed 값만으로 C_B, C_A를 복원할 수 있는지 검증.
   허용 입력: table_idx, r_gauss, norm_q, norm_qtilde, R_matmul_cross, 공개 상수 (M_gauss, TABLE_FRAC_BITS, ZEN 파라미터).
   금지 입력: C_B, C_A 자체, Q_proc, Q~.
   복원 절차:
   (a) dist = table_idx * 2^TABLE_FRAC_BITS + r_gauss (Gaussian lookup 역산)
   (b) cross = (norm_q + norm_qtilde - dist) / 2 (거리 정의 역산)
   (c) cross가 ZEN 등식 `Q_sim * 2^k + R = M * (Q_proc - z_q)(Q~ - z_qtilde) + M'`의
       Q_sim과 일치하는지 확인 (R_matmul_cross 사용).
   모든 원소에서 (a)-(c) 성립 시 통과. 1개라도 불일치하면 STOP.

5. **Clamping 차단 (blocking)**: table_idx에 clamping이 1건이라도 발생하면 STOP.
   clamping = (table_idx <= 0) 또는 (table_idx >= TABLE_SIZE - 1).
   근거: clamping은 Gaussian lookup의 recovery injectivity를 파괴한다.
   동일한 table_idx에 여러 dist가 매핑되어 omega 분류가 무효화된다.
   Margin 보고: min_idx, max_idx, TABLE_SIZE - 1 - max_idx (참고용, 통과 기준은 clamping 0건).

6. **MP 조건 잔차**: 조건 1, 3, 4 각각의 max|diff|를 보고.
   Nystrom epsilon (4096/512/32) 이내인지 확인.
   초과 시 epsilon 조정 필요 (A-gate 측정 항목).

7. **Fidelity**: chain output Y의 역양자화와 float Y의 CosSim >= 0.9.
   O-projection 출력도 동일 기준.

8. **Misc 보고**: dist 통계 (min/max/mean), table_idx 클램핑 비율,
   Newton-Raphson 수렴 잔차 ||A A_k A - A||_F / ||A||_F.

### Stage B 완료 판정 (B-gate)

1. **완전성 (completeness)**: 정직한 witness에 대해 R1CS 만족.
   `cargo test soft_attn_single_layer_test` 통과.

2. **거부 (soundness smoke test)**: 최소 15종 조작 테스트.
   각 조작에 대해 실패 제약 번호를 보고.

   | 조작 | 기대 실패 지점 |
   |------|---------------|
   | aux commit tamper (proj_q) | 사영 Freivalds |
   | C_B tamper (aux) | cross-term Freivalds |
   | C_A tamper (aux) | cross-term Freivalds |
   | kernel_B tamper | Gaussian lookup 등식 |
   | norm_q tamper | 제곱 노름 R1CS |
   | Q_dist_B tamper | dist 등식 (enforce_equal) |
   | R_dist_B tamper | dist 등식 또는 Lasso range check |
   | A^{+} tamper | MP Freivalds |
   | Q_sim_temp tamper | chain Freivalds |
   | Q_sim_mid tamper | chain Freivalds |
   | Q_sim_Y tamper | chain Freivalds 또는 O-proj |
   | Q_raw_proj_o tamper | Poseidon hash |
   | mask tamper (padding) | mask enforcement |
   | W_Q tamper (d2) | 사영 Freivalds (Q) |
   | W_V tamper (d2) | 사영 Freivalds (V) |
   | W_O tamper (d2) | O-proj Freivalds |
   | Q~ landmark tamper | cross-term Freivalds (#3, #4) |
   | table_idx tamper | Gaussian lookup Lasso range check |

3. **제약 수 내역**: 단계별 constraint count 보고.
   Poseidon hash 비율 주의 (축소 fixture에서 80%+ 예상).

---

## 비용 요약 (Nystrom 대비 SOFT 변경)

### 커밋 그룹 변동

| 커밋 그룹 | 추가 | 제거 | 순 변동 |
|-----------|------|------|---------|
| aux | C_B [n×m], C_A [m×m] | proj_k (W_q=W_k) | +n×m + m×m - S×D |
| d1 | R_dist_B [n×m], R_dist_A [m×m], Gauss table_idx ×2, r_gauss ×2 | softmax table_idx ×3, r_sm ×3, R_div ×3 | 상당 감소 |
| d2 | - | W_K | -1종 |
| omega | Q_dist_B, Q_dist_A | C_B, C_A (aux로 이동) | 유사 |

### 제약 변동 (per batch per head)

| 구분 | SOFT 추가 | SOFT 제거 | 비고 |
|------|-----------|-----------|------|
| 제곱 노름 R1CS | (n+m)×D 곱셈 | - | SOFT 전용 |
| dist 등식 (enforce_equal) | (n×m + m×m) × 1 선형 | - | SOFT 전용 |
| R_dist d1 limb | (n×m + m×m) × 2 | - | SOFT 전용 |
| Gaussian lookup | 2세트 | - | softmax lookup 3세트 제거 |
| softmax 정규화 | - | R_div ×3, r_sm ×3 | SOFT는 정규화 없음 |
| K 사영/스케일링 | - | proj_k, scale_k | W_q=W_k |

---

## 구현 작업 단위 분해

### Stage A 작업 목록

| # | 작업 | 예상 파일 | 의존 |
|---|------|-----------|------|
| A1 | SOFTSelfAttention 클래스 구현 (W_q=W_k, segment averaging, Newton-Raphson, chain) | soft_attention.py | - |
| A2 | BERT 모델의 어텐션을 SOFT로 교체하는 진입점 | run_soft_validation.py | A1 |
| A3 | monkey-patch 캡처 함수 (중간값 캡처) | soft_attention_capture.py | A1 |
| A4 | Gaussian kernel 테이블 생성 | gaussian_table.py | - |
| A5 | Q/V 사영 + 스케일링 양자화 (기존 함수 재사용) | soft_attention_pipeline.py | A3 |
| A6 | 제곱 노름 + cross-term matmul 양자화 | soft_attention_pipeline.py | A5 |
| A7 | Gaussian distance 조합 + table lookup | soft_attention_pipeline.py | A4, A6 |
| A8 | MP pseudoinverse (Newton-Raphson 2차) 양자화 | soft_attention_pipeline.py | A7 |
| A9 | Chain matmul (temp, mid, Y) + O-proj 양자화 | soft_attention_pipeline.py | A7, A8 |
| A10 | NPZ 저장 + A-gate 검증 assert | soft_attention_pipeline.py | A9 |
| A11 | 축소 fixture 생성 스크립트 (S=8, m=4, H=2, D=4) | gen_small_soft.py | A10 |

### Stage B 작업 목록

| # | 작업 | 예상 파일 | 의존 |
|---|------|-----------|------|
| B1 | SOFTAttentionCircuit 구조체 정의 | soft_attn_circuit.rs | - |
| B2 | load_soft_circuit NPZ 로더 | soft_attn_circuit.rs | B1 |
| B3 | generate_constraints: 사영 + 스케일링 (Q, V만. K 없음) | soft_attn_circuit.rs | B1 |
| B4 | generate_constraints: 제곱 노름 + cross-term Freivalds | soft_attn_circuit.rs | B3 |
| B5 | generate_constraints: Gaussian lookup 등식 (정규화 없음) | soft_attn_circuit.rs | B4 |
| B6 | generate_constraints: MP pseudoinverse 검증 | soft_attn_circuit.rs | B5 |
| B7 | generate_constraints: chain matmul (temp, mid, Y) | soft_attn_circuit.rs | B5, B6 |
| B8 | generate_constraints: O-proj + Poseidon hash | soft_attn_circuit.rs | B7 |
| B9 | generate_constraints: d1/d2/aux 커밋 push | soft_attn_circuit.rs | B8 |
| B10 | Lasso lookup 함수 (run_all_soft_remainders_with_link) | soft_attn_lookup.rs | B9 |
| B11 | run_soft_paths 전체 파이프라인 | soft_attn_circuit.rs | B10 |
| B12 | naive_matmul wrapper | naive_matmul/soft_attn_circuit.rs | B11 |
| B13 | 완전성 테스트 + soundness smoke test (15종) | 테스트 코드 | B11 |

공통 유틸 변경 필요 여부: **없음**.
enforce_freivalds, 커밋/해시 함수 모두 커널 무관 인터페이스.
