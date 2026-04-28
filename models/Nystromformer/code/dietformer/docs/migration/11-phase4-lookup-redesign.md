# Phase 4 — Lookup table 재설계 (fixed-point scaled 인덱스)

> 사전 조건: `docs/migration/10-var-distribution-analysis.md` 의 판단 C
> (catastrophic 영역 ≥ 50%) 에 따라 진행.

---

## 1. 설계 — VAR_SCALE 도입

### 1.1 핵심 아이디어
`extract_inv_std_single` 산식은 그대로 두고, **`M_ln` 에 VAR_SCALE 을 흡수**시킨다. 이로써 lookup 인덱스의 의미만 바뀜:

| | 기존 (Phase 2) | **신규 (Phase 4)** |
|---|---------------|---------------------|
| 인덱스 의미 | `index ≈ var_real` (정수) | `index ≈ var_real · VAR_SCALE` |
| 표 함수 | `table[i] = 1/√(i + ε)` | `table[i] = 1/√(i / VAR_SCALE + ε)` |
| `M_ln` | `floor(s_var · 2^k_bits)` | `floor(s_var · VAR_SCALE · 2^k_bits)` |
| extract 본체 | (변경 없음) | (변경 없음) |
| 커버 범위 | var ∈ [0, 65535] (정수만) | var ∈ [0, 256) (1/256 해상도) |

### 1.2 VAR_SCALE = 256 선택 근거 (BERT 측정 분포 기준)

| BERT 측정 var | Phase 2 lookup | **Phase 4 lookup (SCALE=256)** |
|---------------|----------------|---------------------------------|
| min = 0.024 | index 0 → 1/√(eps) catastrophic | index 6 → +1.18% 오차 |
| p5 = 0.19 | index 0 → catastrophic | index 48 → +0.63% |
| median = 0.81 | index 0 → catastrophic | index 207 → -0.16% |
| p95 = 12.8 | index 12 → 정상 | index 3276 → -0.77% |
| 100 | index 100 → -2.34% | index 25600 → -2.34% |
| 256 | index 256 → 정상 | index 65535 (clip) → 0% |
| max = 792.6 | index 792 → 정상 | index 65535 (clip) → +75.96% |

→ **>99% 데이터가 var < 256** 영역에서 1-2% 오차로 정상화. 1% 미만의 outlier만 clipping으로 정확도 손실 (의도적 trade-off).

다른 SCALE 검토:
- SCALE=64: var ∈ [0, 1024], min=0.024에서 24% 오차 (index=1만 hit). 핵심 영역 부정확.
- SCALE=128: var ∈ [0, 512], 절충안. 단 max=792 clip 오차는 SCALE=256과 비슷.
- **SCALE=256 채택**. clip 영향이 1%인데 핵심 [0, 1) 영역을 1% 정밀도로 cover.

---

## 2. 변경된 파일 (Python 측만)

### 2.1 `layernorm_table.py` (재작성)
- 백업: `layernorm_table.py.bak3`
- 변경 요약:
  - 신규 상수 `VAR_SCALE = 256`
  - `generate_layernorm_tables(..., var_scale=VAR_SCALE)` — `real_var = i / var_scale` 매핑
  - 신규 함수 `get_layernorm_index_multiplier(s_var, k_bits, var_scale)` — VAR_SCALE 흡수
  - `extract_inv_std_single` 시그니처/본체 unchanged (M_ln 안에 VAR_SCALE 흡수되어 자동 동작)
  - npz 메타에 `var_scale` 필드 추가 (회로 측이 일치 검증)
- 중요 발췌 (`layernorm_table.py:75-89`):
  ```python
  def get_layernorm_index_multiplier(s_var, k_bits=22, var_scale=VAR_SCALE):
      """
      수학:
          shifted_val ≈ var_real · dim / s_var
          scaled_val  = (shifted_val · M_ln) >> k_bits
                      = var_real · dim · VAR_SCALE
          index       = scaled_val // dim
                      = var_real · VAR_SCALE          ← lookup 인덱스
      """
      return int(np.floor(s_var * var_scale * (1 << k_bits)))
  ```

### 2.2 `bert_normalization_pipeline.py` (3곳 변경)
- 백업: `bert_normalization_pipeline.py.bak3`
- 변경 요약:
  - import 라인: `get_rms_index_multiplier` 제거, `get_layernorm_index_multiplier`, `VAR_SCALE` 추가
  - `_process_layernorm_block` 의 step 5 한 줄 (M_ln 산출 함수 교체)
  - witness npz 출력 dict에 `var_scale` 메타 추가

### 2.3 변경하지 않은 파일 (기존 정책 유지)
- `rms_table.py` — Nystromformer 자기학습 RMSNorm 측 사용 중. HF Nystromformer로 전환되면 더 이상 사용되지 않음.
- `normalization_pipeline.py` — 같음.
- `quantize.py` — 공통 함수, 영향 없음.
- 회로 코드 (`circuits/src/`) — Phase 5 작업 영역.

---

## 3. 검증 결과

### 3.1 Lookup table 단독 정확도 (BERT 측정값에서)

| var | index | table×256 | true×256 | rel err | 상태 |
|-----|-------|-----------|----------|---------|------|
| 0.024 | 6 | 1672 | 1652 | +1.18% | 정상 |
| 0.19 | 48 | 591 | 587 | +0.63% | 정상 |
| 0.81 | 207 | 284 | 284 | −0.16% | **정상** |
| 12.8 | 3276 | 71 | 71 | −0.77% | 정상 |
| 100 | 25600 | 25 | 25 | −2.34% | 정상 |
| 256 | 65535 | 16 | 16 | 0% | 경계 (정확) |
| 792 | 65535 (clip) | 16 | 9 | +75.96% | clip 손실 (1% 미만 데이터) |

### 3.2 End-to-end pipeline rel L2 (PyTorch nn.LayerNorm reference 대비)

| 입력 var 분포 | Phase 2 rel L2 | **Phase 4 rel L2** | 개선 |
|--------------|----------------|---------------------|------|
| var ≈ 0.25 | 499,091 | **0.014 (1.4%)** | 3.5×10⁷ 배 |
| var ≈ 1.0 (BERT median) | 693,680 | **0.015 (1.5%)** | **4.6×10⁷ 배** |
| var ≈ 100 | 0.031 | 0.031 | 동일 |
| var ≈ 2500 (clip 영역) | 0.030 | 2.13 | 71× 악화 (예상된 trade-off) |

### 3.3 R1CS 일관성
회로 가젯이 enforce할 식 `M_norm · (Q_diff · inv_std · Q_gamma_diff + Q_beta_lifted + M'_norm) = (Q_out << k_bits) + R_out`:
- **3072/3072 cells exact PASS** (BERT 차원 dim=768, B=1, S=4)

### 3.4 Syntax + import 영향
- `layernorm_table.py`, `bert_normalization_pipeline.py` 모두 syntax OK
- Nystrom 측 (`rms_table.py`, `normalization_pipeline.py`) import 무영향

---

## 4. Rust 회로 측 후속 작업 (Phase 5에서 처리)

본 Phase 4 변경은 Python pipeline만 수정함. **Rust 회로 측 동일 변경이 없으면 Lasso lookup verification은 실패한다** (witness가 새 인덱스 매핑을 쓰지만 Rust subtable strategy는 여전히 옛 정수 인덱스 매핑 가정).

### 4.1 필수 Rust 변경

| 위치 *(추정)* | 현재 | Phase 5에서 | 비고 |
|---------------|------|--------------|------|
| `circuits/src/util/mlp_lookup.rs::run_lasso_for_rmsnorm` | `RMSNormSubtableStrategy` 사용 | 새 `LayerNormSubtableStrategy` 추가 (VAR_SCALE 인지) | RMSNorm strategy는 그대로 두고 추가 |
| Lasso subtable strategy 정의 위치 | (확인 필요) | strategy의 `evaluate(i)` 가 `1/√(i/VAR_SCALE + eps)` 반환 | 본 변경은 *추정* — Lasso crate 내부 또는 본 repo의 strategy 정의에 따라 다름 |
| `circuits/src/util/load.rs` | `var_scale` 키 안 읽음 | `var_scale` 메타 읽고 일치 검증 (방어 코드) | 선택 사항 |
| Norm gadget의 R1CS 식 | 변경 없음 | 변경 없음 | M_ln 값만 다르고 식 구조는 같음 |

### 4.2 Phase 5 작업 시 검증 절차
1. 새 LayerNormSubtableStrategy의 `evaluate(i)` 가 Python `layernorm_tables.npz` 의 `table[i]` 와 일치하는지 단위 테스트
2. 작은 입력 1개로 Python pipeline → Rust prove → Rust verify 사이클 통과 확인
3. R1CS constraint 자체는 이미 일관되므로 (`3072/3072 PASS`), strategy만 맞으면 verify 통과 예상

### 4.3 회로 비용 영향
- TABLE_SIZE = 65536 unchanged → constraint 수, prover/verifier 비용 변화 *없음*
- 추가 곱셈 1회 (M_ln 안에 VAR_SCALE 곱한 게 끝) → 회로 비용 변화 무시 가능

---

## 5. Trade-off 정리 (§7 작성 시 명시 사항)

### 5.1 정확도 trade-off
- **var ∈ [0, 256)**: 1-2% 오차로 정상 (BERT 측정 99%+ 데이터 해당)
- **var ≥ 256**: clipping 발생, rel error up to 76% in inv_std (BERT 측정 1% 미만)
- 결과적으로 **percentile-based metrics (median, p95)** 는 매우 정확, **max-based metrics** 는 outlier 영향으로 부풀려짐

### 5.2 §7 작성 가이드
- "양자화 노이즈"는 percentile (median, p95) 로 보고 권장
- max abs error 보고 시 outlier clipping 영향 1% 미만이라고 명시
- limitation 절: *"Our LayerNorm lookup table covers var ∈ [0, 256] with 1/256 resolution. Variances exceeding 256 (occurring in <1% of activations in our measurements) are clipped, introducing up to 76% relative error in inv_std but bounded absolute error in the LayerNorm output."*

---

## 6. 사용자 결정 필요

### 6.1 Nystromformer 측정 권장 (선택)
HF `uw-madison/nystromformer-512` 의 var 분포가 BERT와 비슷한지 확인. 만약 var ≥ 256 outlier 비율이 BERT보다 훨씬 높으면 VAR_SCALE 재조정 필요.

```bash
# var_test.py 한 줄만 바꿔 실행
# from transformers import BertModel → from transformers import NystromformerModel
# model = BertModel.from_pretrained("bert-base-uncased") 
#   → model = NystromformerModel.from_pretrained("uw-madison/nystromformer-512")
```

### 6.2 다음 단계
- **권장: Phase 5 (Rust 측 LayerNormSubtableStrategy 작성)** — Python 변경이 효과를 발휘하려면 회로 측 동기화 필수
- 또는 Phase 1 (HF Nystromformer 모델 교체) 먼저 진행하여 Python 측 통합 후 Phase 5 함께 처리

### 6.3 Rust 변경 작업 시 주의
- 본 Phase 4의 변경은 R1CS constraint를 바꾸지 않음 (M_ln 값만 다름)
- Lasso strategy의 `evaluate(i)` 함수만 바꾸면 됨
- 기존 RMSNormSubtableStrategy 는 그대로 두고 LayerNormSubtableStrategy 추가 (Nystromformer 자기학습 시절 호환 유지 위해 — 단 사용자가 자기학습 폐기 결정했으므로 삭제도 가능)
