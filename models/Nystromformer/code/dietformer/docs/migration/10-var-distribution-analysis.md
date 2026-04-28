# Phase 0 추가 — LayerNorm 입력 var 분포 측정 및 영향 평가

> 본 문서는 Step 2 검증에서 발견된 lookup table inherent 정밀도 한계
> (`docs/migration/09-bert-layernorm-validation.md` 참조)에 대한
> **정량적 영향 평가**다.
>
> ⚠️ **현 환경 제약**: torch / transformers 미설치 상태에서 작성됨.
> 실제 측정 수치는 사용자가 측정 스크립트를 실행한 뒤 채울 자리만 명시했다.
> 측정 스크립트(`docs/migration/scripts/measure_var_distribution.py`)는 자동으로
> 본 문서와 같은 형식의 markdown을 생성하므로, 결과를 그대로 붙여넣거나
> `--md` 옵션으로 본 문서를 덮어쓸 수 있다.

---

## 1. 배경

기존 RMSNorm pipeline과 새 LayerNorm pipeline 모두 lookup table의 인덱스
도메인이 *정수* `[0, 65535]`이며, 표 함수가 `inv = 1/√(int_index + ε)` 형태다.
이로 인해:

| 입력 var | lookup index | 결과 |
|---------|-------------|------|
| ∈ [0, 1) | 0 | `1/√ε` (catastrophically huge) |
| ∈ [1, 2) | 1 | `1/√1 ≈ 1` (좁은 정밀도) |
| ∈ [10, 100) | 10–99 | 정상 동작 |
| ≥ 100 | 100+ | 정상 동작 |

→ **var 값이 어느 영역에 분포하는지가 본 논문 §7 결과의 신뢰도를 결정.**

---

## 2. 측정 방법

### 2.1 측정 대상
- **`bert-base-uncased`** (BERT baseline) — 12 layer × 2 LN = 24 LayerNorm 위치
- **`uw-madison/nystromformer-512`** (Nystromformer) — 12 layer × 2 LN = 24 LayerNorm 위치

### 2.2 입력 텍스트 (10개, 코드에 박힘 — 재현성)

`docs/migration/scripts/measure_var_distribution.py:23-35` 의 `SAMPLE_TEXTS`:

```python
SAMPLE_TEXTS: list[str] = [
    "The capital of France is [MASK].",
    "Machine learning is a subset of [MASK] intelligence.",
    "Photosynthesis converts [MASK] dioxide into oxygen.",
    "The quick brown fox jumps over the lazy [MASK].",
    "She opened the book and began to [MASK] silently.",
    "Quantum computers use qubits instead of classical [MASK].",
    "The ancient ruins were discovered by an [MASK] team.",
    "Climate change is one of the most pressing [MASK] of our time.",
    "He grabbed his umbrella because the forecast predicted [MASK].",
    "Newton's third law states that for every action there is an equal and opposite [MASK].",
]
```

> Note: `[MASK]` 토큰은 BERT는 그대로 인식되지만, Nystromformer 는 Albert SP 토크나이저라 `[MASK]`가 단어로 split 될 수 있다. 측정은 *MLM 결과*가 아니라 *LayerNorm 입력 분포*이므로 이 차이는 결과에 영향 없음.

### 2.3 측정 절차
1. `BertForMaskedLM.from_pretrained("bert-base-uncased")` / `NystromformerForMaskedLM.from_pretrained("uw-madison/nystromformer-512")` 로드 (`eval()`, `torch.no_grad()`).
2. 각 모델 토크나이저로 `padding=True, truncation=True, max_length=32`로 인코딩 (기본 32, `--max-length`로 조정).
3. 모든 LayerNorm 모듈에 forward hook 부착:
   - `encoder.layer[i].attention.output.LayerNorm` (post-attention)
   - `encoder.layer[i].output.LayerNorm` (post-FFN)
4. forward 1회 → hook이 `inputs[0].var(dim=-1, unbiased=False)` 값 수집.
5. **attention_mask로 padding 토큰 위치는 제외** — pad 토큰의 var은 의미 없음.
6. 통계 산출.

### 2.4 실행 명령
```bash
cd /Users/minhae/Projects/dietformer/Dietformer/models/Nystromformer/code/dietformer
python docs/migration/scripts/measure_var_distribution.py \
    --json docs/migration/var_distribution.json \
    --md   docs/migration/10-var-distribution-analysis.md
```
첫 실행 시 BERT (~440 MB) + Nystromformer (~500 MB) 다운로드. 이후는 캐시 사용.
측정 시간: 각 모델 ~30초 (CPU), 총 ~1분 이내.

---

## 3. 측정 결과 — 1차 (BERT-base-uncased, 짧은 입력 1개)

### 3.1 측정 출처
사용자가 작업 폴더에 직접 작성한 빠른 측정 스크립트 `var_test.py` 실행 결과 (2026-04-28). 측정 조건:
- 모델: `BertModel.from_pretrained("bert-base-uncased")` (eval 모드, `torch.no_grad()`)
- 입력: 단일 문장 6 토큰 (`[CLS] this is a test [SEP]`)
- LayerNorm 위치: 24개 (post-attention 12 + post-FFN 12)
- var 산식: `x.var(dim=-1)` (PyTorch 기본 = unbiased=True, Bessel 보정 / N-1; LayerNorm의 unbiased=False 와 dim=768 에서 차이는 약 0.1%, 결론에 무영향)
- 총 var 관측치: 24 layer × 6 token = **144 sample**

### 3.2 BERT-base-uncased 통계

| 지표 | 값 |
|------|-----|
| min | **0.0239** |
| p5 | **0.1942** |
| median | **0.8098** |
| p95 | **12.8074** |
| max | **792.6041** |

### 3.3 영역별 비율 (분위수에서 추론)

| 영역 | 추론 범위 | 비고 |
|------|----------|------|
| **var ∈ [0, 1) — catastrophic** | **약 50–65%** | median이 0.81로 [0, 1) 안에 위치 → 50%+ 이미 catastrophic. p95=12.8 > 1 이므로 상한 65% 이내. |
| var ∈ [1, 10) | 약 30–45% | median 위에서 p95 사이 대부분 |
| var ∈ [10, 100) | 약 5% 안팎 | p95 = 12.8 |
| var ≥ 100 | 매우 소수 | max 792 — 일부 outlier만 |

> 정확한 frac_lt_1 / frac_1_to_10 등은 raw 데이터가 있어야 산출 가능. 사용자가 선호하면 `measure_var_distribution.py`로 정확 측정 가능 (그쪽은 12 layer × multi-sample × multi-position breakdown 까지 모두 산출).

### 3.4 uw-madison/nystromformer-512 — 미측정

본 단계에서는 측정되지 않음. 단 동일 LayerNorm 구조이고 학습 코퍼스(BookCorpus + Wiki) 비슷하므로 분포 모양도 비슷할 것 *(추정)*. Nystromformer 측정도 동일 스크립트로 가능 (`--model nystrom-only`).

---

## 4. 판단 기준

스크립트가 자동으로 다음 규칙으로 A / B / C 를 산출한다 (`measure_var_distribution.py:_make_judgment`).

### A — 안전 (Phase 4 후순위)
- **두 모델 모두** `p05 ≥ 10` AND `frac_lt_1 < 5%`
- 의미: var의 95%가 `[10, ∞)` 영역에 있어 lookup이 의미 있는 값을 반환. catastrophic 비율 무시 가능.
- 권장: 현재 lookup table 그대로 진행. §7 limitation에 한 줄 명시 ("LayerNorm var lookup uses an integer-domain table; values < 10 fall back to a coarse approximation, but our measurements show <5% of activations land there in practice").

### B — 주의 (Phase 4 권장)
- 위 A 조건 미충족 AND **두 모델 모두** `median ≥ 10` AND `frac_lt_1 < 30%`
- 의미: 양호한 영역이 다수이지만 catastrophic 영역도 무시할 수 없는 비중. 일부 token에서 reconstruction 노이즈가 큼.
- 권장: 마감 일정 여유 있으면 Phase 4 (lookup table 재설계) 진행. 급하면 §7에 limitation 명시 후 진행하되 결과 해석에 주의.

### C — 필수 (Phase 4 즉시)
- 두 모델 중 하나라도 `median < 10` OR `frac_lt_1 > 30%`
- 의미: var의 절반 이상이 lookup table이 의미 있는 값을 못 주는 영역. 현재 lookup으로는 §7 결과가 사실상 noise.
- 권장: Phase 4 + Phase 5 우선 진행. **중요: 기존 dim=128 모델로 측정한 결과(`circuits/src/circuits/freivalds_matmul/experiment.log`)가 같은 한계를 가졌을 가능성 — Phase 4 완료 후 재측정해서 §7 baseline 갱신 필요.**

---

## 5. 판단 — **C (필수, Phase 4 즉시)**

### 5.1 기준 적용

| 판단 | 조건 | BERT 측정값 | 충족? |
|------|------|-------------|------|
| A 안전 | p5 ≥ 10 AND frac_lt_1 < 5% | p5 = 0.19 | ❌ p5 ≪ 10 |
| B 주의 | median ≥ 10 AND frac_lt_1 < 30% | median = 0.81 | ❌ median ≪ 10 |
| **C 필수** | median < 10 OR frac_lt_1 > 30% | median = 0.81 < 10, frac_lt_1 ≥ 50% | ✅ **두 조건 모두 충족** |

판단 기준상 두 모델 중 **하나라도** C 조건을 만족하면 전체 판단이 C. BERT 측정만으로 이미 결정.

### 5.2 의미

- **median이 0.81로 1보다 낮음** = LayerNorm 입력 var 의 절반 이상이 catastrophic [0, 1) 영역에 분포.
- 현재 lookup table (`1/√(int_index + ε)`)은 var < 1 인 입력에 대해 모두 `index = 0` 으로 collapse → `table[0] ≈ 1/√ε ≈ 10⁶` 라는 6 자릿수 잘못된 값 반환.
- 즉 **본 논문 §7 RoBERTa baseline / BERT baseline 결과는 LayerNorm 모듈에서 50% 이상의 위치에서 수치적으로 무의미한 reconstruction 을 가짐**. R1CS constraint 자체는 내부 정합이라 회로 verify는 통과하지만, "양자화 노이즈가 모델 정확도에 미치는 영향" 같은 §7 해석에서 결과를 신뢰할 수 없다.
- 같은 lookup 구조(`extract_inv_rms_single`)를 쓰는 **기존 dim=128 RMSNorm 측정 (`circuits/src/circuits/freivalds_matmul/experiment.log`)도 동일한 한계의 영향을 받았을 가능성이 높다** *(추정 — 같은 표 함수형, 같은 인덱스 매핑이므로)*. Phase 4 완료 후 §7 baseline 자체를 재측정해야 함.

### 5.3 권장 다음 단계 (한 단락)

**Phase 1 (모델 교체) 진입 전에 Phase 4 (lookup table 재설계)를 우선 처리해야 한다.** 현재 구조 그대로 진행하면 §7 결과가 무의미해진다. Phase 4 작업 범위는: (a) `layernorm_table.py` / `rms_table.py` 의 인덱스 매핑을 정수 도메인 → fixed-point scaled 도메인으로 재설계 (`int_index = floor(var · scale)` 형태), (b) 그에 맞춰 `extract_inv_std_single` / `extract_inv_rms_single` 의 인덱스 산출 변경, (c) Rust 회로 측 `RMSNormSubtableStrategy` (`circuits/src/util/mlp_lookup.rs`)의 인덱스 매핑도 동일하게 수정 — 이 부분은 Phase 5 와 묶어서 진행. **Phase 4를 건너뛰고 Phase 1로 가면 BERT/Nystromformer 두 baseline 모두 무의미한 숫자만 산출된다.**

### 5.4 추가 확인 필요

- [ ] Nystromformer (`uw-madison/nystromformer-512`) 측정도 한 번 돌리는 것이 합리적 — `python docs/migration/scripts/measure_var_distribution.py --model nystrom-only` 또는 사용자의 `var_test.py`를 Nystromformer 용으로 한 줄만 바꿔 실행. *(BERT가 이미 C 트리거이므로 결과적으로 영향은 없으나, Phase 4 lookup 설계 시 두 모델의 var 범위 모두 커버해야 하므로 분포 자료가 필요.)*
- [ ] 보다 많은 입력 텍스트로 measure_var_distribution.py (10 sample, max_length=32) 실행 — 6 토큰 단일 입력은 통계적으로 약함. 본 판단(C)은 견고하지만 Phase 4 lookup 설계 시 더 많은 표본이 도움.
- [ ] 측정에서 **layer별 분포** 확인 — 일부 layer만 catastrophic 인지, 전체가 그런지에 따라 lookup 설계가 다름. var_test.py 출력에는 layer breakdown 없음 (`measure_var_distribution.py`는 layer별 표 자동 생성).

---

## 6. 사전 추정 (참고용, 실측 전)

> 아래는 *공개 문헌과 일반적 활성값 분포에 기반한 추정*이며, 실측 결과로 대체되어야 한다.

### 6.1 BERT 계열 (정성 추정)
- BERT-base-uncased는 학습된 모델로, hidden state는 layer마다 다른 normalization 단계를 거친다.
- **post-attention LN 입력 (잔차 연결 직후)**: 잔차 누적으로 magnitude 가 커지는 경향. var이 *수십 ~ 수백* 단위가 흔함 *(추정)*.
- **post-FFN LN 입력**: FFN의 큰 hidden_dim×4 곱셈 결과로 magnitude 가 더 큼. var 이 *수백* 단위 *(추정)*.
- → 실제 BERT의 var 는 거의 [10, 1000] 영역에 분포할 가능성이 높음. **A 판단** 가능성이 가장 높음 *(추정)*.

### 6.2 Nystromformer (정성 추정)
- 원리적으로 BERT와 같은 LayerNorm 구조이므로 분포 모양도 비슷할 것 *(추정)*.
- 단 학습 데이터 규모(BookCorpus + Wiki ~16 GB)가 BERT(~16 GB)와 같지만 학습 step 수가 다를 수 있어 활성값 magnitude가 미세하게 다를 수 있음.

### 6.3 추정의 한계
- 위는 정성적 추측이며 실측이 우선. **사용자 실행 후 본 섹션은 삭제 또는 “실측이 추정과 일치/불일치” 메모만 남길 것.**

---

## 7. 사용자 행동 가이드

### 다음 단계 (단순)
1. 인터넷 접근 가능 환경에서 다음 실행:
   ```bash
   cd /Users/minhae/Projects/dietformer/Dietformer/models/Nystromformer/code/dietformer
   python docs/migration/scripts/measure_var_distribution.py \
       --json docs/migration/var_distribution.json \
       --md   docs/migration/10-var-distribution-analysis.md
   ```
2. 자동 생성된 markdown을 검토 (본 문서가 자동 덮어써짐).
3. 산출된 A / B / C 판단을 보고 Phase 4 진행 여부 결정.

### 실행 환경 노트
- HF cache 확인: `ls ~/.cache/huggingface/hub/` — `models--bert-base-uncased`, `models--uw-madison--nystromformer-512` 디렉터리가 있으면 다운로드 skip.
- 메모리: 두 모델 모두 ~500 MB. CPU만으로 충분 (forward 1회).
- 측정 결과의 결정적 재현: 입력 텍스트가 코드에 박혀 있고 forward에 dropout이 없으므로(eval 모드) 동일 환경에서 항상 동일한 결과.

### 만약 Nystromformer 모델 다운로드가 실패하면
- HF 모델 ID 재확인: `uw-madison/nystromformer-512` (사용자 메시지에서 "mlpen/nystromformer-512" 라 적혔던 부분은 HF에 없음).
- 또는 BERT만 측정: `--model bert-only` 옵션. 이 경우 자동 판단은 BERT 분포만 보고 작성됨 (Nystromformer는 추정으로 남김).

---

## 8. 사용자 결정 필요 사항

측정 결과를 받은 후 다음 결정이 필요:

1. **판단 A인 경우**: Phase 4를 후순위로 두고 곧바로 Phase 1 (Nystromformer HF 모델 교체)로 진행할지.
2. **판단 B인 경우**: Phase 4를 우선할지, §7 limitation 명시로 우회할지.
3. **판단 C인 경우**: Phase 4 lookup 재설계 방식 선택:
   - 옵션 (i): 표 인덱스 도메인을 `int_index = floor(var · scale_factor)` 형태로 바꿔 [0, 1) 영역에도 의미 있는 값.
   - 옵션 (ii): 표 자체는 그대로 두고, var 값을 잘 정의된 quantization grid로 매핑 (`get_quantization_params` 의 `min(0, data_min)` clipping을 var 텐서에 한해 풀어줌).
   - 옵션 (iii): `frac_bits` 를 8 → 16 으로 늘려 fixed-point 정밀도 향상 (단 Rust 회로의 표 크기 상수도 변경 필요 — Phase 5 비용 증가).

각 옵션의 trade-off는 측정 결과를 보고 별도 보고서에서 정리.
