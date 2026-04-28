# Phase 0 — 마이그레이션 계획 (Phase 1–6)

> 본 문서는 작업 폴더(`models/Nystromformer/code/dietformer`) 내부 정보로 작성되었으며, **Rust 회로(`circuits/`) 측 일부 정보는 *(추정)***임을 명시한다.
> Phase 0 완료 후 사용자 승인을 받아 Phase 1로 진행한다.

---

## Phase 의존성 그래프

```
Phase 0 (현재 상태 문서화) ─── 완료
   │
   ▼
Phase 1 (Nystrom 모델 교체)            Phase 2 (Baseline 교체)
   │                                       │
   └──────────────┬────────────────────────┘
                  ▼
          Phase 3 (npz 추출 코드 재작성)
                  │
                  ▼
        Phase 4 (lookup table 재생성)
                  │
                  ▼
            Phase 5 (Rust 가젯 수정)
                  │
                  ▼
          Phase 6 (E2E 검증)
```

- Phase 1과 Phase 2는 **병렬 진행 가능** (서로 다른 entry script).
- Phase 3은 1·2 모두 끝나야 시작 가능 (양쪽 hook 코드를 동시에 정렬해야 함).
- Phase 4는 활성화 함수와 normalization을 양쪽 모델 모두에 대해 한 번에 결정해야 하므로 Phase 3 다음.
- Phase 5는 Rust 코드 수정. Phase 4의 lookup 함수 결정에 의존.
- Phase 6은 E2E (Python 추출 → Rust 회로 검증).

---

## Phase 1 — Nystromformer 직접 학습 → HF 공개 모델 교체

### 1.1 작업 항목 체크리스트
- [ ] `from transformers import NystromformerForMaskedLM, AutoTokenizer` import 추가 (`run_validation.py`)
- [ ] `model = NystromformerForMaskedLM.from_pretrained("uw-madison/nystromformer-512")` 로 모델 로드 변경
- [ ] `tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")` (Albert SP 자동 선택)
- [ ] `bookcorpus-512/config.json`의 `model_max_length=512` → 510으로 정합 (또는 collation에서 510으로 truncate)
- [ ] 로컬 `.model` 체크포인트 로딩 루프 제거 (`run_validation.py:210-213`)
- [ ] `--checkpoint` 인자가 의미 없어지므로 deprecated 처리 또는 인자 제거
- [ ] config의 `vocab_size`/`embedding_dim`/`num_layers` 등은 더 이상 model 생성에 사용하지 않으므로 의존 코드 정리
- [ ] `model.eval()` + `torch.no_grad()` 로 순수 inference 보장 (현재도 그러함)

### 1.2 변경할 파일 목록
- `run_validation.py` (line 36-50, 68-72, 84, 127, 210-213)
- `bookcorpus-512/config.json` (선택: `model.max_seq_len=512` → 510, `embedding_dim=128` → 768 등 — 단 HF 모델의 config가 우선이므로 주석 처리만으로 충분)

### 1.3 의존성
- Phase 0 완료
- 인터넷 접근 또는 사전 캐시된 HF 모델 (`HF_HOME`)

### 1.4 검증 방법
- `python run_validation.py --batch_size 1 --num_batch 1` 실행 (저장 npz 옵션 없이)
- `print(model.config)` 확인 → `hidden_size=768, num_hidden_layers=12, num_landmarks=64, conv_kernel_size=65` 출력 확인
- forward 1회 통과, MLM loss 출력 (정상값 < 10 정도)
- HF cache 디렉터리(`~/.cache/huggingface/hub/models--uw-madison--nystromformer-512/`) 생성 확인

### 1.5 예상 시간
- 코드 변경: 1–2시간
- 첫 실행 (모델 다운로드 + forward): 추가 30분 (모델 약 500MB)

### 1.6 롤백 절차
- `git stash` 또는 변경한 라인을 직접 되돌림
- `bookcorpus-512/config.json`을 원본으로 복구
- 로컬 `cp-{epoch:04}.model` 체크포인트는 디스크에 보존되어 있으므로 코드만 되돌리면 즉시 복구 가능

### 1.7 주의 사항
- Albert tokenizer는 RoBERTa의 BPE와 다르므로, **현재 `dataset.py`의 partition pickle은 Phase 1 환경에서 사용할 수 없다.**
  partition pickle은 RoBERTa tokenizer로 미리 토크나이즈된 token ID 리스트이므로, Albert vocab과 호환되지 않음.
  → Phase 1에서는 raw text 한 줄 또는 `transformers.Dataset` 형태의 작은 fixture로 평가 입력을 단순화하는 것이 합리적.

---

## Phase 2 — Baseline RoBERTa → BERT-base-uncased 교체

### 2.1 작업 항목 체크리스트
- [ ] `from transformers import BertForMaskedLM, BertTokenizerFast, BertConfig` import 추가
- [ ] `from transformers.models.bert.modeling_bert import BertSelfAttention` (hook target)
- [ ] `tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=...)`
- [ ] `model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)`
- [ ] hook target 식별: `isinstance(m, RobertaSelfAttention)` → `isinstance(m, BertSelfAttention)`
- [ ] 모듈 경로: `model.roberta.encoder.layer[i]` → `model.bert.encoder.layer[i]`
- [ ] `--checkpoint` vestigial 인자 제거 또는 명시적 무시
- [ ] 파일명 변경: `run_bert_validation.py`는 그대로 의미가 맞아짐 (실제로 BERT 사용). 또는 명시성 위해 `run_baseline_validation.py`로 일반화 (선택)

### 2.2 변경할 파일 목록
- `run_bert_validation.py` (line 14, 15, 51, 61, 73, 130-172, 183-192)
- `bert_attention_pipeline.py` (변경 없음 예상 — full O(n²) attention path 그대로)
- `bert_normalization_pipeline.py` (Phase 3에서 LayerNorm 정확화 시 변경)

### 2.3 의존성
- Phase 0 완료
- Phase 1과 병렬 가능

### 2.4 검증 방법
- `python run_bert_validation.py --batch_size 1 --num_batch 1` 실행
- `model.config` 출력에서 `BertConfig` 확인
- Hook 부착 확인: `target_modules`에 12개 layer × 1 = 12개 BertSelfAttention 인스턴스
- Forward 1회 통과, MLM loss 정상값 확인

### 2.5 예상 시간
- 코드 변경: 1시간
- 첫 실행 (모델 다운로드 + forward): 추가 20분 (BERT-base 약 440MB)

### 2.6 롤백 절차
- 변경한 라인을 RoBERTa로 되돌림. `git stash`/checkout으로 즉시 복구.
- HF cache의 BERT 가중치는 디스크에 남아 있어 무해.

### 2.7 주의 사항
- BERT는 NSP 객체로도 학습됨 → `BertForMaskedLM`은 MLM head만 사용하므로 영향 없음 (확인용 메모).
- Tokenizer 변경으로 데이터 collation에 [CLS]/[SEP]가 자동 삽입됨. RoBERTa의 `<s>`/`</s>`와 다름. 입력 sequence 길이 계산 시 +2 토큰 차이 인지.

---

## Phase 3 — npz 추출 코드 재작성

### 3.1 작업 항목 체크리스트

#### 3.1.1 Nystrom 측 (`run_validation.py`, `attention_nystrom.py` 대체)
- [ ] `attention_nystrom.py`의 capture/save 로직을 분리 → 새 파일 `hf_nystrom_capture.py` (가칭) 신설
- [ ] `NystromformerSelfAttention.forward`를 monkey-patch 또는 fork 하여 closure 내부 변수를 외부 dict에 push
  - `transformers/models/nystromformer/modeling_nystromformer.py`의 `NystromformerSelfAttention.forward` 본문 복사
  - 각 중간 변수 (`q_landmarks`, `k_landmarks`, `kernel_1/2/3`, `iterative_inv` 결과, `attention_scores`) 직후 `_capture[key] = value.detach().cpu().numpy()` 삽입
  - patched forward로 method 교체
- [ ] 저장 키 이름은 **현재와 동일하게 유지** (`Q_proc`, `K_proc`, `V`, `mask`, `Q_landmarks`, `K_landmarks`, `kernel_*_scores`, `kernel_1/2/3`, `inv_kernel_2`, `temp`, `mid`, `X`, `conv_term`, `X_final`)
  - shape만 자동으로 768/12/510으로 따라감
- [ ] Norm/FFN 캡처: `nn.LayerNorm`/`nn.Linear`/GELU에 forward hook 부착
  - 키 이름은 LayerNorm 정확 표현으로 변경 (`ln_input`, `mean`, `var`, `ln_weight`, `ln_bias`, `ln_output`)
  - `gelu_output` 키는 유지

#### 3.1.2 BERT 측 (`run_bert_validation.py`)
- [ ] Norm hook을 “LayerNorm 위장 RMSNorm 키”에서 진짜 LayerNorm 키로 교체 (현재 `run_bert_validation.py:130-137`)
- [ ] 양 모델 공통 키 스키마 v2 정의:
  ```
  ln_input, mean, var, ln_weight, ln_bias, ln_output
  linear_weight, linear_bias, linear_output
  gelu_output
  linear_weight_ff2, linear_bias_ff2, linear_output_ff2
  ln_input_2, mean_2, var_2, ln_weight_2, ln_bias_2, ln_output_2
  ```

#### 3.1.3 Witness pipeline 변경
- [ ] `normalization_pipeline.py`의 `process_normalization_debug_npz`에서 LayerNorm 식으로 양자화:
  - 현재: `inv_rms = 1/√(ms+eps)` 한 값만 lookup
  - 목표: `mean` quantize, `inv_std = 1/√(var+eps)` lookup, 그리고 `out = w · (x − μ) · inv_std + b` 식으로 reconstruct
  - 새 witness 키: `Q_mean_ln*`, `Q_var_ln*`, `inv_std_total_ln*`, `Q_gamma_ln*`, `Q_beta_ln*`, `M_norm_ln*`
- [ ] `bert_normalization_pipeline.py`도 동일 변경
- [ ] `attention_pipeline.py`/`bert_attention_pipeline.py`는 키 이름 변경 없음, shape만 자동 따라감 (Q_q, R_matmul_*, P_*, limbs 등)

### 3.2 변경할 파일 목록
- `run_validation.py` (norm/ffn capture 부분 line 172-201)
- `attention_nystrom.py` 또는 신설 `hf_nystrom_capture.py`
- `run_bert_validation.py` (line 130-172)
- `normalization_pipeline.py` (전면 재작성)
- `bert_normalization_pipeline.py` (전면 재작성)
- `attention_pipeline.py` (변경 거의 없음, shape 검증만)
- `bert_attention_pipeline.py` (변경 거의 없음)

### 3.3 의존성
- Phase 1, 2 완료
- Phase 4 (lookup table 재생성)와 동시 진행 가능 — 단 Phase 4의 키 이름 컨벤션이 결정되어야 양 측이 일치함

### 3.4 검증 방법
- 양 모델로 npz 생성: `python run_validation.py --save_attn_npz ./outputs` / `python run_bert_validation.py --save_attn_npz ./outputs_bert`
- `inspect_npz.py` 또는 직접 `np.load` 로 키/shape 확인
- 핵심: **양자화 후 dequant 결과가 원본 forward 출력과 충분히 가까운지** (예: relative L2 < 1%) 검증하는 sanity check 함수 추가
- LayerNorm 식이 회로 외부에서 reconstruct되는지: `assert np.allclose(ln_output, ln_weight * (ln_input - mean) / sqrt(var + eps) + ln_bias)`

### 3.5 예상 시간
- monkey-patch + 테스트: 1–2일
- LayerNorm pipeline 재작성: 1일
- 양 모델 양자화 sanity check + 디버깅: 1일
- **총 3–4일**

### 3.6 롤백 절차
- 신설 파일 (`hf_nystrom_capture.py`) 삭제
- 변경된 파일들을 git checkout
- 기존 RMSNorm 위장 캡처 코드는 RoBERTa 환경에서 (Phase 2 롤백과 함께) 복구

---

## Phase 4 — Rust 회로 lookup table 재생성

### 4.1 작업 항목 체크리스트
- [ ] `gelu_table.py`의 함수 분기 추가:
  - `gelu_table_new.npz` (Nystromformer용, `gelu_new` = tanh 기반)
  - `gelu_table_erf.npz` (BERT용, `gelu` = erf 기반)
- [ ] `rms_table.py` → `layernorm_table.py`로 재설계:
  - 입력: 양자화된 분산 `q_var`
  - 출력: `1/√(q_var · scale + eps)` (eps 모델별: 1e-5 vs 1e-12)
- [ ] `softmax_table.py` 변경 없음 — 검증만
- [ ] 새 lookup 표의 binary 출력 (`*.npz` 또는 Rust가 직접 읽는 포맷)을 `circuits/src/util/load.rs`가 읽을 수 있도록 경로 정합

### 4.2 변경할 파일 목록
- `gelu_table.py`
- `rms_table.py` (또는 신설 `layernorm_table.py`)
- (선택) `softmax_table.py` (변경 없음 확인만)

### 4.3 의존성
- Phase 3 완료 (witness 키 컨벤션 확정)
- Phase 5와 동시에 가능, 단 Phase 5가 새 표 형식을 읽으려면 Phase 4가 먼저 완료되어야 함

### 4.4 검증 방법
- `python gelu_table.py` 실행 → 출력 표를 numpy로 로드하여 `gelu_new(x)` 참조 구현과 비교 (max abs error < 1e-3 정도)
- LayerNorm 표: `q_var=0~2^16` 전체 인덱스에 대해 `1/√(real_var+eps) * 2^frac` 값이 정확히 일치하는지 검증
- Rust 측에서 `load_lookup_table()` 실행 후 dump 값 확인

### 4.5 예상 시간
- GELU 표 재생성: 0.5일
- LayerNorm 표 재설계 + 양자화 분포 결정: 1일
- Rust 측 호환 확인: 0.5일
- **총 1.5–2일**

### 4.6 롤백 절차
- 기존 `gelu_table.py`/`rms_table.py`를 git checkout으로 복구
- 새 표 binary 파일 삭제

---

## Phase 5 — Rust 가젯 수정 *(작업 폴더 외부, 추정 비중 큼)*

### 5.1 작업 항목 체크리스트
- [ ] `circuits/src/util/mlp_lookup.rs`:
  - [ ] `run_lasso_for_rmsnorm` (line 27)을 `run_lasso_for_layernorm`으로 교체 또는 분기
    - 새 입력: `(q_mean, q_var, m_ln_idx)` 튜플
    - lookup table: `layernorm_table` (1/√(var+eps))
  - [ ] `run_lasso_for_gelu` (line 64) 인덱스 매핑 재계산
    - sigmoid GELU에서 `q_in_sim` → table index 변환식이 변경됨
    - tanh 기반 `gelu_new`는 `0.5x(1+tanh(0.7978(x+0.044715x³)))`이므로 입력 도메인의 양자화 격자 다시 설계
  - [ ] `run_mlp_remainders` (line 110) 7개 remainder 행렬 키 이름 변경 *(추정: `R_mlp_norm1_shift` → `R_mlp_ln1_shift` 등)*
- [ ] `circuits/src/util/nystrom_attn_lookup.rs`:
  - [ ] conv 차원 상수 `KERNEL=3` → `KERNEL=65` *(추정: 회로 측에 conv kernel size 상수가 있다면)*
  - [ ] dim 128/12/2 → 768/12/12 차원 상수 갱신
  - [ ] Schulz iter 횟수가 HF 구현과 일치하는지 확인 (HF는 보통 6회)
- [ ] `circuits/src/util/roberta_attn_lookup.rs`:
  - [ ] 차원 상수만 갱신. attention 형태(full O(n²))는 동일
- [ ] `circuits/src/circuits/freivalds_matmul/nystrom_attn_circuit.rs`:
  - [ ] conv gadget 차원 갱신 (kernel 65)
  - [ ] norm gadget을 RMSNorm → LayerNorm 식으로 교체
- [ ] `circuits/src/circuits/freivalds_matmul/nystrom_mlp_circuit.rs`:
  - [ ] norm gadget RMSNorm → LayerNorm
  - [ ] activation gadget sigmoid → tanh 기반 gelu_new
- [ ] `circuits/src/circuits/freivalds_matmul/roberta_*_circuit.rs`:
  - [ ] BERT용으로 활용. norm gadget RMSNorm → LayerNorm, activation → erf 기반 gelu
  - [ ] 파일/struct 이름이 `RoBERTa*`인 것은 그대로 두되, 주석 또는 type alias로 “BERT-base용”임 명시
- [ ] `circuits/src/circuits/naive_matmul/*`: 동일 변경 적용
- [ ] `circuits/src/util/load.rs`: 새 npz 키 (`Q_mean_ln*`, `Q_var_ln*`, `inv_std_total_ln*`, `Q_beta_ln*`)를 읽도록 추가

### 5.2 변경할 파일 목록 (추정 비중 큼)
- `circuits/src/util/mlp_lookup.rs`
- `circuits/src/util/nystrom_attn_lookup.rs`
- `circuits/src/util/roberta_attn_lookup.rs`
- `circuits/src/util/load.rs` *(추정)*
- `circuits/src/circuits/freivalds_matmul/{nystrom,roberta}_{attn,mlp}_circuit.rs`
- `circuits/src/circuits/naive_matmul/{nystrom,roberta}_{attn,mlp}_circuit.rs`
- 새 RMSNorm/LayerNorm gadget이 별도 모듈에 있다면 (`circuits/src/gadgets/`) 그쪽도 — *(추정)*

### 5.3 의존성
- Phase 4 완료 (새 lookup 표 형식 확정)
- Phase 3 완료 (witness 키 이름 확정)

### 5.4 검증 방법
- `cargo build --release` 통과
- `cargo test` (만약 unit test가 있다면)
- 작은 입력으로 회로 prove + verify 사이클 1회 통과
- benchmark 비교: 변경 전/후의 constraint 수 출력 비교 — Phase 6에서 본격 측정

### 5.5 예상 시간
- 매우 가변적. 회로 코드 친숙도에 따라:
  - lookup 분기 추가: 1–2일
  - LayerNorm gadget 작성/검증: 2–3일
  - conv kernel 65 차원 변경: 0.5일
  - 모든 회로 변형 (8종) 재컴파일 + 테스트: 1–2일
- **총 1주–10일** *(추정)*

### 5.6 롤백 절차
- Rust 코드는 `git stash` 또는 별도 브랜치(`feat/circuit-ZEN-large-data` → 새 브랜치 분기)로 격리하여 변경
- lookup 표 binary 파일은 backup 후 새로 생성

### 5.7 주의 사항 (전체 *(추정)* 비중 큼)
- 본 단계 이전에 회로 코드를 직접 읽어 다음을 확인해야 함:
  1. RMSNorm gadget의 정확한 위치 (별도 모듈인지, mlp_circuit 내부 inline인지)
  2. conv kernel size가 회로 어디서 hardcoded인지
  3. lookup 표의 wire 형식 (frac_bits, limb_bits 상수가 어디 정의되는지)
  4. CP-Link로 묶이는 commitment 목록이 어떻게 변하는지
- 위 정보 없이 작업 시 회로 검증 실패 가능 → Phase 5 시작 전 “회로 코드 사전 읽기” mini-task 1회 추천

---

## Phase 6 — End-to-End 검증

### 6.1 작업 항목 체크리스트
- [ ] **Smoke test**: 짧은 입력 한 줄로 npz 추출 → witness 변환 → Rust prove → Rust verify 1회 사이클 통과
- [ ] **차원 sweep**: seq_len ∈ {128, 256, 512}, layer 1–12 sweep으로 prover/verifier time, constraint 수, proof size 측정
- [ ] **모델 비교**: HF Nystromformer vs HF BERT 동일 입력에 대해 회로 비용 표 작성 (§7 사용)
- [ ] **Sanity check**: 회로 출력 vs Python forward 출력의 numerical agreement (양자화 오차 < 정해진 threshold)
- [ ] §7 표 자동 생성 스크립트 (선택)

### 6.2 변경할 파일 목록
- 새 e2e 테스트 스크립트 (예: `tests/e2e_nystrom_vs_bert.py`)
- 결과 로깅용 디렉터리 (예: `outputs/benchmarks/`)

### 6.3 의존성
- Phase 5 완료

### 6.4 검증 방법
- prove + verify 사이클 통과 = OK
- Constraint 수가 *(추정)* dim 128→768로 인해 36×, layer 2→12로 인해 6× 증가했는지 확인
- §7 본문에 들어갈 수치가 안정적으로 재현되는지 (same seed → same numbers)

### 6.5 예상 시간
- E2E 디버깅: 2–3일
- benchmark sweep + 결과 정리: 2일
- §7 표 작성: 1일
- **총 5–6일**

### 6.6 롤백 절차
- 본 phase는 “측정/검증”이므로 코드 변경이 거의 없음. 결과 디렉터리만 정리.
- 만약 회로 prove 실패가 잦으면 Phase 5로 회귀.

---

## 전체 예상 일정

| Phase | 예상 기간 |
|-------|----------|
| Phase 1 | 0.5일 |
| Phase 2 | 0.5일 |
| Phase 3 | 3–4일 |
| Phase 4 | 1.5–2일 |
| Phase 5 | 1주–10일 *(추정)* |
| Phase 6 | 5–6일 |
| **합계** | **약 3–4주** |

---

## 작업 폴더 외부에서 추가 확인이 필요한 정보

> 본 문서를 작성하면서 작업 폴더(`models/Nystromformer/code/dietformer`)만 read하였다. Phase 5 진입 전에 다음 항목을 회로 코드(`circuits/`)에서 직접 확인해야 한다. 이는 **(추정)** 표시된 항목들의 정확도를 결정한다.

### Rust 회로 측 확인 필요
1. **RMSNorm gadget의 정확한 위치**:
   - `circuits/src/util/mlp_lookup.rs::run_lasso_for_rmsnorm`만으로 끝인지
   - 또는 `circuits/src/circuits/freivalds_matmul/nystrom_mlp_circuit.rs` 내부에 별도 RMSNorm 검증 R1CS constraint이 있는지
   - 또는 `circuits/src/gadgets/` 하위에 RMSNorm gadget이 분리되어 있는지

2. **Conv kernel size 상수**:
   - `circuits/src/circuits/freivalds_matmul/nystrom_attn_circuit.rs`에서 `KERNEL_SIZE: usize = 3`이 const로 박혀 있는지
   - 또는 동적으로 npz에서 받는지
   - 회로가 conv를 검증하는지, 아니면 conv 결과를 외부에서 받아 inner product만 하는지 (논문 contribution과 정렬)

3. **frac_bits / limb_bits / k_bits 상수 위치**:
   - 현재 `attention_pipeline.py` 측에서 `k_bits=22`로 전달
   - Rust 측 어디에 `const K_BITS: usize = 22;`가 있는지

4. **CP-Link commitment 목록**:
   - `circuits/src/cc_snark/`에서 CP-Link로 묶이는 commitment들이 어떻게 정의되는지
   - 새 LayerNorm 키 (mean, var, beta 등)가 추가되면 commitment 수가 변하는지

5. **`circuits/src/util/load.rs`의 npz 키 hardcoding 여부**:
   - 키 이름이 함수 본문에 string literal로 박혀 있는지
   - generic하게 받는지

### Python 측 추가 확인 필요
6. **`bert-base-uncased`의 정확한 `hidden_act`**: HF config.json 확인 (`gelu` vs `gelu_pytorch_tanh`).
7. **HF Nystromformer의 `iterative_inv` 정확한 구현**: `transformers.models.nystromformer.modeling_nystromformer.NystromformerSelfAttention.iterative_inv`의 step 수 (Schulz 반복 횟수)와 본 회로의 inv 단계 수가 일치하는지.
8. **현재 `bookcorpus-512/config.json`의 모순**: dim=128/layer=2 (디스크) vs 사용자 메시지의 dim=768/layer=12 — 어느 쪽이 맞는지.

### 데이터 측 추가 확인 필요
9. **partition pickle ~822개의 위치**: 현 디스크에서 발견되지 않음. 다른 머신/환경의 것이라면 본 마이그레이션은 무관하지만, 실제 사용 의도가 있다면 `preprocess_data_512.py`를 어떤 코퍼스로 돌릴지 결정 필요.
10. **§7 측정용 입력 데이터**: 사전학습이 끝난 모델을 사용하므로 학습 데이터가 더 이상 필요 없지만, npz 추출용 *평가 입력*은 1개 batch라도 필요. 어떤 텍스트 sample을 사용할지 (예: GLUE dev의 첫 batch?).

---

## Phase 0 종료 — 다음 단계

본 문서 4개를 검토하시고, 이상이 없으면 **Phase 1 (Nystrom 모델 교체)**로 진행할지 알려주세요. Phase 1 시작 전에 위 “추가 확인 필요” 항목 중 다음 두 가지는 먼저 확정되어야 합니다:
- 항목 8: 현재 config의 dim=128/layer=2가 의도된 값인지
- 항목 10: §7 평가 입력 데이터로 무엇을 사용할지
