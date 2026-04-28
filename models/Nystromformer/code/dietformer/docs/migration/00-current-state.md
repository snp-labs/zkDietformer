# Phase 0 — 현재 상태 문서화

> 작성일: 2026-04-27
> 작업 폴더: `models/Nystromformer/code/dietformer`
> 회로 폴더: `circuits/`
> 본 문서는 **읽기 전용 조사** 결과이며 코드를 수정하지 않는다.

---

## 1. 사용 중인 모델 구조

### 1.1 클래스 계층 (Python 측)

| 클래스 | 파일:라인 | 역할 |
|--------|-----------|------|
| `ModelForMaskedLM` | `model_wrapper.py:25` | MLM 헤드 + 백본 래퍼 |
| `Model` | `model.py:208` | Embeddings + N×Transformer |
| `Embeddings` | `model.py` (위쪽 정의) | token + position + (segment) 임베딩 |
| `Transformer` | `model.py:173` | Attention + 2× RMSNorm + 2× Linear + GELU |
| `Attention` | `model.py:117` | Q/K/V projection + `self.attn`로 dispatch |
| `NystromAttention` | `attention_nystrom.py:8` | 실제 Nyström attention 구현 |
| `Approx_RMSNorm` | `model.py:32-57` | RMSNorm + 중간값 캡처 (`_last_*` 속성) |
| `Approx_Linear` | `model.py:59-76` | Linear + 중간값 캡처 |
| `Approx_GeLU` | `model.py:8-30` | `x · σ(1.702 x)` 시그모이드 GELU + 캡처 |

### 1.2 forward 흐름 (1개 Transformer 블록 기준)

```
input x
  └─> Approx_RMSNorm (norm1)  → ms = x².mean, weight only, no bias
  └─> Attention (mha)
        └─> NystromAttention.forward
              ├─ Q_proc, K_proc = (Q·mask·scale, K·mask·scale)
              ├─ Q_landmarks, K_landmarks = segment-mean (num_landmarks=64)
              ├─ kernel_1_scores = Q_proc @ K_landmarks.T
              ├─ kernel_2_scores = Q_landmarks @ K_landmarks.T
              ├─ kernel_3_scores = Q_landmarks @ K_proc.T
              ├─ kernel_1/2/3   = softmax(scores, dim=-1) (row-wise)
              ├─ inv_kernel_2   = Moore-Penrose iterative inverse (Schulz)
              ├─ temp           = kernel_3 @ V
              ├─ mid            = kernel_1 @ inv_kernel_2
              ├─ X              = mid @ temp
              └─ X_final        = X + DepthwiseConv2D(V, kernel=3×1)  (conv enabled)
  └─> residual add
  └─> Approx_RMSNorm (norm2)
  └─> Approx_Linear (ff1, dim → 4·dim)
  └─> Approx_GeLU (1.702·sigmoid)
  └─> Approx_Linear (ff2, 4·dim → dim)
  └─> residual add
```

### 1.3 아키텍처 하이퍼파라미터 (`bookcorpus-512/config.json`)

| 항목 | 값 |
|------|-----|
| `vocab_size` | 50265 (RoBERTa BPE) |
| `max_seq_len` | 512 |
| `embedding_dim` / `transformer_dim` | **128** |
| `transformer_hidden_dim` | 512 |
| `num_layers` | **2** |
| `num_head` / `head_dim` | 2 / 32 |
| `attn_type` | `"nystrom"` |
| `num_landmarks` | 64 |
| `conv_kernel_size` | 3 |
| Normalization | RMSNorm (`Approx_RMSNorm`, weight only) |
| Activation | sigmoid GELU `x·σ(1.702x)` (`Approx_GeLU`) |

> 사용자 메시지 도입부의 *“embedding_dim=768, num_layers=12”*는 디스크 위 `config.json`과 다르다. 본 문서는 디스크 실측치를 기록하며, 실제 학습 의도와의 차이는 §7 “추가 확인 필요”에 적었다.

---

## 2. 데이터 전처리 파이프라인

### 2.1 토크나이저
- `RobertaTokenizerFast.from_pretrained("/code/roberta-base", model_max_length=max_seq_len)` — `run_pretrain.py:47-50`
- `model_config["vocab_size"] = len(tokenizer.get_vocab())` 으로 런타임 덮어쓰기
- 검증·baseline 측은 모두 동일하게 `"roberta-base"` (Hub) 사용 (`run_validation.py:68-72`, `run_bert_validation.py:51`)

### 2.2 raw 코퍼스 → pickle partition
- 스크립트: `data-preprocessing/preprocess_data_512.py`
- 활성화된 데이터셋: `[("bookcorpus", 1.0)]` — Wiki/RealNews/Stories 코드는 주석 처리
- 출력 partition 명: `bookcorpus-{file_idx:05}.pickle`
- `train.json` / `dev.json` 분할: 90 / 10, seed=1

### 2.3 현재 디스크 상태 (실측, 2026-04-27)
| 위치 | 파일 |
|------|------|
| `data-preprocessing/512-roberta/train.json` | 1개 항목 (`bookcorpus-00000.pickle`) |
| `data-preprocessing/512-roberta/dev.json` | (확인 필요, 작은 dev set) |
| `data-preprocessing/bookcorpus/data/` | 2개 partition pickle (0000, 0001) |
| 그 외 위치 | partition 없음 |

> 사용자가 언급한 “총 ~822개 partition”은 디스크에서 확인되지 않음 — *(추정)* `preprocess_data_512.py`가 본격 실행되지 않은 상태로 보인다.

### 2.4 데이터셋 클래스
- `CorpusDataset` — `dataset.py:13-79`, partition pickle을 batch 단위로 lazy 로드
- DataLoader 측 `DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)` 으로 MLM 마스킹 (`run_pretrain.py:55-61`)

---

## 3. 학습 파이프라인 — `run_pretrain.py`

### 3.1 흐름
1. config 로드 (`bookcorpus-512/config.json`) → `model_config`, `pretraining_config`
2. RoBERTa tokenizer 로드, `vocab_size` 갱신
3. `CorpusDataset(..., file_json="train.json")` + DataCollator + DataLoader
4. `ModelForMaskedLM(model_config)` 생성
5. **Optimizer**: `AdamW(lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)` — `run_pretrain.py:117`
6. **Scheduler**: `OneCycleLR(max_lr, pct_start=warmup, anneal=linear, epochs, steps_per_epoch=batches_per_epoch)` — `run_pretrain.py:124`
7. 학습 루프 (`run_pretrain.py:158-203`)
   - 외부: epoch 루프
   - 내부: `batches_per_epoch` 만큼 step
   - gradient accumulation: `partition_inputs()`로 GPU instance 분배
   - 보고/검증: `batches_per_report` 마다

### 3.2 체크포인트 저장 (`run_pretrain.py:205-218`)
- 패턴: `cp-{epoch:04}.cp` (학습 재개용 풀 상태) + `cp-{epoch:04}.model` (가중치만)
- `.model` 형식: `{"model_state_dict": ...}`
- `.cp` 형식: `{"model_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict", "epoch", "inst_pass"}`
- 로드 측은 `.model`만 읽음 (`run_validation.py:213` — `torch.load(...)['model_state_dict']`, `strict=False`)

### 3.3 현 시점 학습 상태 (config 실측)
- `batches_per_epoch=2`, `epoch=1`, `batch_size=1` → effective ≈ 2 step. *(추정: 디버깅용 임시값일 가능성 높음)*

---

## 4. NPZ 추출 파이프라인 — `run_validation.py`

### 4.1 트리거
```bash
python run_validation.py --checkpoint 0 --batch_size 1 --num_batch 1 --save_attn_npz ./outputs
```
- `--save_attn_npz` 인자가 None이 아닐 때만 캡처/저장 활성화 (`run_validation.py:25`)

### 4.2 두 종류의 npz가 생성됨

#### (A) Attention 모듈 디버그 npz
- 위치: `outputs/{prefix}-batch{batch_idx}-{module_name}.npz`
- 저장 코드: `attention_nystrom.py:122-136` (`np.savez_compressed(save_path, **np_save)`)
- 키 (Nystrom + landmark + conv 활성화 시):
  - 입력: `Q_proc`, `K_proc`, `V`, `mask`
  - 랜드마크: `Q_landmarks`, `K_landmarks`
  - 커널 score: `kernel_1_scores`, `kernel_2_scores`, `kernel_3_scores`
  - 커널 (post-softmax): `kernel_1`, `kernel_2`, `kernel_3`
  - pseudo-inverse: `inv_kernel_2`
  - 중간곱: `temp` (= K3·V), `mid` (= K1·K2⁻¹)
  - 출력: `X`, `conv_term`, `X_final`

#### (B) Normalization+FFN 블록 디버그 npz
- 위치: `outputs/{prefix}-batch{batch_idx}-{block_name}.norm_ff_combined.npz`
- 저장 코드: `run_validation.py:181-198`
- 키:
  - RMSNorm1: `rms_input`, `ms`, `rms_weight`, `rms_output`
  - FF1: `linear_weight`, `linear_bias`, `linear_output`
  - GELU: `gelu_output`
  - FF2: `linear_weight_ff2`, `linear_bias_ff2`, `linear_output_ff2`
  - RMSNorm2: `rms_input_rms2`, `ms_rms2`, `rms_weight_rms2`, `rms_output_rms2`

### 4.3 디버그 npz → 양자화 witness npz
이후 회로 입력으로 들어가는 것은 *witness* npz이다.
- Attention witness: `attention_pipeline.py:14` — `process_attn_debug_npz(debug_npz_path, out_npz_path, k_bits=22)`
- Norm/FFN witness: `normalization_pipeline.py:7` — `process_normalization_debug_npz(debug_npz_path, out_npz_path, k_bits=22)`
- 출력 파일명: `{...}.witness.npz` 또는 `{...}.norm.witness.npz`

#### Attention witness 키 (요약, `attention_pipeline.py:189-281`)
- 양자화 입력: `Q_q, z_q, Q_k, z_k`
- 커널별 (k1/k2/k3): `R_mean_k*`, `Q_sim_k*`, `R_matmul_k*`, `M_k*`, `M_prime_k*`, `z_sim_k*`
- softmax 단: `limbs_k*`, `exp_limbs_k*`, `r_sm_k*`, `P_k*`, `R_div_k*`
- pseudo-inverse: `Q_inv_k2`, `M_inv_all`, Schulz 반복 잔차
- 최종: `Q_sim_x`, `R_matmul_x`, `M_x`, `M_prime_x`
- 메타: `sm_tables`, `frac_bits`, `limb_bits`, `k_segments`, `seg_size`, `k_bits`, `mask`

#### Norm/FFN witness 키 (요약, `normalization_pipeline.py:179-226`)
- RMS1/RMS2: `Q_ri_rms*`, `inv_rms_total_rms*`, `Q_gamma_rms*`, `M_norm_rms*`
- LIN1/LIN2: `Q_in_diff_lin*`, `Q_w_diff_lin*`, `Q_out_sim_lin*`, `R_out_lin*`
- GELU: `sigmoid_part_gelu`, `Q_out_sim_gelu`, `R_out_gelu`
- 임베드된 테이블: `rms_table`, `gelu_table`

---

## 5. 회로 측 인터페이스 (어떤 npz를 어디서 읽는가)

회로(Rust) 코드는 **witness npz**를 직접 읽는다 — 디버그 npz는 디버깅용일 뿐 회로 입력이 아니다.

### 5.1 Rust 회로 모듈 (`circuits/src/circuits/mod.rs`, `circuits/src/util/mod.rs`)

```
circuits/src/circuits/
  ├── common.rs               # 공유 가젯 (CP-Link 등)
  ├── freivalds_matmul/
  │     ├── nystrom_attn_circuit.rs   → struct NystromAttentionCircuit<F>
  │     ├── nystrom_mlp_circuit.rs    → struct NystromMLPCircuit<F>
  │     ├── roberta_attn_circuit.rs   → struct RoBERTaAttentionCircuit<F>
  │     └── roberta_mlp_circuit.rs    → struct RoBERTaMLPCircuit<F>
  └── naive_matmul/                  # 위와 동일 구조의 비-Freivalds 변형

circuits/src/util/
  ├── load.rs                  # NPZ I/O
  ├── mlp_lookup.rs            # MLP Lasso lookup proofs
  ├── nystrom_attn_lookup.rs   # Nystrom attention Lasso proofs
  ├── roberta_attn_lookup.rs   # RoBERTa(BERT) attention Lasso proofs
  └── utils.rs
```

### 5.2 회로가 기대하는 키 — Lasso lookup 측면

#### `nystrom_attn_lookup.rs`
- `run_lasso_bundle(SoftmaxBundle)` (line 30): `bundle.limbs[seg][b][h][s][l]` 5D — softmax limb (5-segment lookup)
- `run_all_remainders_with_link(NystromAttentionCircuit)` (line 273):
  - 22-bit range remainders: `R_matmul_k1/k2/k3`, `R_sm_k1/k2/k3`, `R_inv_1/2/3/4`, `R_matmul_temp/mid/x`
  - 16-bit range remainders: `R_exp_k1/k2/k3`, `R_div_k1/k2/k3`

#### `roberta_attn_lookup.rs`
- `run_roberta_lasso_bundle(SoftmaxBundle)` (line 24): `bundle.limbs[seg][b][h][s1][s2]` — single full-matrix softmax
- `run_all_roberta_remainders_with_link(RoBERTaAttentionCircuit)` (line 60):
  - 22-bit: `matmul_attn.r_matmul`, `sm.r_sm`, `matmul_x.r_matmul` (3개)
  - 16-bit: `sm.r_exp` (5D), `sm.r_div` (4D)
  - **No pseudoinverse remainders** — Nystrom 전용

#### `mlp_lookup.rs` (양 모델 공유)
- `run_lasso_for_rmsnorm` (line 27): `bundle.q_ms_sim`, `bundle.m_rms_idx`
- `run_lasso_for_gelu` (line 64): `bundle.q_out_sim`, `bundle.z_in`, `bundle.m_gelu_idx`
- `run_mlp_remainders` (line 110): MLP 회로의 7개 remainder 행렬
  - `R_mlp_norm1_shift`, `R_mlp_norm1_out`, `R_mlp_lin1`, `R_mlp_gelu`, `R_mlp_lin2`, `R_mlp_norm2_shift`, `R_mlp_norm2_out`
  - 모두 22-bit RangeCheck Lasso (C=2, M=2¹⁶)

---

## 6. Lookup table 구성

### 6.1 GeLU — `gelu_table.py`
- TABLE_SIZE = **65536** (= 2¹⁶)
- 입력 매핑: `real_x = (indices - 32768) * (1/512)` → 약 [-64, 64]
- **함수: `sigmoid = 1.0 / (1.0 + exp(-1.702 * real_x))`** (line 14)
- 출력: `uint64`, `sigmoid * 2^frac_bits` (frac_bits 기본 8)
- 즉 **`Approx_GeLU`의 `1.702 * x` 시그모이드 근사 전용**.
  표가 LayerNorm용 `gelu_new` (tanh 기반)와 호환되지 **않음**.

### 6.2 RMSNorm 역수 — `rms_table.py`
- TABLE_SIZE = **65536**
- **함수: `inv_rms = 1.0 / sqrt(real_ms + eps)`** (line 12), `eps = 1e-6`
- 출력: `uint64`, `inv_rms * 2^frac_bits`
- 즉 RMSNorm 정규화 계수의 역수 lookup. LayerNorm은 평균 차감 + 분산이 들어가므로 다른 통계량을 lookup해야 한다 (호환 X).

### 6.3 Softmax exp — `softmax_table.py`
- K_SEGMENTS = **5**, LIMB_BITS = 16, LIMB_SIZE = 2¹⁶
- 함수: `exp(real_x)` (clip at 20.0)
- frac_bits = 32 (32-bit fixed point)
- 출력: 5개 segment 테이블 stack.
- **MLM 모델 종류와 무관하게 호환** (활성화 함수가 아니라 softmax exp이므로 BERT/RoBERTa/Nystromformer 모두 동일).

---

## 7. 사용 중인 외부 모델 — `run_bert_validation.py`

### 7.1 Baseline 모델 정확한 호출
- `run_bert_validation.py:14` — `from transformers import RobertaForMaskedLM, RobertaTokenizerFast, ...`
- `run_bert_validation.py:15` — `from transformers.models.roberta.modeling_roberta import RobertaSelfAttention`
- `run_bert_validation.py:51` — `tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", model_max_length=...)`
- **`run_bert_validation.py:61` — `model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)`**

→ 파일명은 `bert`이지만 실제로는 **`roberta-base`** (HF Hub `FacebookAI/roberta-base`).

### 7.2 가중치 출처
- 인자가 경로가 아닌 *모델 ID* (`"roberta-base"`) → HF cache 우선, 없으면 Hub 다운로드.
- 어떤 추가 학습/fine-tuning도 없음:
  - `model.eval()` (line 196), `torch.no_grad()` (line 197, 251)
  - `optimizer`, `loss.backward()`, `optimizer.step()`, `model.train()` 모두 부재
  - `--checkpoint` 인자 (line 24)는 파싱만 되고 사용되지 않는 vestigial 인자

### 7.3 Hook 부착 지점
- Attention: `isinstance(m, RobertaSelfAttention)` 필터로 모든 layer (line 73)
- Normalization/FFN (line 184-192):
  - `layer.attention.output.LayerNorm` (post-attention LN)
  - `layer.intermediate.dense` (FFN up)
  - `layer.intermediate` (GELU)
  - `layer.output.dense` (FFN down)
  - `layer.output.LayerNorm` (post-FFN LN)

### 7.4 Hook 캡처 트릭
- LayerNorm을 RMSNorm 키 형태로 위장 저장 (`run_bert_validation.py:130-137`):
  ```python
  saved_norm[layer_key] = {
      "rms_input":  x,
      "ms":         x.pow(2).mean(dim=-1, keepdim=True),  # ← LayerNorm이지만 RMS만 저장
      "rms_weight": module.weight,
      "rms_output": output,  # ← LayerNorm 결과
  }
  ```
  → **여기서 `rms_output`은 “LayerNorm 결과”인데 키 이름은 `rms_*`이다.** 회로 측은 `rms_input → ms → rms_output` 관계가 RMSNorm이라고 가정하므로 **수치적 불일치**가 잠재해 있다 (rms_output ≠ rms_weight × rms_input / sqrt(ms+eps)). 본 단계에서는 보고만 하고 수정하지 않는다.

### 7.5 Witness 후처리
- `bert_attention_pipeline.py:13` — `process_standard_attn_debug_npz(debug, out, k_bits=22)`. 입력 키: `Q_proc`, `K_proc`, `V`, `mask`, `X`. 출력: 단일 full O(n²) attention witness (`Q_sim_attn`, `R_matmul_attn`, `M_attn`, `limbs`, `exp_limbs`, `r_sm`, `P_attn`, …).
- `bert_normalization_pipeline.py:10` — `process_standard_normalization_debug_npz(...)`. RMSNorm 가정으로 처리.

---

## 8. 현재 상태 요약 한눈에

| 항목 | 값/위치 |
|------|--------|
| **Nystromformer 모델** | 직접 학습한 small (dim 128, 2 layer, 64 landmark, conv kernel 3) |
| **Nystromformer 가중치** | `bookcorpus-512/model/cp-{epoch:04}.model` |
| **Nystromformer 학습 데이터** | BookCorpus 단일 partition (1개 pickle) |
| **Nystromformer normalization** | RMSNorm (weight only) |
| **Nystromformer activation** | sigmoid GELU `x·σ(1.702x)` |
| **Nystromformer conv** | depthwise 2D, kernel size 3 |
| **Nystromformer tokenizer** | RoBERTa BPE (vocab 50265) |
| **Baseline 모델** | RoBERTa-base (HF Hub `roberta-base`, 파일명만 `bert_*`) |
| **Baseline 가중치** | HF cache (자동 다운로드) |
| **Baseline normalization** | LayerNorm — 단, hook이 RMSNorm 키로 위장 저장 |
| **Baseline activation** | GELU (HF 기본, RobertaIntermediate.intermediate_act_fn) |
| **GELU lookup table** | sigmoid `σ(1.702x)` 전용 (RoBERTa GELU와 불일치) |
| **RMSNorm lookup table** | `1/√(ms+eps)` 전용 (LayerNorm과 불일치) |
| **Softmax lookup table** | exp(x), 5-segment, 모델 무관 호환 |
| **회로 변형** | Freivalds / Naive × (Nystrom / RoBERTa) × (Attention / MLP) = 8개 |

---

## 추가 확인 필요 (Phase 0 전체 적용)

- [ ] `bookcorpus-512/config.json`의 dim=128, layer=2가 의도한 학습 크기인지, 아니면 사용자가 STEP 도입부에 적은 dim=768, layer=12가 다른 config 파일에 있는지.
- [ ] `pretraining_setting.batches_per_epoch=2, epoch=1`이 디버그용 임시값인지 (학습 step 수 약 2).
- [ ] partition pickle ~822개가 어디 있는 환경인지 — 현 디스크에서는 발견되지 않음.
- [ ] `run_bert_validation.py`의 LayerNorm-as-RMSNorm 위장 hook이 의도된 (수치 불일치 인지) 것인지, 아니면 추후 정정 필요한 버그인지.
