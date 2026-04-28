# Phase 0 — 컴포넌트별 변경사항

> **현재(Self-trained Nystrom + RoBERTa-base)** → **목표(`uw-madison/nystromformer-512` + `bert-base-uncased`)**
> *(추정)* 표기는 디스크에서 직접 확인하지 못한 항목.

---

## 1. 컴포넌트 비교표

### 1.1 모델 식별 / 가중치 출처
| 컴포넌트 | 현재 | 목표 |
|----------|------|------|
| Nystrom 모델 클래스 | `model.py:208` `Model` (custom) | `transformers.NystromformerForMaskedLM` |
| Nystrom 가중치 | `bookcorpus-512/model/cp-{epoch:04}.model` (직접 학습) | HF Hub `uw-madison/nystromformer-512` |
| Baseline 모델 클래스 | `transformers.RobertaForMaskedLM` (`run_bert_validation.py:14, 61`) | `transformers.BertForMaskedLM` |
| Baseline 가중치 | HF Hub `roberta-base` | HF Hub `bert-base-uncased` |
| 학습 데이터 | BookCorpus 단일 partition *(추정 < 1GB)* | (사전학습 끝, 추가 학습 없음) |

### 1.2 Normalization
| 항목 | 현재 (Nystrom) | 현재 (RoBERTa baseline) | 목표 (HF Nystrom) | 목표 (BERT) |
|------|---------------|--------------------------|---------------------|---------------|
| 클래스 | `Approx_RMSNorm` (`model.py:32-57`) | `nn.LayerNorm` (위장 캡처: `run_bert_validation.py:130-137`) | `nn.LayerNorm` | `nn.LayerNorm` |
| 식 | `out = w · x / √(mean(x²)+ε)` | `out = w · (x−μ)/√(var+ε) + b` | `out = w · (x−μ)/√(var+ε) + b` | `out = w · (x−μ)/√(var+ε) + b` |
| `eps` | 1e-6 (`rms_table.py:12`) | 1e-5 (HF default) | **1e-5** (`config.json: layer_norm_eps`) | **1e-12** (HF default) |
| bias | 없음 | 있음 | 있음 | 있음 |
| 회로 lookup table | `rms_table.py` = `1/√(ms+ε)` | (현재 코드는 같은 lookup 재사용 — *수치 불일치*) | LayerNorm용 `(μ, σ²)` 분기 lookup으로 재설계 필요 | 같음 |

### 1.3 Activation
| 항목 | 현재 (Nystrom) | 현재 (RoBERTa) | 목표 (HF Nystrom) | 목표 (BERT) |
|------|---------------|---------------|---------------------|---------------|
| 클래스 | `Approx_GeLU` (`model.py:8-30`) | `RobertaIntermediate.intermediate_act_fn` (HF `gelu`) | HF `gelu_new` (config: `hidden_act="gelu_new"`) | HF `gelu` (정확히는 `bert-base-uncased`도 `gelu`) |
| 식 | `x · σ(1.702 x)` | `0.5 x (1+erf(x/√2))` | `0.5 x (1+tanh(√(2/π)·(x+0.044715 x³)))` | `0.5 x (1+erf(x/√2))` |
| 회로 lookup table | `gelu_table.py` = `σ(1.702x)` (`gelu_table.py:14`) | (현재 코드는 같은 lookup 재사용 — *수치 불일치*) | `gelu_new` (tanh 기반)으로 재생성 필요 | `gelu` (erf 기반)으로 재생성 필요 |

### 1.4 Attention 세부
| 항목 | 현재 (Nystrom) | 현재 (RoBERTa) | 목표 (HF Nystrom) | 목표 (BERT) |
|------|---------------|---------------|---------------------|---------------|
| 클래스 | `attention_nystrom.py:8 NystromAttention` | `RobertaSelfAttention` (`run_bert_validation.py:15`) | `transformers.models.nystromformer.modeling_nystromformer.NystromformerSelfAttention` | `transformers.models.bert.modeling_bert.BertSelfAttention` |
| Attention 형태 | low-rank Nyström (3 kernel) | full O(n²) | low-rank Nyström (3 kernel) | full O(n²) |
| `num_landmarks` | 64 | ─ | 64 | ─ |
| `segment_means_seq_len` | (별도 변수 없음, num_landmarks와 통합) | ─ | 64 | ─ |
| `conv_kernel_size` | **3** (depthwise 2D, 3×1) | ─ | **65** (depthwise 2D, 65×1) | ─ |
| Schulz iter (`iterative_inv`) | 사용자 구현 (반복 횟수 명시 필요) | ─ | HF 구현 (반복 횟수 fixed in source) | ─ |
| `inv_coeff_init_option` | (해당 없음) | ─ | `False` | ─ |
| 중간값 노출 (npz hook용) | `Approx_*` 클래스의 `_last_*` 속성으로 직접 노출 | hook으로 일부만 (Q/K/V/X) | closure 내부, monkey-patch 또는 forward fork 필요 | hook으로 일부만 (현 RoBERTa와 동일 전략 사용 가능) |

### 1.5 Tokenizer / Vocab
| 항목 | 현재 (Nystrom) | 현재 (RoBERTa) | 목표 (HF Nystrom) | 목표 (BERT) |
|------|---------------|---------------|---------------------|---------------|
| Tokenizer 클래스 | `RobertaTokenizerFast` | `RobertaTokenizerFast` | `AlbertTokenizer` (SentencePiece) | `BertTokenizer` (WordPiece) |
| `from_pretrained` 인자 | `"/code/roberta-base"` | `"roberta-base"` | `"uw-madison/nystromformer-512"` | `"bert-base-uncased"` |
| `vocab_size` | 50265 | 50265 | **30000** | **30522** |
| Special tokens | `<s>`, `</s>`, `<mask>`, `<pad>` | 같음 | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `<unk>` | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]` |
| Lowercasing | 네 (RoBERTa BPE는 대소문자 구분, 코퍼스 그대로) | 같음 | (Albert default = lowercase) | uncased (lowercase) |
| 사전학습 데이터 | BookCorpus (subset) | BookCorpus + Wiki + CC-News + OpenWebText + Stories (160GB) | **BookCorpus + Wiki** (16GB) | **BookCorpus + Wiki** (16GB) |
| → *코퍼스 일치성* | ─ | ─ | ✅ 두 목표 모델 일치 | ✅ |

### 1.6 Sequence length
| 항목 | 현재 | 목표 (HF Nystrom) | 목표 (BERT) |
|------|------|---------------------|---------------|
| `max_seq_len` (회로 입력) | 512 | **510** (BOS+EOS 제외) | 512 |
| `max_position_embeddings` | (model.py 측 hyperparameter) | 510 | 512 |
| 처리 | `model_max_length=512`로 truncate | 510으로 truncate / 또는 입력에 special token 포함 시 508 | 512 그대로 |

### 1.7 NPZ 키 이름과 shape
모든 shape는 `[B, H, S, D_h]` 또는 `[B, H, S1, S2]` 형식. B=batch, H=num_heads, S=seq_len, D_h=head_dim, M=num_landmarks.

#### Attention 디버그 npz
| 키 | 현재 shape | 목표 shape (Nystrom HF) | 목표 shape (BERT) | 변경? |
|----|-----------|--------------------------|---------------------|------|
| `Q_proc`, `K_proc` | `[1, 2, 512, 32]` | `[1, 12, 510, 64]` | `[1, 12, 512, 64]` | shape 만 변경 |
| `V` | `[1, 2, 512, 32]` | `[1, 12, 510, 64]` | `[1, 12, 512, 64]` | shape 만 변경 |
| `mask` | `[1, 512]` | `[1, 510]` | `[1, 512]` | shape |
| `Q_landmarks`, `K_landmarks` | `[1, 2, 64, 32]` | `[1, 12, 64, 64]` | (해당 없음) | shape / BERT는 키 자체 부재 |
| `kernel_1_scores` | `[1, 2, 512, 64]` | `[1, 12, 510, 64]` | (해당 없음) | shape |
| `kernel_2_scores` | `[1, 2, 64, 64]` | `[1, 12, 64, 64]` | (해당 없음) | shape |
| `kernel_3_scores` | `[1, 2, 64, 512]` | `[1, 12, 64, 510]` | (해당 없음) | shape |
| `kernel_1/2/3` (post-softmax) | 같음 | 같음 | (해당 없음) | shape |
| `inv_kernel_2` | `[1, 2, 64, 64]` | `[1, 12, 64, 64]` | (해당 없음) | shape |
| `temp` | `[1, 2, 64, 32]` | `[1, 12, 64, 64]` | (해당 없음) | shape |
| `mid` | `[1, 2, 512, 64]` | `[1, 12, 510, 64]` | (해당 없음) | shape |
| `X` | `[1, 2, 512, 32]` | `[1, 12, 510, 64]` | `[1, 12, 512, 64]` | shape |
| `conv_term`, `X_final` | `[1, 2, 512, 32]` (kernel=3) | `[1, 12, 510, 64]` (kernel=65) | (해당 없음) | shape + conv kernel size |
| `attention_scores` (BERT 추가) | (해당 없음) | (해당 없음) | `[1, 12, 512, 512]` | **새 키** (full attn matrix) |

→ **키 이름은 거의 그대로 유지 가능**, shape만 바뀜. 단 BERT path에는 Nystrom 전용 키가 없고, 대신 단일 `attention_scores`/`P_attn`이 필요 (이미 `bert_attention_pipeline.py`에 일부 반영).

#### Norm/FFN 디버그 npz
현재 `rms_*` 키가 LayerNorm 위에 위장 저장됨. **목표**:
| 현재 키 | 목표 키 (LayerNorm 정확 표현) | 변경 |
|---------|------------------------------|------|
| `rms_input` | `ln_input` | **이름 변경** |
| `ms` | `mean` + `var` (또는 `inv_std`) | **두 값으로 분리** |
| `rms_weight` | `ln_weight` | 이름 변경 |
| (없음) | `ln_bias` | **새 키** (LayerNorm은 bias 있음) |
| `rms_output` | `ln_output` | 이름 변경 |
| 동일 패턴이 `rms_*_rms2`에도 적용 | `ln_*_2` | |

→ **회로 측 lookup이 `1/√(ms+ε)` 한 값만 받으면 LayerNorm을 못 표현한다.** 키 분리 + lookup 재설계가 필요.

#### Witness npz 키
- 현재 `attention_pipeline.py`/`normalization_pipeline.py`가 출력하는 witness 키들은 회로 측 Lasso lookup이 직접 읽음 (`mlp_lookup.rs`, `nystrom_attn_lookup.rs`).
- shape 변경(dim 128→768)은 witness 차원만 늘림 → 키 이름은 그대로 두는 것이 회로 측 호환성 유지에 가장 안전.
- normalization 키 (`rms_* → ln_*`)는 회로 측에서 분기 처리해야 하므로 새 witness 함수 + 새 lookup 함수 필요.

### 1.8 회로 lookup table
| Table 파일 | 현재 함수 | 목표 함수 (Nystrom 측) | 목표 함수 (BERT 측) | 회로 영향 |
|-----------|-----------|------------------------|---------------------|----------|
| `gelu_table.py` | `σ(1.702 x)` | `gelu_new = 0.5 x (1+tanh(…))` | `gelu = 0.5 x (1+erf(x/√2))` | 표 자체를 새로 생성. 모델별 tanh vs erf 분기 필요 |
| `rms_table.py` | `1/√(ms+ε)` (RMSNorm) | `1/√(var+ε)` + 별도 `mean` 채널 | 같음 | LayerNorm 표 재설계, eps 값도 모델별 (1e-5 vs 1e-12) |
| `softmax_table.py` | `exp(x)` 5-segment | 동일 | 동일 | **변경 없음** ✅ |

### 1.9 회로 변형 (Rust)
| 회로 | 현재 가정 | 목표 가정 | 영향 |
|------|----------|----------|------|
| `freivalds_matmul/nystrom_attn_circuit.rs` | conv k=3, RMSNorm | conv k=65, LayerNorm | conv 차원 상수, norm gadget 교체 |
| `freivalds_matmul/nystrom_mlp_circuit.rs` | RMSNorm + sigmoid GELU | LayerNorm + gelu_new | norm/gelu gadget 교체 |
| `freivalds_matmul/roberta_attn_circuit.rs` | RMSNorm + RoBERTa GELU | LayerNorm + BERT GELU | 이름은 “roberta”이지만 실질은 BERT용으로 사용 가능. norm/gelu gadget 교체 |
| `freivalds_matmul/roberta_mlp_circuit.rs` | 같음 | 같음 | 같음 |
| `naive_matmul/*` | 같음 | 같음 | 동일 변경 적용 |
| `cc_snark` (CP-Link) | 변경 없음 | 변경 없음 | ✅ |
| `gadgets` (Bit decomposition 등) | 변경 없음 | 변경 없음 | ✅ |

---

## 2. 변경이 영향을 주는 파일 목록

### 2.1 Python 측 (`models/Nystromformer/code/dietformer/`)

| 파일:라인 | 현재 동작 | 목표 변경 |
|-----------|-----------|----------|
| `model.py:8-30` (`Approx_GeLU`) | sigmoid 1.702 GELU | (Phase 1+) HF 모델 사용 시 미사용. 단 회로 친화 swap을 쓰려면 retain |
| `model.py:32-57` (`Approx_RMSNorm`) | RMSNorm | 같음 (HF로 교체 시 미사용) |
| `model.py:59-76` (`Approx_Linear`) | Linear + 캡처 | 같음 |
| `model.py:117-171` (`Attention`) | dispatch to NystromAttention | 미사용 |
| `model.py:173-206` (`Transformer`) | RMSNorm+Linear+GELU+Linear+RMSNorm | 미사용 |
| `model.py:208-237` (`Model`) | Embeddings + N×Transformer | 미사용 |
| `model_wrapper.py:25-33` (`ModelForMaskedLM`) | MLM head | 미사용 |
| `attention_nystrom.py:8-162` (`NystromAttention`) | Nyström attention + 캡처 | 미사용. 또는 HF forward fork 시 참조용 |
| `attention_nystrom.py:122-136` (npz 저장) | `np.savez_compressed` | (Phase 3) 새 hook 함수에서 같은 키 이름으로 저장 |
| `run_validation.py:25` | `--save_attn_npz` argparse | 유지 |
| `run_validation.py:36-50` | config 경로 + checkpoint_dir 구성 | (Phase 1) 제거 또는 무력화 |
| `run_validation.py:68-72` | RoBERTa tokenizer + vocab_size 갱신 | Albert tokenizer + Nystromformer config 사용 |
| `run_validation.py:84` | `ModelForMaskedLM(model_config)` | `NystromformerForMaskedLM.from_pretrained("uw-madison/nystromformer-512")` |
| `run_validation.py:127` | `isinstance(m, NystromAttention)` 모듈 식별 | `isinstance(m, NystromformerSelfAttention)` |
| `run_validation.py:172-201` (norm/ffn 캡처) | `Approx_*` 캡처 속성 의존 | (Phase 3) HF `nn.LayerNorm`/`nn.Linear`/GELU에 forward hook + 키 이름 변경 |
| `run_validation.py:210-213` | 로컬 .model 체크포인트 로딩 | 제거 |
| `run_bert_validation.py:14, 51, 61` | RoBERTa 모델/토크나이저 로드 | (Phase 2) `BertForMaskedLM.from_pretrained("bert-base-uncased")` + `BertTokenizerFast.from_pretrained("bert-base-uncased")` |
| `run_bert_validation.py:15, 73` | `RobertaSelfAttention` import & 필터 | `BertSelfAttention` |
| `run_bert_validation.py:130-172` (norm hook) | LayerNorm을 RMSNorm 키로 위장 | (Phase 3) 정확한 LayerNorm 키로 변경 |
| `run_bert_validation.py:183-192` (모듈 경로) | `model.roberta.encoder.layer[i]` | `model.bert.encoder.layer[i]` |
| `attention_pipeline.py:14` (`process_attn_debug_npz`) | k_bits=22, 양자화 | shape 자동 따라감. 내부 contiguous reshape 검증 필요 |
| `normalization_pipeline.py:7` (`process_normalization_debug_npz`) | RMSNorm 가정 | (Phase 3) LayerNorm 가정으로 분기 또는 새 함수 추가 |
| `bert_normalization_pipeline.py:10` | RMSNorm 가정 (위장) | LayerNorm 식으로 재작성 |
| `gelu_table.py:14` | `σ(1.702 x)` | (Phase 4) `gelu_new`/`gelu` 별도 표 생성 |
| `rms_table.py:12` | `1/√(ms+ε)` | (Phase 4) LayerNorm용 표 — `(μ, σ²)` 두 채널 또는 `1/√(var+ε)` 분리 |
| `softmax_table.py` | exp 5-segment | **변경 없음** |
| `dataset.py:13-79` | `CorpusDataset` partition 로드 | (선택) 평가 입력 파이프라인 단순화 가능 (사전학습 추가 학습 불필요하므로) |

### 2.2 Rust 측 (`circuits/src/`)

| 파일 | 현재 | 목표 |
|------|------|------|
| `circuits/src/util/mlp_lookup.rs:27` (`run_lasso_for_rmsnorm`) | RMSNorm lookup 인덱스 매핑 | LayerNorm 분기 추가 (`run_lasso_for_layernorm`) |
| `circuits/src/util/mlp_lookup.rs:64` (`run_lasso_for_gelu`) | sigmoid GELU 인덱스 매핑 | gelu_new / gelu 별 인덱스 매핑 |
| `circuits/src/util/nystrom_attn_lookup.rs` | conv k=3 가정한 차원 (확인 필요) | conv k=65 차원 |
| `circuits/src/util/roberta_attn_lookup.rs` | full O(n²) | 변경 없음 (BERT도 full O(n²)) |
| `circuits/src/circuits/freivalds_matmul/nystrom_attn_circuit.rs` | conv 검증 차원, RMSNorm gadget | conv k=65, LayerNorm gadget |
| `circuits/src/circuits/freivalds_matmul/nystrom_mlp_circuit.rs` | RMSNorm + sigmoid GELU gadget | LayerNorm + gelu_new gadget |
| `circuits/src/circuits/freivalds_matmul/roberta_attn_circuit.rs` | RMSNorm 가정 (mlp 측) | LayerNorm gadget |
| `circuits/src/circuits/freivalds_matmul/roberta_mlp_circuit.rs` | 같음 | 같음 |
| `circuits/src/circuits/naive_matmul/*` | 같음 | 같음 변경 |
| `circuits/src/util/load.rs` (NPZ I/O) | npz 키 이름 hardcoded? *(추정 — 확인 필요)* | 키 이름 추가 또는 분기 |

> Rust 측 변경의 정확한 영향 범위는 **Phase 5에서 회로 코드 직접 검토 후 확정**한다. 본 표는 *(추정)*에 일부 의존한다.

---

## 3. “호환되는 부분”과 “이름만 같고 실질 다른 부분” 구분

| 분류 | 항목 |
|------|------|
| ✅ 그대로 호환 | softmax_table, hidden dim 768 / num_heads 12 (두 목표 모델 동일), Q/K/V projection을 `nn.Linear`로 노출, Freivalds 검증 인프라, CP-SNARK 인프라 |
| ⚠️ 이름만 같음 | `rms_*` 키 (현재 RoBERTa hook이 LayerNorm을 위장 저장 — 수치 불일치 잠재), 파일명 `bert_*` (실제는 RoBERTa) |
| ❌ 회로/lookup 재작성 | RMSNorm→LayerNorm, sigmoid GELU→gelu_new/gelu, conv kernel 3→65 |
| ❌ Python pipeline 재작성 | npz hook (HF closure 노출), tokenizer 분기, model loading 경로 |

---

## 4. 추가 확인 필요

- [ ] `circuits/src/util/load.rs`가 npz 키를 hardcoded로 읽는지, 또는 generic하게 받는지 — Phase 5 진입 전에 확인.
- [ ] `circuits/src/circuits/freivalds_matmul/nystrom_attn_circuit.rs`가 conv를 회로 안에서 검증하는지, 회로 밖으로 분리했는지 — 본 논문 contribution 기준으로는 외부 분리가 자연스러움.
- [ ] `Approx_GeLU`의 `1.702` 계수가 회로 lookup 외에도 다른 곳에서 가정되는지 (예: 양자화 scale 산출에서). `attention_pipeline.py`에서 기본 frac_bits/k_bits 산출이 GELU 모양에 의존하면 추가 변경 필요.
- [ ] `bert-base-uncased`의 정확한 `hidden_act`가 `gelu`인지 `gelu_new`인지 (config.json) — Phase 4 lookup table 생성 전 확정.
