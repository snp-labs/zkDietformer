# Phase 0 — 목표 아키텍처

> 목표 모델 쌍: **`uw-madison/nystromformer-512`** + **`bert-base-uncased`**
> *(사용자 메시지의 “mlpen/nystromformer-512”는 HF에 존재하지 않으며, 정식 공개 ID는 `uw-madison/nystromformer-512`이다 — 저자(uw-madison)가 contribute, novice03이 업로드)*

---

## 1. 목표 아키텍처 다이어그램 (텍스트)

```
            ┌──────────────────────────────────────────┐
            │  HuggingFace Hub                         │
            │  ─ uw-madison/nystromformer-512          │
            │  ─ bert-base-uncased                     │
            └──────────────────────────────────────────┘
                          │ from_pretrained
                          ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│  NystromformerForMaskedLM     │  │  BertForMaskedLM              │
│  ─ AlbertTokenizer (30000 SP) │  │  ─ BertTokenizer (30522 WP)   │
│  ─ NystromformerEncoder       │  │  ─ BertEncoder                │
│       └─ 12× NystromformerLayer│  │       └─ 12× BertLayer       │
│              ├─ self attn      │  │              ├─ self attn    │
│              │   (Nyström, 64 │  │              │   (full O(n²)) │
│              │    landmarks,   │  │              │               │
│              │    conv k=65)   │  │              │               │
│              ├─ LayerNorm     │  │              ├─ LayerNorm    │
│              ├─ FFN(768→3072) │  │              ├─ FFN(768→3072)│
│              ├─ gelu_new      │  │              ├─ gelu          │
│              └─ LayerNorm     │  │              └─ LayerNorm    │
└─────────────┬─────────────────┘  └─────────────┬─────────────────┘
              │ forward (eval, no_grad)          │
              ▼                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  새 NPZ 추출 코드 (Phase 3에서 작성)                             │
│  ─ HF 모듈에 forward hook 부착                                   │
│  ─ Nystrom 측 중간값(Q_proc, kernel_*, inv_kernel_2, …)을        │
│    HF 모델 forward를 monkey-patch / fork 하여 노출 (단순 hook X) │
│  ─ Norm/FFN 측은 LayerNorm 키 + gelu_new로 재설계               │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│  새 양자화 witness 파이프라인 (Phase 3)                          │
│  ─ frac_bits, k_bits 동일 유지 (회로 측 전제)                    │
│  ─ rms_*  → ln_*  로 키 이름 변경 (선택) / 또는 키 호환 위해    │
│    별도 LayerNorm-아래-RMSNorm-키 위장 유지                     │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Rust 회로 (Phase 4–5에서 수정)                                  │
│  ─ gelu_table.py → gelu_new (tanh 기반) 으로 재생성              │
│  ─ rms_table.py → LayerNorm용 (μ, σ²) 분기로 재설계 또는 swap   │
│  ─ NystromAttentionCircuit conv kernel size 3 → 65 반영          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. `uw-madison/nystromformer-512` 구조

### 2.1 config (HF Hub `config.json` 실측)

| 키 | 값 |
|----|------|
| `architectures` | `["NystromformerForMaskedLM"]` |
| `model_type` | `"nystromformer"` |
| `hidden_size` | **768** |
| `num_hidden_layers` | **12** |
| `num_attention_heads` | **12** |
| `intermediate_size` | 3072 |
| `vocab_size` | **30000** |
| `max_position_embeddings` | **510** |
| `type_vocab_size` | 2 |
| `tokenizer_class` | **`AlbertTokenizer`** (SentencePiece) |
| `hidden_act` | **`gelu_new`** |
| `layer_norm_eps` | 1e-5 (LayerNorm) |
| `num_landmarks` | 64 |
| `segment_means_seq_len` | 64 |
| `conv_kernel_size` | **65** |
| `inv_coeff_init_option` | `false` |
| `attention_probs_dropout_prob` | 0.1 |
| `hidden_dropout_prob` | 0.1 |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | 0 / 2 / 1 |
| `torch_dtype` | float32 |

### 2.2 transformers 라이브러리 forward 골격
(참고: `transformers.models.nystromformer.modeling_nystromformer`)

```
NystromformerForMaskedLM
└─ NystromformerModel
    ├─ NystromformerEmbeddings
    └─ NystromformerEncoder
        └─ 12× NystromformerLayer
              ├─ NystromformerAttention
              │   └─ NystromformerSelfAttention   ← Nyström 핵심
              │         ├─ q/k/v Linear
              │         ├─ Q_landmarks = mean over segments
              │         ├─ K_landmarks = mean over segments
              │         ├─ kernel_1, kernel_2, kernel_3 = softmax(QK_lm.T / scale)
              │         ├─ iterative_inv(kernel_2)  ← Schulz iteration
              │         ├─ attention_scores ≈ k1 · k2⁻¹ · k3
              │         └─ depthwise Conv2D(V, kernel_size=65)
              ├─ NystromformerSelfOutput  (Linear + LayerNorm + residual)
              ├─ NystromformerIntermediate (Linear + gelu_new)
              └─ NystromformerOutput      (Linear + LayerNorm + residual)
```

### 2.3 사전학습 정보 (HF 모델 카드)
- 데이터셋: **BookCorpus + English Wikipedia** (약 16GB)
- 객체: Masked Language Modeling
- Sequence length: 512 (모델명 “…-512”), 단 `max_position_embeddings=510` (BOS/EOS 토큰 제외)

---

## 3. `bert-base-uncased` 구조 (비교용)

### 3.1 config (HF 표준)
| 키 | 값 |
|----|------|
| `architectures` | `["BertForMaskedLM"]` |
| `hidden_size` | **768** |
| `num_hidden_layers` | **12** |
| `num_attention_heads` | **12** |
| `intermediate_size` | 3072 |
| `vocab_size` | **30522** (WordPiece) |
| `max_position_embeddings` | **512** |
| `type_vocab_size` | 2 |
| `tokenizer_class` | `BertTokenizer` (WordPiece) |
| `hidden_act` | **`gelu`** (HF default — 정확히는 `gelu_new`/`gelu`/`gelu_pytorch_tanh` 중 하나; bert-base-uncased는 보통 `gelu`) |
| `layer_norm_eps` | 1e-12 |

### 3.2 forward 골격
```
BertForMaskedLM
└─ BertModel
    ├─ BertEmbeddings
    └─ BertEncoder
        └─ 12× BertLayer
              ├─ BertAttention
              │   └─ BertSelfAttention (full O(n²): Q·K^T → softmax → ·V)
              ├─ BertSelfOutput  (Linear + LayerNorm + residual)
              ├─ BertIntermediate (Linear + gelu)
              └─ BertOutput      (Linear + LayerNorm + residual)
```

### 3.3 사전학습 정보
- 데이터셋: **BookCorpus + English Wikipedia** (약 16GB)
- 객체: MLM + NSP (Next Sentence Prediction)
- → Nystromformer 사전학습 코퍼스와 *동일* (단 NSP 객체는 추가됨)

---

## 4. 두 모델의 중간 텐서 흐름 — 어디서 어떤 값이 나오는가

> 회로 측이 필요한 키 (현재 `attention_pipeline.py` / `bert_attention_pipeline.py` 출력) 기준으로, HF forward 그래프 어디서 가져올 수 있는지 매핑.

### 4.1 Nystromformer (HF) 측

| 회로 키 (현 npz) | HF forward 위치 | hook 가능? | 비고 |
|------------------|-----------------|-----------|------|
| `Q_proc`, `K_proc` | `NystromformerSelfAttention.forward` 내부 변수 (q · sqrt(scale)) | ❌ | closure 내부 — forward hook으로 직접 못 잡음. monkey-patch 필요 |
| `V` | 같음 | ❌ | 같은 이유 |
| `Q_landmarks`, `K_landmarks` | 내부 `q_landmarks`, `k_landmarks` | ❌ | monkey-patch 필요 |
| `kernel_1/2/3_scores` | softmax 직전 점수 | ❌ | 노출되지 않음 |
| `kernel_1/2/3` | softmax 결과 | ❌ | 노출되지 않음 |
| `inv_kernel_2` | `iterative_inv(kernel_2)` 결과 | ❌ | 노출되지 않음 |
| `temp` (k3·V), `mid` (k1·k2⁻¹), `X` | 내부 곱셈 결과 | ❌ | 노출되지 않음 |
| `conv_term`, `X_final` | conv2d(V, kernel=65) | ❌ | 노출되지 않음 |
| `mask` | 외부에서 input batch로 넘김 | ✅ | input에서 직접 |

**결론**: HF의 `NystromformerSelfAttention.forward` 본문 안의 모든 중간값은 closure에 갇혀 있어, 단순 forward hook으로는 못 가져온다. 두 가지 방법 중 하나가 필요:
- (a) `NystromformerSelfAttention.forward`를 fork한 함수로 monkey-patch 하여 같은 키 이름으로 외부 dict에 push
- (b) 회로 측이 “Q/K/V projection 결과 + 최종 attention output”만 신뢰하고 중간값은 *증명자 측이 다시 계산*하도록 회로 입력 스펙을 단순화 (Freivalds 검증으로 충분히 가능)

(b) 쪽이 본 논문 contribution과 잘 맞는다. Freivalds + CP-SNARK가 “큰 행렬곱은 회로 밖”이라는 디자인을 강조하므로, *prover 측이 npz를 만들 때 큰 곱은 직접 계산하고 회로는 무작위 검사만 한다*는 흐름이 자연스럽다.

### 4.2 BERT (HF `BertSelfAttention`) 측

| 회로 키 | HF forward 위치 | hook 가능? |
|---------|-----------------|-----------|
| `Q_proc` | `module.query(hidden_states)` | ✅ (현재 `run_bert_validation.py:93-108`이 이렇게 함) |
| `K_proc` | `module.key(hidden_states)` | ✅ |
| `V` | `module.value(hidden_states)` | ✅ |
| `mask` | input | ✅ |
| `X` | `BertSelfAttention.forward` 출력 | ✅ (현재 hook_output[0]로 잡힘) |

**결론**: BERT 측은 현재 RoBERTa baseline과 동일한 hook 전략을 그대로 쓸 수 있다. `RobertaSelfAttention` → `BertSelfAttention` 클래스 이름만 바꾸면 됨.

### 4.3 Norm/FFN 캡처 (양 모델 공통)

| 회로 키 | HF 모듈 | hook 가능? |
|---------|---------|-----------|
| `rms_input`, `rms_output` (LayerNorm 입력·출력) | `*.LayerNorm` | ✅ forward hook |
| `ms` (RMSNorm용 평균제곱) | LayerNorm은 `mean`+`var` 사용 — `ms`만 저장하는 건 **수치적으로 불완전** | ⚠️ |
| `linear_weight`, `linear_bias`, `linear_output` (FFN) | `*.intermediate.dense`, `*.output.dense` | ✅ |
| `gelu_output` | `*.intermediate` (Linear+GELU 합쳐진 모듈) | ✅ |

→ **LayerNorm을 RMSNorm 키 형태로 위장 저장하는 현 방식은 회로의 수학적 검증과 어긋난다.** 회로가 “rms_output = rms_weight × rms_input / sqrt(ms+eps)” 관계를 enforce 하면, 평균을 빼는 LayerNorm 결과로는 이 식이 성립하지 않아 verify가 실패한다. Phase 3에서 키 이름을 `ln_*`로 바꾸고 회로 측도 LayerNorm 식으로 분기시켜야 한다.

---

## 5. 기존 회로와 호환되지 않는 부분 식별

### 5.1 결정적 불일치 (반드시 회로/lookup 수정 필요)

| 항목 | 현재 회로 전제 | 목표 모델 실제 | 회로 영향 |
|------|----------------|----------------|----------|
| Normalization | RMSNorm: `out = w · x / √(mean(x²)+ε)` | LayerNorm: `out = w · (x − μ) / √(var(x)+ε) + b` | **`mlp_lookup.rs::run_lasso_for_rmsnorm` 재작성**, `rms_table.py` 함수 변경, NystromMLPCircuit/RoBERTaMLPCircuit의 norm gadget 변경 |
| Activation | `1.702 σ(1.702 x)` (sigmoid GELU) | `gelu_new = 0.5 x (1 + tanh(√(2/π)·(x + 0.044715 x³)))` (Nystromformer) / `gelu` (BERT) | **`gelu_table.py` 재생성**, `mlp_lookup.rs::run_lasso_for_gelu` 인덱스 매핑 재계산 |
| Attention conv kernel | 3×1 depthwise | **65×1 depthwise** | NystromAttentionCircuit가 conv 결과를 검증한다면 kernel 차원 상수 수정. Conv 결과를 회로 외부에서 처리 중이라면 영향 작음 |
| `max_seq_len` | 512 | Nystromformer 510 / BERT 512 | seq_len 상수 수정. Nystromformer는 BOS+EOS 포함 510이라 데이터 collation 측 처리 필요 |

### 5.2 토크나이저·vocab 불일치 (npz 추출에는 영향, 회로에는 영향 없음)

| 항목 | Nystromformer (HF) | BERT-base-uncased | Phase 3에서의 영향 |
|------|--------------------|--------------------|----------------------|
| Tokenizer | AlbertTokenizer (SentencePiece) | BertTokenizer (WordPiece) | 두 모델에 *각자* 다른 tokenizer 적용. 같은 raw text를 공통 입력으로 쓸 수는 있으나 token sequence는 다름 → 회로 입력 텐서 (Q/K/V 등)가 모델별로 완전 분리됨 |
| vocab_size | 30000 | 30522 | 임베딩 행렬 크기 다름. 회로는 임베딩 검증을 안 하므로 무영향 |
| max_position_embeddings | 510 | 512 | Nystromformer는 sequence length를 510으로 truncate 해야 함 |

> **공정성 주의**: 사용자는 “두 모델이 동일 BookCorpus+Wiki로 사전학습되어 공정한 비교가 가능”하다고 기술. *코퍼스 일치는 사실*이지만, **tokenizer가 서로 다르다** (Albert SP 30k vs BERT WP 30.5k) — 따라서 동일 입력 텍스트가 서로 다른 token 분해를 만든다. 본 논문 contribution이 *회로 비용*이고 *학습 정확도*가 아니므로 이 비대칭이 결과 해석에 직접 영향을 주지는 않으나, §7 limitation에 한 줄 명시할 가치는 있다.

### 5.3 호환되는 부분 (그대로 사용 가능)
- **Softmax**: 두 모델 모두 row-wise softmax. `softmax_table.py` 그대로 재사용 가능.
- **Q/K/V Linear projection**: 두 모델 모두 `nn.Linear` — hook으로 동일하게 추출.
- **Hidden dim 768 / num_heads 12**: 두 모델이 동일. 회로의 차원 상수가 동시에 잘 맞음.
- **Freivalds 무작위 행렬곱 검증**: 모델과 무관 — 행렬 차원만 정확히 알면 그대로 동작.
- **CP-SNARK commit-and-prove 인프라**: 모델 무관, 변경 없음.

### 5.4 회로 비용 측면의 변화 (§7 결과 영향)

| 측정량 | dim 128 → 768 (×6) | layer 2 → 12 (×6) | landmark 64 (동일) | conv kernel 3 → 65 (×22) |
|--------|---------------------|---------------------|---------------------|---------------------------|
| Constraint 수 | ↑ 약 36× (선형 가중) | ↑ 약 6× (layer 비례) | ─ | ↑ (conv 회로 부분만, 비중 작음) |
| Prover time | ↑ ~36× × ~6× ≈ 200×+ | | | |
| Proof size (Groth16) | 거의 상수 | | | |
| Verifier time | 거의 상수 | | | |

→ **회로 비용이 200× 이상 증가할 수 있다.** *(추정)*. 이것이 “실용적 시퀀스 길이”에서 여전히 의미 있는지 §7 sweep에서 명확히 측정해야 한다. 작업 가능한 dim/layer로 줄이려면 다시 self-train이 필요하므로 본 migration의 의의(공개 모델 사용)와 정면 충돌한다.

---

## 6. 추가 확인 필요

- [ ] 사용자가 의도한 “비교 공정성”의 범위 — *코퍼스만* 일치하면 되는지, *tokenizer까지* 일치를 요구하는지.
- [ ] Nystromformer HF의 conv (kernel=65)를 회로에서 검증해야 하는지, 아니면 “회로 외 inner product 검증”으로 분리해도 되는지 (논문 contribution과 잘 맞음).
- [ ] dim 768·layer 12로 회로 비용이 200×↑가 되어도 §7 결과가 “practical”로 표현 가능한지. 만약 안 된다면 distill된 작은 모델 (예: `prajjwal1/bert-tiny` 류)도 함께 사용할 의향이 있는지.
- [ ] Nystromformer HF의 `iterative_inv` 구현이 현재 `attention_nystrom.py`의 Schulz와 동일한지 — 다르면 inv 회로의 step 수도 맞춰야 함.
