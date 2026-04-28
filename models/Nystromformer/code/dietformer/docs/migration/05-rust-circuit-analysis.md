# Phase 0 — Rust 회로 사전 분석

> 본 문서는 작업 폴더 외부 (`circuits/`)를 read-only로 직접 분석한 결과이다.
> 회로 코드 수정은 일절 하지 않았다.
> 명시적으로 코드에서 확인되지 않은 추론은 **(추정)** 으로 표시한다.

---

## 컨텍스트 보정 (사용자 입력 반영)

- `bookcorpus-512/config.json`은 이후 `dim=768, layer=12`로 갱신됨 (Phase 0 보고서가 본 디스크 상태는 옛 값).
- partition pickle 912개는 Linux 서버에 있으며, 직접 학습 폐기 결정에 따라 **`dataset.py`의 partition 로딩 / `train.json`+`dev.json` 분할 / `run_pretrain.py` 자체가 모두 폐기 대상**.
- §7 평가 입력은 raw text 수 개로 충분.

---

## Q1 — RMSNorm 가젯의 정확한 위치

### 발견 결과

**RMSNorm은 두 곳에서 강제된다 — Lasso lookup과 R1CS constraint 양쪽.** 단순 lookup-only가 아니다.

#### (a) R1CS constraint — `circuits/src/gadgets/layernorm/constraints.rs::enforce_rmsnorm`

> 폴더 이름은 `layernorm/`이지만, 실제로 enforce하는 식은 **RMSNorm**이다. 이는 본 마이그레이션의 주요 함정이다.

라인 31–142. 핵심 발췌:

```rust
// constraints.rs:64–81 — mean-squared 누적
for d in 0..bundle.q_ri[b][s].len() {
    let q_ri_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ri[b][s][d]))?;
    let diff = q_ri_var - &z_ri_var;
    sum_sq_circuit += diff.square()?;
    q_diff_vars.push(diff);
}
let lhs_ms = &m_ms_var * (&sum_sq_circuit + &m_prime_var);
let rhs_ms = (&shifted_var * &two_k) + &r_shift_var;
lhs_ms.enforce_equal(&rhs_ms)?;
```

```rust
// constraints.rs:88–94 — 양자화된 1/sqrt 관계
let q_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ms_sim[b][s][0]))?;
let r_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_ms[b][s][0]))?;
shifted_var.enforce_equal(&((&q_ms_var * &dim_var) + &r_ms_var))?;
```

```rust
// constraints.rs:102–124 — 최종 RMSNorm output 양자화 constraint
let inv_rms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.inv_rms[b][s][0]))?;
for d in 0..bundle.q_ri[b][s].len() {
    let q_diff_var = q_diff_vars[d].clone();
    let product = &q_diff_var * &inv_rms_var * &gamma_diffs[d];
    // M_norm * ((Q_ri − z_ri) · inv_rms · (Q_gamma − z_gamma) + M'_norm)
    //        = (Q_out << k_bits) + R_out
    let lhs = &m_norm_var * (product + &m_prime_norm_var);
    let rhs = (&q_out_var * &two_k) + &r_out_var;
    lhs.enforce_equal(&rhs)?;
}
```

#### (b) Lasso lookup — `circuits/src/util/mlp_lookup.rs::run_lasso_for_rmsnorm`

라인 27–61. `bundle.q_ms_sim[b][s][0]`을 인덱스로 받아 RMSNormSubtableStrategy (C=1, M=2¹⁶)에 lookup. 이는 양자화된 `1/√(ms+ε)` 값이 사전 계산된 표 안에 있는지 확인.

#### (c) 회로 호출 지점

- `circuits/src/circuits/freivalds_matmul/nystrom_mlp_circuit.rs:68, 72`:
  ```rust
  let q_norm1_vars = enforce_rmsnorm(cs.clone(), &self.norm1, dim, self.frac_bits, self.k_bits, &mut r_pool, &mut rms1_pool)?;
  let q_norm2_vars = enforce_rmsnorm(cs.clone(), &self.norm2, dim, self.frac_bits, self.k_bits, &mut r_pool, &mut rms2_pool)?;
  ```
- `circuits/src/circuits/freivalds_matmul/roberta_mlp_circuit.rs:29`:
  ```rust
  pub type RoBERTaMLPCircuit<F> = NystromMLPCircuit<F>;
  ```
  → **type alias**. RoBERTa 회로도 동일한 `enforce_rmsnorm`을 호출한다.

### 한 줄 결론
RMSNorm은 lookup이 아니라 **실제 R1CS constraint로 enforce**되며, RoBERTa 회로도 type alias로 같은 enforce 경로를 쓴다.

### Phase 5 영향 범위: **재설계 필요**
LayerNorm으로 가려면 새 가젯 `enforce_layernorm`을 작성하거나 기존 가젯에 mean-차감 + bias 항을 추가해야 한다. NormBundle 구조체에도 `q_mean`, `q_beta`, `z_beta` 필드 추가 필요. 가장 비용이 큰 항목.

---

## Q2 — Conv kernel size 상수

### 발견 결과
**`KERNEL_SIZE`, `kernel_size`, `conv_kernel`, `Conv2d`, `conv_term` 등 conv 관련 식별자가 Rust 회로 코드 어디에도 발견되지 않았다.**

### 한 줄 결론
회로는 conv kernel size를 const로도 동적으로도 사용하지 않는다. conv 자체가 Rust 측 회로의 검증 범위 밖이다.

### Phase 5 영향 범위: **변경 범위 작음**
HF Nystromformer의 conv kernel size 65는 회로에 영향이 없다. Python 측 npz 추출에서 자동으로 따라간다.

---

## Q3 — Conv 검증 범위

### 발견 결과

`circuits/src/circuits/freivalds_matmul/nystrom_attn_circuit.rs:45–80`의 `NystromAttentionCircuit` 구조체 필드:

```rust
pub struct NystromAttentionCircuit<F: PrimeField> {
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,     // [B][H][S][D]
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,
    pub matmul1: MatmulBundle<F>, // QK^T (kernel_1)
    pub matmul2: MatmulBundle<F>, // kernel_2
    pub matmul3: MatmulBundle<F>, // kernel_3
    pub sm1/sm2/sm3: SoftmaxBundle<F>,
    pub pseudo_inv: PseudoinverseBundle<F>,
    pub matmul_temp: MatmulBundle<F>,
    pub matmul_mid: MatmulBundle<F>,
    pub matmul_x: MatmulBundle<F>,
    // ... no conv field
}
```

→ conv 관련 bundle 없음. `q_v`가 직접 attention matmul에 들어감 (`nystrom_attn_circuit.rs:109`).

### 한 줄 결론
회로는 conv 결과를 검증하지 않는다. 회로의 contract는 *“Q/K/V가 주어졌을 때 attention 출력의 정확성”* — Q/K/V 자체의 출처(임베딩, conv 등)는 회로 외부의 신뢰 영역이다.

### Phase 5 영향 범위: **변경 범위 작음**
본 논문 contribution(*“큰 행렬곱·pseudoinverse를 회로 밖으로 분리하고 inner product만 회로 안에서 검증”*)과 정확히 일치한다. conv kernel size 변경은 회로에 무영향. **단** §7 limitation에 *“conv 부분은 회로 외부에서 신뢰됨 — 별도 검증 메커니즘 필요”*를 명시할 가치는 있다.

---

## Q4 — frac_bits / limb_bits / k_bits 정의 위치

### 발견 결과

#### (a) Compile-time 상수 — `circuits/src/config.rs:5–23`

```rust
pub const K_SEGMENTS: usize = 5;          // line 5
pub const LIMB_BITS: usize = 16;          // line 6
pub const LIMB_BASE: u128 = 1u128 << LIMB_BITS;  // line 7
pub const SOFTMAX_C: usize = 5;           // line 10
pub const SOFTMAX_M: usize = 65536;       // line 11
pub const RMSNORM_C: usize = 1;           // line 13
pub const RMSNORM_M: usize = 65536;       // line 14
pub const GELU_C: usize = 1;              // line 16
pub const GELU_M: usize = 65536;          // line 17
pub const KBITS_C: usize = 2;             // line 19
pub const KBITS_M: usize = 65536;         // line 20
pub const FRACBITS_C: usize = 1;          // line 22
pub const FRACBITS_M: usize = 65536;      // line 23
```

#### (b) Runtime 양자화 파라미터 — npz에서 읽음

```rust
// nystrom_attn_circuit.rs:575–578
let frac_bits = npz.by_name::<_, Ix0>("frac_bits.npy")?.into_scalar() as usize;
let limb_bits = npz.by_name::<_, Ix0>("limb_bits.npy")?.into_scalar() as usize;
let k_bits    = npz.by_name::<_, Ix0>("k_bits.npy")?.into_scalar() as usize;

// roberta_attn_circuit.rs:344–351 (같음 + k_segments 추가)
let k_segments = npz.by_name::<_, Ix0>("k_segments.npy")?.into_scalar() as usize;

// nystrom_mlp_circuit.rs:93–94 (k_bits, frac_bits만)
```

#### (c) constraint 함수로의 전파

- `enforce_rmsnorm(cs, bundle, dim, frac_bits, k_bits, ...)` — `gadgets/layernorm/constraints.rs:31`
- `enforce_linear(cs, ..., k_bits, ...)` — `gadgets/linear/constraints.rs:24`
- `enforce_gelu(cs, ..., k_bits, ...)` — `gadgets/activation/gelu.rs:34`

모두 `two_k = F::from(1u64 << k_bits)`로 shift 상수를 만들어 사용.

### 한 줄 결론
양자화 파라미터(frac_bits, limb_bits, k_bits)는 npz에서 동적으로 받아 회로 인스턴스에 박힌다. 단 lookup 표 크기(SOFTMAX_M=2¹⁶ 등)는 compile-time 상수.

### Phase 5 영향 범위: **변경 범위 작음**
Python에서 다른 frac_bits/k_bits를 npz에 쓰면 자동 적용. **단** dim=128 → 768로 가면 양자화 분포가 넓어져 lookup 표(2¹⁶)를 벗어날 가능성이 있다 *(추정)* — 표 크기 재설계 필요할 수 있음. Phase 6의 첫 e2e 시험에서 “lookup index out of bounds” 에러가 나는지 확인이 필요하다.

---

## Q5 — npz 키 hardcoding 위치

### 발견 결과

**`circuits/src/util/load.rs`는 helper 함수만 제공** (`to_field_3d_i64`, `to_field_4d_u64` 등 line 24–217). **string literal 키 자체는 회로 loader 파일에 박혀 있다.**

#### `nystrom_mlp_circuit.rs` — Norm + Linear + GeLU 키 (prefix 기반)

```rust
// nystrom_mlp_circuit.rs:93–94 — metadata
"k_bits.npy", "frac_bits.npy"

// nystrom_mlp_circuit.rs:96–116 — load_norm 클로저 (prefix = "rms1" or "rms2")
"Q_ri_{prefix}.npy"          // line 98
"z_ri_{prefix}.npy"          // line 99
"M_ms_{prefix}.npy"          // line 101
"M_prime_ms_{prefix}.npy"    // line 102
"shifted_val_{prefix}.npy"   // line 103
"R_shift_{prefix}.npy"       // line 104
"Q_ms_sim_{prefix}.npy"      // line 105
"R_ms_{prefix}.npy"          // line 106
"inv_rms_total_{prefix}.npy" // line 107
"Q_gamma_{prefix}.npy"       // line 108
"z_gamma_{prefix}.npy"       // line 109
"Q_rms_output_sim_{prefix}.npy" // line 110
"R_rms_output_{prefix}.npy"  // line 111
"M_{prefix}.npy"             // line 112
"M_norm_{prefix}.npy"        // line 113
"M_prime_norm_{prefix}.npy"  // line 114

// nystrom_mlp_circuit.rs:120–133 — load_linear (prefix = "lin1" or "lin2")
"z_in_{prefix}.npy"   // line 122
"Q_w_diff_{prefix}.npy"   // line 124
"Q_b_diff_{prefix}.npy"   // line 125
"M_{prefix}.npy"          // line 127
"M_prime_{prefix}.npy"    // line 128
"Q_out_sim_{prefix}.npy"  // line 130
"R_out_{prefix}.npy"      // line 131

// nystrom_mlp_circuit.rs:137–148 — load_gelu (prefix = "gelu")
"z_in_{prefix}.npy"        // line 139
"sigmoid_part_{prefix}.npy" // line 140
"M_{prefix}.npy"           // line 142
"M_prime_{prefix}.npy"     // line 143
"Q_out_sim_{prefix}.npy"   // line 144
"R_out_{prefix}.npy"       // line 145
"M_gelu_idx.npy"           // line 146 (no prefix)
```

#### `roberta_attn_circuit.rs` — Full O(n²) attention

```rust
// roberta_attn_circuit.rs:269–279 — 입력
"Q_q.npy", "Q_k.npy", "Q_v.npy", "z_q.npy", "z_k.npy", "z_v.npy"

// 314–339 — attention matmul
"Q_sim_attn.npy", "R_matmul_attn.npy", "M_attn.npy", "M_prime_attn.npy", "z_sim_attn.npy"

// 356–418 — softmax
"limbs.npy", "exp_limbs.npy", "Q_exp.npy", "R_exp.npy", "exp_total.npy",
"P_attn.npy", "R_div_attn.npy", "r_sm.npy", "M_sm.npy"

// 422–449 — output matmul
"Q_sim_x.npy", "R_matmul_x.npy", "M_x.npy", "M_prime_x.npy", "z_sim_x.npy"

// 453 — mask
"mask.npy"
```

#### `nystrom_attn_circuit.rs` — Nyström 3-kernel + pseudo-inverse

```rust
// nystrom_attn_circuit.rs:531–620 (요약)
"Q_q.npy", "Q_k.npy", "Q_v.npy", "seg_size.npy"
// 3 kernel suffixes: "k1", "k2", "k3"
"R_mean_{suffix}.npy", "Q_sim_{suffix}.npy", "R_matmul_{suffix}.npy", ...
// pseudo-inverse 4 step
// final: "Q_sim_x.npy", "M_x.npy", ...
```

### 한 줄 결론
npz 키는 회로 loader 파일들에 string literal로 박혀 있고, Python witness pipeline이 정확히 같은 이름으로 출력해야 한다. `load.rs`는 단순 helper.

### Phase 5 영향 범위: **변경 범위 작음 ~ 큼 (분기에 따라)**
- RMSNorm을 그대로 두면: 키 이름 변경 불필요 (shape만 dim=768로 자동 확장).
- LayerNorm으로 바꾸면: `Q_mean_*`, `Q_beta_*`, `z_beta_*`, `Q_ln_output_*` 등 새 키들이 회로 loader에 추가되어야 함. RMSNorm 키 (`Q_ms_sim_*`, `inv_rms_total_*`, `M_ms_*` 등) 제거 또는 분기.

---

## Q6 — RoBERTa baseline 회로의 RMSNorm vs LayerNorm 충돌 (가장 중요)

### 발견 결과

#### (a) RoBERTa MLP 회로는 type alias

```rust
// roberta_mlp_circuit.rs:28–39
pub type RoBERTaMLPCircuit<F> = NystromMLPCircuit<F>;

pub fn load_roberta_mlp_circuit<E: Pairing>(path: &Path) -> Result<RoBERTaMLPCircuit<E::ScalarField>>
where <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{ load_nystrom_circuit::<E>(path) }
```

→ **RoBERTa 회로는 Nystrom MLP 회로와 정확히 동일한 R1CS** (`enforce_rmsnorm` 포함).

#### (b) enforce_rmsnorm은 RMSNorm 식을 강하게 enforce

`gadgets/layernorm/constraints.rs:122–124` (Q1에서 인용):

```rust
let lhs = &m_norm_var * (product + &m_prime_norm_var);
let rhs = (&q_out_var * &two_k) + &r_out_var;
lhs.enforce_equal(&rhs)?;
```

여기서 `product = (Q_ri − z_ri) · inv_rms · (Q_gamma − z_gamma)` (line 105).

이 식은 정확히 **RMSNorm**: `out ≈ γ · x / √(mean(x²)+ε)`.
**bias 항이 없고, mean 차감도 없다.**

#### (c) LayerNorm 식은 다르다

LayerNorm: `out = γ · (x − μ) / √(var + ε) + β`

→ **mean μ와 bias β 두 항이 추가**된다.

#### (d) 현재 Python의 LayerNorm 위장 hook (`run_bert_validation.py:130-137`)

```python
saved_norm[layer_key] = {
    "rms_input":  x,                               # = LayerNorm 입력
    "ms":         x.pow(2).mean(dim=-1, keepdim=True),  # = 평균제곱 (mean(x²))
    "rms_weight": module.weight,                   # = LayerNorm γ
    "rms_output": output,                          # = LayerNorm 결과: γ·(x−μ)/√(var+ε) + β
}
```

#### (e) 회로 검증의 수치적 결과 (예측)

회로가 enforce하는 식:
```
out_quantized ≈ γ · (input − z_input) / √(mean(input²)+ε)
```

Python이 제공하는 `rms_output`:
```
out_actual = γ · (input − μ) / √(var+ε) + β
```

두 식은 다음 경우에만 같아진다:
1. μ = 0 (입력 평균이 0)
2. β = 0 (bias가 0)
3. mean(x²) ≈ var (μ=0일 때만 성립)

**RoBERTa의 실제 LayerNorm bias β는 0이 아니다.** 또한 입력 hidden state의 평균이 정확히 0이라는 보장도 없다 (보통 학습 중 0에 가깝게 자기조정되지만 정확한 0은 아니다).

→ **R1CS witness 검증 단계에서 `lhs.enforce_equal(&rhs)`가 실패할 가능성이 매우 높다.**

#### (f) 그래도 통과한다면 가능한 이유 *(추정)*

만약 현 코드가 실제로 통과한다면, 다음 중 하나의 “느슨한” 처리가 있을 수 있다:
1. 양자화 오차가 워낙 커서 LayerNorm/RMSNorm 차이를 흡수 (k_bits=22로 제어되는 정밀도 안에서 우연히 일치) — *(추정)*
2. 회로가 실제로 verify까지 돌지 않고 prove 단계의 numerical witness만 쓴 채로 종료 — *(추정)*
3. inv_rms 값을 Python이 미리 RMSNorm 식 결과로 backsolve하여 넣고 있음 — *(추정)*

이 세 가지는 모두 **추정**이며, 실제 동작은 Phase 5/6에서 verify까지 한 사이클 돌려봐야 확인 가능하다.

### 한 줄 결론
**RoBERTa baseline 회로는 RMSNorm 식을 R1CS로 enforce하므로, LayerNorm 출력을 그대로 넣으면 verify가 실패할 것으로 예상된다.** 현재 RoBERTa baseline이 “통과한다고 보였다면” 위 세 가지 우회 중 하나일 가능성이 높고, 결과 신뢰성에 의문이 생긴다.

### Phase 5 영향 범위: **재설계 필요**
- 새 가젯 `enforce_layernorm` 작성 (mean 차감 + bias 항 추가)
- NormBundle에 `q_mean`, `q_beta`, `z_beta` 필드 추가
- npz 키 추가 (`Q_mean_{prefix}`, `M_mean_{prefix}`, `Q_beta_{prefix}`, `z_beta_{prefix}` …)
- 새 lookup 표 (LayerNorm용 `1/√(var+ε)`)
- Phase 6 전에 **현재 RoBERTa baseline 결과가 실제로 verify 통과했는지** 재검증 필요 — 만약 prove만 돌고 verify를 안 돌렸다면 §7 baseline 결과 자체가 무효.

---

## Q7 — 기존 벤치마크 / 비용 측정 기록

### 발견 결과

#### (a) 주요 로그: `circuits/src/circuits/freivalds_matmul/experiment.log`

라인 1–534. Nyström attention 두 layer에 대한 두 번의 e2e prove 사이클 기록.

**Run 1 (`outputs-batch0-model.transformer_0.mha.attn.witness.npz`, lines 148–330):**

| 항목 | 값 |
|------|-----|
| Setup time | 152,981 ms = **152.98 sec** |
| Groth16 prove | 14,501 ms = **14.5 sec** |
| Groth16 verify | 1 ms |
| Lasso total | 6,797 ms = 6.8 sec |
| CP-Link setup | 42,819 ms = 42.8 sec |
| CP-Link prove | 256 ms |
| CP-Link verify | 750 ms |
| **End-to-end** | **222,625 ms = 222.6 sec ≈ 3.7 min** |
| **R1CS constraints** | **3,097,984 (~3.1 M)** |
| Proof size (Groth16) | 384 bytes |
| Lasso commitment size | 266,616 bytes ≈ 260 KB |

**Run 2 (transformer_1):** 거의 동일 (E2E ≈ 224.5 sec, constraints 동일 3.1M).

**시스템 자원 (lines 514–534):**
- Memory resident: 17.2 GB
- Peak memory: 39 GB
- Wall-clock 양 layer 합계: 571.12 sec

#### (b) Lasso 내부 분해

| Lasso 항목 | 시간 |
|-----------|------|
| Softmax k1 (sparsity 65536) | 730 ms |
| Softmax k2 (sparsity 8192) | 329 ms |
| Softmax k3 (sparsity 65536) | 726 ms |
| Matmul remainder k1/k2/k3 | 353 / 169 / 356 ms |
| Softmax div remainder k1/k2/k3 | 197 / 107 / 196 ms |
| Pseudoinverse 4 step | ~150 ms × 4 |
| Output matmul (v/mid/x) | 140 / 345 / 236 ms |

#### (c) 벤치마크가 측정한 회로 규모

이 로그가 dim=128/layer=2 시점인지, 갱신된 dim=768/layer=12 시점인지는 로그 자체에 모델 차원이 명시되어 있지 않아 *(추정)*. 다만:
- `experiment.log`는 git status에서 **untracked**로 표시됨
- 본 회로는 `transformer_0`, `transformer_1` 두 layer만 prove함 (config의 num_layers=2와 일치)
- constraint 수 3.1M은 dim 작은 모델에 부합

→ 본 로그는 **dim=128/layer=2 시점의 측정치일 가능성이 높다** *(추정)*.

#### (d) dim=768/layer=12로의 스케일링 추정

매트멀 R1CS는 보통 `O(B · H · S · d²)` 또는 `O(B · H · S² · d)` 형태로 비용이 든다. dim 6× 증가 → constraint **약 36×** *(추정)*. layer 6× 증가 → constraint **6×** *(추정)*.

| 항목 | 현재 (per layer) | 12-layer 합계 *(추정)* |
|------|------------------|--------------------------|
| Constraints | 3.1M | 3.1M × 36 × 6 ≈ **670M** |
| Setup | 153 sec | 153 × 36 × 6 ≈ **9.2 hr** (parallelizable) |
| Prove | 14.5 sec | 14.5 × 40 × 6 ≈ **58 min** |
| Memory peak | 39 GB | 39 GB × 36 = **1.4 TB** *(매우 비현실적 — 더 정교한 스케일링 모델 필요)* |
| Lasso 표 크기 (RMSNORM_M, GELU_M, SOFTMAX_M) | 2¹⁶ | 동일 (compile-time 고정) — out-of-bounds 위험 *(추정)* |

→ **메모리는 단순 36× 곱으로는 비현실적인 1.4TB가 나오므로, 실제 스케일링 곡선은 비선형일 가능성이 높다.** Phase 5/6 진입 전 dim=256 또는 dim=384로 중간 측정해야 정확한 곡선을 잡을 수 있다.

#### (e) 다른 벤치마크 출력

- README, criterion bench, JSON 결과 파일은 **회로 루트에서 발견되지 않음** (기존 검색 범위 한정).
- `experiment.log`가 사실상 단일 측정 기록.

### 한 줄 결론
현재 회로는 dim=128/layer=2에서 layer당 prove ≈ 15초, E2E ≈ 3.7분, constraint 3.1M. dim=768/layer=12로 가면 prove 1시간 내, constraint 670M *(추정)*, 메모리는 비선형적으로 매우 큼.

### Phase 5 영향 범위: **변경 범위 큼 (사전 측정 필수)**
- Phase 5 진입 전, 현재 회로를 dim=256, dim=384 등 중간 차원에서 한 번씩 돌려 “스케일링 곡선”을 잡아야 함.
- Lasso 표 크기 2¹⁶가 dim=768에서 충분한지 sanity check.
- 메모리가 PC급에서 처리 불가능한 수준이라면, 회로 분할(per-layer 별 prove + recursion / aggregation) 또는 모델 차원 축소(distillation) 전략을 함께 검토해야 함.

---

## 종합 — Phase 5 작업 범위 매트릭스

| Q | 항목 | 현 상태 | Phase 5 영향 |
|---|------|---------|----------------|
| Q1 | RMSNorm gadget | `gadgets/layernorm/constraints.rs::enforce_rmsnorm` 에 R1CS + Lasso 강하게 enforce | **재설계 필요** |
| Q2 | Conv kernel const | 회로에 없음 | **변경 작음** |
| Q3 | Conv 검증 | 검증 안 함 (외부 신뢰) | **변경 작음** |
| Q4 | 양자화 상수 | npz dynamic + config.rs 표 크기는 const | **변경 작음** |
| Q5 | npz 키 hardcoding | 회로 loader에 string literal | **변경 작음** (RMSNorm 유지 시) / **큼** (LayerNorm 전환 시) |
| Q6 | RoBERTa LayerNorm 충돌 | 회로는 RMSNorm enforce / Python은 LayerNorm 출력 → **verify 실패 예상** | **재설계 필요** |
| Q7 | 벤치마크 / 스케일링 | layer당 15s, 3.1M constraints @ dim=128 | **사전 측정 필수**, 비선형 스케일링 가능성 |

---

## 핵심 결정 포인트 (사용자 검토 요청)

### A. Q6의 충돌 해석

다음 두 가지 중 어느 쪽인지 빨리 판단하는 것이 가장 중요합니다:

1. **현재 RoBERTa baseline은 prove만 돌고 verify를 실제로 거치지 않은 상태에서 “돌아간다”고 분류되었을 가능성** — *(추정)*. 이 경우 §7의 RoBERTa baseline 결과는 무효이고, Phase 6에서 verify 사이클을 추가하면 그제서야 실패가 노출된다.
2. **양자화 오차가 LayerNorm/RMSNorm 차이를 흡수해 우연히 통과해왔을 가능성** — *(추정)*. 이 경우 결과는 “수학적으로 부정확하지만 numerical하게 검증 가능한 우회” 상태이고, 논문의 정직성 측면에서 명시적 보정이 필요하다.

→ Phase 5 진입 전, **현재 RoBERTa baseline을 작은 입력 1개로 prove + verify 끝까지 돌려서 실제 통과 여부를 확인**해야 한다. 이것이 “선결 작업 B”의 첫 번째 항목으로 자연스럽다.

### B. RMSNorm 유지 vs LayerNorm 전환

| 옵션 | 장점 | 단점 |
|------|------|------|
| RMSNorm 유지 | 회로 변경 최소, 기존 가젯 그대로 | HF 공개 모델이 LayerNorm을 쓰므로 *“공개 모델 그대로 검증”* 주장이 약해짐. Python에서 LayerNorm을 RMSNorm으로 변환하는 “선처리”를 정의해야 함 — 정직성 의문 |
| LayerNorm 전환 | HF 모델 그대로. §7에서 *“표준 LayerNorm을 회로로 검증”* 주장 가능 | 가젯 신규 작성, NormBundle 확장, Lasso 표 신규 생성 필요. 회로 비용 증가 *(추정 +20%)* |

본 논문 contribution이 *“회로 비용을 O(n²)에서 O(n)으로”* 임을 고려하면, **LayerNorm 전환이 이론적 정합성에 더 잘 맞는다**. 단 작업량은 크다.

### C. 회로 비용 스케일링 사전 측정

Phase 5에 들어가기 전에 dim=256, dim=384에서 한 번씩 prove를 돌려 비선형 스케일링 곡선을 확보해야, dim=768에서 prove가 “수 시간 내” 또는 “수 일 내” 또는 “단일 머신 불가” 중 어디에 떨어질지 알 수 있다. 사용자가 언급한 “선결 작업 B”와 정확히 일치하는 작업.

---

## 추가로 확인 필요 (회로 코드만으로는 단정 불가)

- **`experiment.log`가 dim=128/layer=2 시점의 것인지 dim=768/layer=12 시점의 것인지** — 로그에 명시 없음. 사용자 확인 필요.
- **현재 RoBERTa baseline이 verify까지 통과해 왔는지** — Q6의 충돌 판단을 위해 실측 필요.
- **메모리 39GB peak이 단일 머신 RAM 한계에 가까운지** — 사용자 환경에 따라 다름. dim=768 스케일링 시 swap/OOM 위험 사전 평가 필요.
- **`circuits/src/cc_snark/`의 CP-Link commitment 목록이 LayerNorm 키 추가 시 어떻게 변하는지** — 본 분석에서는 cc_snark 폴더를 깊게 파지 않음. Phase 5 진입 시 추가 분석 필요.
