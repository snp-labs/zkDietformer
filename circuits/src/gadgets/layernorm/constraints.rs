use ark_ff::PrimeField;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_r1cs_std::alloc::{AllocVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::prelude::*;
use ark_r1cs_std::prelude::Boolean;

use crate::gadgets::activation::gelu::split_remainder_and_push;
#[derive(Clone)]
pub struct NormBundle<F: PrimeField> {
    pub q_ri: Vec<Vec<Vec<F>>>,
    pub z_ri: F,
    // pub sum_sq: Vec<Vec<Vec<F>>>,
    pub m_ms: F,
    pub m_prime_ms: F,
    pub shifted_val: Vec<Vec<Vec<F>>>,
    pub r_shift: Vec<Vec<Vec<F>>>,
    pub q_ms_sim: Vec<Vec<Vec<F>>>,
    pub r_ms: Vec<Vec<Vec<F>>>,
    pub inv_rms: Vec<Vec<Vec<F>>>,
    pub q_gamma: Vec<F>,
    pub z_gamma: F,
    pub q_out: Vec<Vec<Vec<F>>>,
    pub r_out: Vec<Vec<Vec<F>>>,
    pub m_rms_idx: F,
    pub m_norm: F,
    pub m_prime_norm: F,
}

// ── Phase 5-B B3-2: LayerNorm bundle ────────────────────────────────────────
//
// 자체학습 RMSNorm 시절의 `NormBundle` 보존 (위) + HF LayerNorm 측 데이터를 위한
// 신규 struct. RMSNorm 16 필드 + LayerNorm 신규 9 필드 (mean 3 + diff 1 + beta 5) +
// 의미 변경 키 (ms→var, inv_rms→inv_std, etc.) = 26 필드 per 블록.
//
// Python `bert_normalization_pipeline.py:_process_layernorm_block` 의 출력과
// 1:1 매핑. 키 매핑 표는 `docs/migration/20-b3-keymap-analysis.md` §4 참조.
//
// `enforce_layernorm` 가젯 impl 은 B4 에서 신규 작성 — 본 단계 (B3-2) 에서는
// struct 정의만 (loader 가 사용).

#[derive(Clone)]
pub struct LayerNormBundle<F: PrimeField> {
    // ── 입력 (RMSNorm 과 동일 의미) ──────────────────────────────────────
    pub q_ri: Vec<Vec<Vec<F>>>,
    pub z_ri: F,

    // ── 평균 (LayerNorm 신규) ────────────────────────────────────────────
    /// Σ Q_ri (mean 계산 입력). shape (B, S, 1)
    pub sum_x: Vec<Vec<Vec<F>>>,
    /// sum_x // dim — 양자화된 평균. shape (B, S, 1)
    pub q_mean: Vec<Vec<Vec<F>>>,
    /// sum_x % dim — division remainder. shape (B, S, 1)
    pub r_mean: Vec<Vec<Vec<F>>>,

    // ── 차이 (zero-mean, LayerNorm 신규) ─────────────────────────────────
    /// Q_ri − Q_mean. zero-point 자동 상쇄. shape (B, S, H)
    pub q_diff: Vec<Vec<Vec<F>>>,

    // ── 분산 (RMSNorm 의 ms 와 의미 다름: var = mean((x−μ)²)) ───────────
    /// Σ Q_diff² (variance 계산 입력). RMSNorm 의 sum_sq 와 다름 (mean 차감 후)
    pub sum_sq: Vec<Vec<Vec<F>>>,
    pub m_var: F,
    pub m_prime_var: F,
    pub shifted_val_var: Vec<Vec<Vec<F>>>,
    pub r_shift_var: Vec<Vec<Vec<F>>>,
    pub q_var_sim: Vec<Vec<Vec<F>>>,
    pub r_var: Vec<Vec<Vec<F>>>,

    // ── inv_std lookup (RMSNorm 의 inv_rms 와 동일 lookup 구조) ──────────
    /// 1/√(var + ε) · 2^FRAC_BITS lookup 결과
    pub inv_std: Vec<Vec<Vec<F>>>,
    /// floor(s_var · VAR_SCALE · 2^k_bits) — Phase 4 lookup 측 multiplier
    pub m_ln_idx: F,

    // ── γ (RMSNorm 과 동일) ──────────────────────────────────────────────
    pub q_gamma: Vec<F>,
    pub z_gamma: F,

    // ── β + lifting (LayerNorm 신규 — RMSNorm 부재) ──────────────────────
    pub q_beta: Vec<F>,
    pub z_beta: F,
    /// β를 inner-product scale 로 lifting 위한 multiplier
    pub m_beta_lift: F,
    /// (Q_beta − z_beta) · M_beta_lift >> k_bits
    pub q_beta_lifted: Vec<F>,
    /// β lifting remainder
    pub r_beta_lift: Vec<F>,

    // ── 출력 rescale (RMSNorm 과 동일 의미) ──────────────────────────────
    pub m_norm: F,
    pub m_prime_norm: F,
    /// 양자화된 LayerNorm 출력. shape (B, S, H)
    pub q_out: Vec<Vec<Vec<F>>>,
    pub r_out: Vec<Vec<Vec<F>>>,
}

pub fn enforce_rmsnorm<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    bundle: &NormBundle<F>,
    dim: usize,
    frac_bits :usize,
    k_bits:usize,
    r_pool: &mut Vec<FpVar<F>>,
    rms_pool: &mut Vec<FpVar<F>>,
) -> Result<Vec<Vec<Vec<FpVar<F>>>>, SynthesisError> {
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;
    let dim_var = FpVar::new_constant(cs.clone(), F::from(dim as u128))?;
    let z_ri_var = FpVar::new_constant(cs.clone(), bundle.z_ri)?;
    let m_ms_var = FpVar::new_constant(cs.clone(), bundle.m_ms)?;
    let m_prime_var = FpVar::new_constant(cs.clone(), bundle.m_prime_ms)?;
    let gamma_vars = bundle.q_gamma.iter().map(|g| FpVar::new_witness(cs.clone(), || Ok(*g))).collect::<Result<Vec<_>, _>>()?;
    let z_gamma_var = FpVar::new_constant(cs.clone(), bundle.z_gamma)?;
    // Precompute (gamma[d] - z_gamma) once — invariant across all (b, s) iterations.
    let gamma_diffs: Vec<FpVar<F>> = gamma_vars.iter()
        .map(|g| g - &z_gamma_var)
        .collect();
    let m_norm_var = FpVar::new_constant(cs.clone(), bundle.m_norm)?;
    let m_prime_norm_var = FpVar::new_constant(cs.clone(), bundle.m_prime_norm)?;
    let mut all_q_out_vars = Vec::new();
    
    let mut shift_low_vars = Vec::new();
    let mut shift_high_vars = Vec::new();
    let mut out_low_vars = Vec::new();
    let mut out_high_vars = Vec::new();
    let bit_limit = (dim as f64).log2().ceil() as usize;

    for b in 0..bundle.q_ri.len() {
        let mut batch_q_out_vars = Vec::new();
        for s in 0..bundle.q_ri[b].len() {
            // sum_sq = sum((Q_ri - z_ri)^2)
            let mut sum_sq_circuit = FpVar::zero();
            let mut q_diff_vars = Vec::new();
            for d in 0..bundle.q_ri[b][s].len() {
                let q_ri_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ri[b][s][d]))?;
                let diff = q_ri_var - &z_ri_var;
                sum_sq_circuit += diff.square()?;
                q_diff_vars.push(diff);
            }
            // let sum_sq_witness = FpVar::new_witness(cs.clone(), || Ok(bundle.sum_sq[b][s][0]))?;
            // sum_sq_circuit.enforce_equal(&sum_sq_witness)?;

            // M * (sum_sq + M') = (shifted << k) + R_shift
            let shifted_var = FpVar::new_witness(cs.clone(), || Ok(bundle.shifted_val[b][s][0]))?;
            let r_shift_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_shift[b][s][0]))?;
            let lhs_ms = &m_ms_var * (&sum_sq_circuit + &m_prime_var);
            let rhs_ms = (&shifted_var * &two_k) + &r_shift_var;
            lhs_ms.enforce_equal(&rhs_ms)?;

            // r_pool.push(r_shift_var.clone()); // 추가
            let (s_low, s_high) = split_remainder_and_push(cs.clone(), &r_shift_var, bundle.r_shift[b][s][0])?;
            shift_low_vars.push(s_low);
            shift_high_vars.push(s_high);

            // shifted = (Q_ms * dim) + R_ms
            let q_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ms_sim[b][s][0]))?;

            rms_pool.push(q_ms_var.clone()); // 추가
            
            let r_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_ms[b][s][0]))?;
            shifted_var.enforce_equal(&((&q_ms_var * &dim_var) + &r_ms_var))?;

            // range_check: r_ms_var < dim (bit_limit is loop-invariant, hoisted to caller)
            let r_bits = r_ms_var.to_bits_le()?;
            for bit in r_bits.iter().skip(bit_limit) {
                bit.enforce_equal(&Boolean::constant(false))?;
            }

            let inv_rms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.inv_rms[b][s][0]))?;
            let mut seq_q_out_vars = Vec::new();
            
            for d in 0..bundle.q_ri[b][s].len() {
                // Reuse the linear combination computed in the first loop — no new witness needed.
                let q_diff_var = q_diff_vars[d].clone();

                // (Q_diff * inv_rms * Q_gamma_diff)
                // gamma_diffs[d] = (gamma[d] - z_gamma) is precomputed above.
                let product = &q_diff_var * &inv_rms_var * &gamma_diffs[d];
                
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out[b][s][d]))?;

                // r_pool.push(r_out_var.clone()); // 추가
                let (o_low, o_high) = split_remainder_and_push(cs.clone(), &r_out_var, bundle.r_out[b][s][d])?;
                out_low_vars.push(o_low);
                out_high_vars.push(o_high);

                // M_norm * (product + M_prime_norm) = (Q_out << k) + R_out
                let lhs = &m_norm_var * (product + &m_prime_norm_var);
                let rhs = (&q_out_var * &two_k) + &r_out_var;
                lhs.enforce_equal(&rhs)?;
                
                seq_q_out_vars.push(q_out_var);
            }
            batch_q_out_vars.push(seq_q_out_vars);
        }
        all_q_out_vars.push(batch_q_out_vars);
    }
    
    // 루프가 끝난 후 Lasso의 기대 순서(dim[0], dim[1])에 맞춰 r_pool을 채움
    // Matrix 1 (R_shift): Lows -> Highs
    r_pool.extend(shift_low_vars);
    r_pool.extend(shift_high_vars);

    // Matrix 2 (R_out): Lows -> Highs
    r_pool.extend(out_low_vars);
    r_pool.extend(out_high_vars);

    Ok(all_q_out_vars)
}

// ── Phase 5-B B4-2: enforce_layernorm 가젯 신규 작성 ────────────────────────
//
// Python `bert_normalization_pipeline.py:_process_layernorm_block` (라인 15-124)
// 의 R1CS 식을 정확히 모방. Python 측 검증 결과 3072/3072 cells exact (작업 4E).
//
// `enforce_rmsnorm` (위) 와 평행한 패턴 — 차이만 추가:
//   - Step A (mean):       sum_x = Σ Q_ri,  Q_mean × dim + R_mean = sum_x   ← 신규
//   - Step B (diff):       Q_diff = Q_ri − Q_mean (witness 차감)             ← RMSNorm 의 'Q_ri − z_ri' 와 형태 변경
//   - Step C (variance):   M_var × (sum_sq + M'_var) = ... + R_shift_var     ← 동일
//   - Step D (lookup):     inv_std (Lasso 외부)                               ← 동일
//   - Step E (β lifting):  M_beta_lift × (Q_beta − z_beta) = ... + R_beta_lift  ← 신규
//   - Step F (final + β):  M_norm × (inner + Q_beta_lifted_d + M'_norm) = ... ← β 항 추가
//
// 매핑 reference: docs/migration/21-b4-layernorm-rcs-analysis.md
//
// r_pool 순서 (Lasso lookup 측이 기대) — 3 그룹 (B5-1, option A 채택):
//   1. R_shift_var (low → high)    — Step C
//   2. R_out (low → high)          — Step F
//   3. R_beta_lift (low → high)    — Step E
//
// Note: R_mean 은 Step A 등식 (Q_mean × dim + R_mean = sum_x) + to_bits_le
// (< 2^bit_limit) 만으로 R_mean < dim 근사 보장. RMSNorm 의 R_ms 와 동일하게
// Lasso 22-bit pool 에는 포함하지 않음 (over-kill 회피, CP-Link mismatch 방지).
//
// 본 r_pool 순서가 mlp_lookup.rs::run_mlp_remainders_layernorm 와 정합해야 함.
// RMSNorm 측 (shift + out) 의 패턴 + β lifting 추가.
pub fn enforce_layernorm<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    bundle: &LayerNormBundle<F>,
    dim: usize,
    frac_bits: usize,
    k_bits: usize,
    r_pool: &mut Vec<FpVar<F>>,
    ln_pool: &mut Vec<FpVar<F>>,
) -> Result<Vec<Vec<Vec<FpVar<F>>>>, SynthesisError> {
    // ── Step 0: 상수 alloc ──────────────────────────────────────────────
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;
    let dim_var = FpVar::new_constant(cs.clone(), F::from(dim as u128))?;

    // M_var, M_prime_var, M_norm, M_prime_norm, M_beta_lift — Python 측 상수
    let m_var_var = FpVar::new_constant(cs.clone(), bundle.m_var)?;
    let m_prime_var_var = FpVar::new_constant(cs.clone(), bundle.m_prime_var)?;
    let m_norm_var = FpVar::new_constant(cs.clone(), bundle.m_norm)?;
    let m_prime_norm_var = FpVar::new_constant(cs.clone(), bundle.m_prime_norm)?;
    let m_beta_lift_var = FpVar::new_constant(cs.clone(), bundle.m_beta_lift)?;

    // γ alloc + (γ - z_gamma) precompute (loop-invariant)
    let z_gamma_var = FpVar::new_constant(cs.clone(), bundle.z_gamma)?;
    let gamma_vars = bundle
        .q_gamma
        .iter()
        .map(|g| FpVar::new_witness(cs.clone(), || Ok(*g)))
        .collect::<Result<Vec<_>, _>>()?;
    let gamma_diffs: Vec<FpVar<F>> = gamma_vars.iter().map(|g| g - &z_gamma_var).collect();

    // β alloc + (β - z_beta) precompute (Step E 사용)
    let z_beta_var = FpVar::new_constant(cs.clone(), bundle.z_beta)?;
    let beta_vars = bundle
        .q_beta
        .iter()
        .map(|b| FpVar::new_witness(cs.clone(), || Ok(*b)))
        .collect::<Result<Vec<_>, _>>()?;
    let beta_diffs: Vec<FpVar<F>> = beta_vars.iter().map(|b| b - &z_beta_var).collect();

    // 출력 + remainder pool 누적기 (3 그룹 × low/high — option A: R_mean 제외)
    let mut all_q_out_vars: Vec<Vec<Vec<FpVar<F>>>> = Vec::new();

    let mut shift_low_vars: Vec<FpVar<F>> = Vec::new();
    let mut shift_high_vars: Vec<FpVar<F>> = Vec::new();
    let mut out_low_vars: Vec<FpVar<F>> = Vec::new();
    let mut out_high_vars: Vec<FpVar<F>> = Vec::new();
    let mut beta_low_vars: Vec<FpVar<F>> = Vec::new();
    let mut beta_high_vars: Vec<FpVar<F>> = Vec::new();

    let bit_limit = (dim as f64).log2().ceil() as usize;

    // ── Step E (loop-invariant β lifting, H 회) ─────────────────────────
    // Python: line 68-73 (numerator_beta = M_beta_lift * Q_beta_diff)
    // 식: M_beta_lift × (Q_beta_h − z_beta) = (Q_beta_lifted_h << k) + R_beta_lift_h
    let q_beta_lifted_vars: Vec<FpVar<F>> = bundle
        .q_beta_lifted
        .iter()
        .enumerate()
        .map(|(h, &val)| {
            let lifted_var = FpVar::new_witness(cs.clone(), || Ok(val))?;
            let r_beta_h = FpVar::new_witness(cs.clone(), || Ok(bundle.r_beta_lift[h]))?;

            let lhs_beta = &m_beta_lift_var * &beta_diffs[h];
            let rhs_beta = (&lifted_var * &two_k) + &r_beta_h;
            lhs_beta.enforce_equal(&rhs_beta)?;

            // R_beta_lift range check (split low/high → r_pool)
            let (lo, hi) = split_remainder_and_push(cs.clone(), &r_beta_h, bundle.r_beta_lift[h])?;
            beta_low_vars.push(lo);
            beta_high_vars.push(hi);

            Ok(lifted_var)
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    // ── 메인 루프 (B × S) ─────────────────────────────────────────────
    for b in 0..bundle.q_ri.len() {
        let mut batch_q_out_vars: Vec<Vec<FpVar<F>>> = Vec::new();
        for s in 0..bundle.q_ri[b].len() {
            // ── Step A: mean (Python line 26-28) ──────────────────────────
            // sum_x = Σ Q_ri_d  (linear combination, 무비용)
            // Q_mean × dim + R_mean = sum_x
            let q_ri_vars: Vec<FpVar<F>> = bundle.q_ri[b][s]
                .iter()
                .map(|&v| FpVar::new_witness(cs.clone(), || Ok(v)))
                .collect::<Result<Vec<_>, _>>()?;

            let mut sum_x_circuit = FpVar::zero();
            for q in &q_ri_vars {
                sum_x_circuit += q;
            }

            let q_mean_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_mean[b][s][0]))?;
            let r_mean_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_mean[b][s][0]))?;
            sum_x_circuit.enforce_equal(&((&q_mean_var * &dim_var) + &r_mean_var))?;

            // R_mean range check (< 2^bit_limit ≈ dim, 경량 R1CS-only).
            // option A: 22-bit Lasso pool 미포함 (소스 < dim 이므로 over-kill).
            // 등식 (Q_mean × dim + R_mean = sum_x) + 본 to_bits_le 로 R_mean 의 자유도 제한.
            let r_mean_bits = r_mean_var.to_bits_le()?;
            for bit in r_mean_bits.iter().skip(bit_limit) {
                bit.enforce_equal(&Boolean::constant(false))?;
            }

            // ── Step B: Q_diff = Q_ri − Q_mean (Python line 31) + Step C 의 sum_sq accumulator ──
            // RMSNorm 의 'Q_ri - z_ri' 와 형태 변경: Q_mean 이 witness (constant 아님)
            let mut q_diff_vars: Vec<FpVar<F>> = Vec::new();
            let mut sum_sq_circuit = FpVar::zero();
            for d in 0..bundle.q_ri[b][s].len() {
                let diff = &q_ri_vars[d] - &q_mean_var;
                sum_sq_circuit += diff.square()?;
                q_diff_vars.push(diff);
            }

            // ── Step C: variance shift + division (Python line 38-45) ─────
            // M_var × (sum_sq + M'_var) = (shifted_val_var << k) + R_shift_var
            let shifted_var =
                FpVar::new_witness(cs.clone(), || Ok(bundle.shifted_val_var[b][s][0]))?;
            let r_shift_var =
                FpVar::new_witness(cs.clone(), || Ok(bundle.r_shift_var[b][s][0]))?;
            let lhs_var = &m_var_var * (&sum_sq_circuit + &m_prime_var_var);
            let rhs_var = (&shifted_var * &two_k) + &r_shift_var;
            lhs_var.enforce_equal(&rhs_var)?;

            let (sh_low, sh_high) = split_remainder_and_push(
                cs.clone(),
                &r_shift_var,
                bundle.r_shift_var[b][s][0],
            )?;
            shift_low_vars.push(sh_low);
            shift_high_vars.push(sh_high);

            // shifted_val_var = Q_var_sim × dim + R_var
            let q_var_sim_var =
                FpVar::new_witness(cs.clone(), || Ok(bundle.q_var_sim[b][s][0]))?;
            ln_pool.push(q_var_sim_var.clone()); // Lasso 측 인덱스로 사용

            let r_var_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_var[b][s][0]))?;
            shifted_var.enforce_equal(&((&q_var_sim_var * &dim_var) + &r_var_var))?;

            // R_var range check (< dim)
            let r_var_bits = r_var_var.to_bits_le()?;
            for bit in r_var_bits.iter().skip(bit_limit) {
                bit.enforce_equal(&Boolean::constant(false))?;
            }

            // ── Step D: inv_std lookup 결과 witness alloc (Lasso 외부 검증) ──
            let inv_std_var = FpVar::new_witness(cs.clone(), || Ok(bundle.inv_std[b][s][0]))?;

            // ── Step F: 최종 rescale (Python line 76-85, β 추가) ──────────
            // inner_d = Q_diff_d × inv_std × (Q_gamma_d − z_gamma)
            // M_norm × (inner_d + Q_beta_lifted_d + M'_norm) = (Q_out_d << k) + R_out_d
            let mut seq_q_out_vars: Vec<FpVar<F>> = Vec::new();
            for d in 0..bundle.q_ri[b][s].len() {
                let q_diff_var = q_diff_vars[d].clone();
                let inner = &q_diff_var * &inv_std_var * &gamma_diffs[d];

                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out[b][s][d]))?;

                let (o_low, o_high) =
                    split_remainder_and_push(cs.clone(), &r_out_var, bundle.r_out[b][s][d])?;
                out_low_vars.push(o_low);
                out_high_vars.push(o_high);

                // β broadcast: Q_beta_lifted_d (per dim, shared across (b, s))
                let lhs = &m_norm_var
                    * (inner + &q_beta_lifted_vars[d] + &m_prime_norm_var);
                let rhs = (&q_out_var * &two_k) + &r_out_var;
                lhs.enforce_equal(&rhs)?;

                seq_q_out_vars.push(q_out_var);
            }
            batch_q_out_vars.push(seq_q_out_vars);
        }
        all_q_out_vars.push(batch_q_out_vars);
    }

    // ── r_pool 채움 (Lasso 측 기대 순서: shift → out → beta) — 3 그룹 ─────
    // Matrix 1 (R_shift_var):  Lows → Highs
    r_pool.extend(shift_low_vars);
    r_pool.extend(shift_high_vars);

    // Matrix 2 (R_out):        Lows → Highs
    r_pool.extend(out_low_vars);
    r_pool.extend(out_high_vars);

    // Matrix 3 (R_beta_lift):  Lows → Highs
    r_pool.extend(beta_low_vars);
    r_pool.extend(beta_high_vars);

    Ok(all_q_out_vars)
}