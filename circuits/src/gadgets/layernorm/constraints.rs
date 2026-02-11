use ark_ff::PrimeField;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_r1cs_std::alloc::{AllocVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

#[derive(Clone)]
pub struct NormBundle<F: PrimeField> {
    pub q_ri: Vec<Vec<Vec<F>>>,       // [B][S][D]
    pub z_ri: F,
    pub sum_sq: Vec<Vec<Vec<F>>>,     // [B][S][1] - 수정됨
    pub m_ms: F,
    pub m_prime_ms: F,
    pub shifted_val: Vec<Vec<Vec<F>>>,// [B][S][1] - 수정됨
    pub r_shift: Vec<Vec<Vec<F>>>,    // [B][S][1] - 수정됨
    pub q_ms_sim: Vec<Vec<Vec<F>>>,   // [B][S][1] - 수정됨
    pub r_ms: Vec<Vec<Vec<F>>>,       // [B][S][1] - 수정됨
    pub inv_rms: Vec<Vec<Vec<F>>>,    // [B][S][1] - 수정됨
    pub q_gamma: Vec<F>,              // [D]
    pub q_out: Vec<Vec<Vec<F>>>,      // [B][S][D]
    pub r_out: Vec<Vec<Vec<F>>>,      // [B][S][D]
    pub weight_bits: usize,
}

pub fn enforce_rmsnorm<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    bundle: &NormBundle<F>,
    dim: usize,
    frac_bits :usize,
    k_bits:usize
) -> Result<(), SynthesisError> {
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;
    let two_total = FpVar::new_constant(cs.clone(), F::from(1u128 << (frac_bits + bundle.weight_bits)))?;
    let dim_var = FpVar::new_constant(cs.clone(), F::from(dim as u128))?;
    let z_ri_var = FpVar::new_constant(cs.clone(), bundle.z_ri)?;
    let m_ms_var = FpVar::new_constant(cs.clone(), bundle.m_ms)?;
    let m_prime_var = FpVar::new_constant(cs.clone(), bundle.m_prime_ms)?;
    let gamma_vars = bundle.q_gamma.iter().map(|g| FpVar::new_witness(cs.clone(), || Ok(*g))).collect::<Result<Vec<_>, _>>()?;

    for b in 0..bundle.q_ri.len() {
        for s in 0..bundle.q_ri[b].len() {
            // 1. MS(Mean Square) 입력 준비: sum_sq = sum((Q_ri - z_ri)^2)
            let mut sum_sq_circuit = FpVar::zero();
            let mut q_diff_vars = Vec::new();
            for d in 0..bundle.q_ri[b][s].len() {
                let q_ri_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ri[b][s][d]))?;
                let diff = q_ri_var - &z_ri_var;
                sum_sq_circuit += diff.square()?;
                q_diff_vars.push(diff);
            }
            let sum_sq_witness = FpVar::new_witness(cs.clone(), || Ok(bundle.sum_sq[b][s][0]))?;
            sum_sq_circuit.enforce_equal(&sum_sq_witness)?;

            // 2. MS 시프트 검증: M * (sum_sq + M') = (shifted << k) + R_shift
            let shifted_var = FpVar::new_witness(cs.clone(), || Ok(bundle.shifted_val[b][s][0]))?;
            let r_shift_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_shift[b][s][0]))?;
            let lhs_ms = &m_ms_var * (&sum_sq_witness + &m_prime_var);
            let rhs_ms = (&shifted_var * &two_k) + &r_shift_var;
            lhs_ms.enforce_equal(&rhs_ms)?;

            // 3. 차원 나눗셈 검증: shifted = (Q_ms * dim) + R_ms
            let q_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ms_sim[b][s][0]))?;
            let r_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_ms[b][s][0]))?;
            shifted_var.enforce_equal(&((&q_ms_var * &dim_var) + &r_ms_var))?;

            // 4. 최종 출력 검증 (Weight 포함)
            // (Q_diff * inv_rms * Q_gamma) = (Q_out << (frac + weight)) + R_out
            let inv_rms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.inv_rms[b][s][0]))?;
            for d in 0..q_diff_vars.len() {
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out[b][s][d]))?;
                
                let lhs_final = &q_diff_vars[d] * &inv_rms_var * &gamma_vars[d];
                let rhs_final = (&q_out_var * &two_total) + &r_out_var;
                lhs_final.enforce_equal(&rhs_final)?;
            }
        }
    }
    Ok(())
}