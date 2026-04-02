use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

use crate::gadgets::activation::gelu::split_remainder_and_push;

#[derive(Clone)]
pub struct LinearBundle<F: PrimeField> {
    pub z_in: F,
    pub q_w_diff: Vec<Vec<F>>,       // [Out_D][In_D] (Q_w - z_w)
    pub q_b_diff: Vec<F>,            // [Out_D] (Q_b - z_b)
    pub m_lin: F,
    pub m_prime_lin: F,
    pub q_out_sim: Vec<Vec<Vec<F>>>, // [B][S][Out_D]
    pub r_out_lin: Vec<Vec<Vec<F>>>, // [B][S][Out_D]
}

pub fn enforce_linear<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_in_vars: &Vec<Vec<Vec<FpVar<F>>>>, // [B][S][In_D]
    bundle: &LinearBundle<F>,
    k_bits: usize,
    r_pool: &mut Vec<FpVar<F>>,
) -> Result<(Vec<Vec<Vec<FpVar<F>>>>, Vec<Vec<Vec<FpVar<F>>>>), SynthesisError> { // 튜플 반환으로 수정
    let z_in_var = FpVar::new_constant(cs.clone(), bundle.z_in)?;
    let m_var = FpVar::new_constant(cs.clone(), bundle.m_lin)?;
    let m_prime_var = FpVar::new_constant(cs.clone(), bundle.m_prime_lin)?;
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;

    let b_dim = q_in_vars.len();
    let s_dim = if b_dim > 0 { q_in_vars[0].len() } else { 0 };
    let out_dim = bundle.q_w_diff.len();

    let mut all_q_vars = Vec::with_capacity(b_dim);
    let mut all_r_vars = Vec::with_capacity(b_dim);

    let mut current_low_vars = Vec::with_capacity(b_dim * s_dim * out_dim);
    let mut current_high_vars = Vec::with_capacity(b_dim * s_dim * out_dim);

    for b in 0..b_dim {
        let mut batch_q = Vec::with_capacity(s_dim);
        let mut batch_r = Vec::with_capacity(s_dim);
        for s in 0..q_in_vars[b].len() {
            let mut seq_q = Vec::with_capacity(out_dim);
            let mut seq_r = Vec::with_capacity(out_dim);
            
            // 1. 입력 Zero-centering: (Q_in - z_in)
            let q_in_diff: Vec<FpVar<F>> = q_in_vars[b][s].iter()
                .map(|q| q - &z_in_var)
                .collect();

            for out_d in 0..bundle.q_w_diff.len() {
                // 2. MatMul + Bias: sum((Q_in-z_in)*(Q_w-z_w)) + (Q_b-z_b)
                let mut acc = FpVar::new_constant(cs.clone(), bundle.q_b_diff[out_d])?;
                for in_d in 0..q_in_diff.len() {
                    let w_diff_var = FpVar::new_constant(cs.clone(), bundle.q_w_diff[out_d][in_d])?;
                    acc += &q_in_diff[in_d] * &w_diff_var;
                }

                // 3. Rescaling: numerator = M * (acc + M_prime)
                let numerator = &m_var * (&acc + &m_prime_var);

                // 4. Result Witness & Equality: numerator = (Q_out << k) + R_out
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out_sim[b][s][out_d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out_lin[b][s][out_d]))?;
                
                // r_pool.push(r_out_var.clone()); // 추가
                let (low, high) = split_remainder_and_push(cs.clone(), &r_out_var, bundle.r_out_lin[b][s][out_d])?;
                current_low_vars.push(low);
                current_high_vars.push(high);

                let rhs = (&q_out_var * &two_k) + &r_out_var;
                numerator.enforce_equal(&rhs)?;

                seq_q.push(q_out_var);
                seq_r.push(r_out_var); 
            }
            batch_q.push(seq_q);
            batch_r.push(seq_r);
        }
        all_q_vars.push(batch_q);
        all_r_vars.push(batch_r);
    }

    r_pool.extend(current_low_vars);
    r_pool.extend(current_high_vars);
    
    Ok((all_q_vars, all_r_vars)) 
}