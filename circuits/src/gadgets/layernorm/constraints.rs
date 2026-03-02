use ark_ff::PrimeField;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_r1cs_std::alloc::{AllocVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::prelude::*;
use ark_r1cs_std::prelude::Boolean;
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
    let m_norm_var = FpVar::new_constant(cs.clone(), bundle.m_norm)?;
    let m_prime_norm_var = FpVar::new_constant(cs.clone(), bundle.m_prime_norm)?;
    let mut all_q_out_vars = Vec::new();
    
    
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

            r_pool.push(r_shift_var.clone()); // 추가

            // shifted = (Q_ms * dim) + R_ms
            let q_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ms_sim[b][s][0]))?;

            rms_pool.push(q_ms_var.clone()); // 추가
            
            let r_ms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_ms[b][s][0]))?;
            shifted_var.enforce_equal(&((&q_ms_var * &dim_var) + &r_ms_var))?;

            // range_check: r_ms_var < dim
            let bit_limit = (dim as f64).log2().ceil() as usize; 
            let r_bits = r_ms_var.to_bits_le()?; 
            for bit in r_bits.iter().skip(bit_limit) {
                bit.enforce_equal(&Boolean::constant(false))?;
            }

            let inv_rms_var = FpVar::new_witness(cs.clone(), || Ok(bundle.inv_rms[b][s][0]))?;
            let mut seq_q_out_vars = Vec::new();
            
            for d in 0..bundle.q_ri[b][s].len() {
                let q_diff_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_ri[b][s][d] - bundle.z_ri))?;
                let q_gamma_diff = &gamma_vars[d] - &z_gamma_var;
                
                // (Q_diff * inv_rms * Q_gamma_diff)
                let product = &q_diff_var * &inv_rms_var * &q_gamma_diff;
                
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out[b][s][d]))?;

                r_pool.push(r_out_var.clone()); // 추가

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
    
    Ok(all_q_out_vars)
}