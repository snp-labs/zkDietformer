use ark_ff::PrimeField;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_relations::r1cs::SynthesisError;
use ark_relations::r1cs::ConstraintSystemRef;
use super::{ActivationFunction, ActivationGadget, constraints};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
#[derive(Clone)]
pub struct GeLUBundle<F: PrimeField> {
    pub z_in: F,                  
    pub sigmoid_table_val: Vec<Vec<Vec<F>>>, 
    // pub acc_gelu: Vec<Vec<Vec<F>>>,          // (Q_in - z_in) * sigmoid_val
    pub m_gelu: F,
    pub m_prime_gelu: F,
    pub q_out_sim: Vec<Vec<Vec<F>>>,         
    pub r_out_gelu: Vec<Vec<Vec<F>>>,        
    pub m_gelu_idx: F,
}


pub fn enforce_gelu<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_in_vars: &Vec<Vec<Vec<FpVar<F>>>>, // Linear1 출력
    r_in_vars: &Vec<Vec<Vec<FpVar<F>>>>, // Linear1 나머지
    bundle: &GeLUBundle<F>,
    k_bits: usize,
    committed_vars: &mut Vec<FpVar<F>>,
) -> Result<Vec<Vec<Vec<FpVar<F>>>>, SynthesisError> {
    let z_in_var = FpVar::new_constant(cs.clone(), bundle.z_in)?;
    let m_var = FpVar::new_constant(cs.clone(), bundle.m_gelu)?;
    let m_prime_var = FpVar::new_constant(cs.clone(), bundle.m_prime_gelu)?;
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;

    let mut all_out_vars = Vec::new();

    for b in 0..q_in_vars.len() {
        let mut batch_vars = Vec::new();
        for s in 0..q_in_vars[b].len() {
            let mut seq_vars = Vec::new();
            for d in 0..q_in_vars[b][s].len() {
                let sigmoid_val_var = FpVar::new_witness(cs.clone(), || Ok(bundle.sigmoid_table_val[b][s][d]))?;

                // acc = (Q_in - z_in) * sigmoid_val
                let q_in_diff = &q_in_vars[b][s][d] - &z_in_var;
                let acc_var = &q_in_diff * &sigmoid_val_var;
                
                // let acc_witness = FpVar::new_witness(cs.clone(), || Ok(bundle.acc_gelu[b][s][d]))?;
                // acc_var.enforce_equal(&acc_witness)?;

                // Rescaling: numerator = M * (acc + M_prime)
                let numerator = &m_var * (&acc_var + &m_prime_var);

                // numerator = (Q_out << k) + R_out
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out_sim[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out_gelu[b][s][d]))?;
                
                committed_vars.push(r_out_var.clone()); // 추가

                let rhs = (&q_out_var * &two_k) + &r_out_var;
                numerator.enforce_equal(&rhs)?;

                seq_vars.push(q_out_var);
            }
            batch_vars.push(seq_vars);
        }
        all_out_vars.push(batch_vars);
    }
    Ok(all_out_vars)
}