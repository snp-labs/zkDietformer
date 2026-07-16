use ark_ff::PrimeField;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_relations::r1cs::SynthesisError;
use ark_relations::r1cs::ConstraintSystemRef;
use super::{ActivationFunction, ActivationGadget, constraints};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, R1CSVar};
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

pub fn split_remainder_and_push<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    r_out_var: &FpVar<F>, 
    r_val_raw: F,        
) -> Result<(FpVar<F>, FpVar<F>), SynthesisError> {
    let r_u64 = r_val_raw.into_bigint().as_ref()[0];
    let r_low_val = r_u64 & 0xFFFF; // 16 bits for low part
    let r_high_val = (r_u64 >> 16) & 0x3FF; // 10 bits for high part (26-bit total)

    let r_low_var = FpVar::new_witness(cs.clone(), || Ok(F::from(r_low_val)))?;
    let r_high_var = FpVar::new_witness(cs.clone(), || Ok(F::from(r_high_val)))?;

    let shift_const = F::from(1u128 << 16);
    r_out_var.enforce_equal(&(&r_high_var * shift_const + &r_low_var))?;
    Ok((r_low_var, r_high_var))
}

pub fn enforce_gelu<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_in_vars: &Vec<Vec<Vec<FpVar<F>>>>, // Linear1 출력 (Q_in)
    r_in_vars: &Vec<Vec<Vec<FpVar<F>>>>, // Linear1 나머지 (R_in)
    bundle: &GeLUBundle<F>,
    k_bits: usize,
    r_pool: &mut Vec<FpVar<F>>,
    gelu_pool: &mut Vec<FpVar<F>>,        // CP-Link용 인덱스 수집 풀
) -> Result<Vec<Vec<Vec<FpVar<F>>>>, SynthesisError> {
    let z_in_var = FpVar::new_constant(cs.clone(), bundle.z_in)?;
    let m_var = FpVar::new_constant(cs.clone(), bundle.m_gelu)?;
    let m_prime_var = FpVar::new_constant(cs.clone(), bundle.m_prime_gelu)?;
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;
    
    // Lasso와 동일한 상수 설정
    let offset = 32768i128;
    let gelu_m_limit = 65536i128;

    // Hoist loop-invariant BigInt conversions — these are constant across all (b, s, d).
    let z_in_i128 = bundle.z_in.into_bigint().as_ref()[0] as i128;
    let m_gelu_idx_u128 = bundle.m_gelu_idx.into_bigint().as_ref()[0] as u128;

    let mut all_out_vars = Vec::new();
    let mut current_low_vars = Vec::new();
    let mut current_high_vars = Vec::new();

    for b in 0..q_in_vars.len() {
        let mut batch_vars = Vec::new();
        for s in 0..q_in_vars[b].len() {
            let mut seq_vars = Vec::new();
            for d in 0..q_in_vars[b][s].len() {
                // 1. GeLU 연산 제약
                let sigmoid_val_var = FpVar::new_witness(cs.clone(), || Ok(bundle.sigmoid_table_val[b][s][d]))?;
                // acc = (Q_in - z_in) * sigmoid_val
                let q_in_diff = &q_in_vars[b][s][d] - &z_in_var;
                let acc_var = &q_in_diff * &sigmoid_val_var;
                // Rescaling: numerator = M * (acc + M_prime)
                let numerator = &m_var * (&acc_var + &m_prime_var);

                // numerator = (Q_out << k) + R_out
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out_sim[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out_gelu[b][s][d]))?;

                let (low, high) = split_remainder_and_push(cs.clone(), &r_out_var, bundle.r_out_gelu[b][s][d])?;
                current_low_vars.push(low);
                current_high_vars.push(high);

                let rhs = (&q_out_var * &two_k) + &r_out_var;
                numerator.enforce_equal(&rhs)?;

                seq_vars.push(q_out_var);

                // 2. CP-Link를 위한 Lasso 인덱스 생성 및 수집
                let idx_var = FpVar::new_witness(cs.clone(), || {
                    let q_in_val = q_in_vars[b][s][d].value()?.into_bigint().as_ref()[0] as i128;
                    let r_in_val = r_in_vars[b][s][d].value()?.into_bigint().as_ref()[0] as i128;

                    let diff = q_in_val - z_in_i128;
                    let x_diff_full = (diff << k_bits) + r_in_val;
                    let scaled_diff = (x_diff_full * m_gelu_idx_u128 as i128) >> k_bits;
                    let mut idx = scaled_diff + offset;

                    // Clipping
                    if idx < 0 { idx = 0; }
                    if idx >= gelu_m_limit { idx = gelu_m_limit - 1; }

                    Ok(F::from(idx as u64))
                })?;

                gelu_pool.push(idx_var);
            }
            batch_vars.push(seq_vars);
        }
        all_out_vars.push(batch_vars);
    }

    r_pool.extend(current_low_vars);  // 모든 Low 데이터 (dim[0] 매칭)
    r_pool.extend(current_high_vars); // 모든 High 데이터 (dim[1] 매칭)
    Ok(all_out_vars)
}