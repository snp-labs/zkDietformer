use ark_ff::PrimeField;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_relations::r1cs::SynthesisError;
use ark_relations::r1cs::ConstraintSystemRef;
use super::{ActivationFunction, ActivationGadget, constraints};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
#[derive(Clone)]
pub struct GeLUBundle<F: PrimeField> {
    pub z_in: F,                  // Linear1의 출력 제로포인트 (GeLU의 입력 영점)
    pub sigmoid_table_val: Vec<Vec<Vec<F>>>, // [B][S][D] 테이블 조회 결과
    pub acc_gelu: Vec<Vec<Vec<F>>>,          // (Q_in - z_in) * sigmoid_val
    pub m_gelu: F,
    pub m_prime_gelu: F,
    pub q_out_sim: Vec<Vec<Vec<F>>>,         // 최종 GeLU 출력 정수
    pub r_out_gelu: Vec<Vec<Vec<F>>>,        // 최종 GeLU 시프트 나머지
}

pub fn enforce_gelu<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_in_vars: &Vec<Vec<Vec<FpVar<F>>>>,  // Linear1 출력 Wire
    r_in_vars: &Vec<Vec<Vec<FpVar<F>>>>,  // Linear1 나머지 Wire (고정밀 인덱스용)
    bundle: &GeLUBundle<F>,
    k_bits: usize,
    frac_bits: usize,
) -> Result<Vec<Vec<Vec<FpVar<F>>>>, SynthesisError> {
    let z_in_var = FpVar::new_constant(cs.clone(), bundle.z_in)?;
    let m_var = FpVar::new_constant(cs.clone(), bundle.m_gelu)?;
    let m_prime_var = FpVar::new_constant(cs.clone(), bundle.m_prime_gelu)?;
    let two_k = FpVar::new_constant(cs.clone(), F::from(1u128 << k_bits))?;
    let two_frac = FpVar::new_constant(cs.clone(), F::from(1u128 << frac_bits))?;

    let mut all_out_vars = Vec::new();

    for b in 0..q_in_vars.len() {
        let mut batch_vars = Vec::new();
        for s in 0..q_in_vars[b].len() {
            let mut seq_vars = Vec::new();
            for d in 0..q_in_vars[b][s].len() {
                // 1. 고정밀 인덱스 생성 (Lasso 증명부와 일관성 유지)
                // index = (Q_in << k) + R_in
                let _index_var = (&q_in_vars[b][s][d] * &two_k) + &r_in_vars[b][s][d];

                // 2. 테이블 조회 결과 Witness 할당 (실제 Lasso 증명은 별도 수행)
                let sigmoid_val_var = FpVar::new_witness(cs.clone(), || Ok(bundle.sigmoid_table_val[b][s][d]))?;

                // 3. GeLU 알맹이 계산: acc = (Q_in - z_in) * sigmoid_val
                let q_in_diff = &q_in_vars[b][s][d] - &z_in_var;
                let acc_var = &q_in_diff * &sigmoid_val_var;
                
                // 파이썬 acc_gelu와 일치하는지 중간 검증
                let acc_witness = FpVar::new_witness(cs.clone(), || Ok(bundle.acc_gelu[b][s][d]))?;
                acc_var.enforce_equal(&acc_witness)?;

                // 4. Rescaling: numerator = M * (acc + M_prime)
                let numerator = &m_var * (&acc_witness + &m_prime_var);

                // 5. 최종 결과 검증: numerator = (Q_out << k) + R_out
                let q_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.q_out_sim[b][s][d]))?;
                let r_out_var = FpVar::new_witness(cs.clone(), || Ok(bundle.r_out_gelu[b][s][d]))?;
                
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