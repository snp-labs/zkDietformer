use anyhow::Context;
use ark_ff::PrimeField;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_r1cs_std::alloc::AllocVar;
use ark_relations::r1cs::SynthesisError;
use ark_relations::r1cs::ConstraintSystemRef;
use super::{ActivationFunction, ActivationGadget, constraints};

#[derive(Clone)]
pub struct GeluBundle<F: PrimeField> {
    pub inputs_q: Vec<F>,
    pub limbs: Vec<Vec<F>>, // 림브로 분해된 입력 값들
    pub exp_limb_vals: Vec<Vec<F>>,
    pub sigmoid_outputs: Vec<F>,
    pub sigmoid_remainders: Vec<F>, // 추가적인 시그모이드 잔여값
    pub gelu_outputs: Vec<F>,
    pub gelu_remainders: Vec<F>,
    pub input_0: F,
    pub scale_input: F,
    pub frac_bits: u32,
    pub multiplier: F,
}

pub fn enforce_gelu<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    bundle: &GeluBundle<F>,
) -> Result<(), SynthesisError> {
    let n = bundle.inputs_q.len();
    let f = bundle.frac_bits;
    let k_segments = bundle.limbs[0].len() as u32;

    // 1. 스케일 상수 계산
    // prod_scale = (2^f)^k = 2^(f*k)
    let prod_scale = F::from(2u64).pow(&[(f * k_segments) as u64]);
    // target_scale = (2^f)^(k+1) = 2^(f*(k+1))
    // 이는 sig_q(2^f) * (prod_scale + exp_prod)의 결과와 일치해야 함
    let target_scale = F::from(2u64).pow(&[(f * (k_segments + 1)) as u64]);

    let scale_input_var = FpVar::Constant(bundle.scale_input);
    let input_0_var = FpVar::Constant(bundle.input_0);

    for i in 0..n {
        // 2. 변수 할당 (Witness Allocation)
        let x_q_var = FpVar::new_witness(cs.clone(), || Ok(bundle.inputs_q[i]))?;
        let sig_var = FpVar::new_witness(cs.clone(), || Ok(bundle.sigmoid_outputs[i]))?;
        let sig_rem_var = FpVar::new_witness(cs.clone(), || Ok(bundle.sigmoid_remainders[i]))?; // 나머지 할당
        let y_q_var = FpVar::new_witness(cs.clone(), || Ok(bundle.gelu_outputs[i]))?;
        let y_rem_var = FpVar::new_witness(cs.clone(), || Ok(bundle.gelu_remainders[i]))?;

        // 3. Limb 재조립 검증 (입력 무결성 체크)
        // x_q = sum(limb_k * (2^limb_bits)^k)
        let mut recomposed = FpVar::zero();
        let mut mul_const = F::one();
        let limb_base = F::from(1u64 << crate::config::LIMB_BITS);
        
        for k_idx in 0..bundle.limbs[i].len() {
            let l_v = FpVar::new_witness(cs.clone(), || Ok(bundle.limbs[i][k_idx]))?;
            recomposed += l_v * FpVar::Constant(mul_const);
            mul_const *= limb_base;
        }
        recomposed.enforce_equal(&x_q_var)?;

        // 4. 지수 곱 계산 (누적 스케일 반영)
        // exp_prod = exp_limb_0 * exp_limb_1 * ... * exp_limb_k
        let mut exp_prod_var = FpVar::Constant(F::one());
        for k_idx in 0..bundle.exp_limb_vals[i].len() {
            let e_v = FpVar::new_witness(cs.clone(), || Ok(bundle.exp_limb_vals[i][k_idx]))?;
            if k_idx == 0 {
                exp_prod_var = e_v;
            } else {
                exp_prod_var = exp_prod_var * e_v;
            }
        }

        // 5. Sigmoid 제약: sig_q * (prod_scale + exp_prod) + sig_rem = target_scale
        let sig_lhs = (&sig_var * &(FpVar::Constant(prod_scale) + &exp_prod_var)) + &sig_rem_var;
        sig_lhs.enforce_equal(&FpVar::Constant(target_scale))?;

        // 6. GeLU 제약: y_q * (scale_input * multiplier) + y_rem = (x_q - input_0 * multiplier) * sig_q
        let offset_m = bundle.input_0 * bundle.multiplier;
        let x_raw_var = &x_q_var - FpVar::Constant(offset_m);
        let den_var = FpVar::Constant(bundle.scale_input * bundle.multiplier);
        
        let gelu_lhs = (y_q_var * den_var) + y_rem_var;
        let gelu_rhs = x_raw_var * sig_var;
        gelu_lhs.enforce_equal(&gelu_rhs)?;
    }

    Ok(())
}