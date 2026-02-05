use crate::config::{K_SEGMENTS, LIMB_BITS}; 
use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{
    fields::{FieldVar, fp::FpVar},
    prelude::{AllocVar, EqGadget},
};

/// GeLU 활성함수 증명을 위한 서킷 구조체
#[derive(Debug, Clone)]
pub struct GeluCircuit<F: PrimeField> {
    /// 양자화된 입력 벡터 (x_raw + input_0)
    pub inputs_q: Vec<F>,             
    /// 입력값에서 분해된 각 K-segments Limb 값들
    pub limbs: Vec<Vec<F>>,           
    /// 각 Limb에 대응하는 테이블 내 지수 기여분 (exp_limb_val)
    pub exp_limb_vals: Vec<Vec<F>>,   
    /// 시그모이드 중간 결과 위트니스
    pub sigmoid_outputs: Vec<F>,      
    /// 최종 GeLU 양자화 출력값 y_q
    pub gelu_outputs: Vec<F>,         
    
    /// 고정소수점 정밀도 설정을 위한 메타데이터
    pub input_0: F,
    pub scale_input: F,
    pub frac_bits: u32,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for GeluCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let n = self.inputs_q.len();
        let frac_unit = F::from(1u64 << self.frac_bits);
        let limb_base = F::from(1u64 << LIMB_BITS);

        // GeLU는 파라미터가 없으므로 모든 입력 원소에 대해 독립적인 제약을 생성함
        for i in 0..n {
            // 1. 변수 할당 (Witness Allocation)
            let x_q_var = FpVar::new_witness(cs.clone(), || Ok(self.inputs_q[i]))?;
            let sig_var = FpVar::new_witness(cs.clone(), || Ok(self.sigmoid_outputs[i]))?;
            let y_q_var = FpVar::new_witness(cs.clone(), || Ok(self.gelu_outputs[i]))?;
            let input_0_var = FpVar::Constant(self.input_0);

            // 2. Limb 분해 정당성 검증 (x_q = \sum limb_k * (2^LIMB_BITS)^k)
            let mut recomposed = FpVar::zero();
            let mut current_offset = F::one();
            for k in 0..self.limbs[i].len() {
                let limb_v = FpVar::new_witness(cs.clone(), || Ok(self.limbs[i][k]))?;
                recomposed += limb_v * FpVar::Constant(current_offset);
                current_offset *= limb_base;
            }
            recomposed.enforce_equal(&x_q_var)?;

            // 3. 지수 곱 계산 (고정소수점 나눗셈 제거를 위한 곱셈 제약)
            // 수식: Product(exp_limbs) = P_final * (frac_unit ^ (K-1))
            let mut limb_exp_prod = FpVar::Constant(F::one());
            for k in 0..self.exp_limb_vals[i].len() {
                let e_v = FpVar::new_witness(cs.clone(), || Ok(self.exp_limb_vals[i][k]))?;
                limb_exp_prod *= e_v;
            }

            let exp_prod_final = FpVar::new_witness(cs.clone(), || {
                // 평문에서 계산된 지수 곱 위트니스
                let mut p = self.exp_limb_vals[i][0];
                for k in 1..self.exp_limb_vals[i].len() {
                    // 고정소수점 정밀도 유지를 위해 중간마다 frac_bits만큼 스케일 다운
                    let val = F::from(p) * F::from(self.exp_limb_vals[i][k]);
                    p = F::from(val.into_bigint() >> self.frac_bits); // 이걸 해야하나?
                }
                Ok(p)
            })?;

            let frac_pow = frac_unit.pow([(self.exp_limb_vals[i].len() - 1) as u64]);
            (exp_prod_final.clone() * FpVar::Constant(frac_pow)).enforce_equal(&limb_exp_prod)?;

            // 4. Sigmoid 수식 검증: sig_q * (frac_unit + exp_prod) = frac_unit^2
            // Sigmoid(z) = 1 / (1 + exp(-z)) 의 양자화 형태
            let sig_lhs = sig_var.clone() * (FpVar::Constant(frac_unit) + exp_prod_final);
            sig_lhs.enforce_equal(&FpVar::Constant(frac_unit * frac_unit))?;

            // 5. GeLU 최종 결과 검증: y_q * scale_input = (x_q - input_0) * sig_q
            // y = x * Sigmoid(1.702x) 수식을 증명함
            let x_raw_var = &x_q_var - &input_0_var;
            let gelu_lhs = y_q_var * FpVar::Constant(self.scale_input);
            let gelu_rhs = x_raw_var * sig_var;
            gelu_lhs.enforce_equal(&gelu_rhs)?;
        }

        Ok(())
    }
}