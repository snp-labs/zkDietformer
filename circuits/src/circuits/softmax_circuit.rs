use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{
    fields::{FieldVar, fp::FpVar},
    prelude::{AllocVar, Boolean, EqGadget, ToBitsGadget},
};

// 세크먼트 및 림브 설정(파이썬 설정값과 일치해야 함)
pub const K_SEGMENTS: usize = 2;
pub const LIMB_BITS: usize = 8;
pub const LIMB_BASE: u128 = 1u128 << LIMB_BITS;

#[derive(Debug, Clone)]
pub struct SoftmaxCircuit<F: PrimeField> {
    pub matmul_outputs: Vec<Vec<Vec<Vec<F>>>>,
    // limbs: for each scalar matmul output, array of K limbs
    pub limbs: Vec<Vec<Vec<Vec<[F; K_SEGMENTS]>>>>,
    // exp_limb_vals: for each scalar, K per-limb exponent values
    pub exp_limb_vals: Vec<Vec<Vec<Vec<[F; K_SEGMENTS]>>>>,
    pub softmax_outputs: Vec<Vec<Vec<Vec<F>>>>,

    pub frac_bits: u32,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for SoftmaxCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        
        let batch_size = self.matmul_outputs.len();
        if batch_size == 0 {
            return Ok(());
        }
        let head_size = self.matmul_outputs[0].len();

        for n in 0..batch_size {
            for h in 0..head_size {
                let rows = self.matmul_outputs[n][h].len(); // seq_len (queries)
                let cols = if rows > 0 { self.matmul_outputs[n][h][0].len() } else { 0 }; // seq_len (keys)

                for i in 0..rows {
                    // 수집할 벡터들: 이 행 i에 대해 모든 열 j의 exp_prod 및 softmax 값들
                    let mut exp_prods_vars: Vec<FpVar<F>> = Vec::with_capacity(cols);
                    let mut softmax_vars: Vec<FpVar<F>> = Vec::with_capacity(cols);

                    // 각 열 j에 대해 제약(재조립, 범위, per-limb exp 곱, softmax witness 할당)
                    for j in 0..cols {
                        // 행렬곱 결과값 가져오고
                        let x_var = FpVar::<F>::new_witness(ark_relations::ns!(cs, "input_var"), || Ok(self.matmul_outputs[n][h][i][j]))?;

                        // i에 대한 결과값 e^i에 대한 테이블 룩업 증명은 따로 있고, 이 값이 테이블에서 가져온 exp 세그먼트들로 재조립이 된것과 동치하는지 제약
                        // limb들 할당 : 우리는 예를 들어 2^16 진법을 쓴다고 생각.
                        // 22 * 2^16 = 하여 나온 첫번째 limb이 22, 두번째 limb이 0 이런식. > limb_vars에 들어감
                        let mut limb_vars: [FpVar<F>; K_SEGMENTS] = {
                            let mut tmp = Vec::with_capacity(K_SEGMENTS);
                            for k in 0..K_SEGMENTS {
                                let v = self.limbs[n][h][i][j][k];
                                tmp.push(FpVar::<F>::new_witness(ark_relations::ns!(cs, "limb_var"), || Ok(v))?);
                            }
                            tmp.try_into().unwrap()
                        };
                        let mut recomposed = FpVar::<F>::zero();
                        let mut mul_const: u128 = 1;
                        for k in 0..K_SEGMENTS {
                            let c = F::from(mul_const as u128);
                            recomposed = recomposed + (&limb_vars[k] * FpVar::Constant(c));
                            mul_const <<= LIMB_BITS;
                        }
                        recomposed.enforce_equal(&x_var)?;

                        // limb range check (to_bits_le 사용) 이 인덱스가 테이블 안에 있다는 것을 증명
                        /*for k in 0..K_SEGMENTS {
                            let bits = limb_vars[k].to_bits_le()?;
                            if bits.len() > LIMB_BITS {
                                for b in bits.iter().skip(LIMB_BITS) {
                                    b.enforce_equal(&Boolean::constant(false))?;
                                }
                            }
                        }*/

                        // per-limb exp 할당 및 곱
                        let mut exp_prod_var = FpVar::<F>::Constant(F::one());
                        for k in 0..K_SEGMENTS { // e^limb_val 합쳐서 5개의 segement를 하나로 만들고
                            let e_val = self.exp_limb_vals[n][h][i][j][k];
                            let e_var = FpVar::<F>::new_witness(ark_relations::ns!(cs, "exp"), || Ok(e_val))?;
                            exp_prod_var = exp_prod_var * e_var;
                        }
                        exp_prods_vars.push(exp_prod_var); 

                        // softmax witness 할당
                        let s_val = self.softmax_outputs[n][h][i][j];
                        let s_var = FpVar::<F>::new_witness(ark_relations::ns!(cs, "output_var"), || Ok(s_val))?;
                        softmax_vars.push(s_var);
                    }

                    // 소프트맥스 분모 부분 계산
                    let mut denom = FpVar::<F>::zero();
                    for ep in exp_prods_vars.iter() { denom = denom + ep; }

                    // 분모 * 소프트맥스 결과값 = 분자 확인
                    for (s_var, ep_var) in softmax_vars.iter().zip(exp_prods_vars.iter()) {
                        let lhs = s_var * &denom;
                        lhs.enforce_equal(ep_var)?;
                    }

                    // 모든 소프트맥스 값들의 합이 1인지 확인
                    let mut sum_soft = FpVar::<F>::zero();
                    for s in softmax_vars.iter() { sum_soft = sum_soft + s; }
                    sum_soft.enforce_equal(&FpVar::Constant(F::one()))?;
                } 
            } 
        } 

        Ok(())
    }
} 