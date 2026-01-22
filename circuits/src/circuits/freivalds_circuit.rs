use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{fields::fp::FpVar, fields::FieldVar, prelude::AllocVar};
use ark_r1cs_std::eq::EqGadget;
use crate::gadgets::matmul::constraints::{
    new_vector_witness, 
    new_matrix_witness, 
    mat_vec_mul, 
    MatrixVar
};

#[derive(Debug, Clone)]
pub struct FreivaldsCircuit<F:PrimeField> {
    pub a: Vec<Vec<Vec<Vec<F>>>>, // [Batch Size, Num Head, Seq Len, Head dim]
    pub b: Vec<Vec<Vec<Vec<F>>>>, // [Batch Size, Num Head, 토큰 개수, 토큰 하나의 길이]
    pub c: Vec<Vec<Vec<Vec<F>>>>,
    pub r: F,      // Random vector r, Fiat-Shamir challenge(public input)

    //these two variables are used to restore the real y
    pub remainder: Vec<Vec<Vec<Vec<F>>>>,
    pub div: Vec<Vec<Vec<Vec<F>>>>,

    //zero points for quantization
    pub input_0: F, // 현재 Q,K에 대한것
    pub c_0_converted: F, // C에 대한 것

    //multiplier for quantization. s1*s2/s3
    pub multiplier: F,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for FreivaldsCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        
        let batch_size = self.a.len();
        // erase
        if batch_size == 0 {
            return Ok(());
        }
        let head_size = self.a[0].len();

        // scaling constants
        let two_power_8: F = (2u64.pow(8)).into();
        let two_power_8_constant = FpVar::<F>::Constant(two_power_8);
        let two_power_22: F = (2u64.pow(22)).into();
        let two_power_22_constant = FpVar::<F>::Constant(two_power_22);

        let m_var = FpVar::new_witness(ark_relations::ns!(cs, "m_var"), || Ok(self.multiplier))?;

        let c_0_var = FpVar::Constant(self.c_0_converted);
        let a_0_var = FpVar::Constant(self.input_0);
        let b_0_var = FpVar::Constant(self.input_0);
        
        // random r
        let r_var = FpVar::new_input(ark_relations::ns!(cs, "r_scalar"), || Ok(self.r))?;

        for n in 0..batch_size {
            for h in 0..head_size {

                let a_var = new_matrix_witness(cs.clone(), &self.a[n][h]); // seq_len x head_dim
                let b_var = new_matrix_witness(cs.clone(), &self.b[n][h]); // head_dim x seq_len 전치된 상태로 들어옴
                let c_var = new_matrix_witness(cs.clone(), &self.c[n][h]); // seq_len x seq_len
                let div_var = new_matrix_witness(cs.clone(), &self.div[n][h]);
                let rem_var = new_matrix_witness(cs.clone(), &self.remainder[n][h]);

                let r_rows = b_var[0].len(); // seq_len
                let a_cols = if !a_var.is_empty() { a_var[0].len() } else { 0 }; // head_dim
                let a_rows = a_var.len(); // seq_len
                let b_rows = b_var.len(); // head_dim
                let b_cols = if !b_var.is_empty() { b_var[0].len() } else { 0 }; // seq_len
                // Ensure shapes are consistent
                if a_cols != b_rows {
                    return Err(SynthesisError::Unsatisfiable);
                }
                if r_rows != b_cols {
                    return Err(SynthesisError::Unsatisfiable);
                }

                // r_scalar_var로부터 벡터 r_pows = [r^0, r^1, ..., r^{r_rows-1}] 생성
                let mut r_pows: Vec<FpVar<F>> = Vec::with_capacity(r_rows);
                if r_rows > 0 {
                    // 첫 항 r^0 = 1
                    let one_var = FpVar::<F>::Constant(F::one());
                    r_pows.push(one_var);
                    for idx in 1..r_rows {
                        // r^{k} = r^{k-1} * r
                        let next = &r_pows[idx - 1] * &r_var;
                        r_pows.push(next);
                    }
                }

                let sum_r = r_pows.iter().fold(FpVar::zero(), |acc, x| acc + x);

                // Left side: m * ((A-a0)(B-b0) r + c0_corr) 
                // 1. (AB)r = A(Br)
                let br = mat_vec_mul(&b_var, &r_pows)?; // length head_dim
                let term1 = mat_vec_mul(&a_var, &br)?; // length seq_len

                // 2. (Ab0)r = b0 * sum_r * (RowSums of A)
                let b0_sum_r = &b_0_var * &sum_r;
                let term2: Vec<FpVar<F>> = a_var.iter()
                    .map(|row| {
                        let row_sum = row.iter().fold(FpVar::zero(), |acc, x| acc + x);
                        row_sum * &b0_sum_r
                    })
                    .collect();

                // 3. (a0B)r = a0 * sum_k br_k  (scalar)
                let sum_br = br.iter().fold(FpVar::zero(), |acc, x| acc + x); // scalar
                let term3_scalar = &sum_br * &a_0_var;

                // 4. (a0b0)r = head_dim * a0 * b0 * sum_r  (scalar)
                let head_dim_const: F = F::from(a_cols as u64);
                let head_dim_fpvar = FpVar::Constant(head_dim_const);
                let term4_val = &a_0_var * &b_0_var * &sum_r * &head_dim_fpvar;

                // 5. c0 correction = c0 * sum_r
                let c0_corr = &c_0_var * &sum_r;

                // Combine left side per row
                let mut left_side: Vec<FpVar<F>> = Vec::with_capacity(a_rows); // a_row = seq_len
                for i in 0..a_rows {
                    let res = (&term1[i] - &term2[i] - &term3_scalar + &term4_val + &c0_corr) * &m_var;
                    left_side.push(res);
                }

                // Right side: ((C + div*2^8)*2^22 + rem) * r 
                let mut cc_var_matrix: MatrixVar<F> = Vec::with_capacity(a_rows);
                for i in 0..a_rows {
                    let mut row = Vec::with_capacity(c_var[0].len());
                    for j in 0..c_var[0].len() {
                        let val = (&c_var[i][j] + (&div_var[i][j] * &two_power_8_constant)) 
                                  * &two_power_22_constant + &rem_var[i][j];
                        row.push(val);
                    }
                    cc_var_matrix.push(row);
                }

                let right_side = mat_vec_mul(&cc_var_matrix, &r_pows)?;

                // enforce equality
                for (l, r) in left_side.iter().zip(right_side.iter()) {
                    l.enforce_equal(r)?;
                }
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use crate::{cc_snark::cc_gro::{CcGroth16, prepare_verifying_key}, circuits::freivalds_circuit::FreivaldsCircuit};
    use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
    use ark_ff::PrimeField;
    use ark_bls12_381::Bls12_381; 
    use ark_ec::pairing::Pairing;
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
    use ark_std::rand::{SeedableRng, rngs::StdRng};
    use ark_std::rand::Rng;

    fn setup_test_data<F: PrimeField>(
        batch_size: usize,
        num_head: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> (
        Vec<Vec<Vec<Vec<F>>>>, // a
        Vec<Vec<Vec<Vec<F>>>>, // b
        Vec<Vec<Vec<Vec<F>>>>, // c
        Vec<Vec<Vec<Vec<F>>>>, // remainder
        Vec<Vec<Vec<Vec<F>>>>, // div
        F,                // r
        F,                     // input_0
        F,                     // c_0_converted
        F,                     // multiplier
    ) {
        let mut rng = StdRng::seed_from_u64(42);

        // a: seq_len x head_dim
        // b: head_dim x seq_len
        let mut a = vec![vec![vec![vec![0u128; head_dim]; seq_len]; num_head]; batch_size];
        let mut b = vec![vec![vec![vec![0u128; seq_len]; head_dim]; num_head]; batch_size];

        let mut c = vec![vec![vec![vec![0u128; seq_len]; seq_len]; num_head]; batch_size];
        let mut rem = vec![vec![vec![vec![0u128; seq_len]; seq_len]; num_head]; batch_size];
        let mut div = vec![vec![vec![vec![0u128; seq_len]; seq_len]; num_head]; batch_size];

        let r: u64 = rng.gen_range(2..6) as u64;

        let input_0: u128 = rng.gen_range(1..5) as u128;
        let c_0_converted: u128 = rng.gen_range(1..10) as u128;
        let multiplier: u128 = rng.gen_range(1..10) as u128;

        for n in 0..batch_size {
            for h in 0..num_head {
                for i in 0..seq_len {
                    for k in 0..head_dim {
                        let a_val = rng.gen_range(10..20) as u128;
                        a[n][h][i][k] = a_val;
                        
                        let b_val = rng.gen_range(10..20) as u128;
                        b[n][h][k][i] = b_val;
                    }
                }
            }
        }

        // 상수: 비트마스크/쉬프트
        let mask_22: u128 = (1u128 << 22) - 1u128;

        for n in 0..batch_size {
            for h in 0..num_head {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // ab_sum = sum_k ( (a[i][k] - input_0) * (b[k][j] - input_0) )
                        let mut ab_sum: i128 = 0;
                        for k in 0..head_dim {
                            // (a - input0)와 (b - input0) > 0
                            let a_minus = (a[n][h][i][k] as i128) - (input_0 as i128);
                            let b_minus = (b[n][h][k][j] as i128) - (input_0 as i128);
                            ab_sum += a_minus * b_minus;
                        }
                        // ab_sum이 음수일 경우 0으로 클램핑
                        let ab_sum_nonneg = if ab_sum < 0 { 0i128 } else { ab_sum };

                        // target_val = (ab_sum + c_0_converted) * multiplier
                        let target_val_u128 = ((ab_sum_nonneg as u128) + c_0_converted) * multiplier;

                        let div_22 = target_val_u128 >> 22;                  // floor(target / 2^22)
                        let rem_22 = target_val_u128 & mask_22;                 // target % 2^22
                        let div_8 = div_22 >> 8;                             // floor(div_22 / 2^8)
                        let c_val = div_22 & 0xffu128;                       // lower 8 bits of div_22

                        c[n][h][i][j] = c_val;
                        div[n][h][i][j] = div_8;
                        rem[n][h][i][j] = rem_22;
                    }
                }
            }
        }

        let mut a_f = vec![vec![vec![vec![F::zero(); head_dim]; seq_len]; num_head]; batch_size];
        let mut b_f = vec![vec![vec![vec![F::zero(); seq_len]; head_dim]; num_head]; batch_size];
        let mut c_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch_size];
        let mut rem_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch_size];
        let mut div_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch_size];

        for n in 0..batch_size {
            for h in 0..num_head {
                for i in 0..seq_len {
                    for k in 0..head_dim {
                        a_f[n][h][i][k] = F::from(a[n][h][i][k] as u64);
                        b_f[n][h][k][i] = F::from(b[n][h][k][i] as u64);
                    }
                }

                for i in 0..seq_len {
                    for j in 0..seq_len {
                        c_f[n][h][i][j] = F::from(c[n][h][i][j] as u64);
                        div_f[n][h][i][j] = F::from(div[n][h][i][j] as u64);
                        rem_f[n][h][i][j] = F::from(rem[n][h][i][j] as u64);
                    }
                }
            }
        }

        let r_f = F::from(r);

        let input_0_f = F::from(input_0 as u64);
        let c_0_converted_f = F::from(c_0_converted as u64);
        let multiplier_f = F::from(multiplier as u64);

        (a_f, b_f, c_f, rem_f, div_f, r_f, input_0_f, c_0_converted_f, multiplier_f)
    }

    fn test_prove_and_verify_cc<E: Pairing>(batch: usize, num_head: usize) {
        let mut rng = StdRng::seed_from_u64(42);
        let seq_len = 5;
        let head_dim = 4;

        let (a, b, c, remainder, div, r, input_0, c_0_converted, multiplier) =
            setup_test_data::<E::ScalarField>(batch, num_head, seq_len, head_dim);

        let circuit = FreivaldsCircuit {
            a: a.clone(), b: b.clone(), c: c.clone(), r: r.clone(),
            remainder: remainder.clone(), div: div.clone(),
            input_0, c_0_converted, multiplier,
        };

        let cs = ConstraintSystem::<E::ScalarField>::new_ref();
        circuit.clone().generate_constraints(cs.clone()).expect("generate_constraints failed");

        let num_constraints = cs.num_constraints();
        println!("Generated constraints: {}", num_constraints);
        assert!(num_constraints > 0, "constraints should be greater than 0");

        let (pk, vk) = CcGroth16::<E>::setup(circuit.clone(), &mut rng).unwrap();
        let pvk = prepare_verifying_key::<E>(&vk);

        let proof = CcGroth16::<E>::prove(&pk, circuit, &mut rng).unwrap();

        // 현재 서킷에 Public Input(new_input_variable)이 없다면 빈 벡터 []를 전달.
        let public_inputs = vec![r.clone()]; 
        let result = CcGroth16::<E>::verify_with_processed_vk(&pvk, &public_inputs, &proof).unwrap();
        
        assert!(result, "cc-Groth16 proof verification failed!");
    }

    #[test]
    fn prove_and_verify_nystrom_bls12() {
        // 소규모 테스트 실행
        test_prove_and_verify_cc::<Bls12_381>(1, 1);
    }
}