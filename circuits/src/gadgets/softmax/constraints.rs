use crate::config::{K_SEGMENTS, LIMB_BITS}; 
use crate::gadgets::matmul::constraints::MatrixVar;
use crate::gadgets::matmul::quantized::new_matrix_witness_4d;
use std::fs::File;
use std::path::Path;
use std::error::Error;

use ark_ff::{PrimeField, Zero, One};
use ndarray::{Array2, Array0};
use ndarray_npy::NpzReader;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{
    fields::{FieldVar, fp::FpVar},
    prelude::{AllocVar, EqGadget},
};

pub struct SoftmaxWitnessVar<F: PrimeField> {
    pub limbs: Vec<MatrixVar<F>>,      
    pub exp_limbs: Vec<MatrixVar<F>>,  
    pub exp_total: MatrixVar<F>,       
    pub softmax_res: MatrixVar<F>,     
    pub quotients: Vec<MatrixVar<F>>,   // [3][S][L] 중간 몫
    pub remainders: Vec<MatrixVar<F>>,  // [4][S][L] 나머지
    pub softmax_rem: MatrixVar<F>,
}

pub fn new_matrix_witness_5d<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    data: &Vec<Vec<Vec<Vec<Vec<F>>>>>,
) -> Vec<Vec<Vec<MatrixVar<F>>>> {
    data.iter()
        .map(|d4| new_matrix_witness_4d(cs.clone(), d4))
        .collect()
}

// load.rs 또는 constraints.rs에 추가
pub fn new_matrix_witness_3d<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    data: &Vec<Vec<Vec<F>>>,
) -> Result<Vec<Vec<Vec<FpVar<F>>>>, SynthesisError> {
    let mut out_b = Vec::with_capacity(data.len());
    for b in 0..data.len() {
        let mut out_h = Vec::with_capacity(data[b].len());
        for h in 0..data[b].len() {
            let mut out_s = Vec::with_capacity(data[b][h].len());
            for s in 0..data[b][h].len() {
                out_s.push(FpVar::new_witness(cs.clone(), || Ok(data[b][h][s]))?);
            }
            out_h.push(out_s);
        }
        out_b.push(out_h);
    }
    Ok(out_b)
}

pub fn get_softmax_witness_at<F: PrimeField>(
    b: usize,
    h: usize,
    limbs_all: &Vec<Vec<Vec<MatrixVar<F>>>>,
    exp_limbs_all: &Vec<Vec<Vec<MatrixVar<F>>>>,
    exp_total_all: &Vec<Vec<MatrixVar<F>>>,
    softmax_all: &Vec<Vec<MatrixVar<F>>>,
    quotients_all: &Vec<Vec<Vec<MatrixVar<F>>>>,
    remainders_all: &Vec<Vec<Vec<MatrixVar<F>>>>,
    softmax_rem_all: &Vec<Vec<MatrixVar<F>>>,
) -> SoftmaxWitnessVar<F> {
    let mut limbs = Vec::new();
    let mut exp_limbs = Vec::new();
    let mut quotients = Vec::new();
    let mut remainders = Vec::new();

    for seg in 0..limbs_all.len() {
        limbs.push(limbs_all[seg][b][h].clone());
        exp_limbs.push(exp_limbs_all[seg][b][h].clone());
    }
    for seg in 0..quotients_all.len() {
        quotients.push(quotients_all[seg][b][h].clone());
    }
    for seg in 0..remainders_all.len() {
        remainders.push(remainders_all[seg][b][h].clone());
    }

    SoftmaxWitnessVar {
        limbs,
        exp_limbs,
        quotients,
        exp_total: exp_total_all[b][h].clone(),
        softmax_res: softmax_all[b][h].clone(),
        remainders,
        softmax_rem: softmax_rem_all[b][h].clone(),
    }
}

// constraints.rs

pub fn enforce_softmax<F: PrimeField>(
    q_sim: &MatrixVar<F>,       // [S][L]
    r_matmul: &MatrixVar<F>,    // [S][L]
    limbs: &[MatrixVar<F>],     // [K_SEGMENTS] 리스트, 각 Matrix는 [S][L]
    exp_limbs: &[MatrixVar<F>], // [K_SEGMENTS] 리스트
    q_exp: &[MatrixVar<F>],     // [K_SEGMENTS-2] 리스트
    r_exp: &[MatrixVar<F>],     // [K_SEGMENTS-1] 리스트
    exp_total: &MatrixVar<F>,    // [S][L]
    exp_sum: &[FpVar<F>],       // [S] (L축 합계)
    p_k1: &MatrixVar<F>,         // [S][L]
    r_div: &MatrixVar<F>,        // [S][L]
    frac_bits: usize,
    two_k: &FpVar<F>,
) -> Result<(), SynthesisError> {
    let s_len = q_sim.len();
    let l_len = q_sim[0].len();
    let frac_scale = F::from(1u128 << frac_bits);

    for i in 0..s_len {
        for j in 0..l_len {
            // 1. Bit-Decomposition (Limb) 검증 [cite: 203]
            let mut x_recovered = (&q_sim[i][j] * two_k) + &r_matmul[i][j];
            let mut limb_sum = FpVar::zero();
            for k in 0..5 {
                limb_sum += &limbs[k][i][j] * F::from(1u128 << (16 * k));
            }
            x_recovered.enforce_equal(&limb_sum)?;

            // 2. Remainder-based Exp Multiplication 
            let mut current_val = exp_limbs[0][i][j].clone();
            for k in 0..4 {
                let lhs = &current_val * &exp_limbs[k+1][i][j];
                let next_val = if k == 3 { exp_total[i][j].clone() } else { q_exp[k][i][j].clone() };
                let rhs = (&next_val * frac_scale) + &r_exp[k][i][j];
                lhs.enforce_equal(&rhs)?;
                current_val = next_val;
            }

            // 3. Division Check (나머지 기반 나눗셈) 
            let lhs_div = &exp_total[i][j] * frac_scale;
            let rhs_div = (&p_k1[i][j] * &exp_sum[i]) + &r_div[i][j];
            lhs_div.enforce_equal(&rhs_div)?;
        }
    }
    Ok(())
}

pub fn enforce_softmax_without_sum<F: PrimeField>(
    q_sim: &MatrixVar<F>,
    r_matmul: &MatrixVar<F>,
    limbs: &[MatrixVar<F>],
    exp_limbs: &[MatrixVar<F>],
    q_exp: &[MatrixVar<F>],
    r_exp: &[MatrixVar<F>],
    exp_total: &MatrixVar<F>,    // [S][L]
    p_k1: &MatrixVar<F>,
    r_div: &MatrixVar<F>,
    frac_bits: usize,
    two_k: &FpVar<F>,
) -> Result<(), SynthesisError> {
    let s_len = q_sim.len();
    let l_len = q_sim[0].len();
    let frac_scale = F::from(1u128 << frac_bits);

    for i in 0..s_len {
        // --- 내부에서 직접 Sum 계산 ---
        // 각 행(i)의 모든 열(j)에 있는 exp_total 값을 합산합니다.
        let mut row_exp_sum = FpVar::zero();
        for j in 0..l_len {
            row_exp_sum += &exp_total[i][j];
        }

        for j in 0..l_len {
            // 1. Bit-Decomposition (Limb) 검증
            let mut x_recovered = (&q_sim[i][j] * two_k) + &r_matmul[i][j];
            let mut limb_sum = FpVar::zero();
            for k in 0..5 {
                limb_sum += &limbs[k][i][j] * F::from(1u128 << (16 * k));
            }
            x_recovered.enforce_equal(&limb_sum)?;

            // 2. Remainder-based Exp Multiplication
            let mut current_val = exp_limbs[0][i][j].clone();
            for k in 0..4 {
                let lhs = &current_val * &exp_limbs[k+1][i][j];
                let next_val = if k == 3 { exp_total[i][j].clone() } else { q_exp[k][i][j].clone() };
                let rhs = (&next_val * frac_scale) + &r_exp[k][i][j];
                lhs.enforce_equal(&rhs)?;
                current_val = next_val;
            }

            // 3. Division Check (계산된 row_exp_sum 사용)
            // (exp_total << 32) == (P * row_exp_sum) + R_div
            let lhs_div = &exp_total[i][j] * frac_scale;
            let rhs_div = (&p_k1[i][j] * &row_exp_sum) + &r_div[i][j];
            lhs_div.enforce_equal(&rhs_div)?;
        }
    }
    Ok(())
}