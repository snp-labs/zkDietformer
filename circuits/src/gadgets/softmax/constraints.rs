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
    prelude::*, R1CSVar,
};

#[derive(Clone)]
pub struct SoftmaxBundle<F: PrimeField> {
    pub limbs: Vec<Vec<Vec<Vec<Vec<F>>>>>,
    pub exp_limbs: Vec<Vec<Vec<Vec<Vec<F>>>>>,
    pub q_exp: Vec<Vec<Vec<Vec<Vec<F>>>>>,
    pub r_exp: Vec<Vec<Vec<Vec<Vec<F>>>>>,
    pub exp_total: Vec<Vec<Vec<Vec<F>>>>,
    pub p: Vec<Vec<Vec<Vec<F>>>>,
    pub r_div: Vec<Vec<Vec<Vec<F>>>>,
    pub r_sm: Vec<Vec<Vec<Vec<F>>>>,
    pub m_sm: F,
}

impl<F: PrimeField> SoftmaxBundle<F> {
    pub fn alloc_vars(&self, cs: ConstraintSystemRef<F>) -> Result<SoftmaxVarBundle<F>, SynthesisError> {
        Ok(SoftmaxVarBundle {
            limbs: self.limbs.iter().map(|l| new_matrix_witness_4d(cs.clone(), l)).collect(),
            exp_limbs: self.exp_limbs.iter().map(|e| new_matrix_witness_4d(cs.clone(), e)).collect(),
            q_exp: self.q_exp.iter().map(|q| new_matrix_witness_4d(cs.clone(), q)).collect(),
            r_exp: self.r_exp.iter().map(|r| new_matrix_witness_4d(cs.clone(), r)).collect(),
            exp_total: new_matrix_witness_4d(cs.clone(), &self.exp_total),
            p: new_matrix_witness_4d(cs.clone(), &self.p),
            r_div: new_matrix_witness_4d(cs.clone(), &self.r_div),
            r_sm: new_matrix_witness_4d(cs.clone(), &self.r_sm),
            m_sm: FpVar::new_constant(cs.clone(), self.m_sm)?,
        })
    }
}

#[derive(Clone)]
pub struct SoftmaxVarBundle<F: PrimeField> {
    pub limbs: Vec<Vec<Vec<MatrixVar<F>>>>,      
    pub exp_limbs: Vec<Vec<Vec<MatrixVar<F>>>>,
    pub q_exp: Vec<Vec<Vec<MatrixVar<F>>>>,
    pub r_exp: Vec<Vec<Vec<MatrixVar<F>>>>,
    pub exp_total: Vec<Vec<MatrixVar<F>>>,       
    pub p: Vec<Vec<MatrixVar<F>>>,
    pub r_div: Vec<Vec<MatrixVar<F>>>,
    pub r_sm: Vec<Vec<MatrixVar<F>>>, 
    pub m_sm: FpVar<F>,
}

/*pub fn enforce_softmax<F: PrimeField>(
    q_sim: &MatrixVar<F>,       
    r_matmul: &MatrixVar<F>,    
    r_sm: &MatrixVar<F>,        
    m_sm: &FpVar<F>,            
    limbs: &[MatrixVar<F>],     
    exp_limbs: &[MatrixVar<F>], 
    q_exp: &[MatrixVar<F>],     
    r_exp: &[MatrixVar<F>],     
    exp_total: &MatrixVar<F>,    
    p_out: &MatrixVar<F>,        
    r_div: &MatrixVar<F>,        
    frac_bits: usize,
    two_k: &FpVar<F>,           
) -> Result<(), SynthesisError> {
    let s_len = q_sim.len();
    let l_len = q_sim[0].len();
    let frac_scale = F::from(1u128 << frac_bits);

    for i in 0..s_len {
        let mut row_exp_sum = FpVar::zero();
        for j in 0..l_len {
            row_exp_sum += &exp_total[i][j];
        }

        for j in 0..l_len {
            let x_recovered = (&q_sim[i][j] * two_k) + &r_matmul[i][j];
            
            let mut scaled_x = FpVar::zero();
            for k in 0..5 {
                scaled_x += &limbs[k][i][j] * F::from(1u128 << (16 * k));
            }

            let lhs_scale = &x_recovered * m_sm;
            let rhs_scale = (&scaled_x * two_k) + &r_sm[i][j];
            lhs_scale.enforce_equal(&rhs_scale)?;

            let mut current_acc = exp_limbs[0][i][j].clone();
            for k in 0..4 {
                let lhs_mult = &current_acc * &exp_limbs[k+1][i][j];
                let next_q = if k == 3 { exp_total[i][j].clone() } else { q_exp[k][i][j].clone() };
                let rhs_mult = (&next_q * frac_scale) + &r_exp[k][i][j];
                lhs_mult.enforce_equal(&rhs_mult)?;
                current_acc = next_q;
            }

            let lhs_div = &exp_total[i][j] * frac_scale;
            let rhs_div = (&p_out[i][j] * &row_exp_sum) + &r_div[i][j];
            lhs_div.enforce_equal(&rhs_div)?;
        }
    }
    Ok(())
}*/

pub fn enforce_softmax<F: PrimeField>(
    q_sim: &MatrixVar<F>,       
    r_matmul: &MatrixVar<F>,    
    r_sm: &MatrixVar<F>,        
    m_sm: &FpVar<F>,            
    limbs: &[MatrixVar<F>],     
    exp_limbs: &[MatrixVar<F>], 
    q_exp: &[MatrixVar<F>], 
    p_out: &MatrixVar<F>,     
    r_div: &MatrixVar<F>,          
    frac_bits: usize,
    two_k: &FpVar<F>,           
) -> Result<(), SynthesisError> {
    let s_len = q_sim.len();
    let l_len = q_sim[0].len();
    let frac_scale = F::from(1u128 << frac_bits);

    for i in 0..s_len {
        let mut row_exp_sum = FpVar::zero();
        let mut row_exp_totals = Vec::with_capacity(l_len);

        for j in 0..l_len {
            let x_recovered = (&q_sim[i][j] * two_k) + &r_matmul[i][j];
            
            let mut scaled_x = FpVar::zero();
            for k in 0..5 {
                scaled_x += &limbs[k][i][j] * F::from(1u128 << (16 * k));
            }

            let lhs_scale = &x_recovered * m_sm;
            let rhs_scale = (&scaled_x * two_k) + &r_sm[i][j];
            lhs_scale.enforce_equal(&rhs_scale)?;

            let mut current_acc = exp_limbs[0][i][j].clone();
            for k in 0..4 {
                let lhs_mult = &current_acc * &exp_limbs[k+1][i][j];
                let next_q = q_exp[k][i][j].clone(); 
                current_acc = next_q; 
            }

            let exp_total_var = current_acc; 
            row_exp_sum += &exp_total_var;
            row_exp_totals.push(exp_total_var);
        }

        for j in 0..l_len {
            let lhs_div = &row_exp_totals[j] * frac_scale;
            let rhs_div = (&p_out[i][j] * &row_exp_sum) + &r_div[i][j];
            lhs_div.enforce_equal(&rhs_div)?;
        }
    }
    Ok(())
}