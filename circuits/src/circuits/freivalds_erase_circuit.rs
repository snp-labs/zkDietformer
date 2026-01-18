#![allow(missing_docs)]
use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{fields::fp::FpVar, fields::FieldVar, prelude::AllocVar};
use ark_r1cs_std::eq::EqGadget;
use crate::gadgets::matmul::constraints::{
    new_matrix_witness, 
    mat_vec_mul, 
    MatrixVar
};

/// 임시 : A (m x n) * B (n x p) = C (m x p)를 검증해주기위한 서킷
#[derive(Debug, Clone)]
pub struct FreivaldsMatMulCircuit<F: PrimeField> {
    pub a: Vec<Vec<Vec<Vec<F>>>>, // [Batch Size, Num Head, Seq Len, Head dim]
    pub b: Vec<Vec<Vec<Vec<F>>>>, // [batch][head][n:토큰 개수][p:토큰 하나의 길이]
    pub c: Vec<Vec<Vec<Vec<F>>>>, // [batch][head][m][p]
    pub r: F,                     // Random vector r
}


impl<F: PrimeField> ConstraintSynthesizer<F> for FreivaldsMatMulCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let batch = self.a.len();
        let head = if batch > 0 { self.a[0].len() } else { 0 };

        // r scalar witness
        let r_var = ark_r1cs_std::fields::fp::FpVar::<F>::new_witness(ark_relations::ns!(cs, "r_scalar"), || Ok(self.r))?;

        for n in 0..batch {
            for h in 0..head {
                let a_var = new_matrix_witness(cs.clone(), &self.a[n][h]); // m x n
                let b_var = new_matrix_witness(cs.clone(), &self.b[n][h]); // n x p
                let c_var = new_matrix_witness(cs.clone(), &self.c[n][h]); // m x p

                let m = a_var.len(); // rows of A
                let ncols = if m > 0 { a_var[0].len() } else { 0 }; // n (inner)
                let p = if !c_var.is_empty() { c_var[0].len() } else { 0 }; // p

                // check shape consistency
                if ncols != b_var.len() { return Err(SynthesisError::Unsatisfiable); }
                if m != c_var.len() { return Err(SynthesisError::Unsatisfiable); }
                if p == 0 { continue; }

                // build r_pows = [1, r, r^2, ..., r^{p-1}] length p
                let mut r_pows: Vec<ark_r1cs_std::fields::fp::FpVar<F>> = Vec::with_capacity(p);
                r_pows.push(ark_r1cs_std::fields::fp::FpVar::Constant(F::one()));
                for idx in 1..p {
                    let next = &r_pows[idx - 1] * &r_var;
                    r_pows.push(next);
                }

                // br = B * r_pows  -> length ncols
                let br = mat_vec_mul(&b_var, &r_pows)?;
                // left = A * (B r)
                let left = mat_vec_mul(&a_var, &br)?;
                // right = C * r_pows
                let right = mat_vec_mul(&c_var, &r_pows)?;

                // enforce left == right component-wise
                for i in 0..m {
                    left[i].enforce_equal(&right[i])?;
                }
            }
        }
        Ok(())
    }
}