//! Constraint implementations for dot product gadgets.

use ark_ff::PrimeField;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};

/// Computes the dot product of two variable vectors.
///
/// Returns `sum_i (a_i * b_i)`.
///
/// # Panics
/// Panics if the input lengths do not match.
pub fn dot_product<F: PrimeField>(a: &[FpVar<F>], b: &[FpVar<F>]) -> FpVar<F> {
    assert_eq!(a.len(), b.len());
    let mut acc = FpVar::<F>::constant(F::zero());
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc += ai * bi;
    }
    acc
}

/// Computes the dot product of a variable vector and a constant vector.
///
/// Returns `sum_i (a_i * w_i)`.
///
/// # Panics
/// Panics if the input lengths do not match.
pub fn dot_prod_const<F: PrimeField>(a: &[FpVar<F>], w: &[F]) -> FpVar<F> {
    assert_eq!(a.len(), w.len());

    let mut acc = FpVar::<F>::constant(F::ZERO);
    for (ai, wi) in a.iter().zip(w.iter()) {
        acc += ai * FpVar::<F>::constant(*wi);
    }
    acc
}

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_r1cs_std::{R1CSVar, alloc::AllocVar};
    use ark_relations::r1cs::ConstraintSystem;
    use ark_std::Zero;

    #[test]
    fn test_dot_prod() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let a_vec = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let b_vec = vec![Fr::from(4u64), Fr::from(5u64), Fr::from(6u64)];

        let a = Vec::<FpVar<Fr>>::new_witness(cs.clone(), || Ok(a_vec.clone())).unwrap();
        let b = Vec::<FpVar<Fr>>::new_witness(cs.clone(), || Ok(b_vec.clone())).unwrap();

        let out = dot_product(&a, &b);

        let expected = a_vec
            .iter()
            .zip(b_vec.iter())
            .fold(Fr::zero(), |acc, (x, y)| acc + (*x * *y));

        assert_eq!(out.value().unwrap(), expected);
        println!("number of constraints: {}", cs.num_constraints());
        assert!(cs.is_satisfied().unwrap());
    }
}
