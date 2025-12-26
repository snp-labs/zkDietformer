//! Constraint implementations for activation gadgets.

use ark_ff::{BigInteger, PrimeField};
use ark_r1cs_std::{boolean::Boolean, fields::fp::FpVar, prelude::*};
use ark_relations::r1cs::SynthesisError;
use core::cmp::Ordering;

/// Returns `(p - 1) / 2`, where `p` is the field modulus.
pub fn modulus_minus_one_div_two<F: PrimeField>() -> F {
    F::from_bigint(F::MODULUS_MINUS_ONE_DIV_TWO).unwrap()
}

/// Returns `(p - 1) / 2` as a constant circuit variable.
pub fn modulus_minus_one_div_two_var<F: PrimeField>() -> FpVar<F> {
    FpVar::<F>::constant(modulus_minus_one_div_two::<F>())
}

/// Checks whether a field element is treated as positive.
///
/// Values `<= (p - 1) / 2` are considered non-negative.
pub fn is_positive<F: PrimeField>(x: &FpVar<F>) -> Result<Boolean<F>, SynthesisError> {
    let half = modulus_minus_one_div_two_var::<F>();
    x.is_cmp_unchecked(&half, Ordering::Greater, true)
}

/// Checks whether a field element is treated as negative.
///
/// Values `> (p - 1) / 2` are considered negative.
pub fn is_negative<F: PrimeField>(x: &FpVar<F>) -> Result<Boolean<F>, SynthesisError> {
    let half = modulus_minus_one_div_two_var::<F>();
    x.is_cmp_unchecked(&half, Ordering::Greater, false)
}
