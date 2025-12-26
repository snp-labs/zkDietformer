//! Activation function gadgets.
//!
//! This module defines traits and gadgets for non-linear activation
//! functions used in Transformer circuits, such as ReLU and GeLU.
//!

/// Constraint implementations for activation gadgets.
pub mod constraints;

/// ReLU activation function gadget.
pub mod relu;

pub use constraints::*;
pub use relu::*;

use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::SynthesisError;

/// Trait for native (non-circuit) activation functions.
pub trait ActivationFunction<F: PrimeField> {
    /// Name of the activation function.
    const NAME: &'static str;

    /// Evaluates the activation function on a native field element.
    fn eval_native(x: F) -> F;
}

/// Trait for activation gadgets inside an R1CS circuit.
pub trait ActivationGadget<AF: ActivationFunction<F>, F: PrimeField> {
    /// Applies the activation function to a single circuit variable.
    fn apply(x: &FpVar<F>) -> Result<FpVar<F>, SynthesisError>;

    /// Applies the activation function element-wise to a vector.
    fn apply_vec(xs: &[FpVar<F>]) -> Result<Vec<FpVar<F>>, SynthesisError> {
        xs.iter().map(|x| Self::apply(x)).collect()
    }

    /// Applies the activation function element-wise to a matrix.
    fn apply_matrix(m: &[Vec<FpVar<F>>]) -> Result<Vec<Vec<FpVar<F>>>, SynthesisError> {
        m.iter()
            .map(|row| Self::apply_vec(row))
            .collect::<Result<Vec<_>, _>>()
    }
}
