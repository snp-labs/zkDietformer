//! zkSNARK circuits and gadgets for Transformer-style neural networks
//! components for building zero-knowledge proofs of Transformer models,
//! including matrix multiplication, linear layers, and activation functions.
#![deny(missing_docs)]
#![allow(
    dead_code,
    unused_imports,
    unused_mut,
    unused_variables,
    unused_macros,
    unused_assignments,
    unreachable_patterns
)]

/// High-level zkSNARK circuits for Transformer models.
pub mod circuits;

/// Reusable low-level constraint gadgets used by circuits.
pub mod gadgets;

/// cc_snark: zkSNARK implementation based on the Groth16 protocol.
pub mod cc_snark;

pub mod harisa;
pub mod linker;
pub mod lookup;
/// util
pub mod util;

pub use ark_crypto_primitives::*;
pub use ark_ec::*;
pub use ark_ff::*;
pub use ark_poly::*;

pub use pairing::*;

pub(crate) type ConstraintF<C> = <<C as CurveGroup>::BaseField as Field>::BasePrimeField;
pub(crate) type BasePrimeField<E> =
    <<<E as Pairing>::G1Affine as CurveGroup>::BaseField as Field>::BasePrimeField;

#[cfg(feature = "r1cs")]
#[macro_use]
extern crate derivative;

pub use ark_std::{marker::PhantomData, vec::Vec};
