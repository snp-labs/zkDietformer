//! zkSNARK circuits and gadgets for Transformer-style neural networks
//! components for building zero-knowledge proofs of Transformer models,
//! including matrix multiplication, linear layers, and activation functions.
#![allow(missing_docs)]
#![feature(extend_one)]
#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![allow(
    dead_code,
    unused_imports,
    unused_mut,
    unused_variables,
    unused_macros,
    unused_assignments,
    unreachable_patterns
)]

#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![allow(incomplete_features)]

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
/// High-level zkSNARK circuits for Transformer models.
pub mod circuits;

/// Reusable low-level constraint gadgets used by circuits.
pub mod gadgets;

/// cc_snark: zkSNARK implementation based on the Groth16 protocol.
pub mod cc_snark;
/// util
pub mod util;
pub mod config;

pub mod lasso;

pub use ark_crypto_primitives::*;
pub use ark_ff::{Field, PrimeField, BigInteger, One, Zero}; 
pub use ark_ec::{CurveGroup, pairing::Pairing, AffineRepr};
pub use ark_poly::{DenseUVPolynomial, Polynomial};
pub use ark_std::{marker::PhantomData, vec::Vec};

pub use ark_ec::pairing;

pub(crate) type ConstraintF<C> = <<C as CurveGroup>::BaseField as Field>::BasePrimeField;
pub(crate) type BasePrimeField<E> =
    <<<E as Pairing>::G1Affine as CurveGroup>::BaseField as Field>::BasePrimeField;

#[cfg(feature = "r1cs")]
#[macro_use]
extern crate derivative;
