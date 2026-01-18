#![allow(missing_docs)]
pub mod data_structure;
pub mod precompute;
pub mod prover;
pub mod setup;
pub mod verifier;

pub mod arithm;
pub mod constants;
pub mod harisa;

pub mod hash_to_prime;
pub mod type_conversion;

pub mod di_hash;

mod test;

use std::{marker::PhantomData, str::FromStr};

pub use crate::cc_snark::*;
use crate::linker::Linker;

use ark_crypto_primitives::snark::*;
use ark_ec::pairing::Pairing;
use ark_r1cs_std::pairing::PairingVar;
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError};
use ark_std::rand::{CryptoRng, Rng, RngCore};
use num_bigint::BigInt;

pub type Error = Box<dyn ark_std::error::Error>;

pub trait Membership<E: Pairing, LNK: Linker<E>> {
    type Parameters: Clone;
    type Table;
    type Proof;

    fn setup<Arithm: ConstraintSynthesizer<E::ScalarField>, R: RngCore + CryptoRng + Rng>(
        set: Vec<BigInt>,
        arithm_circuit: Arithm,
        rng: &mut R,
    ) -> Result<(Self::Parameters, Self::Table, Vec<E::G1Affine>), Error>;

    fn prove<R: RngCore + CryptoRng + Rng>(
        pp: Self::Parameters,
        tree: Self::Table,
        accum: BigInt,
        cm_u: E::G1Affine,
        u: Vec<BigInt>,
        o_u: E::ScalarField,
        rng: &mut R,
        is_lookup: bool,
    ) -> Result<Self::Proof, Error>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug;

    fn verify(
        pp: Self::Parameters,
        accum: BigInt,
        c_u: E::G1Affine,
        proof: Self::Proof,
    ) -> Result<bool, Error>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug;
}
