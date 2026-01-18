#![allow(missing_docs)]
pub mod constants;
pub mod data_structure;
pub mod lookup;
pub mod prover;
pub mod setup;
pub mod verifier;

mod test;

pub mod copy_this_or_that;
pub mod well_transformed;


use std::str::FromStr;

use crate::{harisa::Membership, linker::Linker};

use ark_ec::pairing::Pairing;
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_std::rand::{CryptoRng, Rng, RngCore};
use num_bigint::BigInt;

pub type Error = Box<dyn ark_std::error::Error>;

pub trait Lookup<E: Pairing, M: Membership<E, LNK>, LNK: Linker<E>> {
    type PP;
    type Table;
    type Accum;
    type Proof;
    type CM;

    fn setup<
        CTT: ConstraintSynthesizer<E::ScalarField>,
        WT: ConstraintSynthesizer<E::ScalarField>,
        Arithm: ConstraintSynthesizer<E::ScalarField>,
        R: RngCore + CryptoRng + Rng,
    >(
        set: Vec<BigInt>,
        arithm_circuit: Option<Arithm>,
        ctt_circuit: Option<CTT>,
        wt_circuit: Option<WT>,
        rng: &mut R,
    ) -> Result<(Self::PP, Self::Table), Error>;

    fn prove<
        CTT: ConstraintSynthesizer<E::ScalarField>,
        WT: ConstraintSynthesizer<E::ScalarField>,
        R: RngCore + CryptoRng + Rng,
    >(
        pp: Self::PP,
        acc: Self::Accum,
        tree: Self::Table,
        lookup: Vec<BigInt>,
        elem: Vec<BigInt>,
        rand: Vec<BigInt>,
        ctt_circuit: Option<CTT>,
        wt_circuit: Option<WT>,
        rng: &mut R,
        non_proven_elem: Vec<BigInt>,
    ) -> Result<Self::Proof, Error>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug;

    fn verify(
        pp: Self::PP,
        acc: Self::Accum,
        cm_u: Self::CM,
        cm_a: Self::CM,
        cm_z: Self::CM,
        prf: Self::Proof,
    ) -> Result<bool, Error>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug;
}
