use std::{marker::PhantomData, str::FromStr};

use crate::{
    cc_snark::cc_gro::{LibsnarkReduction, R1CSToQAP},
    harisa::Membership,
    linker::Linker,
    lookup::{
        data_structure::{LookupPP, LookupProof},
        Lookup,
    },
};

use ark_ec::pairing::Pairing;
use ark_relations::r1cs::ConstraintSynthesizer;
use ark_std::rand::{CryptoRng, Rng, RngCore};
use num_bigint::BigInt;

pub type Error = Box<dyn ark_std::error::Error>;

pub struct HarisaPlus<
    E: Pairing,
    M: Membership<E, LNK>,
    LNK: Linker<E>,
    QAP: R1CSToQAP = LibsnarkReduction,
> {
    _curve: PhantomData<(E, M, LNK, QAP)>,
}

impl<E: Pairing, M: Membership<E, LNK>, LNK: Linker<E>> Lookup<E, M, LNK>
    for HarisaPlus<E, M, LNK>
{
    type Accum = BigInt;
    type Table = M::Table;
    type CM = E::G1Affine;
    type PP = LookupPP<E, M, LNK>;
    type Proof = LookupProof<E, M, LNK>;

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
    ) -> Result<(Self::PP, Self::Table), Error> {
        let (pp, table) = Self::generate_lookup_parameters(
            set,
            ctt_circuit.unwrap(),
            wt_circuit.unwrap(),
            arithm_circuit.unwrap(),
            rng,
        )
        .unwrap();

        Ok((pp, table))
    }

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
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let prf = Self::generate_lookup_proof(
            pp,
            acc,
            tree,
            lookup,
            elem,
            rand,
            ctt_circuit.unwrap(),
            wt_circuit.unwrap(),
            rng,
            non_proven_elem,
        )
        .unwrap();

        Ok(prf)
    }

    fn verify(
        pp: Self::PP,
        acc: Self::Accum,
        cm_u: Self::CM,
        cm_a: Self::CM,
        cm_z: Self::CM,
        prf: Self::Proof,
    ) -> Result<bool, Error>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let res = Self::verify_lookup(pp, acc, cm_u, cm_a, cm_z, prf).unwrap();

        Ok(res)
    }
}
