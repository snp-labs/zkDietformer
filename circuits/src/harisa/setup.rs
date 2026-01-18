use std::marker::PhantomData;
use std::str::FromStr;

use crate::linker::Linker;
use crate::ConstraintF;
use crate::{
    cc_snark::cc_gro::{
        data_structure::{ProvingKey, VerifyingKey},
        r1cs_to_qap::R1CSToQAP,
        CcGroth16,
    },
    harisa::constants::RSA_2048,
};

use super::data_structure::HarisaPP;
use super::harisa::Harisa;
use super::precompute::*;

use ark_crypto_primitives::snark::*;
use ark_ec::pairing::Pairing;
use ark_r1cs_std::pairing::PairingVar;
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError};
use ark_std::{
    rand::{CryptoRng, Rng, RngCore},
    UniformRand, start_timer, end_timer
};
use num_bigint::BigInt;

impl<E: Pairing, LNK: Linker<E>, QAP: R1CSToQAP> Harisa<E, LNK, QAP> {
    pub fn generate_link_parameters<R: RngCore + CryptoRng>(
        n: usize,
        ck: Vec<E::G1Affine>,
        snark_ck: Vec<E::G1Affine>,
        mode: &str,
        rng: &mut R,
    ) -> Result<(LNK::PP, LNK::EK, LNK::VK), SynthesisError> {
        let (link_pp, link_crs) = LNK::setup(n, ck, snark_ck, mode, rng);
        let (link_ek, link_vk) = LNK::keygen(&link_pp.clone(), link_crs, rng);

        Ok((link_pp, link_ek, link_vk))
    }

    pub fn generate_cc_snark_parameters<
        C: ConstraintSynthesizer<E::ScalarField>,
        R: RngCore + CryptoRng,
    >(
        circuit: C,
        rng: &mut R,
    ) -> Result<(ProvingKey<E>, VerifyingKey<E>), SynthesisError> {
        let cc_snark_generator = start_timer!(|| "ccGroth::Generator");
        let (cc_ek, cc_vk) = CcGroth16::<E>::circuit_specific_setup(circuit, rng).unwrap();
        end_timer!(cc_snark_generator);
        Ok((cc_ek, cc_vk))
    }

    pub fn generate_harisa_parameters<
        Arithm: ConstraintSynthesizer<E::ScalarField>,
        R: RngCore + CryptoRng + Rng,
    >(
        set: Vec<BigInt>,
        arithm_circuit: Arithm,
        rng: &mut R,
    ) -> Result<(HarisaPP<E, LNK>, Vec<BigInt>, Vec<E::G1Affine>), SynthesisError> {
        let num = set.len();
        let harisa_generation = start_timer!(|| "HARiSA::Generator");

        let mut ck = Vec::new();

        for _ in 0..num + 1 {
            ck.push(E::G1Affine::rand(rng));
        }

        let arithm_generation = start_timer!(|| "arithm::generator");
        let (arithm_ek, arithm_vk) =
            Self::generate_cc_snark_parameters(arithm_circuit, rng).unwrap();

        let (arithm_lnk_pp, arithm_lnk_ek, arithm_lnk_vk) = Self::generate_link_parameters(
            arithm_ek.ck.len() - 6,
            ck.clone(),
            arithm_ek.ck.as_slice().to_vec(),
            "arithm",
            rng,
        )
        .unwrap();
        end_timer!(arithm_generation);

        end_timer!(harisa_generation);

        let (g, mod_n) = rsa_setup();

        let preprocessing = start_timer!(|| "HARiSA::Preprocess");
        let table = precompute(g.clone(), mod_n.clone(), set);
        end_timer!(preprocessing);

        Ok((
            HarisaPP {
                arithm_ek: arithm_ek.clone(),
                arithm_vk: arithm_vk.clone(),
                arithm_lnk_pp,
                arithm_lnk_ek,
                arithm_lnk_vk,
                g: g.clone(),
                mod_n: mod_n.clone(),
                ck: ck.clone(),
            },
            table,
            ck,
        ))
    }
}
