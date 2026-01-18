use crate::{
    cc_snark::cc_gro::{prepare_verifying_key, CcGroth16, ProvingKey, VerifyingKey},
    harisa::{harisa::Harisa, Membership},
    linker::Linker,
    lookup::{
        data_structure::{LookupPP, LookupProof},
        lookup::HarisaPlus,
    },
};

use ark_crypto_primitives::snark::SNARK;
use ark_ec::pairing::Pairing;
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError};
use ark_std::{
    rand::{CryptoRng, Rng, RngCore},
    UniformRand, end_timer, start_timer
};
use num_bigint::BigInt;

impl<E, M, LNK> HarisaPlus<E, M, LNK>
where
    E: Pairing,
    M: Membership<E, LNK>,
    LNK: Linker<E>,
{
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

    pub fn generate_lookup_parameters<
        CTT: ConstraintSynthesizer<E::ScalarField>,
        WT: ConstraintSynthesizer<E::ScalarField>,
        Arithm: ConstraintSynthesizer<E::ScalarField>,
        R: Rng + RngCore + CryptoRng,
    >(
        set: Vec<BigInt>,
        ctt_circuit: CTT,
        wt_circuit: WT,
        arithm_circuit: Arithm,
        rng: &mut R,
    ) -> Result<(LookupPP<E, M, LNK>, M::Table), SynthesisError> {
        let num = set.len();

        let lookup_generation = start_timer!(|| "HARiSA+::Generator");

        let (m_pp, tree, ck) = M::setup(set, arithm_circuit, rng).unwrap();

        let ctt_generation = start_timer!(|| "ctt::generator");
        let (ctt_ek, ctt_vk) = Self::generate_cc_snark_parameters(ctt_circuit, rng).unwrap();

        let (ctt_lnk_pp, ctt_lnk_ek, ctt_lnk_vk) = Self::generate_link_parameters(
            (ctt_ek.ck.len() - 1) / 2,
            ck.clone(),
            ctt_ek.ck.as_slice().to_vec(),
            "ctt",
            rng,
        )
        .unwrap();
        end_timer!(ctt_generation);

        let wt_generation = start_timer!(|| "wt::generator");
        let (wt_ek, wt_vk) = Self::generate_cc_snark_parameters(wt_circuit, rng).unwrap();

        let (wt_lnk_pp, wt_lnk_ek, wt_lnk_vk) = Self::generate_link_parameters(
            (wt_ek.ck.len() - 1) / 3,
            ck.clone(),
            wt_ek.ck.as_slice().to_vec(),
            "wt",
            rng,
        )
        .unwrap();
        end_timer!(wt_generation);

        end_timer!(lookup_generation);

        Ok((
            LookupPP {
                m_pp,
                ctt_ek,
                ctt_vk,
                ctt_lnk_pp,
                ctt_lnk_ek,
                ctt_lnk_vk,
                wt_ek,
                wt_vk,
                wt_lnk_pp,
                wt_lnk_ek,
                wt_lnk_vk,
                ck,
            },
            tree,
        ))
    }
}
