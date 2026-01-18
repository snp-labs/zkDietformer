use crate::cc_snark::cc_gro::CcGroth16;
use crate::harisa::constants::*;
use crate::harisa::hash_to_prime::{hash_to_prime, round_keys_contants_to_vec};
use crate::linker::Linker;
use crate::ConstraintF;

use ark_ec::pairing::Pairing;
use ark_r1cs_std::pairing::PairingVar;
use ark_relations::r1cs::SynthesisError;
use ark_std::{One, start_timer, end_timer};
use core::ops::{AddAssign, MulAssign};
use num_bigint::BigInt;
use std::str::FromStr;

use super::cc_gro::prepare_verifying_key;
use super::cc_gro::r1cs_to_qap::R1CSToQAP;
use super::{
    data_structure::{HarisaPP, HarisaProof},
    harisa::Harisa,
};

impl<E: Pairing, LNK: Linker<E>, QAP: R1CSToQAP> Harisa<E, LNK, QAP> {
    pub fn harisa_verify(
        pp: HarisaPP<E, LNK>,
        accum: BigInt,
        cm_u: E::G1Affine,
        proof: HarisaProof<E, LNK>,
    ) -> Result<bool, SynthesisError>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let harisa_verifier = start_timer!(|| "Harisa::verify");
        // pstar
        let mut p_star = BigInt::one();

        let mut p = Vec::new();

        for i in 0..ODD_PRIME.len() {
            let p_i = BigInt::from(ODD_PRIME[i]);
            p.push(p_i.clone());
            p_star *= p_i;
        }

        let accum_hat = accum.clone().modpow(&p_star, &pp.mod_n.clone());

        // hash h
        let constants = round_keys_contants_to_vec::<E::ScalarField>(&MIMC_7_91_BN254_ROUND_KEYS);
        let mut h = hash_to_prime(accum.clone(), proof.w_hat.clone(), &constants);
        h = hash_to_prime(h, proof.r.clone(), &constants);

        // 1. acc_hat = acc^{h * prod_pi} + R
        let poke_verify = start_timer!(|| "PoKE::verify");
        let acc_hat = (accum_hat.modpow(&h, &pp.mod_n.clone()) * proof.r) % pp.mod_n.clone();

        let l = hash_to_prime(proof.w_hat.clone(), acc_hat.clone(), &constants);

        assert!(proof.k.clone() >= BigInt::from(0), "[PoKE] Wrong range (k)");
        assert!(proof.k.clone() < l.clone(), "[PoKE] Wrong range (k)");

        // PoKE verify
        assert_eq!(
            (proof.q.modpow(&l, &pp.mod_n.clone())
                * (proof.w_hat.modpow(&proof.k, &pp.mod_n.clone())))
                % pp.mod_n,
            acc_hat,
            "[PoKE] Verification Failed"
        );
        end_timer!(poke_verify);

        let arithm_instance = LNK::generate_instance(
            vec![proof.cm_u, proof.cm_sr],
            proof.arithm_lnk_cm,
            proof.arithm_lnk_cm_aux,
        );

        let arithm_pvk = prepare_verifying_key(&pp.arithm_vk.clone());
        let arithm_verify = start_timer!(|| "cparithm::verify");
        let arithm_result =
            CcGroth16::<E, QAP>::verify_proof(&arithm_pvk, &proof.arithm_prf, &[]).unwrap();
        let arithm_link_result = LNK::verify(
            &pp.arithm_lnk_pp,
            &pp.arithm_lnk_vk,
            &arithm_instance,
            &proof.arithm_lnk_prf,
        );
        end_timer!(arithm_verify);

        end_timer!(harisa_verifier);

        assert_eq!(arithm_result, true, "[Arithm] Verification Failed");
        assert_eq!(arithm_link_result, true, "[Arithm] Linker Failed");

        Ok(true)
    }
}
