use std::str::FromStr;

use crate::{
    cc_snark::cc_gro::{prepare_verifying_key, CcGroth16, R1CSToQAP},
    harisa::{Membership, cc_gro::Proof},
    linker::Linker,
    lookup::{
        data_structure::{LookupPP, LookupProof},
        lookup::HarisaPlus,
    },
};

use ark_crypto_primitives::snark::SNARK;
use ark_ec::pairing::Pairing;
use ark_relations::r1cs::SynthesisError;
use num_bigint::BigInt;
use ark_std::{end_timer,start_timer};

impl<E, M, LNK, QAP> HarisaPlus<E, M, LNK, QAP>
where
    E: Pairing,
    M: Membership<E, LNK>,
    LNK: Linker<E>,
    QAP: R1CSToQAP,
{
    pub fn verify_lookup(
        pp: LookupPP<E, M, LNK>,
        accum: BigInt,
        cm_u: E::G1Affine,
        cm_f: E::G1Affine,
        cm_z: E::G1Affine,
        proof: LookupProof<E, M, LNK>,
    ) -> Result<bool, SynthesisError>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let lookup_verify = start_timer!(|| "Harisa+::verify");
        let mem_verify = start_timer!(|| "mem::verify");
        let mem_result = M::verify(pp.m_pp, accum, cm_u, proof.m_prf).unwrap();
        end_timer!(mem_verify);

        let ctt_instance = LNK::generate_instance(
            // vec![proof.cm_f_hat, proof.cm_f_hat],
            vec![proof.cm_f_prime, proof.cm_f_prime],
            proof.ctt_prf.cm,
            proof.ctt_lnk_cm_aux,
        );

        let ctt_pvk = prepare_verifying_key(&pp.ctt_vk.clone());
        let ctt_verify = start_timer!(|| "cpctt::verify");
        let ctt_result = CcGroth16::<E, QAP>::verify_proof(&ctt_pvk, &proof.ctt_prf, &[]).unwrap();
        let ctt_lnk_result = LNK::verify(
            &pp.ctt_lnk_pp,
            &pp.ctt_lnk_vk,
            &ctt_instance,
            &proof.ctt_lnk_prf,
        );
        end_timer!(ctt_verify);

        let wt_instance = LNK::generate_instance(
            vec![proof.cm_f_hat, proof.cm_f, proof.cm_z],
            proof.wt_prf.cm,
            proof.wt_lnk_cm_aux,
        );

        let wt_pvk = prepare_verifying_key(&pp.wt_vk.clone());
        let wt_verify = start_timer!(|| "cpwt::verify");
        let wt_result = CcGroth16::<E, QAP>::verify_proof(&wt_pvk, &proof.wt_prf, &[]).unwrap();
        let wt_lnk_result = LNK::verify(
            &pp.wt_lnk_pp,
            &pp.wt_lnk_vk,
            &wt_instance,
            &proof.wt_lnk_prf,
        );
        end_timer!(wt_verify);
        end_timer!(lookup_verify);

        assert_eq!(mem_result, true, "[HARiSA] Membership Check Failed");
        assert_eq!(ctt_result, true, "[Copy this or that] Verification Failed");
        assert_eq!(ctt_lnk_result, true, "[Copy this or that] Linker Failed");
        assert_eq!(wt_result, true, "[Well Transformed] Verification Failed");
        assert_eq!(wt_lnk_result, true, "[Well Transformed] Linker Failed");

        Ok(true)
    }
}
