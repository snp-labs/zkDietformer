use std::str::FromStr;

use crate::{
    cc_snark::cc_gro::{CcGroth16, Proof, ProvingKey, R1CSToQAP},
    harisa::{harisa::Harisa, type_conversion::bigint_to_fr, Membership},
    linker::{matrix::inner_product, Linker},
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
    UniformRand,start_timer, end_timer
};
use num_bigint::BigInt;

impl<E, M, LNK, QAP> HarisaPlus<E, M, LNK, QAP>
where
    E: Pairing,
    M: Membership<E, LNK>,
    LNK: Linker<E>,
    QAP: R1CSToQAP,
{
    fn generate_link_proof<R>(
        pp: LNK::PP,
        ek: LNK::EK,
        rand: Vec<E::ScalarField>,
        witness: Vec<E::ScalarField>,
        snark_witness: Vec<E::ScalarField>,
        rng: &mut R,
    ) -> Result<(LNK::Proof, LNK::CM), SynthesisError>
    where
        R: Rng + RngCore + CryptoRng,
    {
        let link_witness = LNK::generate_witness(rand, witness, snark_witness);

        let (link_prf, link_cm) = LNK::prove(&pp, &ek, link_witness, rng);

        Ok((link_prf, link_cm))
    }

    fn generate_cc_proof<C, R>(
        pk: &ProvingKey<E>,
        circuit: C,
        rng: &mut R,
    ) -> Result<Proof<E>, SynthesisError>
    where
        C: ConstraintSynthesizer<E::ScalarField>,
        R: Rng + RngCore + CryptoRng,
    {
        let cc_snark_prover_time = start_timer!(|| "ccGroth::Prover");

        let cc_prf = CcGroth16::<E, QAP>::prove(&pk, circuit, rng).unwrap();

        end_timer!(cc_snark_prover_time);

        Ok(cc_prf)
    }

    pub fn generate_lookup_proof<
        CTT: ConstraintSynthesizer<E::ScalarField>,
        WT: ConstraintSynthesizer<E::ScalarField>,
        R: Rng + RngCore + CryptoRng,
    >(
        pp: LookupPP<E, M, LNK>,
        accum: BigInt,
        tree: M::Table,
        lookup: Vec<BigInt>,
        elem: Vec<BigInt>,
        rand: Vec<BigInt>,
        ctt_circuit: CTT,
        wt_circuit: WT,
        rng: &mut R,
        non_proven_elem: Vec<BigInt>,
    ) -> Result<LookupProof<E, M, LNK>, SynthesisError>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let mut f_hat_scalar: Vec<E::ScalarField> = Vec::new();
        for u_i in lookup.clone() {
            f_hat_scalar.push(bigint_to_fr(u_i));
        }

        let o_f_hat = E::ScalarField::rand(rng);

        let mut cm_f_hat = (pp.ck[0].clone() * o_f_hat).into();

        for (g_i, u_i) in pp
            .ck
            .clone()
            .iter()
            .skip(1)
            .zip(f_hat_scalar.clone().into_iter())
        {
            cm_f_hat = (cm_f_hat + *g_i * u_i).into();
        }

        let mut f_scalar: Vec<E::ScalarField> = Vec::new();
        for f_i in elem.clone() {
            f_scalar.push(bigint_to_fr(f_i));
        }

        let o_f = E::ScalarField::rand(rng);

        let mut cm_f = (pp.ck[0].clone() * o_f).into();

        for (g_i, u_i) in pp
            .ck
            .clone()
            .iter()
            .skip(1)
            .zip(f_scalar.clone().into_iter())
        {
            cm_f = (cm_f + *g_i * u_i).into();
        }

        let mut z_scalar: Vec<E::ScalarField> = Vec::new();
        for u_i in rand.clone() {
            z_scalar.push(bigint_to_fr(u_i));
        }

        let o_z = E::ScalarField::rand(rng);

        let mut cm_z = (pp.ck[0].clone() * o_z).into();

        for (g_i, u_i) in pp
            .ck
            .clone()
            .iter()
            .skip(1)
            .zip(z_scalar.clone().into_iter())
        {
            cm_z = (cm_z + *g_i * u_i).into();
        }

        let harisa_elem = [lookup, rand].concat();

        let harisa_scalar = [f_hat_scalar.clone(), z_scalar.clone()].concat();

        let o_f_prime = E::ScalarField::rand(rng);

        let mut cm_f_prime = (pp.ck[0].clone() * o_f_prime).into();

        for (g_i, u_i) in pp
            .ck
            .clone()
            .iter()
            .skip(1)
            .zip(harisa_scalar.clone().into_iter())
        {
            cm_f_prime = (cm_f_prime + *g_i * u_i).into();
        }

        let lookup_prover = start_timer!(|| "Harisa+::prove");
        // let m_prf = M::prove(pp.m_pp, tree, accum, cm_f_hat, lookup, o_f_hat, rng).unwrap();
        let m_prf = M::prove(
            pp.m_pp,
            tree,
            accum,
            cm_f_prime,
            harisa_elem,
            o_f_prime,
            rng,
            true,
        )
        .unwrap();

        let ctt_prove = start_timer!(|| "cpctt::prove");

        let ctt_prf = Self::generate_cc_proof(&pp.ctt_ek, ctt_circuit, rng).unwrap();

        let (ctt_lnk_prf, ctt_lnk_cm_aux) = Self::generate_link_proof(
            pp.ctt_lnk_pp.clone(),
            pp.ctt_lnk_ek.clone(),
            vec![o_f_prime, o_f_prime],
            [harisa_scalar.clone(), harisa_scalar.clone()].concat(),
            vec![ctt_prf.open],
            rng,
        )
        .unwrap();

        end_timer!(ctt_prove);

        let wt_prove = start_timer!(|| "cpwt::prove");

        let wt_prf = Self::generate_cc_proof(&pp.wt_ek, wt_circuit, rng).unwrap();

        let (wt_lnk_prf, wt_lnk_cm_aux) = Self::generate_link_proof(
            pp.wt_lnk_pp.clone(),
            pp.wt_lnk_ek.clone(),
            vec![o_f_hat, o_f, o_z],
            [f_hat_scalar, f_scalar, z_scalar].concat(),
            vec![wt_prf.open],
            rng,
        )
        .unwrap();

        end_timer!(wt_prove);
        end_timer!(lookup_prover);

        Ok(LookupProof {
            m_prf,
            ctt_prf,
            ctt_lnk_prf,
            ctt_lnk_cm_aux,
            wt_prf,
            wt_lnk_prf,
            wt_lnk_cm_aux,
            cm_f_prime,
            cm_f_hat,
            cm_f,
            cm_z,
        })
    }
}
