use std::str::FromStr;

use super::{
    arithm::ArithmCircuit,
    constants::{MIMC_7_91_BN254_ROUND_KEYS, RSA_2048},
    data_structure::{HarisaPP, HarisaProof},
    harisa::Harisa,
    hash_to_prime::{gen_bigint_range, hash_to_prime, round_keys_contants_to_vec},
    precompute::*,
    cc_gro::r1cs_to_qap::LibsnarkReduction,
    type_conversion::*,
};
use crate::{
    cc_snark::cc_gro::{
        data_structure::{Proof, ProvingKey},
        r1cs_to_qap::R1CSToQAP,
        CcGroth16,
    },
    harisa::constants::ODD_PRIME,
    linker::{matrix::inner_product, snark::LinkSnark, Linker},
};

use ark_crypto_primitives::snark::*;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField};
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisError,
};
use ark_std::{
    rand::{CryptoRng, Rng, RngCore},
    One, UniformRand, Zero, start_timer, end_timer
};
use num_bigint::BigInt;

impl<E: Pairing, LNK: Linker<E>, QAP: R1CSToQAP> Harisa<E, LNK, QAP> {
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

    pub fn generate_harisa_proof<R: RngCore + CryptoRng + Rng>(
        pp: HarisaPP<E, LNK>,
        accum: BigInt,
        w: BigInt,
        cm_u: E::G1Affine,
        u: Vec<BigInt>,
        o_u: E::ScalarField,
        rng: &mut R,
    ) -> Result<HarisaProof<E, LNK>, SynthesisError>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let harisa_prover = start_timer!(|| "Harisa::prove");
        // pstar
        let mut p_star = BigInt::one();

        let mut p = Vec::new();

        for i in 0..ODD_PRIME.len() {
            let p_i = BigInt::from(ODD_PRIME[i]);
            p.push(p_i.clone());
            p_star *= p_i;
        }

        // accum_hat
        let accum_hat = accum.clone().modpow(&p_star, &pp.mod_n.clone());

        // ustar
        let mut u_star = BigInt::one();
        for u_i in u.clone() {
            u_star *= u_i;
        }

        // sample b
        let b_rand = E::ScalarField::rand(rng);
        let mut b_bits = b_rand.into_bigint().to_bits_le();
        b_bits.truncate(u.clone().into_iter().len());

        for _ in b_bits.len()..p.len() {
            b_bits.push(false);
        }

        b_bits.reverse();

        // calculate s, s_bar
        let (mut s, mut s_bar) = (BigInt::one(), BigInt::one());

        for (p_i, b_bits_i) in p.clone().iter().zip(b_bits.clone().into_iter()) {
            match b_bits_i {
                false => s_bar *= p_i,
                true => s *= p_i,
            };
        }

        // calculate w_hat
        let w_hat = w.modpow(&s_bar, &pp.mod_n.clone());

        // sample r
        let r_rand = gen_bigint_range(rng, &BigInt::from(2), &(pp.mod_n.clone() - 1));

        // calculate R
        let r = w_hat.clone().modpow(&r_rand.clone(), &pp.mod_n.clone());

        // hash h
        let constants = round_keys_contants_to_vec::<E::ScalarField>(&MIMC_7_91_BN254_ROUND_KEYS);

        let mut h = hash_to_prime(accum.clone(), w_hat.clone(), &constants);
        h = hash_to_prime(h, r.clone(), &constants);

        // calculate k
        let k = r_rand.clone() + u_star.clone() * s.clone() * h.clone();

        // PoKE => prf1
        let large_b =
            (accum_hat.modpow(&h.clone(), &pp.mod_n.clone()) * r.clone()) % pp.mod_n.clone();

        let poke_prove = start_timer!(|| "PoKE::prove");

        let l = hash_to_prime(w_hat.clone(), large_b.clone(), &constants);

        let quot = k.clone() / l.clone();
        let rem = k.clone() % l.clone();
        let q = w_hat.clone().modpow(&quot.clone(), &pp.mod_n.clone());

        end_timer!(poke_prove);

        let mut circuit_u = Vec::new();

        for u_i in u.clone() {
            circuit_u.push(bigint_to_fr(u_i));
        }

        let circuit_h = bigint_to_fr(h);
        let circuit_l = bigint_to_fr(l);
        let circuit_k = bigint_to_fr(k);
        let circuit_s = bigint_to_fr(s);
        let circuit_r = bigint_to_fr(r_rand);

        let small_prime = E::ScalarField::from(ODD_PRIME[255]);

        // arithm => prf2
        let arithm_circuit = ArithmCircuit::<E::ScalarField>::new(
            circuit_h,
            circuit_l,
            circuit_k,
            circuit_u.clone(),
            circuit_s,
            circuit_r,
        );

        let arithm_prove = start_timer!(|| "cparithm::prove");
        let arithm_prf =
            Self::generate_cc_proof(&pp.arithm_ek.clone(), arithm_circuit, rng).unwrap();

        let o_sr = E::ScalarField::rand(rng);

        let cm_sr = (pp.ck[1].clone() * circuit_s.clone()
            + pp.ck[2].clone() * circuit_r.clone()
            + pp.ck[0].clone() * o_sr)
            .into();

        let arithm_witness = [
            vec![arithm_prf.open],
            circuit_u.clone(),
            vec![circuit_s, circuit_r],
        ]
        .concat();

        let mut arithm_ck = pp.arithm_ek.ck.clone();

        arithm_ck.truncate(u.clone().len() + 3);

        let arithm_lnk_cm = inner_product::<E>(arithm_witness.as_slice(), arithm_ck.as_slice());

        let (arithm_lnk_prf, arithm_lnk_cm_aux) = Self::generate_link_proof(
            pp.arithm_lnk_pp.clone(),
            pp.arithm_lnk_ek.clone(),
            vec![o_u, o_sr],
            [
                circuit_u.clone(),
                vec![circuit_s.clone(), circuit_r.clone()],
            ]
            .concat(),
            vec![arithm_prf.open],
            rng,
        )
        .unwrap();
        end_timer!(arithm_prove);

        end_timer!(harisa_prover);

        Ok(HarisaProof {
            cm_u,
            cm_sr,
            w_hat,
            r,
            q,
            k: rem,
            arithm_prf,
            arithm_lnk_prf,
            arithm_lnk_cm,
            arithm_lnk_cm_aux,
        })
    }

    pub fn generate_harisa_opt_proof<R: RngCore + CryptoRng + Rng>(
        pp: HarisaPP<E, LNK>,
        tree: Vec<BigInt>,
        accum: BigInt,
        cm_u: E::G1Affine,
        u: Vec<BigInt>,
        o_u: E::ScalarField,
        rng: &mut R,
        is_lookup: bool,
    ) -> Result<HarisaProof<E, LNK>, SynthesisError>
    where
        <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
        let mut w: Vec<BigInt> = Vec::new();

        let u_len = u.len();

        for i in 0..u_len {
            w.push(tree[i].clone());
        }

        let mut u_vec = u.clone();

        let mut w_len = u_len;

        while w_len > 1 {
            w_len >>= 1;

            for i in 0..w_len {
                (w[i], u_vec[i]) = assemble(
                    pp.mod_n.clone(),
                    u_vec[2 * i].clone(),
                    u_vec[2 * i + 1].clone(),
                    w[2 * i].clone(),
                    w[2 * i + 1].clone(),
                );
            }
            w.truncate(w_len);
            u_vec.truncate(w_len);
        }

        let w_u = w.first().unwrap();

        let proof = Self::generate_harisa_proof(pp, accum, w_u.clone(), cm_u, u, o_u, rng).unwrap();

        Ok(proof)
    }
}
