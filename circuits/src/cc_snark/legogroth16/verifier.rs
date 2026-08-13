use super::{PreparedVerifyingKey, ProofWithLink, VerifyingKeyWithLink};
use crate::cc_snark::legogroth16::link::{PESubspaceSnark, SubspaceSnark};
use crate::cc_snark::legogroth16::{Proof, Result, VerifyingKey, error::Error};
use ark_ec::VariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_ff::{One, PrimeField};
use ark_relations::r1cs::SynthesisError;
use ark_std::{cfg_iter, vec, vec::Vec};
use core::ops::{AddAssign, Neg};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Prepare the verifying key `vk` for use in proof verification.
pub fn prepare_verifying_key<E: Pairing>(vk: &VerifyingKey<E>) -> PreparedVerifyingKey<E> {
    PreparedVerifyingKey {
        vk: vk.clone(),
        alpha_g1_beta_g2: E::pairing(vk.alpha_g1, vk.beta_g2),
        gamma_g2_neg_pc: vk.gamma_g2.into_group().neg().into_affine().into(),
        delta_g2_neg_pc: vk.delta_g2.into_group().neg().into_affine().into(),
    }
}

/// Prepare proof inputs for use with [`verify_proof`], wrt the prepared
/// verification key `pvk` and instance public inputs.
pub fn prepare_inputs<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    public_inputs: &[E::ScalarField],
) -> Result<E::G1> {
    if (public_inputs.len() + 1) > pvk.vk.gamma_abc_g1.len() {
        return Err(SynthesisError::MalformedVerifyingKey).map_err(|e| e.into());
    }
    if public_inputs.len() > 2 {
        let mut inp = Vec::with_capacity(1 + public_inputs.len());
        inp.push(E::ScalarField::one());
        inp.extend_from_slice(public_inputs);
        let inp = cfg_iter!(inp).map(|a| a.into_bigint()).collect::<Vec<_>>();

        // Note: gamma_abc_g1 contains [PublicInputs | Group1 | Group2]
        // We only want to MSM the PublicInput part here.
        // Assuming public_inputs.len() matches the number of public inputs in VK.
        let num_public = inp.len();
        Ok(E::G1::msm_bigint(&pvk.vk.gamma_abc_g1[0..num_public], &inp))
    } else {
        let mut d = pvk.vk.gamma_abc_g1[0].into_group();
        for (i, b) in public_inputs.iter().zip(pvk.vk.gamma_abc_g1.iter().skip(1)) {
            d.add_assign(&b.mul_bigint(i.into_bigint()));
        }
        Ok(d)
    }
}

/// Verify the proof of the Subspace Snark on the equality of openings of cp_link and proof.d1
pub fn verify_link_proof<E: Pairing>(
    vk: &VerifyingKeyWithLink<E>,
    proof: &ProofWithLink<E>,
) -> Result<()> {
    // Note: Link is checked against d1
    let commitments = vec![proof.link_d.clone(), proof.groth16_proof.d1.clone()];
    PESubspaceSnark::<E>::verify(&vk.link_pp, &vk.link_vk, &commitments, &proof.link_pi)
        .map_err(|e| e.into())
}

pub fn verify_qap_proof<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    a: E::G1Affine,
    b: E::G2Affine,
    c: E::G1Affine,
    d: E::G1Affine,
) -> Result<()> {
    let qap = E::multi_miller_loop(
        [a, c, d],
        [
            b.into(),
            pvk.delta_g2_neg_pc.clone(),
            pvk.gamma_g2_neg_pc.clone(),
        ],
    );
    if E::final_exponentiation(qap).ok_or(SynthesisError::UnexpectedIdentity)?
        != pvk.alpha_g1_beta_g2
    {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Verify a LegoGroth16 proof `proof` against the prepared verification key `pvk`
pub fn verify_proof<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs: &[E::ScalarField],
) -> Result<()> {
    // We pass the aggregated D (Public + D1 + D2) to the QAP verification
    verify_qap_proof(
        pvk,
        proof.a,
        proof.b,
        proof.c,
        calculate_d(pvk, proof, public_inputs)?,
    )
}

pub fn calculate_d<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs: &[E::ScalarField],
) -> Result<E::G1Affine> {
    // Accumulate public inputs * gamma_abc
    let mut d = prepare_inputs(pvk, public_inputs)?;
    d += proof.d1;
    d += proof.d2;
    d += proof.d3;

    Ok(d.into_affine())
}

/// Verify a LegoGroth16 proof `proof` against the prepared verification key `pvk`
pub fn verify_proof_incl_cp_link<E: Pairing>(
    pvk: &PreparedVerifyingKey<E>,
    vk: &VerifyingKeyWithLink<E>,
    proof: &ProofWithLink<E>,
    public_inputs: &[E::ScalarField],
) -> Result<()> {
    verify_link_proof(vk, proof)?;
    verify_proof(pvk, &proof.groth16_proof, public_inputs)
}
