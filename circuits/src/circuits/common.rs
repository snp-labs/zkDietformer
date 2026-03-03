//! Shared circuit utilities used by both MLP and attention circuits.

use anyhow::Result;
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_std::{UniformRand, rand::{Rng, RngCore}};
use crate::{
    cc_snark::legogroth16::{LinkPublicGenerators, VerifyingKey},
    cc_snark::linker::{
        Linker,
        snark::{LinkSnark, PP},
        relation_generator::{generate_cp_factored_instance, generate_cp_factored_relation, generate_cp_factored_witness},
    },
    lasso::poly::{
        commitments::MultiCommitGens,
        dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds},
    },
};

/// Generates Pedersen commitment public generators for CP-Link.
pub fn get_link_public_gens<R: RngCore, E: Pairing>(
    rng: &mut R,
    count: u32,
) -> LinkPublicGenerators<E> {
    let pedersen_gens = (0..count)
        .map(|_| E::G1::rand(rng).into_affine())
        .collect::<Vec<_>>();
    let g1 = E::G1::rand(rng).into_affine();
    let g2 = E::G2::rand(rng).into_affine();
    LinkPublicGenerators {
        pedersen_gens,
        g1,
        g2,
    }
}

/// Proves and verifies the factored multi-Lasso CP-Link that connects
/// the Groth16 committed variables to the Lasso lookup polynomial commitments.
pub fn prove_and_verify_factored_multi_link<E: Pairing>(
    vk: &VerifyingKey<E>,
    num_pub: usize,
    v_groth: E::ScalarField,
    cm_groth: E::G1Affine,
    lasso_results: Vec<(PolyCommitment<E::G1>, PolyCommitmentBlinds<E::ScalarField>, MultiCommitGens<E::G1>)>,
    u_total: Vec<E::ScalarField>,
    rng: &mut impl Rng,
) -> Result<()> {
    let mut l_rows_vec = Vec::new();
    let mut r_cols_vec = Vec::new();
    let mut cks_lasso_vec = Vec::new();
    let mut rs = vec![v_groth];
    let mut lasso_cms_all = Vec::new();

    for (cm, blinds, gens) in &lasso_results {
        l_rows_vec.push(cm.C.len());
        r_cols_vec.push(gens.n);

        let mut ck = vec![gens.h.into_affine()];
        ck.extend(gens.G.iter().map(|g| g.into_affine()));
        cks_lasso_vec.push(ck);

        rs.extend(&blinds.blinds);
        lasso_cms_all.extend(cm.C.iter().map(|c| c.into_affine()));
    }

    let mut ck_joint = vec![vk.eta1_gamma_inv_g1];
    let start = 1 + num_pub;
    ck_joint.extend(vk.gamma_abc_g1[start..start + u_total.len()].iter().cloned());

    let crs = generate_cp_factored_relation::<E>(
        lasso_results.len(), l_rows_vec, r_cols_vec, ck_joint, cks_lasso_vec,
    );

    let pp = PP::new(crs.nr, crs.nc, &E::G1::rand(rng).into_affine(), &E::G2::rand(rng).into_affine());
    let (ek, vk_link) = LinkSnark::<E>::keygen(&pp, crs, rng);

    let witness = generate_cp_factored_witness::<E>(rs, u_total);
    let instance = generate_cp_factored_instance::<E>(cm_groth, lasso_cms_all);

    let (proof, _) = LinkSnark::<E>::prove(&pp, &ek, witness, rng);
    if LinkSnark::<E>::verify(&pp, &vk_link, &instance, &proof) {
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Factored link verification failed"))
    }
}
