use anyhow::Result;
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_std::{UniformRand, rand::Rng, Zero};
use crate::{
    cc_snark::legogroth16::VerifyingKey,
    cc_snark::linker::{
        Linker,
        snark::{EK, LinkSnark, PP, VK},
        matrix::SparseMatrix,
        relation_generator::{
            generate_cp_factored_instance,
            generate_cp_factored_relation,
            generate_cp_factored_witness,
        },
    },
    lasso::poly::{
        commitments::MultiCommitGens,
        dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds},
    },
};


type LassoResult<E> = (
    PolyCommitment<<E as Pairing>::G1>,
    PolyCommitmentBlinds<<E as Pairing>::ScalarField>,
    MultiCommitGens<<E as Pairing>::G1>,
    usize, // actual sparsity before rounding to the next power of two
);

// ── BenchmarkResult ─────────────────────────────────────────────────────────

/// Holds all performance metrics for one witness file's proving pipeline.
///
/// Lasso helpers bundle prove+verify internally, so `lasso_ms` covers both.
/// `prove_and_verify_factored_multi_link` bundles setup+keygen+prove+verify,
/// so `cp_link_ms` covers all four phases. Verify-only times for these are
/// reported as 0 because they cannot be separated without modifying the helpers.
///
/// Call `print_json()` to emit a single `BENCHMARK_RESULT` JSON line:
///
/// ```
/// grep 'BENCHMARK_RESULT' test.log | jq '.prove.total_ms'
/// ```
pub struct BenchmarkResult {
    pub file: String,
    pub circuit: String,        // "mlp" or "attention"
    pub constraints: usize,
    pub setup_ms: u128,
    pub proof_size_bytes: usize,
    pub groth16_prove_ms: u128,
    pub lasso_ms: u128,         // prove + verify bundled inside Lasso helpers
    pub cp_link_ms: u128,       // setup + keygen + prove + verify bundled
    pub groth16_verify_ms: u128,
    pub end_to_end_ms: u128,
}

impl BenchmarkResult {
    /// Prints a single-line JSON record prefixed with `BENCHMARK_RESULT`.
    /// Parse with: `grep BENCHMARK_RESULT output.log | jq .`
    pub fn print_json(&self) {
        let total_prove_ms  = self.groth16_prove_ms + self.lasso_ms + self.cp_link_ms;
        let total_verify_ms = self.groth16_verify_ms;
        println!(
            concat!(
                "BENCHMARK_RESULT ",
                "{{",
                "\"file\":{file:?},",
                "\"circuit\":{circuit:?},",
                "\"constraints\":{c},",
                "\"setup_ms\":{s},",
                "\"proof_size_bytes\":{p},",
                "\"prove\":{{\"groth16_ms\":{g16p},\"lasso_ms\":{lp},\"cp_link_ms\":{cp},\"total_ms\":{tp}}},",
                "\"verify\":{{\"groth16_ms\":{g16v},\"lasso_ms\":0,\"cp_link_ms\":0,\"total_ms\":{tv}}},",
                "\"end_to_end_ms\":{e2e},",
                "\"memory_mb\":null",
                "}}"
            ),
            file    = self.file,
            circuit = self.circuit,
            c       = self.constraints,
            s       = self.setup_ms,
            p       = self.proof_size_bytes,
            g16p    = self.groth16_prove_ms,
            lp      = self.lasso_ms,
            cp      = self.cp_link_ms,
            tp      = total_prove_ms,
            g16v    = self.groth16_verify_ms,
            tv      = total_verify_ms,
            e2e     = self.end_to_end_ms,
        );
    }
}

pub fn factored_link_setup<E: Pairing, R: Rng>(
    vk: &VerifyingKey<E>,
    num_pub: usize,
    u_len: usize,
    lasso_results: &[LassoResult<E>],
    rng: &mut R,
) -> (PP<E::G1Affine, E::G2Affine>, SparseMatrix<E::G1Affine>) {
    let mut l_rows_vec = Vec::new();
    let mut r_cols_vec = Vec::new();
    let mut actual_sparsities = Vec::new();
    let mut cks_lasso_vec = Vec::new();

    for (cm, _, gens, actual_sp) in lasso_results {
        l_rows_vec.push(cm.C.len());
        r_cols_vec.push(gens.n);
        actual_sparsities.push(*actual_sp);
        let mut ck = vec![gens.h.into_affine()];
        ck.extend(gens.G.iter().map(|g| g.into_affine()));
        cks_lasso_vec.push(ck);
    }

    // Reuse Groth16 commitment keys: the same bases used to compute proof.d1.
    let mut ck_joint = vec![vk.eta1_gamma_inv_g1];
    let start = 1 + num_pub;
    ck_joint.extend(vk.gamma_abc_g1[start..start + u_len].iter().cloned());

    let crs = generate_cp_factored_relation::<E>(
        lasso_results.len(),
        l_rows_vec,
        r_cols_vec,
        actual_sparsities,
        ck_joint,
        cks_lasso_vec,
    );

    // Fresh random generators for the link SNARK's pairing check.
    let pp = PP::new(
        crs.nr,
        crs.nc,
        &E::G1::rand(rng).into_affine(),
        &E::G2::rand(rng).into_affine(),
    );
    (pp, crs)
}


pub fn factored_link_keygen<E: Pairing, R: Rng>(
    pp: &PP<E::G1Affine, E::G2Affine>,
    crs: SparseMatrix<E::G1Affine>,
    rng: &mut R,
) -> (EK<E::G1Affine>, VK<E::G2Affine>) {
    LinkSnark::<E>::keygen(pp, crs, rng)
}


pub fn factored_link_prove<E: Pairing, R: Rng>(
    pp: &PP<E::G1Affine, E::G2Affine>,
    ek: &EK<E::G1Affine>,
    v_groth: E::ScalarField,
    cm_groth: E::G1Affine,
    lasso_results: &[LassoResult<E>],
    u_total: Vec<E::ScalarField>,
    rng: &mut R,
) -> (E::G1Affine, Vec<E::G1Affine>) {
    let mut rs = vec![v_groth];
    let mut lasso_cms_all = Vec::new();
    for (cm, blinds, _, _) in lasso_results {
        rs.extend(&blinds.blinds);
        lasso_cms_all.extend(cm.C.iter().map(|c| c.into_affine()));
    }

    let witness = generate_cp_factored_witness::<E>(rs, u_total);
    let instance = generate_cp_factored_instance::<E>(cm_groth, lasso_cms_all);
    let (proof, _) = LinkSnark::<E>::prove(pp, ek, witness, rng);
    (proof, instance)
}


pub fn factored_link_verify<E: Pairing>(
    pp: &PP<E::G1Affine, E::G2Affine>,
    vk_link: &VK<E::G2Affine>,
    instance: &Vec<E::G1Affine>,
    proof: &E::G1Affine,
) -> bool {
    LinkSnark::<E>::verify(pp, vk_link, instance, proof)
}


pub fn prove_and_verify_factored_multi_link<E: Pairing>(
    vk: &VerifyingKey<E>,
    num_pub: usize,
    v_groth: E::ScalarField,
    cm_groth: E::G1Affine,
    lasso_results: Vec<LassoResult<E>>,
    u_total: Vec<E::ScalarField>,
    rng: &mut impl Rng,
) -> Result<()> {
    // Build a padded u_total that has exactly total_msg_len elements.
    //
    // The Lasso polynomials are zero-padded internally to the next power of two
    // (s = actual_sparsity.next_power_of_two()).  The Groth16 committed_group1
    // only stores the actual (unpadded) values.  For CP-Link the witness vector
    // must cover all s_i columns (including padding zeros), so we interleave zeros
    // at the padding positions of each result.
    let mut u_padded: Vec<E::ScalarField> = Vec::new();
    let mut committed_offset = 0usize;
    for (cm, _, gens, actual_sparsity) in &lasso_results {
        let s = cm.C.len() * gens.n; // rounded sparsity (= l_size * r_size)
        u_padded.extend_from_slice(&u_total[committed_offset..committed_offset + actual_sparsity]);
        u_padded.extend(std::iter::repeat(E::ScalarField::zero()).take(s - actual_sparsity));
        committed_offset += actual_sparsity;
    }

    // [1] Setup
    let (pp, crs) =
        factored_link_setup::<E, _>(vk, num_pub, u_total.len(), &lasso_results, rng);

    // [2] KeyGen
    let (ek, vk_link) = factored_link_keygen::<E, _>(&pp, crs, rng);

    // [3] Prove
    let (proof, instance) =
        factored_link_prove::<E, _>(&pp, &ek, v_groth, cm_groth, &lasso_results, u_padded, rng);

    // [4] Verify
    if factored_link_verify::<E>(&pp, &vk_link, &instance, &proof) {
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Factored link verification failed"))
    }
}
