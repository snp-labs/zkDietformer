use anyhow::Result;
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_serialize::{CanonicalSerialize, Compress};
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

/// Compressed serialized size of the Groth16 Pedersen commitment (`cm1`).
pub fn groth16_commitment_size<E: Pairing>(cm1: &E::G1Affine) -> usize {
    cm1.serialized_size(Compress::Yes)
}

/// Compressed serialized size of all Lasso polynomial commitments combined.
pub fn lasso_commitment_size<E: Pairing>(lasso_results: &[LassoResult<E>]) -> usize {
    lasso_results
        .iter()
        .map(|(poly_cm, _, _, _)| poly_cm.serialized_size(Compress::Yes))
        .sum()
}

// ── Lasso Metrics (per-invocation) ──────────────────────────────────────────

/// Fine-grained timing and size metrics for a single Lasso prove invocation.
#[derive(Debug, Clone, Default)]
pub struct LassoMetrics {
    pub commit_proof_ms: u128,
    pub prove_ms: u128,
    pub verify_ms: u128,
    pub commitment_cplink_link_bytes: usize,
    pub commitment_proof_bytes: usize,
    pub proof_size_bytes: usize,
}

/// Aggregated Lasso metrics across multiple invocations.
#[derive(Debug, Clone, Default)]
pub struct LassoMetricsAggregate {
    pub commit_proof_ms: u128,
    pub prove_ms: u128,
    pub verify_ms: u128,
    pub commitment_cplink_link_bytes: usize,
    pub commitment_proof_bytes: usize,
    pub proof_size_bytes: usize,
    pub total_ms: u128,
}

impl LassoMetricsAggregate {
    pub fn add(&mut self, m: &LassoMetrics) {
        self.commit_proof_ms += m.commit_proof_ms;
        self.prove_ms += m.prove_ms;
        self.verify_ms += m.verify_ms;
        self.commitment_cplink_link_bytes += m.commitment_cplink_link_bytes;
        self.commitment_proof_bytes += m.commitment_proof_bytes;
        self.proof_size_bytes += m.proof_size_bytes;
    }

    pub fn add_aggregate(&mut self, other: &LassoMetricsAggregate) {
        self.commit_proof_ms += other.commit_proof_ms;
        self.prove_ms += other.prove_ms;
        self.verify_ms += other.verify_ms;
        self.commitment_cplink_link_bytes += other.commitment_cplink_link_bytes;
        self.commitment_proof_bytes += other.commitment_proof_bytes;
        self.proof_size_bytes += other.proof_size_bytes;
    }
}

// ── BenchmarkResult (nested structure) ─────────────────────────────────────

pub struct Groth16Metrics {
    pub setup_ms: u128,
    pub commit_ms: u128,
    pub prove_ms: u128,
    pub verify_ms: u128,
    pub proof_size_bytes: usize,
    pub commitment_size_bytes: usize,
}

pub struct LassoBenchmarkMetrics {
    pub total_ms: u128,
    pub commit_proof_ms: u128,
    pub prove_ms: u128,
    pub verify_ms: u128,
    pub commitment_cplink_link_bytes: usize,
    pub commitment_proof_bytes: usize,
    pub proof_size_bytes: usize,
}

impl LassoBenchmarkMetrics {
    pub fn from_aggregate(agg: &LassoMetricsAggregate) -> Self {
        LassoBenchmarkMetrics {
            total_ms: agg.total_ms,
            commit_proof_ms: agg.commit_proof_ms,
            prove_ms: agg.prove_ms,
            verify_ms: agg.verify_ms,
            commitment_cplink_link_bytes: agg.commitment_cplink_link_bytes,
            commitment_proof_bytes: agg.commitment_proof_bytes,
            proof_size_bytes: agg.proof_size_bytes,
        }
    }
}

pub struct CpLinkMetrics {
    pub setup_ms: u128,
    pub prove_ms: u128,
    pub verify_ms: u128,
    pub proof_size_bytes: usize,
}

pub struct BenchmarkResult {
    pub file: String,
    pub circuit: String,
    pub constraints: usize,
    pub groth16: Groth16Metrics,
    pub lasso: LassoBenchmarkMetrics,
    pub cp_link: CpLinkMetrics,
    pub end_to_end_ms: u128,
}

impl BenchmarkResult {
    /// Prints a single-line JSON record prefixed with `BENCHMARK_RESULT`.
    /// Parse with: `grep BENCHMARK_RESULT output.log | jq .`
    pub fn print_json(&self) {
        println!(
            concat!(
                "BENCHMARK_RESULT {{",
                "\"file\":{file:?},",
                "\"circuit\":{circuit:?},",
                "\"constraints\":{c},",
                "\"groth16\":{{",
                  "\"setup_ms\":{g16_setup},",
                  "\"commit_ms\":{g16_cm_ms},",
                  "\"prove_ms\":{g16_p},",
                  "\"verify_ms\":{g16_v},",
                  "\"proof_size_bytes\":{g16_pf},",
                  "\"commitment_size_bytes\":{g16_cm}",
                "}},",
                "\"lasso\":{{",
                  "\"total_ms\":{l_total},",
                  "\"commit_proof_ms\":{l_cp},",
                  "\"prove_ms\":{l_p},",
                  "\"verify_ms\":{l_v},",
                  "\"commitment_cplink_link_bytes\":{l_ccl_b},",
                  "\"commitment_proof_bytes\":{l_cp_b},",
                  "\"proof_size_bytes\":{l_pf}",
                "}},",
                "\"cp_link\":{{",
                  "\"setup_ms\":{cp_s},",
                  "\"prove_ms\":{cp_p},",
                  "\"verify_ms\":{cp_v},",
                  "\"proof_size_bytes\":{cp_pf}",
                "}},",
                "\"end_to_end_ms\":{e2e}",
                "}}"
            ),
            file       = self.file,
            circuit    = self.circuit,
            c          = self.constraints,
            g16_setup  = self.groth16.setup_ms,
            g16_cm_ms  = self.groth16.commit_ms,
            g16_p      = self.groth16.prove_ms,
            g16_v      = self.groth16.verify_ms,
            g16_pf     = self.groth16.proof_size_bytes,
            g16_cm     = self.groth16.commitment_size_bytes,
            l_total    = self.lasso.total_ms,
            l_cp       = self.lasso.commit_proof_ms,
            l_p        = self.lasso.prove_ms,
            l_v        = self.lasso.verify_ms,
            l_ccl_b    = self.lasso.commitment_cplink_link_bytes,
            l_cp_b     = self.lasso.commitment_proof_bytes,
            l_pf       = self.lasso.proof_size_bytes,
            cp_s       = self.cp_link.setup_ms,
            cp_p       = self.cp_link.prove_ms,
            cp_v       = self.cp_link.verify_ms,
            cp_pf      = self.cp_link.proof_size_bytes,
            e2e        = self.end_to_end_ms,
        );
    }
}

// ── CP-Link helpers ──────────────────────────────────────────────────────────

/// Interleave zeros into `raw_u` so every Lasso segment reaches its rounded
/// sparsity (`cm.C.len() × gens.n`).
///
/// Lasso pads its polynomial internally to the next power of two, so the
/// CP-Link witness vector must cover all padded slots.  `raw_u` contains only
/// the *actual* (unpadded) committed values from Groth16; this function
/// re-inserts the zero-padding at each segment boundary.
pub fn pad_lasso_u_values<E: Pairing>(
    lasso_results: &[LassoResult<E>],
    raw_u: &[E::ScalarField],
) -> Vec<E::ScalarField> {
    let mut u_padded = Vec::new();
    let mut offset = 0usize;
    for (cm, _, gens, actual_sp) in lasso_results {
        let s_padded = cm.C.len() * gens.n;
        u_padded.extend_from_slice(&raw_u[offset..offset + actual_sp]);
        u_padded.extend(std::iter::repeat(E::ScalarField::zero()).take(s_padded - actual_sp));
        offset += actual_sp;
    }
    u_padded
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

    // Collect all Lasso generator points into a single buffer for one
    // batched Montgomery inversion chain instead of N separate ones.
    let mut all_projective = Vec::new();
    let mut ck_lengths = Vec::new(); // track per-Lasso slice boundaries

    for (cm, _, gens, actual_sp) in lasso_results {
        l_rows_vec.push(cm.C.len());
        r_cols_vec.push(gens.n);
        actual_sparsities.push(*actual_sp);
        let len = 1 + gens.G.len(); // h + G
        ck_lengths.push(len);
        all_projective.push(gens.h);
        all_projective.extend_from_slice(&gens.G);
    }

    // Single normalize_batch for all Lasso generators combined.
    let all_affine = E::G1::normalize_batch(&all_projective);

    // Split the flat affine buffer back into per-Lasso commitment key vectors.
    let mut cks_lasso_vec = Vec::with_capacity(ck_lengths.len());
    let mut offset = 0;
    for len in &ck_lengths {
        cks_lasso_vec.push(all_affine[offset..offset + len].to_vec());
        offset += len;
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
    let mut lasso_cms_projective = Vec::new();
    for (cm, blinds, _, _) in lasso_results {
        rs.extend(&blinds.blinds);
        lasso_cms_projective.extend_from_slice(&cm.C);
    }
    // Batch affine conversion: single Montgomery inversion chain instead of
    // per-element into_affine() calls.
    let lasso_cms_all = E::G1::normalize_batch(&lasso_cms_projective);

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


/// Convenience wrapper that runs all four CP-Link phases in sequence.
pub fn prove_and_verify_factored_multi_link<E: Pairing>(
    vk: &VerifyingKey<E>,
    num_pub: usize,
    v_groth: E::ScalarField,
    cm_groth: E::G1Affine,
    lasso_results: Vec<LassoResult<E>>,
    u_total: Vec<E::ScalarField>,
    rng: &mut impl Rng,
) -> Result<()> {
    let u_padded = pad_lasso_u_values::<E>(&lasso_results, &u_total);
    let (pp, crs)      = factored_link_setup::<E, _>(vk, num_pub, u_total.len(), &lasso_results, rng);
    let (ek, vk_link)  = factored_link_keygen::<E, _>(&pp, crs, rng);
    let (proof, inst)  = factored_link_prove::<E, _>(&pp, &ek, v_groth, cm_groth, &lasso_results, u_padded, rng);
    if factored_link_verify::<E>(&pp, &vk_link, &inst, &proof) {
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Factored link verification failed"))
    }
}
