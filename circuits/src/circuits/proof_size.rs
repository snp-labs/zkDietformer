//! Detailed proof-size measurement and logging for composite proofs
//! (LegoGroth16 + Lasso/Hyrax PCS + CP-Link).

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalSerialize, Compress};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cc_snark::legogroth16::data_structures::Proof;
use crate::lasso::{
    lasso::{
        memory_checking::MemoryCheckingProof,
        surge::{PrimarySumcheck, SparsePolynomialCommitment, SparsePolynomialEvaluationProof},
    },
    subtables::SubtableStrategy,
};

// ─── Size-report structs ─────────────────────────────────────────────────────

/// Groth16 (LegoGroth16) proof size breakdown.
#[derive(Debug, Clone)]
pub struct Groth16SizeReport {
    pub g1_count: usize,
    pub g1_bytes: usize,
    pub g2_count: usize,
    pub g2_bytes: usize,
    pub total: usize,
}

/// CP-Link proof size breakdown.
///
/// Separates the pure link proof (Prover-generated, counted in Grand Total)
/// from the public instance (Verifier-known commitments, excluded from Grand Total).
#[derive(Debug, Clone)]
pub struct CpLinkSizeReport {
    // Pure Link Proof elements (Prover-generated)
    pub proof_g_count: usize,
    pub proof_g_bytes: usize,
    pub proof_f_count: usize,
    pub proof_f_bytes: usize,
    pub proof_total: usize,
    // Public Instance (Verifier-known, excluded from Grand Total)
    pub instance_g_count: usize,
    pub instance_g_bytes: usize,
}

/// Lasso evaluation proof + commitment size breakdown.
#[derive(Debug, Clone)]
pub struct LassoProofSizeReport {
    // (a)-1 Oracle Commitments (dim, read, final from SparsePolynomialCommitment)
    pub oracle_l_variate_g: usize,
    pub oracle_log_m_variate_g: usize,
    pub oracle_bytes: usize,
    // (a)-2 Lookup Vector Commitment (E_i from comm_derefs)
    pub deref_commitment_g: usize,
    pub deref_bytes: usize,
    // (a)-3 Hyrax Opening Proofs (BulletReductionProof L/R + delta/beta)
    pub opening_proof_g: usize,
    pub opening_proof_f: usize,
    pub opening_proof_bytes: usize,
    pub is_batched: bool,
    // (b)-1 Primary Sum-Check Coefficients
    pub primary_sc_f: usize,
    pub primary_sc_bytes: usize,
    // (b)-2 Grand Product Coefficients
    pub grand_product_f: usize,
    pub grand_product_bytes: usize,
    // (b)-3 Claimed Evaluations
    pub claimed_eval_f: usize,
    pub claimed_eval_bytes: usize,
    pub total: usize,
}

// ─── Global Lasso Aggregator ─────────────────────────────────────────────────

static LASSO_TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);
static LASSO_PROOF_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Reset the global Lasso aggregator. Call before a benchmark run.
pub fn reset_lasso_aggregator() {
    LASSO_TOTAL_BYTES.store(0, Ordering::SeqCst);
    LASSO_PROOF_COUNT.store(0, Ordering::SeqCst);
}

/// Get the aggregated Lasso proof sizes.
pub fn get_lasso_aggregate() -> (usize, usize) {
    (
        LASSO_PROOF_COUNT.load(Ordering::SeqCst),
        LASSO_TOTAL_BYTES.load(Ordering::SeqCst),
    )
}

fn add_to_lasso_aggregate(bytes: usize) {
    LASSO_TOTAL_BYTES.fetch_add(bytes, Ordering::SeqCst);
    LASSO_PROOF_COUNT.fetch_add(1, Ordering::SeqCst);
}

// ─── Helper: compressed serialized size ──────────────────────────────────────

fn compressed_size<T: CanonicalSerialize>(val: &T) -> usize {
    val.serialized_size(Compress::Yes)
}

/// Return the compressed size of a single element of type T, using a
/// representative instance.
fn element_size<T: CanonicalSerialize>(representative: &T) -> usize {
    compressed_size(representative)
}

// ─── Groth16 report ──────────────────────────────────────────────────────────

/// Count individual elements in a LegoGroth16 `Proof`.
///
/// Layout: A(G1) + B(G2) + C(G1) + d1(G1) + d2(G1)
pub fn compute_groth16_report<E: Pairing>(proof: &Proof<E>) -> Groth16SizeReport {
    let g1_size = compressed_size(&proof.a);
    let g2_size = compressed_size(&proof.b);

    // G1 elements: A, C, d1, d2  (4 total)
    let g1_count = 4;
    let g1_bytes = g1_count * g1_size;

    // G2 elements: B  (1 total)
    let g2_count = 1;
    let g2_bytes = g2_count * g2_size;

    let total = g1_bytes + g2_bytes;

    Groth16SizeReport {
        g1_count,
        g1_bytes,
        g2_count,
        g2_bytes,
        total,
    }
}

impl Groth16SizeReport {
    pub fn print_log(&self) {
        println!("PROOF_SIZE_DETAIL [Groth16/LegoGroth16]");
        println!("  G1 elements: {} ({} bytes)", self.g1_count, self.g1_bytes);
        println!("    - A (G1), C (G1), d1 (G1), d2 (G1)");
        println!("  G2 elements: {} ({} bytes)", self.g2_count, self.g2_bytes);
        println!("    - B (G2)");
        println!("  Total: {} bytes", self.total);
    }
}

// ─── CP-Link report ─────────────────────────────────────────────────────────

/// Count elements in the CP-Link proof, separating pure proof from public instance.
///
/// proof_link: single G1Affine  (Prover-generated linking element)
/// instance:   Vec<G1Affine>    (public commitments known to Verifier)
pub fn compute_cplink_report<E: Pairing>(
    proof: &E::G1Affine,
    instance: &[E::G1Affine],
) -> CpLinkSizeReport {
    let g_size = compressed_size(proof);

    // Pure Link Proof: single G1 element (MSM of witness × evaluation key)
    // The CP-Link protocol produces no scalar (F) responses.
    let proof_g_count = 1;
    let proof_g_bytes = proof_g_count * g_size;
    let proof_f_count = 0;
    let proof_f_bytes = 0;
    let proof_total = proof_g_bytes + proof_f_bytes;

    // Public Instance: cm_groth (1 G1) + Lasso commitments (N G1)
    // These are Verifier-known and excluded from the proof size Grand Total.
    let instance_g_count = instance.len();
    let instance_g_bytes = instance_g_count * g_size;

    CpLinkSizeReport {
        proof_g_count,
        proof_g_bytes,
        proof_f_count,
        proof_f_bytes,
        proof_total,
        instance_g_count,
        instance_g_bytes,
    }
}

impl CpLinkSizeReport {
    pub fn print_log(&self) {
        println!("PROOF_SIZE_DETAIL [CP-Link]");
        println!("  Pure Link Proof G elements: {} ({} bytes)",
            self.proof_g_count, self.proof_g_bytes);
        println!("  Pure Link Proof F elements: {} ({} bytes)",
            self.proof_f_count, self.proof_f_bytes);
        println!("  [Public Instance G elements]: {} ({} bytes) *excluded from Grand Total",
            self.instance_g_count, self.instance_g_bytes);
        println!("  => CP-Link Pure Proof Total: {} bytes", self.proof_total);
    }
}

// ─── Lasso proof report ─────────────────────────────────────────────────────

/// Traverse the entire Lasso proof tree and commitment to produce a detailed
/// size report.
pub fn compute_lasso_proof_report<
    G: CurveGroup,
    const C: usize,
    const M: usize,
    S: SubtableStrategy<G::ScalarField, C, M>,
>(
    proof: &SparsePolynomialEvaluationProof<G, C, M, S>,
    commitment: &SparsePolynomialCommitment<G>,
) -> LassoProofSizeReport
where
    [(); S::NUM_MEMORIES]: Sized,
    [(); S::NUM_SUBTABLES]: Sized,
    [(); S::NUM_MEMORIES + 1]: Sized,
{
    // ── Per-element sizes (auto-adapts to any curve) ──
    // Use commitment vectors to get representative G element
    let g_size = if !commitment.l_variate_polys_commitment.C.is_empty() {
        compressed_size(&commitment.l_variate_polys_commitment.C[0])
    } else if !commitment.log_m_variate_polys_commitment.C.is_empty() {
        compressed_size(&commitment.log_m_variate_polys_commitment.C[0])
    } else {
        0
    };

    let f_size = compressed_size(&proof.primary_sumcheck.claimed_evaluation);

    // ── (a)-1 Oracle Commitments ──
    let oracle_l_variate_g = commitment.l_variate_polys_commitment.C.len();
    let oracle_log_m_variate_g = commitment.log_m_variate_polys_commitment.C.len();
    let oracle_bytes = (oracle_l_variate_g + oracle_log_m_variate_g) * g_size;

    // ── (a)-2 Lookup Vector Commitment (comm_derefs) ──
    let deref_commitment_g = proof.comm_derefs.comm_ops_val.C.len();
    let deref_bytes = deref_commitment_g * g_size;

    // ── (a)-3 Hyrax Opening Proofs ──
    // Collect all DotProductProofLog instances from:
    //   - primary_sumcheck.proof_derefs.proof_table_eval.proof (DotProductProofLog)
    //   - memory_check.proof_hash_layer.proof_ops.proof (DotProductProofLog)
    //   - memory_check.proof_hash_layer.proof_mem.proof (DotProductProofLog)
    //   - memory_check.proof_hash_layer.proof_derefs.proof_table_eval.proof (DotProductProofLog)

    let mut opening_g_count: usize = 0;
    let mut opening_f_count: usize = 0;

    // Helper: count from a DotProductProofLog
    let count_dot_product = |dp: &crate::lasso::subprotocols::dot_product::DotProductProofLog<G>| -> (usize, usize) {
        let l_len = dp.bullet_reduction_proof.L_vec.len();
        let r_len = dp.bullet_reduction_proof.R_vec.len();
        // G elements: L_vec + R_vec + delta + beta
        let g = l_len + r_len + 2;
        // F elements: z1 + z2
        let f = 2;
        (g, f)
    };

    // primary_sumcheck.proof_derefs
    {
        let dp = &proof.primary_sumcheck.proof_derefs.proof_table_eval.proof;
        let (g, f) = count_dot_product(dp);
        opening_g_count += g;
        opening_f_count += f;
    }

    // memory_check.proof_hash_layer.proof_ops
    {
        let dp = &proof.memory_check.proof_hash_layer.proof_ops.proof;
        let (g, f) = count_dot_product(dp);
        opening_g_count += g;
        opening_f_count += f;
    }

    // memory_check.proof_hash_layer.proof_mem
    {
        let dp = &proof.memory_check.proof_hash_layer.proof_mem.proof;
        let (g, f) = count_dot_product(dp);
        opening_g_count += g;
        opening_f_count += f;
    }

    // memory_check.proof_hash_layer.proof_derefs
    {
        let dp = &proof.memory_check.proof_hash_layer.proof_derefs.proof_table_eval.proof;
        let (g, f) = count_dot_product(dp);
        opening_g_count += g;
        opening_f_count += f;
    }

    let opening_proof_bytes = opening_g_count * g_size + opening_f_count * f_size;

    // Hyrax opening proofs are ALWAYS batched by structural design:
    //   - CombinedTableEvalProof performs n-to-1 reduction (NUM_MEMORIES/C evals → 1 opening)
    //   - HashLayerProof uses combined joint polynomials (interleaved dim/read/final)
    // Result: exactly 4 DotProductProofLog instances cover all polynomial openings,
    // regardless of NUM_MEMORIES or C.  Without batching there would be
    // (2*NUM_MEMORIES + 3*C) individual IPA proofs.
    let is_batched = true;

    let prod_layer = &proof.memory_check.proof_prod_layer;

    // ── (b)-1 Primary Sum-Check Coefficients ──
    let mut primary_sc_f: usize = 0;
    for cpoly in &proof.primary_sumcheck.proof.compressed_polys {
        // CompressedUniPoly stores all coefficients except linear term
        // Total coefficients per poly = coeffs_except_linear_term.len() + 1 (linear)
        // But what's actually transmitted is coeffs_except_linear_term only
        primary_sc_f += cpoly.coeffs_except_linear_term.len();
    }
    let primary_sc_bytes = primary_sc_f * f_size;

    // ── (b)-2 Grand Product Coefficients ──
    let mut grand_product_f: usize = 0;

    // proof_mem layers
    for layer in &prod_layer.proof_mem.proof {
        for cpoly in &layer.proof.compressed_polys {
            grand_product_f += cpoly.coeffs_except_linear_term.len();
        }
        // claims_prod_left and claims_prod_right are F elements per layer
        grand_product_f += layer.claims_prod_left.len();
        grand_product_f += layer.claims_prod_right.len();
    }

    // proof_ops layers
    for layer in &prod_layer.proof_ops.proof {
        for cpoly in &layer.proof.compressed_polys {
            grand_product_f += cpoly.coeffs_except_linear_term.len();
        }
        grand_product_f += layer.claims_prod_left.len();
        grand_product_f += layer.claims_prod_right.len();
    }
    let grand_product_bytes = grand_product_f * f_size;

    // ── (b)-3 Claimed Evaluations ──
    let mut claimed_eval_f: usize = 0;

    // primary_sumcheck: claimed_evaluation (1) + eval_derefs (NUM_MEMORIES)
    claimed_eval_f += 1;
    claimed_eval_f += S::NUM_MEMORIES;

    // proof_prod_layer: grand_product_evals (4 × NUM_MEMORIES)
    claimed_eval_f += 4 * S::NUM_MEMORIES;

    // proof_hash_layer: eval_dim(C) + eval_read(C) + eval_final(C) + eval_derefs(NUM_MEMORIES)
    claimed_eval_f += C; // eval_dim
    claimed_eval_f += C; // eval_read
    claimed_eval_f += C; // eval_final
    claimed_eval_f += S::NUM_MEMORIES; // eval_derefs

    let claimed_eval_bytes = claimed_eval_f * f_size;

    // ── Total ──
    let total = oracle_bytes
        + deref_bytes
        + opening_proof_bytes
        + primary_sc_bytes
        + grand_product_bytes
        + claimed_eval_bytes;

    LassoProofSizeReport {
        oracle_l_variate_g,
        oracle_log_m_variate_g,
        oracle_bytes,
        deref_commitment_g,
        deref_bytes,
        opening_proof_g: opening_g_count,
        opening_proof_f: opening_f_count,
        opening_proof_bytes,
        is_batched,
        primary_sc_f,
        primary_sc_bytes,
        grand_product_f,
        grand_product_bytes,
        claimed_eval_f,
        claimed_eval_bytes,
        total,
    }
}

impl LassoProofSizeReport {
    pub fn print_log(&self, label: &str) {
        println!("PROOF_SIZE_DETAIL [Lasso: {}]", label);
        println!("  (a) Commitments:");
        println!(
            "    (a)-1 Oracle Commitments: {} l-variate + {} log_m-variate G = {} bytes",
            self.oracle_l_variate_g, self.oracle_log_m_variate_g, self.oracle_bytes
        );
        println!(
            "    (a)-2 Lookup Vector (comm_derefs): {} G = {} bytes",
            self.deref_commitment_g, self.deref_bytes
        );
        println!(
            "    (a)-3 Hyrax Opening Proofs: {} G + {} F = {} bytes (batched: {})",
            self.opening_proof_g, self.opening_proof_f, self.opening_proof_bytes, self.is_batched
        );
        println!("  (b) Interactive Proof (Scalars):");
        println!(
            "    (b)-1 Primary Sum-Check Coeffs: {} F = {} bytes",
            self.primary_sc_f, self.primary_sc_bytes
        );
        println!(
            "    (b)-2 Grand Product Coeffs+Claims: {} F = {} bytes",
            self.grand_product_f, self.grand_product_bytes
        );
        println!(
            "    (b)-3 Claimed Evaluations: {} F = {} bytes",
            self.claimed_eval_f, self.claimed_eval_bytes
        );
        println!("  Total: {} bytes", self.total);

        // Add to global aggregator
        add_to_lasso_aggregate(self.total);
    }
}

// ─── Grand total report ──────────────────────────────────────────────────────

pub fn print_full_report(
    g16: &Groth16SizeReport,
    cplink: &CpLinkSizeReport,
) {
    let (lasso_count, lasso_total) = get_lasso_aggregate();

    println!();
    println!("PROOF_SIZE_REPORT ════════════════════════════════════════════");
    println!("  Groth16 (LegoGroth16): {} bytes", g16.total);
    println!("    {} G1 ({} B) + {} G2 ({} B)",
        g16.g1_count, g16.g1_bytes, g16.g2_count, g16.g2_bytes);
    println!("  CP-Link (pure proof): {} bytes", cplink.proof_total);
    println!("    {} G1 ({} B) + {} F ({} B)",
        cplink.proof_g_count, cplink.proof_g_bytes,
        cplink.proof_f_count, cplink.proof_f_bytes);
    println!("    [public instance: {} G1 ({} B) — not counted]",
        cplink.instance_g_count, cplink.instance_g_bytes);
    println!("  Lasso (all {} proofs): {} bytes", lasso_count, lasso_total);
    let grand_total = g16.total + cplink.proof_total + lasso_total;
    println!("  ──────────────────────────────────────");
    println!("  Grand Total: {} bytes ({:.2} KB)", grand_total, grand_total as f64 / 1024.0);
    println!("════════════════════════════════════════════════════════════");
    println!();
}
