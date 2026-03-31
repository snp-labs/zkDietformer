use anyhow::Result;
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::{UniformRand, rand::Rng, Zero};

// ── Memory measurement ───────────────────────────────────────────────────────

/// Returns the process's **peak** RSS in megabytes via `getrusage(RUSAGE_SELF)`.
///
/// On macOS `ru_maxrss` is in **bytes**; on Linux it is in **kilobytes**.
/// Returns `0.0` on unsupported platforms.
///
/// Because `getrusage` reports the high-water mark since process start, two
/// consecutive calls yield a monotonically non-decreasing value.  The delta
/// between two calls therefore measures the additional peak memory incurred by
/// the code that ran between them — a lightweight, zero-dependency proxy for
/// per-phase memory cost.
pub fn rss_mb() -> f64 {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        // Minimal repr of POSIX `struct rusage` — only the fields we need.
        // Both macOS-aarch64/x86_64 and Linux-x86_64 lay out the first three
        // logical fields identically: two `struct timeval` (each 16 bytes on
        // 64-bit) followed by `long ru_maxrss`.  We pad the remainder to the
        // known struct size (144 bytes = 18 × i64) so the stack frame is safe.
        #[repr(C)]
        struct RUsage {
            ru_utime:  [i64; 2],   // struct timeval (tv_sec + tv_usec, 16 B)
            ru_stime:  [i64; 2],   // struct timeval
            ru_maxrss: i64,        // peak RSS (bytes on macOS, KB on Linux)
            _rest:     [i64; 13],  // remaining fields, unused
        }
        unsafe extern "C" { fn getrusage(who: i32, usage: *mut RUsage) -> i32; }

        let mut u = RUsage {
            ru_utime: [0; 2], ru_stime: [0; 2], ru_maxrss: 0, _rest: [0; 13],
        };
        unsafe { getrusage(0 /* RUSAGE_SELF */, &mut u as *mut _); }
        #[cfg(target_os = "macos")]
        return u.ru_maxrss as f64 / (1024.0 * 1024.0); // bytes → MB
        #[cfg(target_os = "linux")]
        return u.ru_maxrss as f64 / 1024.0;             // KB → MB
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    0.0
}
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

/// Compute total serialized size (compressed) of the Groth16 Pedersen
/// commitment (`cm1`) plus all Lasso polynomial commitments.
pub fn total_commitment_size<E: Pairing>(
    cm1: &E::G1Affine,
    lasso_results: &[LassoResult<E>],
) -> usize {
    let cm1_size = cm1.serialized_size(Compress::Yes);
    let lasso_size: usize = lasso_results
        .iter()
        .map(|(poly_cm, _, _, _)| poly_cm.serialized_size(Compress::Yes))
        .sum();
    cm1_size + lasso_size
}

// ── BenchmarkResult ─────────────────────────────────────────────────────────

/// Holds all performance metrics for one witness file's proving pipeline.
///
/// # Timing notes
/// * `setup_ms`         — LegoGroth16 `generate_random_parameters`
/// * `groth16_commit_ms`— `cm1_from_vk` (Pedersen commitment from the VK)
/// * `groth16_prove_ms` — `create_random_proof`
/// * `groth16_verify_ms`— `verify_proof`
/// * `lasso_ms`         — all Lasso `run_lasso_for_*` calls; commit + prove +
///                        verify are bundled inside the helpers and cannot be
///                        separated without modifying them
/// * `cp_link_setup_ms` — `factored_link_setup` + `factored_link_keygen`
/// * `cp_link_prove_ms` — `factored_link_prove`
/// * `cp_link_verify_ms`— `factored_link_verify`
///
/// # Memory notes
/// Fields are **peak-RSS deltas** (MB) measured with `rss_mb()`.  Because
/// `getrusage` reports the high-water mark since process start, the delta
/// between two consecutive calls measures newly allocated (and not yet freed)
/// memory for that phase.
///
/// Call `print_json()` to emit a single `BENCHMARK_RESULT` JSON line:
/// ```text
/// grep 'BENCHMARK_RESULT' test.log | jq '.prove.total_ms'
/// ```
pub struct BenchmarkResult {
    pub file:    String,
    pub circuit: String,
    pub constraints: usize,
    // ── timing (milliseconds) ─────────────────────────────────────────────
    pub setup_ms:          u128,
    pub groth16_commit_ms: u128,
    pub groth16_prove_ms:  u128,
    pub groth16_verify_ms: u128,
    pub lasso_ms:          u128,
    pub cp_link_setup_ms:  u128,
    pub cp_link_prove_ms:  u128,
    pub cp_link_verify_ms: u128,
    pub end_to_end_ms:     u128,
    pub proof_size_bytes:  usize,
    pub commitment_size_bytes: usize,
    // ── memory (peak-RSS delta per phase, MB) ─────────────────────────────
    pub memory_setup_mb:   f64,
    pub memory_commit_mb:  f64,
    pub memory_prove_mb:   f64,
    pub memory_verify_mb:  f64,
}

impl BenchmarkResult {
    /// Prints a single-line JSON record prefixed with `BENCHMARK_RESULT`.
    /// Parse with: `grep BENCHMARK_RESULT output.log | jq .`
    pub fn print_json(&self) {
        let total_prove_ms  = self.groth16_prove_ms + self.lasso_ms
                            + self.cp_link_setup_ms + self.cp_link_prove_ms;
        let total_verify_ms = self.groth16_verify_ms + self.cp_link_verify_ms;
        println!(
            concat!(
                "BENCHMARK_RESULT {{",
                "\"file\":{file:?},",
                "\"circuit\":{circuit:?},",
                "\"constraints\":{c},",
                "\"setup_ms\":{s},",
                "\"proof_size_bytes\":{p},",
                "\"commitment_size_bytes\":{cm_size},",
                "\"groth16\":{{",
                  "\"commit_ms\":{gc},",
                  "\"prove_ms\":{gp},",
                  "\"verify_ms\":{gv}",
                "}},",
                "\"lasso\":{{\"total_ms\":{lp}}},",
                "\"cp_link\":{{",
                  "\"setup_ms\":{cs},",
                  "\"prove_ms\":{cp},",
                  "\"verify_ms\":{cv}",
                "}},",
                "\"prove\":{{\"total_ms\":{tp}}},",
                "\"verify\":{{\"total_ms\":{tv}}},",
                "\"memory_mb\":{{",
                  "\"setup\":{ms:.2},",
                  "\"commit\":{mc:.2},",
                  "\"prove\":{mp:.2},",
                  "\"verify\":{mv:.2}",
                "}},",
                "\"end_to_end_ms\":{e2e}",
                "}}"
            ),
            file    = self.file,
            circuit = self.circuit,
            c       = self.constraints,
            s       = self.setup_ms,
            p       = self.proof_size_bytes,
            cm_size = self.commitment_size_bytes,
            gc      = self.groth16_commit_ms,
            gp      = self.groth16_prove_ms,
            gv      = self.groth16_verify_ms,
            lp      = self.lasso_ms,
            cs      = self.cp_link_setup_ms,
            cp      = self.cp_link_prove_ms,
            cv      = self.cp_link_verify_ms,
            tp      = total_prove_ms,
            tv      = total_verify_ms,
            ms      = self.memory_setup_mb,
            mc      = self.memory_commit_mb,
            mp      = self.memory_prove_mb,
            mv      = self.memory_verify_mb,
            e2e     = self.end_to_end_ms,
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
