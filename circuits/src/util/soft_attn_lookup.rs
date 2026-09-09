use std::time::Instant;
use anyhow::Result;
use ark_bn254::{Fr, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, PrimeField, Zero};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::rand::rngs::StdRng;
use ark_std::UniformRand;
use merlin::Transcript;
use crate::{
    circuits::common::{LassoMetrics, LassoMetricsAggregate},
    circuits::freivalds_matmul::soft_attn_circuit::SOFTAttentionCircuit,
    circuits::proof_size,
    config::{KBITS_C, KBITS_M, SM_KBITS_C, SM_KBITS_M, FRACBITS_C, FRACBITS_M, SOFTMAX_C, SOFTMAX_M},
    lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
        subtables::gaussian::GaussianSubtableStrategy,
        utils::random::RandomTape,
    },
    util::nystrom_attn_lookup::{
        prove_lasso_for_range_22, prove_lasso_for_range_52,
        prove_mp_range_15, prove_mp_range_14, prove_mp_range_13,
        prove_mp_range_12, prove_mp_range_11, prove_mp_range_10,
        prove_mp_range_9, prove_mp_range_8, prove_mp_range_7,
        prove_mp_range_6, prove_mp_range_5, prove_mp_range_4,
        prove_mp_range_3,
    },
};

type G1 = G1Projective;
type LassoResult = (PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize);

/// Dynamic dispatch for MP bounded-diff Lasso range check.
/// Dispatches to the appropriate compile-time prove_mp_range_N based on log_r.
fn dispatch_mp_range(
    log_r: usize,
    nz: Vec<[usize; FRACBITS_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<(LassoResult, LassoMetrics)> {
    match log_r {
        3 => prove_mp_range_3(nz, gens, rng, label),
        4 => prove_mp_range_4(nz, gens, rng, label),
        5 => prove_mp_range_5(nz, gens, rng, label),
        6 => prove_mp_range_6(nz, gens, rng, label),
        7 => prove_mp_range_7(nz, gens, rng, label),
        8 => prove_mp_range_8(nz, gens, rng, label),
        9 => prove_mp_range_9(nz, gens, rng, label),
        10 => prove_mp_range_10(nz, gens, rng, label),
        11 => prove_mp_range_11(nz, gens, rng, label),
        12 => prove_mp_range_12(nz, gens, rng, label),
        13 => prove_mp_range_13(nz, gens, rng, label),
        14 => prove_mp_range_14(nz, gens, rng, label),
        15 => prove_mp_range_15(nz, gens, rng, label),
        _ => Err(anyhow::anyhow!(
            "unsupported MP range check log_r={}",
            log_r
        )),
    }
}

/// Run Lasso Gaussian function lookup for a SOFT table_idx 4D array.
///
/// Uses GaussianSubtableStrategy (exp(-x) function table, C=1, M=65536).
/// Analogous to Nystrom's run_lasso_bundle which uses SoftmaxSubtableStrategy.
pub fn run_soft_lasso_bundle(
    table_idx: &[Vec<Vec<Vec<Fr>>>],
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    let b_dim = table_idx.len();
    let h_dim = table_idx[0].len();
    let dim1 = table_idx[0][0].len();
    let dim2 = table_idx[0][0][0].len();
    let total_sparsity = b_dim * h_dim * dim1 * dim2;

    let gens = SparsePolyCommitmentGens::<G1>::new(
        label, SOFTMAX_C,
        total_sparsity.next_power_of_two(),
        SOFTMAX_C,
        SOFTMAX_M.trailing_zeros() as usize,
    );

    let mut metrics_agg = LassoMetricsAggregate::default();
    let start = Instant::now();
    let (results, metrics) = prove_lasso_for_gaussian(table_idx, &gens, rng, label)?;
    metrics_agg.add(&metrics);
    println!(
        "  - Lasso Total Time (SOFT Gaussian {:?}): {:?}",
        std::str::from_utf8(label).unwrap_or("?"),
        start.elapsed()
    );

    Ok((results, metrics_agg))
}

/// Gaussian function lookup Lasso proof (C=1, M=65536).
/// Mirrors prove_lasso_for_softmax but uses GaussianSubtableStrategy.
pub fn prove_lasso_for_gaussian(
    table_idx: &[Vec<Vec<Vec<Fr>>>],
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<(Vec<LassoResult>, LassoMetrics)> {
    let b_dim = table_idx.len();
    let h_dim = table_idx[0].len();
    let s_dim = table_idx[0][0].len();
    let l_dim = table_idx[0][0][0].len();
    let mut metrics = LassoMetrics::default();

    let mut nz_indices: Vec<[usize; SOFTMAX_C]> = Vec::with_capacity(b_dim * h_dim * s_dim * l_dim);
    for b in 0..b_dim {
        for h in 0..h_dim {
            for s in 0..s_dim {
                for l in 0..l_dim {
                    nz_indices.push([table_idx[b][h][s][l].into_bigint().as_ref()[0] as usize]);
                }
            }
        }
    }

    let sparsity = nz_indices.len();
    let log_m = SOFTMAX_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<SOFTMAX_C>::new(nz_indices, log_m);
    let mut densified = DensifiedRepresentation::<Fr, SOFTMAX_C>::from(&lookup_matrix);

    let commit_start = Instant::now();
    let commitment = densified.commit::<G1>(gens);
    metrics.commit_proof_ms = commit_start.elapsed().as_millis();

    let log_s = sparsity.next_power_of_two().trailing_zeros() as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();
    let mut random_tape = RandomTape::new(b"lasso_tape");
    let mut transcript = Transcript::new(b"lasso_gaussian");

    let prove_start = Instant::now();
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, SOFTMAX_C, SOFTMAX_M, GaussianSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );
    metrics.prove_ms = prove_start.elapsed().as_millis();

    let lasso_label = std::str::from_utf8(label).unwrap_or("gaussian");
    let report = proof_size::compute_lasso_proof_report::<G1, SOFTMAX_C, SOFTMAX_M, GaussianSubtableStrategy>(&proof_lasso, &commitment);
    report.print_log(lasso_label);
    metrics.proof_size_bytes = report.total;
    metrics.commitment_proof_bytes = report.oracle_bytes + report.deref_bytes;

    let mut vt = Transcript::new(b"lasso_gaussian");
    let verify_start = Instant::now();
    proof_lasso.verify(&commitment, &r_point, gens, &mut vt)
        .map_err(|_| anyhow::anyhow!("Gaussian Lasso verification failed: {:?}", lasso_label))?;
    metrics.verify_ms = verify_start.elapsed().as_millis();
    println!("    -> Gaussian Lasso verified: {:?}", lasso_label);

    // CP-Link: extract dim[i] slices from combined commitment
    let r_size = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks = densified.dim[0].len() / r_size;
    assert!(chunks > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let mut results = Vec::new();
    for i in 0..SOFTMAX_C {
        let start = i * chunks;
        let end = start + chunks;
        let cm = PolyCommitment { C: commitment.l_variate_polys_commitment.C[start..end].to_vec() };
        let blinds = PolyCommitmentBlinds { blinds: vec![Fr::zero(); chunks] };
        metrics.commitment_cplink_link_bytes += cm.serialized_size(Compress::Yes);
        results.push((cm, blinds, full_gens.clone(), sparsity));
    }

    Ok((results, metrics))
}

/// Run all range-check Lasso proofs for a SOFT attention circuit.
///
/// d1 ordering (must match circuit push order in generate_constraints B9):
///
///   | # | d1 push (circuit B9)       | Lasso target (this fn)     | C | type  |
///   |---|----------------------------|----------------------------|---|-------|
///   | 1 | gauss_b_idx                | gauss_b_idx                | 1 | gauss |
///   | 2 | gauss_a_idx                | gauss_a_idx                | 1 | gauss |
///   | 3 | R_dist_B  lows, highs      | R_dist_B  lows, highs      | 2 | 26bit |
///   | 4 | R_dist_A  lows, highs      | R_dist_A  lows, highs      | 2 | 26bit |
///   | 5 | R_cross_B lows, highs      | R_cross_B lows, highs      | 2 | 26bit |
///   | 6 | R_cross_A lows, highs      | R_cross_A lows, highs      | 2 | 26bit |
///   | 7 | R_inv[0]  lows, highs      | R_inv[0]  lows, highs      | 2 | 26bit |
///   | 8 | R_inv[1]  lows, highs      | R_inv[1]  lows, highs      | 2 | 26bit |
///   | 9 | R_inv[2]  lows, highs      | R_inv[2]  lows, highs      | 2 | 26bit |
///   |10 | R_inv[3]  lows, highs      | R_inv[3]  lows, highs      | 2 | 26bit |
///   |11 | R_temp    lows, highs      | R_temp    lows, highs      | 2 | 26bit |
///   |12 | R_mid     lows, highs      | R_mid     lows, highs      | 2 | 26bit |
///   |13 | R_Y       lows, highs      | R_Y       lows, highs      | 2 | 26bit |
///   |14 | R_proj_q  lows, highs      | R_proj_q  lows, highs      | 2 | 26bit |
///   |15 | R_proj_v  lows, highs      | R_proj_v  lows, highs      | 2 | 26bit |
///   |16 | R_proj_o  lows, highs      | R_proj_o  lows, highs      | 2 | 26bit |
///   |17 | R_scale_q lows, highs      | R_scale_q lows, highs      | 2 | 26bit |
///   |18 | r_gauss_B limbs[0..3]      | r_gauss_B limbs[0..3]      | 4 | 52bit |
///   |19 | r_gauss_A limbs[0..3]      | r_gauss_A limbs[0..3]      | 4 | 52bit |
///   |20 | mp_cond1_pool              | mp_cond1_pool              | 1 | MP(log_r1) |
///   |21 | mp_cond3_pool              | mp_cond3_pool              | 1 | MP(log_r3) |
///   |22 | mp_cond4_pool              | mp_cond4_pool              | 1 | MP(log_r4) |
///
/// Total Lasso results: 1+1 + 15*2 + 4+4 + 1+1+1 = 43
pub fn run_all_soft_remainders_with_link(
    circuit: &SOFTAttentionCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] SOFT Attention: Verifying All 26-bit Remainders ===");

    let mut all_results = Vec::new();
    let mut metrics_agg = LassoMetricsAggregate::default();

    // 26-bit range checks (C=2: 16+10) -- order must match d1 push order
    let target_26bit: Vec<(&Vec<Vec<Vec<Vec<Fr>>>>, &str)> = vec![
        (&circuit.dist_b.r_dist,                   "soft_r_dist_B"),
        (&circuit.dist_a.r_dist,                   "soft_r_dist_A"),
        (&circuit.cross_b.r_matmul,                "soft_r_cross_B"),
        (&circuit.cross_a.r_matmul,                "soft_r_cross_A"),
        (&circuit.pseudo_inv.r_inv_remainders[0],  "soft_r_inv0"),
        (&circuit.pseudo_inv.r_inv_remainders[1],  "soft_r_inv1"),
        (&circuit.pseudo_inv.r_inv_remainders[2],  "soft_r_inv2"),
        (&circuit.pseudo_inv.r_inv_remainders[3],  "soft_r_inv3"),
        (&circuit.matmul_temp.r_matmul,            "soft_r_temp"),
        (&circuit.matmul_mid.r_matmul,             "soft_r_mid"),
        (&circuit.matmul_x.r_matmul,               "soft_r_Y"),
        (&circuit.proj_q.r_out_proj,               "soft_r_proj_q"),
        (&circuit.proj_v.r_out_proj,               "soft_r_proj_v"),
        (&circuit.proj_o.r_out_proj,               "soft_r_proj_o"),
        (&circuit.scale_q.r_scale,                 "soft_r_scale_q"),
    ];

    for (r_matrix, label) in &target_26bit {
        let b_dim = r_matrix.len();
        let h_dim = r_matrix[0].len();
        let row_dim = r_matrix[0][0].len();
        let col_dim = r_matrix[0][0][0].len();
        let total_sparsity = b_dim * h_dim * row_dim * col_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for r in 0..row_dim {
                    for c in 0..col_dim {
                        let val_u64 = r_matrix[b][h][r][c].into_bigint().as_ref()[0];
                        let idx0 = (val_u64 & 0xFFFF) as usize;
                        let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;
                        nz.push([idx0, idx1]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), KBITS_C,
            total_sparsity.next_power_of_two(),
            KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    // 52-bit r_gauss range checks (C=4: 16+16+16+4)
    let target_52bit: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&circuit.gauss_b.r_gauss, "soft_r_gauss_B_52bit"),
        (&circuit.gauss_a.r_gauss, "soft_r_gauss_A_52bit"),
    ];

    for (r_gauss_4d, label) in target_52bit {
        let b_dim = r_gauss_4d.len();
        let h_dim = r_gauss_4d[0].len();
        let s_dim = r_gauss_4d[0][0].len();
        let l_dim = r_gauss_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz: Vec<[usize; SM_KBITS_C]> = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = r_gauss_4d[b][h][s][l].into_bigint().as_ref()[0];
                        nz.push([
                            (val_u64 & 0xFFFF) as usize,
                            ((val_u64 >> 16) & 0xFFFF) as usize,
                            ((val_u64 >> 32) & 0xFFFF) as usize,
                            ((val_u64 >> 48) & 0xF) as usize,
                        ]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), SM_KBITS_C,
            total_sparsity.next_power_of_two(),
            SM_KBITS_C,
            SM_KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_52(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    Ok((all_results, metrics_agg))
}

/// Run Moore-Penrose bounded-difference Lasso proofs for conditions 1, 3, 4.
/// Uses per-layer epsilon and log_r from the SOFTAttentionCircuit struct.
pub fn run_soft_mp_conditions_with_link(
    circuit: &SOFTAttentionCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] SOFT MP Bounded-Difference Range Checks ===");
    println!("  eps1={} (log_r={}), eps3={} (log_r={}), eps4={} (log_r={})",
        circuit.eps1, circuit.mp_log_r1, circuit.eps3, circuit.mp_log_r3,
        circuit.eps4, circuit.mp_log_r4);

    let mut all_results = Vec::new();
    let mut metrics_agg = LassoMetricsAggregate::default();

    let q_y = &circuit.pseudo_inv.q_y_intermediates;
    let kernel_a = &circuit.gauss_a.kernel;
    let b_dim = q_y[0].len();
    let h_dim = q_y[0][0].len();
    let m_dim = q_y[0][0][0].len();

    // Condition 1: |Q_y2 - kernel_A| < eps1
    {
        let delta_u64 = circuit.eps1 as u64;
        let delta = Fr::from(delta_u64);
        let range_limit = delta_u64 * 2;
        let total = b_dim * h_dim * m_dim * m_dim;
        let mut nz = Vec::with_capacity(total);
        let mut max_val: usize = 0;
        for b in 0..b_dim {
            for h in 0..h_dim {
                for i in 0..m_dim {
                    for j in 0..q_y[1][b][h][i].len() {
                        let shifted = q_y[1][b][h][i][j] - kernel_a[b][h][i][j] + delta;
                        let val = shifted.into_bigint().as_ref()[0] as usize;
                        assert!(
                            (val as u64) < range_limit,
                            "soft mp_cond1: shifted {} >= range {} at [b={},h={},i={},j={}]",
                            val, range_limit, b, h, i, j,
                        );
                        max_val = max_val.max(val);
                        nz.push([val]);
                    }
                }
            }
        }
        println!("  [soft_mp_cond1] max shifted: {} (limit: {})", max_val, range_limit);
        let gens = SparsePolyCommitmentGens::<G1>::new(
            b"soft_mp_cond1", FRACBITS_C,
            total.next_power_of_two(),
            FRACBITS_C, FRACBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = dispatch_mp_range(
            circuit.mp_log_r1, nz, &gens, rng, b"soft_mp_cond1")?;
        all_results.push(link_data);
        metrics_agg.add(&metrics);
        println!("  - soft_mp_cond1 (|Q_y2-A|<{}, log_r={}) verified | Count: {} | Time: {:?}",
            circuit.eps1, circuit.mp_log_r1, total, start.elapsed());
    }

    // Condition 3: |Q_y1[i][j] - Q_y1[j][i]| < eps3
    {
        let delta_u64 = circuit.eps3 as u64;
        let delta = Fr::from(delta_u64);
        let range_limit = delta_u64 * 2;
        let pairs_per_bh = m_dim * (m_dim - 1) / 2;
        let total = b_dim * h_dim * pairs_per_bh;
        let mut nz = Vec::with_capacity(total);
        let mut max_val: usize = 0;
        for b in 0..b_dim {
            for h in 0..h_dim {
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted = q_y[0][b][h][i][j] - q_y[0][b][h][j][i] + delta;
                        let val = shifted.into_bigint().as_ref()[0] as usize;
                        assert!(
                            (val as u64) < range_limit,
                            "soft mp_cond3: shifted {} >= range {} at [b={},h={},i={},j={}]",
                            val, range_limit, b, h, i, j,
                        );
                        max_val = max_val.max(val);
                        nz.push([val]);
                    }
                }
            }
        }
        println!("  [soft_mp_cond3] max shifted: {} (limit: {})", max_val, range_limit);
        let gens = SparsePolyCommitmentGens::<G1>::new(
            b"soft_mp_cond3", FRACBITS_C,
            total.next_power_of_two(),
            FRACBITS_C, FRACBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = dispatch_mp_range(
            circuit.mp_log_r3, nz, &gens, rng, b"soft_mp_cond3")?;
        all_results.push(link_data);
        metrics_agg.add(&metrics);
        println!("  - soft_mp_cond3 (|Q_y1-Q_y1^T|<{}, log_r={}) verified | Count: {} | Time: {:?}",
            circuit.eps3, circuit.mp_log_r3, total, start.elapsed());
    }

    // Condition 4: |Q_y3[i][j] - Q_y3[j][i]| < eps4
    {
        let delta_u64 = circuit.eps4 as u64;
        let delta = Fr::from(delta_u64);
        let range_limit = delta_u64 * 2;
        let pairs_per_bh = m_dim * (m_dim - 1) / 2;
        let total = b_dim * h_dim * pairs_per_bh;
        let mut nz = Vec::with_capacity(total);
        let mut max_val: usize = 0;
        for b in 0..b_dim {
            for h in 0..h_dim {
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted = q_y[2][b][h][i][j] - q_y[2][b][h][j][i] + delta;
                        let val = shifted.into_bigint().as_ref()[0] as usize;
                        assert!(
                            (val as u64) < range_limit,
                            "soft mp_cond4: shifted {} >= range {} at [b={},h={},i={},j={}]",
                            val, range_limit, b, h, i, j,
                        );
                        max_val = max_val.max(val);
                        nz.push([val]);
                    }
                }
            }
        }
        println!("  [soft_mp_cond4] max shifted: {} (limit: {})", max_val, range_limit);
        let gens = SparsePolyCommitmentGens::<G1>::new(
            b"soft_mp_cond4", FRACBITS_C,
            total.next_power_of_two(),
            FRACBITS_C, FRACBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = dispatch_mp_range(
            circuit.mp_log_r4, nz, &gens, rng, b"soft_mp_cond4")?;
        all_results.push(link_data);
        metrics_agg.add(&metrics);
        println!("  - soft_mp_cond4 (|Q_y3-Q_y3^T|<{}, log_r={}) verified | Count: {} | Time: {:?}",
            circuit.eps4, circuit.mp_log_r4, total, start.elapsed());
    }

    Ok((all_results, metrics_agg))
}
