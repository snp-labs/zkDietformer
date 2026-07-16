use std::time::Instant;
use anyhow::Result;
use ark_bn254::{Fr, G1Projective};
use ark_ff::{BigInteger, PrimeField};
use ark_std::rand::rngs::StdRng;
use crate::{
    circuits::common::{LassoMetrics, LassoMetricsAggregate},
    circuits::freivalds_matmul::bert_attn_circuit::BERTAttentionCircuit,
    config::{KBITS_C, KBITS_M, SM_KBITS_C, SM_KBITS_M, SOFTMAX_C, SOFTMAX_M},
    gadgets::softmax::constraints::SoftmaxBundle,
    lasso::{
        lasso::surge::SparsePolyCommitmentGens,
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
    },
    util::nystrom_attn_lookup::{prove_lasso_for_softmax, prove_lasso_for_range_22, prove_lasso_for_range_52},
};

type G1 = G1Projective;
type LassoResult = (PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize);

/// Run Lasso proof for the single BERT full-sequence softmax bundle.
///
/// BERT uses one softmax over the full [S×S] attention score matrix,
/// unlike Nystromformer which has three landmark-based softmax kernels.
pub fn run_bert_lasso_bundle(
    bundle: &SoftmaxBundle<Fr>,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    // v3: table_idx is 4D [B][H][S][L]
    let b_dim = bundle.table_idx.len();
    let h_dim = bundle.table_idx[0].len();
    let dim1 = bundle.table_idx[0][0].len();
    let dim2 = bundle.table_idx[0][0][0].len();
    let total_sparsity = b_dim * h_dim * dim1 * dim2;

    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        SOFTMAX_C,
        total_sparsity.next_power_of_two(),
        SOFTMAX_C,
        SOFTMAX_M.trailing_zeros() as usize,
    );

    let mut metrics_agg = LassoMetricsAggregate::default();

    let start = Instant::now();
    let (results, metrics) = prove_lasso_for_softmax(&bundle.table_idx, &gens, rng, label)?;
    metrics_agg.add(&metrics);
    println!(
        "  - Lasso Total Time (BERT Softmax {:?}): {:?}",
        std::str::from_utf8(label).unwrap_or("?"),
        start.elapsed()
    );

    Ok((results, metrics_agg))
}

/// Run all range-check Lasso proofs for a BERT attention circuit.
///
/// Targets:
///   22-bit (C=2): matmul_attn.r_matmul, matmul_x.r_matmul, projections, scaling
///   52-bit (C=4): sm.r_sm  (v3: 4 limbs [16,16,16,4])
///   16-bit 4D:    sm.r_div
pub fn run_all_bert_remainders_with_link(
    circuit: &BERTAttentionCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] BERT Attention: Verifying All Remainders ===");

    let mut all_results = Vec::new();
    let mut metrics_agg = LassoMetricsAggregate::default();

    // ── 26-bit range checks (sm.r_sm handled separately as 52-bit) ─────────
    let target_26bit: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&circuit.matmul_attn.r_matmul, "bert_attn_remainder"),
        (&circuit.matmul_x.r_matmul,   "bert_x_remainder"),
        // Projection remainders
        (&circuit.proj_q.r_out_proj,    "bert_proj_q_remainder"),
        (&circuit.proj_k.r_out_proj,    "bert_proj_k_remainder"),
        (&circuit.proj_v.r_out_proj,    "bert_proj_v_remainder"),
        (&circuit.proj_o.r_out_proj,    "bert_proj_o_remainder"),
        // Scaling remainders
        (&circuit.scale_q.r_scale,      "bert_scale_q_remainder"),
        (&circuit.scale_k.r_scale,      "bert_scale_k_remainder"),
    ];

    for (r_matrix, label) in target_26bit {
        let b_dim   = r_matrix.len();
        let h_dim   = r_matrix[0].len();
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
                        let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;  // 10 bits (26-bit total)
                        nz.push([idx0, idx1]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            KBITS_C,
            total_sparsity.next_power_of_two(),
            KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!(
            "  - {} verified | Count: {} | Time: {:?}",
            label, total_sparsity, start.elapsed()
        );
    }

    // ── 52-bit r_sm range check (v3: 4 limbs [16,16,16,4]) ─────────────────
    {
        let r_sm_4d = &circuit.sm.r_sm;
        let label = "bert_sm_r_sm_52bit";

        let b_dim = r_sm_4d.len();
        let h_dim = r_sm_4d[0].len();
        let s_dim = r_sm_4d[0][0].len();
        let l_dim = r_sm_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz: Vec<[usize; SM_KBITS_C]> = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = r_sm_4d[b][h][s][l].into_bigint().as_ref()[0];
                        nz.push([
                            (val_u64 & 0xFFFF) as usize,
                            ((val_u64 >> 16) & 0xFFFF) as usize,
                            ((val_u64 >> 32) & 0xFFFF) as usize,
                            ((val_u64 >> 48) & 0xF) as usize,    // 4 bits
                        ]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            SM_KBITS_C,
            total_sparsity.next_power_of_two(),
            SM_KBITS_C,
            SM_KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_52(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!(
            "  - {} verified | Count: {} | Time: {:?}",
            label, total_sparsity, start.elapsed()
        );
    }

    // ── 26-bit 4D: sm.r_div (bounded by exp_sum, can exceed 16-bit) ────────
    {
        let matrix_4d = &circuit.sm.r_div;
        let label = "bert_sm_r_div";

        let b_dim = matrix_4d.len();
        let h_dim = matrix_4d[0].len();
        let s_dim = matrix_4d[0][0].len();
        let l_dim = matrix_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = matrix_4d[b][h][s][l].into_bigint().as_ref()[0];
                        let idx0 = (val_u64 & 0xFFFF) as usize;
                        let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;  // 10 bits (26-bit total)
                        nz.push([idx0, idx1]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            KBITS_C,
            total_sparsity.next_power_of_two(),
            KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        println!(
            "  - {} verified | Count: {} | Time: {:?}",
            label, total_sparsity, start.elapsed()
        );
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
    }

    Ok((all_results, metrics_agg))
}
