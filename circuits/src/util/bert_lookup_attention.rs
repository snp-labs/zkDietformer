use std::time::Instant;
use anyhow::Result;
use ark_bn254::{Fr, G1Projective};
use ark_ff::{BigInteger, PrimeField};
use ark_std::rand::rngs::StdRng;
use crate::{
    circuits::bert_circuit::BertAttentionCircuit,
    config::{FRACBITS_C, FRACBITS_M, KBITS_C, KBITS_M, SOFTMAX_C, SOFTMAX_M},
    gadgets::softmax::constraints::SoftmaxBundle,
    lasso::{
        lasso::surge::SparsePolyCommitmentGens,
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
    },
    util::lookup_attention::{prove_lasso_for_softmax, prove_lasso_for_range_16, prove_lasso_for_range_22},
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
) -> Result<Vec<LassoResult>> {
    let b_dim = bundle.limbs[0].len();
    let h_dim = bundle.limbs[0][0].len();
    let dim1 = bundle.limbs[0][0][0].len();
    let dim2 = bundle.limbs[0][0][0][0].len();
    let total_sparsity = b_dim * h_dim * dim1 * dim2;

    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        SOFTMAX_C,
        total_sparsity.next_power_of_two(),
        SOFTMAX_C,
        SOFTMAX_M.trailing_zeros() as usize,
    );

    let start = Instant::now();
    let results = prove_lasso_for_softmax(&bundle.limbs, &gens, rng, label)?;
    println!(
        "  - Lasso Total Time (BERT Softmax {:?}): {:?}",
        std::str::from_utf8(label).unwrap_or("?"),
        start.elapsed()
    );

    Ok(results)
}

/// Run all range-check Lasso proofs for a BERT attention circuit.
///
/// Targets:
///   22-bit (C=2): matmul_attn.r_matmul, sm.r_sm, matmul_x.r_matmul
///   16-bit 5D:    sm.r_exp
///   16-bit 4D:    sm.r_div
pub fn run_all_bert_remainders_with_link(
    circuit: &BertAttentionCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<Vec<LassoResult>> {
    println!("\n=== [Lasso Lookup] BERT Attention: Verifying All Remainders ===");

    let mut all_results = Vec::new();

    // ── 22-bit range checks ──────────────────────────────────────────────────
    let target_22bit: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&circuit.matmul_attn.r_matmul, "bert_attn_remainder"),
        (&circuit.sm.r_sm,              "bert_sm_remainder"),
        (&circuit.matmul_x.r_matmul,   "bert_x_remainder"),
    ];

    for (r_matrix, label) in target_22bit {
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
                        let idx1 = ((val_u64 >> 16) & 0x3F) as usize;
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
        let link_data = prove_lasso_for_range_22(&nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        println!(
            "  - {} verified | Count: {} | Time: {:?}",
            label, total_sparsity, start.elapsed()
        );
    }

    // ── 16-bit 5D: sm.r_exp ──────────────────────────────────────────────────
    {
        let r_exp_5d = &circuit.sm.r_exp;
        let label = "bert_sm_r_exp";

        let c_seg  = r_exp_5d.len();                    // C (K_SEGMENTS-1)
        let b_batch = r_exp_5d[0].len();                 // B
        let h_num  = r_exp_5d[0][0].len();               // H
        let s1_num = r_exp_5d[0][0][0].len();            // S
        let s2_num = r_exp_5d[0][0][0][0].len();         // S
        let total_sparsity = c_seg * b_batch * h_num * s1_num * s2_num;

        // Iterate in (b, h, s1, s2, k) order to match the circuit r2_pool
        // accumulation order in enforce_softmax: for i { for j { for k { r2_pool.push(r_exp[k][i][j]) }}}
        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_batch {
            for h in 0..h_num {
                for s1 in 0..s1_num {
                    for s2 in 0..s2_num {
                        for k in 0..c_seg {
                            nz.push([r_exp_5d[k][b][h][s1][s2].into_bigint().as_ref()[0] as usize]);
                        }
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            FRACBITS_C,
            total_sparsity.next_power_of_two(),
            FRACBITS_C,
            FRACBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let link_data = prove_lasso_for_range_16(&nz, &gens, rng, label.as_bytes())?;
        println!(
            "  - {} verified | Count: {} | Time: {:?}",
            label, total_sparsity, start.elapsed()
        );
        all_results.push(link_data);
    }

    // ── 16-bit 4D: sm.r_div ──────────────────────────────────────────────────
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
                        nz.push([matrix_4d[b][h][s][l].into_bigint().as_ref()[0] as usize]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            FRACBITS_C,
            total_sparsity.next_power_of_two(),
            FRACBITS_C,
            FRACBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let link_data = prove_lasso_for_range_16(&nz, &gens, rng, label.as_bytes())?;
        println!(
            "  - {} verified | Count: {} | Time: {:?}",
            label, total_sparsity, start.elapsed()
        );
        all_results.push(link_data);
    }

    Ok(all_results)
}
