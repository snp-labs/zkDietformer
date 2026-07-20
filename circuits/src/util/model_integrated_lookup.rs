//! Lasso lookup helpers for the integrated circuits (Version A & B).
//!
//! Provides `run_layer_attn_lasso` and `run_layer_mlp_lasso` which dispatch
//! to variant-specific Lasso proof functions. Each uses layer-indexed labels
//! to ensure transcript uniqueness across multi-layer proofs.

use std::time::Instant;
use anyhow::Result;
use ark_bn254::{Fr, G1Projective};
use ark_ff::{PrimeField, BigInteger, Zero};
use ark_std::rand::rngs::StdRng;
use crate::{
    circuits::{
        common::{LassoMetrics, LassoMetricsAggregate},
        freivalds_matmul::model_integrated_circuit::{
            AttnWitness, MlpWitness, NystromAttnWitness, BertAttnWitness,
            NystromMLPRMSNormData, NystromMLPLayerNormData, LassoResult,
        },
    },
    config::{FRACBITS_C, FRACBITS_M, KBITS_C, KBITS_M, SM_KBITS_C, SM_KBITS_M, SOFTMAX_C, SOFTMAX_M,
             RMSNORM_C, RMSNORM_M, GELU_C, GELU_M},
    lasso::{
        lasso::surge::SparsePolyCommitmentGens,
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
    },
    util::{
        nystrom_attn_lookup::{prove_lasso_for_softmax, prove_lasso_for_range_22, prove_lasso_for_range_52},
        mlp_lookup::{prove_lasso_for_rmsnorm, prove_lasso_for_gelu, prove_lasso_for_range},
    },
};

type G1 = G1Projective;

/// Generate a unique Lasso label from layer index and component name.
/// Uses `Box::leak` for `&'static [u8]` lifetime (acceptable for benchmarks).
pub fn make_lasso_label(layer_idx: usize, component: &str) -> &'static [u8] {
    let s = format!("model_L{}_{}", layer_idx, component);
    Box::leak(s.into_boxed_str()).as_bytes()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Top-level Dispatchers
// ═══════════════════════════════════════════════════════════════════════════════

/// Run all Lasso proofs for one attention layer (variant-dispatched).
pub fn run_layer_attn_lasso(
    attn: &AttnWitness<Fr>,
    layer_idx: usize,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    match attn {
        AttnWitness::Nystrom(w) => run_nystrom_attn_lasso(w, layer_idx, rng),
        AttnWitness::Bert(w) => run_bert_attn_lasso(w, layer_idx, rng),
    }
}

/// Run all Lasso proofs for one MLP layer (variant-dispatched).
pub fn run_layer_mlp_lasso(
    mlp: &MlpWitness<Fr>,
    layer_idx: usize,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    match mlp {
        MlpWitness::LayerNorm(data) => run_layernorm_mlp_lasso(data, layer_idx, rng),
        MlpWitness::RMSNorm(data) => run_rmsnorm_mlp_lasso(data, layer_idx, rng),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Nystrom Attention Lasso
// ═══════════════════════════════════════════════════════════════════════════════

fn run_nystrom_attn_lasso(
    w: &NystromAttnWitness<Fr>,
    layer_idx: usize,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    let mut results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    // Softmax 3 bundles (sm1, sm2, sm3) — v3: table_idx is 4D
    for (sm_bundle, name) in [(&w.sm1, "attn_sm1"), (&w.sm2, "attn_sm2"), (&w.sm3, "attn_sm3")] {
        let label = make_lasso_label(layer_idx, name);
        let b_dim = sm_bundle.table_idx.len();
        let h_dim = sm_bundle.table_idx[0].len();
        let dim1 = sm_bundle.table_idx[0][0].len();
        let dim2 = sm_bundle.table_idx[0][0][0].len();
        let total_sparsity = b_dim * h_dim * dim1 * dim2;

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, SOFTMAX_C, total_sparsity.next_power_of_two(), SOFTMAX_C, SOFTMAX_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_results, metrics) = prove_lasso_for_softmax(&sm_bundle.table_idx, &gens, rng, label)?;
        agg.add(&metrics);
        results.extend(link_results);
        println!("  - Lasso Softmax L{} {} | Time: {:?}", layer_idx, name, start.elapsed());
    }

    // Remainder proofs (22-bit range checks)
    let target_22bit: Vec<(&Vec<Vec<Vec<Vec<Fr>>>>, &str)> = vec![
        (&w.matmul1.r_matmul, "attn_k1_rem"),
        (&w.matmul2.r_matmul, "attn_k2_rem"),
        (&w.matmul3.r_matmul, "attn_k3_rem"),
    ];

    // Pseudoinverse steps
    let mut full_22bit = target_22bit;
    for (i, r_inv) in w.pseudo_inv.r_inv_remainders.iter().enumerate() {
        let label_str: &'static str = Box::leak(format!("attn_inv{}_rem", i).into_boxed_str());
        full_22bit.push((r_inv, label_str));
    }
    full_22bit.push((&w.matmul_temp.r_matmul, "attn_temp_rem"));
    full_22bit.push((&w.matmul_mid.r_matmul, "attn_mid_rem"));
    full_22bit.push((&w.matmul_x.r_matmul, "attn_x_rem"));
    // Projection remainders
    full_22bit.push((&w.proj_q.r_out_proj, "proj_q_rem"));
    full_22bit.push((&w.proj_k.r_out_proj, "proj_k_rem"));
    full_22bit.push((&w.proj_v.r_out_proj, "proj_v_rem"));
    full_22bit.push((&w.proj_o.r_out_proj, "proj_o_rem"));
    // Scaling remainders
    full_22bit.push((&w.scale_q.r_scale, "scale_q_rem"));
    full_22bit.push((&w.scale_k.r_scale, "scale_k_rem"));

    for (r_matrix, name) in &full_22bit {
        let label = make_lasso_label(layer_idx, name);
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
                        nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3F) as usize]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, KBITS_C, total_sparsity.next_power_of_two(), KBITS_C, KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} {} verified | Count: {} | Time: {:?}", layer_idx, name, total_sparsity, start.elapsed());
    }

    // 52-bit 4D: r_sm for 3 softmaxes (v3: 4 limbs [16,16,16,4])
    for (r_sm_4d, name) in [(&w.sm1.r_sm, "attn_sm1_rsm_52bit"), (&w.sm2.r_sm, "attn_sm2_rsm_52bit"), (&w.sm3.r_sm, "attn_sm3_rsm_52bit")] {
        let label = make_lasso_label(layer_idx, name);
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
            label, SM_KBITS_C, total_sparsity.next_power_of_two(), SM_KBITS_C, SM_KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_52(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} {} verified | Count: {} | Time: {:?}", layer_idx, name, total_sparsity, start.elapsed());
    }

    // 26-bit 4D: r_div for 3 softmaxes (bounded by exp_sum, can exceed 16-bit)
    for (r_div_4d, name) in [(&w.sm1.r_div, "attn_sm1_rdiv"), (&w.sm2.r_div, "attn_sm2_rdiv"), (&w.sm3.r_div, "attn_sm3_rdiv")] {
        let label = make_lasso_label(layer_idx, name);
        let b_dim = r_div_4d.len();
        let h_dim = r_div_4d[0].len();
        let s_dim = r_div_4d[0][0].len();
        let l_dim = r_div_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = r_div_4d[b][h][s][l].into_bigint().as_ref()[0];
                        let idx0 = (val_u64 & 0xFFFF) as usize;
                        let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;
                        nz.push([idx0, idx1]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, KBITS_C, total_sparsity.next_power_of_two(), KBITS_C, KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} {} verified | Count: {} | Time: {:?}", layer_idx, name, total_sparsity, start.elapsed());
    }

    Ok((results, agg))
}

// ═══════════════════════════════════════════════════════════════════════════════
// BERT Attention Lasso
// ═══════════════════════════════════════════════════════════════════════════════

fn run_bert_attn_lasso(
    w: &BertAttnWitness<Fr>,
    layer_idx: usize,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    let mut results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    // Single softmax — v3: table_idx is 4D
    {
        let label = make_lasso_label(layer_idx, "attn_sm");
        let b_dim = w.sm.table_idx.len();
        let h_dim = w.sm.table_idx[0].len();
        let dim1 = w.sm.table_idx[0][0].len();
        let dim2 = w.sm.table_idx[0][0][0].len();
        let total_sparsity = b_dim * h_dim * dim1 * dim2;

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, SOFTMAX_C, total_sparsity.next_power_of_two(), SOFTMAX_C, SOFTMAX_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_results, metrics) = prove_lasso_for_softmax(&w.sm.table_idx, &gens, rng, label)?;
        agg.add(&metrics);
        results.extend(link_results);
        println!("  - Lasso Softmax L{} bert_sm | Time: {:?}", layer_idx, start.elapsed());
    }

    // 22-bit remainder checks
    let target_22bit: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&w.matmul_attn.r_matmul, "attn_rem"),
        (&w.matmul_x.r_matmul, "attn_x_rem"),
        // Projection remainders
        (&w.proj_q.r_out_proj, "proj_q_rem"),
        (&w.proj_k.r_out_proj, "proj_k_rem"),
        (&w.proj_v.r_out_proj, "proj_v_rem"),
        (&w.proj_o.r_out_proj, "proj_o_rem"),
        // Scaling remainders
        (&w.scale_q.r_scale, "scale_q_rem"),
        (&w.scale_k.r_scale, "scale_k_rem"),
    ];

    for (r_matrix, name) in target_22bit {
        let label = make_lasso_label(layer_idx, name);
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
                        nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3F) as usize]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, KBITS_C, total_sparsity.next_power_of_two(), KBITS_C, KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} {} verified | Count: {} | Time: {:?}", layer_idx, name, total_sparsity, start.elapsed());
    }

    // 52-bit 4D: sm.r_sm (v3: 4 limbs [16,16,16,4])
    {
        let r_sm_4d = &w.sm.r_sm;
        let label = make_lasso_label(layer_idx, "attn_sm_rsm_52bit");
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
            label, SM_KBITS_C, total_sparsity.next_power_of_two(), SM_KBITS_C, SM_KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_52(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} attn_sm_rsm_52bit verified | Count: {} | Time: {:?}", layer_idx, total_sparsity, start.elapsed());
    }

    // 26-bit 4D: sm.r_div (bounded by exp_sum, can exceed 16-bit)
    {
        let matrix_4d = &w.sm.r_div;
        let label = make_lasso_label(layer_idx, "attn_sm_rdiv");
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
                        let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;
                        nz.push([idx0, idx1]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, KBITS_C, total_sparsity.next_power_of_two(), KBITS_C, KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} attn_sm_rdiv verified | Count: {} | Time: {:?}", layer_idx, total_sparsity, start.elapsed());
    }

    Ok((results, agg))
}

// ═══════════════════════════════════════════════════════════════════════════════
// RMSNorm MLP Lasso
// ═══════════════════════════════════════════════════════════════════════════════

fn run_rmsnorm_mlp_lasso(
    data: &NystromMLPRMSNormData<Fr>,
    layer_idx: usize,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    let mut results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    // rms1 lookup
    {
        let label = make_lasso_label(layer_idx, "mlp_rms1");
        let b_dim = data.norm1.q_ms_sim.len();
        let s_dim = data.norm1.q_ms_sim[0].len();
        let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
        for b in 0..b_dim { for s in 0..s_dim { indices_x[b][s][0] = data.norm1.q_ms_sim[b][s][0]; } }
        let total_sparsity = b_dim * s_dim;
        let gens = SparsePolyCommitmentGens::<G1>::new(label, RMSNORM_C, total_sparsity.next_power_of_two(), RMSNORM_C, RMSNORM_M.trailing_zeros() as usize);
        let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
        results.push((cm, blinds, cgens, sp));
        agg.add(&metrics);
    }

    // gelu lookup
    {
        let label = make_lasso_label(layer_idx, "mlp_gelu");
        let q_in_sim: Vec<Vec<Vec<Fr>>> = data.ff1.q_out_sim.iter()
            .map(|batch| batch.iter()
                .map(|seq| seq.iter().map(|val: &Fr| Fr::from(val.into_bigint().as_ref()[0] as i64)).collect())
                .collect())
            .collect();
        let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_gelu_integrated(
            &data.gelu1, &q_in_sim, &data.ff1.r_out_lin, data.k_bits, label, rng,
        )?;
        results.push((cm, blinds, cgens, sp));
        agg.add(&metrics);
    }

    // rms2 lookup
    {
        let label = make_lasso_label(layer_idx, "mlp_rms2");
        let b_dim = data.norm2.q_ms_sim.len();
        let s_dim = data.norm2.q_ms_sim[0].len();
        let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
        for b in 0..b_dim { for s in 0..s_dim { indices_x[b][s][0] = data.norm2.q_ms_sim[b][s][0]; } }
        let total_sparsity = b_dim * s_dim;
        let gens = SparsePolyCommitmentGens::<G1>::new(label, RMSNORM_C, total_sparsity.next_power_of_two(), RMSNORM_C, RMSNORM_M.trailing_zeros() as usize);
        let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
        results.push((cm, blinds, cgens, sp));
        agg.add(&metrics);
    }

    // MLP remainders (7 matrices for RMSNorm)
    let target_matrices: Vec<(&Vec<Vec<Vec<Fr>>>, &str)> = vec![
        (&data.norm1.r_shift, "mlp_rms1_rshift"),
        (&data.norm1.r_out, "mlp_rms1_rout"),
        (&data.ff1.r_out_lin, "mlp_lin1_r"),
        (&data.gelu1.r_out_gelu, "mlp_gelu_r"),
        (&data.ff2.r_out_lin, "mlp_lin2_r"),
        (&data.norm2.r_shift, "mlp_rms2_rshift"),
        (&data.norm2.r_out, "mlp_rms2_rout"),
    ];

    for (r_matrix, name) in &target_matrices {
        let label = make_lasso_label(layer_idx, name);
        let b_dim = r_matrix.len();
        let row_dim = r_matrix[0].len();
        let col_dim = r_matrix[0][0].len();
        let total_sparsity = b_dim * row_dim * col_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for r in 0..row_dim {
                for c in 0..col_dim {
                    let val_u64 = r_matrix[b][r][c].into_bigint().as_ref()[0];
                    nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3F) as usize]);
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, KBITS_C, total_sparsity.next_power_of_two(), KBITS_C, KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} {} verified | Count: {} | Time: {:?}", layer_idx, name, total_sparsity, start.elapsed());
    }

    Ok((results, agg))
}

// ═══════════════════════════════════════════════════════════════════════════════
// LayerNorm MLP Lasso
// ═══════════════════════════════════════════════════════════════════════════════

fn run_layernorm_mlp_lasso(
    data: &NystromMLPLayerNormData<Fr>,
    layer_idx: usize,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    let mut results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    // ln1 lookup (uses q_var_sim for LayerNorm)
    {
        let label = make_lasso_label(layer_idx, "mlp_ln1");
        let b_dim = data.norm1.q_var_sim.len();
        let s_dim = data.norm1.q_var_sim[0].len();
        let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
        for b in 0..b_dim { for s in 0..s_dim { indices_x[b][s][0] = data.norm1.q_var_sim[b][s][0]; } }
        let total_sparsity = b_dim * s_dim;
        let gens = SparsePolyCommitmentGens::<G1>::new(label, RMSNORM_C, total_sparsity.next_power_of_two(), RMSNORM_C, RMSNORM_M.trailing_zeros() as usize);
        let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
        results.push((cm, blinds, cgens, sp));
        agg.add(&metrics);
    }

    // gelu lookup
    {
        let label = make_lasso_label(layer_idx, "mlp_gelu");
        let q_in_sim: Vec<Vec<Vec<Fr>>> = data.ff1.q_out_sim.iter()
            .map(|batch| batch.iter()
                .map(|seq| seq.iter().map(|val: &Fr| Fr::from(val.into_bigint().as_ref()[0] as i64)).collect())
                .collect())
            .collect();
        let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_gelu_integrated(
            &data.gelu1, &q_in_sim, &data.ff1.r_out_lin, data.k_bits, label, rng,
        )?;
        results.push((cm, blinds, cgens, sp));
        agg.add(&metrics);
    }

    // ln2 lookup
    {
        let label = make_lasso_label(layer_idx, "mlp_ln2");
        let b_dim = data.norm2.q_var_sim.len();
        let s_dim = data.norm2.q_var_sim[0].len();
        let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
        for b in 0..b_dim { for s in 0..s_dim { indices_x[b][s][0] = data.norm2.q_var_sim[b][s][0]; } }
        let total_sparsity = b_dim * s_dim;
        let gens = SparsePolyCommitmentGens::<G1>::new(label, RMSNORM_C, total_sparsity.next_power_of_two(), RMSNORM_C, RMSNORM_M.trailing_zeros() as usize);
        let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
        results.push((cm, blinds, cgens, sp));
        agg.add(&metrics);
    }

    // MLP remainders (9 matrices for LayerNorm, including r_beta_lift)
    enum RTarget<'a> {
        D3(&'a Vec<Vec<Vec<Fr>>>),
        D1(&'a Vec<Fr>),
    }

    let targets: Vec<(RTarget, &str)> = vec![
        (RTarget::D3(&data.norm1.r_shift_var), "mlp_ln1_rshift"),
        (RTarget::D3(&data.norm1.r_out), "mlp_ln1_rout"),
        (RTarget::D1(&data.norm1.r_beta_lift), "mlp_ln1_rbeta"),
        (RTarget::D3(&data.ff1.r_out_lin), "mlp_lin1_r"),
        (RTarget::D3(&data.gelu1.r_out_gelu), "mlp_gelu_r"),
        (RTarget::D3(&data.ff2.r_out_lin), "mlp_lin2_r"),
        (RTarget::D3(&data.norm2.r_shift_var), "mlp_ln2_rshift"),
        (RTarget::D3(&data.norm2.r_out), "mlp_ln2_rout"),
        (RTarget::D1(&data.norm2.r_beta_lift), "mlp_ln2_rbeta"),
    ];

    for (target, name) in targets {
        let label = make_lasso_label(layer_idx, name);
        let mut nz: Vec<[usize; KBITS_C]> = Vec::new();
        match target {
            RTarget::D3(m) => {
                for batch in m {
                    for row in batch {
                        for v in row {
                            let val_u64 = v.into_bigint().as_ref()[0];
                            nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3F) as usize]);
                        }
                    }
                }
            }
            RTarget::D1(m) => {
                for v in m {
                    let val_u64 = v.into_bigint().as_ref()[0];
                    nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3F) as usize]);
                }
            }
        }
        let total_sparsity = nz.len();

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label, KBITS_C, total_sparsity.next_power_of_two(), KBITS_C, KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range(nz, &gens, rng, label)?;
        results.extend(link_data);
        agg.add(&metrics);
        println!("  - L{} {} verified | Count: {} | Time: {:?}", layer_idx, name, total_sparsity, start.elapsed());
    }

    Ok((results, agg))
}

// ═══════════════════════════════════════════════════════════════════════════════
// GeLU Helper (integrated version with layer labels)
// ═══════════════════════════════════════════════════════════════════════════════

use crate::gadgets::activation::gelu::GeLUBundle;

fn prove_lasso_for_gelu_integrated(
    bundle: &GeLUBundle<Fr>,
    q_in_sim: &Vec<Vec<Vec<Fr>>>,
    r_in_lin: &Vec<Vec<Vec<Fr>>>,
    k_bits: usize,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<((PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize), LassoMetrics)> {
    let b_dim = bundle.q_out_sim.len();
    let s_dim = bundle.q_out_sim[0].len();
    let d_dim = bundle.q_out_sim[0][0].len();

    let mut indices = Vec::new();
    let offset = 32768u128;
    let m_gelu_idx = bundle.m_gelu_idx.into_bigint().as_ref()[0] as u128;

    for b in 0..b_dim {
        for s in 0..s_dim {
            for d in 0..d_dim {
                let q_in_diff = (q_in_sim[b][s][d].into_bigint().as_ref()[0] as i128)
                    - (bundle.z_in.into_bigint().as_ref()[0] as i128);
                let r_in = r_in_lin[b][s][d].into_bigint().as_ref()[0] as i128;
                let x_diff_full = (q_in_diff << k_bits) + r_in;
                let scaled_diff = (x_diff_full * m_gelu_idx as i128) >> (2 * k_bits);
                let mut idx = scaled_diff + offset as i128;
                if idx < 0 { idx = 0; }
                if idx >= GELU_M as i128 { idx = (GELU_M - 1) as i128; }
                indices.push([idx as usize]);
            }
        }
    }

    let sparsity = indices.len();
    let gens = SparsePolyCommitmentGens::<G1>::new(label, GELU_C, sparsity.next_power_of_two(), GELU_C, 16);
    let start = Instant::now();
    let ((cm, blinds, cgens, sp), metrics) = prove_lasso_for_gelu(indices, &gens, rng, label)?;
    println!("  - Lasso GeLU {:?} | Time: {:?}", std::str::from_utf8(label).unwrap_or("?"), start.elapsed());
    Ok(((cm, blinds, cgens, sp), metrics))
}
