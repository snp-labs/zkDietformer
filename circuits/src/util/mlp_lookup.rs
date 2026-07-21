use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{fs, UniformRand, log2, marker::PhantomData, vec::Vec, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{select::CondSelectGadget,alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use merlin::Transcript;
use crate::{
    cc_snark::legogroth16::{LinkPublicGenerators, create_random_proof_incl_cp_link, generate_random_parameters_incl_cp_link, prepare_verifying_key, verify_proof_incl_cp_link}, circuits::{common::{LassoMetrics, LassoMetricsAggregate}, freivalds_matmul::nystrom_mlp_circuit::{NystromMLPCircuit, NystromMLPLayerNormCircuit}, proof_size}, config::{GELU_C, GELU_M, KBITS_C, KBITS_M, RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M}, gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu}, layernorm::constraints::{NormBundle, LayerNormBundle, enforce_rmsnorm}, linear::constraints::{LinearBundle, enforce_linear}
    }, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialCommitment, SparsePolynomialEvaluationProof},
        }, poly::{commitments::{Commitments, MultiCommitGens}, dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentBlinds}}, subtables::{
            SubtableStrategy, and::AndSubtableStrategy, gelu::GeLUSubtableStrategy, layernorm::RMSNormSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
        }, utils::{math::Math, random::RandomTape}
    } 
};

type G1 = G1Projective;

pub fn run_lasso_for_rmsnorm(
    bundle: &NormBundle<Fr>,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<((PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize), LassoMetrics)> {
    let b_dim = bundle.q_ms_sim.len();
    let s_dim = bundle.q_ms_sim[0].len();
    let dim = bundle.q_out[0][0].len();

    let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];

    // Index = ((shifted_val * M_rms_idx) >> k) // dim
    let m_idx = bundle.m_rms_idx.into_bigint().as_ref()[0] as u128;

    for b in 0..b_dim {
        for s in 0..s_dim {
            indices_x[b][s][0] = bundle.q_ms_sim[b][s][0];
        }
    }

    let total_sparsity = b_dim * s_dim;
    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        RMSNORM_C,
        total_sparsity.next_power_of_two(),
        RMSNORM_C,
        RMSNORM_M.trailing_zeros() as usize,
    );

    let start = Instant::now();
    let ((pure_data_commitment, blinds, lasso_gens_n, sparsity), metrics) = prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
    println!("  - Lasso Total Time (RMSNorm): {:?}", start.elapsed());

    Ok(((pure_data_commitment, blinds, lasso_gens_n.clone(), sparsity), metrics))
}


// ── Phase 5-B B5-2: LayerNorm 측 inv_std lookup (Phase 4 65536-entry table) ──
//
// `run_lasso_for_rmsnorm` 와 평행. RMSNormSubtableStrategy 를 LayerNorm 으로
// alias 한 (B1-B2) 동일 strategy 사용. 차이는 인덱스 출처:
//   - RMSNorm: `bundle.q_ms_sim` (mean-square index)
//   - LayerNorm: `bundle.q_var_sim` (variance index, mean-제거 후)
//
// 표 자체는 set_layernorm_table_dir() / LAYERNORM_TABLE_PATH 로 결정 (B1).
// var_scale=1000 sanity check 은 표 로드 시 (subtables/layernorm.rs:get_or_init).
pub fn run_lasso_for_layernorm(
    bundle: &LayerNormBundle<Fr>,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<((PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize), LassoMetrics)> {
    let b_dim = bundle.q_var_sim.len();
    let s_dim = bundle.q_var_sim[0].len();

    let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];

    for b in 0..b_dim {
        for s in 0..s_dim {
            indices_x[b][s][0] = bundle.q_var_sim[b][s][0];
        }
    }

    let total_sparsity = b_dim * s_dim;
    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        RMSNORM_C,
        total_sparsity.next_power_of_two(),
        RMSNORM_C,
        RMSNORM_M.trailing_zeros() as usize,
    );

    let start = Instant::now();
    let ((pure_data_commitment, blinds, lasso_gens_n, sparsity), metrics) =
        prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
    println!("  - Lasso Total Time (LayerNorm): {:?}", start.elapsed());

    Ok(((pure_data_commitment, blinds, lasso_gens_n.clone(), sparsity), metrics))
}


pub fn run_lasso_for_gelu(
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
    let offset = 32768u128; // TABLE_SIZE // 2 for fixed-resolution 0-point
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
    let ((pure_data_commitment, blinds, lasso_gens_n, sparsity_out), metrics) = prove_lasso_for_gelu(indices, &gens, rng, label)?;
    Ok(((pure_data_commitment, blinds, lasso_gens_n, sparsity_out), metrics))
}


pub fn run_mlp_remainders<F: PrimeField, G: CurveGroup<ScalarField = F>>(
    circuit: &NystromMLPCircuit<F>,
    rng: &mut StdRng,
) -> Result<(Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>, LassoMetricsAggregate)>  {
    println!("\n=== [MLP] Lasso Lookup Proofs: 26-bit Remainders (R) ===");
    let mut all_link_results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    type Range26Lasso = RangeCheckSubtableStrategy<26>;

    let mut target_matrices = Vec::new();

    target_matrices.push((&circuit.norm1.r_shift, "R_mlp_norm1_shift"));
    target_matrices.push((&circuit.norm1.r_out, "R_mlp_norm1_out"));
    target_matrices.push((&circuit.ff1.r_out_lin, "R_mlp_lin1"));
    target_matrices.push((&circuit.gelu1.r_out_gelu, "R_mlp_gelu"));
    target_matrices.push((&circuit.ff2.r_out_lin, "R_mlp_lin2"));
    target_matrices.push((&circuit.norm2.r_shift, "R_mlp_norm2_shift"));
    target_matrices.push((&circuit.norm2.r_out, "R_mlp_norm2_out"));

    for (r_matrix, label) in target_matrices {
        let b_dim = r_matrix.len();
        let row_dim = r_matrix[0].len();
        let col_dim = r_matrix[0][0].len();
        let total_sparsity = b_dim * row_dim * col_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for r in 0..row_dim {
                for c in 0..col_dim {
                    let val_u64 = r_matrix[b][r][c].into_bigint().as_ref()[0];
                    let idx0 = (val_u64 & 0xFFFF) as usize;
                    let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;
                    nz.push([idx0, idx1]);
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
        let (link_data, metrics) = prove_lasso_for_range(
            nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;
        all_link_results.extend(link_data);
        agg.add(&metrics);

        println!(" - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());

    }
    println!("All MLP remainders verified sequentially.");
    Ok((all_link_results, agg))

}

// ── Phase 5-B B5-1: LayerNorm 측 22-bit Lasso remainder lookup ───────────────
//
// `NystromMLPLayerNormCircuit` 의 r_pool 순서 (enforce_layernorm + generate_constraints
// 정합) 를 따라 9 행렬 처리:
//   1. norm1.r_shift_var (3D)        — Step C (variance shift)
//   2. norm1.r_out (3D)              — Step F (final rescale)
//   3. norm1.r_beta_lift (1D)        — Step E (β lifting)
//   4. ff1.r_out_lin (3D)            — Linear1
//   5. gelu1.r_out_gelu (3D)         — GELU
//   6. ff2.r_out_lin (3D)            — Linear2
//   7. norm2.r_shift_var (3D)        — Step C
//   8. norm2.r_out (3D)              — Step F
//   9. norm2.r_beta_lift (1D)        — Step E
//
// RMSNorm 측 `run_mlp_remainders` 의 7 행렬 패턴을 LayerNorm 의 β lifting
// 추가분 (× 2) 만큼 확장. R_mean 은 option A 채택 (B5-1 분석) 으로 r_pool 미포함.
pub fn run_mlp_remainders_layernorm<F: PrimeField, G: CurveGroup<ScalarField = F>>(
    circuit: &NystromMLPLayerNormCircuit<F>,
    rng: &mut StdRng,
) -> Result<(Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>, LassoMetricsAggregate)>  {
    println!("\n=== [MLP-LayerNorm] Lasso Lookup Proofs: 22-bit Remainders (R) ===");
    let mut all_link_results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    enum RTarget<'a, F: PrimeField> {
        D3(&'a Vec<Vec<Vec<F>>>),
        D1(&'a Vec<F>),
    }

    let targets: Vec<(RTarget<F>, &'static str)> = vec![
        (RTarget::D3(&circuit.norm1.r_shift_var), "R_mlp_ln1_shift"),
        (RTarget::D3(&circuit.norm1.r_out),       "R_mlp_ln1_out"),
        (RTarget::D1(&circuit.norm1.r_beta_lift), "R_mlp_ln1_beta"),
        (RTarget::D3(&circuit.ff1.r_out_lin),     "R_mlp_lin1"),
        (RTarget::D3(&circuit.gelu1.r_out_gelu),  "R_mlp_gelu"),
        (RTarget::D3(&circuit.ff2.r_out_lin),     "R_mlp_lin2"),
        (RTarget::D3(&circuit.norm2.r_shift_var), "R_mlp_ln2_shift"),
        (RTarget::D3(&circuit.norm2.r_out),       "R_mlp_ln2_out"),
        (RTarget::D1(&circuit.norm2.r_beta_lift), "R_mlp_ln2_beta"),
    ];

    for (target, label) in targets {
        let mut nz: Vec<[usize; KBITS_C]> = Vec::new();
        match target {
            RTarget::D3(m) => {
                for batch in m {
                    for row in batch {
                        for v in row {
                            let val_u64 = v.into_bigint().as_ref()[0];
                            let idx0 = (val_u64 & 0xFFFF) as usize;
                            let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;
                            nz.push([idx0, idx1]);
                        }
                    }
                }
            }
            RTarget::D1(m) => {
                for v in m {
                    let val_u64 = v.into_bigint().as_ref()[0];
                    let idx0 = (val_u64 & 0xFFFF) as usize;
                    let idx1 = ((val_u64 >> 16) & 0x3FF) as usize;
                    nz.push([idx0, idx1]);
                }
            }
        }
        let total_sparsity = nz.len();

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            KBITS_C,
            total_sparsity.next_power_of_two(),
            KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range(
            nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;
        all_link_results.extend(link_data);
        agg.add(&metrics);

        println!(" - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }
    println!("All MLP-LayerNorm remainders verified sequentially.");
    Ok((all_link_results, agg))
}

pub fn prove_lasso_for_range(
    nz_indices: Vec<[usize; KBITS_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<(Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>, LassoMetrics)> {
    let sparsity = nz_indices.len();
    let mut metrics = LassoMetrics::default();

    let log_m = KBITS_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<KBITS_C>::new(nz_indices, log_m);
    let mut densified = DensifiedRepresentation::<Fr, KBITS_C>::from(&lookup_matrix);

    println!("\n[Debug] Lasso Range (Label: {:?}, Total: {})", std::str::from_utf8(label).unwrap(), densified.dim[0].len());
    for i in 0..KBITS_C {
        let data = &densified.dim[i].Z;
        let print_count = std::cmp::min(10, data.len());
        if data.len() > 0 {
            println!("  - Dim[{}] Head (10): {:?}", i, data[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  - Dim[{}] Tail (10): {:?}", i, data[data.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }
    }

    let commit_proof_start = Instant::now();
    let commitment = densified.commit::<G1>(gens);
    metrics.commit_proof_ms = commit_proof_start.elapsed().as_millis();

    let log_s = sparsity.next_power_of_two().trailing_zeros() as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape");
    let mut transcript = Transcript::new(b"lasso_range");

    let prove_start = Instant::now();
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, KBITS_C, KBITS_M, RangeCheckSubtableStrategy<26>>::prove(
        &mut densified,
        &r_point,
        gens,
        &mut transcript,
        &mut random_tape,
    );
    metrics.prove_ms = prove_start.elapsed().as_millis();

    let verify_start = Instant::now();
    let mut verifier_transcript = Transcript::new(b"lasso_range");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso Range verification failed for label: {:?}", std::str::from_utf8(label)))?;
    metrics.verify_ms = verify_start.elapsed().as_millis();

    // proof size report
    let report = proof_size::compute_lasso_proof_report::<G1, KBITS_C, KBITS_M, RangeCheckSubtableStrategy<26>>(&proof_lasso, &commitment);
    report.print_log(std::str::from_utf8(label).unwrap_or("range"));
    metrics.proof_size_bytes = report.total;
    metrics.commitment_proof_bytes = report.oracle_bytes + report.deref_bytes;

    println!("    -> Lasso proof verified for: {:?}", std::str::from_utf8(label).unwrap());
    // CP-Link: extract dim[i] slices from combined commitment (no commit_inner recomputation)
    let r_size_combined = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks_per_poly = densified.dim[0].len() / r_size_combined;
    assert!(chunks_per_poly > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let mut results = Vec::new();
    for i in 0..KBITS_C {
        let start = i * chunks_per_poly;
        let end = start + chunks_per_poly;

        let pure_cm = PolyCommitment {
            C: commitment.l_variate_polys_commitment.C[start..end].to_vec(),
        };
        let blinds = PolyCommitmentBlinds {
            blinds: vec![Fr::zero(); chunks_per_poly],
        };
        metrics.commitment_cplink_link_bytes += pure_cm.serialized_size(Compress::Yes);

        results.push((pure_cm, blinds, full_gens.clone(), sparsity));
    }

    Ok((results, metrics))
}


pub fn prove_lasso_for_rmsnorm(
    indices: &Vec<Vec<Vec<Fr>>>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<((PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize), LassoMetrics)> {
    let b_dim = indices.len();
    let s_dim = indices[0].len();
    let mut metrics = LassoMetrics::default();

    let mut nz_indices: Vec<[usize; RMSNORM_C]> = Vec::with_capacity(b_dim * s_dim);
    for b in 0..b_dim {
        for s in 0..s_dim {
            let idx_u64 = indices[b][s][0].into_bigint().as_ref()[0];
            if idx_u64 >= RMSNORM_M as u64 {
                return Err(anyhow::anyhow!("Lasso Index out of bounds"));
            }
            nz_indices.push([idx_u64 as usize]);
        }
    }

    let sparsity = nz_indices.len();
    let log_M = RMSNORM_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<RMSNORM_C>::new(nz_indices, log_M);
    let mut densified = DensifiedRepresentation::<Fr, RMSNORM_C>::from(&lookup_matrix);

    println!("\n[Debug] Lasso RMSNorm (Label: {:?}, Total: {})", std::str::from_utf8(label).unwrap(), densified.dim[0].len());
    let data = &densified.dim[0].Z;
    let print_count = std::cmp::min(10, data.len());
    if data.len() > 0 {
        println!("  Head (10): {:?}", data[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        println!("  Tail (10): {:?}", data[data.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
    }

    let mut random_tape = RandomTape::new(b"lasso_tape_rms");
    let mut transcript = Transcript::new(b"lasso_rmsnorm");

    let commit_proof_start = Instant::now();
    let commitment = densified.commit::<G1>(gens);
    metrics.commit_proof_ms = commit_proof_start.elapsed().as_millis();

    let log_s = (sparsity.next_power_of_two().trailing_zeros()) as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let prove_start = Instant::now();
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, RMSNORM_C, RMSNORM_M, RMSNormSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );
    metrics.prove_ms = prove_start.elapsed().as_millis();

    let verify_start = Instant::now();
    let mut verifier_transcript = Transcript::new(b"lasso_rmsnorm");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso verification failed"))?;
    metrics.verify_ms = verify_start.elapsed().as_millis();

    // proof size report
    let report = proof_size::compute_lasso_proof_report::<G1, RMSNORM_C, RMSNORM_M, RMSNormSubtableStrategy>(&proof_lasso, &commitment);
    report.print_log(std::str::from_utf8(label).unwrap_or("rmsnorm"));
    metrics.proof_size_bytes = report.total;
    metrics.commitment_proof_bytes = report.oracle_bytes + report.deref_bytes;

    // CP-Link: extract dim[0] slice from combined commitment (no commit_inner recomputation)
    let r_size_combined = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks_per_poly = densified.dim[0].len() / r_size_combined;
    assert!(chunks_per_poly > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let pure_data_commitment = PolyCommitment {
        C: commitment.l_variate_polys_commitment.C[..chunks_per_poly].to_vec(),
    };
    let blinds = PolyCommitmentBlinds {
        blinds: vec![Fr::zero(); chunks_per_poly],
    };
    metrics.commitment_cplink_link_bytes = pure_data_commitment.serialized_size(Compress::Yes);

    Ok(((pure_data_commitment, blinds, full_gens, sparsity), metrics))
}

pub fn prove_lasso_for_gelu(
    nz_indices: Vec<[usize; GELU_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<((PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize), LassoMetrics)> {
    let sparsity = nz_indices.len();
    let mut metrics = LassoMetrics::default();

    let log_m = GELU_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<GELU_C>::new(nz_indices, log_m);
    let mut densified = DensifiedRepresentation::<Fr, GELU_C>::from(&lookup_matrix);

    println!("\n[Debug] Lasso GeLU (Label: {:?}, Total: {})", std::str::from_utf8(label).unwrap(), densified.dim[0].len());
    let data = &densified.dim[0].Z;
    let print_count = std::cmp::min(10, data.len());
    if data.len() > 0 {
        println!("  Head (10): {:?}", data[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        println!("  Tail (10): {:?}", data[data.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
    }

    let mut random_tape = RandomTape::new(b"lasso_tape_gelu");
    let mut transcript = Transcript::new(b"lasso_gelu");

    let commit_proof_start = Instant::now();
    let commitment = densified.commit::<G1>(gens);
    metrics.commit_proof_ms = commit_proof_start.elapsed().as_millis();

    let r_point: Vec<Fr> = (0..sparsity.next_power_of_two().trailing_zeros() as usize)
        .map(|_| Fr::rand(rng)).collect();

    let prove_start = Instant::now();
    let proof = SparsePolynomialEvaluationProof::<G1, GELU_C, GELU_M, GeLUSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );
    metrics.prove_ms = prove_start.elapsed().as_millis();

    let verify_start = Instant::now();
    let mut v_transcript = Transcript::new(b"lasso_gelu");
    proof.verify(&commitment, &r_point, gens, &mut v_transcript)
        .map_err(|_| anyhow::anyhow!("GeLU Lasso verification failed"))?;
    metrics.verify_ms = verify_start.elapsed().as_millis();

    // proof size report
    let report = proof_size::compute_lasso_proof_report::<G1, GELU_C, GELU_M, GeLUSubtableStrategy>(&proof, &commitment);
    report.print_log(std::str::from_utf8(label).unwrap_or("gelu"));
    metrics.proof_size_bytes = report.total;
    metrics.commitment_proof_bytes = report.oracle_bytes + report.deref_bytes;

    println!("GeLU Lasso proof verified: {:?}", std::str::from_utf8(label).unwrap());

    // CP-Link: extract dim[0] slice from combined commitment (no commit_inner recomputation)
    let r_size_combined = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks_per_poly = densified.dim[0].len() / r_size_combined;
    assert!(chunks_per_poly > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let pure_data_commitment = PolyCommitment {
        C: commitment.l_variate_polys_commitment.C[..chunks_per_poly].to_vec(),
    };
    let blinds = PolyCommitmentBlinds {
        blinds: vec![Fr::zero(); chunks_per_poly],
    };
    metrics.commitment_cplink_link_bytes = pure_data_commitment.serialized_size(Compress::Yes);

    Ok(((pure_data_commitment, blinds, full_gens, sparsity), metrics))
}