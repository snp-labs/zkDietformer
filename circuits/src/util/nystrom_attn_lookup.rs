use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix1, Ix4, Array2, Ix2, Ix3, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{fs, UniformRand, log2, marker::PhantomData, rand::{Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use merlin::Transcript;
use ark_r1cs_std::select::CondSelectGadget;
use crate::{
    circuits::{
        common::{LassoMetrics, LassoMetricsAggregate},
        freivalds_matmul::nystrom_attn_circuit::NystromAttentionCircuit,
        proof_size,
    },
    config::{FRACBITS_C, FRACBITS_M, KBITS_C, KBITS_M, SOFTMAX_C, SOFTMAX_M}, gadgets::{
        matmul::{constraints::{MatrixVar, new_matrix_witness},
    quantized::{MatmulBundle, MatmulData, PseudoinverseBundle, ZenParams, compute_circuit_landmarks, gen_challenge_vector, new_matrix_witness_4d, transpose_matrix}}, softmax::constraints::{SoftmaxBundle, enforce_softmax}}, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        }, poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}}, subtables::{
            SubtableStrategy, and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
        }, utils::{math::Math, random::RandomTape}
    }, util::load::{read_array_i64, to_field_2d_i64, to_field_3d_u64, to_field_4d_i64, to_field_4d_u64, to_field_5d_i64, to_field_5d_u64}
};

type G1 = G1Projective;

pub fn run_lasso_bundle(
    bundle: &SoftmaxBundle<Fr>,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<(Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>, LassoMetricsAggregate)> {
    // 총 룩업 횟수(Sparsity) 계산: C, B, H, S, L 차원을 가짐
    let b_dim = bundle.limbs[0].len();
    let h_dim = bundle.limbs[0][0].len();
    let dim1 = bundle.limbs[0][0][0].len();
    let dim2 = bundle.limbs[0][0][0][0].len();
    let total_sparsity = b_dim * h_dim * dim1 * dim2;

    // Lasso용 Generator 생성 (각 번들의 고유 sparsity 반영)
    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        SOFTMAX_C,
        total_sparsity.next_power_of_two(),
        SOFTMAX_C,
        SOFTMAX_M.trailing_zeros() as usize,
    );

    let start = Instant::now();
    let (link_results, metrics) = prove_lasso_for_softmax(&bundle.limbs, &gens, rng, label)?;
    println!("  - Lasso Total Time ({:?}): {:?}", std::str::from_utf8(label).unwrap(), start.elapsed());

    let mut agg = LassoMetricsAggregate::default();
    agg.add(&metrics);
    Ok((link_results, agg))
}

pub fn prove_lasso_for_softmax(
    limbs: &Vec<Vec<Vec<Vec<Vec<Fr>>>>>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<(Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>, LassoMetrics)> {

    use ark_ff::{BigInteger, PrimeField};
    let c_seg = limbs.len();          // SOFTMAX_C (5)
    let b_dim = limbs[0].len();       // Batch
    let h_dim = limbs[0][0].len();    // Head
    let s_dim = limbs[0][0][0].len(); // Sequence
    let l_dim = limbs[0][0][0][0].len(); // Landmark (L)

    let mut metrics = LassoMetrics::default();

    // Lasso 입력용 nz_indices (Sparsity m = B*H*S*L)
    let mut nz_indices: Vec<[usize; SOFTMAX_C]> = Vec::with_capacity(b_dim * h_dim * s_dim * l_dim);

    for b in 0..b_dim {
        for h in 0..h_dim {
            for s in 0..s_dim {
                for l in 0..l_dim {
                    let mut idx_arr = [0usize; SOFTMAX_C];
                    for seg in 0..c_seg {
                        idx_arr[seg] = limbs[seg][b][h][s][l].into_bigint().as_ref()[0] as usize;
                    }
                    nz_indices.push(idx_arr);
                }
            }
        }
    }

    let sparsity = nz_indices.len();
    println!("    * Sparsity (m): {}", sparsity);

    let log_M = SOFTMAX_M.trailing_zeros() as usize; //
    let lookup_matrix = SparseLookupMatrix::<SOFTMAX_C>::new(nz_indices, log_M);
    let mut densified = DensifiedRepresentation::<Fr, SOFTMAX_C>::from(&lookup_matrix);

    println!("\n[Debug] Lasso Range (Label: {:?}, Total: {})", std::str::from_utf8(label).unwrap(), densified.dim[0].len());
    for i in 0..KBITS_C {
        let data = &densified.dim[i].Z;
        let print_count = std::cmp::min(10, data.len());
        if data.len() > 0 {
            println!("  - Dim[{}] Head (10): {:?}", i, data[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  - Dim[{}] Tail (10): {:?}", i, data[data.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }
    }

    let commit_start = Instant::now();
    let commitment = densified.commit::<G1>(gens);
    let commit_duration = commit_start.elapsed();
    metrics.commit_proof_ms = commit_duration.as_millis();
    println!("    * Commitment Time: {:?}", commit_duration);

    let log_s = (sparsity.next_power_of_two().trailing_zeros()) as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape");
    let mut transcript = Transcript::new(b"lasso_softmax");

    println!("    * Starting Lasso Prove (log_s: {})...", log_s);
    let prove_start = Instant::now();
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, SOFTMAX_C, SOFTMAX_M, SoftmaxSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let prove_duration = prove_start.elapsed();
    metrics.prove_ms = prove_duration.as_millis();
    println!("    * Lasso Prover Time: {:?}", prove_duration);

    // ── Lasso proof size report ──
    let lasso_label = std::str::from_utf8(label).unwrap_or("softmax");
    let lasso_report = proof_size::compute_lasso_proof_report::<G1, SOFTMAX_C, SOFTMAX_M, SoftmaxSubtableStrategy>(&proof_lasso, &commitment);
    lasso_report.print_log(lasso_label);
    metrics.proof_size_bytes = lasso_report.total;
    metrics.commitment_proof_bytes = lasso_report.oracle_bytes + lasso_report.deref_bytes;

    let mut verifier_transcript = Transcript::new(b"lasso_softmax");
    let verify_start = Instant::now();
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso verification failed for label: {:?}", std::str::from_utf8(label)))?;
    let verify_duration = verify_start.elapsed();
    metrics.verify_ms = verify_duration.as_millis();

    println!("Lasso proof verified for: {:?}", lasso_label);
    println!("    * Lasso Verifier Time: {:?}", verify_duration);

    // CP-Link: extract dim[i] slices from combined commitment (no commit_inner recomputation)
    let r_size_combined = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks_per_poly = densified.dim[0].len() / r_size_combined;
    assert!(chunks_per_poly > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let mut results = Vec::new();
    for i in 0..SOFTMAX_C {
        let start = i * chunks_per_poly;
        let end = start + chunks_per_poly;

        let ccl_start = Instant::now();
        let pure_cm = PolyCommitment {
            C: commitment.l_variate_polys_commitment.C[start..end].to_vec(),
        };
        let blinds = PolyCommitmentBlinds {
            blinds: vec![Fr::zero(); chunks_per_poly],
        };
        metrics.commit_cplink_link_ms += ccl_start.elapsed().as_millis();
        metrics.commitment_cplink_link_bytes += pure_cm.serialized_size(Compress::Yes);

        results.push((pure_cm, blinds, full_gens.clone(), sparsity));
    }

    Ok((results, metrics))
}


pub fn prove_lasso_for_range_22(
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
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, KBITS_C, KBITS_M, RangeCheckSubtableStrategy<22>>::prove(
        &mut densified,
        &r_point,
        gens,
        &mut transcript,
        &mut random_tape,
    );
    metrics.prove_ms = prove_start.elapsed().as_millis();

    // ── Lasso proof size report ──
    let range_label = std::str::from_utf8(label).unwrap_or("range_22");
    let lasso_report = proof_size::compute_lasso_proof_report::<G1, KBITS_C, KBITS_M, RangeCheckSubtableStrategy<22>>(&proof_lasso, &commitment);
    lasso_report.print_log(range_label);
    metrics.proof_size_bytes = lasso_report.total;
    metrics.commitment_proof_bytes = lasso_report.oracle_bytes + lasso_report.deref_bytes;

    let verify_start = Instant::now();
    let mut verifier_transcript = Transcript::new(b"lasso_range");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso Range verification failed for label: {:?}", std::str::from_utf8(label)))?;
    metrics.verify_ms = verify_start.elapsed().as_millis();

    println!("    -> Lasso proof verified for: {:?}", range_label);

    // CP-Link: extract dim[i] slices from combined commitment (no commit_inner recomputation)
    let r_size_combined = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks_per_poly = densified.dim[0].len() / r_size_combined;
    assert!(chunks_per_poly > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let mut results = Vec::new();
    for i in 0..KBITS_C {
        let start = i * chunks_per_poly;
        let end = start + chunks_per_poly;

        let ccl_start = Instant::now();
        let pure_cm = PolyCommitment {
            C: commitment.l_variate_polys_commitment.C[start..end].to_vec(),
        };
        let blinds = PolyCommitmentBlinds {
            blinds: vec![Fr::zero(); chunks_per_poly],
        };
        metrics.commit_cplink_link_ms += ccl_start.elapsed().as_millis();
        metrics.commitment_cplink_link_bytes += pure_cm.serialized_size(Compress::Yes);

        results.push((pure_cm, blinds, full_gens.clone(), sparsity));
    }

    Ok((results, metrics))
}

pub fn prove_lasso_for_range_16(
    nz_indices: Vec<[usize; FRACBITS_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<((PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize), LassoMetrics)> {
    let sparsity = nz_indices.len();
    let mut metrics = LassoMetrics::default();

    let log_m = FRACBITS_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<FRACBITS_C>::new(nz_indices, log_m);
    let mut densified = DensifiedRepresentation::<Fr, FRACBITS_C>::from(&lookup_matrix);

    for i in 0..FRACBITS_C {
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
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, FRACBITS_C, FRACBITS_M, RangeCheckSubtableStrategy<22>>::prove(
        &mut densified,
        &r_point,
        gens,
        &mut transcript,
        &mut random_tape,
    );
    metrics.prove_ms = prove_start.elapsed().as_millis();

    // ── Lasso proof size report ──
    let range_label = std::str::from_utf8(label).unwrap_or("range_16");
    let lasso_report = proof_size::compute_lasso_proof_report::<G1, FRACBITS_C, FRACBITS_M, RangeCheckSubtableStrategy<22>>(&proof_lasso, &commitment);
    lasso_report.print_log(range_label);
    metrics.proof_size_bytes = lasso_report.total;
    metrics.commitment_proof_bytes = lasso_report.oracle_bytes + lasso_report.deref_bytes;

    let verify_start = Instant::now();
    let mut verifier_transcript = Transcript::new(b"lasso_range");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso Range verification failed for label: {:?}", std::str::from_utf8(label)))?;
    metrics.verify_ms = verify_start.elapsed().as_millis();

    println!("    -> Lasso proof verified for: {:?}", range_label);

    // CP-Link: extract dim[0] slice from combined commitment (no commit_inner recomputation)
    let r_size_combined = gens.gens_combined_l_variate.gens.gens_n.n;
    let chunks_per_poly = densified.dim[0].len() / r_size_combined;
    assert!(chunks_per_poly > 0, "sparsity too small for CP-Link extraction");
    let full_gens = gens.gens_combined_l_variate.gens.gens_n.clone();

    let ccl_start = Instant::now();
    let pure_cm = PolyCommitment {
        C: commitment.l_variate_polys_commitment.C[..chunks_per_poly].to_vec(),
    };
    let blinds = PolyCommitmentBlinds {
        blinds: vec![Fr::zero(); chunks_per_poly],
    };
    metrics.commit_cplink_link_ms = ccl_start.elapsed().as_millis();
    metrics.commitment_cplink_link_bytes = pure_cm.serialized_size(Compress::Yes);

    Ok(((pure_cm, blinds, full_gens, sparsity), metrics))
}

pub fn run_all_remainders_with_link<F: PrimeField, G: CurveGroup<ScalarField = F>>(
    circuit: &NystromAttentionCircuit<F>,
    rng: &mut StdRng,
) -> Result<(Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] Verifying All 22-bit Remainders (R_*) ===");

    let mut all_results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    // 22비트 범위 체크를 위한 전략 설정 (M=2^16, C=2)
    type Range22Lasso = RangeCheckSubtableStrategy<22>;
    const C: usize = 2;
    const M: usize = 1 << 16;

    // 검증할 나머지 행렬들을 수집
    let mut target_22bit = Vec::new();
    target_22bit.push((&circuit.matmul1.r_matmul, "matmul_k1_remainder"));
    target_22bit.push((&circuit.matmul2.r_matmul, "matmul_k2_remainder"));
    target_22bit.push((&circuit.matmul3.r_matmul, "matmul_k3_remainder"));

    target_22bit.push((&circuit.sm1.r_sm, "softmax_k1_remainder"));
    target_22bit.push((&circuit.sm2.r_sm, "softmax_k2_remainder"));
    target_22bit.push((&circuit.sm3.r_sm, "softmax_k3_remainder"));

    for (i, r_inv) in circuit.pseudo_inv.r_inv_remainders.iter().enumerate() {
        let label: &'static str = Box::leak(format!("pseudoinverse_step{}_remainder", i).into_boxed_str());
        target_22bit.push((r_inv, label));
    }

    target_22bit.push((&circuit.matmul_temp.r_matmul, "matmul_v_remainder"));
    target_22bit.push((&circuit.matmul_mid.r_matmul, "matmul_mid_remainder"));
    target_22bit.push((&circuit.matmul_x.r_matmul, "matmul_x_remainder"));

    for (r_matrix, label) in target_22bit {
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
            KBITS_M.trailing_zeros() as usize, // log2(M)
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(
            nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;
        all_results.extend(link_data);
        agg.add(&metrics);

        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    type Range16Lasso = RangeCheckSubtableStrategy<16>;
    let mut target_16bit_5d = Vec::new();
    let mut target_16bit_4d = Vec::new();
    // Softmax r_exp 수집 (5차원: [B][H][S][L][C])
    target_16bit_5d.push((&circuit.sm1.r_exp, "R_sm1_exp"));
    target_16bit_5d.push((&circuit.sm2.r_exp, "R_sm2_exp"));
    target_16bit_5d.push((&circuit.sm3.r_exp, "R_sm3_exp"));

    target_16bit_4d.push((&circuit.sm1.r_div, "softmax_k1_div_remainder"));
    target_16bit_4d.push((&circuit.sm2.r_div, "softmax_k2_div_remainder"));
    target_16bit_4d.push((&circuit.sm3.r_div, "softmax_k3_div_remainder"));

    for (r_exp_5d, label) in target_16bit_5d {
        let b_dim = r_exp_5d.len();
        let h_dim = r_exp_5d[0].len();
        let s_dim = r_exp_5d[0][0].len();
        let l_dim = r_exp_5d[0][0][0].len();
        let c_dim = r_exp_5d[0][0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim * c_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        // 5차원 순회: r_exp는 [B][H][S][L][C] 구조
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        for c in 0..c_dim {
                            nz.push([r_exp_5d[b][h][s][l][c].into_bigint().as_ref()[0] as usize]);
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
            FRACBITS_M.trailing_zeros() as usize, // log2(M)
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_16(
            nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;

        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
        all_results.push(link_data);
        agg.add(&metrics);
    }

    for (matrix_4d, label) in target_16bit_4d {
        let b_dim = matrix_4d.len();
        let h_dim = matrix_4d[0].len();
        let s_dim = matrix_4d[0][0].len();
        let l_dim = matrix_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..matrix_4d.len() {
            for h in 0..matrix_4d[b].len() {
                for s in 0..matrix_4d[b][h].len() {
                    for l in 0..matrix_4d[b][h][s].len() {
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
            FRACBITS_M.trailing_zeros() as usize, // log2(M)
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_16(
            nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;

        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
        all_results.push(link_data);
        agg.add(&metrics);
    }

    Ok((all_results, agg))
}