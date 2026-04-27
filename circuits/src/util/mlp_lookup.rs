use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{fs, UniformRand, log2, marker::PhantomData, vec::Vec, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{select::CondSelectGadget,alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use merlin::Transcript;
use crate::{
    cc_snark::legogroth16::{LinkPublicGenerators, create_random_proof_incl_cp_link, generate_random_parameters_incl_cp_link, prepare_verifying_key, verify_proof_incl_cp_link}, circuits::freivalds_matmul::nystrom_mlp_circuit::NystromMLPCircuit, config::{GELU_C, GELU_M, KBITS_C, KBITS_M, RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M}, gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu}, layernorm::constraints::{NormBundle, enforce_rmsnorm}, linear::constraints::{LinearBundle, enforce_linear}
    }, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialCommitment, SparsePolynomialEvaluationProof},
        }, poly::{commitments::{Commitments, MultiCommitGens}, dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentBlinds}, eq_poly::EqPolynomial}, subtables::{
            SubtableStrategy, and::AndSubtableStrategy, gelu::GeLUSubtableStrategy, layernorm::RMSNormSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
        }, utils::{math::Math, random::RandomTape}
    } 
};

type G1 = G1Projective;

pub fn run_lasso_for_rmsnorm(
    bundle: &NormBundle<Fr>,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)> {
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
    let (pure_data_commitment, blinds, lasso_gens_n, sparsity) = prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
    println!("  - Lasso Total Time (RMSNorm): {:?}", start.elapsed());

    Ok((pure_data_commitment, blinds, lasso_gens_n.clone(), sparsity))
}


pub fn run_lasso_for_gelu(
    bundle: &GeLUBundle<Fr>,
    q_in_sim: &Vec<Vec<Vec<Fr>>>, 
    r_in_lin: &Vec<Vec<Vec<Fr>>>, 
    k_bits: usize,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)> {
    let b_dim = bundle.q_out_sim.len();
    let s_dim = bundle.q_out_sim[0].len();
    let d_dim = bundle.q_out_sim[0][0].len();

    let mut indices = Vec::new();
    let offset = 32768u128; // TABLE_SIZE // 2 for fixed-resolution 0-point
    let m_gelu_idx = bundle.m_gelu_idx.into_bigint().as_ref()[0] as u128;

    for b in 0..b_dim {
        for s in 0..s_dim {
            for d in 0..d_dim {
                // 1. Reconstruct high-precision X_diff: (Q_in - z_in) * 2^k + R_in
                let q_in_diff = (q_in_sim[b][s][d].into_bigint().as_ref()[0] as i128) 
                                - (bundle.z_in.into_bigint().as_ref()[0] as i128);
                let r_in = r_in_lin[b][s][d].into_bigint().as_ref()[0] as i128;
                
                let x_diff_full = (q_in_diff << k_bits) + r_in;

                // 2. Map to Fixed Table Index: Index = (X_diff * M_gelu_idx >> k) + Offset
                let scaled_diff = (x_diff_full * m_gelu_idx as i128) >> k_bits;
                let mut idx = scaled_diff + offset as i128;

                // 3. Clipping to TABLE_SIZE (65536)
                if idx < 0 { idx = 0; }
                if idx >= GELU_M as i128 { idx = (GELU_M - 1) as i128; }
                
                indices.push([idx as usize]);
            }
        }
    }

    let sparsity = indices.len();
    let gens = SparsePolyCommitmentGens::<G1>::new(label, GELU_C, sparsity.next_power_of_two(), GELU_C, 16);
    let (pure_data_commitment, blinds, lasso_gens_n, sparsity_out) = prove_lasso_for_gelu(indices, &gens, rng, label)?;
    Ok((pure_data_commitment, blinds, lasso_gens_n, sparsity_out))
}


pub fn run_mlp_remainders<F: PrimeField, G: CurveGroup<ScalarField = F>>(
    circuit: &NystromMLPCircuit<F>,
    rng: &mut StdRng,
) -> Result<Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>>  {
    println!("\n=== [MLP] Lasso Lookup Proofs: 22-bit Remainders (R) ===");
    let mut all_link_results = Vec::new();

    type Range22Lasso = RangeCheckSubtableStrategy<22>;

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
                    let idx1 = ((val_u64 >> 16) & 0x3F) as usize;
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
        let link_data = prove_lasso_for_range(
            nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;
        all_link_results.extend(link_data);
        
        println!(" - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
        
    }
    println!("All MLP remainders verified sequentially.");
    Ok(all_link_results)
    
}

pub fn prove_lasso_for_range(
    nz_indices: Vec<[usize; KBITS_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<Vec<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)>> {
    let sparsity = nz_indices.len();

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

    let commitment = densified.commit::<G1>(gens);

    let log_s = sparsity.next_power_of_two().trailing_zeros() as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape");
    let mut transcript = Transcript::new(b"lasso_range");
    
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, KBITS_C, KBITS_M, RangeCheckSubtableStrategy<22>>::prove(
        &mut densified, 
        &r_point, 
        gens, 
        &mut transcript, 
        &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"lasso_range");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso Range verification failed for label: {:?}", std::str::from_utf8(label)))?;

    println!("    -> Lasso proof verified for: {:?}", std::str::from_utf8(label).unwrap());
    // C=2이므로 dim[0]과 dim[1]을 각각 독립적인 CP-Link 블록으로 추출
    let mut results = Vec::new();
    let ell = densified.dim[0].get_num_vars();
    let (left_vars, right_vars) = EqPolynomial::<Fr>::compute_factored_lens(ell);
    let l_size = 1 << left_vars;   // 행 개수
    let r_size = 1 << right_vars;  // 열/베이스 개수 (보통 32)

    for i in 0..KBITS_C {
        // Lasso의 1024개용 베이스에서 현재 차원의 Factored 베이스(32개) 추출
        let (factored_gens, _) = gens.gens_combined_l_variate.gens.gens_n.split_at(r_size);
        
        // 해당 차원(dim[i])에 대한 blinds 생성 (random_tape 활용)
        let blinds = PolyCommitmentBlinds {
            blinds: random_tape.random_vector(b"poly_blinds", l_size),
        };

        // dim[i]에 대해 행렬 구조 커밋 생성
        let pure_cm = densified.dim[i].commit_inner(&blinds.blinds, &factored_gens);
        
        results.push((pure_cm, blinds, factored_gens, sparsity));
    }

    Ok(results)
}


pub fn prove_lasso_for_rmsnorm(
    indices: &Vec<Vec<Vec<Fr>>>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)> {
    let b_dim = indices.len();
    let s_dim = indices[0].len();

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

    // 2. 행렬 구조(L, R) 수동 결정 (데이터 512개 -> 16 x 32)
    let ell = densified.dim[0].get_num_vars(); // 512의 경우 9
    let (left_vars, right_vars) = EqPolynomial::<Fr>::compute_factored_lens(ell);
    let l_size = 1 << left_vars;   // 16
    let r_size = 1 << right_vars;  // 32

    // 3. Lasso의 제너레이터에서 현재 데이터 폭(R=32)에 맞는 부분만 추출
    // gens_combined_l_variate 내부의 gens_n은 결합 다항식용으로 더 클 수 있으므로 split_at으로 크기를 맞춥니다.
    let (factored_gens, _) = gens.gens_combined_l_variate.gens.gens_n.split_at(r_size);

    // 4. Lasso 기존 로직과 동일하게 RandomTape에서 16개의 Blinds 생성
    let blinds = PolyCommitmentBlinds {
        blinds: random_tape.random_vector(b"poly_blinds", l_size),
    };

    // 5. 수동으로 행렬 구조 커밋 생성 (이제 베이스 32 vs 데이터 32로 일치하여 패닉 방지)
    let pure_data_commitment = densified.dim[0].commit_inner(&blinds.blinds, &factored_gens);

    let commitment = densified.commit::<G1>(gens);
    let log_s = (sparsity.next_power_of_two().trailing_zeros()) as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let proof_lasso = SparsePolynomialEvaluationProof::<G1, RMSNORM_C, RMSNORM_M, RMSNormSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"lasso_rmsnorm");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso verification failed"))?;

    Ok((pure_data_commitment, blinds, factored_gens, sparsity))
}

pub fn prove_lasso_for_gelu(
    nz_indices: Vec<[usize; GELU_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<(PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize)> {
    let sparsity = nz_indices.len();

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

    // 데이터 크기에 따른 행렬 구조 계산
    let ell = densified.dim[0].get_num_vars();
    let (left_vars, right_vars) = EqPolynomial::<Fr>::compute_factored_lens(ell);
    let l_size = 1 << left_vars;
    let r_size = 1 << right_vars;

    // 제너레이터를 정확한 R 크기로 분할하여 차원 불일치 해결
    let (factored_gens, _) = gens.gens_combined_l_variate.gens.gens_n.split_at(r_size);

    // RandomTape에서 Blinds 수집
    let blinds = PolyCommitmentBlinds {
        blinds: random_tape.random_vector(b"poly_blinds", l_size),
    };

    // 행렬 구조로 커밋먼트 조각들(C_i) 생성
    let pure_data_commitment = densified.dim[0].commit_inner(&blinds.blinds, &factored_gens);

    let commitment = densified.commit::<G1>(gens);
    let r_point: Vec<Fr> = (0..sparsity.next_power_of_two().trailing_zeros() as usize)
        .map(|_| Fr::rand(rng)).collect();

    let proof = SparsePolynomialEvaluationProof::<G1, GELU_C, GELU_M, GeLUSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut v_transcript = Transcript::new(b"lasso_gelu");
    proof.verify(&commitment, &r_point, gens, &mut v_transcript)
        .map_err(|_| anyhow::anyhow!("GeLU Lasso verification failed"))?;

    println!("GeLU Lasso proof verified: {:?}", std::str::from_utf8(label).unwrap());

    Ok((pure_data_commitment, blinds, factored_gens, sparsity))
}