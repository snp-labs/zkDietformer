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
    circuits::mlp_circuit::NystromMLPCircuit,
    cc_snark::legogroth16::{LinkPublicGenerators, prepare_verifying_key, generate_random_parameters_incl_cp_link, create_random_proof_incl_cp_link, verify_proof_incl_cp_link},
    config::{RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M, GELU_M, GELU_C, KBITS_C, KBITS_M}, 
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu}, layernorm::constraints::{NormBundle, enforce_rmsnorm}, linear::constraints::{LinearBundle, enforce_linear}
    }, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        subtables::{
            SubtableStrategy, and::AndSubtableStrategy, 
            layernorm::RMSNormSubtableStrategy, 
            lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, 
            softmax::SoftmaxSubtableStrategy, gelu::GeLUSubtableStrategy
        }, utils::{math::Math, random::RandomTape},
    }, 
};

type G1 = G1Projective;

pub fn run_lasso_for_rmsnorm(
    bundle: &NormBundle<Fr>,
    k_bits: usize,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<()> {
    let b_dim = bundle.q_ms_sim.len();
    let s_dim = bundle.q_ms_sim[0].len();
    let dim = bundle.q_out[0][0].len();

    let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
    
    // Index = ((shifted_val * M_rms_idx) >> k) // dim
    let m_idx = bundle.m_rms_idx.into_bigint().as_ref()[0] as u128;

    for b in 0..b_dim {
        for s in 0..s_dim {
            let val = bundle.shifted_val[b][s][0].into_bigint().as_ref()[0] as u128;
            
            let numerator = val * m_idx;
            let scaled = numerator >> k_bits;
            let mut combined_idx = (scaled / (dim as u128)) as u64;

            if combined_idx >= RMSNORM_M as u64 {
                combined_idx = (RMSNORM_M - 1) as u64;
            }
            indices_x[b][s][0] = Fr::from(combined_idx);
        }
    }
    
    let total_sparsity = b_dim * s_dim;
    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        RMSNORM_C,
        total_sparsity,
        RMSNORM_C,
        RMSNORM_M.trailing_zeros() as usize,
    );

    let start = Instant::now();
    prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
    println!("  - Lasso Total Time (RMSNorm): {:?}", start.elapsed());
    
    Ok(())
}

pub fn prove_lasso_for_rmsnorm(
    indices: &Vec<Vec<Vec<Fr>>>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<()> {
    let b_dim = indices.len();
    let s_dim = indices[0].len();

    let mut nz_indices: Vec<[usize; RMSNORM_C]> = Vec::with_capacity(b_dim * s_dim);
    
    for b in 0..b_dim {
        for s in 0..s_dim {
            let val_bigint = indices[b][s][0].into_bigint();
            let limbs = val_bigint.as_ref(); 
            let idx_u64 = limbs[0]; 

            if idx_u64 >= RMSNORM_M as u64 {
                return Err(anyhow::anyhow!(
                    "Lasso Index Final Check Failed: index {} >= table size {} at [Batch: {}, Seq: {}]. \
                    This should have been clipped in run_lasso_for_rmsnorm.", 
                    idx_u64, RMSNORM_M, b, s
                ));
            }
            
            nz_indices.push([idx_u64 as usize]);
        }
    }

    let sparsity = nz_indices.len();
    if sparsity == 0 { return Ok(()); }

    let log_M = RMSNORM_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<RMSNORM_C>::new(nz_indices, log_M);
    let mut densified = DensifiedRepresentation::<Fr, RMSNORM_C>::from(&lookup_matrix);

    let commitment = densified.commit::<G1>(gens);
    let log_s = (sparsity.next_power_of_two().trailing_zeros()) as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape_rms");
    let mut transcript = Transcript::new(b"lasso_rmsnorm");
    
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, RMSNORM_C, RMSNORM_M, RMSNormSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"lasso_rmsnorm");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso verification failed for: {:?}", std::str::from_utf8(label)))?;

    println!("Lasso proof verified: {:?}", std::str::from_utf8(label).unwrap());
    Ok(())
}

pub fn run_lasso_for_gelu(
    bundle: &GeLUBundle<Fr>,
    q_in_sim: &Vec<Vec<Vec<Fr>>>, 
    r_in_lin: &Vec<Vec<Vec<Fr>>>, 
    k_bits: usize,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<()> {
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
    let gens = SparsePolyCommitmentGens::<G1>::new(label, GELU_C, sparsity, GELU_C, 16);
    prove_lasso_for_gelu(&indices, &gens, rng, label)
}

pub fn prove_lasso_for_gelu(
    nz_indices: &Vec<[usize; GELU_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<()> {
    let sparsity = nz_indices.len();
    
    let log_m = GELU_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<GELU_C>::new(nz_indices.clone(), log_m);
    let mut densified = DensifiedRepresentation::<Fr, GELU_C>::from(&lookup_matrix);

    let commitment = densified.commit::<G1>(gens);
    let r_point: Vec<Fr> = (0..sparsity.next_power_of_two().trailing_zeros() as usize)
        .map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape_gelu");
    let mut transcript = Transcript::new(b"lasso_gelu");
    
    let proof = SparsePolynomialEvaluationProof::<G1, GELU_C, GELU_M, GeLUSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut v_transcript = Transcript::new(b"lasso_gelu");
    proof.verify(&commitment, &r_point, gens, &mut v_transcript)
        .map_err(|_| anyhow::anyhow!("GeLU Lasso verification failed"))?;

    println!("GeLU Lasso proof verified: {:?}", std::str::from_utf8(label).unwrap());
    Ok(())
}


pub fn run_mlp_remainders<F: PrimeField, G: CurveGroup<ScalarField = F>>(
    circuit: &NystromMLPCircuit<F>,
    rng: &mut StdRng,
) -> Result<()> {    
    println!("\n=== [MLP] Lasso Lookup Proofs: 22-bit Remainders (R) ===");

    type Range22Lasso = RangeCheckSubtableStrategy<22>;

    let mut target_matrices = Vec::new();

    target_matrices.push((&circuit.norm1.r_shift, "R_mlp_norm1_shift"));
    target_matrices.push((&circuit.norm1.r_out, "R_mlp_norm1_out"));
    target_matrices.push((&circuit.norm2.r_shift, "R_mlp_norm2_shift"));
    target_matrices.push((&circuit.norm2.r_out, "R_mlp_norm2_out"));
    target_matrices.push((&circuit.gelu1.r_out_gelu, "R_mlp_gelu"));
    target_matrices.push((&circuit.ff1.r_out_lin, "R_mlp_lin1"));
    target_matrices.push((&circuit.ff2.r_out_lin, "R_mlp_lin2"));

    for (r_matrix, label) in target_matrices {
        let b_dim = r_matrix.len();
        let row_dim = r_matrix[0].len();
        let col_dim = r_matrix[0][0].len();
        let total_sparsity = b_dim * row_dim * col_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for r in 0..row_dim {
                for c in 0..col_dim {
                    let val_f = r_matrix[b][r][c];
                    let val_bytes = val_f.into_bigint().to_bytes_le();
                    let val_u64 = u64::from_le_bytes(val_bytes[..8].try_into().unwrap());
                    
                    let idx0 = (val_u64 & 0xFFFF) as usize;       
                    let idx1 = ((val_u64 >> 16) & 0x3F) as usize; 
                    
                    nz.push([idx0, idx1]);
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(),
            KBITS_C,
            total_sparsity,
            KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        prove_lasso_for_range(
            &nz,
            &gens,
            rng,
            label.as_bytes(),
        )?;
        
        println!(" - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    println!("All MLP remainders verified sequentially.");
    Ok(())
}

pub fn prove_lasso_for_range(
    nz_indices: &Vec<[usize; KBITS_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<()> {
    let sparsity = nz_indices.len();
    if sparsity == 0 { return Ok(()); }

    let log_m = KBITS_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<KBITS_C>::new(nz_indices.clone(), log_m);
    let mut densified = DensifiedRepresentation::<Fr, KBITS_C>::from(&lookup_matrix);

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
    Ok(())
}