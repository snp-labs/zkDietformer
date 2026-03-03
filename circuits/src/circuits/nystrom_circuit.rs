use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix1, Ix4, Array2, Ix2, Ix3, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{rc::Rc, fs, UniformRand, log2, marker::PhantomData, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}, cell::RefCell};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{R1CSVar, alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use merlin::Transcript;
use ark_r1cs_std::select::CondSelectGadget;
use crate::{
    cc_snark::{legogroth16::{ProvingKey, create_random_proof, 
        verify_proof, VerifyingKey,LinkPublicGenerators, create_random_proof_incl_cp_link, generate_random_parameters_incl_cp_link, prepare_verifying_key, verify_proof_incl_cp_link}, 
    linker::{Linker,snark::LinkSnark,
        relation_generator::generate_cp_arithm_relation,relation_generator::{generate_cp_factored_instance, generate_cp_factored_relation, generate_cp_factored_witness}, snark::PP}},
    gadgets::{
        matmul::{constraints::{MatrixVar, new_matrix_witness}, 
    quantized::{MatmulBundle, MatmulData, PseudoinverseBundle, ZenParams, compute_circuit_landmarks, enforce_freivald, gen_challenge_vector, new_matrix_witness_4d, transpose_matrix}}, softmax::constraints::{SoftmaxBundle, enforce_softmax}}, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        }, poly::{commitments::{Commitments, MultiCommitGens}, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
        subtables::{
            SubtableStrategy, and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
        }, utils::{math::Math, random::RandomTape},
    }, util::{
        load::{read_array_i64, to_field_2d_i64, to_field_3d_u64, to_field_4d_i64, to_field_4d_u64, to_field_5d_i64, to_field_5d_u64}, lookup_attention::{prove_lasso_for_softmax, run_lasso_bundle, run_all_remainders_with_link}, utils::{cm1_from_vk, hash_cm_to_field}
    }
};

pub fn get_link_public_gens<R: RngCore, E: Pairing>(
    rng: &mut R,
    count: u32,
) -> LinkPublicGenerators<E> {
    let pedersen_gens = (0..count)
        .map(|_| E::G1::rand(rng).into_affine())
        .collect::<Vec<_>>();
    let g1 = E::G1::rand(rng).into_affine();
    let g2 = E::G2::rand(rng).into_affine();
    LinkPublicGenerators {
        pedersen_gens,
        g1,
        g2,
    }
}

type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromAttentionCircuit<F: PrimeField> {
    // 공통 입력 (한 번만 로드)
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,       
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,     
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,  
    pub z_q: F,      
    pub z_k: F, 
    pub z_v: F,

    pub matmul1: MatmulBundle<F>,
    pub matmul2: MatmulBundle<F>,
    pub matmul3: MatmulBundle<F>,


    pub sm1: SoftmaxBundle<F>,
    pub sm2: SoftmaxBundle<F>,
    pub sm3: SoftmaxBundle<F>,

    pub pseudo_inv: PseudoinverseBundle<F>,

    pub matmul_temp: MatmulBundle<F>,
    pub matmul_mid: MatmulBundle<F>,
    pub matmul_x: MatmulBundle<F>,

    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub r_challenge: F, 
    pub seg_size: usize,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,

    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        
        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        let mut r2_pool = Vec::new(); // FRAC_BITS remainder 5d
        let mut r3_pool = Vec::new(); // FRAC_BITS remainder 4d

        // 1. 공통 입력 및 위트니스 할당 ([B][H][S][D])
        let q_proc = new_matrix_witness_4d(cs.clone(), &self.q_q);
        let k_proc = new_matrix_witness_4d(cs.clone(), &self.q_k);
        let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &self.matmul1.r_mean);
        let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &self.matmul2.r_mean);

        // 2. Kernel 결과 데이터 할당
        let (k1_data, params_k1) = self.matmul1.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (k2_data, params_k2) = self.matmul2.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (k3_data, params_k3) = self.matmul3.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm1_vars = self.sm1.alloc_vars(cs.clone())?;
        let sm2_vars = self.sm2.alloc_vars(cs.clone())?;
        let sm3_vars = self.sm3.alloc_vars(cs.clone())?;

        // temp
        let v_proc = new_matrix_witness_4d(cs.clone(), &self.q_v); 
        let (temp_data, params_temp) = self.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);
        let (mid_data, params_mid) = self.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), self.pseudo_inv.z_inv, &two_k_var);
        let (x_data, params_x) = self.matmul_x.alloc_and_setup(cs.clone(), self.matmul_mid.z_sim, self.matmul_temp.z_sim, &two_k_var);
        
        let mut global_k1_lows = Vec::new(); let mut global_k1_highs = Vec::new();
        let mut global_k2_lows = Vec::new(); let mut global_k2_highs = Vec::new();
        let mut global_k3_lows = Vec::new(); let mut global_k3_highs = Vec::new();

        let mut global_sm1_idx = vec![Vec::new(); 5];
        let mut global_sm1_r_sm = vec![Vec::new(); 2];
        let mut global_sm2_idx = vec![Vec::new(); 5];
        let mut global_sm2_r_sm = vec![Vec::new(); 2];
        let mut global_sm3_idx = vec![Vec::new(); 5];
        let mut global_sm3_r_sm = vec![Vec::new(); 2];

        let mut global_inv0_lows = Vec::new(); let mut global_inv0_highs = Vec::new();
        let mut global_inv1_lows = Vec::new(); let mut global_inv1_highs = Vec::new();
        let mut global_inv2_lows = Vec::new(); let mut global_inv2_highs = Vec::new();
        let mut global_inv3_lows = Vec::new(); let mut global_inv3_highs = Vec::new();

        let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
        let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
        let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();

        for b in 0..q_proc.len() {
            // 해당 배치의 Landmarks 계산
            let k_landmarks = compute_circuit_landmarks(cs.clone(), &k_proc[b], &r_mean_k1[b], self.seg_size)?;
            let q_landmarks = compute_circuit_landmarks(cs.clone(), &q_proc[b], &r_mean_k2[b], self.seg_size)?;

            for h in 0..q_proc[b].len() {
                let k_l_t = transpose_matrix(&k_landmarks[h]);
                let r_pows_k1 = gen_challenge_vector(&r_var, k1_data.q_sim[b][h][0].len());
                
                // Kernel 1
                let mut h_k1_lows = Vec::new(); 
                let mut h_k1_highs = Vec::new();
                enforce_freivald(
                    cs.clone(),
                    &q_proc[b][h], 
                    &k_l_t, 
                    &k1_data.q_sim[b][h], 
                    &k1_data.r_matmul[b][h], 
                    &self.matmul1.r_matmul[b][h],
                    &params_k1, 
                    &r_pows_k1,
                    &mut h_k1_lows, 
                    &mut h_k1_highs,
                )?;
                global_k1_lows.extend(h_k1_lows); global_k1_highs.extend(h_k1_highs);

                
                // Kernel 2
                let mut h_k2_lows = Vec::new(); 
                let mut h_k2_highs = Vec::new();
                let q_l_h = &q_landmarks[h];
                let r_pows_k2: Vec<FpVar<F>> = gen_challenge_vector(&r_var, k2_data.q_sim[b][h][0].len());
                enforce_freivald(
                    cs.clone(),
                    q_l_h, 
                    &k_l_t, 
                    &k2_data.q_sim[b][h], 
                    &k2_data.r_matmul[b][h], 
                    &self.matmul2.r_matmul[b][h],
                    &params_k2, 
                    &r_pows_k2,
                    &mut h_k2_lows, 
                    &mut h_k2_highs,
                )?;
                global_k2_lows.extend(h_k2_lows); global_k2_highs.extend(h_k2_highs);
                

                // Kernel 3
                let mut h_k3_lows = Vec::new(); 
                let mut h_k3_highs = Vec::new();
                let k_full_t = transpose_matrix(&k_proc[b][h]);
                let r_pows_k3 = gen_challenge_vector(&r_var, k3_data.q_sim[b][h][0].len()); 
                enforce_freivald(
                    cs.clone(),
                    &q_landmarks[h], 
                    &k_full_t, 
                    &k3_data.q_sim[b][h], 
                    &k3_data.r_matmul[b][h], 
                    &self.matmul3.r_matmul[b][h],
                    &params_k3, 
                    &r_pows_k3,
                    &mut h_k3_lows, 
                    &mut h_k3_highs,
                )?;
                global_k3_lows.extend(h_k3_lows); global_k3_highs.extend(h_k3_highs);
                
                // Softmax 1
                let mut h_sm1_idx = [const { Vec::new() }; 5];
                let mut h_sm1_r_sm = [const { Vec::new() }; 2];
                enforce_softmax(
                    cs.clone(),
                    &k1_data.q_sim[b][h], 
                    &k1_data.r_matmul[b][h],
                    &sm1_vars.r_sm[b][h],        
                    &sm1_vars.m_sm,
                    &sm1_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm1_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm1_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm1_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    //&sm1_vars.exp_total[b][h],
                    &sm1_vars.p[b][h],
                    &sm1_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut h_sm1_idx,
                    &mut h_sm1_r_sm, 
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm1_idx[k].extend(h_sm1_idx[k].clone()); }
                for k in 0..2 { global_sm1_r_sm[k].extend(h_sm1_r_sm[k].clone()); }

                // Softmax 2
                let mut h_sm2_idx = [const { Vec::new() }; 5];
                let mut h_sm2_r_sm = [const { Vec::new() }; 2];
                enforce_softmax(
                    cs.clone(),
                    &k2_data.q_sim[b][h], 
                    &k2_data.r_matmul[b][h],
                    &sm2_vars.r_sm[b][h],        
                    &sm2_vars.m_sm,
                    &sm2_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm2_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm2_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm2_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm2_vars.p[b][h],
                    &sm2_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut h_sm2_idx,
                    &mut h_sm2_r_sm, 
                    &mut r2_pool, 
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm2_idx[k].extend(h_sm2_idx[k].clone()); }
                for k in 0..2 { global_sm2_r_sm[k].extend(h_sm2_r_sm[k].clone()); }

                // Softmax 3
                let zero_var = FpVar::zero();
                let num_landmarks = k3_data.q_sim[b][h].len();
                let seq_len = k3_data.q_sim[b][h][0].len();

                let mut masked_q_sim_k3 = k3_data.q_sim[b][h].clone();
                for l in 0..num_landmarks {
                    for s in 0..seq_len {
                        let is_not_padding = mask_var[b][s].is_one()?; 
                        
                        masked_q_sim_k3[l][s] = FpVar::conditionally_select(
                            &is_not_padding,
                            &k3_data.q_sim[b][h][l][s], 
                            &zero_var
                        )?;
                    }
                }

                let mut h_sm3_idx = [const { Vec::new() }; 5];
                let mut h_sm3_r_sm = [const { Vec::new() }; 2];
                enforce_softmax(
                    cs.clone(),
                    &masked_q_sim_k3, 
                    &k3_data.r_matmul[b][h],
                    &sm3_vars.r_sm[b][h],        
                    &sm3_vars.m_sm,
                    &sm3_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm3_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm3_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm3_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    //&sm3_vars.exp_total[b][h],
                    &sm3_vars.p[b][h],
                    &sm3_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut h_sm3_idx,
                    &mut h_sm3_r_sm, 
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm3_idx[k].extend(h_sm3_idx[k].clone()); }
                for k in 0..2 { global_sm3_r_sm[k].extend(h_sm3_r_sm[k].clone()); }

                // moore-penrose
                let q_inv_var = new_matrix_witness_4d(cs.clone(), &self.pseudo_inv.q_inv);
                let q_y_vars = self.pseudo_inv.q_y_intermediates.iter()
                    .map(|q_step| new_matrix_witness_4d(cs.clone(), q_step))
                    .collect::<Vec<_>>();
                let r_inv_vars = self.pseudo_inv.r_inv_remainders.iter()
                    .map(|r_step| new_matrix_witness_4d(cs.clone(), r_step))
                    .collect::<Vec<_>>();

                let z_zero = FpVar::zero();
                let z_inv = FpVar::Constant(self.pseudo_inv.z_inv);

                let mut inv_params = Vec::new();
                for i in 0..4 {
                    let (zw, zx, zy) = match i {
                        0 => (&z_zero, &z_inv, &z_zero), 
                        1 => (&z_zero, &z_zero, &z_zero), 
                        2 => (&z_inv, &z_zero, &z_inv),  
                        3 => (&z_inv, &z_inv, &z_inv),   
                        _ => unreachable!(),
                    };

                    inv_params.push(ZenParams {
                        m: FpVar::Constant(self.pseudo_inv.m_inv[i]),
                        m_prime: FpVar::Constant(self.pseudo_inv.m_prime_inv[i]),
                        z_w: zw.clone(),
                        z_x: zx.clone(),
                        z_y: zy.clone(),
                        two_k: two_k_var.clone(),
                    });
                }

                let r_pows_inv = gen_challenge_vector(&r_var, q_inv_var[b][h].len());

                // A * A_inv * A = A
                // Step 1: P_k2 * Q_inv_k2 = Q_y1
                let mut h_inv0_lows = Vec::new(); 
                let mut h_inv0_highs = Vec::new();
                enforce_freivald(
                    cs.clone(),
                    &sm2_vars.p[b][h], 
                    &q_inv_var[b][h], 
                    &q_y_vars[0][b][h], 
                    &r_inv_vars[0][b][h], 
                    &self.pseudo_inv.r_inv_remainders[0][b][h],
                    &inv_params[0], 
                    &r_pows_inv,
                    &mut h_inv0_lows, 
                    &mut h_inv0_highs,
                )?;
                global_inv0_lows.extend(h_inv0_lows); global_inv0_highs.extend(h_inv0_highs);

                // Step 2: Q_y1 * P_k2 = P_k2
                let mut h_inv1_lows = Vec::new(); 
                let mut h_inv1_highs = Vec::new();
                enforce_freivald(
                    cs.clone(),
                    &q_y_vars[0][b][h], 
                    &sm2_vars.p[b][h], 
                    &q_y_vars[1][b][h],
                    &r_inv_vars[1][b][h], 
                    &self.pseudo_inv.r_inv_remainders[1][b][h],
                    &inv_params[1], 
                    &r_pows_inv,
                    &mut h_inv1_lows, 
                    &mut h_inv1_highs,
                )?;
                global_inv1_lows.extend(h_inv1_lows); global_inv1_highs.extend(h_inv1_highs);
                
                // A_inv * A * A_inv = A_inv
                // Step 2-1: Q_inv (A+) * P_k2 (A) = Q_y3
                let mut h_inv2_lows = Vec::new(); 
                let mut h_inv2_highs = Vec::new();
                enforce_freivald(
                    cs.clone(),
                    &q_inv_var[b][h],             
                    &sm2_vars.p[b][h], 
                    &q_y_vars[2][b][h],             
                    &r_inv_vars[2][b][h],  
                    &self.pseudo_inv.r_inv_remainders[2][b][h],      
                    &inv_params[2],              
                    &r_pows_inv,
                    &mut h_inv2_lows, 
                    &mut h_inv2_highs,
                )?;
                global_inv2_lows.extend(h_inv2_lows); global_inv2_highs.extend(h_inv2_highs);

                // Step 2-2: Q_y3 * Q_inv (A+) = Q_y4
                let mut h_inv3_lows = Vec::new(); 
                let mut h_inv3_highs = Vec::new();
                enforce_freivald(
                    cs.clone(),
                    &q_y_vars[2][b][h],            
                    &q_inv_var[b][h],
                    &q_y_vars[3][b][h],             
                    &r_inv_vars[3][b][h],   
                    &self.pseudo_inv.r_inv_remainders[3][b][h],    
                    &inv_params[3],             
                    &r_pows_inv,
                    &mut h_inv3_lows, 
                    &mut h_inv3_highs,
                )?;
                global_inv3_lows.extend(h_inv3_lows); global_inv3_highs.extend(h_inv3_highs);


                // temp
                let mut h_temp_lows = Vec::new(); 
                let mut h_temp_highs = Vec::new();
                let r_pows_temp = gen_challenge_vector(&r_var, temp_data.q_sim[b][h][0].len());
                enforce_freivald(
                    cs.clone(),
                    &sm3_vars.p[b][h],            
                    &v_proc[b][h],                
                    &temp_data.q_sim[b][h],       
                    &temp_data.r_matmul[b][h],    
                    &self.matmul_temp.r_matmul[b][h],
                    &params_temp, 
                    &r_pows_temp,
                    &mut h_temp_lows, 
                    &mut h_temp_highs,                
                )?;
                global_temp_lows.extend(h_temp_lows); global_temp_highs.extend(h_temp_highs);

                // mid
                let mut h_mid_lows = Vec::new(); 
                let mut h_mid_highs = Vec::new();
                let r_pows_mid = gen_challenge_vector(&r_var, mid_data.q_sim[b][h][0].len()); // 길이 L
                enforce_freivald(
                    cs.clone(),
                    &sm1_vars.p[b][h],        
                    &q_inv_var[b][h],         
                    &mid_data.q_sim[b][h],    
                    &mid_data.r_matmul[b][h], 
                    &self.matmul_mid.r_matmul[b][h],
                    &params_mid, 
                    &r_pows_mid,
                    &mut h_mid_lows, 
                    &mut h_mid_highs,
                )?;
                global_mid_lows.extend(h_mid_lows); global_mid_highs.extend(h_mid_highs);

                // X
                let mut h_x_lows = Vec::new(); 
                let mut h_x_highs = Vec::new();
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len()); // 길이 D
                enforce_freivald(
                    cs.clone(),
                    &mid_data.q_sim[b][h],    
                    &temp_data.q_sim[b][h],   
                    &x_data.q_sim[b][h],      
                    &x_data.r_matmul[b][h],  
                    &self.matmul_x.r_matmul[b][h],
                    &params_x, 
                    &r_pows_x,
                    &mut h_x_lows, 
                    &mut h_x_highs,
                )?;
                global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);
            }
        }

        {
            let mut final_vars = self.committed_vars.borrow_mut();
        
            // 1. Softmax 인덱스 블록 (C=5)
            for seg in 0..5 { final_vars.extend(global_sm1_idx[seg].clone()); }
            for seg in 0..5 { final_vars.extend(global_sm2_idx[seg].clone()); }
            for seg in 0..5 { final_vars.extend(global_sm3_idx[seg].clone()); }
        
            // MatMul Kernels
            final_vars.extend(global_k1_lows); final_vars.extend(global_k1_highs);
            final_vars.extend(global_k2_lows); final_vars.extend(global_k2_highs);
            final_vars.extend(global_k3_lows); final_vars.extend(global_k3_highs);
            
            // Softmax R_sm
            for k in 0..2 { final_vars.extend(global_sm1_r_sm[k].clone()); }
            for k in 0..2 { final_vars.extend(global_sm2_r_sm[k].clone()); }
            for k in 0..2 { final_vars.extend(global_sm3_r_sm[k].clone()); }

            // Moore-Penrose Steps
            final_vars.extend(global_inv0_lows); final_vars.extend(global_inv0_highs);
            final_vars.extend(global_inv1_lows); final_vars.extend(global_inv1_highs);
            final_vars.extend(global_inv2_lows); final_vars.extend(global_inv2_highs);
            final_vars.extend(global_inv3_lows); final_vars.extend(global_inv3_highs);

            // Final MatMuls
            final_vars.extend(global_temp_lows); final_vars.extend(global_temp_highs);
            final_vars.extend(global_mid_lows); final_vars.extend(global_mid_highs);
            final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);

            // --- 16-bit Remainders (C=1 Blocks) ---
            final_vars.extend(r2_pool); 
            final_vars.extend(r3_pool);
        }

        Ok(())
    }
}

pub fn load_nystrom_circuit<E: Pairing>(path: &Path) -> Result<NystromAttentionCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let r = E::ScalarField::zero(); // 혹은 결정론적 해시 결과값
    
    let q_q = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?.into_dimensionality()?;
    let q_k = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?.into_dimensionality()?;
    let z_q = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar();
    let z_k = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar();
    let seg= npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("seg_size.npy")?.into_scalar() as usize;

    let load_matmul = |npz: &mut NpzReader<File>, suffix: &str, has_rmean: bool| -> Result<MatmulBundle<E::ScalarField>> {
        Ok(MatmulBundle {
            // 커널 3처럼 r_mean이 없는 경우 빈 벡터 할당
            r_mean: if has_rmean {
                to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_mean_{}.npy", suffix))?.into_dimensionality()?)
            } else {
                vec![] 
            },
            q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_sim_{}.npy", suffix))?.into_dimensionality()?),
            r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_matmul_{}.npy", suffix))?.into_dimensionality()?),
            m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", suffix))?.into_scalar()),
            m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", suffix))?.into_scalar()),
            z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_sim_{}.npy", suffix))?.into_scalar()),
        })
    };

    let matmul1 = load_matmul(&mut npz, "k1", true)?;
    let matmul2 = load_matmul(&mut npz, "k2", true)?;
    let matmul3 = load_matmul(&mut npz, "k3", false)?;
    
    let load_softmax = |npz: &mut NpzReader<File>, suffix: &str| -> Result<SoftmaxBundle<E::ScalarField>> {
        Ok(SoftmaxBundle {
            limbs: to_field_5d_i64(&npz.by_name::<_, IxDyn>(&format!("limbs_{}.npy", suffix))?),
            exp_limbs: to_field_5d_u64(&npz.by_name::<_, IxDyn>(&format!("exp_limbs_{}.npy", suffix))?),
            q_exp: to_field_5d_u64(&npz.by_name::<_, IxDyn>(&format!("Q_exp_{}.npy", suffix))?),
            r_exp: to_field_5d_u64(&npz.by_name::<_, IxDyn>(&format!("R_exp_{}.npy", suffix))?),
            exp_total: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("exp_total_{}.npy", suffix))?.into_dimensionality()?),
            p: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("P_{}.npy", suffix))?.into_dimensionality()?),
            r_div: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("R_div_{}.npy", suffix))?.into_dimensionality()?),
            r_sm: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("r_sm_{}.npy", suffix))?.into_dimensionality()?),
            m_sm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar()), // 또는 커널별 suffix 적용
        })
    };

    let sm1 = load_softmax(&mut npz, "k1")?;
    let sm2 = load_softmax(&mut npz, "k2")?;
    let sm3 = load_softmax(&mut npz, "k3")?;

    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let limb_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("limb_bits.npy")?.into_scalar() as usize;
    let k_segments = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_segments.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    // Pseudoinverse 관련 데이터 로드
    let q_inv_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_inv_k2.npy")?.into_dimensionality()?;
    let z_inv_k2 = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_inv_k2.npy")?.into_scalar();

    // M_inv_all (4개), M_prime_inv_all (4개)
    let m_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_inv_all.npy")?;
    let m_prime_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_prime_inv_all.npy")?;
    let q_y_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("Q_y_inv_intermediates.npy")?;
    let r_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("R_inv_remainders.npy")?;

    let pseudo_inv = PseudoinverseBundle {
        q_inv: to_field_4d_i64(&q_inv_k2),
        z_inv: E::ScalarField::from(z_inv_k2),
        m_inv: [
            E::ScalarField::from(m_inv_all[0]),
            E::ScalarField::from(m_inv_all[1]),
            E::ScalarField::from(m_inv_all[2]),
            E::ScalarField::from(m_inv_all[3]),
        ],
        m_prime_inv: [
            E::ScalarField::from(m_prime_inv_all[0]),
            E::ScalarField::from(m_prime_inv_all[1]),
            E::ScalarField::from(m_prime_inv_all[2]),
            E::ScalarField::from(m_prime_inv_all[3]),
        ],
        q_y_intermediates: to_field_5d_i64(&q_y_inv_raw),
        r_inv_remainders: to_field_5d_i64(&r_inv_raw),
    };

    let q_v = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_v.npy")?.into_dimensionality()?;
    let z_v = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar();
    let matmul_temp = load_matmul(&mut npz, "temp", false)?;
    let matmul_mid = load_matmul(&mut npz, "mid", false)?; // P_k1 @ Q_inv_k2
    let matmul_x = load_matmul(&mut npz, "x", false)?;     // mid @ temp

    Ok(NystromAttentionCircuit {
        q_q: to_field_4d_i64(&q_q),
        q_k: to_field_4d_i64(&q_k),
        q_v: to_field_4d_i64(&q_v),
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),
        z_v: E::ScalarField::from(z_v),

        matmul1: matmul1,
        matmul2: matmul2,
        matmul3: matmul3,

        sm1: sm1,
        sm2: sm2,
        sm3: sm3,

        pseudo_inv,
        matmul_temp,
        matmul_mid,
        matmul_x,

        frac_bits: frac_bits,
        limb_bits: limb_bits,
        k_segments: k_segments,
        r_challenge: r,
        seg_size: seg,
        k_bits: k_bits,
        mask: to_field_2d_i64(&mask_raw),

        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })

    
}

pub fn prove_and_verify_factored_multi_link<E: Pairing>(
    vk: &VerifyingKey<E>,
    num_pub: usize,
    v_groth: E::ScalarField,
    cm_groth: E::G1Affine,
    lasso_results: Vec<(PolyCommitment<E::G1>, PolyCommitmentBlinds<E::ScalarField>, MultiCommitGens<E::G1>)>,
    u_total: Vec<E::ScalarField>,
    rng: &mut impl Rng,
) -> Result<()> {
    let mut l_rows_vec = Vec::new();
    let mut r_cols_vec = Vec::new();
    let mut cks_lasso_vec = Vec::new();
    let mut rs = vec![v_groth];
    let mut lasso_cms_all = Vec::new();

    for (cm, blinds, gens) in &lasso_results {
        l_rows_vec.push(cm.C.len()); 
        r_cols_vec.push(gens.n);    
        
        let mut ck = vec![gens.h.into_affine()];
        ck.extend(gens.G.iter().map(|g| g.into_affine()));
        cks_lasso_vec.push(ck);
        
        rs.extend(&blinds.blinds); 
        lasso_cms_all.extend(cm.C.iter().map(|c| c.into_affine()));
    }

    let mut ck_joint = vec![vk.eta1_gamma_inv_g1];
    let start = 1 + num_pub;
    ck_joint.extend(vk.gamma_abc_g1[start..start + u_total.len()].iter().cloned());

    let crs = generate_cp_factored_relation::<E>(
        lasso_results.len(), l_rows_vec, r_cols_vec, ck_joint, cks_lasso_vec
    );

    let pp = PP::new(crs.nr, crs.nc, &E::G1::rand(rng).into_affine(), &E::G2::rand(rng).into_affine());
    let (ek, vk_link) = LinkSnark::<E>::keygen(&pp, crs, rng);

    let witness = generate_cp_factored_witness::<E>(rs, u_total);
    let instance = generate_cp_factored_instance::<E>(cm_groth, lasso_cms_all);

    let (proof, _) = LinkSnark::<E>::prove(&pp, &ek, witness, rng);
    if LinkSnark::<E>::verify(&pp, &vk_link, &instance, &proof) {
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Factored link verification failed"))
    }
}

pub fn run_paths<E: Pairing>(path: &Path) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    type ScalarF<E> = <E as Pairing>::ScalarField;
    let mut targets: Vec<std::path::PathBuf> = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path).context("reading directory")? {
            let ent = entry?.path();
            if ent.is_file() && ent.extension().map_or(false, |s| s == "npz") {
                targets.push(ent);
            }
        }
        targets.sort();
    } else {
        targets.push(path.to_path_buf());
    }

    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets.iter() {
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);
    
        let num_commit_1 = {
            let temp_circuit = load_nystrom_circuit::<E>(&file_path)?;

            let vars_ptr = Rc::clone(&temp_circuit.committed_vars);

            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;

            vars_ptr.borrow().len() as u32
        };

        
        println!("최종 측정된 커밋 개수: {:?}", num_commit_1);

        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        // LegoGroth16 Setup
        let link_gens = get_link_public_gens::<_, E>(&mut rng, num_commit_1 + 1);
        println!("=== [0] LegoGroth16 Setup (Num Commit: {:?}) ===", num_commit_1);
        
        let params = generate_random_parameters_incl_cp_link::<E, _, _>(
            sample_circuit.clone(), 
            link_gens, 
            num_commit_1, 
            0, 
            &mut rng
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        
        let pvk = prepare_verifying_key::<E>(&params.vk.groth16_vk);

        // Witness 값 추출을 위해 sample_circuit을 한 번 실행 (Synthesize)
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> = vars.iter()
                .map(|v| v.value().unwrap_or(Fr::zero()))
                .collect();
            
            if values.len() != num_commit_1 as usize {
                return Err(anyhow::anyhow!(
                    "Commitment size mismatch: expected {}, found {}", 
                    num_commit_1, 
                    values.len()
                ));
            }
            values
        };

        println!("\n[Debug] LegoGroth16 committed_group1 (Total: {})", committed_group1.len());
        let print_count = std::cmp::min(10, committed_group1.len());
        if committed_group1.len() > 0 {
            println!("  Head (10): {:?}", committed_group1[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  Tail (10): {:?}", committed_group1[committed_group1.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let link_v = Fr::rand(&mut rng);
        
        // 서킷의 공개 입력 개수
        let num_pub = cs_witness.num_instance_variables() - 1; 

        let cm1 = cm1_from_vk::<E>(&params.vk.groth16_vk, num_pub, &committed_group1, &v1);
        let r_fs = hash_cm_to_field::<E>(&cm1);

        let groth16_pk = ProvingKey {
            vk: params.vk.groth16_vk.clone(),
            common: params.common.clone(),
        };

        println!("=== [1] LegoGroth16 GenProof ===");

        let final_circuit = NystromAttentionCircuit {
            r_challenge: r_fs,
            ..sample_circuit.clone()
        };

        // 1. 증명 생성: CP-Link 없이 LegoGroth16 SNARK 본체만 생성
        // v1, v2는 위트니스 그룹 1, 2에 대한 blinding factor입니다.
        /*let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(), 
            v1, 
            v2, 
            &groth16_pk, 
            &mut rng
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;

        // 2. 증명 검증: CP-Link 검증을 제외하고 SNARK 제약 조건 및 커밋먼트 정당성만 검증
        let public_inputs = vec![final_circuit.r_challenge]; 
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);

        println!("  - LegoGroth16 SNARK Verification: {:?}", result);

        if let Err(e) = result {
            println!("    Reason: {:?}", e);
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }*/

        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        let lasso_sm1 = run_lasso_bundle(&final_circuit.sm1, b"lasso_k1", &mut rng)?;
        let lasso_sm2 = run_lasso_bundle(&final_circuit.sm2, b"lasso_k2", &mut rng)?;
        let lasso_sm3 = run_lasso_bundle(&final_circuit.sm3, b"lasso_k3", &mut rng)?;
        let lasso_remainders = run_all_remainders_with_link::<Fr, G1>(&final_circuit, &mut rng)?;

        let mut lasso_results = vec![];
        lasso_results.extend(lasso_sm1);        // SM1 (5개 블록)
        lasso_results.extend(lasso_sm2);        // SM2 (5개 블록)
        lasso_results.extend(lasso_sm3);
        lasso_results.extend(lasso_remainders);

        prove_and_verify_factored_multi_link::<E>(
            &params.vk.groth16_vk,
            num_pub,
            v1,
            cm1,
            lasso_results,
            committed_group1, // u_values
            &mut rng,
        )?;
    }
    Ok(())
}

#[test]
fn attention_test() {
    let p = Path::new("outputs/outputs-model.transformer_0.mha.attn.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            panic!("proof generate failed: {}", e);
        }
    }
}