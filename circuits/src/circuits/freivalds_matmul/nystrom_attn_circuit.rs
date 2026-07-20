use std::{fs::File, ops::{AddAssign, Mul}, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{IxDyn, Ix0, Ix2, Ix3, Ix4};
use ndarray_npy::NpzReader;
use ark_std::{rc::Rc, fs, UniformRand, marker::PhantomData, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}, cell::RefCell};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{R1CSVar, alloc::AllocVar, eq::EqGadget, fields::{fp::FpVar, FieldVar}};
use ark_r1cs_std::select::CondSelectGadget;
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        proof_size,
    },
    gadgets::{
        matmul::{
            constraints::{MatrixVar, new_matrix_witness},
            quantized::{MatmulBundle, MatmulData, ProjectionBundle, ScalingBundle, PseudoinverseBundle, ZenParams, compute_circuit_landmarks, enforce_freivalds, enforce_scaling, gen_challenge_vector, new_matrix_witness_4d, transpose_matrix},
        },
        softmax::constraints::{SoftmaxBundle, SoftmaxVarBundle, enforce_softmax},
    },
    lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
        subtables::range_check::RangeCheckSubtableStrategy,
        utils::random::RandomTape,
    },
    util::{
        load::{read_array_i64, to_field_2d_i64, to_field_3d_i64, to_field_3d_u64, to_field_4d_i64, to_field_4d_u64, to_field_5d_i64},
        nystrom_attn_lookup::{prove_lasso_for_softmax, run_lasso_bundle, run_all_remainders_with_link},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;

type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromAttentionCircuit<F: PrimeField> {
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,

    // === Projection inputs ===
    pub q_x_input: Vec<Vec<Vec<F>>>,      // [B][S][H*D] quantized hidden_states
    pub z_x_input: F,

    // === Q/K/V Projection ===
    pub proj_q: ProjectionBundle<F>,
    pub proj_k: ProjectionBundle<F>,
    pub proj_v: ProjectionBundle<F>,

    // === Q/K Scaling ===
    pub scale_q: ScalingBundle<F>,
    pub scale_k: ScalingBundle<F>,

    // === Output Projection ===
    pub z_attn_output: F,
    pub q_attn_output: Vec<Vec<Vec<F>>>,  // [B][S][HD] — O-proj input (post-conv if conv exists)
    pub proj_o: ProjectionBundle<F>,

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
    pub r_challenge: F,
    pub seg_size: usize,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,

    /// Poseidon hash of flatten(q_x_input) — native precomputed, declared as public input
    pub input_hash: F,
    /// Poseidon hash of flatten(proj_o_out) — native precomputed, declared as public input
    pub output_hash: F,

    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    /// d2 group: W_Q, W_K, W_V weights — committed via v2 but NOT CP-linked to Lasso.
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        // v3: r2_pool removed (no exp chain in v3)
        // r_div: 26-bit split (low 16 + high 10) — R_div can exceed 2^16
        let mut r_div_k1_lows = Vec::new();
        let mut r_div_k1_highs = Vec::new();
        let mut r_div_k2_lows = Vec::new();
        let mut r_div_k2_highs = Vec::new();
        let mut r_div_k3_lows = Vec::new();
        let mut r_div_k3_highs = Vec::new();

        // 1. 공통 입력 할당
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

        // === Projection input allocation (witness) ===
        // X_input [B][S][HD] — common input for Q/K/V projections
        let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = self.q_x_input.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        // O-projection input [B][S][HD] — may differ from x_data.q_sim when conv exists
        let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = self.q_attn_output.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        // Projection output witnesses [B][H][S][D]
        let proj_q_out = new_matrix_witness_4d(cs.clone(), &self.proj_q.q_out_proj);
        let proj_q_r = new_matrix_witness_4d(cs.clone(), &self.proj_q.r_out_proj);
        let proj_k_out = new_matrix_witness_4d(cs.clone(), &self.proj_k.q_out_proj);
        let proj_k_r = new_matrix_witness_4d(cs.clone(), &self.proj_k.r_out_proj);
        let proj_v_out = new_matrix_witness_4d(cs.clone(), &self.proj_v.q_out_proj);
        let proj_v_r = new_matrix_witness_4d(cs.clone(), &self.proj_v.r_out_proj);

        // Scaling witnesses [B][H][S][D] (Q/K only)
        // scale_q_out / scale_k_out are used directly as kernel chain inputs (= q_q / q_k)
        let scale_q_out = new_matrix_witness_4d(cs.clone(), &self.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &self.scale_q.r_scale);
        let scale_k_out = new_matrix_witness_4d(cs.clone(), &self.scale_k.q_scaled);
        let scale_k_r = new_matrix_witness_4d(cs.clone(), &self.scale_k.r_scale);

        // Allocate W_Q, W_K, W_V weights as witnesses (transposed form: [H][HD][D])
        let num_heads = self.proj_q.q_w_diff.len();
        let w_q_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = self.proj_q.q_w_diff[h].len();
            let hd_dim = self.proj_q.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(self.proj_q.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();
        let w_k_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = self.proj_k.q_w_diff[h].len();
            let hd_dim = self.proj_k.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(self.proj_k.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();
        let w_v_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = self.proj_v.q_w_diff[h].len();
            let hd_dim = self.proj_v.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(self.proj_v.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();

        let w_o_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = self.proj_o.q_w_diff[h].len();
            let hd_dim = self.proj_o.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(self.proj_o.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();

        // d2: commit W_Q, W_K, W_V, W_O weights
        {
            let mut vars_2 = self.committed_vars_2.borrow_mut();
            for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_k_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        }

        let proj_o_out = new_matrix_witness_4d(cs.clone(), &self.proj_o.q_out_proj);
        let proj_o_r = new_matrix_witness_4d(cs.clone(), &self.proj_o.r_out_proj);

        // Projection remainder accumulators
        let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
        let mut global_proj_k_lows = Vec::new(); let mut global_proj_k_highs = Vec::new();
        let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
        let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

        let (temp_data, params_temp) = self.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);
        let (mid_data, params_mid) = self.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), self.pseudo_inv.z_inv, &two_k_var);
        let (x_data, params_x) = self.matmul_x.alloc_and_setup(cs.clone(), self.matmul_mid.z_sim, self.matmul_temp.z_sim, &two_k_var);

        let mut global_k1_lows = Vec::new(); let mut global_k1_highs = Vec::new();
        let mut global_k2_lows = Vec::new(); let mut global_k2_highs = Vec::new();
        let mut global_k3_lows = Vec::new(); let mut global_k3_highs = Vec::new();

        // v3: single pool (C=1) per softmax
        let mut global_sm1_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_sm1_r_sm: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut global_sm2_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_sm2_r_sm: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut global_sm3_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_sm3_r_sm: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        let mut global_inv0_lows = Vec::new(); let mut global_inv0_highs = Vec::new();
        let mut global_inv1_lows = Vec::new(); let mut global_inv1_highs = Vec::new();
        let mut global_inv2_lows = Vec::new(); let mut global_inv2_highs = Vec::new();
        let mut global_inv3_lows = Vec::new(); let mut global_inv3_highs = Vec::new();

        let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
        let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
        let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();

        for b in 0..scale_q_out.len() {
            // 해당 배치의 Landmarks 계산
            let _nc_pre_lm = cs.num_constraints();
            let k_landmarks = compute_circuit_landmarks(cs.clone(), &scale_k_out[b], &r_mean_k1[b], self.seg_size)?;
            let q_landmarks = compute_circuit_landmarks(cs.clone(), &scale_q_out[b], &r_mean_k2[b], self.seg_size)?;
            eprintln!("[CCOUNT] b={} after landmarks: {} (delta={})", b, cs.num_constraints(), cs.num_constraints() - _nc_pre_lm);

            for h in 0..scale_q_out[b].len() {
                let _nc_h_start = cs.num_constraints();
                // === Q/K/V Projection + Q/K Scaling constraints ===

                // Q Projection: Q_raw_q = X_input @ W_q_h^T + b_q_h
                {
                    let d_dim = self.proj_q.q_w_diff[h].len();
                    let proj_q_params = ZenParams {
                        m: FpVar::Constant(self.proj_q.m_proj),
                        m_prime: FpVar::Constant(self.proj_q.m_prime_proj),
                        z_w: FpVar::Constant(self.z_x_input),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    let r_pows_pq = gen_challenge_vector(&r_var, d_dim);
                    enforce_freivalds(
                        cs.clone(), &x_input_vars[b], &w_q_vars[h],
                        &proj_q_out[b][h], &proj_q_r[b][h],
                        &self.proj_q.r_out_proj[b][h],
                        &proj_q_params, &r_pows_pq,
                        &mut global_proj_q_lows, &mut global_proj_q_highs,
                        Some(&self.proj_q.q_b_diff[h]),
                    )?;
                }

                // K Projection: K_raw_k = X_input @ W_k_h^T + b_k_h
                {
                    let d_dim = self.proj_k.q_w_diff[h].len();
                    let proj_k_params = ZenParams {
                        m: FpVar::Constant(self.proj_k.m_proj),
                        m_prime: FpVar::Constant(self.proj_k.m_prime_proj),
                        z_w: FpVar::Constant(self.z_x_input),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    let r_pows_pk = gen_challenge_vector(&r_var, d_dim);
                    enforce_freivalds(
                        cs.clone(), &x_input_vars[b], &w_k_vars[h],
                        &proj_k_out[b][h], &proj_k_r[b][h],
                        &self.proj_k.r_out_proj[b][h],
                        &proj_k_params, &r_pows_pk,
                        &mut global_proj_k_lows, &mut global_proj_k_highs,
                        Some(&self.proj_k.q_b_diff[h]),
                    )?;
                }

                // V Projection: V_raw = X_input @ W_v_h^T + b_v_h
                {
                    let d_dim = self.proj_v.q_w_diff[h].len();
                    let proj_v_params = ZenParams {
                        m: FpVar::Constant(self.proj_v.m_proj),
                        m_prime: FpVar::Constant(self.proj_v.m_prime_proj),
                        z_w: FpVar::Constant(self.z_x_input),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    let r_pows_pv = gen_challenge_vector(&r_var, d_dim);
                    enforce_freivalds(
                        cs.clone(), &x_input_vars[b], &w_v_vars[h],
                        &proj_v_out[b][h], &proj_v_r[b][h],
                        &self.proj_v.r_out_proj[b][h],
                        &proj_v_params, &r_pows_pv,
                        &mut global_proj_v_lows, &mut global_proj_v_highs,
                        Some(&self.proj_v.q_b_diff[h]),
                    )?;
                }

                // Q Scaling: Q_proc = Q_raw_q / scale
                {
                    let z_raw_q = FpVar::Constant(self.proj_q.z_raw);
                    let m_scale_q = FpVar::Constant(self.scale_q.m_scale);
                    let m_prime_q = FpVar::Constant(self.scale_q.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &proj_q_out[b][h], &scale_q_out[b][h], &scale_q_r[b][h],
                        &self.scale_q.r_scale[b][h],
                        &z_raw_q, &m_scale_q, &m_prime_q, &two_k_var,
                        &mut global_scale_q_lows, &mut global_scale_q_highs,
                    )?;
                }

                // K Scaling: K_proc = K_raw_k / scale
                {
                    let z_raw_k = FpVar::Constant(self.proj_k.z_raw);
                    let m_scale_k = FpVar::Constant(self.scale_k.m_scale);
                    let m_prime_k = FpVar::Constant(self.scale_k.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &proj_k_out[b][h], &scale_k_out[b][h], &scale_k_r[b][h],
                        &self.scale_k.r_scale[b][h],
                        &z_raw_k, &m_scale_k, &m_prime_k, &two_k_var,
                        &mut global_scale_k_lows, &mut global_scale_k_highs,
                    )?;
                }

                eprintln!("[CCOUNT] b={} h={} after proj+scale: {} (delta={})", b, h, cs.num_constraints(), cs.num_constraints() - _nc_h_start);

                let k_l_t = transpose_matrix(&k_landmarks[h]);
                let r_pows_k1 = gen_challenge_vector(&r_var, k1_data.q_sim[b][h][0].len());

                // Kernel 1
                let mut h_k1_lows = Vec::new(); 
                let mut h_k1_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &scale_q_out[b][h],
                    &k_l_t,
                    &k1_data.q_sim[b][h],
                    &k1_data.r_matmul[b][h],
                    &self.matmul1.r_matmul[b][h],
                    &params_k1,
                    &r_pows_k1,
                    &mut h_k1_lows,
                    &mut h_k1_highs,
                    None,
                )?;
                global_k1_lows.extend(h_k1_lows); global_k1_highs.extend(h_k1_highs);
                eprintln!("[CCOUNT] b={} h={} after K1: {}", b, h, cs.num_constraints());

                // Kernel 2
                let mut h_k2_lows = Vec::new(); 
                let mut h_k2_highs = Vec::new();
                let q_l_h = &q_landmarks[h];
                let r_pows_k2: Vec<FpVar<F>> = gen_challenge_vector(&r_var, k2_data.q_sim[b][h][0].len());
                enforce_freivalds(
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
                    None,
                )?;
                global_k2_lows.extend(h_k2_lows); global_k2_highs.extend(h_k2_highs);
                

                // Kernel 3
                let mut h_k3_lows = Vec::new(); 
                let mut h_k3_highs = Vec::new();
                let k_full_t = transpose_matrix(&scale_k_out[b][h]);
                let r_pows_k3 = gen_challenge_vector(&r_var, k3_data.q_sim[b][h][0].len()); 
                enforce_freivalds(
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
                    None,
                )?;
                global_k3_lows.extend(h_k3_lows); global_k3_highs.extend(h_k3_highs);
                eprintln!("[CCOUNT] b={} h={} after K2+K3: {}", b, h, cs.num_constraints());

                // Softmax 1 (v3)
                enforce_softmax(
                    cs.clone(),
                    &k1_data.q_sim[b][h],
                    &k1_data.r_matmul[b][h],
                    &sm1_vars.r_sm[b][h],
                    &sm1_vars.m_sm,
                    &sm1_vars.table_idx[b][h],
                    &sm1_vars.exp_total[b][h],
                    &sm1_vars.p[b][h],
                    &sm1_vars.r_div[b][h],
                    &self.sm1.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut global_sm1_idx,
                    &mut global_sm1_r_sm,
                    &mut r_div_k1_lows,
                    &mut r_div_k1_highs,
                )?;

                eprintln!("[CCOUNT] b={} h={} after SM1: {}", b, h, cs.num_constraints());

                // Softmax 2 (v3)
                enforce_softmax(
                    cs.clone(),
                    &k2_data.q_sim[b][h],
                    &k2_data.r_matmul[b][h],
                    &sm2_vars.r_sm[b][h],
                    &sm2_vars.m_sm,
                    &sm2_vars.table_idx[b][h],
                    &sm2_vars.exp_total[b][h],
                    &sm2_vars.p[b][h],
                    &sm2_vars.r_div[b][h],
                    &self.sm2.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut global_sm2_idx,
                    &mut global_sm2_r_sm,
                    &mut r_div_k2_lows,
                    &mut r_div_k2_highs,
                )?;


                eprintln!("[CCOUNT] b={} h={} after SM2: {}", b, h, cs.num_constraints());

                // Softmax 3 (v3) — mask applied to exp_total (not q_sim)
                let zero_var = FpVar::zero();

                // v3: enforce softmax without masking q_sim
                enforce_softmax(
                    cs.clone(),
                    &k3_data.q_sim[b][h],
                    &k3_data.r_matmul[b][h],
                    &sm3_vars.r_sm[b][h],
                    &sm3_vars.m_sm,
                    &sm3_vars.table_idx[b][h],
                    &sm3_vars.exp_total[b][h],
                    &sm3_vars.p[b][h],
                    &sm3_vars.r_div[b][h],
                    &self.sm3.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut global_sm3_idx,
                    &mut global_sm3_r_sm,
                    &mut r_div_k3_lows,
                    &mut r_div_k3_highs,
                )?;

                // v3 mask: verify exp_total=0 at padding positions
                {
                    let num_landmarks = sm3_vars.exp_total[b][h].len();
                    let seq_len = sm3_vars.exp_total[b][h][0].len();
                    for l in 0..num_landmarks {
                        for s in 0..seq_len {
                            let is_valid = mask_var[b][s].is_one()?;
                            // If NOT valid (padding), exp_total must be zero
                            let should_be_zero = FpVar::conditionally_select(
                                &is_valid,
                                &zero_var,
                                &sm3_vars.exp_total[b][h][l][s],
                            )?;
                            should_be_zero.enforce_equal(&zero_var)?;
                        }
                    }
                }


                eprintln!("[CCOUNT] b={} h={} after SM3+mask: {}", b, h, cs.num_constraints());

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
                enforce_freivalds(
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
                    None,
                )?;
                global_inv0_lows.extend(h_inv0_lows); global_inv0_highs.extend(h_inv0_highs);

                // Step 2: Q_y1 * P_k2 = P_k2
                let mut h_inv1_lows = Vec::new(); 
                let mut h_inv1_highs = Vec::new();
                enforce_freivalds(
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
                    None,
                )?;
                global_inv1_lows.extend(h_inv1_lows); global_inv1_highs.extend(h_inv1_highs);

                // A_inv * A * A_inv = A_inv
                // Step 2-1: Q_inv (A+) * P_k2 (A) = Q_y3
                let mut h_inv2_lows = Vec::new(); 
                let mut h_inv2_highs = Vec::new();
                enforce_freivalds(
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
                    None,
                )?;
                global_inv2_lows.extend(h_inv2_lows); global_inv2_highs.extend(h_inv2_highs);

                // Step 2-2: Q_y3 * Q_inv (A+) = Q_y4
                let mut h_inv3_lows = Vec::new(); 
                let mut h_inv3_highs = Vec::new();
                enforce_freivalds(
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
                    None,
                )?;
                global_inv3_lows.extend(h_inv3_lows); global_inv3_highs.extend(h_inv3_highs);
                eprintln!("[CCOUNT] b={} h={} after pseudo-inv: {}", b, h, cs.num_constraints());

                // temp
                let mut h_temp_lows = Vec::new(); 
                let mut h_temp_highs = Vec::new();
                let r_pows_temp = gen_challenge_vector(&r_var, temp_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &sm3_vars.p[b][h],
                    &proj_v_out[b][h],
                    &temp_data.q_sim[b][h],
                    &temp_data.r_matmul[b][h],
                    &self.matmul_temp.r_matmul[b][h],
                    &params_temp,
                    &r_pows_temp,
                    &mut h_temp_lows,
                    &mut h_temp_highs,
                    None,
                )?;
                global_temp_lows.extend(h_temp_lows); global_temp_highs.extend(h_temp_highs);

                // mid
                let mut h_mid_lows = Vec::new(); 
                let mut h_mid_highs = Vec::new();
                let r_pows_mid = gen_challenge_vector(&r_var, mid_data.q_sim[b][h][0].len()); // 길이 L
                enforce_freivalds(
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
                    None,
                )?;
                global_mid_lows.extend(h_mid_lows); global_mid_highs.extend(h_mid_highs);

                // X
                let mut h_x_lows = Vec::new(); 
                let mut h_x_highs = Vec::new();
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len()); // 길이 D
                enforce_freivalds(
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
                    None,
                )?;
                global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);
                eprintln!("[CCOUNT] b={} h={} after temp/mid/x: {}", b, h, cs.num_constraints());

                // === O Projection: O = attn_output @ W_o_h^T + b_o_h ===
                // Use q_attn_output [B][S][HD] (post-conv if conv exists, else same as x_data.q_sim)
                {
                    let o_input = &attn_output_vars[b];

                    let w_o_h_t = &w_o_vars[h];
                    let d_dim = self.proj_o.q_w_diff[h].len();
                    let proj_o_params = ZenParams {
                        m: FpVar::Constant(self.proj_o.m_proj),
                        m_prime: FpVar::Constant(self.proj_o.m_prime_proj),
                        z_w: FpVar::Constant(self.z_attn_output),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    let r_pows_po = gen_challenge_vector(&r_var, d_dim);
                    enforce_freivalds(
                        cs.clone(), &o_input, w_o_h_t,
                        &proj_o_out[b][h], &proj_o_r[b][h],
                        &self.proj_o.r_out_proj[b][h],
                        &proj_o_params, &r_pows_po,
                        &mut global_proj_o_lows, &mut global_proj_o_highs,
                        Some(&self.proj_o.q_b_diff[h]),
                    )?;
                }
                eprintln!("[CCOUNT] b={} h={} after O-proj: {} (h_total={})", b, h, cs.num_constraints(), cs.num_constraints() - _nc_h_start);
            }
        }

        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // 1. Softmax 인덱스 블록 (v3: C=1, single pool)
            final_vars.extend(global_sm1_idx);
            final_vars.extend(global_sm2_idx);
            final_vars.extend(global_sm3_idx);

            // === 26-bit remainders (C=2: 16+10) ===
            // Order must match run_all_remainders_with_link target_26bit
            final_vars.extend(global_k1_lows); final_vars.extend(global_k1_highs);
            final_vars.extend(global_k2_lows); final_vars.extend(global_k2_highs);
            final_vars.extend(global_k3_lows); final_vars.extend(global_k3_highs);
            // Moore-Penrose Steps
            final_vars.extend(global_inv0_lows); final_vars.extend(global_inv0_highs);
            final_vars.extend(global_inv1_lows); final_vars.extend(global_inv1_highs);
            final_vars.extend(global_inv2_lows); final_vars.extend(global_inv2_highs);
            final_vars.extend(global_inv3_lows); final_vars.extend(global_inv3_highs);
            // Final MatMuls
            final_vars.extend(global_temp_lows); final_vars.extend(global_temp_highs);
            final_vars.extend(global_mid_lows); final_vars.extend(global_mid_highs);
            final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
            // Projection remainders
            final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
            final_vars.extend(global_proj_k_lows); final_vars.extend(global_proj_k_highs);
            final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
            final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);
            // Scaling remainders
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

            // === 52-bit r_sm (C=4: 16+16+16+4) ===
            for k in 0..4 { final_vars.extend(global_sm1_r_sm[k].clone()); }
            for k in 0..4 { final_vars.extend(global_sm2_r_sm[k].clone()); }
            for k in 0..4 { final_vars.extend(global_sm3_r_sm[k].clone()); }

            // === R_div per-softmax (26-bit C=2: 16+10) ===
            // Order must match Lasso results: sm1[low,high], sm2[low,high], sm3[low,high]
            final_vars.extend(r_div_k1_lows);
            final_vars.extend(r_div_k1_highs);
            final_vars.extend(r_div_k2_lows);
            final_vars.extend(r_div_k2_highs);
            final_vars.extend(r_div_k3_lows);
            final_vars.extend(r_div_k3_highs);
        }

        // === Poseidon hash of x_input → public input ===
        {
            use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
            use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

            let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();
            let declared_input_hash = FpVar::new_input(cs.clone(), || Ok(self.input_hash))?;

            let mut x_flat: Vec<FpVar<F>> = Vec::new();
            for batch in &x_input_vars {
                for seq in batch {
                    x_flat.extend(seq.iter().cloned());
                }
            }
            let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
            sponge_var.absorb(&x_flat)?;
            let computed_hash = sponge_var.squeeze_field_elements(1)?;
            computed_hash[0].enforce_equal(&declared_input_hash)?;
        }

        // === Poseidon hash of output → public input ===
        {
            use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
            use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

            let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();

            let declared_hash = FpVar::new_input(cs.clone(), || Ok(self.output_hash))?;

            // 2. Flatten proj_o_out and compute in-circuit Poseidon hash
            let mut output_flat: Vec<FpVar<F>> = Vec::new();
            for batch in &proj_o_out {
                for head in batch {
                    for seq in head {
                        output_flat.extend(seq.iter().cloned());
                    }
                }
            }
            let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
            sponge_var.absorb(&output_flat)?;
            let computed_hash = sponge_var.squeeze_field_elements(1)?;

            // 3. Constraint: in-circuit hash == declared public input
            computed_hash[0].enforce_equal(&declared_hash)?;
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

    let r = E::ScalarField::zero();
    
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
    
    // v3: single table lookup — load table_idx, exp_total, P, R_div, r_sm, M_sm per kernel
    let load_softmax = |npz: &mut NpzReader<File>, suffix: &str| -> Result<SoftmaxBundle<E::ScalarField>> {
        Ok(SoftmaxBundle {
            table_idx: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("table_idx_{}.npy", suffix))?.into_dimensionality()?),
            exp_total: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("exp_total_{}.npy", suffix))?.into_dimensionality()?),
            p: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("P_{}.npy", suffix))?.into_dimensionality()?),
            r_div: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("R_div_{}.npy", suffix))?.into_dimensionality()?),
            r_sm: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("r_sm_{}.npy", suffix))?.into_dimensionality()?),
            m_sm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_sm_{}.npy", suffix))?.into_scalar()),
        })
    };

    let sm1 = load_softmax(&mut npz, "k1")?;
    let sm2 = load_softmax(&mut npz, "k2")?;
    let sm3 = load_softmax(&mut npz, "k3")?;

    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    // v3: limb_bits and k_segments removed from NPZ
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

    let z_v = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar();
    let matmul_temp = load_matmul(&mut npz, "temp", false)?;
    let matmul_mid = load_matmul(&mut npz, "mid", false)?; // P_k1 @ Q_inv_k2
    let matmul_x = load_matmul(&mut npz, "x", false)?;     // mid @ temp

    // === Projection witness loading ===
    let q_x_input = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_x_input.npy")?.into_dimensionality()?);
    let z_x_input = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_x_input.npy")?.into_scalar());

    let load_projection = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ProjectionBundle<E::ScalarField>> {
        Ok(ProjectionBundle {
            q_w_diff: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_w_diff_{}.npy", suffix))?.into_dimensionality()?),
            q_b_diff: to_field_2d_i64(&npz.by_name::<_, Ix2>(&format!("Q_b_diff_{}.npy", suffix))?.into_dimensionality()?),
            z_w: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_w_{}.npy", suffix))?.into_scalar()),
            m_proj: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_proj_{}.npy", suffix))?.into_scalar()),
            m_prime_proj: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_proj_{}.npy", suffix))?.into_scalar()),
            q_out_proj: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_raw_proj_{}.npy", suffix))?.into_dimensionality()?),
            r_out_proj: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_proj_{}.npy", suffix))?.into_dimensionality()?),
            z_raw: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_raw_{}.npy", suffix))?.into_scalar()),
        })
    };

    let proj_q = load_projection(&mut npz, "q")?;
    let proj_k = load_projection(&mut npz, "k")?;
    let proj_v = load_projection(&mut npz, "v")?;

    let load_scaling = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ScalingBundle<E::ScalarField>> {
        Ok(ScalingBundle {
            q_scaled: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_scaled_{}.npy", suffix))?.into_dimensionality()?),
            r_scale: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_scale_{}.npy", suffix))?.into_dimensionality()?),
            m_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_scale_{}.npy", suffix))?.into_scalar()),
            m_prime_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_scale_{}.npy", suffix))?.into_scalar()),
        })
    };

    let scale_q = load_scaling(&mut npz, "q")?;
    let scale_k = load_scaling(&mut npz, "k")?;

    // Output projection
    let z_attn_output = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_attn_output.npy")?.into_scalar());
    let q_attn_output = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_attn_output.npy")?.into_dimensionality()?);
    let proj_o = load_projection(&mut npz, "o")?;

    Ok(NystromAttentionCircuit {
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),
        z_v: E::ScalarField::from(z_v),

        q_x_input,
        z_x_input,
        proj_q,
        proj_k,
        proj_v,
        scale_q,
        scale_k,
        z_attn_output,
        q_attn_output,
        proj_o,

        matmul1,
        matmul2,
        matmul3,

        sm1,
        sm2,
        sm3,

        pseudo_inv,
        matmul_temp,
        matmul_mid,
        matmul_x,

        frac_bits,
        r_challenge: r,
        seg_size: seg,
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        input_hash: E::ScalarField::zero(),
        output_hash: E::ScalarField::zero(),

        committed_vars: Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
    })
}

pub fn run_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);

        // ── Dry-run: extract committed witness indices ───────────────────────
        let (d1_indices, d2_indices) = {
            let temp_circuit = load_nystrom_circuit::<E>(&file_path)?;
            let vars1_ptr = Rc::clone(&temp_circuit.committed_vars);
            let vars2_ptr = Rc::clone(&temp_circuit.committed_vars_2);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;
            (
                crate::util::utils::extract_witness_indices(&vars1_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&vars2_ptr.borrow()),
            )
        };
        let num_commit_1 = d1_indices.len() as u32;
        let num_commit_2 = d2_indices.len() as u32;
        println!("최종 측정된 커밋 개수: d1={:?}, d2={:?}", num_commit_1, num_commit_2);

        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (d1={:?}, d2={:?}) ===", num_commit_1, num_commit_2);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            &d1_indices,
            &d2_indices,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction + constraint count ────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        // (constraint check moved to final_circuit below)

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
        if !committed_group1.is_empty() {
            println!("  Head (10): {:?}", committed_group1[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  Tail (10): {:?}", committed_group1[committed_group1.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }

        let v1 = Fr::rand(&mut rng);
        // v2 blinds d2 (q_q / q_k / q_v). proof.d2 carries this commitment; it is
        // NOT included in the CP-link, which links only d1 to the Lasso proofs.
        let v2 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit (Pedersen cm1 from VK) ───────────────────────────
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        let r_fs = hash_cm_to_field::<E>(&cm1);

        // ── Compute Poseidon hashes (native) ──────────────────────────────────
        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<Fr>();
        let input_hash = {
            let mut x_flat: Vec<Fr> = Vec::new();
            for batch in &sample_circuit.q_x_input {
                for seq in batch {
                    x_flat.extend(seq.iter());
                }
            }
            crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &x_flat)
        };
        let output_hash = {
            let mut output_flat: Vec<Fr> = Vec::new();
            for batch in &sample_circuit.proj_o.q_out_proj {
                for head in batch {
                    for seq in head {
                        output_flat.extend(seq.iter());
                    }
                }
            }
            crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &output_flat)
        };

        println!("=== [1] Building final circuit with Fiat-Shamir challenge ===");
        let final_circuit = NystromAttentionCircuit {
            r_challenge: r_fs,
            input_hash,
            output_hash,
            ..sample_circuit.clone()
        };

        // R1CS constraint satisfaction check (with correct hashes)
        {
            let cs_check = ConstraintSystem::<Fr>::new_ref();
            final_circuit.clone().generate_constraints(cs_check.clone())?;
            let is_sat = cs_check.is_satisfied().unwrap_or(false);
            println!("[DEBUG] R1CS is_satisfied (final_circuit) = {}", is_sat);
            if !is_sat {
                println!("[DEBUG] num_constraints={}", cs_check.num_constraints());
                if let Some(unsat) = cs_check.which_is_unsatisfied().unwrap() {
                    println!("[DEBUG] First unsatisfied: {}", unsat);
                }
                return Err(anyhow::anyhow!("R1CS not satisfied (final_circuit)"));
            }
        }

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(),
            v1,
            v2,
            &params,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;
        let groth16_prove_time = groth16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // ── [1] Groth16 Verify ───────────────────────────────────────────────
        let public_inputs = vec![
            final_circuit.r_challenge,
            final_circuit.input_hash,
            final_circuit.output_hash,
        ];
        let groth16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let groth16_verify_time = groth16_verify_start.elapsed();

        println!("  - LegoGroth16 SNARK Verification: {:?}", result);
        if let Err(e) = result {
            println!("    Reason: {:?}", e);
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        // ── Log circuit output (proj_o_out) ──────────────────────────────────
        /*println!("\n=== Circuit Output (proj_o_out) ===");
        for (b, batch) in final_circuit.proj_o.q_out_proj.iter().enumerate() {
            for (h, head) in batch.iter().enumerate() {
                println!("  [batch={}, head={}] shape=[{}][{}]", b, h, head.len(), head[0].len());
                for (s, seq) in head.iter().enumerate() {
                    let vals: Vec<_> = seq.iter()
                        .map(|v| v.into_bigint())
                        .collect();
                    println!("    seq[{}]: {:?}", s, vals);
                }
            }
        }*/
        println!("  output_hash = {:?}", final_circuit.output_hash.into_bigint());

        // ── Groth16 proof size report ──
        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // ── [2] Lasso (commit + prove + verify bundled in helpers) ───────────
        // Reset Lasso aggregator before individual Lasso proofs
        proof_size::reset_lasso_aggregator();
        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let (lasso_sm1, sm1_agg) = run_lasso_bundle(&final_circuit.sm1, b"lasso_k1", &mut rng)?;
        lasso_agg.add_aggregate(&sm1_agg);
        let (lasso_sm2, sm2_agg) = run_lasso_bundle(&final_circuit.sm2, b"lasso_k2", &mut rng)?;
        lasso_agg.add_aggregate(&sm2_agg);
        let (lasso_sm3, sm3_agg) = run_lasso_bundle(&final_circuit.sm3, b"lasso_k3", &mut rng)?;
        lasso_agg.add_aggregate(&sm3_agg);
        let (lasso_remainders, rem_agg) = run_all_remainders_with_link::<Fr, G1>(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = vec![];
        lasso_results.extend(lasso_sm1);
        lasso_results.extend(lasso_sm2);
        lasso_results.extend(lasso_sm3);
        lasso_results.extend(lasso_remainders);

        // ── [3] CP-Link (setup/keygen, prove, verify timed separately) ───────
        let u_padded = pad_lasso_u_values::<E>(&lasso_results, &committed_group1);

        let cp_setup_start = Instant::now();
        let (pp, crs) = factored_link_setup::<E, _>(&params.vk, num_pub, committed_group1.len(), &lasso_results, &mut rng);
        let (ek, vk_link) = factored_link_keygen::<E, _>(&pp, crs, &mut rng);
        let cp_setup_ms = cp_setup_start.elapsed().as_millis();

        let cp_prove_start = Instant::now();
        let (proof_link, instance) = factored_link_prove::<E, _>(&pp, &ek, v1, cm1, &lasso_results, u_padded, &mut rng);
        let cp_prove_ms = cp_prove_start.elapsed().as_millis();

        let cp_verify_start = Instant::now();
        if !factored_link_verify::<E>(&pp, &vk_link, &instance, &proof_link) {
            return Err(anyhow::anyhow!("Factored link verification failed"));
        }
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        let cp_verify_ms = cp_verify_start.elapsed().as_millis();

        // ── CP-Link proof size report ──
        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        cplink_report.print_log();

        // ── Grand total proof size report ──
        proof_size::print_full_report(&g16_report, &cplink_report);

        let e2e_time = e2e_start.elapsed();

        // ── Benchmark output ─────────────────────────────────────────────────
        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "attention".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: g16_commit_ms,
                prove_ms: groth16_prove_time.as_millis(),
                verify_ms: groth16_verify_time.as_millis(),
                proof_size_bytes: proof_size,
                commitment_size_bytes: groth16_commitment_size::<E>(&cm1),
            },
            lasso: LassoBenchmarkMetrics::from_aggregate(&lasso_agg),
            cp_link: CpLinkMetrics {
                setup_ms: cp_setup_ms,
                prove_ms: cp_prove_ms,
                verify_ms: cp_verify_ms,
                proof_size_bytes: cplink_report.proof_total,
            },
            end_to_end_ms: e2e_time.as_millis(),
        }
        .print_json();
    }
    Ok(())
}

#[test]
fn nystrom_attn_hf_single_layer_test() {
    use std::env;

    // ── 환경 변수 ───────────────────────────────────────────────────────────
    let witness_dir = env::var("NYSTROM_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_m64".to_string());
    let layer_idx: Option<usize> = env::var("NYSTROM_ATTN_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] NYSTROM_ATTN_WITNESS_DIR={} not a directory. \
             Set the env var to a valid Phase 4 outputs_m{{N}}/ dir to run.",
            witness_dir
        );
        return;
    }

    // ── softmax 표 경로 결정: env var 미설정 시 witness_dir 안의 softmax_tables.npz ─
    // (자체학습 attention_test 는 default `outputs/softmax_tables.npz` fallback 그대로)
    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
        println!(
            "[info] SOFTMAX_TABLE_PATH not set; using {}/softmax_tables.npz",
            witness_dir
        );
    }

    // ── npz 수집 + 정렬 + layer_idx 만 단일 추출 ───────────────────────────
    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention.self.witness.npz"))
        {
            all_targets.push(path);
        }
    }
    all_targets.sort();

    if all_targets.is_empty() {
        eprintln!(
            "[skip] no '*attention.self.witness.npz' files in {}. \
             Run Phase 4 attention pipeline first.",
            witness_dir
        );
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!(
                "NYSTROM_ATTN_LAYER_IDX={} out of range (have {} layers).",
                idx,
                all_targets.len()
            );
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [Nystrom Attention prove+verify] dir={} | files={} | mode={} ===",
        witness_dir,
        targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!(
            "\n=== [Nystrom Attention] ALL {} files PASS | total_wall_ms={} ===",
            targets.len(), total_ms,
        ),
        Err(e) => panic!(
            "[Nystrom Attention] proof generate/verify failed after {}ms: {}",
            total_ms, e,
        ),
    }
}
