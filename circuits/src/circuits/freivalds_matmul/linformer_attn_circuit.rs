use std::{fs::File, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::pairing::Pairing;
use ndarray::{Ix0, Ix2, Ix3, Ix4};
use ndarray_npy::NpzReader;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{Rng, SeedableRng, rngs::StdRng},
    cell::RefCell,
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero, BigInteger};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{fp::FpVar, FieldVar}, R1CSVar, select::CondSelectGadget};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            global_link_setup, global_link_keygen, global_link_prove, global_link_verify,
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        proof_size,
    },
    gadgets::{
        matmul::{
            constraints::new_matrix_witness,
            quantized::{
                MatmulBundle, ProjectionBundle, ScalingBundle, ZenParams,
                enforce_freivalds, enforce_scaling, gen_challenge_vector,
                new_matrix_witness_4d, transpose_matrix,
            },
        },
        softmax::constraints::{SoftmaxBundle, SoftmaxVarBundle, enforce_softmax},
    },
    lasso::poly::{
        commitments::MultiCommitGens,
        dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds},
    },
    util::{
        load::{to_field_2d_i64, to_field_3d_i64, to_field_4d_i64, to_field_4d_u64},
        linformer_attn_lookup::{run_linformer_lasso_bundle, run_all_linformer_remainders_with_link},
        utils::{cm1_from_vk, cm2_from_vk, cm_aux_from_vk, generate_global_ck, commit_global, hash_commitments_to_field_v3},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;

type G1 = G1Projective;

/// Linformer attention circuit: BERT base + E_i/F_i projection matrices.
///
/// Linformer replaces O(n^2) attention with O(nk) by projecting K and V
/// from n to k dimensions using per-head learned projection matrices E_i, F_i.
///
/// Matmul chain:
///   E_i @ K_proc -> EK [k×D]
///   F_i @ V      -> FV [k×D]
///   Q_proc @ EK^T -> logit [S×k]
///   softmax(logit) -> P [S×k]
///   P @ FV -> x [S×D]
#[derive(Clone)]
pub struct LinformerAttentionCircuit<F: PrimeField> {
    // === Quantized Q/K/V zero-points (direct quantization of float refs) ===
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,

    // === Projection inputs ===
    pub q_x_input: Vec<Vec<Vec<F>>>,       // [B][S][HD]
    pub z_x_input: F,

    // === Q/K/V Projection (BERT-identical) ===
    pub proj_q: ProjectionBundle<F>,
    pub proj_k: ProjectionBundle<F>,
    pub proj_v: ProjectionBundle<F>,

    // === Q/K Scaling (BERT-identical) ===
    pub scale_q: ScalingBundle<F>,
    pub scale_k: ScalingBundle<F>,

    // === Output Projection (BERT-identical) ===
    pub z_attn_output: F,
    pub q_attn_output: Vec<Vec<Vec<F>>>,   // [B][S][HD]
    pub proj_o: ProjectionBundle<F>,

    // === Linformer projection weights ===
    pub q_w_diff_e: Vec<Vec<Vec<F>>>,      // [H][k][n] — E_i weight (diff)
    pub z_w_e: F,
    pub q_w_diff_f: Vec<Vec<Vec<F>>>,      // [H][k][n] — F_i weight (diff)
    pub z_w_f: F,

    // === Linformer matmuls ===
    /// E_i @ K_proc: [B][H][k][D]
    pub matmul_ek: MatmulBundle<F>,
    /// F_i @ V: [B][H][k][D]
    pub matmul_fv: MatmulBundle<F>,
    /// Q_proc @ EK^T: [B][H][S][k] (pre-softmax logit)
    pub matmul_logit: MatmulBundle<F>,

    /// Softmax over [B][H][S][k]
    pub sm: SoftmaxBundle<F>,

    /// P @ FV: [B][H][S][D]
    pub matmul_x: MatmulBundle<F>,

    pub frac_bits: usize,
    pub r_challenge: F,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,                 // [B][S]

    pub input_hash: F,
    pub output_hash: F,

    /// d1: Lasso-linked (remainder splits, softmax indices)
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    /// d2: W_Q, W_K, W_V, W_O, E_i, F_i weights
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
    /// aux: Freivalds-constrained intermediates (proj_q/k/v, EK, FV, x=P*FV)
    pub committed_vars_aux: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for LinformerAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();
        self.committed_vars_aux.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        let mut r_div_lows = Vec::new();
        let mut r_div_highs = Vec::new();

        // === Projection input allocation ===
        let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = self.q_x_input.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        let proj_q_out = new_matrix_witness_4d(cs.clone(), &self.proj_q.q_out_proj);
        let proj_q_r = new_matrix_witness_4d(cs.clone(), &self.proj_q.r_out_proj);
        let proj_k_out = new_matrix_witness_4d(cs.clone(), &self.proj_k.q_out_proj);
        let proj_k_r = new_matrix_witness_4d(cs.clone(), &self.proj_k.r_out_proj);
        let proj_v_out = new_matrix_witness_4d(cs.clone(), &self.proj_v.q_out_proj);
        let proj_v_r = new_matrix_witness_4d(cs.clone(), &self.proj_v.r_out_proj);

        let scale_q_out = new_matrix_witness_4d(cs.clone(), &self.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &self.scale_q.r_scale);
        let scale_k_out = new_matrix_witness_4d(cs.clone(), &self.scale_k.q_scaled);
        let scale_k_r = new_matrix_witness_4d(cs.clone(), &self.scale_k.r_scale);

        // === W_Q/K/V/O weight allocation (transposed: [H][HD][D]) ===
        let num_heads = self.proj_q.q_w_diff.len();
        let w_q_vars = alloc_weight_vars(cs.clone(), &self.proj_q.q_w_diff);
        let w_k_vars = alloc_weight_vars(cs.clone(), &self.proj_k.q_w_diff);
        let w_v_vars = alloc_weight_vars(cs.clone(), &self.proj_v.q_w_diff);
        let w_o_vars = alloc_weight_vars(cs.clone(), &self.proj_o.q_w_diff);

        // === E_i, F_i weight allocation [H][k][n] ===
        let e_vars = alloc_weight_vars_direct(cs.clone(), &self.q_w_diff_e);
        let f_vars = alloc_weight_vars_direct(cs.clone(), &self.q_w_diff_f);

        // === d2: commit W_Q, W_K, W_V, W_O, E_i, F_i ===
        {
            let mut vars_2 = self.committed_vars_2.borrow_mut();
            for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_k_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &e_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &f_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        }

        // === aux: commit proj_q/k/v outputs (Freivalds-constrained) ===
        {
            let mut vars_aux = self.committed_vars_aux.borrow_mut();
            for b in &proj_q_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
            for b in &proj_k_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
            for b in &proj_v_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
        }

        let proj_o_out = new_matrix_witness_4d(cs.clone(), &self.proj_o.q_out_proj);
        let proj_o_r = new_matrix_witness_4d(cs.clone(), &self.proj_o.r_out_proj);

        // === Allocate Linformer matmul witnesses ===
        // EK [B][H][k][D], FV [B][H][k][D]
        // z_w=0 because e_vars/f_vars are already (Q-z) diff form;
        // ZEN subtracts z_w from left matrix, so using z_w=z_e would double-subtract.
        let (ek_data, params_ek) =
            self.matmul_ek.alloc_and_setup(cs.clone(), F::zero(), self.z_k, &two_k_var);
        let (fv_data, params_fv) =
            self.matmul_fv.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);

        // aux: commit EK, FV (Freivalds-only, no element-wise constraint)
        {
            let mut vars_aux = self.committed_vars_aux.borrow_mut();
            for b in &ek_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
            for b in &fv_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
        }

        // logit [B][H][S][k]: z_left=z_q (Q_proc chain output), z_right=z_ek
        let (logit_data, params_logit) =
            self.matmul_logit.alloc_and_setup(cs.clone(), self.z_q, self.matmul_ek.z_sim, &two_k_var);

        // P*FV [B][H][S][D]: z_left=0 (softmax), z_right=z_fv
        let (x_data, params_x) =
            self.matmul_x.alloc_and_setup(cs.clone(), F::zero(), self.matmul_fv.z_sim, &two_k_var);

        // aux: commit x_data.q_sim (P*FV output, Freivalds-only)
        {
            let mut vars_aux = self.committed_vars_aux.borrow_mut();
            for b in &x_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
        }

        // O-proj input: rearranged reference to x_data.q_sim (arithmetic-free reshape)
        let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = {
            let num_h = x_data.q_sim[0].len();
            let head_dim = x_data.q_sim[0][0][0].len();
            x_data.q_sim.iter().map(|batch| {
                let seq_len = batch[0].len();
                (0..seq_len).map(|s| {
                    (0..num_h * head_dim).map(|hd| {
                        batch[hd / head_dim][s][hd % head_dim].clone()
                    }).collect()
                }).collect()
            }).collect()
        };

        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm_vars = self.sm.alloc_vars(cs.clone())?;

        // === Remainder pools ===
        let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
        let mut global_proj_k_lows = Vec::new(); let mut global_proj_k_highs = Vec::new();
        let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
        let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

        let mut global_ek_lows = Vec::new(); let mut global_ek_highs = Vec::new();
        let mut global_fv_lows = Vec::new(); let mut global_fv_highs = Vec::new();
        let mut global_logit_lows = Vec::new(); let mut global_logit_highs = Vec::new();
        let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();

        let mut global_sm_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_sm_r_sm: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        let zero_var = FpVar::<F>::zero();

        // Pre-compute mask booleans [B][S] for conditional_select masking
        let mut mask_bools: Vec<Vec<_>> = Vec::new();
        for b_mask in &mask_var {
            let mut batch_bools = Vec::new();
            for m_val in b_mask {
                batch_bools.push(m_val.is_one()?);
            }
            mask_bools.push(batch_bools);
        }
        let z_k_mask = FpVar::Constant(self.z_k);
        let z_q_mask = FpVar::Constant(self.z_q);
        let z_v_mask = FpVar::Constant(self.z_v);

        // === Per-(batch, head) constraint loop ===
        #[cfg(debug_assertions)]
        eprintln!("[linformer] before loop: {} constraints", cs.num_constraints());
        for b in 0..scale_q_out.len() {
            for h in 0..scale_q_out[b].len() {
                // --- Q Projection (Freivalds) ---
                {
                    let d_dim = self.proj_q.q_w_diff[h].len();
                    let proj_q_params = ZenParams {
                        m: FpVar::Constant(self.proj_q.m_proj),
                        m_prime: FpVar::Constant(self.proj_q.m_prime_proj),
                        z_w: FpVar::Constant(self.z_x_input),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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
                // --- K Projection ---
                {
                    let d_dim = self.proj_k.q_w_diff[h].len();
                    let proj_k_params = ZenParams {
                        m: FpVar::Constant(self.proj_k.m_proj),
                        m_prime: FpVar::Constant(self.proj_k.m_prime_proj),
                        z_w: FpVar::Constant(self.z_x_input),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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
                // --- V Projection ---
                {
                    let d_dim = self.proj_v.q_w_diff[h].len();
                    let proj_v_params = ZenParams {
                        m: FpVar::Constant(self.proj_v.m_proj),
                        m_prime: FpVar::Constant(self.proj_v.m_prime_proj),
                        z_w: FpVar::Constant(self.z_x_input),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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
                // --- Q Scaling ---
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
                // --- K Scaling ---
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

                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after scaling: {} constraints", b, h, cs.num_constraints());
                // --- Masking: set K_proc, Q_proc, V to zero-point at padding ---
                // Linformer F@V mixes all seq positions without softmax gating,
                // so padding must be zeroed. conditional_select after enforce_scaling.
                let seq_len = scale_k_out[b][h].len();
                let mut masked_k: Vec<Vec<FpVar<F>>> = Vec::with_capacity(seq_len);
                let mut masked_q: Vec<Vec<FpVar<F>>> = Vec::with_capacity(seq_len);
                let mut masked_v: Vec<Vec<FpVar<F>>> = Vec::with_capacity(seq_len);
                for s in 0..seq_len {
                    let is_valid = &mask_bools[b][s];
                    let d_k = scale_k_out[b][h][s].len();
                    let mut row_k = Vec::with_capacity(d_k);
                    let mut row_q = Vec::with_capacity(d_k);
                    let mut row_v = Vec::with_capacity(d_k);
                    for d in 0..d_k {
                        row_k.push(FpVar::conditionally_select(
                            is_valid, &scale_k_out[b][h][s][d], &z_k_mask,
                        )?);
                        row_q.push(FpVar::conditionally_select(
                            is_valid, &scale_q_out[b][h][s][d], &z_q_mask,
                        )?);
                        row_v.push(FpVar::conditionally_select(
                            is_valid, &proj_v_out[b][h][s][d], &z_v_mask,
                        )?);
                    }
                    masked_k.push(row_k);
                    masked_q.push(row_q);
                    masked_v.push(row_v);
                }

                // --- [L1] E_i @ K_proc -> EK [k×D] ---
                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after masking: {} constraints", b, h, cs.num_constraints());
                // E_i [k×n] @ K_proc [n×D] — no transpose needed
                {
                    let r_pows_ek = gen_challenge_vector(&r_var, ek_data.q_sim[b][h][0].len());
                    let mut h_ek_lows = Vec::new();
                    let mut h_ek_highs = Vec::new();
                    enforce_freivalds(
                        cs.clone(),
                        &e_vars[h],              // E_i [k×n]
                        &masked_k,               // K_proc [n×D] (masked)
                        &ek_data.q_sim[b][h],
                        &ek_data.r_matmul[b][h],
                        &self.matmul_ek.r_matmul[b][h],
                        &params_ek,
                        &r_pows_ek,
                        &mut h_ek_lows,
                        &mut h_ek_highs,
                        None,
                    )?;
                    global_ek_lows.extend(h_ek_lows);
                    global_ek_highs.extend(h_ek_highs);
                }

                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after EK: {} constraints", b, h, cs.num_constraints());
                // --- [L2] F_i @ V -> FV [k×D] ---
                // F_i [k×n] @ V [n×D] — no transpose needed
                {
                    let r_pows_fv = gen_challenge_vector(&r_var, fv_data.q_sim[b][h][0].len());
                    let mut h_fv_lows = Vec::new();
                    let mut h_fv_highs = Vec::new();
                    enforce_freivalds(
                        cs.clone(),
                        &f_vars[h],              // F_i [k×n]
                        &masked_v,               // V [n×D] (masked)
                        &fv_data.q_sim[b][h],
                        &fv_data.r_matmul[b][h],
                        &self.matmul_fv.r_matmul[b][h],
                        &params_fv,
                        &r_pows_fv,
                        &mut h_fv_lows,
                        &mut h_fv_highs,
                        None,
                    )?;
                    global_fv_lows.extend(h_fv_lows);
                    global_fv_highs.extend(h_fv_highs);
                }

                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after FV: {} constraints", b, h, cs.num_constraints());
                // --- [L3] Q_proc @ EK^T -> logit [S×k] ---
                // Q_proc [S×D] @ EK^T [D×k] — EK needs transpose
                {
                    let ek_t = transpose_matrix(&ek_data.q_sim[b][h]);
                    let r_pows_logit = gen_challenge_vector(&r_var, logit_data.q_sim[b][h][0].len());
                    let mut h_logit_lows = Vec::new();
                    let mut h_logit_highs = Vec::new();
                    enforce_freivalds(
                        cs.clone(),
                        &masked_q,               // Q_proc [S×D] (masked)
                        &ek_t,                    // EK^T [D×k]
                        &logit_data.q_sim[b][h],
                        &logit_data.r_matmul[b][h],
                        &self.matmul_logit.r_matmul[b][h],
                        &params_logit,
                        &r_pows_logit,
                        &mut h_logit_lows,
                        &mut h_logit_highs,
                        None,
                    )?;
                    global_logit_lows.extend(h_logit_lows);
                    global_logit_highs.extend(h_logit_highs);
                }

                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after logit: {} constraints", b, h, cs.num_constraints());
                // --- [L4] Softmax over logit [S×k] ---
                // No mask on softmax: K_proc pre-masking handles padding
                enforce_softmax(
                    cs.clone(),
                    &logit_data.q_sim[b][h],
                    &logit_data.r_matmul[b][h],
                    &sm_vars.r_sm[b][h],
                    &sm_vars.m_sm,
                    &sm_vars.table_idx[b][h],
                    &sm_vars.exp_total[b][h],
                    &sm_vars.p[b][h],
                    &sm_vars.r_div[b][h],
                    &self.sm.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut global_sm_idx,
                    &mut global_sm_r_sm,
                    &mut r_div_lows,
                    &mut r_div_highs,
                )?;

                // Linformer masking: K_proc, Q_proc, V are masked via conditional_select
                // before the matmuls (above). This zeroes padding contributions in
                // EK, FV, and logit. No additional softmax mask needed -- the k
                // dimension is a projection, not raw sequence positions.

                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after softmax: {} constraints", b, h, cs.num_constraints());
                // --- [L5] P @ FV -> x [S×D] ---
                // P [S×k] @ FV [k×D] — no transpose needed
                {
                    let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len());
                    let mut h_x_lows = Vec::new();
                    let mut h_x_highs = Vec::new();
                    enforce_freivalds(
                        cs.clone(),
                        &sm_vars.p[b][h],        // P [S×k]
                        &fv_data.q_sim[b][h],    // FV [k×D]
                        &x_data.q_sim[b][h],
                        &x_data.r_matmul[b][h],
                        &self.matmul_x.r_matmul[b][h],
                        &params_x,
                        &r_pows_x,
                        &mut h_x_lows,
                        &mut h_x_highs,
                        None,
                    )?;
                    global_x_lows.extend(h_x_lows);
                    global_x_highs.extend(h_x_highs);
                }

                #[cfg(debug_assertions)]
                eprintln!("[linformer] b={} h={} after P@FV: {} constraints", b, h, cs.num_constraints());
                // --- O Projection ---
                {
                    let o_input = &attn_output_vars[b];
                    let w_o_h_t = &w_o_vars[h];
                    let d_dim = self.proj_o.q_w_diff[h].len();
                    let proj_o_params = ZenParams {
                        m: FpVar::Constant(self.proj_o.m_proj),
                        m_prime: FpVar::Constant(self.proj_o.m_prime_proj),
                        z_w: FpVar::Constant(self.z_attn_output),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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
            }
        }

        // === Populate d1 committed_vars ===
        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // Softmax index (C=1)
            final_vars.extend(global_sm_idx);

            // 26-bit remainders — order must match run_all_linformer_remainders_with_link
            final_vars.extend(global_ek_lows); final_vars.extend(global_ek_highs);
            final_vars.extend(global_fv_lows); final_vars.extend(global_fv_highs);
            final_vars.extend(global_logit_lows); final_vars.extend(global_logit_highs);
            final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
            final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
            final_vars.extend(global_proj_k_lows); final_vars.extend(global_proj_k_highs);
            final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
            final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

            // 52-bit r_sm (C=4)
            for k in 0..4 {
                final_vars.extend(global_sm_r_sm[k].clone());
            }

            // R_div (26-bit C=2)
            final_vars.extend(r_div_lows);
            final_vars.extend(r_div_highs);
        }

        #[cfg(debug_assertions)]
        eprintln!("[linformer] after all b/h + final_vars: {} constraints", cs.num_constraints());
        // === Poseidon hash of x_input -> public input ===
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

        // === Poseidon hash of output -> public input ===
        {
            use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
            use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

            let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();
            let declared_hash = FpVar::new_input(cs.clone(), || Ok(self.output_hash))?;

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
            computed_hash[0].enforce_equal(&declared_hash)?;
        }

        Ok(())
    }
}

// === Helper: allocate weight vars with transposed indexing [H][HD][D] -> [H][HD][D] ===
fn alloc_weight_vars<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_w_diff: &Vec<Vec<Vec<F>>>,
) -> Vec<Vec<Vec<FpVar<F>>>> {
    let num_heads = q_w_diff.len();
    (0..num_heads).map(|h| {
        let d_dim = q_w_diff[h].len();
        let hd_dim = q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| {
                FpVar::new_witness(cs.clone(), || Ok(q_w_diff[h][d][hd])).unwrap()
            }).collect()
        }).collect()
    }).collect()
}

// === Helper: allocate weight vars with direct indexing [H][k][n] ===
fn alloc_weight_vars_direct<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_w_diff: &Vec<Vec<Vec<F>>>,
) -> Vec<Vec<Vec<FpVar<F>>>> {
    q_w_diff.iter().map(|head| {
        head.iter().map(|row| {
            row.iter().map(|v| {
                FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()
            }).collect()
        }).collect()
    }).collect()
}


// === NPZ Loader ===

pub fn load_linformer_circuit<E: Pairing>(
    path: &Path,
) -> Result<LinformerAttentionCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening Linformer witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    // --- Q/K/V zero-points (direct quantization) ---
    let z_q = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar());
    let z_k = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar());
    let z_v = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar());

    // --- Metadata ---
    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;

    // --- Projection inputs ---
    let q_x_input = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_x_input.npy")?.into_dimensionality()?);
    let z_x_input = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_x_input.npy")?.into_scalar());

    // --- Q/K/V/O Projections ---
    let load_projection = |npz: &mut NpzReader<std::fs::File>, suffix: &str| -> Result<ProjectionBundle<E::ScalarField>> {
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

    let load_scaling = |npz: &mut NpzReader<std::fs::File>, suffix: &str| -> Result<ScalingBundle<E::ScalarField>> {
        Ok(ScalingBundle {
            q_scaled: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_scaled_{}.npy", suffix))?.into_dimensionality()?),
            r_scale: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_scale_{}.npy", suffix))?.into_dimensionality()?),
            m_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_scale_{}.npy", suffix))?.into_scalar()),
            m_prime_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_scale_{}.npy", suffix))?.into_scalar()),
        })
    };

    let scale_q = load_scaling(&mut npz, "q")?;
    let scale_k = load_scaling(&mut npz, "k")?;

    let z_attn_output = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_attn_output.npy")?.into_scalar());
    let q_attn_output = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_attn_output.npy")?.into_dimensionality()?);
    let proj_o = load_projection(&mut npz, "o")?;

    // --- Linformer E_i, F_i weights [H][k][n] ---
    let q_w_diff_e = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_w_diff_e.npy")?.into_dimensionality()?);
    let z_w_e = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_w_e.npy")?.into_scalar());
    let q_w_diff_f = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_w_diff_f.npy")?.into_dimensionality()?);
    let z_w_f = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_w_f.npy")?.into_scalar());

    // --- E_i @ K -> EK [B][H][k][D] ---
    let matmul_ek = MatmulBundle {
        r_mean: vec![],
        q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_sim_ek.npy")?.into_dimensionality()?),
        r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_matmul_ek.npy")?.into_dimensionality()?),
        m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_ek.npy")?.into_scalar()),
        m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_ek.npy")?.into_scalar()),
        z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_ek.npy")?.into_scalar()),
    };

    // --- F_i @ V -> FV [B][H][k][D] ---
    let matmul_fv = MatmulBundle {
        r_mean: vec![],
        q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_sim_fv.npy")?.into_dimensionality()?),
        r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_matmul_fv.npy")?.into_dimensionality()?),
        m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_fv.npy")?.into_scalar()),
        m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_fv.npy")?.into_scalar()),
        z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_fv.npy")?.into_scalar()),
    };

    // --- Q @ EK^T -> logit [B][H][S][k] ---
    let matmul_logit = MatmulBundle {
        r_mean: vec![],
        q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_sim_logit.npy")?.into_dimensionality()?),
        r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_matmul_logit.npy")?.into_dimensionality()?),
        m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_logit.npy")?.into_scalar()),
        m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_logit.npy")?.into_scalar()),
        z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_logit.npy")?.into_scalar()),
    };

    // --- Softmax [B][H][S][k] ---
    let sm = SoftmaxBundle {
        table_idx: to_field_4d_i64(&npz.by_name::<_, Ix4>("table_idx.npy")?.into_dimensionality()?),
        exp_total: to_field_4d_u64(&npz.by_name::<_, Ix4>("exp_total.npy")?.into_dimensionality()?),
        p: to_field_4d_u64(&npz.by_name::<_, Ix4>("P_lin.npy")?.into_dimensionality()?),
        r_div: to_field_4d_u64(&npz.by_name::<_, Ix4>("R_div_lin.npy")?.into_dimensionality()?),
        r_sm: to_field_4d_i64(&npz.by_name::<_, Ix4>("r_sm.npy")?.into_dimensionality()?),
        m_sm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar()),
    };

    // --- P @ FV -> x [B][H][S][D] ---
    let matmul_x = MatmulBundle {
        r_mean: vec![],
        q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_sim_x.npy")?.into_dimensionality()?),
        r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_matmul_x.npy")?.into_dimensionality()?),
        m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_x.npy")?.into_scalar()),
        m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_x.npy")?.into_scalar()),
        z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_x.npy")?.into_scalar()),
    };

    // --- Mask [B][S] ---
    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    Ok(LinformerAttentionCircuit {
        z_q,
        z_k,
        z_v,
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
        q_w_diff_e,
        z_w_e,
        q_w_diff_f,
        z_w_f,
        matmul_ek,
        matmul_fv,
        matmul_logit,
        sm,
        matmul_x,
        frac_bits,
        r_challenge: E::ScalarField::zero(),
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        input_hash: E::ScalarField::zero(),
        output_hash: E::ScalarField::zero(),
        committed_vars: Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
        committed_vars_aux: Rc::new(RefCell::new(Vec::new())),
    })
}


// === Proving Pipeline ===

pub fn run_linformer_attention_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [Linformer Attention] Processing: {:?} ===", file_path);

        // Dry-run: extract committed witness indices
        let (d1_indices, d2_indices, d3_indices) = {
            let temp = load_linformer_circuit::<E>(file_path)?;
            let v1_ptr = Rc::clone(&temp.committed_vars);
            let v2_ptr = Rc::clone(&temp.committed_vars_2);
            let v_aux_ptr = Rc::clone(&temp.committed_vars_aux);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            (
                crate::util::utils::extract_witness_indices(&v1_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&v2_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&v_aux_ptr.borrow()),
            )
        };
        let num_commit_1 = d1_indices.len() as u32;
        let num_commit_2 = d2_indices.len() as u32;
        let num_commit_3 = d3_indices.len() as u32;
        println!("Commit counts: d1={}, d2={}, d3={}", num_commit_1, num_commit_2, num_commit_3);

        let sample_circuit = load_linformer_circuit::<E>(file_path)?;

        // [0] Setup
        println!("=== [0] LegoGroth16 Setup (d1={}, d2={}, d3={}) ===", num_commit_1, num_commit_2, num_commit_3);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            &d1_indices, &d2_indices, &d3_indices,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // Witness extraction (committed values only; satisfaction check deferred to final_circuit)
        {
            let cs_tmp = ConstraintSystem::<Fr>::new_ref();
            sample_circuit.clone().generate_constraints(cs_tmp)?;
        }

        let committed_group1 = extract_committed_values(&sample_circuit.committed_vars, num_commit_1)?;
        let committed_group2 = extract_committed_values(&sample_circuit.committed_vars_2, num_commit_2)?;
        let committed_aux = {
            let vars = sample_circuit.committed_vars_aux.borrow();
            vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect::<Vec<Fr>>()
        };

        println!("[Debug] d1={}, d2={}, aux={}", committed_group1.len(), committed_group2.len(), committed_aux.len());

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let v3 = Fr::rand(&mut rng);
        // num_pub: 3 public inputs (r_challenge, input_hash, output_hash)
        let num_pub = 3usize;

        // Groth16 commit
        let g16_commit_start = Instant::now();
        let cm1_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let cm1_ms = cm1_start.elapsed().as_millis();
        let cm2_start = Instant::now();
        let cm2 = cm2_from_vk::<E>(&params.vk, num_pub, &committed_group2, &v2);
        let cm2_ms = cm2_start.elapsed().as_millis();
        let cm_aux_start = Instant::now();
        let cm_aux = cm_aux_from_vk::<E>(&params.vk, num_pub, &committed_aux, &v3);
        let cm_aux_ms = cm_aux_start.elapsed().as_millis();
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        // Global weight commitment
        let global_commit_start = Instant::now();
        let ck_glob = generate_global_ck::<E, _>(committed_group2.len(), &mut rng);
        let o_glob = Fr::rand(&mut rng);
        let c_w = commit_global::<E>(&ck_glob, &committed_group2, &o_glob);
        let global_commit_ms = global_commit_start.elapsed().as_millis();

        // Poseidon hashes (native)
        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<Fr>();
        let input_hash = {
            let mut x_flat: Vec<Fr> = Vec::new();
            for batch in &sample_circuit.q_x_input {
                for seq in batch { x_flat.extend(seq.iter()); }
            }
            crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &x_flat)
        };
        let output_hash = {
            let mut output_flat: Vec<Fr> = Vec::new();
            for batch in &sample_circuit.proj_o.q_out_proj {
                for head in batch { for seq in head { output_flat.extend(seq.iter()); } }
            }
            crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &output_flat)
        };

        // Fiat-Shamir challenge (v3: includes cm_aux)
        let r_fs = hash_commitments_to_field_v3::<E>(&c_w, &cm1, &cm2, &cm_aux, &input_hash, &output_hash);

        println!("=== [1] Rebuilding circuit with Fiat-Shamir challenge ===");
        let final_circuit = LinformerAttentionCircuit {
            r_challenge: r_fs,
            input_hash,
            output_hash,
            ..sample_circuit.clone()
        };

        // R1CS satisfaction check (with correct hashes)
        let num_constraints = {
            let cs_check = ConstraintSystem::<Fr>::new_ref();
            final_circuit.clone().generate_constraints(cs_check.clone())?;
            let nc = cs_check.num_constraints();
            let is_sat = cs_check.is_satisfied().unwrap_or(false);
            println!("[DEBUG] R1CS is_satisfied (final_circuit) = {}", is_sat);
            if !is_sat {
                if let Some(unsat) = cs_check.which_is_unsatisfied().unwrap() {
                    println!("[DEBUG] First unsatisfied: {}", unsat);
                }
                return Err(anyhow::anyhow!(
                    "R1CS unsatisfied! constraints={}", nc,
                ));
            }
            println!("  R1CS satisfied: {} constraints", nc);
            nc
        };

        // [1] Groth16 Prove
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(), v1, v2, v3, &params, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
        let g16_prove_time = g16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // [1] Groth16 Verify
        let public_inputs = vec![final_circuit.r_challenge, final_circuit.input_hash, final_circuit.output_hash];
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let g16_verify_time = g16_verify_start.elapsed();

        println!("  - Linformer Attention SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }
        println!("  output_hash = {:?}", final_circuit.output_hash.into_bigint());

        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // [2] Lasso
        proof_size::reset_lasso_aggregator();
        println!("=== [2] Linformer Attention Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let (lasso_sm, sm_agg) = run_linformer_lasso_bundle(&final_circuit.sm, b"lin_lasso_sm", &mut rng)?;
        lasso_agg.add_aggregate(&sm_agg);
        let (lasso_remainders, rem_agg) = run_all_linformer_remainders_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = Vec::new();
        lasso_results.extend(lasso_sm);
        lasso_results.extend(lasso_remainders);

        // [3] CP-Link #1: d1 <-> Lasso
        let u_padded = pad_lasso_u_values::<E>(&lasso_results, &committed_group1);
        let cpl_setup_start = Instant::now();
        let (pp_lasso, crs_lasso) = factored_link_setup::<E, _>(&params.vk, num_pub, committed_group1.len(), &lasso_results, &mut rng);
        let (ek_lasso, vk_link_lasso) = factored_link_keygen::<E, _>(&pp_lasso, crs_lasso, &mut rng);
        let cpl_setup_ms = cpl_setup_start.elapsed().as_millis();

        let cpl_prove_start = Instant::now();
        let (proof_link_lasso, instance_lasso) = factored_link_prove::<E, _>(&pp_lasso, &ek_lasso, v1, cm1, &lasso_results, u_padded, &mut rng);
        let cpl_prove_ms = cpl_prove_start.elapsed().as_millis();

        let cpl_verify_start = Instant::now();
        if !factored_link_verify::<E>(&pp_lasso, &vk_link_lasso, &instance_lasso, &proof_link_lasso) {
            return Err(anyhow::anyhow!("CP-Link #1 (d1 <-> Lasso) verification failed"));
        }
        println!("  -> [Success] CP-Link #1 (d1 <-> Lasso) verified");
        let cpl_verify_ms = cpl_verify_start.elapsed().as_millis();

        let cplink_lasso_report = proof_size::compute_cplink_report::<E>(&proof_link_lasso, &instance_lasso);
        cplink_lasso_report.print_log();

        // [4] CP-Link #2: c_W <-> d2
        let cpg_setup_start = Instant::now();
        let (pp_glob, crs_glob) = global_link_setup::<E, _>(&params.vk, num_pub, &ck_glob, &mut rng);
        let (ek_glob, vk_link_glob) = global_link_keygen::<E, _>(&pp_glob, crs_glob, &mut rng);
        let cpg_setup_ms = cpg_setup_start.elapsed().as_millis();

        let cpg_prove_start = Instant::now();
        let (proof_link_glob, instance_glob) = global_link_prove::<E, _>(
            &pp_glob, &ek_glob, o_glob, v2, &committed_group2, c_w, proof.d2, &mut rng,
        );
        let cpg_prove_ms = cpg_prove_start.elapsed().as_millis();

        let cpg_verify_start = Instant::now();
        if !global_link_verify::<E>(&pp_glob, &vk_link_glob, &instance_glob, &proof_link_glob) {
            return Err(anyhow::anyhow!("CP-Link #2 (c_W <-> d2) verification failed"));
        }
        println!("  -> [Success] CP-Link #2 (c_W <-> d2) verified");
        let cpg_verify_ms = cpg_verify_start.elapsed().as_millis();

        let cplink_global_report = proof_size::compute_cplink_report::<E>(&proof_link_glob, &instance_glob);
        cplink_global_report.print_log();

        // Fiat-Shamir verification
        let r_fs_verify = hash_commitments_to_field_v3::<E>(
            &c_w, &proof.d1, &proof.d2, &proof.d3,
            &public_inputs[1], &public_inputs[2],
        );
        assert_eq!(r_fs, r_fs_verify, "Fiat-Shamir recomputation mismatch");
        println!("  -> [Success] Fiat-Shamir recomputation verified");

        proof_size::print_full_report(&g16_report, &cplink_lasso_report, &cplink_global_report);

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
            circuit: "linformer_attention".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: g16_commit_ms,
                cm1_ms,
                cm2_ms,
                cm_aux_ms,
                prove_ms: g16_prove_time.as_millis(),
                verify_ms: g16_verify_time.as_millis(),
                proof_size_bytes: proof_size,
                commitment_size_bytes: groth16_commitment_size::<E>(&cm1),
            },
            lasso: LassoBenchmarkMetrics::from_aggregate(&lasso_agg),
            cp_link_lasso: CpLinkMetrics {
                commit_ms: 0,
                setup_ms: cpl_setup_ms,
                prove_ms: cpl_prove_ms,
                verify_ms: cpl_verify_ms,
                proof_size_bytes: cplink_lasso_report.proof_total,
            },
            cp_link_global: CpLinkMetrics {
                commit_ms: global_commit_ms,
                setup_ms: cpg_setup_ms,
                prove_ms: cpg_prove_ms,
                verify_ms: cpg_verify_ms,
                proof_size_bytes: cplink_global_report.proof_total,
            },
            end_to_end_ms: e2e_time.as_millis(),
        }
        .print_json();
    }
    Ok(())
}

fn extract_committed_values(
    vars: &Rc<RefCell<Vec<FpVar<Fr>>>>,
    expected: u32,
) -> Result<Vec<Fr>> {
    let v = vars.borrow();
    let values: Vec<Fr> = v.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
    if values.len() != expected as usize {
        return Err(anyhow::anyhow!(
            "commit size mismatch: expected {}, got {}",
            expected, values.len()
        ));
    }
    Ok(values)
}


#[test]
fn linformer_attn_single_layer_test() {
    use std::env;

    let witness_dir = env::var("LINFORMER_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_linformer".to_string());
    let layer_idx: Option<usize> = env::var("LINFORMER_ATTN_LAYER_IDX")
        .ok().and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!("[skip] LINFORMER_ATTN_WITNESS_DIR={} not a directory.", witness_dir);
        return;
    }

    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
    }

    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name().and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention_self.witness.npz"))
        {
            all_targets.push(path);
        }
    }
    all_targets.sort();

    if all_targets.is_empty() {
        eprintln!("[skip] no '*attention.self.witness.npz' files in {}.", witness_dir);
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!("LINFORMER_ATTN_LAYER_IDX={} out of range (have {} layers).", idx, all_targets.len());
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [Linformer Attention prove+verify] dir={} | files={} ===",
        witness_dir, targets.len(),
    );

    let total_start = std::time::Instant::now();
    let res = run_linformer_attention_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!("\n=== [Linformer Attention] ALL {} files PASS | total_wall_ms={} ===", targets.len(), total_ms),
        Err(e) => panic!("[Linformer Attention] proof generate/verify failed after {}ms: {}", total_ms, e),
    }
}

// === B-2 Rejection tests ===

/// Load small fixture, return None and skip if witness dir missing.
fn load_small_linformer_fixture() -> Option<LinformerAttentionCircuit<Fr>> {
    let witness_dir = std::env::var("LINFORMER_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_n8_k4_L1_h8_linformer".to_string());
    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!("[skip] LINFORMER_ATTN_WITNESS_DIR={} not found", witness_dir);
        return None;
    }
    if std::env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
    }
    let mut targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name().and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention_self.witness.npz"))
        {
            targets.push(path);
        }
    }
    targets.sort();
    if targets.is_empty() { return None; }
    load_linformer_circuit::<Bn254>(&targets[0]).ok()
}

/// Compute native Poseidon hashes for a circuit and set them along with r_challenge.
fn prepare_circuit_hashes(c: &mut LinformerAttentionCircuit<Fr>) {
    let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<Fr>();
    c.input_hash = {
        let mut x_flat = Vec::new();
        for batch in &c.q_x_input {
            for seq in batch { x_flat.extend(seq.iter()); }
        }
        crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &x_flat)
    };
    c.output_hash = {
        let mut out_flat = Vec::new();
        for batch in &c.proj_o.q_out_proj {
            for head in batch { for seq in head { out_flat.extend(seq.iter()); } }
        }
        crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &out_flat)
    };
    // Non-zero challenge so Freivalds checks all columns
    c.r_challenge = Fr::from(42u64);
}

/// Check that a tampered circuit's R1CS is unsatisfied. Returns the failed constraint name.
fn assert_r1cs_rejects(
    circuit: LinformerAttentionCircuit<Fr>,
    label: &str,
) {
    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.generate_constraints(cs.clone()).expect("constraint gen");
    let is_sat = cs.is_satisfied().unwrap_or(false);
    if is_sat {
        panic!("[REJECT FAIL] {} -- circuit should be unsatisfied but is_satisfied=true", label);
    }
    let unsat = cs.which_is_unsatisfied().unwrap();
    println!("[REJECT OK] {} -- unsatisfied at constraint {:?}", label, unsat);
}

#[test]
fn linformer_reject_ek_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.matmul_ek.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "Q_sim_ek tamper");
}

#[test]
fn linformer_reject_fv_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.matmul_fv.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "Q_sim_fv tamper");
}

#[test]
fn linformer_reject_logit_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.matmul_logit.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "Q_sim_logit tamper");
}

#[test]
fn linformer_reject_softmax_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.sm.table_idx[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "softmax table_idx tamper");
}

#[test]
fn linformer_reject_x_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.matmul_x.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "Q_sim_x (P*FV) tamper");
}

#[test]
fn linformer_reject_oproj_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.proj_o.q_out_proj[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "Q_raw_proj_o tamper");
}

// === 1-1: Mask constraint rejection tests ===

#[test]
fn linformer_reject_mask_k_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    // Find a padding position (mask=0) and set K_proc to non-zero-point value.
    // mask is [B][S]; padding positions have mask=0.
    let s_len = c.mask[0].len();
    let mut pad_idx = None;
    for s in 0..s_len {
        if c.mask[0][s] == Fr::from(0u64) { pad_idx = Some(s); break; }
    }
    let s = pad_idx.expect("no padding position found in mask");
    // Tamper Q_scaled_k at padding: set to (z_k + 100) instead of z_k
    let num_heads = c.scale_k.q_scaled[0].len();
    let d_dim = c.scale_k.q_scaled[0][0][s].len();
    for h in 0..num_heads {
        for d in 0..d_dim {
            c.scale_k.q_scaled[0][h][s][d] = c.z_k + Fr::from(100u64);
        }
    }
    assert_r1cs_rejects(c, "mask K_proc tamper at padding");
}

#[test]
fn linformer_reject_mask_q_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    let s_len = c.mask[0].len();
    let mut pad_idx = None;
    for s in 0..s_len {
        if c.mask[0][s] == Fr::from(0u64) { pad_idx = Some(s); break; }
    }
    let s = pad_idx.expect("no padding position found in mask");
    let num_heads = c.scale_q.q_scaled[0].len();
    let d_dim = c.scale_q.q_scaled[0][0][s].len();
    for h in 0..num_heads {
        for d in 0..d_dim {
            c.scale_q.q_scaled[0][h][s][d] = c.z_q + Fr::from(100u64);
        }
    }
    assert_r1cs_rejects(c, "mask Q_proc tamper at padding");
}

#[test]
fn linformer_reject_mask_v_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    let s_len = c.mask[0].len();
    let mut pad_idx = None;
    for s in 0..s_len {
        if c.mask[0][s] == Fr::from(0u64) { pad_idx = Some(s); break; }
    }
    let s = pad_idx.expect("no padding position found in mask");
    let num_heads = c.proj_v.q_out_proj[0].len();
    let d_dim = c.proj_v.q_out_proj[0][0][s].len();
    for h in 0..num_heads {
        for d in 0..d_dim {
            c.proj_v.q_out_proj[0][h][s][d] = c.z_v + Fr::from(100u64);
        }
    }
    assert_r1cs_rejects(c, "mask V tamper at padding");
}

// === 1-2: E_i/F_i weight tamper rejection tests ===

#[test]
fn linformer_reject_e_weight_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.q_w_diff_e[0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "E_i weight tamper");
}

#[test]
fn linformer_reject_f_weight_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.q_w_diff_f[0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "F_i weight tamper");
}

// === B-3 Adaptive manipulation tests ===
//
// Freivalds kernel attack analysis:
// r_pows = [r, r^2, ..., r^D] (gen_challenge_vector, quantized.rs:107-118).
// For a SPECIFIC r, the kernel of the dot product sum_j v[j]*r^(j+1) = 0 is
// non-trivial (D-1 degrees of freedom). An adversary who knows r can construct
// a perturbation v that passes the EK Freivalds check.
//
// Defense: the Fiat-Shamir challenge r is derived from commitments including
// cm_aux (which contains EK). Tampering EK changes cm_aux, which changes r,
// invalidating the kernel vector. (hash_commitments_to_field_v3, utils.rs:248)
//
// R1CS-level test: we fix r=42, construct the kernel vector, and add it to
// one row of Q_sim_ek. EK Freivalds passes, but the logit Freivalds
// (Q_proc @ EK^T, line 444) uses the tampered EK^T and fails.
// This demonstrates that even if EK Freivalds is bypassed, the constraint
// chain catches the inconsistency. In the full pipeline, the commitment-
// challenge binding provides an additional defense layer.

#[test]
fn linformer_reject_adaptive_kernel() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    // r_challenge=42, r_pows=[r, r^2, r^3, r^4] for D=4.
    // Kernel vector: 42*v[0] + 42^2*v[1] = 0 => v = [-42, 1, 0, 0] mod p.
    let r = Fr::from(42u64);
    let neg_r = Fr::from(0u64) - r; // = p - 42
    let row = 0;
    c.matmul_ek.q_sim[0][0][row][0] += neg_r;
    c.matmul_ek.q_sim[0][0][row][1] += Fr::from(1u64);
    // EK Freivalds should pass (kernel nullifies dot product).
    // Logit Freivalds (Q_proc @ EK^T) should fail (tampered EK^T).
    assert_r1cs_rejects(c, "adaptive kernel (EK bypass -> logit fail)");
}

#[test]
fn linformer_reject_adaptive_zero_ek() {
    // Brute-force EK tamper: zero out all elements.
    // EK Freivalds constraint fails directly.
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    for s in 0..c.matmul_ek.q_sim[0][0].len() {
        for d in 0..c.matmul_ek.q_sim[0][0][s].len() {
            c.matmul_ek.q_sim[0][0][s][d] = Fr::from(0u64);
        }
    }
    assert_r1cs_rejects(c, "adaptive zero EK (Freivalds fail)");
}

#[test]
fn linformer_reject_aux_commit_tamper() {
    let mut c = match load_small_linformer_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_circuit_hashes(&mut c);
    c.proj_q.q_out_proj[0][0][0][0] += Fr::from(1u64);
    assert_r1cs_rejects(c, "aux commit tamper (proj_q)");
}
