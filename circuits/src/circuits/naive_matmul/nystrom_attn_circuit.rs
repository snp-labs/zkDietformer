use std::{path::Path, str::FromStr, time::Instant};
use anyhow::Result;
use ark_ec::pairing::Pairing;
use ark_std::{rc::Rc, fs, UniformRand, rand::{SeedableRng, rngs::StdRng}, cell::RefCell};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero, BigInteger};
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
        freivalds_matmul::nystrom_attn_circuit::{NystromAttentionCircuit, load_nystrom_circuit},
    },
    gadgets::{
        matmul::{
            constraints::new_matrix_witness,
            quantized::{
                ZenParams, compute_circuit_landmarks,
                new_matrix_witness_4d, transpose_matrix, enforce_scaling,
                // enforce_freivald and gen_challenge_vector removed
            },
            naive_quantized::enforce_naive_matmul,  // ← CHANGED: naive O(n^3) matmul
        },
        softmax::constraints::enforce_softmax,
    },
    util::{
        nystrom_attn_lookup::{run_lasso_bundle, run_all_remainders_with_link},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;
use crate::circuits::proof_size;

type G1 = G1Projective;

/// Newtype wrapper around `NystromAttentionCircuit` that uses naive matmul
/// instead of Freivalds for `generate_constraints`.
#[derive(Clone)]
pub struct NystromAttentionCircuitNot<F: PrimeField>(pub NystromAttentionCircuit<F>);

impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuitNot<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let inner = self.0;
        inner.committed_vars.borrow_mut().clear();
        inner.committed_vars_2.borrow_mut().clear();

        // Public input: Fiat-Shamir challenge (kept for interface compatibility)
        let _r_var = FpVar::new_input(cs.clone(), || Ok(inner.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << inner.k_bits))?;

        let mut r2_pool = Vec::new();
        let mut r3_pool = Vec::new();

        // 1. Common inputs and witnesses ([B][H][S][D])
        let q_proc = new_matrix_witness_4d(cs.clone(), &inner.scale_q.q_scaled);
        let k_proc = new_matrix_witness_4d(cs.clone(), &inner.scale_k.q_scaled);
        let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &inner.matmul1.r_mean);
        let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &inner.matmul2.r_mean);

        // 2. Kernel result data allocation
        let (k1_data, params_k1) = inner.matmul1.alloc_and_setup(cs.clone(), inner.z_q, inner.z_k, &two_k_var);
        let (k2_data, params_k2) = inner.matmul2.alloc_and_setup(cs.clone(), inner.z_q, inner.z_k, &two_k_var);
        let (k3_data, params_k3) = inner.matmul3.alloc_and_setup(cs.clone(), inner.z_q, inner.z_k, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &inner.mask);
        let sm1_vars = inner.sm1.alloc_vars(cs.clone())?;
        let sm2_vars = inner.sm2.alloc_vars(cs.clone())?;
        let sm3_vars = inner.sm3.alloc_vars(cs.clone())?;

        let v_proc = new_matrix_witness_4d(cs.clone(), &inner.proj_v.q_out_proj);

        // Allocate W_Q, W_K, W_V weights as witnesses (transposed form: [H][HD][D])
        let num_heads = inner.proj_q.q_w_diff.len();
        let w_q_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = inner.proj_q.q_w_diff[h].len();
            let hd_dim = inner.proj_q.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(inner.proj_q.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();
        let w_k_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = inner.proj_k.q_w_diff[h].len();
            let hd_dim = inner.proj_k.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(inner.proj_k.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();
        let w_v_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
            let d_dim = inner.proj_v.q_w_diff[h].len();
            let hd_dim = inner.proj_v.q_w_diff[h][0].len();
            (0..hd_dim).map(|hd| {
                (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(inner.proj_v.q_w_diff[h][d][hd])).unwrap()).collect()
            }).collect()
        }).collect();

        // d2: commit W_Q, W_K, W_V weights
        {
            let mut vars_2 = inner.committed_vars_2.borrow_mut();
            for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_k_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        }

        // === Projection input allocation (public input) ===
        let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.q_x_input.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_input(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        {
            let mut vars_2 = inner.committed_vars_2.borrow_mut();
            for batch in &x_input_vars {
                for seq in batch {
                    vars_2.extend(seq.iter().cloned());
                }
            }
        }

        let proj_q_out = new_matrix_witness_4d(cs.clone(), &inner.proj_q.q_out_proj);
        let proj_q_r = new_matrix_witness_4d(cs.clone(), &inner.proj_q.r_out_proj);
        let proj_k_out = new_matrix_witness_4d(cs.clone(), &inner.proj_k.q_out_proj);
        let proj_k_r = new_matrix_witness_4d(cs.clone(), &inner.proj_k.r_out_proj);
        let proj_v_out = new_matrix_witness_4d(cs.clone(), &inner.proj_v.q_out_proj);
        let proj_v_r = new_matrix_witness_4d(cs.clone(), &inner.proj_v.r_out_proj);

        let scale_q_out = new_matrix_witness_4d(cs.clone(), &inner.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &inner.scale_q.r_scale);
        let scale_k_out = new_matrix_witness_4d(cs.clone(), &inner.scale_k.q_scaled);
        let scale_k_r = new_matrix_witness_4d(cs.clone(), &inner.scale_k.r_scale);

        let proj_o_out = new_matrix_witness_4d(cs.clone(), &inner.proj_o.q_out_proj);
        let proj_o_r = new_matrix_witness_4d(cs.clone(), &inner.proj_o.r_out_proj);

        let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
        let mut global_proj_k_lows = Vec::new(); let mut global_proj_k_highs = Vec::new();
        let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
        let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

        let (temp_data, params_temp) = inner.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), inner.z_v, &two_k_var);
        let (mid_data, params_mid) = inner.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), inner.pseudo_inv.z_inv, &two_k_var);
        let (x_data, params_x) = inner.matmul_x.alloc_and_setup(cs.clone(), inner.matmul_mid.z_sim, inner.matmul_temp.z_sim, &two_k_var);

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
            let k_landmarks = compute_circuit_landmarks(cs.clone(), &k_proc[b], &r_mean_k1[b], inner.seg_size)?;
            let q_landmarks = compute_circuit_landmarks(cs.clone(), &q_proc[b], &r_mean_k2[b], inner.seg_size)?;

            for h in 0..q_proc[b].len() {
                // === Q/K/V Projection + Q/K Scaling (naive) ===

                // Q Projection
                {
                    let proj_q_params = ZenParams {
                        m: FpVar::Constant(inner.proj_q.m_proj),
                        m_prime: FpVar::Constant(inner.proj_q.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_x_input),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    enforce_naive_matmul(
                        cs.clone(), &x_input_vars[b], &w_q_vars[h],
                        &proj_q_out[b][h], &proj_q_r[b][h],
                        &inner.proj_q.r_out_proj[b][h],
                        &proj_q_params,
                        &mut global_proj_q_lows, &mut global_proj_q_highs,
                        Some(&inner.proj_q.q_b_diff[h]),
                    )?;
                }

                // K Projection
                {
                    let proj_k_params = ZenParams {
                        m: FpVar::Constant(inner.proj_k.m_proj),
                        m_prime: FpVar::Constant(inner.proj_k.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_x_input),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    enforce_naive_matmul(
                        cs.clone(), &x_input_vars[b], &w_k_vars[h],
                        &proj_k_out[b][h], &proj_k_r[b][h],
                        &inner.proj_k.r_out_proj[b][h],
                        &proj_k_params,
                        &mut global_proj_k_lows, &mut global_proj_k_highs,
                        Some(&inner.proj_k.q_b_diff[h]),
                    )?;
                }

                // V Projection
                {
                    let proj_v_params = ZenParams {
                        m: FpVar::Constant(inner.proj_v.m_proj),
                        m_prime: FpVar::Constant(inner.proj_v.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_x_input),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    enforce_naive_matmul(
                        cs.clone(), &x_input_vars[b], &w_v_vars[h],
                        &proj_v_out[b][h], &proj_v_r[b][h],
                        &inner.proj_v.r_out_proj[b][h],
                        &proj_v_params,
                        &mut global_proj_v_lows, &mut global_proj_v_highs,
                        Some(&inner.proj_v.q_b_diff[h]),
                    )?;
                }

                // Q Scaling
                {
                    let z_raw_q = FpVar::Constant(inner.proj_q.z_raw);
                    let m_scale_q = FpVar::Constant(inner.scale_q.m_scale);
                    let m_prime_q = FpVar::Constant(inner.scale_q.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &proj_q_out[b][h], &scale_q_out[b][h], &scale_q_r[b][h],
                        &inner.scale_q.r_scale[b][h],
                        &z_raw_q, &m_scale_q, &m_prime_q, &two_k_var,
                        &mut global_scale_q_lows, &mut global_scale_q_highs,
                    )?;
                }

                // K Scaling
                {
                    let z_raw_k = FpVar::Constant(inner.proj_k.z_raw);
                    let m_scale_k = FpVar::Constant(inner.scale_k.m_scale);
                    let m_prime_k = FpVar::Constant(inner.scale_k.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &proj_k_out[b][h], &scale_k_out[b][h], &scale_k_r[b][h],
                        &inner.scale_k.r_scale[b][h],
                        &z_raw_k, &m_scale_k, &m_prime_k, &two_k_var,
                        &mut global_scale_k_lows, &mut global_scale_k_highs,
                    )?;
                }

                let k_l_t = transpose_matrix(&k_landmarks[h]);
                // CHANGED: removed all gen_challenge_vector / r_pows

                // Kernel 1: CHANGED enforce_freivald → enforce_naive_matmul
                let mut h_k1_lows = Vec::new();
                let mut h_k1_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &q_proc[b][h],
                    &k_l_t,
                    &k1_data.q_sim[b][h],
                    &k1_data.r_matmul[b][h],
                    &inner.matmul1.r_matmul[b][h],
                    &params_k1,
                    &mut h_k1_lows,
                    &mut h_k1_highs,
                    None,
                )?;
                global_k1_lows.extend(h_k1_lows); global_k1_highs.extend(h_k1_highs);

                // Kernel 2: CHANGED enforce_freivald → enforce_naive_matmul
                let mut h_k2_lows = Vec::new();
                let mut h_k2_highs = Vec::new();
                let q_l_h = &q_landmarks[h];
                enforce_naive_matmul(
                    cs.clone(),
                    q_l_h,
                    &k_l_t,
                    &k2_data.q_sim[b][h],
                    &k2_data.r_matmul[b][h],
                    &inner.matmul2.r_matmul[b][h],
                    &params_k2,
                    &mut h_k2_lows,
                    &mut h_k2_highs,
                    None,
                )?;
                global_k2_lows.extend(h_k2_lows); global_k2_highs.extend(h_k2_highs);

                // Kernel 3: CHANGED enforce_freivald → enforce_naive_matmul
                let mut h_k3_lows = Vec::new();
                let mut h_k3_highs = Vec::new();
                let k_full_t = transpose_matrix(&k_proc[b][h]);
                enforce_naive_matmul(
                    cs.clone(),
                    &q_landmarks[h],
                    &k_full_t,
                    &k3_data.q_sim[b][h],
                    &k3_data.r_matmul[b][h],
                    &inner.matmul3.r_matmul[b][h],
                    &params_k3,
                    &mut h_k3_lows,
                    &mut h_k3_highs,
                    None,
                )?;
                global_k3_lows.extend(h_k3_lows); global_k3_highs.extend(h_k3_highs);

                // Softmax 1 (unchanged)
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
                    &sm1_vars.p[b][h],
                    &sm1_vars.r_div[b][h],
                    inner.frac_bits,
                    &two_k_var,
                    &mut h_sm1_idx,
                    &mut h_sm1_r_sm,
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm1_idx[k].extend(h_sm1_idx[k].clone()); }
                for k in 0..2 { global_sm1_r_sm[k].extend(h_sm1_r_sm[k].clone()); }

                // Softmax 2 (unchanged)
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
                    inner.frac_bits,
                    &two_k_var,
                    &mut h_sm2_idx,
                    &mut h_sm2_r_sm,
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm2_idx[k].extend(h_sm2_idx[k].clone()); }
                for k in 0..2 { global_sm2_r_sm[k].extend(h_sm2_r_sm[k].clone()); }

                // Softmax 3 (unchanged — includes masking)
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
                            &zero_var,
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
                    &sm3_vars.p[b][h],
                    &sm3_vars.r_div[b][h],
                    inner.frac_bits,
                    &two_k_var,
                    &mut h_sm3_idx,
                    &mut h_sm3_r_sm,
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm3_idx[k].extend(h_sm3_idx[k].clone()); }
                for k in 0..2 { global_sm3_r_sm[k].extend(h_sm3_r_sm[k].clone()); }

                // Moore-Penrose pseudoinverse verification
                let q_inv_var = new_matrix_witness_4d(cs.clone(), &inner.pseudo_inv.q_inv);
                let q_y_vars = inner.pseudo_inv.q_y_intermediates.iter()
                    .map(|q_step| new_matrix_witness_4d(cs.clone(), q_step))
                    .collect::<Vec<_>>();
                let r_inv_vars = inner.pseudo_inv.r_inv_remainders.iter()
                    .map(|r_step| new_matrix_witness_4d(cs.clone(), r_step))
                    .collect::<Vec<_>>();

                let z_zero = FpVar::zero();
                let z_inv = FpVar::Constant(inner.pseudo_inv.z_inv);

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
                        m: FpVar::Constant(inner.pseudo_inv.m_inv[i]),
                        m_prime: FpVar::Constant(inner.pseudo_inv.m_prime_inv[i]),
                        z_w: zw.clone(),
                        z_x: zx.clone(),
                        z_y: zy.clone(),
                        two_k: two_k_var.clone(),
                    });
                }

                // Pseudoinverse Step 1: P_k2 * Q_inv_k2 = Q_y1
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_inv0_lows = Vec::new();
                let mut h_inv0_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &sm2_vars.p[b][h],
                    &q_inv_var[b][h],
                    &q_y_vars[0][b][h],
                    &r_inv_vars[0][b][h],
                    &inner.pseudo_inv.r_inv_remainders[0][b][h],
                    &inv_params[0],
                    &mut h_inv0_lows,
                    &mut h_inv0_highs,
                    None,
                )?;
                global_inv0_lows.extend(h_inv0_lows); global_inv0_highs.extend(h_inv0_highs);

                // Pseudoinverse Step 2: Q_y1 * P_k2 = P_k2
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_inv1_lows = Vec::new();
                let mut h_inv1_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &q_y_vars[0][b][h],
                    &sm2_vars.p[b][h],
                    &q_y_vars[1][b][h],
                    &r_inv_vars[1][b][h],
                    &inner.pseudo_inv.r_inv_remainders[1][b][h],
                    &inv_params[1],
                    &mut h_inv1_lows,
                    &mut h_inv1_highs,
                    None,
                )?;
                global_inv1_lows.extend(h_inv1_lows); global_inv1_highs.extend(h_inv1_highs);

                // Pseudoinverse Step 2-1: Q_inv * P_k2 = Q_y3
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_inv2_lows = Vec::new();
                let mut h_inv2_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &q_inv_var[b][h],
                    &sm2_vars.p[b][h],
                    &q_y_vars[2][b][h],
                    &r_inv_vars[2][b][h],
                    &inner.pseudo_inv.r_inv_remainders[2][b][h],
                    &inv_params[2],
                    &mut h_inv2_lows,
                    &mut h_inv2_highs,
                    None,
                )?;
                global_inv2_lows.extend(h_inv2_lows); global_inv2_highs.extend(h_inv2_highs);

                // Pseudoinverse Step 2-2: Q_y3 * Q_inv = Q_y4
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_inv3_lows = Vec::new();
                let mut h_inv3_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &q_y_vars[2][b][h],
                    &q_inv_var[b][h],
                    &q_y_vars[3][b][h],
                    &r_inv_vars[3][b][h],
                    &inner.pseudo_inv.r_inv_remainders[3][b][h],
                    &inv_params[3],
                    &mut h_inv3_lows,
                    &mut h_inv3_highs,
                    None,
                )?;
                global_inv3_lows.extend(h_inv3_lows); global_inv3_highs.extend(h_inv3_highs);

                // temp: P_k3 * V
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_temp_lows = Vec::new();
                let mut h_temp_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &sm3_vars.p[b][h],
                    &v_proc[b][h],
                    &temp_data.q_sim[b][h],
                    &temp_data.r_matmul[b][h],
                    &inner.matmul_temp.r_matmul[b][h],
                    &params_temp,
                    &mut h_temp_lows,
                    &mut h_temp_highs,
                    None,
                )?;
                global_temp_lows.extend(h_temp_lows); global_temp_highs.extend(h_temp_highs);

                // mid: P_k1 * Q_inv_k2
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_mid_lows = Vec::new();
                let mut h_mid_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &sm1_vars.p[b][h],
                    &q_inv_var[b][h],
                    &mid_data.q_sim[b][h],
                    &mid_data.r_matmul[b][h],
                    &inner.matmul_mid.r_matmul[b][h],
                    &params_mid,
                    &mut h_mid_lows,
                    &mut h_mid_highs,
                    None,
                )?;
                global_mid_lows.extend(h_mid_lows); global_mid_highs.extend(h_mid_highs);

                // X: mid * temp
                // CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_x_lows = Vec::new();
                let mut h_x_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &mid_data.q_sim[b][h],
                    &temp_data.q_sim[b][h],
                    &x_data.q_sim[b][h],
                    &x_data.r_matmul[b][h],
                    &inner.matmul_x.r_matmul[b][h],
                    &params_x,
                    &mut h_x_lows,
                    &mut h_x_highs,
                    None,
                )?;
                global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);

                // === O Projection: O = attn_output @ W_o_h^T + b_o_h ===
                // Construct [S][H*D] input directly from x_data.q_sim
                {
                    let num_heads = x_data.q_sim[b].len();
                    let s_dim = x_data.q_sim[b][0].len();
                    let o_input: Vec<Vec<FpVar<F>>> = (0..s_dim).map(|s| {
                        (0..num_heads).flat_map(|hh| x_data.q_sim[b][hh][s].iter().cloned()).collect()
                    }).collect();

                    let d_dim = inner.proj_o.q_w_diff[h].len();
                    let hd_dim = inner.proj_o.q_w_diff[h][0].len();
                    let w_o_h_t: Vec<Vec<FpVar<F>>> = (0..hd_dim)
                        .map(|hd| (0..d_dim).map(|d| FpVar::Constant(inner.proj_o.q_w_diff[h][d][hd])).collect())
                        .collect();
                    let proj_o_params = ZenParams {
                        m: FpVar::Constant(inner.proj_o.m_proj),
                        m_prime: FpVar::Constant(inner.proj_o.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_attn_output),
                        z_x: FpVar::zero(),
                        z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    enforce_naive_matmul(
                        cs.clone(), &o_input, &w_o_h_t,
                        &proj_o_out[b][h], &proj_o_r[b][h],
                        &inner.proj_o.r_out_proj[b][h],
                        &proj_o_params,
                        &mut global_proj_o_lows, &mut global_proj_o_highs,
                        Some(&inner.proj_o.q_b_diff[h]),
                    )?;
                }
            }
        }

        {
            let mut final_vars = inner.committed_vars.borrow_mut();

            // Softmax index blocks (C=5)
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

            // Projection remainders
            final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
            final_vars.extend(global_proj_k_lows); final_vars.extend(global_proj_k_highs);
            final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
            final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);

            // Scaling remainders
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

            // 16-bit Remainders
            final_vars.extend(r2_pool);
            final_vars.extend(r3_pool);
        }

        // === Poseidon hash of output → public input ===
        {
            use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
            use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

            let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();

            let declared_hash = FpVar::new_input(cs.clone(), || Ok(inner.output_hash))?;

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

pub fn run_paths_not<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [Nystrom NOT] Processing: {:?} ===", file_path);

        // ── Dry-run: extract committed witness indices ───────────────────────
        let (d1_indices, d2_indices) = {
            let temp_circuit = load_nystrom_circuit::<E>(&file_path)?;
            let vars1_ptr = Rc::clone(&temp_circuit.committed_vars);
            let vars2_ptr = Rc::clone(&temp_circuit.committed_vars_2);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            NystromAttentionCircuitNot(temp_circuit).generate_constraints(cs_analyze)?;
            (
                crate::util::utils::extract_witness_indices(&vars1_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&vars2_ptr.borrow()),
            )
        };
        let num_commit_1 = d1_indices.len() as u32;
        let num_commit_2 = d2_indices.len() as u32;
        println!("Commit counts: d1={:?}, d2={:?}", num_commit_1, num_commit_2);

        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (d1={:?}, d2={:?}) ===", num_commit_1, num_commit_2);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            NystromAttentionCircuitNot(sample_circuit.clone()),
            &d1_indices,
            &d2_indices,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction + constraint count ────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        NystromAttentionCircuitNot(sample_circuit.clone()).generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

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

        println!("\n[Debug] committed_group1 total={}", committed_group1.len());
        let print_count = std::cmp::min(10, committed_group1.len());
        if !committed_group1.is_empty() {
            println!("  Head (10): {:?}", committed_group1[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  Tail (10): {:?}", committed_group1[committed_group1.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit ───────────────────────────────────────────────────
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        let r_fs = hash_cm_to_field::<E>(&cm1);

        // ── Compute Poseidon hash of proj_o_out (native) ─────────────────────
        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<Fr>();
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
        let final_inner = NystromAttentionCircuit {
            r_challenge: r_fs,
            output_hash,
            ..sample_circuit.clone()
        };

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            NystromAttentionCircuitNot(final_inner.clone()),
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
        let mut public_inputs = vec![final_inner.r_challenge];
        for batch in &final_inner.q_x_input {
            for seq in batch {
                public_inputs.extend(seq.iter().cloned());
            }
        }
        public_inputs.push(final_inner.output_hash);
        let groth16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let groth16_verify_time = groth16_verify_start.elapsed();

        println!("  - Nystrom NOT SNARK Verification: {:?}", result);
        if let Err(e) = result {
            println!("    Reason: {:?}", e);
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        // ── Log circuit output (proj_o_out) ──────────────────────────────────
        println!("\n=== Circuit Output (proj_o_out) ===");
        for (b, batch) in final_inner.proj_o.q_out_proj.iter().enumerate() {
            for (h, head) in batch.iter().enumerate() {
                println!("  [batch={}, head={}] shape=[{}][{}]", b, h, head.len(), head[0].len());
                for (s, seq) in head.iter().enumerate() {
                    let vals: Vec<_> = seq.iter()
                        .map(|v| v.into_bigint())
                        .collect();
                    println!("    seq[{}]: {:?}", s, vals);
                }
            }
        }
        println!("  output_hash = {:?}", final_inner.output_hash.into_bigint());

        // ── [2] Lasso (uses original circuit type — data fields identical) ───
        println!("=== [2] Nystrom NOT Lasso Lookup Proofs ===");
        proof_size::reset_lasso_aggregator();
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let (lasso_sm1, sm1_agg) = run_lasso_bundle(&final_inner.sm1, b"lasso_k1", &mut rng)?;
        lasso_agg.add_aggregate(&sm1_agg);
        let (lasso_sm2, sm2_agg) = run_lasso_bundle(&final_inner.sm2, b"lasso_k2", &mut rng)?;
        lasso_agg.add_aggregate(&sm2_agg);
        let (lasso_sm3, sm3_agg) = run_lasso_bundle(&final_inner.sm3, b"lasso_k3", &mut rng)?;
        lasso_agg.add_aggregate(&sm3_agg);
        let (lasso_remainders, rem_agg) = run_all_remainders_with_link::<Fr, G1>(&final_inner, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = vec![];
        lasso_results.extend(lasso_sm1);
        lasso_results.extend(lasso_sm2);
        lasso_results.extend(lasso_sm3);
        lasso_results.extend(lasso_remainders);

        // ── [3] CP-Link ──────────────────────────────────────────────────────
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

        let e2e_time = e2e_start.elapsed();

        // ── Groth16 proof size report ──
        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        cplink_report.print_log();
        proof_size::print_full_report(&g16_report, &cplink_report);

        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "attention_not".to_string(),
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


// 환경 변수 (모두 optional):
//   NAIVE_NYSTROM_ATTN_WITNESS_DIR : witness npz 디렉터리 (기본 "outputs_m64")
//   NAIVE_NYSTROM_ATTN_LAYER_IDX   : 처리할 layer index (기본 0)
//   SOFTMAX_TABLE_PATH             : B6-A1 setter — 미설정 시 witness_dir 사용
#[test]
fn nystrom_attn_naive_single_layer_test() {
    use std::env;

    let witness_dir = env::var("NAIVE_NYSTROM_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_m64".to_string());
    let layer_idx: Option<usize> = env::var("NAIVE_NYSTROM_ATTN_LAYER_IDX")
        .ok().and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] NAIVE_NYSTROM_ATTN_WITNESS_DIR={} not a directory.",
            witness_dir
        );
        return;
    }

    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
        println!(
            "[info] SOFTMAX_TABLE_PATH not set; using {}/softmax_tables.npz",
            witness_dir
        );
    }

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
            "[skip] no '*attention.self.witness.npz' files in {}.",
            witness_dir
        );
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!(
                "NAIVE_NYSTROM_ATTN_LAYER_IDX={} out of range (have {} layers).",
                idx,
                all_targets.len()
            );
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [Nystrom Attention NAIVE prove+verify] dir={} | files={} | mode={} ===",
        witness_dir,
        targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_paths_not::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!(
            "\n=== [Nystrom Attention NAIVE] ALL {} files PASS | total_wall_ms={} ===",
            targets.len(), total_ms,
        ),
        Err(e) => panic!(
            "[Nystrom Attention NAIVE] proof generate/verify failed after {}ms: {}",
            total_ms, e,
        ),
    }
}
