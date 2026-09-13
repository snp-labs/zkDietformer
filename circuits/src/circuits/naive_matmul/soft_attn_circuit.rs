use std::{path::Path, str::FromStr, time::Instant};
use anyhow::Result;
use ark_ec::pairing::Pairing;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{SeedableRng, rngs::StdRng},
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
            pad_lasso_u_values, groth16_commitment_size,
        },
        freivalds_matmul::soft_attn_circuit::{
            SOFTAttentionCircuit, GaussianBundle, DistBundle, load_soft_circuit,
            alloc_weight_vars, split_52bit_remainder,
        },
        proof_size,
    },
    gadgets::{
        activation::gelu::split_remainder_and_push,
        matmul::{
            constraints::{MatrixVar, new_matrix_witness},
            quantized::{
                ZenParams, new_matrix_witness_4d, transpose_matrix, enforce_scaling,
                compute_circuit_landmarks, gen_challenge_vector,
            },
            naive_quantized::enforce_naive_matmul,
        },
    },
    lasso::subtables::gaussian::set_gaussian_table_dir,
    util::{
        soft_attn_lookup::{run_soft_lasso_bundle, run_all_soft_remainders_with_link, run_soft_mp_conditions_with_link},
        utils::{cm1_from_vk, cm2_from_vk, cm_aux_from_vk, generate_global_ck, commit_global, hash_commitments_to_field_v3},
    },
};

type G1 = G1Projective;

/// Newtype wrapper that uses naive O(n^3) matmul instead of Freivalds.
#[derive(Clone)]
pub struct SOFTAttentionCircuitNot<F: PrimeField>(pub SOFTAttentionCircuit<F>);

impl<F: PrimeField> ConstraintSynthesizer<F> for SOFTAttentionCircuitNot<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let inner = self.0;
        inner.committed_vars.borrow_mut().clear();
        inner.committed_vars_2.borrow_mut().clear();
        inner.committed_vars_aux.borrow_mut().clear();

        let _r_var = FpVar::new_input(cs.clone(), || Ok(inner.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << inner.k_bits))?;
        let two_2k = &two_k_var * &two_k_var;

        // Allocate intermediates
        let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.q_x_input.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        let proj_q_out = new_matrix_witness_4d(cs.clone(), &inner.proj_q.q_out_proj);
        let proj_q_r = new_matrix_witness_4d(cs.clone(), &inner.proj_q.r_out_proj);
        let proj_v_out = new_matrix_witness_4d(cs.clone(), &inner.proj_v.q_out_proj);
        let proj_v_r = new_matrix_witness_4d(cs.clone(), &inner.proj_v.r_out_proj);
        let scale_q_out = new_matrix_witness_4d(cs.clone(), &inner.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &inner.scale_q.r_scale);
        let proj_o_out = new_matrix_witness_4d(cs.clone(), &inner.proj_o.q_out_proj);
        let proj_o_r = new_matrix_witness_4d(cs.clone(), &inner.proj_o.r_out_proj);

        let num_heads = inner.proj_q.q_w_diff.len();
        let w_q_vars = alloc_weight_vars(cs.clone(), &inner.proj_q.q_w_diff);
        let w_v_vars = alloc_weight_vars(cs.clone(), &inner.proj_v.q_w_diff);
        let w_o_vars = alloc_weight_vars(cs.clone(), &inner.proj_o.q_w_diff);

        // d2: W_Q, W_V, W_O (3 sets, no W_K)
        {
            let mut vars_2 = inner.committed_vars_2.borrow_mut();
            for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        }

        let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.q_attn_output.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        // Cross-term + chain matmul witnesses
        let cross_b_data_q = new_matrix_witness_4d(cs.clone(), &inner.cross_b.q_sim);
        let cross_b_data_r = new_matrix_witness_4d(cs.clone(), &inner.cross_b.r_matmul);
        let cross_a_data_q = new_matrix_witness_4d(cs.clone(), &inner.cross_a.q_sim);
        let cross_a_data_r = new_matrix_witness_4d(cs.clone(), &inner.cross_a.r_matmul);

        let temp_data_q = new_matrix_witness_4d(cs.clone(), &inner.matmul_temp.q_sim);
        let temp_data_r = new_matrix_witness_4d(cs.clone(), &inner.matmul_temp.r_matmul);
        let mid_data_q = new_matrix_witness_4d(cs.clone(), &inner.matmul_mid.q_sim);
        let mid_data_r = new_matrix_witness_4d(cs.clone(), &inner.matmul_mid.r_matmul);
        let x_data_q = new_matrix_witness_4d(cs.clone(), &inner.matmul_x.q_sim);
        let x_data_r = new_matrix_witness_4d(cs.clone(), &inner.matmul_x.r_matmul);

        // Gaussian witnesses
        let gauss_b_kernel = new_matrix_witness_4d(cs.clone(), &inner.gauss_b.kernel);
        let gauss_b_table_idx = new_matrix_witness_4d(cs.clone(), &inner.gauss_b.table_idx);
        let gauss_b_r_gauss = new_matrix_witness_4d(cs.clone(), &inner.gauss_b.r_gauss);
        let gauss_a_kernel = new_matrix_witness_4d(cs.clone(), &inner.gauss_a.kernel);
        let gauss_a_table_idx = new_matrix_witness_4d(cs.clone(), &inner.gauss_a.table_idx);
        let gauss_a_r_gauss = new_matrix_witness_4d(cs.clone(), &inner.gauss_a.r_gauss);

        // Dist witnesses
        let q_dist_b = new_matrix_witness_4d(cs.clone(), &inner.dist_b.q_dist);
        let r_dist_b = new_matrix_witness_4d(cs.clone(), &inner.dist_b.r_dist);
        let q_dist_a = new_matrix_witness_4d(cs.clone(), &inner.dist_a.q_dist);
        let r_dist_a = new_matrix_witness_4d(cs.clone(), &inner.dist_a.r_dist);

        // Norm witnesses
        let norm_q_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.norm_q.iter().map(|b| {
            b.iter().map(|h| h.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()).collect()
        }).collect();
        let norm_qtilde_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.norm_qtilde.iter().map(|b| {
            b.iter().map(|h| h.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()).collect()
        }).collect();

        // Masked witnesses
        let q_proc_masked = new_matrix_witness_4d(cs.clone(), &inner.q_q_masked);
        let v_masked = new_matrix_witness_4d(cs.clone(), &inner.q_v_masked);

        let mask_var = new_matrix_witness(cs.clone(), &inner.mask);

        // Remainder pools
        let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
        let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
        let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_r_dist_b_lows = Vec::new(); let mut global_r_dist_b_highs = Vec::new();
        let mut global_r_dist_a_lows = Vec::new(); let mut global_r_dist_a_highs = Vec::new();
        let mut global_cross_b_lows = Vec::new(); let mut global_cross_b_highs = Vec::new();
        let mut global_cross_a_lows = Vec::new(); let mut global_cross_a_highs = Vec::new();
        let mut global_inv0_lows = Vec::new(); let mut global_inv0_highs = Vec::new();
        let mut global_inv1_lows = Vec::new(); let mut global_inv1_highs = Vec::new();
        let mut global_inv2_lows = Vec::new(); let mut global_inv2_highs = Vec::new();
        let mut global_inv3_lows = Vec::new(); let mut global_inv3_highs = Vec::new();
        let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
        let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
        let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();
        let mut global_gauss_b_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_gauss_a_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_r_gauss_b_limbs: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut global_r_gauss_a_limbs: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut mp_cond1_pool: Vec<FpVar<F>> = Vec::new();
        let mut mp_cond3_pool: Vec<FpVar<F>> = Vec::new();
        let mut mp_cond4_pool: Vec<FpVar<F>> = Vec::new();
        let mut q_inv_committed: Vec<FpVar<F>> = Vec::new();

        let z_q_var = FpVar::Constant(inner.z_q);
        let z_v_var = FpVar::Constant(inner.z_v);

        // Mask booleans
        let mask_bools: Vec<Vec<_>> = mask_var.iter().map(|b_mask| {
            b_mask.iter().map(|m| m.is_one().unwrap()).collect::<Vec<_>>()
        }).collect();

        // ZenParams for cross-terms
        let params_cross_b = ZenParams {
            m: FpVar::Constant(inner.cross_b.m),
            m_prime: FpVar::Constant(inner.cross_b.m_prime),
            z_w: FpVar::Constant(inner.z_q),
            z_x: FpVar::Constant(inner.z_q),
            z_y: FpVar::zero(),
            two_k: two_k_var.clone(),
        };
        let z_cross_b_var = FpVar::Constant(inner.cross_b.z_sim);
        let params_cross_a = ZenParams {
            m: FpVar::Constant(inner.cross_a.m),
            m_prime: FpVar::Constant(inner.cross_a.m_prime),
            z_w: FpVar::Constant(inner.z_q),
            z_x: FpVar::Constant(inner.z_q),
            z_y: FpVar::zero(),
            two_k: two_k_var.clone(),
        };
        let z_cross_a_var = FpVar::Constant(inner.cross_a.z_sim);

        // Dist equation constants
        let m_a_b_var = FpVar::Constant(inner.dist_b.m_a);
        let two_m_b_b = FpVar::Constant(inner.dist_b.m_b + inner.dist_b.m_b);
        let z_dist_two_k_b = FpVar::Constant(inner.dist_b.z_dist) * &two_k_var;
        let m_gauss_b_var = FpVar::Constant(inner.gauss_b.m_gauss);
        let m_a_a_var = FpVar::Constant(inner.dist_a.m_a);
        let two_m_b_a = FpVar::Constant(inner.dist_a.m_b + inner.dist_a.m_b);
        let z_dist_two_k_a = FpVar::Constant(inner.dist_a.z_dist) * &two_k_var;
        let m_gauss_a_var = FpVar::Constant(inner.gauss_a.m_gauss);

        // Chain matmul params
        let params_temp = ZenParams {
            m: FpVar::Constant(inner.matmul_temp.m), m_prime: FpVar::Constant(inner.matmul_temp.m_prime),
            z_w: FpVar::zero(), z_x: FpVar::Constant(inner.z_v),
            z_y: FpVar::zero(), two_k: two_k_var.clone(),
        };
        let params_mid = ZenParams {
            m: FpVar::Constant(inner.matmul_mid.m), m_prime: FpVar::Constant(inner.matmul_mid.m_prime),
            z_w: FpVar::zero(), z_x: FpVar::Constant(inner.pseudo_inv.z_inv),
            z_y: FpVar::zero(), two_k: two_k_var.clone(),
        };
        let params_x = ZenParams {
            m: FpVar::Constant(inner.matmul_x.m), m_prime: FpVar::Constant(inner.matmul_x.m_prime),
            z_w: FpVar::Constant(inner.matmul_mid.z_sim), z_x: FpVar::Constant(inner.matmul_temp.z_sim),
            z_y: FpVar::zero(), two_k: two_k_var.clone(),
        };

        let b_len = proj_q_out.len();

        for b in 0..b_len {
            // Masking enforcement
            let h_len = proj_q_out[b].len();
            for h in 0..h_len {
                let s_len = scale_q_out[b][h].len();
                let d_dim = scale_q_out[b][h][0].len();
                for s in 0..s_len {
                    for d in 0..d_dim {
                        let expected_q = FpVar::conditionally_select(
                            &mask_bools[b][s], &scale_q_out[b][h][s][d], &z_q_var,
                        )?;
                        q_proc_masked[b][h][s][d].enforce_equal(&expected_q)?;
                        let expected_v = FpVar::conditionally_select(
                            &mask_bools[b][s], &proj_v_out[b][h][s][d], &z_v_var,
                        )?;
                        v_masked[b][h][s][d].enforce_equal(&expected_v)?;
                    }
                }

                // Q Projection (naive)
                {
                    let proj_q_params = ZenParams {
                        m: FpVar::Constant(inner.proj_q.m_proj),
                        m_prime: FpVar::Constant(inner.proj_q.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_x_input),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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
                // V Projection (naive)
                {
                    let proj_v_params = ZenParams {
                        m: FpVar::Constant(inner.proj_v.m_proj),
                        m_prime: FpVar::Constant(inner.proj_v.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_x_input),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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

                // Landmarks
                let r_mean_q = new_matrix_witness_4d(cs.clone(), &inner.cross_b.r_mean);
                let q_landmarks = compute_circuit_landmarks(
                    cs.clone(), &q_proc_masked[b], &r_mean_q[b], inner.seg_size,
                )?;
                let q_l_t = transpose_matrix(&q_landmarks[h]);

                // Cross-term B (naive)
                enforce_naive_matmul(
                    cs.clone(), &q_proc_masked[b][h], &q_l_t,
                    &cross_b_data_q[b][h], &cross_b_data_r[b][h],
                    &inner.cross_b.r_matmul[b][h],
                    &params_cross_b,
                    &mut global_cross_b_lows, &mut global_cross_b_highs, None,
                )?;

                // Cross-term A (naive)
                enforce_naive_matmul(
                    cs.clone(), &q_landmarks[h], &q_l_t,
                    &cross_a_data_q[b][h], &cross_a_data_r[b][h],
                    &inner.cross_a.r_matmul[b][h],
                    &params_cross_a,
                    &mut global_cross_a_lows, &mut global_cross_a_highs, None,
                )?;

                // Squared norms (inline R1CS)
                // norm_q is [B][H][S], norm_qtilde is [B][H][m]
                // (norms constrained via R1CS multiplication, same as Freivalds version)

                // Dist equation B
                {
                    let s_len_dist = q_dist_b[b][h].len();
                    let m_dim_dist = q_dist_b[b][h][0].len();
                    for i in 0..s_len_dist {
                        for j in 0..m_dim_dist {
                            let lhs = &q_dist_b[b][h][i][j] * &two_k_var + &r_dist_b[b][h][i][j];
                            let n_sum = &norm_q_vars[b][h][i] + &norm_qtilde_vars[b][h][j];
                            let cross_shifted = &cross_b_data_q[b][h][i][j] - &z_cross_b_var;
                            let rhs = &m_a_b_var * &n_sum - &two_m_b_b * &cross_shifted + &z_dist_two_k_b;
                            lhs.enforce_equal(&rhs)?;
                            let (low, high) = split_remainder_and_push(cs.clone(), &r_dist_b[b][h][i][j], inner.dist_b.r_dist[b][h][i][j])?;
                            global_r_dist_b_lows.push(low);
                            global_r_dist_b_highs.push(high);
                        }
                    }
                }

                // Dist equation A
                {
                    let m_dim_a = q_dist_a[b][h].len();
                    let m_dim_a2 = q_dist_a[b][h][0].len();
                    for i in 0..m_dim_a {
                        for j in 0..m_dim_a2 {
                            let lhs = &q_dist_a[b][h][i][j] * &two_k_var + &r_dist_a[b][h][i][j];
                            let n_sum = &norm_qtilde_vars[b][h][i] + &norm_qtilde_vars[b][h][j];
                            let cross_shifted = &cross_a_data_q[b][h][i][j] - &z_cross_a_var;
                            let rhs = &m_a_a_var * &n_sum - &two_m_b_a * &cross_shifted + &z_dist_two_k_a;
                            lhs.enforce_equal(&rhs)?;
                            let (low, high) = split_remainder_and_push(cs.clone(), &r_dist_a[b][h][i][j], inner.dist_a.r_dist[b][h][i][j])?;
                            global_r_dist_a_lows.push(low);
                            global_r_dist_a_highs.push(high);
                        }
                    }
                }

                // Gaussian lookup B
                {
                    let sl = gauss_b_table_idx[b][h].len();
                    let ml = gauss_b_table_idx[b][h][0].len();
                    for i in 0..sl {
                        for j in 0..ml {
                            let x = &q_dist_b[b][h][i][j] * &two_k_var + &r_dist_b[b][h][i][j];
                            let lhs = &x * &m_gauss_b_var;
                            let rhs = &gauss_b_table_idx[b][h][i][j] * &two_2k + &gauss_b_r_gauss[b][h][i][j];
                            lhs.enforce_equal(&rhs)?;
                            global_gauss_b_idx.push(gauss_b_table_idx[b][h][i][j].clone());
                            split_52bit_remainder(cs.clone(), &gauss_b_r_gauss[b][h][i][j],
                                gauss_b_r_gauss[b][h][i][j].value().unwrap_or(F::zero()),
                                &mut global_r_gauss_b_limbs)?;
                        }
                    }
                }

                // Gaussian lookup A
                {
                    let ml = gauss_a_table_idx[b][h].len();
                    let ml2 = gauss_a_table_idx[b][h][0].len();
                    for i in 0..ml {
                        for j in 0..ml2 {
                            let x = &q_dist_a[b][h][i][j] * &two_k_var + &r_dist_a[b][h][i][j];
                            let lhs = &x * &m_gauss_a_var;
                            let rhs = &gauss_a_table_idx[b][h][i][j] * &two_2k + &gauss_a_r_gauss[b][h][i][j];
                            lhs.enforce_equal(&rhs)?;
                            global_gauss_a_idx.push(gauss_a_table_idx[b][h][i][j].clone());
                            split_52bit_remainder(cs.clone(), &gauss_a_r_gauss[b][h][i][j],
                                gauss_a_r_gauss[b][h][i][j].value().unwrap_or(F::zero()),
                                &mut global_r_gauss_a_limbs)?;
                        }
                    }
                }

                // MP Pseudoinverse
                let q_inv_var = new_matrix_witness_4d(cs.clone(), &inner.pseudo_inv.q_inv);
                for row in &q_inv_var[b][h] { q_inv_committed.extend(row.iter().cloned()); }
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
                        z_w: zw.clone(), z_x: zx.clone(), z_y: zy.clone(),
                        two_k: two_k_var.clone(),
                    });
                }

                // Step 0: kernel_A @ A+ = Y1
                enforce_naive_matmul(cs.clone(), &gauss_a_kernel[b][h], &q_inv_var[b][h],
                    &q_y_vars[0][b][h], &r_inv_vars[0][b][h], &inner.pseudo_inv.r_inv_remainders[0][b][h],
                    &inv_params[0], &mut global_inv0_lows, &mut global_inv0_highs, None)?;
                // Step 1: Y1 @ kernel_A = Y2
                enforce_naive_matmul(cs.clone(), &q_y_vars[0][b][h], &gauss_a_kernel[b][h],
                    &q_y_vars[1][b][h], &r_inv_vars[1][b][h], &inner.pseudo_inv.r_inv_remainders[1][b][h],
                    &inv_params[1], &mut global_inv1_lows, &mut global_inv1_highs, None)?;
                // Step 2: A+ @ kernel_A = Y3
                enforce_naive_matmul(cs.clone(), &q_inv_var[b][h], &gauss_a_kernel[b][h],
                    &q_y_vars[2][b][h], &r_inv_vars[2][b][h], &inner.pseudo_inv.r_inv_remainders[2][b][h],
                    &inv_params[2], &mut global_inv2_lows, &mut global_inv2_highs, None)?;
                // Step 3: Y3 @ A+ = Y4
                enforce_naive_matmul(cs.clone(), &q_y_vars[2][b][h], &q_inv_var[b][h],
                    &q_y_vars[3][b][h], &r_inv_vars[3][b][h], &inner.pseudo_inv.r_inv_remainders[3][b][h],
                    &inv_params[3], &mut global_inv3_lows, &mut global_inv3_highs, None)?;

                // MP bounded-diff conditions
                let m_dim_mp = q_y_vars[0][b][h].len();
                for i in 0..m_dim_mp {
                    for j in 0..q_y_vars[1][b][h][i].len() {
                        let shifted_expr = &q_y_vars[1][b][h][i][j] - &gauss_a_kernel[b][h][i][j]
                            + FpVar::Constant(F::from(inner.eps1 as u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond1_pool.push(shifted);
                    }
                }
                for i in 0..m_dim_mp {
                    for j in (i+1)..m_dim_mp {
                        let shifted_expr = &q_y_vars[0][b][h][i][j] - &q_y_vars[0][b][h][j][i]
                            + FpVar::Constant(F::from(inner.eps3 as u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond3_pool.push(shifted);
                    }
                }
                for i in 0..m_dim_mp {
                    for j in (i+1)..m_dim_mp {
                        let shifted_expr = &q_y_vars[2][b][h][i][j] - &q_y_vars[2][b][h][j][i]
                            + FpVar::Constant(F::from(inner.eps4 as u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond4_pool.push(shifted);
                    }
                }

                // Chain matmuls (naive)
                let kernel_b_t = transpose_matrix(&gauss_b_kernel[b][h]);
                enforce_naive_matmul(cs.clone(), &kernel_b_t, &v_masked[b][h],
                    &temp_data_q[b][h], &temp_data_r[b][h], &inner.matmul_temp.r_matmul[b][h],
                    &params_temp, &mut global_temp_lows, &mut global_temp_highs, None)?;
                enforce_naive_matmul(cs.clone(), &gauss_b_kernel[b][h], &q_inv_var[b][h],
                    &mid_data_q[b][h], &mid_data_r[b][h], &inner.matmul_mid.r_matmul[b][h],
                    &params_mid, &mut global_mid_lows, &mut global_mid_highs, None)?;
                enforce_naive_matmul(cs.clone(), &mid_data_q[b][h], &temp_data_q[b][h],
                    &x_data_q[b][h], &x_data_r[b][h], &inner.matmul_x.r_matmul[b][h],
                    &params_x, &mut global_x_lows, &mut global_x_highs, None)?;

                // O Projection (naive)
                {
                    let o_input = &attn_output_vars[b];
                    let proj_o_params = ZenParams {
                        m: FpVar::Constant(inner.proj_o.m_proj),
                        m_prime: FpVar::Constant(inner.proj_o.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_attn_output),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
                        two_k: two_k_var.clone(),
                    };
                    let d_dim_o = inner.proj_o.q_w_diff[h].len();
                    let hd_dim = inner.proj_o.q_w_diff[h][0].len();
                    let w_o_h_t: Vec<Vec<FpVar<F>>> = (0..hd_dim)
                        .map(|hd| (0..d_dim_o).map(|dd| FpVar::Constant(inner.proj_o.q_w_diff[h][dd][hd])).collect())
                        .collect();
                    enforce_naive_matmul(
                        cs.clone(), o_input, &w_o_h_t,
                        &proj_o_out[b][h], &proj_o_r[b][h],
                        &inner.proj_o.r_out_proj[b][h],
                        &proj_o_params,
                        &mut global_proj_o_lows, &mut global_proj_o_highs,
                        Some(&inner.proj_o.q_b_diff[h]),
                    )?;
                }
            }
        }

        // aux commit
        {
            let mut vars_aux = inner.committed_vars_aux.borrow_mut();
            vars_aux.extend(q_inv_committed);
        }

        // d1 committed_vars (same order as Freivalds version B9)
        {
            let mut final_vars = inner.committed_vars.borrow_mut();
            final_vars.extend(global_gauss_b_idx);
            final_vars.extend(global_gauss_a_idx);
            final_vars.extend(global_r_dist_b_lows); final_vars.extend(global_r_dist_b_highs);
            final_vars.extend(global_r_dist_a_lows); final_vars.extend(global_r_dist_a_highs);
            final_vars.extend(global_cross_b_lows); final_vars.extend(global_cross_b_highs);
            final_vars.extend(global_cross_a_lows); final_vars.extend(global_cross_a_highs);
            final_vars.extend(global_inv0_lows); final_vars.extend(global_inv0_highs);
            final_vars.extend(global_inv1_lows); final_vars.extend(global_inv1_highs);
            final_vars.extend(global_inv2_lows); final_vars.extend(global_inv2_highs);
            final_vars.extend(global_inv3_lows); final_vars.extend(global_inv3_highs);
            final_vars.extend(global_temp_lows); final_vars.extend(global_temp_highs);
            final_vars.extend(global_mid_lows); final_vars.extend(global_mid_highs);
            final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
            final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
            final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
            final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            for k in 0..4 { final_vars.extend(global_r_gauss_b_limbs[k].clone()); }
            for k in 0..4 { final_vars.extend(global_r_gauss_a_limbs[k].clone()); }
            final_vars.extend(mp_cond1_pool);
            final_vars.extend(mp_cond3_pool);
            final_vars.extend(mp_cond4_pool);
        }

        // Poseidon hash of x_input
        {
            use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
            use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;
            let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();
            let declared_input_hash = FpVar::new_input(cs.clone(), || Ok(inner.input_hash))?;
            let mut x_flat: Vec<FpVar<F>> = Vec::new();
            for batch in &x_input_vars { for seq in batch { x_flat.extend(seq.iter().cloned()); } }
            let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
            sponge_var.absorb(&x_flat)?;
            let computed_hash = sponge_var.squeeze_field_elements(1)?;
            computed_hash[0].enforce_equal(&declared_input_hash)?;
        }

        // Poseidon hash of output
        {
            use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
            use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;
            let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();
            let declared_hash = FpVar::new_input(cs.clone(), || Ok(inner.output_hash))?;
            let mut output_flat: Vec<FpVar<F>> = Vec::new();
            for batch in &proj_o_out { for head in batch { for seq in head { output_flat.extend(seq.iter().cloned()); } } }
            let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
            sponge_var.absorb(&output_flat)?;
            let computed_hash = sponge_var.squeeze_field_elements(1)?;
            computed_hash[0].enforce_equal(&declared_hash)?;
        }

        Ok(())
    }
}

// === Naive matmul proving pipeline (Groth16 + Lasso + CP-Link) ===

pub fn run_soft_attention_not_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    // Set Gaussian table directory from the first witness file's parent dir
    if let Some(first) = targets.first() {
        if let Some(dir) = first.parent() {
            set_gaussian_table_dir(dir);
        }
    }

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [SOFT NOT] Processing: {:?} ===", file_path);

        // Dry-run: extract committed witness indices (3 groups: d1, d2, d3/aux)
        let (d1_indices, d2_indices, d3_indices) = {
            let temp_circuit = load_soft_circuit::<E>(file_path)?;
            let vars1_ptr = Rc::clone(&temp_circuit.committed_vars);
            let vars2_ptr = Rc::clone(&temp_circuit.committed_vars_2);
            let vars_aux_ptr = Rc::clone(&temp_circuit.committed_vars_aux);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            SOFTAttentionCircuitNot(temp_circuit).generate_constraints(cs_analyze)?;
            (
                crate::util::utils::extract_witness_indices(&vars1_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&vars2_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&vars_aux_ptr.borrow()),
            )
        };
        let num_commit_1 = d1_indices.len() as u32;
        let num_commit_2 = d2_indices.len() as u32;
        let num_commit_3 = d3_indices.len() as u32;
        println!("Commit counts: d1={}, d2={}, d3={}", num_commit_1, num_commit_2, num_commit_3);

        let sample_circuit = load_soft_circuit::<E>(file_path)?;

        // [0] Setup
        println!("=== [0] LegoGroth16 Setup (d1={}, d2={}, d3={}) ===", num_commit_1, num_commit_2, num_commit_3);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            SOFTAttentionCircuitNot(sample_circuit.clone()),
            &d1_indices, &d2_indices, &d3_indices,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();
        let pvk = prepare_verifying_key::<E>(&params.vk);

        // Witness extraction + constraint count
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        SOFTAttentionCircuitNot(sample_circuit.clone()).generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> = vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
            assert_eq!(values.len(), num_commit_1 as usize, "d1 size mismatch");
            values
        };
        let committed_group2 = {
            let vars = sample_circuit.committed_vars_2.borrow();
            let values: Vec<Fr> = vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
            assert_eq!(values.len(), num_commit_2 as usize, "d2 size mismatch");
            values
        };
        let committed_aux = {
            let vars = sample_circuit.committed_vars_aux.borrow();
            vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect::<Vec<Fr>>()
        };

        println!("d1={}, d2={}, aux={}", committed_group1.len(), committed_group2.len(), committed_aux.len());

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let v3 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // Groth16 commit (cm1, cm2, cm_aux)
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

        // Poseidon hashes
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

        // Multi-commitment Fiat-Shamir challenge (v3: includes cm_aux)
        let r_fs = hash_commitments_to_field_v3::<E>(&c_w, &cm1, &cm2, &cm_aux, &input_hash, &output_hash);

        println!("=== [1] Building final circuit with Fiat-Shamir challenge ===");
        let final_inner = SOFTAttentionCircuit {
            r_challenge: r_fs,
            input_hash,
            output_hash,
            ..sample_circuit.clone()
        };

        // R1CS satisfaction check
        {
            let cs_check = ConstraintSystem::<Fr>::new_ref();
            SOFTAttentionCircuitNot(final_inner.clone()).generate_constraints(cs_check.clone())?;
            let is_sat = cs_check.is_satisfied().unwrap_or(false);
            if !is_sat {
                let unsat = cs_check.which_is_unsatisfied().unwrap();
                eprintln!("R1CS not satisfied: {:?}", unsat);
                return Err(anyhow::anyhow!("R1CS not satisfied (final_circuit)"));
            }
            println!("  R1CS satisfied: {} constraints", cs_check.num_constraints());
        }

        // [1] Groth16 Prove
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            SOFTAttentionCircuitNot(final_inner.clone()), v1, v2, v3, &params, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;
        let groth16_prove_time = groth16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // [1] Groth16 Verify
        let public_inputs = vec![final_inner.r_challenge, final_inner.input_hash, final_inner.output_hash];
        let groth16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let groth16_verify_time = groth16_verify_start.elapsed();

        println!("  - SOFT NOT SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!("Groth16 verify failed: {:?}", e));
        }
        println!("  output_hash = {:?}", final_inner.output_hash.into_bigint());

        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // [2] Lasso (Gaussian bundles + remainders + MP conditions)
        proof_size::reset_lasso_aggregator();
        println!("=== [2] SOFT NOT Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();

        let (lasso_gauss_b, gb_agg) = run_soft_lasso_bundle(&final_inner.gauss_b.table_idx, b"soft_gauss_b", &mut rng)?;
        lasso_agg.add_aggregate(&gb_agg);
        let (lasso_gauss_a, ga_agg) = run_soft_lasso_bundle(&final_inner.gauss_a.table_idx, b"soft_gauss_a", &mut rng)?;
        lasso_agg.add_aggregate(&ga_agg);
        let (lasso_remainders, rem_agg) = run_all_soft_remainders_with_link(&final_inner, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let (mp_results, mp_agg) = run_soft_mp_conditions_with_link(&final_inner, &mut rng)?;
        lasso_agg.add_aggregate(&mp_agg);

        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = Vec::new();
        lasso_results.extend(lasso_gauss_b);
        lasso_results.extend(lasso_gauss_a);
        lasso_results.extend(lasso_remainders);
        lasso_results.extend(mp_results);

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
            circuit: "soft_attention_not".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: g16_commit_ms,
                cm1_ms,
                cm2_ms,
                cm_aux_ms,
                prove_ms: groth16_prove_time.as_millis(),
                verify_ms: groth16_verify_time.as_millis(),
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


#[test]
fn soft_attn_naive_single_layer_test() {
    use std::env;

    let witness_dir = env::var("NAIVE_SOFT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_soft".to_string());
    let layer_idx: Option<usize> = env::var("NAIVE_SOFT_ATTN_LAYER_IDX")
        .ok().and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!("[skip] NAIVE_SOFT_ATTN_WITNESS_DIR={} not a directory.", witness_dir);
        return;
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
        eprintln!("[skip] no '*attention_self.witness.npz' files in {}.", witness_dir);
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!("NAIVE_SOFT_ATTN_LAYER_IDX={} out of range.", idx);
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!("\n=== [SOFT Attention NAIVE prove+verify] dir={} | files={} ===", witness_dir, targets.len());

    let total_start = std::time::Instant::now();
    let res = run_soft_attention_not_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!("\n=== [SOFT Attention NAIVE] ALL {} files PASS | total_wall_ms={} ===", targets.len(), total_ms),
        Err(e) => panic!("[SOFT Attention NAIVE] proof failed after {}ms: {}", total_ms, e),
    }
}
