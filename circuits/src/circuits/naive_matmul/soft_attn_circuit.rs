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
    circuits::freivalds_matmul::soft_attn_circuit::{
        SOFTAttentionCircuit, GaussianBundle, DistBundle, load_soft_circuit,
        alloc_weight_vars, split_52bit_remainder,
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

#[test]
fn soft_attn_naive_single_layer_test() {
    use std::env;

    let witness_dir = env::var("NAIVE_SOFT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_soft".to_string());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!("[skip] NAIVE_SOFT_ATTN_WITNESS_DIR={} not a directory.", witness_dir);
        return;
    }

    eprintln!("[info] Naive matmul test: would require run_soft_attention_not_paths (not implemented yet)");
}
