use std::{fs::File, ops::{AddAssign, Mul}, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{IxDyn, Ix0, Ix2, Ix3, Ix4};
use ndarray_npy::NpzReader;
use ark_std::{rc::Rc, fs, UniformRand, marker::PhantomData, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}, cell::RefCell};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{R1CSVar, alloc::AllocVar, eq::EqGadget, fields::{fp::FpVar, FieldVar}};
use ark_r1cs_std::select::CondSelectGadget;
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size,
        },
        proof_size,
    },
    config::{FRACBITS_C, FRACBITS_M, KBITS_C, KBITS_M, SM_KBITS_C, SM_KBITS_M},
    gadgets::{
        matmul::{
            constraints::{MatrixVar, new_matrix_witness},
            quantized::{MatmulBundle, MatmulData, PseudoinverseBundle, ScalingBundle, ZenParams, compute_circuit_landmarks, enforce_freivalds, enforce_scaling, gen_challenge_vector, new_matrix_witness_4d, transpose_matrix},
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
        load::{to_field_2d_i64, to_field_4d_i64, to_field_4d_u64, to_field_5d_i64},
        nystrom_attn_lookup::{run_lasso_bundle, prove_lasso_for_range_22, prove_lasso_for_range_52, prove_mp_range_13, prove_mp_range_10, prove_mp_range_7, prove_mp_range_6, prove_mp_range_3},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;
use merlin::Transcript;

type G1 = G1Projective;
type LassoResult = (PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize);

/// Nystrom attention circuit proving ONLY the core attention equation.
/// Excludes: input projections (XW_Q/K/V), output projection (XW_O),
/// weight matrices, Poseidon hashing, global weight commitment (c_W),
/// CP-Link #2, d2 group.
#[derive(Clone)]
pub struct NystromAttentionOnlyeqCircuit<F: PrimeField> {
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,

    /// Raw Q [B][H][S][D] (before scaling) — loaded from Q_raw_proj_q.npy
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,
    /// Raw K [B][H][S][D] (before scaling) — loaded from Q_raw_proj_k.npy
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,
    /// V [B][H][S][D] — loaded from Q_raw_proj_v.npy
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,
    pub z_raw_q: F,
    pub z_raw_k: F,

    pub scale_q: ScalingBundle<F>,
    pub scale_k: ScalingBundle<F>,

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

    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionOnlyeqCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        let mut r_div_k1_lows = Vec::new();
        let mut r_div_k1_highs = Vec::new();
        let mut r_div_k2_lows = Vec::new();
        let mut r_div_k2_highs = Vec::new();
        let mut r_div_k3_lows = Vec::new();
        let mut r_div_k3_highs = Vec::new();

        let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &self.matmul1.r_mean);
        let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &self.matmul2.r_mean);

        let (k1_data, params_k1) = self.matmul1.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (k2_data, params_k2) = self.matmul2.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (k3_data, params_k3) = self.matmul3.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm1_vars = self.sm1.alloc_vars(cs.clone())?;
        let sm2_vars = self.sm2.alloc_vars(cs.clone())?;
        let sm3_vars = self.sm3.alloc_vars(cs.clone())?;

        // Raw Q/K witnesses (replaces proj_q_out, proj_k_out)
        let q_q_vars = new_matrix_witness_4d(cs.clone(), &self.q_q);
        let q_k_vars = new_matrix_witness_4d(cs.clone(), &self.q_k);
        // V witness (replaces proj_v_out)
        let q_v_vars = new_matrix_witness_4d(cs.clone(), &self.q_v);

        // Scaling witnesses [B][H][S][D]
        let scale_q_out = new_matrix_witness_4d(cs.clone(), &self.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &self.scale_q.r_scale);
        let scale_k_out = new_matrix_witness_4d(cs.clone(), &self.scale_k.q_scaled);
        let scale_k_r = new_matrix_witness_4d(cs.clone(), &self.scale_k.r_scale);

        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

        let (temp_data, params_temp) = self.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);
        let (mid_data, params_mid) = self.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), self.pseudo_inv.z_inv, &two_k_var);
        let (x_data, params_x) = self.matmul_x.alloc_and_setup(cs.clone(), self.matmul_mid.z_sim, self.matmul_temp.z_sim, &two_k_var);

        let mut global_k1_lows = Vec::new(); let mut global_k1_highs = Vec::new();
        let mut global_k2_lows = Vec::new(); let mut global_k2_highs = Vec::new();
        let mut global_k3_lows = Vec::new(); let mut global_k3_highs = Vec::new();

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

        let mut mp_cond1_pool: Vec<FpVar<F>> = Vec::new();
        let mut mp_cond3_pool: Vec<FpVar<F>> = Vec::new();
        let mut mp_cond4_pool: Vec<FpVar<F>> = Vec::new();

        let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
        let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
        let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();

        for b in 0..scale_q_out.len() {
            let k_landmarks = compute_circuit_landmarks(cs.clone(), &scale_k_out[b], &r_mean_k1[b], self.seg_size)?;
            let q_landmarks = compute_circuit_landmarks(cs.clone(), &scale_q_out[b], &r_mean_k2[b], self.seg_size)?;

            for h in 0..scale_q_out[b].len() {
                // Q Scaling: raw_Q → scaled_Q
                {
                    let z_raw_q = FpVar::Constant(self.z_raw_q);
                    let m_scale_q = FpVar::Constant(self.scale_q.m_scale);
                    let m_prime_q = FpVar::Constant(self.scale_q.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &q_q_vars[b][h], &scale_q_out[b][h], &scale_q_r[b][h],
                        &self.scale_q.r_scale[b][h],
                        &z_raw_q, &m_scale_q, &m_prime_q, &two_k_var,
                        &mut global_scale_q_lows, &mut global_scale_q_highs,
                    )?;
                }

                // K Scaling: raw_K → scaled_K
                {
                    let z_raw_k = FpVar::Constant(self.z_raw_k);
                    let m_scale_k = FpVar::Constant(self.scale_k.m_scale);
                    let m_prime_k = FpVar::Constant(self.scale_k.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &q_k_vars[b][h], &scale_k_out[b][h], &scale_k_r[b][h],
                        &self.scale_k.r_scale[b][h],
                        &z_raw_k, &m_scale_k, &m_prime_k, &two_k_var,
                        &mut global_scale_k_lows, &mut global_scale_k_highs,
                    )?;
                }

                // Kernel 1
                let k_l_t = transpose_matrix(&k_landmarks[h]);
                let r_pows_k1 = gen_challenge_vector(&r_var, k1_data.q_sim[b][h][0].len());
                let mut h_k1_lows = Vec::new();
                let mut h_k1_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &scale_q_out[b][h], &k_l_t,
                    &k1_data.q_sim[b][h], &k1_data.r_matmul[b][h],
                    &self.matmul1.r_matmul[b][h],
                    &params_k1, &r_pows_k1,
                    &mut h_k1_lows, &mut h_k1_highs,
                    None,
                )?;
                global_k1_lows.extend(h_k1_lows); global_k1_highs.extend(h_k1_highs);

                // Kernel 2
                let q_l_h = &q_landmarks[h];
                let r_pows_k2 = gen_challenge_vector(&r_var, k2_data.q_sim[b][h][0].len());
                let mut h_k2_lows = Vec::new();
                let mut h_k2_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    q_l_h, &k_l_t,
                    &k2_data.q_sim[b][h], &k2_data.r_matmul[b][h],
                    &self.matmul2.r_matmul[b][h],
                    &params_k2, &r_pows_k2,
                    &mut h_k2_lows, &mut h_k2_highs,
                    None,
                )?;
                global_k2_lows.extend(h_k2_lows); global_k2_highs.extend(h_k2_highs);

                // Kernel 3
                let k_full_t = transpose_matrix(&scale_k_out[b][h]);
                let r_pows_k3 = gen_challenge_vector(&r_var, k3_data.q_sim[b][h][0].len());
                let mut h_k3_lows = Vec::new();
                let mut h_k3_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &q_landmarks[h], &k_full_t,
                    &k3_data.q_sim[b][h], &k3_data.r_matmul[b][h],
                    &self.matmul3.r_matmul[b][h],
                    &params_k3, &r_pows_k3,
                    &mut h_k3_lows, &mut h_k3_highs,
                    None,
                )?;
                global_k3_lows.extend(h_k3_lows); global_k3_highs.extend(h_k3_highs);

                // Softmax 1
                enforce_softmax(
                    cs.clone(),
                    &k1_data.q_sim[b][h], &k1_data.r_matmul[b][h],
                    &sm1_vars.r_sm[b][h], &sm1_vars.m_sm,
                    &sm1_vars.table_idx[b][h], &sm1_vars.exp_total[b][h],
                    &sm1_vars.p[b][h], &sm1_vars.r_div[b][h],
                    &self.sm1.r_div[b][h],
                    self.frac_bits, &two_k_var,
                    &mut global_sm1_idx, &mut global_sm1_r_sm,
                    &mut r_div_k1_lows, &mut r_div_k1_highs,
                )?;

                // Softmax 2
                enforce_softmax(
                    cs.clone(),
                    &k2_data.q_sim[b][h], &k2_data.r_matmul[b][h],
                    &sm2_vars.r_sm[b][h], &sm2_vars.m_sm,
                    &sm2_vars.table_idx[b][h], &sm2_vars.exp_total[b][h],
                    &sm2_vars.p[b][h], &sm2_vars.r_div[b][h],
                    &self.sm2.r_div[b][h],
                    self.frac_bits, &two_k_var,
                    &mut global_sm2_idx, &mut global_sm2_r_sm,
                    &mut r_div_k2_lows, &mut r_div_k2_highs,
                )?;

                // Softmax 3
                enforce_softmax(
                    cs.clone(),
                    &k3_data.q_sim[b][h], &k3_data.r_matmul[b][h],
                    &sm3_vars.r_sm[b][h], &sm3_vars.m_sm,
                    &sm3_vars.table_idx[b][h], &sm3_vars.exp_total[b][h],
                    &sm3_vars.p[b][h], &sm3_vars.r_div[b][h],
                    &self.sm3.r_div[b][h],
                    self.frac_bits, &two_k_var,
                    &mut global_sm3_idx, &mut global_sm3_r_sm,
                    &mut r_div_k3_lows, &mut r_div_k3_highs,
                )?;

                // v3 mask: verify exp_total=0 at padding positions
                {
                    let zero_var = FpVar::zero();
                    let num_landmarks = sm3_vars.exp_total[b][h].len();
                    let seq_len = sm3_vars.exp_total[b][h][0].len();
                    for l in 0..num_landmarks {
                        for s in 0..seq_len {
                            let is_valid = mask_var[b][s].is_one()?;
                            let should_be_zero = FpVar::conditionally_select(
                                &is_valid,
                                &zero_var,
                                &sm3_vars.exp_total[b][h][l][s],
                            )?;
                            should_be_zero.enforce_equal(&zero_var)?;
                        }
                    }
                }

                // Moore-Penrose pseudo-inverse
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
                        z_w: zw.clone(), z_x: zx.clone(), z_y: zy.clone(),
                        two_k: two_k_var.clone(),
                    });
                }

                let r_pows_inv = gen_challenge_vector(&r_var, q_inv_var[b][h].len());

                // Step 1: P_k2 * Q_inv_k2 = Q_y1
                let mut h_inv0_lows = Vec::new(); let mut h_inv0_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(), &sm2_vars.p[b][h], &q_inv_var[b][h],
                    &q_y_vars[0][b][h], &r_inv_vars[0][b][h],
                    &self.pseudo_inv.r_inv_remainders[0][b][h],
                    &inv_params[0], &r_pows_inv,
                    &mut h_inv0_lows, &mut h_inv0_highs, None,
                )?;
                global_inv0_lows.extend(h_inv0_lows); global_inv0_highs.extend(h_inv0_highs);

                // Step 2: Q_y1 * P_k2 = Q_y2
                let mut h_inv1_lows = Vec::new(); let mut h_inv1_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(), &q_y_vars[0][b][h], &sm2_vars.p[b][h],
                    &q_y_vars[1][b][h], &r_inv_vars[1][b][h],
                    &self.pseudo_inv.r_inv_remainders[1][b][h],
                    &inv_params[1], &r_pows_inv,
                    &mut h_inv1_lows, &mut h_inv1_highs, None,
                )?;
                global_inv1_lows.extend(h_inv1_lows); global_inv1_highs.extend(h_inv1_highs);

                // Step 2-1: Q_inv * P_k2 = Q_y3
                let mut h_inv2_lows = Vec::new(); let mut h_inv2_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(), &q_inv_var[b][h], &sm2_vars.p[b][h],
                    &q_y_vars[2][b][h], &r_inv_vars[2][b][h],
                    &self.pseudo_inv.r_inv_remainders[2][b][h],
                    &inv_params[2], &r_pows_inv,
                    &mut h_inv2_lows, &mut h_inv2_highs, None,
                )?;
                global_inv2_lows.extend(h_inv2_lows); global_inv2_highs.extend(h_inv2_highs);

                // Step 2-2: Q_y3 * Q_inv = Q_y4
                let mut h_inv3_lows = Vec::new(); let mut h_inv3_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(), &q_y_vars[2][b][h], &q_inv_var[b][h],
                    &q_y_vars[3][b][h], &r_inv_vars[3][b][h],
                    &self.pseudo_inv.r_inv_remainders[3][b][h],
                    &inv_params[3], &r_pows_inv,
                    &mut h_inv3_lows, &mut h_inv3_highs, None,
                )?;
                global_inv3_lows.extend(h_inv3_lows); global_inv3_highs.extend(h_inv3_highs);

                // MP bounded-difference conditions
                let m_dim = q_y_vars[0][b][h].len();

                // Condition 1: |Q_y2 - P_k2| < 4096
                for i in 0..m_dim {
                    for j in 0..q_y_vars[1][b][h][i].len() {
                        let shifted_expr = &q_y_vars[1][b][h][i][j] - &sm2_vars.p[b][h][i][j]
                            + FpVar::Constant(F::from(4096u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond1_pool.push(shifted);
                    }
                }

                // Condition 3: |Q_y1[i][j] - Q_y1[j][i]| < 512
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted_expr = &q_y_vars[0][b][h][i][j] - &q_y_vars[0][b][h][j][i]
                            + FpVar::Constant(F::from(512u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond3_pool.push(shifted);
                    }
                }

                // Condition 4: |Q_y3[i][j] - Q_y3[j][i]| < 32
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted_expr = &q_y_vars[2][b][h][i][j] - &q_y_vars[2][b][h][j][i]
                            + FpVar::Constant(F::from(32u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond4_pool.push(shifted);
                    }
                }

                // matmul_temp: sm3.p @ V
                let mut h_temp_lows = Vec::new(); let mut h_temp_highs = Vec::new();
                let r_pows_temp = gen_challenge_vector(&r_var, temp_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &sm3_vars.p[b][h], &q_v_vars[b][h],
                    &temp_data.q_sim[b][h], &temp_data.r_matmul[b][h],
                    &self.matmul_temp.r_matmul[b][h],
                    &params_temp, &r_pows_temp,
                    &mut h_temp_lows, &mut h_temp_highs, None,
                )?;
                global_temp_lows.extend(h_temp_lows); global_temp_highs.extend(h_temp_highs);

                // matmul_mid: sm1.p @ Q_inv
                let mut h_mid_lows = Vec::new(); let mut h_mid_highs = Vec::new();
                let r_pows_mid = gen_challenge_vector(&r_var, mid_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &sm1_vars.p[b][h], &q_inv_var[b][h],
                    &mid_data.q_sim[b][h], &mid_data.r_matmul[b][h],
                    &self.matmul_mid.r_matmul[b][h],
                    &params_mid, &r_pows_mid,
                    &mut h_mid_lows, &mut h_mid_highs, None,
                )?;
                global_mid_lows.extend(h_mid_lows); global_mid_highs.extend(h_mid_highs);

                // matmul_x: mid @ temp
                let mut h_x_lows = Vec::new(); let mut h_x_highs = Vec::new();
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &mid_data.q_sim[b][h], &temp_data.q_sim[b][h],
                    &x_data.q_sim[b][h], &x_data.r_matmul[b][h],
                    &self.matmul_x.r_matmul[b][h],
                    &params_x, &r_pows_x,
                    &mut h_x_lows, &mut h_x_highs, None,
                )?;
                global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);
            }
        }

        // Populate d1 committed_vars (no projection remainders, no Poseidon hash)
        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // Softmax index pools (v3: C=1, single pool)
            final_vars.extend(global_sm1_idx);
            final_vars.extend(global_sm2_idx);
            final_vars.extend(global_sm3_idx);

            // 26-bit remainders (C=2: 16+10) — NO projection remainders
            final_vars.extend(global_k1_lows); final_vars.extend(global_k1_highs);
            final_vars.extend(global_k2_lows); final_vars.extend(global_k2_highs);
            final_vars.extend(global_k3_lows); final_vars.extend(global_k3_highs);
            final_vars.extend(global_inv0_lows); final_vars.extend(global_inv0_highs);
            final_vars.extend(global_inv1_lows); final_vars.extend(global_inv1_highs);
            final_vars.extend(global_inv2_lows); final_vars.extend(global_inv2_highs);
            final_vars.extend(global_inv3_lows); final_vars.extend(global_inv3_highs);
            final_vars.extend(global_temp_lows); final_vars.extend(global_temp_highs);
            final_vars.extend(global_mid_lows); final_vars.extend(global_mid_highs);
            final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
            // Scaling remainders (KEPT)
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

            // 52-bit r_sm (C=4: 16+16+16+4)
            for k in 0..4 { final_vars.extend(global_sm1_r_sm[k].clone()); }
            for k in 0..4 { final_vars.extend(global_sm2_r_sm[k].clone()); }
            for k in 0..4 { final_vars.extend(global_sm3_r_sm[k].clone()); }

            // R_div per-softmax (26-bit C=2: 16+10)
            final_vars.extend(r_div_k1_lows);
            final_vars.extend(r_div_k1_highs);
            final_vars.extend(r_div_k2_lows);
            final_vars.extend(r_div_k2_highs);
            final_vars.extend(r_div_k3_lows);
            final_vars.extend(r_div_k3_highs);

            // Moore-Penrose bounded-difference pools (C=1)
            final_vars.extend(mp_cond1_pool);
            final_vars.extend(mp_cond3_pool);
            final_vars.extend(mp_cond4_pool);
        }

        Ok(())
    }
}

pub fn load_nystrom_onlyeq_circuit<E: Pairing>(path: &Path) -> Result<NystromAttentionOnlyeqCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let z_q = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar();
    let z_k = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar();
    let z_v = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar();
    let seg = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("seg_size.npy")?.into_scalar() as usize;

    // Raw Q/K/V (projection outputs, before scaling for Q/K)
    let q_q = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_raw_proj_q.npy")?.into_dimensionality()?);
    let q_k = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_raw_proj_k.npy")?.into_dimensionality()?);
    let q_v = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_raw_proj_v.npy")?.into_dimensionality()?);
    let z_raw_q = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_raw_q.npy")?.into_scalar());
    let z_raw_k = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_raw_k.npy")?.into_scalar());

    let load_matmul = |npz: &mut NpzReader<File>, suffix: &str, has_rmean: bool| -> Result<MatmulBundle<E::ScalarField>> {
        Ok(MatmulBundle {
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
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    // Pseudo-inverse
    let q_inv_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_inv_k2.npy")?.into_dimensionality()?;
    let z_inv_k2 = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_inv_k2.npy")?.into_scalar();
    let m_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_inv_all.npy")?;
    let m_prime_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_prime_inv_all.npy")?;
    let q_y_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("Q_y_inv_intermediates.npy")?;
    let r_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("R_inv_remainders.npy")?;

    let pseudo_inv = PseudoinverseBundle {
        q_inv: to_field_4d_i64(&q_inv_k2),
        z_inv: E::ScalarField::from(z_inv_k2),
        m_inv: [
            E::ScalarField::from(m_inv_all[0]), E::ScalarField::from(m_inv_all[1]),
            E::ScalarField::from(m_inv_all[2]), E::ScalarField::from(m_inv_all[3]),
        ],
        m_prime_inv: [
            E::ScalarField::from(m_prime_inv_all[0]), E::ScalarField::from(m_prime_inv_all[1]),
            E::ScalarField::from(m_prime_inv_all[2]), E::ScalarField::from(m_prime_inv_all[3]),
        ],
        q_y_intermediates: to_field_5d_i64(&q_y_inv_raw),
        r_inv_remainders: to_field_5d_i64(&r_inv_raw),
    };

    let matmul_temp = load_matmul(&mut npz, "temp", false)?;
    let matmul_mid = load_matmul(&mut npz, "mid", false)?;
    let matmul_x = load_matmul(&mut npz, "x", false)?;

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

    Ok(NystromAttentionOnlyeqCircuit {
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),
        z_v: E::ScalarField::from(z_v),
        q_q, q_k, q_v,
        z_raw_q, z_raw_k,
        scale_q, scale_k,
        matmul1, matmul2, matmul3,
        sm1, sm2, sm3,
        pseudo_inv,
        matmul_temp, matmul_mid, matmul_x,
        frac_bits,
        r_challenge: E::ScalarField::zero(),
        seg_size: seg,
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })
}

// ── Lasso remainder functions (inline, referencing onlyeq circuit) ───────────

fn run_onlyeq_remainders_with_link(
    circuit: &NystromAttentionOnlyeqCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] Onlyeq: Verifying All 26-bit Remainders ===");
    let mut all_results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    // 26-bit range checks — NO projection remainders
    let mut target_26bit: Vec<(&Vec<Vec<Vec<Vec<Fr>>>>, &str)> = Vec::new();
    target_26bit.push((&circuit.matmul1.r_matmul, "matmul_k1_remainder"));
    target_26bit.push((&circuit.matmul2.r_matmul, "matmul_k2_remainder"));
    target_26bit.push((&circuit.matmul3.r_matmul, "matmul_k3_remainder"));
    for (i, r_inv) in circuit.pseudo_inv.r_inv_remainders.iter().enumerate() {
        let label: &'static str = Box::leak(format!("pseudoinverse_step{}_remainder", i).into_boxed_str());
        target_26bit.push((r_inv, label));
    }
    target_26bit.push((&circuit.matmul_temp.r_matmul, "matmul_v_remainder"));
    target_26bit.push((&circuit.matmul_mid.r_matmul, "matmul_mid_remainder"));
    target_26bit.push((&circuit.matmul_x.r_matmul, "matmul_x_remainder"));
    target_26bit.push((&circuit.scale_q.r_scale, "scale_q_remainder"));
    target_26bit.push((&circuit.scale_k.r_scale, "scale_k_remainder"));

    for (r_matrix, label) in &target_26bit {
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
                        nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3FF) as usize]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), KBITS_C,
            total_sparsity.next_power_of_two(), KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    // 52-bit r_sm range checks
    let target_52bit: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&circuit.sm1.r_sm, "softmax_k1_r_sm_52bit"),
        (&circuit.sm2.r_sm, "softmax_k2_r_sm_52bit"),
        (&circuit.sm3.r_sm, "softmax_k3_r_sm_52bit"),
    ];
    for (r_sm_4d, label) in target_52bit {
        let b_dim = r_sm_4d.len();
        let h_dim = r_sm_4d[0].len();
        let s_dim = r_sm_4d[0][0].len();
        let l_dim = r_sm_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz: Vec<[usize; SM_KBITS_C]> = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = r_sm_4d[b][h][s][l].into_bigint().as_ref()[0];
                        nz.push([
                            (val_u64 & 0xFFFF) as usize,
                            ((val_u64 >> 16) & 0xFFFF) as usize,
                            ((val_u64 >> 32) & 0xFFFF) as usize,
                            ((val_u64 >> 48) & 0xF) as usize,
                        ]);
                    }
                }
            }
        }
        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), SM_KBITS_C,
            total_sparsity.next_power_of_two(), SM_KBITS_C,
            SM_KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_52(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    // R_div 26-bit
    let target_rdiv: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&circuit.sm1.r_div, "softmax_k1_div_remainder"),
        (&circuit.sm2.r_div, "softmax_k2_div_remainder"),
        (&circuit.sm3.r_div, "softmax_k3_div_remainder"),
    ];
    for (matrix_4d, label) in target_rdiv {
        let b_dim = matrix_4d.len();
        let h_dim = matrix_4d[0].len();
        let s_dim = matrix_4d[0][0].len();
        let l_dim = matrix_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = matrix_4d[b][h][s][l].into_bigint().as_ref()[0];
                        nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3FF) as usize]);
                    }
                }
            }
        }
        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), KBITS_C,
            total_sparsity.next_power_of_two(), KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    Ok((all_results, agg))
}

fn run_onlyeq_mp_conditions_with_link(
    circuit: &NystromAttentionOnlyeqCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] Onlyeq: Moore-Penrose Bounded-Difference ===");
    let mut all_results = Vec::new();
    let mut agg = LassoMetricsAggregate::default();

    let q_y = &circuit.pseudo_inv.q_y_intermediates;
    let p_k2 = &circuit.sm2.p;
    let b_dim = q_y[0].len();
    let h_dim = q_y[0][0].len();
    let m_dim = q_y[0][0][0].len();

    // Condition 1: |Q_y2 - P_k2| < 4096  (13-bit range [0, 8192))
    {
        let delta_u64 = 4096u64;
        let delta = Fr::from(delta_u64);
        let range_limit = delta_u64 * 2;
        let total = b_dim * h_dim * m_dim * m_dim;
        let mut nz = Vec::with_capacity(total);
        let mut max_val: usize = 0;
        for b in 0..b_dim {
            for h in 0..h_dim {
                for i in 0..m_dim {
                    for j in 0..q_y[1][b][h][i].len() {
                        let shifted = q_y[1][b][h][i][j] - p_k2[b][h][i][j] + delta;
                        let val = shifted.into_bigint().as_ref()[0] as usize;
                        assert!(
                            (val as u64) < range_limit,
                            "mp_cond1: shifted value {} exceeds range [0, {}) at [b={},h={},i={},j={}]. \
                             Increase delta (currently {}).",
                            val, range_limit, b, h, i, j, delta_u64,
                        );
                        max_val = max_val.max(val);
                        nz.push([val]);
                    }
                }
            }
        }
        println!("  [mp_cond1] max shifted value: {} (range limit: {})", max_val, range_limit);
        let gens = SparsePolyCommitmentGens::<G1>::new(
            b"mp_cond1", FRACBITS_C, total.next_power_of_two(), FRACBITS_C, FRACBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_mp_range_13(nz, &gens, rng, b"mp_cond1")?;
        all_results.push(link_data);
        agg.add(&metrics);
        println!("  - mp_cond1 verified | Count: {} | Time: {:?}", total, start.elapsed());
    }

    // Condition 3: |Q_y1[i][j] - Q_y1[j][i]| < 512  (10-bit range [0, 1024))
    {
        let delta_u64 = 512u64;
        let delta = Fr::from(delta_u64);
        let range_limit = delta_u64 * 2;
        let pairs_per_bh = m_dim * (m_dim - 1) / 2;
        let total = b_dim * h_dim * pairs_per_bh;
        let mut nz = Vec::with_capacity(total);
        let mut max_val: usize = 0;
        for b in 0..b_dim {
            for h in 0..h_dim {
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted = q_y[0][b][h][i][j] - q_y[0][b][h][j][i] + delta;
                        let val = shifted.into_bigint().as_ref()[0] as usize;
                        assert!(
                            (val as u64) < range_limit,
                            "mp_cond3: shifted value {} exceeds range [0, {}) at [b={},h={},i={},j={}]. \
                             Increase delta (currently {}).",
                            val, range_limit, b, h, i, j, delta_u64,
                        );
                        max_val = max_val.max(val);
                        nz.push([val]);
                    }
                }
            }
        }
        println!("  [mp_cond3] max shifted value: {} (range limit: {})", max_val, range_limit);
        let gens = SparsePolyCommitmentGens::<G1>::new(
            b"mp_cond3", FRACBITS_C, total.next_power_of_two(), FRACBITS_C, FRACBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_mp_range_10(nz, &gens, rng, b"mp_cond3")?;
        all_results.push(link_data);
        agg.add(&metrics);
        println!("  - mp_cond3 verified | Count: {} | Time: {:?}", total, start.elapsed());
    }

    // Condition 4: |Q_y3[i][j] - Q_y3[j][i]| < 32  (6-bit range [0, 64))
    {
        let delta_u64 = 32u64;
        let delta = Fr::from(delta_u64);
        let range_limit = delta_u64 * 2;
        let pairs_per_bh = m_dim * (m_dim - 1) / 2;
        let total = b_dim * h_dim * pairs_per_bh;
        let mut nz = Vec::with_capacity(total);
        let mut max_val: usize = 0;
        for b in 0..b_dim {
            for h in 0..h_dim {
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted = q_y[2][b][h][i][j] - q_y[2][b][h][j][i] + delta;
                        let val = shifted.into_bigint().as_ref()[0] as usize;
                        assert!(
                            (val as u64) < range_limit,
                            "mp_cond4: shifted value {} exceeds range [0, {}) at [b={},h={},i={},j={}]. \
                             Increase delta (currently {}).",
                            val, range_limit, b, h, i, j, delta_u64,
                        );
                        max_val = max_val.max(val);
                        nz.push([val]);
                    }
                }
            }
        }
        println!("  [mp_cond4] max shifted value: {} (range limit: {})", max_val, range_limit);
        let gens = SparsePolyCommitmentGens::<G1>::new(
            b"mp_cond4", FRACBITS_C, total.next_power_of_two(), FRACBITS_C, FRACBITS_M.trailing_zeros() as usize,
        );
        let start = Instant::now();
        let (link_data, metrics) = prove_mp_range_6(nz, &gens, rng, b"mp_cond4")?;
        all_results.push(link_data);
        agg.add(&metrics);
        println!("  - mp_cond4 verified | Count: {} | Time: {:?}", total, start.elapsed());
    }

    Ok((all_results, agg))
}

// ── Proving pipeline ─────────────────────────────────────────────────────────

pub fn run_nystrom_onlyeq_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [Nystrom Onlyeq] Processing: {:?} ===", file_path);

        // Dry-run: extract d1 indices only (no d2)
        let d1_indices = {
            let temp = load_nystrom_onlyeq_circuit::<E>(file_path)?;
            let v1_ptr = Rc::clone(&temp.committed_vars);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            crate::util::utils::extract_witness_indices(&v1_ptr.borrow())
        };
        let num_commit_1 = d1_indices.len() as u32;
        let d2_indices: Vec<u32> = vec![];
        println!("Commit counts: d1={}, d2=0", num_commit_1);

        let sample_circuit = load_nystrom_onlyeq_circuit::<E>(file_path)?;

        // [0] Setup — d1 only
        println!("=== [0] LegoGroth16 Setup (d1={}, d2=0) ===", num_commit_1);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(), &d1_indices, &d2_indices, &[], &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();
        let pvk = prepare_verifying_key::<E>(&params.vk);

        // Witness extraction
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> = vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
            assert_eq!(values.len(), num_commit_1 as usize, "d1 size mismatch");
            values
        };

        let v1 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // cm1 only (no cm2, no c_W)
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        // Simple Fiat-Shamir: hash(cm1)
        let r_fs = hash_cm_to_field::<E>(&cm1);

        println!("=== [1] Building circuit with Fiat-Shamir challenge ===");
        let final_circuit = NystromAttentionOnlyeqCircuit {
            r_challenge: r_fs,
            ..sample_circuit.clone()
        };

        // [1] Groth16 Prove (v2=Fr::zero(), no d2)
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(), v1, Fr::zero(), Fr::zero(), &params, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
        let g16_prove_time = g16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // [1] Groth16 Verify — public_inputs = [r_challenge] only
        let public_inputs = vec![final_circuit.r_challenge];
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let g16_verify_time = g16_verify_start.elapsed();
        println!("  - Nystrom Onlyeq SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // [2] Lasso proofs
        proof_size::reset_lasso_aggregator();
        println!("=== [2] Nystrom Onlyeq Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let (lasso_sm1, sm1_agg) = run_lasso_bundle(&final_circuit.sm1, b"lasso_k1", &mut rng)?;
        lasso_agg.add_aggregate(&sm1_agg);
        let (lasso_sm2, sm2_agg) = run_lasso_bundle(&final_circuit.sm2, b"lasso_k2", &mut rng)?;
        lasso_agg.add_aggregate(&sm2_agg);
        let (lasso_sm3, sm3_agg) = run_lasso_bundle(&final_circuit.sm3, b"lasso_k3", &mut rng)?;
        lasso_agg.add_aggregate(&sm3_agg);
        let (lasso_remainders, rem_agg) = run_onlyeq_remainders_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let (mp_results, mp_agg) = run_onlyeq_mp_conditions_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&mp_agg);
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = vec![];
        lasso_results.extend(lasso_sm1);
        lasso_results.extend(lasso_sm2);
        lasso_results.extend(lasso_sm3);
        lasso_results.extend(lasso_remainders);
        lasso_results.extend(mp_results);

        // [3] CP-Link #1: d1 ↔ Lasso
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
            return Err(anyhow::anyhow!("CP-Link #1 (d1 ↔ Lasso) verification failed"));
        }
        println!("  -> [Success] CP-Link #1 (d1 ↔ Lasso) verified");
        let cpl_verify_ms = cpl_verify_start.elapsed().as_millis();

        let cplink_lasso_report = proof_size::compute_cplink_report::<E>(&proof_link_lasso, &instance_lasso);
        cplink_lasso_report.print_log();

        // Fiat-Shamir verification
        let r_fs_verify = hash_cm_to_field::<E>(&proof.d1);
        assert_eq!(r_fs, r_fs_verify, "Fiat-Shamir recomputation mismatch");
        println!("  -> [Success] Fiat-Shamir recomputation verified");

        // Proof size report (no global link)
        let empty_cplink = proof_size::CpLinkSizeReport {
            proof_g_count: 0, proof_g_bytes: 0,
            proof_f_count: 0, proof_f_bytes: 0,
            proof_total: 0,
            instance_g_count: 0, instance_g_bytes: 0,
        };
        proof_size::print_full_report(&g16_report, &cplink_lasso_report, &empty_cplink);

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
            circuit: "nystrom_attention_onlyeq".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: g16_commit_ms,
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
                commit_ms: 0, setup_ms: 0, prove_ms: 0, verify_ms: 0, proof_size_bytes: 0,
            },
            end_to_end_ms: e2e_time.as_millis(),
        }
        .print_json();
    }
    Ok(())
}

#[test]
fn nystrom_attn_onlyeq_single_layer_test() {
    use std::env;

    let witness_dir = env::var("NYSTROM_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_m64".to_string());
    let layer_idx: Option<usize> = env::var("NYSTROM_ATTN_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] NYSTROM_ATTN_WITNESS_DIR={} not a directory.",
            witness_dir
        );
        return;
    }

    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
        println!("[info] SOFTMAX_TABLE_PATH not set; using {}/softmax_tables.npz", witness_dir);
    }

    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name().and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention.self.witness.npz"))
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
        assert!(idx < all_targets.len(), "NYSTROM_ATTN_LAYER_IDX={} out of range", idx);
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [Nystrom Onlyeq prove+verify] dir={} | files={} | mode={} ===",
        witness_dir, targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_nystrom_onlyeq_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!("\n=== [Nystrom Onlyeq] ALL {} files PASS | total_wall_ms={} ===", targets.len(), total_ms),
        Err(e) => panic!("[Nystrom Onlyeq] proof failed after {}ms: {}", total_ms, e),
    }
}
