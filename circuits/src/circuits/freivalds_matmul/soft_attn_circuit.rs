use std::{fs::File, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::pairing::Pairing;
use ndarray::{IxDyn, Ix0, Ix2, Ix3, Ix4};
use ndarray_npy::NpzReader;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{SeedableRng, rngs::StdRng},
    cell::RefCell,
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero, BigInteger};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{
    R1CSVar, alloc::AllocVar, eq::EqGadget,
    fields::{fp::FpVar, FieldVar},
    select::CondSelectGadget,
};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            global_link_setup, global_link_keygen, global_link_prove, global_link_verify,
            pad_lasso_u_values, groth16_commitment_size,
        },
        proof_size,
    },
    gadgets::{
        activation::gelu::split_remainder_and_push,
        matmul::{
            constraints::{MatrixVar, new_matrix_witness},
            quantized::{
                MatmulBundle, MatmulData, ProjectionBundle, ScalingBundle,
                PseudoinverseBundle, ZenParams,
                compute_circuit_landmarks, enforce_freivalds, enforce_scaling,
                gen_challenge_vector, new_matrix_witness_4d, transpose_matrix,
            },
        },
    },
    util::{
        load::{
            to_field_2d_i64, to_field_3d_i64, to_field_3d_u64,
            to_field_4d_i64, to_field_4d_u64, to_field_5d_i64,
        },
        soft_attn_lookup::{run_soft_lasso_bundle, run_all_soft_remainders_with_link, run_soft_mp_conditions_with_link},
        utils::{cm1_from_vk, cm2_from_vk, cm_aux_from_vk, generate_global_ck, commit_global, hash_commitments_to_field_v3},
    },
    lasso::subtables::gaussian::set_gaussian_table_dir,
};
use crate::circuits::freivalds_matmul::model_integrated_circuit::ConvBundle;

type G1 = G1Projective;

// ═══════════════════════════════════════════════════════════════════════════════
// B1: SOFT-specific helper structs
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct GaussianBundle<F: PrimeField> {
    pub kernel: Vec<Vec<Vec<Vec<F>>>>,      // [B][H][R][C] table output values
    pub table_idx: Vec<Vec<Vec<Vec<F>>>>,   // [B][H][R][C] table indices
    pub r_gauss: Vec<Vec<Vec<Vec<F>>>>,     // [B][H][R][C] 52-bit remainders
    pub m_gauss: F,                          // scaling constant
}

#[derive(Clone)]
pub struct DistBundle<F: PrimeField> {
    pub q_dist: Vec<Vec<Vec<Vec<F>>>>,     // [B][H][R][C] Euclidean quotient
    pub r_dist: Vec<Vec<Vec<Vec<F>>>>,     // [B][H][R][C] Euclidean remainder (26-bit)
    pub m_a: F,                             // norm multiplier
    pub m_b: F,                             // cross-term multiplier
    pub z_dist: F,                          // floor guard offset (= 1)
}

// ═══════════════════════════════════════════════════════════════════════════════
// B1: SOFTAttentionCircuit main struct
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct SOFTAttentionCircuit<F: PrimeField> {
    // Zero-points (2 instead of 3: no z_k since W_Q = W_K)
    pub z_q: F,           // = z_qtilde (Q_proc and landmarks share zero-point)
    pub z_v: F,

    // Projection inputs
    pub q_x_input: Vec<Vec<Vec<F>>>,   // [B][S][HD]
    pub z_x_input: F,

    // Projections (2 instead of 3: no proj_k)
    pub proj_q: ProjectionBundle<F>,
    pub proj_v: ProjectionBundle<F>,

    // Scaling (1 instead of 2: no scale_k)
    pub scale_q: ScalingBundle<F>,

    // Output projection
    pub z_attn_output: F,
    pub q_attn_output: Vec<Vec<Vec<F>>>,   // [B][S][HD]
    pub proj_o: ProjectionBundle<F>,

    // SOFT-specific: cross-term matmuls
    pub cross_b: MatmulBundle<F>,       // Q_proc @ Q~^T -> C_B [B][H][S][m]
    pub cross_a: MatmulBundle<F>,       // Q~ @ Q~^T -> C_A [B][H][m][m]

    // SOFT-specific: Gaussian kernel data
    pub gauss_b: GaussianBundle<F>,     // kernel B lookup data
    pub gauss_a: GaussianBundle<F>,     // kernel A lookup data

    // SOFT-specific: squared norms
    pub norm_q: Vec<Vec<Vec<F>>>,       // [B][H][S] integer squared norms
    pub norm_qtilde: Vec<Vec<Vec<F>>>,  // [B][H][m]

    // SOFT-specific: distance equation params
    pub dist_b: DistBundle<F>,
    pub dist_a: DistBundle<F>,

    // Pseudoinverse (same as Nystrom)
    pub pseudo_inv: PseudoinverseBundle<F>,

    // Chain matmuls (same 3-step pattern, different inputs)
    pub matmul_temp: MatmulBundle<F>,   // B^T @ V
    pub matmul_mid: MatmulBundle<F>,    // B @ A^{+}
    pub matmul_x: MatmulBundle<F>,      // mid @ temp = Y

    // Config
    pub frac_bits: usize,              // Gaussian FRAC_BITS (16)
    pub r_challenge: F,
    pub seg_size: usize,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,             // [B][S]

    // MP epsilon (per-layer, loaded from witness metadata)
    pub eps1: usize,                   // condition 1 half-range
    pub eps3: usize,                   // condition 3 half-range
    pub eps4: usize,                   // condition 4 half-range
    pub mp_log_r1: usize,             // Lasso range bits for cond1: [0, 2^log_r1)
    pub mp_log_r3: usize,             // Lasso range bits for cond3
    pub mp_log_r4: usize,             // Lasso range bits for cond4

    // Poseidon hashes
    pub input_hash: F,
    pub output_hash: F,

    // Commitment groups
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
    pub committed_vars_aux: Rc<RefCell<Vec<FpVar<F>>>>,

    // Conv (optional, same as Nystrom)
    pub conv: Option<ConvBundle<F>>,

    // Pre-masked witnesses for masking verification
    pub q_q_masked: Vec<Vec<Vec<Vec<F>>>>,  // [B][H][S][D]
    pub q_v_masked: Vec<Vec<Vec<Vec<F>>>>,  // [B][H][S][D]
}

/// Compute the minimum Lasso range-check LOG_R for a given epsilon.
/// Range check covers [0, 2*eps), so LOG_R = ceil(log2(2*eps)).
pub(crate) fn eps_to_log_r(eps: usize) -> usize {
    assert!(eps > 0, "epsilon must be positive");
    let range = 2 * eps;
    let bits = (usize::BITS - (range - 1).leading_zeros()) as usize;
    bits
}

// ═══════════════════════════════════════════════════════════════════════════════
// Private helper: split_52bit_remainder (duplicated from softmax/constraints.rs)
// ═══════════════════════════════════════════════════════════════════════════════

pub(crate) fn split_52bit_remainder<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    r_var: &FpVar<F>,
    r_val: F,
    limbs_out: &mut [Vec<FpVar<F>>; 4],
) -> Result<(), SynthesisError> {
    let r_u64 = r_val.into_bigint().as_ref()[0];
    let l0 = r_u64 & 0xFFFF;                 // bits [0:15]  — 16 bits
    let l1 = (r_u64 >> 16) & 0xFFFF;         // bits [16:31] — 16 bits
    let l2 = (r_u64 >> 32) & 0xFFFF;         // bits [32:47] — 16 bits
    let l3 = (r_u64 >> 48) & 0xF;            // bits [48:51] — 4 bits

    let l0_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l0)))?;
    let l1_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l1)))?;
    let l2_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l2)))?;
    let l3_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l3)))?;

    let shift16 = F::from(1u128 << 16);
    let shift32 = F::from(1u128 << 32);
    let shift48 = F::from(1u128 << 48);

    let reconstructed = &l0_var
        + &l1_var * shift16
        + &l2_var * shift32
        + &l3_var * shift48;

    r_var.enforce_equal(&reconstructed)?;

    limbs_out[0].push(l0_var);
    limbs_out[1].push(l1_var);
    limbs_out[2].push(l2_var);
    limbs_out[3].push(l3_var);

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: allocate weight vars with transposed indexing [H][D][HD] -> [H][HD][D]
// ═══════════════════════════════════════════════════════════════════════════════

pub(crate) fn alloc_weight_vars<F: PrimeField>(
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

// ═══════════════════════════════════════════════════════════════════════════════
// B3-B9: ConstraintSynthesizer implementation
// ═══════════════════════════════════════════════════════════════════════════════

impl<F: PrimeField> ConstraintSynthesizer<F> for SOFTAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();
        self.committed_vars_aux.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        // ── B3: Witness allocation ──────────────────────────────────────────

        // Landmark remainders (only 1 set: Q~ = K~)
        let r_mean_q = new_matrix_witness_4d(cs.clone(), &self.cross_b.r_mean);

        // Cross-term matmul data allocation
        // Both z_w and z_x are z_q since W_Q = W_K and Q_proc shares zero-point
        let (cross_b_data, params_cross_b) =
            self.cross_b.alloc_and_setup(cs.clone(), self.z_q, self.z_q, &two_k_var);
        let (cross_a_data, params_cross_a) =
            self.cross_a.alloc_and_setup(cs.clone(), self.z_q, self.z_q, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &self.mask);

        // X_input [B][S][HD] as witnesses
        let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = self.q_x_input.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        // Projection output witnesses [B][H][S][D]
        let proj_q_out = new_matrix_witness_4d(cs.clone(), &self.proj_q.q_out_proj);
        let proj_q_r = new_matrix_witness_4d(cs.clone(), &self.proj_q.r_out_proj);
        let proj_v_out = new_matrix_witness_4d(cs.clone(), &self.proj_v.q_out_proj);
        let proj_v_r = new_matrix_witness_4d(cs.clone(), &self.proj_v.r_out_proj);

        // Scaling witnesses [B][H][S][D] (Q only, no K scaling)
        let scale_q_out = new_matrix_witness_4d(cs.clone(), &self.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &self.scale_q.r_scale);

        // W_Q, W_V, W_O weights (3 sets, no W_K)
        let num_heads = self.proj_q.q_w_diff.len();
        let w_q_vars = alloc_weight_vars(cs.clone(), &self.proj_q.q_w_diff);
        let w_v_vars = alloc_weight_vars(cs.clone(), &self.proj_v.q_w_diff);
        let w_o_vars = alloc_weight_vars(cs.clone(), &self.proj_o.q_w_diff);

        // d2: commit W_Q, W_V, W_O weights
        {
            let mut vars_2 = self.committed_vars_2.borrow_mut();
            for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        }

        // aux: commit proj_q, proj_v outputs (2 sets, no K)
        {
            let mut vars_aux = self.committed_vars_aux.borrow_mut();
            for b in &proj_q_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
            for b in &proj_v_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
        }

        let proj_o_out = new_matrix_witness_4d(cs.clone(), &self.proj_o.q_out_proj);
        let proj_o_r = new_matrix_witness_4d(cs.clone(), &self.proj_o.r_out_proj);

        // ── Remainder pools ──
        let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
        let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
        let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();

        let mut global_cross_b_lows = Vec::new(); let mut global_cross_b_highs = Vec::new();
        let mut global_cross_a_lows = Vec::new(); let mut global_cross_a_highs = Vec::new();

        // Chain matmul allocation
        let (temp_data, params_temp) =
            self.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);
        let (mid_data, params_mid) =
            self.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), self.pseudo_inv.z_inv, &two_k_var);
        let (x_data, params_x) =
            self.matmul_x.alloc_and_setup(cs.clone(), self.matmul_mid.z_sim, self.matmul_temp.z_sim, &two_k_var);

        // aux: commit cross_b, cross_a, chain intermediates
        {
            let mut vars_aux = self.committed_vars_aux.borrow_mut();
            for b in &cross_b_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
            for b in &cross_a_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
            for b in &temp_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
            for b in &mid_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
            for b in &x_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
        }

        // O-proj input derivation
        let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = if self.conv.is_some() {
            self.q_attn_output.iter().map(|batch| {
                batch.iter().map(|seq| {
                    seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
                }).collect()
            }).collect()
        } else {
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

        // ── Gaussian lookup pools (C=1 indices) ──
        let mut global_gauss_b_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_gauss_a_idx: Vec<FpVar<F>> = Vec::new();

        // ── r_gauss 52-bit limbs (C=4) ──
        let mut global_r_gauss_b_limbs: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut global_r_gauss_a_limbs: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        // ── R_dist 26-bit remainder pools ──
        let mut global_r_dist_b_lows = Vec::new(); let mut global_r_dist_b_highs = Vec::new();
        let mut global_r_dist_a_lows = Vec::new(); let mut global_r_dist_a_highs = Vec::new();

        // ── Moore-Penrose remainder pools ──
        let mut global_inv0_lows = Vec::new(); let mut global_inv0_highs = Vec::new();
        let mut global_inv1_lows = Vec::new(); let mut global_inv1_highs = Vec::new();
        let mut global_inv2_lows = Vec::new(); let mut global_inv2_highs = Vec::new();
        let mut global_inv3_lows = Vec::new(); let mut global_inv3_highs = Vec::new();

        // Moore-Penrose bounded-difference pools (C=1)
        let mut mp_cond1_pool: Vec<FpVar<F>> = Vec::new();
        let mut mp_cond3_pool: Vec<FpVar<F>> = Vec::new();
        let mut mp_cond4_pool: Vec<FpVar<F>> = Vec::new();

        let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
        let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
        let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();

        // Z* (q_inv_var) collector for aux
        let mut q_inv_committed: Vec<FpVar<F>> = Vec::new();

        // ── Allocate Gaussian kernel and dist witnesses ──
        let gauss_b_kernel = new_matrix_witness_4d(cs.clone(), &self.gauss_b.kernel);
        let gauss_b_table_idx = new_matrix_witness_4d(cs.clone(), &self.gauss_b.table_idx);
        let gauss_b_r_gauss = new_matrix_witness_4d(cs.clone(), &self.gauss_b.r_gauss);
        let m_gauss_b_var = FpVar::new_constant(cs.clone(), self.gauss_b.m_gauss)?;

        let gauss_a_kernel = new_matrix_witness_4d(cs.clone(), &self.gauss_a.kernel);
        let gauss_a_table_idx = new_matrix_witness_4d(cs.clone(), &self.gauss_a.table_idx);
        let gauss_a_r_gauss = new_matrix_witness_4d(cs.clone(), &self.gauss_a.r_gauss);
        let m_gauss_a_var = FpVar::new_constant(cs.clone(), self.gauss_a.m_gauss)?;

        let q_dist_b = new_matrix_witness_4d(cs.clone(), &self.dist_b.q_dist);
        let r_dist_b = new_matrix_witness_4d(cs.clone(), &self.dist_b.r_dist);
        let m_a_b_var = FpVar::new_constant(cs.clone(), self.dist_b.m_a)?;
        let m_b_b_var = FpVar::new_constant(cs.clone(), self.dist_b.m_b)?;
        let z_dist_b_var = FpVar::new_constant(cs.clone(), self.dist_b.z_dist)?;

        let q_dist_a = new_matrix_witness_4d(cs.clone(), &self.dist_a.q_dist);
        let r_dist_a = new_matrix_witness_4d(cs.clone(), &self.dist_a.r_dist);
        let m_a_a_var = FpVar::new_constant(cs.clone(), self.dist_a.m_a)?;
        let m_b_a_var = FpVar::new_constant(cs.clone(), self.dist_a.m_b)?;
        let z_dist_a_var = FpVar::new_constant(cs.clone(), self.dist_a.z_dist)?;

        // Squared norm witnesses
        let norm_q_vars: Vec<Vec<Vec<FpVar<F>>>> = self.norm_q.iter().map(|batch| {
            batch.iter().map(|head| {
                head.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        let norm_qtilde_vars: Vec<Vec<Vec<FpVar<F>>>> = self.norm_qtilde.iter().map(|batch| {
            batch.iter().map(|head| {
                head.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        // ── Pre-kernel masking witnesses ──
        let q_proc_masked = new_matrix_witness_4d(cs.clone(), &self.q_q_masked);
        let v_masked = new_matrix_witness_4d(cs.clone(), &self.q_v_masked);

        // Build mask booleans [B][S]
        let mask_bools: Vec<Vec<_>> = mask_var.iter().map(|b_mask| {
            b_mask.iter().map(|m| m.is_one().unwrap()).collect::<Vec<_>>()
        }).collect();

        let z_q_var = FpVar::Constant(self.z_q);
        let z_v_var = FpVar::Constant(self.z_v);

        // Precompute two_m_b and z_dist*2^k for dist equation
        let two_var = FpVar::Constant(F::from(2u64));
        let two_m_b_b = &two_var * &m_b_b_var;
        let z_dist_two_k_b = &z_dist_b_var * &two_k_var;
        let two_m_b_a = &two_var * &m_b_a_var;
        let z_dist_two_k_a = &z_dist_a_var * &two_k_var;

        // Cross-term z_sim for dist equation
        let z_cross_b_var = FpVar::Constant(self.cross_b.z_sim);
        let z_cross_a_var = FpVar::Constant(self.cross_a.z_sim);

        let two_2k = &two_k_var * &two_k_var;  // 2^(2*k_bits)

        // ── Per-(batch, head) constraint loop ──────────────────────────────
        #[cfg(debug_assertions)]
        eprintln!("[soft] before loop: {} constraints", cs.num_constraints());

        for b in 0..scale_q_out.len() {
            // Landmarks (only 1 set: Q~ = K~)
            let q_landmarks = compute_circuit_landmarks(
                cs.clone(), &q_proc_masked[b], &r_mean_q[b], self.seg_size,
            )?;

            // ── Enforce pre-kernel masking ──
            {
                let s_len = scale_q_out[b][0].len();
                let d_dim = scale_q_out[b][0][0].len();
                for h in 0..scale_q_out[b].len() {
                    for s in 0..s_len {
                        for d in 0..d_dim {
                            // Q_proc masking
                            let expected_q = FpVar::conditionally_select(
                                &mask_bools[b][s],
                                &scale_q_out[b][h][s][d],
                                &z_q_var,
                            )?;
                            q_proc_masked[b][h][s][d].enforce_equal(&expected_q)?;

                            // V masking (uses proj_v_out: SOFT has no V scaling)
                            let expected_v = FpVar::conditionally_select(
                                &mask_bools[b][s],
                                &proj_v_out[b][h][s][d],
                                &z_v_var,
                            )?;
                            v_masked[b][h][s][d].enforce_equal(&expected_v)?;
                        }
                    }
                }
            }

            for h in 0..scale_q_out[b].len() {
                #[cfg(debug_assertions)]
                let _nc_h_start = cs.num_constraints();

                // ── B3: Q Projection ──
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

                // ── B3: V Projection ──
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

                // ── B3: Q Scaling ──
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

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after proj+scale: {} (delta={})",
                    b, h, cs.num_constraints(), cs.num_constraints() - _nc_h_start);

                // ── B4: Squared norms ──
                // norm_q[b][h][i] = sum_d (q_proc_masked[b][h][i][d] - z_q)^2
                {
                    let s_len = q_proc_masked[b][h].len();
                    let d_dim = q_proc_masked[b][h][0].len();
                    for i in 0..s_len {
                        let mut norm_sum = FpVar::zero();
                        for d in 0..d_dim {
                            let diff = &q_proc_masked[b][h][i][d] - &z_q_var;
                            // diff * diff produces an R1CS multiplication constraint
                            let sq = &diff * &diff;
                            norm_sum = norm_sum + &sq;
                        }
                        norm_sum.enforce_equal(&norm_q_vars[b][h][i])?;
                    }
                }

                // norm_qtilde[b][h][j] = sum_d (q_landmarks[h][j][d] - z_q)^2
                {
                    let m_dim = q_landmarks[h].len();
                    let d_dim = q_landmarks[h][0].len();
                    for j in 0..m_dim {
                        let mut norm_sum = FpVar::zero();
                        for d in 0..d_dim {
                            let diff = &q_landmarks[h][j][d] - &z_q_var;
                            let sq = &diff * &diff;
                            norm_sum = norm_sum + &sq;
                        }
                        norm_sum.enforce_equal(&norm_qtilde_vars[b][h][j])?;
                    }
                }

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after norms: {}", b, h, cs.num_constraints());

                // ── B4: Cross-term Freivalds ──
                let q_l_t = transpose_matrix(&q_landmarks[h]);

                // C_B: Q_proc_masked @ Q~^T -> [S][m]
                {
                    let r_pows_cb = gen_challenge_vector(
                        &r_var, cross_b_data.q_sim[b][h][0].len(),
                    );
                    let mut h_cb_lows = Vec::new();
                    let mut h_cb_highs = Vec::new();
                    enforce_freivalds(
                        cs.clone(),
                        &q_proc_masked[b][h],
                        &q_l_t,
                        &cross_b_data.q_sim[b][h],
                        &cross_b_data.r_matmul[b][h],
                        &self.cross_b.r_matmul[b][h],
                        &params_cross_b,
                        &r_pows_cb,
                        &mut h_cb_lows,
                        &mut h_cb_highs,
                        None,
                    )?;
                    global_cross_b_lows.extend(h_cb_lows);
                    global_cross_b_highs.extend(h_cb_highs);
                }

                // C_A: Q~ @ Q~^T -> [m][m]
                {
                    let r_pows_ca = gen_challenge_vector(
                        &r_var, cross_a_data.q_sim[b][h][0].len(),
                    );
                    let mut h_ca_lows = Vec::new();
                    let mut h_ca_highs = Vec::new();
                    enforce_freivalds(
                        cs.clone(),
                        &q_landmarks[h],
                        &q_l_t,
                        &cross_a_data.q_sim[b][h],
                        &cross_a_data.r_matmul[b][h],
                        &self.cross_a.r_matmul[b][h],
                        &params_cross_a,
                        &r_pows_ca,
                        &mut h_ca_lows,
                        &mut h_ca_highs,
                        None,
                    )?;
                    global_cross_a_lows.extend(h_ca_lows);
                    global_cross_a_highs.extend(h_ca_highs);
                }

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after cross-terms: {}", b, h, cs.num_constraints());

                // ── B5: Distance equation + Gaussian lookup ──

                // Dist equation B: for each (i, j) in [S][m]
                {
                    let s_len = q_dist_b[b][h].len();
                    let m_dim = q_dist_b[b][h][0].len();
                    for i in 0..s_len {
                        for j in 0..m_dim {
                            // LHS = Q_dist * 2^k + R_dist
                            let lhs = &q_dist_b[b][h][i][j] * &two_k_var + &r_dist_b[b][h][i][j];
                            // RHS = M_A*(norm_q[i]+norm_qtilde[j]) - 2*M_B*(C_B[i][j]-z_cross) + z_dist*2^k
                            let n_sum = &norm_q_vars[b][h][i] + &norm_qtilde_vars[b][h][j];
                            let cross_shifted = &cross_b_data.q_sim[b][h][i][j] - &z_cross_b_var;
                            let rhs = &m_a_b_var * &n_sum - &two_m_b_b * &cross_shifted + &z_dist_two_k_b;
                            lhs.enforce_equal(&rhs)?;

                            // R_dist 26-bit split
                            let (low, high) = split_remainder_and_push(
                                cs.clone(),
                                &r_dist_b[b][h][i][j],
                                self.dist_b.r_dist[b][h][i][j],
                            )?;
                            global_r_dist_b_lows.push(low);
                            global_r_dist_b_highs.push(high);
                        }
                    }
                }

                // Dist equation A: for each (i, j) in [m][m]
                {
                    let m_dim = q_dist_a[b][h].len();
                    let m_dim2 = q_dist_a[b][h][0].len();
                    for i in 0..m_dim {
                        for j in 0..m_dim2 {
                            let lhs = &q_dist_a[b][h][i][j] * &two_k_var + &r_dist_a[b][h][i][j];
                            let n_sum = &norm_qtilde_vars[b][h][i] + &norm_qtilde_vars[b][h][j];
                            let cross_shifted = &cross_a_data.q_sim[b][h][i][j] - &z_cross_a_var;
                            let rhs = &m_a_a_var * &n_sum - &two_m_b_a * &cross_shifted + &z_dist_two_k_a;
                            lhs.enforce_equal(&rhs)?;

                            let (low, high) = split_remainder_and_push(
                                cs.clone(),
                                &r_dist_a[b][h][i][j],
                                self.dist_a.r_dist[b][h][i][j],
                            )?;
                            global_r_dist_a_lows.push(low);
                            global_r_dist_a_highs.push(high);
                        }
                    }
                }

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after dist eq: {}", b, h, cs.num_constraints());

                // Gaussian lookup B: (Q_dist * 2^k + R_dist) * M_gauss = table_idx * 2^(2k) + r_gauss
                {
                    let s_len = gauss_b_table_idx[b][h].len();
                    let m_dim = gauss_b_table_idx[b][h][0].len();
                    for i in 0..s_len {
                        for j in 0..m_dim {
                            let x = &q_dist_b[b][h][i][j] * &two_k_var + &r_dist_b[b][h][i][j];
                            let lhs = &x * &m_gauss_b_var;
                            let rhs = &gauss_b_table_idx[b][h][i][j] * &two_2k + &gauss_b_r_gauss[b][h][i][j];
                            lhs.enforce_equal(&rhs)?;

                            // Collect table_idx for Lasso (C=1)
                            global_gauss_b_idx.push(gauss_b_table_idx[b][h][i][j].clone());

                            // r_gauss 52-bit split
                            split_52bit_remainder(
                                cs.clone(),
                                &gauss_b_r_gauss[b][h][i][j],
                                gauss_b_r_gauss[b][h][i][j].value().unwrap_or(F::zero()),
                                &mut global_r_gauss_b_limbs,
                            )?;
                        }
                    }
                }

                // Gaussian lookup A
                {
                    let m_dim = gauss_a_table_idx[b][h].len();
                    let m_dim2 = gauss_a_table_idx[b][h][0].len();
                    for i in 0..m_dim {
                        for j in 0..m_dim2 {
                            let x = &q_dist_a[b][h][i][j] * &two_k_var + &r_dist_a[b][h][i][j];
                            let lhs = &x * &m_gauss_a_var;
                            let rhs = &gauss_a_table_idx[b][h][i][j] * &two_2k + &gauss_a_r_gauss[b][h][i][j];
                            lhs.enforce_equal(&rhs)?;

                            global_gauss_a_idx.push(gauss_a_table_idx[b][h][i][j].clone());

                            split_52bit_remainder(
                                cs.clone(),
                                &gauss_a_r_gauss[b][h][i][j],
                                gauss_a_r_gauss[b][h][i][j].value().unwrap_or(F::zero()),
                                &mut global_r_gauss_a_limbs,
                            )?;
                        }
                    }
                }

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after Gaussian lookup: {}", b, h, cs.num_constraints());

                // ── B6: MP Pseudoinverse ──
                let q_inv_var = new_matrix_witness_4d(cs.clone(), &self.pseudo_inv.q_inv);
                for row in &q_inv_var[b][h] {
                    q_inv_committed.extend(row.iter().cloned());
                }
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
                        0 => (&z_zero, &z_inv, &z_zero),  // A @ A+ = Y1
                        1 => (&z_zero, &z_zero, &z_zero), // Y1 @ A = Y2
                        2 => (&z_inv, &z_zero, &z_inv),   // A+ @ A = Y3
                        3 => (&z_inv, &z_inv, &z_inv),    // Y3 @ A+ = Y4
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

                // Step 0: kernel_A @ A+ = Y1  (z_w=0 since Gaussian outputs are unsigned, z_x=z_inv)
                let mut h_inv0_lows = Vec::new();
                let mut h_inv0_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &gauss_a_kernel[b][h],  // Gaussian kernel A (replaces sm2_vars.p)
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
                global_inv0_lows.extend(h_inv0_lows);
                global_inv0_highs.extend(h_inv0_highs);

                // Step 1: Y1 @ kernel_A = Y2
                let mut h_inv1_lows = Vec::new();
                let mut h_inv1_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &q_y_vars[0][b][h],
                    &gauss_a_kernel[b][h],
                    &q_y_vars[1][b][h],
                    &r_inv_vars[1][b][h],
                    &self.pseudo_inv.r_inv_remainders[1][b][h],
                    &inv_params[1],
                    &r_pows_inv,
                    &mut h_inv1_lows,
                    &mut h_inv1_highs,
                    None,
                )?;
                global_inv1_lows.extend(h_inv1_lows);
                global_inv1_highs.extend(h_inv1_highs);

                // Step 2: A+ @ kernel_A = Y3
                let mut h_inv2_lows = Vec::new();
                let mut h_inv2_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &q_inv_var[b][h],
                    &gauss_a_kernel[b][h],
                    &q_y_vars[2][b][h],
                    &r_inv_vars[2][b][h],
                    &self.pseudo_inv.r_inv_remainders[2][b][h],
                    &inv_params[2],
                    &r_pows_inv,
                    &mut h_inv2_lows,
                    &mut h_inv2_highs,
                    None,
                )?;
                global_inv2_lows.extend(h_inv2_lows);
                global_inv2_highs.extend(h_inv2_highs);

                // Step 3: Y3 @ A+ = Y4
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
                global_inv3_lows.extend(h_inv3_lows);
                global_inv3_highs.extend(h_inv3_highs);

                // ── MP bounded-difference conditions ──
                let m_dim = q_y_vars[0][b][h].len();

                // Condition 1: |Q_y2 - kernel_A| < eps1
                for i in 0..m_dim {
                    for j in 0..q_y_vars[1][b][h][i].len() {
                        let shifted_expr = &q_y_vars[1][b][h][i][j]
                            - &gauss_a_kernel[b][h][i][j]
                            + FpVar::Constant(F::from(self.eps1 as u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond1_pool.push(shifted);
                    }
                }

                // Condition 3: |Q_y1[i][j] - Q_y1[j][i]| < eps3
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted_expr = &q_y_vars[0][b][h][i][j]
                            - &q_y_vars[0][b][h][j][i]
                            + FpVar::Constant(F::from(self.eps3 as u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond3_pool.push(shifted);
                    }
                }

                // Condition 4: |Q_y3[i][j] - Q_y3[j][i]| < eps4
                for i in 0..m_dim {
                    for j in (i+1)..m_dim {
                        let shifted_expr = &q_y_vars[2][b][h][i][j]
                            - &q_y_vars[2][b][h][j][i]
                            + FpVar::Constant(F::from(self.eps4 as u64));
                        let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                        shifted_expr.enforce_equal(&shifted)?;
                        mp_cond4_pool.push(shifted);
                    }
                }

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after pseudo-inv + MP: {}", b, h, cs.num_constraints());

                // ── B7: Chain matmuls ──

                // temp: B^T @ V  (need transpose of kernel_b)
                let kernel_b_t = transpose_matrix(&gauss_b_kernel[b][h]); // [S][m] -> [m][S]
                let mut h_temp_lows = Vec::new();
                let mut h_temp_highs = Vec::new();
                let r_pows_temp = gen_challenge_vector(&r_var, temp_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &kernel_b_t,             // B^T [m×S]
                    &v_masked[b][h],         // V [S×D]
                    &temp_data.q_sim[b][h],
                    &temp_data.r_matmul[b][h],
                    &self.matmul_temp.r_matmul[b][h],
                    &params_temp,
                    &r_pows_temp,
                    &mut h_temp_lows,
                    &mut h_temp_highs,
                    None,
                )?;
                global_temp_lows.extend(h_temp_lows);
                global_temp_highs.extend(h_temp_highs);

                // mid: B @ A^{+}  (kernel_b is [S][m], q_inv is [m][m])
                let mut h_mid_lows = Vec::new();
                let mut h_mid_highs = Vec::new();
                let r_pows_mid = gen_challenge_vector(&r_var, mid_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &gauss_b_kernel[b][h],   // B [S×m]
                    &q_inv_var[b][h],        // A^{+} [m×m]
                    &mid_data.q_sim[b][h],
                    &mid_data.r_matmul[b][h],
                    &self.matmul_mid.r_matmul[b][h],
                    &params_mid,
                    &r_pows_mid,
                    &mut h_mid_lows,
                    &mut h_mid_highs,
                    None,
                )?;
                global_mid_lows.extend(h_mid_lows);
                global_mid_highs.extend(h_mid_highs);

                // Y: mid @ temp  ([S×m] @ [m×D] = [S×D])
                let mut h_x_lows = Vec::new();
                let mut h_x_highs = Vec::new();
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len());
                enforce_freivalds(
                    cs.clone(),
                    &mid_data.q_sim[b][h],   // mid [S×m]
                    &temp_data.q_sim[b][h],  // temp [m×D]
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

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after temp/mid/x: {}", b, h, cs.num_constraints());

                // ── B8: O Projection ──
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
                        cs.clone(), o_input, w_o_h_t,
                        &proj_o_out[b][h], &proj_o_r[b][h],
                        &self.proj_o.r_out_proj[b][h],
                        &proj_o_params, &r_pows_po,
                        &mut global_proj_o_lows, &mut global_proj_o_highs,
                        Some(&self.proj_o.q_b_diff[h]),
                    )?;
                }

                #[cfg(debug_assertions)]
                eprintln!("[soft] b={} h={} after O-proj: {}", b, h, cs.num_constraints());
            }
        }

        // aux: commit Z* (q_inv_var)
        {
            let mut vars_aux = self.committed_vars_aux.borrow_mut();
            vars_aux.extend(q_inv_committed);
        }

        // ── B9: Populate d1 committed_vars ──
        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // C=1 indices: Gaussian table lookups
            final_vars.extend(global_gauss_b_idx);
            final_vars.extend(global_gauss_a_idx);

            // 26-bit remainders (C=2: 16+10)
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

            // 52-bit r_gauss (C=4: 16+16+16+4)
            for k in 0..4 { final_vars.extend(global_r_gauss_b_limbs[k].clone()); }
            for k in 0..4 { final_vars.extend(global_r_gauss_a_limbs[k].clone()); }

            // MP bounded-diff (C=1)
            final_vars.extend(mp_cond1_pool);
            final_vars.extend(mp_cond3_pool);
            final_vars.extend(mp_cond4_pool);
        }

        // ── B8: Poseidon hash of x_input -> public input ──
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

        // ── B8: Poseidon hash of output -> public input ──
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

// ═══════════════════════════════════════════════════════════════════════════════
// B2: NPZ Loader
// ═══════════════════════════════════════════════════════════════════════════════

pub fn load_soft_circuit<E: Pairing>(path: &Path) -> Result<SOFTAttentionCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening SOFT witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    // --- Zero-points (2: no z_k) ---
    let z_q = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar());
    let z_v = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar());

    // --- Metadata ---
    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("gauss_frac_bits.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let seg_size = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("seg_size.npy")?.into_scalar() as usize;

    // --- Projection inputs ---
    let q_x_input = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_x_input.npy")?.into_dimensionality()?);
    let z_x_input = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_x_input.npy")?.into_scalar());

    // --- Projection loaders ---
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

    let load_scaling = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ScalingBundle<E::ScalarField>> {
        Ok(ScalingBundle {
            q_scaled: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_scaled_{}.npy", suffix))?.into_dimensionality()?),
            r_scale: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_scale_{}.npy", suffix))?.into_dimensionality()?),
            m_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_scale_{}.npy", suffix))?.into_scalar()),
            m_prime_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_scale_{}.npy", suffix))?.into_scalar()),
        })
    };

    let load_matmul = |npz: &mut NpzReader<File>, suffix: &str, has_rmean: bool| -> Result<MatmulBundle<E::ScalarField>> {
        Ok(MatmulBundle {
            r_mean: if has_rmean {
                to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_mean_{}.npy", suffix))?.into_dimensionality()?)
            } else {
                vec![]
            },
            q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_sim_{}.npy", suffix))?.into_dimensionality()?),
            r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_{}.npy", suffix))?.into_dimensionality()?),
            m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", suffix))?.into_scalar()),
            m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", suffix))?.into_scalar()),
            z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_sim_{}.npy", suffix))?.into_scalar()),
        })
    };

    let proj_q = load_projection(&mut npz, "q")?;
    let proj_v = load_projection(&mut npz, "v")?;
    let scale_q = load_scaling(&mut npz, "q")?;

    let z_attn_output = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_attn_output.npy")?.into_scalar());
    let q_attn_output = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_attn_output.npy")?.into_dimensionality()?);
    let proj_o = load_projection(&mut npz, "o")?;

    // --- Cross-term matmuls ---
    let mut cross_b = load_matmul(&mut npz, "cross_B", false)?;
    // R_mean is for landmark segment averaging, stored in cross_b for convenience.
    // NPZ key is "R_mean_q" (not "R_mean_cross_B").
    cross_b.r_mean = to_field_4d_i64(&npz.by_name::<_, Ix4>("R_mean_q.npy")?.into_dimensionality()?);
    let cross_a = load_matmul(&mut npz, "cross_A", false)?;

    // --- Gaussian bundles ---
    let gauss_b = GaussianBundle {
        kernel: to_field_4d_u64(&npz.by_name::<_, Ix4>("kernel_B.npy")?.into_dimensionality()?),
        table_idx: to_field_4d_i64(&npz.by_name::<_, Ix4>("table_idx_B.npy")?.into_dimensionality()?),
        // TODO: Python saves r_gauss_B as object dtype (big int). Ensure i64 save in pipeline.
        r_gauss: to_field_4d_i64(&npz.by_name::<_, Ix4>("r_gauss_B.npy")?.into_dimensionality()?),
        m_gauss: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_gauss_B.npy")?.into_scalar()),
    };

    let gauss_a = GaussianBundle {
        kernel: to_field_4d_u64(&npz.by_name::<_, Ix4>("kernel_A.npy")?.into_dimensionality()?),
        table_idx: to_field_4d_i64(&npz.by_name::<_, Ix4>("table_idx_A.npy")?.into_dimensionality()?),
        r_gauss: to_field_4d_i64(&npz.by_name::<_, Ix4>("r_gauss_A.npy")?.into_dimensionality()?),
        m_gauss: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_gauss_A.npy")?.into_scalar()),
    };

    // --- Squared norms ---
    let norm_q = to_field_3d_i64(&npz.by_name::<_, Ix3>("norm_q_int.npy")?.into_dimensionality()?);
    let norm_qtilde = to_field_3d_i64(&npz.by_name::<_, Ix3>("norm_qtilde_int.npy")?.into_dimensionality()?);

    // --- Dist bundles ---
    let dist_b = DistBundle {
        q_dist: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_dist_B.npy")?.into_dimensionality()?),
        r_dist: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_dist_B.npy")?.into_dimensionality()?),
        m_a: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_A_B.npy")?.into_scalar()),
        m_b: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_B_B.npy")?.into_scalar()),
        z_dist: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_dist_B.npy")?.into_scalar()),
    };

    let dist_a = DistBundle {
        q_dist: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_dist_A.npy")?.into_dimensionality()?),
        r_dist: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_dist_A.npy")?.into_dimensionality()?),
        m_a: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_A_A.npy")?.into_scalar()),
        m_b: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_B_A.npy")?.into_scalar()),
        z_dist: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_dist_A.npy")?.into_scalar()),
    };

    // --- Pseudoinverse ---
    let q_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_inv_A.npy")?.into_dimensionality()?;
    let z_inv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_inv.npy")?.into_scalar();
    let m_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_inv_all.npy")?;
    let m_prime_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_prime_inv_all.npy")?;
    let q_y_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("Q_y_inv_intermediates.npy")?;
    let r_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("R_inv_remainders.npy")?;

    let pseudo_inv = PseudoinverseBundle {
        q_inv: to_field_4d_i64(&q_inv_raw),
        z_inv: E::ScalarField::from(z_inv),
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

    // --- Chain matmuls ---
    let matmul_temp = load_matmul(&mut npz, "temp", false)?;
    let matmul_mid = load_matmul(&mut npz, "mid", false)?;
    let matmul_x = load_matmul(&mut npz, "Y", false)?;  // NPZ uses suffix "Y"

    // --- Mask ---
    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    // --- Masked witnesses ---
    let q_q_masked = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_q_masked.npy")?.into_dimensionality()?);
    let q_v_masked = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_v_masked.npy")?.into_dimensionality()?);

    // --- Conv (optional) ---
    let conv = match npz.by_name::<ndarray::OwnedRepr<i64>, Ix3>("Q_t_conv.npy") {
        Ok(q_t_conv_raw) => {
            let q_t_conv_raw = q_t_conv_raw.into_dimensionality()?;
            let z_t_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_t_conv.npy")?.into_scalar();
            let m_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_conv.npy")?.into_scalar();
            let m_prime_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_conv.npy")?.into_scalar();
            let z_conv_out = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_conv_out.npy")?.into_scalar();
            let m_add_x = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_add_x.npy")?.into_scalar();
            let m_add_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_add_conv.npy")?.into_scalar();
            let z_xf = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_xf.npy")?.into_scalar();
            Some(ConvBundle {
                q_t_conv: to_field_3d_i64(&q_t_conv_raw),
                z_t_conv: E::ScalarField::from(z_t_conv),
                q_conv: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_conv.npy")?.into_dimensionality()?),
                r_conv: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_conv.npy")?.into_dimensionality()?),
                m_conv: E::ScalarField::from(m_conv),
                m_prime_conv: E::ScalarField::from(m_prime_conv),
                z_conv_out: E::ScalarField::from(z_conv_out),
                q_xf: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_xf.npy")?.into_dimensionality()?),
                r_xf: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_xf.npy")?.into_dimensionality()?),
                m_add_x: E::ScalarField::from(m_add_x),
                m_add_conv: E::ScalarField::from(m_add_conv),
                z_xf: E::ScalarField::from(z_xf),
            })
        }
        Err(_) => None,
    };

    let (eps1, eps1_src) = match npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("eps1.npy") {
        Ok(a) => (a.into_scalar() as usize, "npz"),
        Err(_) => (4096, "fallback"),
    };
    let (eps3, eps3_src) = match npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("eps3.npy") {
        Ok(a) => (a.into_scalar() as usize, "npz"),
        Err(_) => (2048, "fallback"),
    };
    let (eps4, eps4_src) = match npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("eps4.npy") {
        Ok(a) => (a.into_scalar() as usize, "npz"),
        Err(_) => (32, "fallback"),
    };
    let mp_log_r1 = eps_to_log_r(eps1);
    let mp_log_r3 = eps_to_log_r(eps3);
    let mp_log_r4 = eps_to_log_r(eps4);
    eprintln!(
        "[soft] eps1={} [{}] (log_r={}), eps3={} [{}] (log_r={}), eps4={} [{}] (log_r={})",
        eps1, eps1_src, mp_log_r1, eps3, eps3_src, mp_log_r3, eps4, eps4_src, mp_log_r4,
    );

    Ok(SOFTAttentionCircuit {
        z_q,
        z_v,
        q_x_input,
        z_x_input,
        proj_q,
        proj_v,
        scale_q,
        z_attn_output,
        q_attn_output,
        proj_o,
        cross_b,
        cross_a,
        gauss_b,
        gauss_a,
        norm_q,
        norm_qtilde,
        dist_b,
        dist_a,
        pseudo_inv,
        matmul_temp,
        matmul_mid,
        matmul_x,
        frac_bits,
        r_challenge: E::ScalarField::zero(),
        seg_size,
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        eps1,
        eps3,
        eps4,
        mp_log_r1,
        mp_log_r3,
        mp_log_r4,
        input_hash: E::ScalarField::zero(),
        output_hash: E::ScalarField::zero(),
        committed_vars: Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
        committed_vars_aux: Rc::new(RefCell::new(Vec::new())),
        conv,
        q_q_masked,
        q_v_masked,
    })
}

// =============================================================================
// B11: Full pipeline (Groth16 + Lasso + CP-Link)
// =============================================================================

pub fn run_soft_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
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
        println!("\n=== [SOFT] Processing witness NPZ: {:?} ===", file_path);

        // Dry-run: extract committed witness indices
        let (d1_indices, d2_indices, d3_indices) = {
            let temp_circuit = load_soft_circuit::<E>(file_path)?;
            let vars1_ptr = Rc::clone(&temp_circuit.committed_vars);
            let vars2_ptr = Rc::clone(&temp_circuit.committed_vars_2);
            let vars_aux_ptr = Rc::clone(&temp_circuit.committed_vars_aux);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;
            (
                crate::util::utils::extract_witness_indices(&vars1_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&vars2_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&vars_aux_ptr.borrow()),
            )
        };
        let num_commit_1 = d1_indices.len() as u32;
        let num_commit_2 = d2_indices.len() as u32;
        let num_commit_3 = d3_indices.len() as u32;
        println!("d1={}, d2={}, d3={}", num_commit_1, num_commit_2, num_commit_3);

        let sample_circuit = load_soft_circuit::<E>(file_path)?;

        // [0] Setup
        println!("=== [0] LegoGroth16 Setup (d1={}, d2={}, d3={}) ===", num_commit_1, num_commit_2, num_commit_3);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(), &d1_indices, &d2_indices, &d3_indices, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();
        let pvk = prepare_verifying_key::<E>(&params.vk);

        // Witness extraction + constraint count
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
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

        // Global weight commitment c_W
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

        // Multi-commitment Fiat-Shamir challenge
        let r_fs = hash_commitments_to_field_v3::<E>(&c_w, &cm1, &cm2, &cm_aux, &input_hash, &output_hash);

        println!("=== [1] Building final circuit with Fiat-Shamir challenge ===");
        let final_circuit = SOFTAttentionCircuit {
            r_challenge: r_fs,
            input_hash,
            output_hash,
            ..sample_circuit.clone()
        };

        // R1CS constraint satisfaction check
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

        // [1] Groth16 Prove
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(), v1, v2, v3, &params, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;
        let groth16_prove_time = groth16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // [1] Groth16 Verify
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
            return Err(anyhow::anyhow!("Groth16 verify failed: {:?}", e));
        }

        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // [2] Lasso (Gaussian bundles + remainders + MP conditions)
        proof_size::reset_lasso_aggregator();
        println!("=== [2] Lasso Lookup Proofs: SOFT Gaussian + Remainders ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();

        let (lasso_gauss_b, gb_agg) = run_soft_lasso_bundle(&final_circuit.gauss_b.table_idx, b"soft_gauss_b", &mut rng)?;
        lasso_agg.add_aggregate(&gb_agg);
        let (lasso_gauss_a, ga_agg) = run_soft_lasso_bundle(&final_circuit.gauss_a.table_idx, b"soft_gauss_a", &mut rng)?;
        lasso_agg.add_aggregate(&ga_agg);

        let (lasso_remainders, rem_agg) = run_all_soft_remainders_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let (mp_results, mp_agg) = run_soft_mp_conditions_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&mp_agg);

        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = vec![];
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

        // [4] CP-Link #2: c_W <-> d2 (global weight commitment link)
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

        // Grand total proof size report
        proof_size::print_full_report(&g16_report, &cplink_lasso_report, &cplink_global_report);

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "soft_attention".to_string(),
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
fn soft_attn_single_layer_test() {
    use std::env;

    let witness_dir = env::var("SOFT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_soft".to_string());
    let layer_idx: Option<usize> = env::var("SOFT_ATTN_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] SOFT_ATTN_WITNESS_DIR={} not a directory.",
            witness_dir
        );
        return;
    }

    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name()
                .and_then(|n| n.to_str())
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
            panic!("SOFT_ATTN_LAYER_IDX={} out of range.", idx);
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!("\n=== [SOFT Attention] dir={} | files={} ===", witness_dir, targets.len());
    run_soft_paths::<Bn254>(&targets).unwrap();
}

// =============================================================================
// B-gate: Rejection test helpers
// =============================================================================

fn load_small_soft_fixture() -> Option<SOFTAttentionCircuit<Fr>> {
    let witness_dir = std::env::var("SOFT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_n8_m4_L1_h8_soft_bgate".to_string());
    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!("[skip] SOFT_ATTN_WITNESS_DIR={} not found", witness_dir);
        return None;
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
    load_soft_circuit::<Bn254>(&targets[0]).ok()
}

fn prepare_soft_hashes(c: &mut SOFTAttentionCircuit<Fr>) {
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
    c.r_challenge = Fr::from(42u64);
}

fn assert_soft_r1cs_rejects(circuit: SOFTAttentionCircuit<Fr>, label: &str) {
    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.generate_constraints(cs.clone()).expect("constraint gen");
    let is_sat = cs.is_satisfied().unwrap_or(false);
    if is_sat {
        panic!("[REJECT FAIL] {} -- circuit should be unsatisfied but is_satisfied=true", label);
    }
    let unsat = cs.which_is_unsatisfied().unwrap();
    println!("[REJECT OK] {} -- unsatisfied at {:?}", label, unsat);
}

// =============================================================================
// B-gate: 18 rejection tests
// =============================================================================

// --- 1-3: Weight projection tamper ---

#[test]
fn soft_reject_wq_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.proj_q.q_w_diff[0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "W_Q tamper");
}

#[test]
fn soft_reject_wv_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.proj_v.q_w_diff[0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "W_V tamper");
}

#[test]
fn soft_reject_wo_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.proj_o.q_w_diff[0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "W_O tamper");
}

// --- 4: Landmark tamper ---

#[test]
fn soft_reject_landmark_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.cross_b.r_mean[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "R_mean_q (landmark) tamper");
}

// --- 5-6: Squared norm tamper ---

#[test]
fn soft_reject_norm_q_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.norm_q[0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "norm_q tamper");
}

#[test]
fn soft_reject_norm_qtilde_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.norm_qtilde[0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "norm_qtilde tamper");
}

// --- 7-8: Cross-term matmul tamper ---

#[test]
fn soft_reject_cross_b_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.cross_b.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "C_B (cross_b) tamper");
}

#[test]
fn soft_reject_cross_a_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.cross_a.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "C_A (cross_a) tamper");
}

// --- 9-10: Distance equation tamper ---

#[test]
fn soft_reject_q_dist_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.dist_b.q_dist[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "Q_dist_B tamper");
}

#[test]
fn soft_reject_r_dist_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.dist_b.r_dist[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "R_dist_B tamper");
}

// --- 11-12: Gaussian lookup tamper ---
// Note: test 11 (table_idx_B) is caught by the Gaussian lookup R1CS (constraint ~1204-1300).
// Test 12 (kernel_B) is a free witness not directly bound by Gaussian lookup R1CS;
// it is caught downstream by chain matmul temp (B^T @ V) Freivalds (constraint ~1684-2136).
// This is expected: comm_derefs linkage is not implemented, so kernel values
// propagate only through arithmetic constraints that consume them.

#[test]
fn soft_reject_gauss_table_idx_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.gauss_b.table_idx[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "table_idx_B (Gaussian) tamper");
}

#[test]
fn soft_reject_gauss_kernel_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.gauss_b.kernel[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "kernel_B (Gaussian) tamper");
}

// --- 13: Pseudoinverse tamper ---

#[test]
fn soft_reject_pseudo_inv_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.pseudo_inv.q_inv[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "A^+ (pseudo_inv) tamper");
}

// --- 14-16: Chain matmul tamper ---

#[test]
fn soft_reject_temp_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.matmul_temp.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "temp (B^T @ V) tamper");
}

#[test]
fn soft_reject_mid_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.matmul_mid.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "mid (B @ A^+) tamper");
}

#[test]
fn soft_reject_y_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    c.matmul_x.q_sim[0][0][0][0] += Fr::from(1u64);
    assert_soft_r1cs_rejects(c, "Y (mid @ temp) tamper");
}

// --- 17-18: Mask constraint tamper at padding ---

#[test]
fn soft_reject_mask_q_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    let s_len = c.mask[0].len();
    let mut pad_idx = None;
    for s in 0..s_len {
        if c.mask[0][s] == Fr::from(0u64) { pad_idx = Some(s); break; }
    }
    let s = pad_idx.expect("no padding position in mask");
    let num_heads = c.q_q_masked[0].len();
    let d_dim = c.q_q_masked[0][0][s].len();
    for h in 0..num_heads {
        for d in 0..d_dim {
            c.q_q_masked[0][h][s][d] = c.z_q + Fr::from(100u64);
        }
    }
    assert_soft_r1cs_rejects(c, "mask Q_proc tamper at padding");
}

#[test]
fn soft_reject_mask_v_tamper() {
    let mut c = match load_small_soft_fixture() {
        Some(c) => c, None => { eprintln!("[skip]"); return; }
    };
    prepare_soft_hashes(&mut c);
    let s_len = c.mask[0].len();
    let mut pad_idx = None;
    for s in 0..s_len {
        if c.mask[0][s] == Fr::from(0u64) { pad_idx = Some(s); break; }
    }
    let s = pad_idx.expect("no padding position in mask");
    let num_heads = c.q_v_masked[0].len();
    let d_dim = c.q_v_masked[0][0][s].len();
    for h in 0..num_heads {
        for d in 0..d_dim {
            c.q_v_masked[0][h][s][d] = c.z_v + Fr::from(100u64);
        }
    }
    assert_soft_r1cs_rejects(c, "mask V tamper at padding");
}
