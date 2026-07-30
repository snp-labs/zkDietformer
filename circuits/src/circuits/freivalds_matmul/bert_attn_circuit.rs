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
        bert_attn_lookup::{run_bert_lasso_bundle, run_all_bert_remainders_with_link},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;

type G1 = G1Projective;

#[derive(Clone)]
pub struct BERTAttentionCircuit<F: PrimeField> {
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,

    // === Projection inputs ===
    pub q_x_input: Vec<Vec<Vec<F>>>,
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
    pub q_attn_output: Vec<Vec<Vec<F>>>,  // [B][S][HD] — O-proj input
    pub proj_o: ProjectionBundle<F>,

    /// QK^T matmul: q_sim/r_matmul shape [batch][head][seq][seq], r_mean empty
    pub matmul_attn: MatmulBundle<F>,

    /// Full-sequence softmax over [batch][head][seq][seq]
    pub sm: SoftmaxBundle<F>,

    /// P * V output: q_sim/r_matmul shape [batch][head][seq][head_dim], r_mean empty
    pub matmul_x: MatmulBundle<F>,

    pub frac_bits: usize,
    pub r_challenge: F,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,

    /// Poseidon hash of flatten(q_x_input) — native precomputed, declared as public input
    pub input_hash: F,
    /// Poseidon hash of flatten(proj_o_out) — native precomputed, declared as public input
    pub output_hash: F,

    /// d1 committed group: Lasso-linked wires (remainder splits, softmax indices)
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    /// d2 committed group: W_Q, W_K, W_V weights committed via v2, NOT CP-linked to Lasso
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for BERTAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();

        // Public input: Fiat-Shamir challenge r
        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        // v3: r2_pool removed (no exp chain in v3)
        let mut r_div_lows = Vec::new();
        let mut r_div_highs = Vec::new();

        // === Projection input allocation (witness) ===
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

        // O-projection input: separate witness
        let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = self.q_attn_output.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
        let mut global_proj_k_lows = Vec::new(); let mut global_proj_k_highs = Vec::new();
        let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
        let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

        // Allocate QK^T output witnesses; ZenParams for QK^T (z_left=z_q, z_right=z_k)
        let (attn_data, params_attn) =
            self.matmul_attn.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        // Allocate P*V output witnesses; ZenParams (z_left=0 for softmax output, z_right=z_v)
        let (x_data, params_x) =
            self.matmul_x.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);

        // Softmax: mask and bundle vars
        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm_vars = self.sm.alloc_vars(cs.clone())?;

        // ── Per-(batch, head) loop ────────────────────────────────────────────
        let mut global_attn_lows  = Vec::new();
        let mut global_attn_highs = Vec::new();
        // v3: single pool (C=1)
        let mut global_sm_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_sm_r_sm: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut global_x_lows   = Vec::new();
        let mut global_x_highs  = Vec::new();

        let zero_var = FpVar::zero();

        for b in 0..scale_q_out.len() {
            for h in 0..scale_q_out[b].len() {
                // === Q/K/V Projection + Q/K Scaling ===
                // Q Projection
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
                // K Projection
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
                // V Projection
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
                // Q Scaling
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
                // K Scaling
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

                // [1] QK^T: scale_q_out[B][H] [S×D] × scale_k_out[B][H]^T [D×S] → [S×S]
                let k_t = transpose_matrix(&scale_k_out[b][h]);
                let r_pows_attn =
                    gen_challenge_vector(&r_var, attn_data.q_sim[b][h][0].len());
                let mut h_attn_lows  = Vec::new();
                let mut h_attn_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &scale_q_out[b][h],
                    &k_t,
                    &attn_data.q_sim[b][h],
                    &attn_data.r_matmul[b][h],
                    &self.matmul_attn.r_matmul[b][h],
                    &params_attn,
                    &r_pows_attn,
                    &mut h_attn_lows,
                    &mut h_attn_highs,
                    None,
                )?;
                global_attn_lows.extend(h_attn_lows);
                global_attn_highs.extend(h_attn_highs);

                // [2] + [3] v3 softmax: no q_sim masking, mask applied to exp_total
                enforce_softmax(
                    cs.clone(),
                    &attn_data.q_sim[b][h],
                    &attn_data.r_matmul[b][h],
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

                // v3 mask: verify exp_total=0 at padding positions
                {
                    let seq_len = sm_vars.exp_total[b][h].len();
                    let col_len = sm_vars.exp_total[b][h][0].len();
                    for row in 0..seq_len {
                        for col in 0..col_len {
                            let is_valid = mask_var[b][col].is_one()?;
                            // If NOT valid (padding), exp_total must be zero
                            let should_be_zero = FpVar::conditionally_select(
                                &is_valid,
                                &zero_var,
                                &sm_vars.exp_total[b][h][row][col],
                            )?;
                            should_be_zero.enforce_equal(&zero_var)?;
                        }
                    }
                }

                // [4] P * V: sm_vars.p[B][H] [S×S] × proj_v_out[B][H] [S×D] → [S×D]
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len());
                let mut h_x_lows  = Vec::new();
                let mut h_x_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &sm_vars.p[b][h],
                    &proj_v_out[b][h],
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

                // === O Projection: O = attn_output @ W_o_h^T + b_o_h ===
                // Uses q_attn_output witness as input.
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

        // ── Populate d1 committed_vars ────────────────────────────────────────
        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // === Softmax index lookup vars (C=1, single pool) ===
            final_vars.extend(global_sm_idx);

            // === 26-bit remainders (C=2: 16+10) ===
            // Order must match run_all_bert_remainders_with_link target_26bit
            // QK^T matmul remainder splits (low / high)
            final_vars.extend(global_attn_lows);
            final_vars.extend(global_attn_highs);
            // P*V matmul remainder splits
            final_vars.extend(global_x_lows);
            final_vars.extend(global_x_highs);
            // Projection remainders
            final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
            final_vars.extend(global_proj_k_lows); final_vars.extend(global_proj_k_highs);
            final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
            final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);
            // Scaling remainders
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

            // === 52-bit r_sm (C=4: 16+16+16+4) ===
            for k in 0..4 {
                final_vars.extend(global_sm_r_sm[k].clone());
            }

            // === R_div per-softmax (26-bit C=2: 16+10) ===
            final_vars.extend(r_div_lows);
            final_vars.extend(r_div_highs);
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

            // 1. Public input: declared hash value
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


pub fn load_bert_attention_circuit<E: Pairing>(
    path: &Path,
) -> Result<BERTAttentionCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening BERT witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    // ── Inputs ───────────────────────────────────────────────────────────────
    let q_q_raw: ndarray::Array4<i64> = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?;
    let q_k_raw: ndarray::Array4<i64> = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?;
    let q_v_raw: ndarray::Array4<i64> = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_v.npy")?;
    let z_q = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar(),
    );
    let z_k = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar(),
    );
    let z_v = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar(),
    );

    // Shape: Q/K/V must be [B, H, S, D]
    assert_eq!(q_q_raw.ndim(), 4, "Q_q must be 4D [B, H, S, D]");
    assert_eq!(q_k_raw.ndim(), 4, "Q_k must be 4D [B, H, S, D]");
    assert_eq!(q_v_raw.ndim(), 4, "Q_v must be 4D [B, H, S, D]");
    assert_eq!(
        q_q_raw.shape()[1], q_k_raw.shape()[1],
        "Q and K must have the same number of heads"
    );
    assert_eq!(
        q_q_raw.shape()[2], q_k_raw.shape()[2],
        "Q and K must have the same seq_len"
    );
    assert_eq!(
        q_q_raw.shape()[3], q_k_raw.shape()[3],
        "Q and K must have the same head_dim"
    );
    assert_eq!(
        q_q_raw.shape()[2], q_v_raw.shape()[2],
        "Q and V must have the same seq_len"
    );
    assert_eq!(
        q_q_raw.shape()[3], q_v_raw.shape()[3],
        "Q and V must have the same head_dim"
    );

    let b = q_q_raw.shape()[0];
    let h = q_q_raw.shape()[1];
    let s = q_q_raw.shape()[2];
    let d = q_q_raw.shape()[3];

    // ── QK^T matmul ──────────────────────────────────────────────────────────
    let q_sim_attn =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_attn.npy")?.into_dimensionality()?;
    let r_matmul_attn =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_attn.npy")?.into_dimensionality()?;

    // Shape: [B, H, S, S]
    assert_eq!(
        q_sim_attn.shape(), &[b, h, s, s],
        "Q_sim_attn must be [B={b}, H={h}, S={s}, S={s}]"
    );
    assert_eq!(
        r_matmul_attn.shape(), q_sim_attn.shape(),
        "R_matmul_attn must match Q_sim_attn shape"
    );

    let matmul_attn = MatmulBundle {
        r_mean:   vec![],
        q_sim:    to_field_4d_i64(&q_sim_attn),
        r_matmul: to_field_4d_i64(&r_matmul_attn),
        m:        E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_attn.npy")?.into_scalar(),
        ),
        m_prime:  E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_attn.npy")?.into_scalar(),
        ),
        z_sim:    E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_attn.npy")?.into_scalar(),
        ),
    };

    // ── Metadata ─────────────────────────────────────────────────────────────
    let frac_bits =
        npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    // v3: limb_bits and k_segments removed from NPZ
    let k_bits =
        npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;

    // ── Softmax v3 ──────────────────────────────────────────────────────────
    // All 4D arrays: [B, H, S, S]
    let table_idx_raw =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("table_idx.npy")?.into_dimensionality()?;
    let exp_total_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("exp_total.npy")?.into_dimensionality()?;
    let p_attn_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("P_attn.npy")?.into_dimensionality()?;
    let r_div_attn_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("R_div_attn.npy")?.into_dimensionality()?;
    let r_sm_raw =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("r_sm.npy")?.into_dimensionality()?;

    // Shape: all must be [B, H, S, S]
    assert_eq!(table_idx_raw.shape(), &[b, h, s, s], "table_idx must be [B={b}, H={h}, S={s}, S={s}]");
    assert_eq!(p_attn_raw.shape(), &[b, h, s, s], "P_attn must be [B={b}, H={h}, S={s}, S={s}]");
    assert_eq!(exp_total_raw.shape(), p_attn_raw.shape(), "exp_total shape must match P_attn");
    assert_eq!(r_div_attn_raw.shape(), p_attn_raw.shape(), "R_div_attn shape must match P_attn");
    assert_eq!(r_sm_raw.shape(), p_attn_raw.shape(), "r_sm shape must match P_attn");

    let sm = SoftmaxBundle {
        table_idx: to_field_4d_i64(&table_idx_raw),
        exp_total: to_field_4d_u64(&exp_total_raw),
        p:         to_field_4d_u64(&p_attn_raw),
        r_div:     to_field_4d_u64(&r_div_attn_raw),
        r_sm:      to_field_4d_i64(&r_sm_raw),
        m_sm:      E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar(),
        ),
    };

    // ── P*V output matmul ─────────────────────────────────────────────────────
    let q_sim_x =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_x.npy")?.into_dimensionality()?;
    let r_matmul_x =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_x.npy")?.into_dimensionality()?;

    // Shape: [B, H, S, D]
    assert_eq!(
        q_sim_x.shape(), &[b, h, s, d],
        "Q_sim_x must be [B={b}, H={h}, S={s}, D={d}]"
    );
    assert_eq!(
        r_matmul_x.shape(), q_sim_x.shape(),
        "R_matmul_x must match Q_sim_x shape"
    );

    let matmul_x = MatmulBundle {
        r_mean:   vec![],
        q_sim:    to_field_4d_i64(&q_sim_x),
        r_matmul: to_field_4d_i64(&r_matmul_x),
        m:        E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_x.npy")?.into_scalar(),
        ),
        m_prime:  E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_x.npy")?.into_scalar(),
        ),
        z_sim:    E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_x.npy")?.into_scalar(),
        ),
    };

    // ── Mask ─────────────────────────────────────────────────────────────────
    let mask_raw =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    // Shape: [B, S]
    assert_eq!(
        mask_raw.shape(), &[b, s],
        "mask must be [B={b}, S={s}]"
    );

    // === Projection witness loading ===
    let q_x_input = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_x_input.npy")?.into_dimensionality()?);
    let z_x_input = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_x_input.npy")?.into_scalar());

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

    Ok(BERTAttentionCircuit {
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
        matmul_attn,
        sm,
        matmul_x,
        frac_bits,
        r_challenge: E::ScalarField::zero(),
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        input_hash: E::ScalarField::zero(),
        output_hash: E::ScalarField::zero(),
        committed_vars:   Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
    })
}

// ── Proving pipeline ─────────────────────────────────────────────────────────

pub fn run_bert_attention_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [BERT Attention] Processing: {:?} ===", file_path);

        // ── Dry-run: extract committed witness indices ───────────────────────
        let (d1_indices, d2_indices) = {
            let temp = load_bert_attention_circuit::<E>(file_path)?;
            let v1_ptr = Rc::clone(&temp.committed_vars);
            let v2_ptr = Rc::clone(&temp.committed_vars_2);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            (
                crate::util::utils::extract_witness_indices(&v1_ptr.borrow()),
                crate::util::utils::extract_witness_indices(&v2_ptr.borrow()),
            )
        };
        let num_commit_1 = d1_indices.len() as u32;
        let num_commit_2 = d2_indices.len() as u32;
        println!("Commit counts: d1={}, d2={}", num_commit_1, num_commit_2);

        let sample_circuit = load_bert_attention_circuit::<E>(file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (d1={}, d2={}) ===", num_commit_1, num_commit_2);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            &d1_indices,
            &d2_indices,
            &mut rng,
        )
        .map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction ────────────────────────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> =
                vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
            if values.len() != num_commit_1 as usize {
                return Err(anyhow::anyhow!(
                    "d1 size mismatch: expected {}, got {}",
                    num_commit_1,
                    values.len()
                ));
            }
            values
        };

        println!(
            "[Debug] committed_group1 total={}",
            committed_group1.len()
        );

        let v1 = Fr::rand(&mut rng);
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

        println!("=== [1] Rebuilding circuit with Fiat-Shamir challenge ===");
        let final_circuit = BERTAttentionCircuit {
            r_challenge: r_fs,
            input_hash,
            output_hash,
            ..sample_circuit.clone()
        };

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(),
            v1,
            v2,
            &params,
            &mut rng,
        )
        .map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
        let g16_prove_time = g16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof
            .serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // ── [1] Groth16 Verify ───────────────────────────────────────────────
        let public_inputs = vec![
            final_circuit.r_challenge,
            final_circuit.input_hash,
            final_circuit.output_hash,
        ];
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let g16_verify_time = g16_verify_start.elapsed();

        println!("  - BERT Attention SNARK Verification: {:?}", result);
        if let Err(e) = result {
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
        proof_size::reset_lasso_aggregator();
        println!("=== [2] BERT Attention Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let (lasso_sm, sm_agg) =
            run_bert_lasso_bundle(&final_circuit.sm, b"bert_lasso_sm", &mut rng)?;
        lasso_agg.add_aggregate(&sm_agg);
        let (lasso_remainders, rem_agg) =
            run_all_bert_remainders_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = Vec::new();
        lasso_results.extend(lasso_sm);
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
        proof_size::print_full_report(&g16_report, &cplink_report, &proof_size::CpLinkSizeReport { proof_g_count: 0, proof_g_bytes: 0, proof_f_count: 0, proof_f_bytes: 0, proof_total: 0, instance_g_count: 0, instance_g_bytes: 0 });

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "bert_attention".to_string(),
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
                setup_ms: cp_setup_ms,
                prove_ms: cp_prove_ms,
                verify_ms: cp_verify_ms,
                proof_size_bytes: cplink_report.proof_total,
            },
            cp_link_global: CpLinkMetrics {
                commit_ms: 0,
                setup_ms: 0,
                prove_ms: 0,
                verify_ms: 0,
                proof_size_bytes: 0,
            },
            end_to_end_ms: e2e_time.as_millis(),
        }
        .print_json();
    }
    Ok(())
}


#[test]
fn bert_attn_hf_single_layer_test() {
    use std::env;

    // ── 환경 변수 ───────────────────────────────────────────────────────────
    let witness_dir = env::var("BERT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_bert".to_string());
    let layer_idx: Option<usize> = env::var("BERT_ATTN_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] BERT_ATTN_WITNESS_DIR={} not a directory. \
             Set the env var to a valid HF BERT outputs_bert/ dir to run.",
            witness_dir
        );
        return;
    }

    // ── softmax 표 경로 결정: env var 미설정 시 witness_dir 안의 softmax_tables.npz ─
    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
        println!(
            "[info] SOFTMAX_TABLE_PATH not set; using {}/softmax_tables.npz",
            witness_dir
        );
    }

    // ── npz 수집 + 정렬 + layer_idx 단일 추출 ──────────────────────────────
    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path
                .file_name()
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
             Run HF BERT attention pipeline first.",
            witness_dir
        );
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!(
                "BERT_ATTN_LAYER_IDX={} out of range (have {} layers).",
                idx,
                all_targets.len()
            );
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [BERT Attention prove+verify] dir={} | files={} | mode={} ===",
        witness_dir,
        targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_bert_attention_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!(
            "\n=== [BERT Attention] ALL {} files PASS | total_wall_ms={} ===",
            targets.len(), total_ms,
        ),
        Err(e) => panic!(
            "[BERT Attention] proof generate/verify failed after {}ms: {}",
            total_ms, e,
        ),
    }
}
