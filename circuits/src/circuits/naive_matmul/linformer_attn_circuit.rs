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
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        freivalds_matmul::linformer_attn_circuit::{LinformerAttentionCircuit, load_linformer_circuit},
    },
    gadgets::{
        matmul::{
            constraints::new_matrix_witness,
            quantized::{ZenParams, new_matrix_witness_4d, transpose_matrix, enforce_scaling},
            naive_quantized::enforce_naive_matmul,
        },
        softmax::constraints::enforce_softmax,
    },
    util::{
        linformer_attn_lookup::{run_linformer_lasso_bundle, run_all_linformer_remainders_with_link},
        utils::{cm1_from_vk, cm2_from_vk, generate_global_ck, commit_global, hash_commitments_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;
use crate::circuits::proof_size;

type G1 = G1Projective;

/// Newtype wrapper that uses naive O(n^3) matmul instead of Freivalds.
#[derive(Clone)]
pub struct LinformerAttentionCircuitNot<F: PrimeField>(pub LinformerAttentionCircuit<F>);

impl<F: PrimeField> ConstraintSynthesizer<F> for LinformerAttentionCircuitNot<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let inner = self.0;
        inner.committed_vars.borrow_mut().clear();
        inner.committed_vars_2.borrow_mut().clear();

        let _r_var = FpVar::new_input(cs.clone(), || Ok(inner.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << inner.k_bits))?;

        let mut r_div_lows = Vec::new();
        let mut r_div_highs = Vec::new();

        // === Allocate intermediates ===
        let q_proc = new_matrix_witness_4d(cs.clone(), &inner.scale_q.q_scaled);
        let k_proc = new_matrix_witness_4d(cs.clone(), &inner.scale_k.q_scaled);
        let v_proc = new_matrix_witness_4d(cs.clone(), &inner.proj_v.q_out_proj);

        let num_heads = inner.proj_q.q_w_diff.len();
        let w_q_vars = alloc_weight_vars_transposed(cs.clone(), &inner.proj_q.q_w_diff);
        let w_k_vars = alloc_weight_vars_transposed(cs.clone(), &inner.proj_k.q_w_diff);
        let w_v_vars = alloc_weight_vars_transposed(cs.clone(), &inner.proj_v.q_w_diff);
        let w_o_vars = alloc_weight_vars_transposed(cs.clone(), &inner.proj_o.q_w_diff);

        // E_i, F_i: direct indexing [H][k][n]
        let e_vars = alloc_weight_vars_direct(cs.clone(), &inner.q_w_diff_e);
        let f_vars = alloc_weight_vars_direct(cs.clone(), &inner.q_w_diff_f);

        // d2: W_Q/K/V/O + E_i + F_i
        {
            let mut vars_2 = inner.committed_vars_2.borrow_mut();
            for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_k_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &e_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
            for h_w in &f_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        }

        let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.q_x_input.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

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

        // O-proj input
        let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = inner.q_attn_output.iter().map(|batch| {
            batch.iter().map(|seq| {
                seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();

        // Linformer matmul witnesses
        // z_w=0: e_vars/f_vars are already (Q-z) diff form; ZEN subtracts z_w, so z_w=z_e would double-subtract.
        let (ek_data, params_ek) = inner.matmul_ek.alloc_and_setup(cs.clone(), F::zero(), inner.z_k, &two_k_var);
        let (fv_data, params_fv) = inner.matmul_fv.alloc_and_setup(cs.clone(), F::zero(), inner.z_v, &two_k_var);
        let (logit_data, params_logit) = inner.matmul_logit.alloc_and_setup(cs.clone(), inner.z_q, inner.matmul_ek.z_sim, &two_k_var);
        let (x_data, params_x) = inner.matmul_x.alloc_and_setup(cs.clone(), F::zero(), inner.matmul_fv.z_sim, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &inner.mask);
        let sm_vars = inner.sm.alloc_vars(cs.clone())?;

        // Remainder pools
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
        let z_k_mask = FpVar::Constant(inner.z_k);
        let z_q_mask = FpVar::Constant(inner.z_q);
        let z_v_mask = FpVar::Constant(inner.z_v);

        for b in 0..q_proc.len() {
            for h in 0..q_proc[b].len() {
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
                // K Projection
                {
                    let proj_k_params = ZenParams {
                        m: FpVar::Constant(inner.proj_k.m_proj),
                        m_prime: FpVar::Constant(inner.proj_k.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_x_input),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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
                // Q/K Scaling
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

                // --- Masking: set K_proc, Q_proc, V to zero-point at padding ---
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

                // E_i @ K_proc (naive)
                enforce_naive_matmul(
                    cs.clone(), &e_vars[h], &masked_k,
                    &ek_data.q_sim[b][h], &ek_data.r_matmul[b][h],
                    &inner.matmul_ek.r_matmul[b][h],
                    &params_ek,
                    &mut global_ek_lows, &mut global_ek_highs, None,
                )?;

                // F_i @ V (naive)
                enforce_naive_matmul(
                    cs.clone(), &f_vars[h], &masked_v,
                    &fv_data.q_sim[b][h], &fv_data.r_matmul[b][h],
                    &inner.matmul_fv.r_matmul[b][h],
                    &params_fv,
                    &mut global_fv_lows, &mut global_fv_highs, None,
                )?;

                // Q_proc @ EK^T (naive)
                {
                    let ek_t = transpose_matrix(&ek_data.q_sim[b][h]);
                    enforce_naive_matmul(
                        cs.clone(), &masked_q, &ek_t,
                        &logit_data.q_sim[b][h], &logit_data.r_matmul[b][h],
                        &inner.matmul_logit.r_matmul[b][h],
                        &params_logit,
                        &mut global_logit_lows, &mut global_logit_highs, None,
                    )?;
                }

                // Softmax
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
                    &inner.sm.r_div[b][h],
                    inner.frac_bits,
                    &two_k_var,
                    &mut global_sm_idx,
                    &mut global_sm_r_sm,
                    &mut r_div_lows,
                    &mut r_div_highs,
                )?;

                // P @ FV (naive)
                enforce_naive_matmul(
                    cs.clone(), &sm_vars.p[b][h], &fv_data.q_sim[b][h],
                    &x_data.q_sim[b][h], &x_data.r_matmul[b][h],
                    &inner.matmul_x.r_matmul[b][h],
                    &params_x,
                    &mut global_x_lows, &mut global_x_highs, None,
                )?;

                // O Projection (naive)
                {
                    let o_input = &attn_output_vars[b];
                    let d_dim = inner.proj_o.q_w_diff[h].len();
                    let hd_dim = inner.proj_o.q_w_diff[h][0].len();
                    let w_o_h_t: Vec<Vec<FpVar<F>>> = (0..hd_dim)
                        .map(|hd| (0..d_dim).map(|d| FpVar::Constant(inner.proj_o.q_w_diff[h][d][hd])).collect())
                        .collect();
                    let proj_o_params = ZenParams {
                        m: FpVar::Constant(inner.proj_o.m_proj),
                        m_prime: FpVar::Constant(inner.proj_o.m_prime_proj),
                        z_w: FpVar::Constant(inner.z_attn_output),
                        z_x: FpVar::zero(), z_y: FpVar::zero(),
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

        // d1 committed_vars
        {
            let mut final_vars = inner.committed_vars.borrow_mut();
            final_vars.extend(global_sm_idx);
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
            for k in 0..4 { final_vars.extend(global_sm_r_sm[k].clone()); }
            final_vars.extend(r_div_lows);
            final_vars.extend(r_div_highs);
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

fn alloc_weight_vars_transposed<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_w_diff: &Vec<Vec<Vec<F>>>,
) -> Vec<Vec<Vec<FpVar<F>>>> {
    let num_heads = q_w_diff.len();
    (0..num_heads).map(|h| {
        let d_dim = q_w_diff[h].len();
        let hd_dim = q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect()
}

fn alloc_weight_vars_direct<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_w_diff: &Vec<Vec<Vec<F>>>,
) -> Vec<Vec<Vec<FpVar<F>>>> {
    q_w_diff.iter().map(|head| {
        head.iter().map(|row| {
            row.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
        }).collect()
    }).collect()
}


#[test]
fn linformer_attn_naive_single_layer_test() {
    use std::env;

    let witness_dir = env::var("NAIVE_LINFORMER_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_linformer".to_string());
    let layer_idx: Option<usize> = env::var("NAIVE_LINFORMER_ATTN_LAYER_IDX")
        .ok().and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!("[skip] NAIVE_LINFORMER_ATTN_WITNESS_DIR={} not a directory.", witness_dir);
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
        if idx >= all_targets.len() {
            panic!("NAIVE_LINFORMER_ATTN_LAYER_IDX={} out of range.", idx);
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!("\n=== [Linformer Attention NAIVE] dir={} | files={} ===", witness_dir, targets.len());
    eprintln!("[info] Naive matmul test: would require run_linformer_attention_not_paths (not implemented yet)");
}
