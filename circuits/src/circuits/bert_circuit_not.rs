use std::{path::Path, str::FromStr, time::Instant};
use anyhow::Result;
use ark_ec::pairing::Pairing;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{SeedableRng, rngs::StdRng},
    cell::RefCell,
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, fields::{fp::FpVar, FieldVar}, R1CSVar, select::CondSelectGadget};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        bert_circuit::{BertAttentionCircuit, load_bert_attention_circuit},
    },
    gadgets::{
        matmul::{
            constraints::new_matrix_witness,
            quantized::{new_matrix_witness_4d, transpose_matrix},
            naive_quantized::enforce_naive_matmul,  // ← CHANGED: naive O(n^3) matmul
        },
        softmax::constraints::enforce_softmax,
    },
    util::{
        bert_lookup_attention::{run_bert_lasso_bundle, run_all_bert_remainders_with_link},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};

type G1 = G1Projective;

/// Newtype wrapper around `BertAttentionCircuit` that uses naive matmul
/// instead of Freivalds for `generate_constraints`.
#[derive(Clone)]
pub struct BertAttentionCircuitNot<F: PrimeField>(pub BertAttentionCircuit<F>);

impl<F: PrimeField> ConstraintSynthesizer<F> for BertAttentionCircuitNot<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let inner = self.0;
        inner.committed_vars.borrow_mut().clear();
        inner.committed_vars_2.borrow_mut().clear();

        // Public input: Fiat-Shamir challenge r (kept for interface compatibility)
        let _r_var = FpVar::new_input(cs.clone(), || Ok(inner.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << inner.k_bits))?;

        let mut r2_pool = Vec::new();
        let mut r3_pool = Vec::new();

        // Q/K/V: [B][H][S][D]
        let q_proc = new_matrix_witness_4d(cs.clone(), &inner.q_q);
        let k_proc = new_matrix_witness_4d(cs.clone(), &inner.q_k);
        let v_proc = new_matrix_witness_4d(cs.clone(), &inner.q_v);

        // d2: commit Q, K, V
        {
            let mut vars_2 = inner.committed_vars_2.borrow_mut();
            for batch in &q_proc {
                for head in batch {
                    for seq in head {
                        vars_2.extend(seq.iter().cloned());
                    }
                }
            }
            for batch in &k_proc {
                for head in batch {
                    for seq in head {
                        vars_2.extend(seq.iter().cloned());
                    }
                }
            }
            for batch in &v_proc {
                for head in batch {
                    for seq in head {
                        vars_2.extend(seq.iter().cloned());
                    }
                }
            }
        }

        let (attn_data, params_attn) =
            inner.matmul_attn.alloc_and_setup(cs.clone(), inner.z_q, inner.z_k, &two_k_var);
        let (x_data, params_x) =
            inner.matmul_x.alloc_and_setup(cs.clone(), F::zero(), inner.z_v, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &inner.mask);
        let sm_vars = inner.sm.alloc_vars(cs.clone())?;

        let mut global_attn_lows  = Vec::new();
        let mut global_attn_highs = Vec::new();
        let mut global_sm_idx   = vec![Vec::new(); 5];
        let mut global_sm_r_sm  = vec![Vec::new(); 2];
        let mut global_x_lows   = Vec::new();
        let mut global_x_highs  = Vec::new();

        let zero_var = FpVar::zero();

        for b in 0..q_proc.len() {
            for h in 0..q_proc[b].len() {
                // [1] QK^T: CHANGED: enforce_freivald → enforce_naive_matmul
                let k_t = transpose_matrix(&k_proc[b][h]);
                let mut h_attn_lows  = Vec::new();
                let mut h_attn_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &q_proc[b][h],
                    &k_t,
                    &attn_data.q_sim[b][h],
                    &attn_data.r_matmul[b][h],
                    &inner.matmul_attn.r_matmul[b][h],
                    &params_attn,
                    &mut h_attn_lows,
                    &mut h_attn_highs,
                )?;
                global_attn_lows.extend(h_attn_lows);
                global_attn_highs.extend(h_attn_highs);

                // [2] Mask padding (unchanged)
                let seq_len = attn_data.q_sim[b][h].len();
                let col_len = attn_data.q_sim[b][h][0].len();
                let mut masked_q_sim = attn_data.q_sim[b][h].clone();
                for row in 0..seq_len {
                    for col in 0..col_len {
                        let is_valid = mask_var[b][col].is_one()?;
                        masked_q_sim[row][col] = FpVar::conditionally_select(
                            &is_valid,
                            &attn_data.q_sim[b][h][row][col],
                            &zero_var,
                        )?;
                    }
                }

                // [3] Softmax (unchanged)
                let mut h_sm_idx = [const { Vec::new() }; 5];
                let mut h_sm_r_sm = [const { Vec::new() }; 2];
                enforce_softmax(
                    cs.clone(),
                    &masked_q_sim,
                    &attn_data.r_matmul[b][h],
                    &sm_vars.r_sm[b][h],
                    &sm_vars.m_sm,
                    &sm_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.p[b][h],
                    &sm_vars.r_div[b][h],
                    inner.frac_bits,
                    &two_k_var,
                    &mut h_sm_idx,
                    &mut h_sm_r_sm,
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm_idx[k].extend(h_sm_idx[k].clone()); }
                for k in 0..2 { global_sm_r_sm[k].extend(h_sm_r_sm[k].clone()); }

                // [4] P * V: CHANGED: enforce_freivald → enforce_naive_matmul
                let mut h_x_lows  = Vec::new();
                let mut h_x_highs = Vec::new();
                enforce_naive_matmul(
                    cs.clone(),
                    &sm_vars.p[b][h],
                    &v_proc[b][h],
                    &x_data.q_sim[b][h],
                    &x_data.r_matmul[b][h],
                    &inner.matmul_x.r_matmul[b][h],
                    &params_x,
                    &mut h_x_lows,
                    &mut h_x_highs,
                )?;
                global_x_lows.extend(h_x_lows);
                global_x_highs.extend(h_x_highs);
            }
        }

        // Populate d1 committed_vars (unchanged)
        {
            let mut final_vars = inner.committed_vars.borrow_mut();
            for seg in 0..5 {
                final_vars.extend(global_sm_idx[seg].clone());
            }
            final_vars.extend(global_attn_lows);
            final_vars.extend(global_attn_highs);
            for k in 0..2 {
                final_vars.extend(global_sm_r_sm[k].clone());
            }
            final_vars.extend(global_x_lows);
            final_vars.extend(global_x_highs);
            final_vars.extend(r2_pool);
            final_vars.extend(r3_pool);
        }

        Ok(())
    }
}

// ── Proving pipeline ─────────────────────────────────────────────────────────

pub fn run_bert_attention_not_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [BERT Attention NOT] Processing: {:?} ===", file_path);

        // ── Dry-run: count committed witnesses ───────────────────────────────
        let (num_commit_1, num_commit_2) = {
            let temp = load_bert_attention_circuit::<E>(file_path)?;
            let v1_ptr = Rc::clone(&temp.committed_vars);
            let v2_ptr = Rc::clone(&temp.committed_vars_2);
            let cs = ConstraintSystem::<Fr>::new_ref();
            BertAttentionCircuitNot(temp).generate_constraints(cs)?;
            (v1_ptr.borrow().len() as u32, v2_ptr.borrow().len() as u32)
        };
        println!("Commit counts: d1={}, d2={}", num_commit_1, num_commit_2);

        let sample_circuit = load_bert_attention_circuit::<E>(file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (d1={}, d2={}) ===", num_commit_1, num_commit_2);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            BertAttentionCircuitNot(sample_circuit.clone()),
            num_commit_1,
            num_commit_2,
            &mut rng,
        )
        .map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction ────────────────────────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        BertAttentionCircuitNot(sample_circuit.clone()).generate_constraints(cs_witness.clone())?;
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

        println!("[Debug] committed_group1 total={}", committed_group1.len());

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit ───────────────────────────────────────────────────
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        let r_fs = hash_cm_to_field::<E>(&cm1);

        println!("=== [1] Rebuilding circuit with Fiat-Shamir challenge ===");
        let final_inner = BertAttentionCircuit {
            r_challenge: r_fs,
            ..sample_circuit.clone()
        };

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            BertAttentionCircuitNot(final_inner.clone()),
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
        let public_inputs = vec![final_inner.r_challenge];
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let g16_verify_time = g16_verify_start.elapsed();

        println!("  - BERT Attention NOT SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        // ── [2] Lasso (uses original circuit type — data fields identical) ───
        println!("=== [2] BERT Attention NOT Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let lasso_sm =
            run_bert_lasso_bundle(&final_inner.sm, b"bert_lasso_sm", &mut rng)?;
        let lasso_remainders =
            run_all_bert_remainders_with_link(&final_inner, &mut rng)?;
        let lasso_time = lasso_start.elapsed();

        let mut lasso_results = Vec::new();
        lasso_results.extend(lasso_sm);
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

        BenchmarkResult {
            file: file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit:           "bert_attention_not".to_string(),
            constraints:       num_constraints,
            setup_ms:          setup_time.as_millis(),
            groth16_commit_ms: g16_commit_ms,
            groth16_prove_ms:  g16_prove_time.as_millis(),
            groth16_verify_ms: g16_verify_time.as_millis(),
            lasso_ms:          lasso_time.as_millis(),
            cp_link_setup_ms:  cp_setup_ms,
            cp_link_prove_ms:  cp_prove_ms,
            cp_link_verify_ms: cp_verify_ms,
            end_to_end_ms:     e2e_time.as_millis(),
            proof_size_bytes:  proof_size,
            groth16_commitment_size_bytes: groth16_commitment_size::<E>(&cm1),
            lasso_commitment_size_bytes:   lasso_commitment_size::<E>(&lasso_results),
        }
        .print_json();
    }
    Ok(())
}

#[test]
fn bert_attention_not_test() {
    let outputs = Path::new("outputs_bert");
    if !outputs.is_dir() {
        return;
    }
    let mut targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(outputs).expect("reading outputs_bert dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention.self.witness.npz"))
        {
            targets.push(path);
        }
    }
    targets.sort();
    println!(
        "Running BERT attention NOT circuit test on {} witness file(s)",
        targets.len()
    );
    let res = run_bert_attention_not_paths::<Bn254>(&targets);
    if let Err(e) = res {
        panic!("BERT attention NOT proof failed: {}", e);
    }
}
