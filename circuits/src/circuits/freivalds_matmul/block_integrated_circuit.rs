//! Version B: Per-Block Integrated Circuit
//!
//! Each transformer block (1 Attention + 1 MLP) is proven as a single Groth16 proof.
//! For a model with N layers, this produces N independent proofs.
//! Supports both Nystrom/BERT attention and RMSNorm/LayerNorm MLP variants.
//!
//! **Constraint**: Existing files in `freivalds_matmul/` are NOT modified.
//! Reuses common types and constraint helpers from `model_integrated_circuit`.

use std::{cell::RefCell, path::Path, str::FromStr, time::Instant};
use anyhow::Result;
use ark_ec::pairing::Pairing;
use ark_std::{
    rc::Rc, UniformRand,
    rand::{SeedableRng, rngs::StdRng},
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero};
use ark_r1cs_std::R1CSVar;
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{
    ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError,
};
use ark_r1cs_std::{
    alloc::AllocVar,
    fields::fp::FpVar,
};
use crate::{
    cc_snark::legogroth16::{
        create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    },
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics,
            LassoMetricsAggregate, factored_link_setup, factored_link_keygen,
            factored_link_prove, factored_link_verify, pad_lasso_u_values,
            groth16_commitment_size,
        },
        freivalds_matmul::model_integrated_circuit::{
            AttentionVariant, AttnWitness, MlpWitness, LassoResult,
            DiscoveredWitnesses,
            discover_witnesses,
            load_nystrom_attn_witness, load_bert_attn_witness, load_rmsnorm_mlp_data,
            generate_nystrom_attn_constraints, generate_bert_attn_constraints,
            generate_mlp_rmsnorm_constraints, generate_mlp_layernorm_constraints,
        },
        proof_size,
    },
    util::{
        norm_detect::NormType,
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;
use crate::lasso::subtables::layernorm::set_layernorm_table_dir;
use crate::lasso::subtables::gelu::set_gelu_table_dir;

type G1 = G1Projective;

// ═══════════════════════════════════════════════════════════════════════════════
// Version B: BlockIntegratedCircuit
// ═══════════════════════════════════════════════════════════════════════════════

/// A single transformer block (Attention + MLP) as one Groth16 circuit.
#[derive(Clone)]
pub struct BlockIntegratedCircuit<F: PrimeField> {
    /// This block's Attention witness
    pub attn: AttnWitness<F>,
    /// This block's MLP witness
    pub mlp: MlpWitness<F>,
    /// Layer index (for logging/labeling)
    pub layer_idx: usize,
    /// Quantization parameters (shared across components)
    pub frac_bits: usize,
    pub k_bits: usize,
    /// Fiat-Shamir challenge (set after cm1 hashing)
    pub r_challenge: F,
    /// d1: softmax indices + remainder splits (Attn + MLP combined, CP-Link target)
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    /// d2: Q, K, V raw values (CP-Link non-target)
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for BlockIntegratedCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();

        // Single public input: Fiat-Shamir challenge
        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        // Attention constraints (push to d1 + d2)
        match &self.attn {
            AttnWitness::Nystrom(w) => {
                generate_nystrom_attn_constraints(
                    cs.clone(), w, &r_var, &two_k_var,
                    &self.committed_vars, &self.committed_vars_2,
                )?;
            }
            AttnWitness::Bert(w) => {
                generate_bert_attn_constraints(
                    cs.clone(), w, &r_var, &two_k_var,
                    &self.committed_vars, &self.committed_vars_2,
                )?;
            }
        }

        // MLP constraints (push to d1)
        match &self.mlp {
            MlpWitness::LayerNorm(data) => {
                generate_mlp_layernorm_constraints(
                    cs.clone(), data, &self.committed_vars,
                )?;
            }
            MlpWitness::RMSNorm(data) => {
                generate_mlp_rmsnorm_constraints(
                    cs.clone(), data, &self.committed_vars,
                )?;
            }
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loading
// ═══════════════════════════════════════════════════════════════════════════════

/// Load a single block (layer_idx) as a BlockIntegratedCircuit.
pub fn load_block_integrated_circuit<E: Pairing>(
    discovered: &DiscoveredWitnesses,
    layer_idx: usize,
) -> Result<BlockIntegratedCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let attn = match discovered.attn_variant {
        AttentionVariant::Nystrom => {
            AttnWitness::Nystrom(load_nystrom_attn_witness::<E>(&discovered.attn_files[layer_idx])?)
        }
        AttentionVariant::Bert => {
            AttnWitness::Bert(load_bert_attn_witness::<E>(&discovered.attn_files[layer_idx])?)
        }
    };

    let (frac_bits, k_bits) = match &attn {
        AttnWitness::Nystrom(w) => (w.frac_bits, w.k_bits),
        AttnWitness::Bert(w) => (w.frac_bits, w.k_bits),
    };

    let mlp = match discovered.norm_variant {
        NormType::LayerNorm => {
            use crate::circuits::freivalds_matmul::nystrom_mlp_circuit::load_nystrom_layernorm_data;
            MlpWitness::LayerNorm(load_nystrom_layernorm_data::<E>(&discovered.mlp_files[layer_idx])?)
        }
        NormType::RMSNorm => {
            MlpWitness::RMSNorm(load_rmsnorm_mlp_data::<E>(&discovered.mlp_files[layer_idx])?)
        }
    };

    Ok(BlockIntegratedCircuit {
        attn,
        mlp,
        layer_idx,
        frac_bits,
        k_bits,
        r_challenge: E::ScalarField::zero(),
        committed_vars: Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Proving Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/// Full proving pipeline for block-integrated circuits (one proof per layer).
pub fn run_block_integrated_paths<E: Pairing>(
    witness_dir: &Path,
) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    use crate::util::model_integrated_lookup::{run_layer_attn_lasso, run_layer_mlp_lasso};

    let mut rng = StdRng::seed_from_u64(42u64);

    // Initialize all lookup tables
    if std::env::var("SOFTMAX_TABLE_PATH").is_err() { set_softmax_table_dir(witness_dir); }
    if std::env::var("LAYERNORM_TABLE_PATH").is_err() { set_layernorm_table_dir(witness_dir); }
    if std::env::var("GELU_TABLE_PATH").is_err() { set_gelu_table_dir(witness_dir); }

    let discovered = discover_witnesses(witness_dir)?;
    println!("\n=== [Block Integrated] {} layers, {:?} attn, {:?} norm ===",
        discovered.num_layers, discovered.attn_variant, discovered.norm_variant);

    // Optional: filter to a single layer via env var
    let layer_range: Vec<usize> = match std::env::var("BLOCK_LAYER_IDX") {
        Ok(val) => {
            let idx: usize = val.parse().expect("BLOCK_LAYER_IDX must be a number");
            assert!(idx < discovered.num_layers, "BLOCK_LAYER_IDX {} >= num_layers {}", idx, discovered.num_layers);
            vec![idx]
        }
        Err(_) => (0..discovered.num_layers).collect(),
    };

    for &layer_idx in &layer_range {
        let e2e_start = Instant::now();
        println!("\n--- [Block L{}] Start ---", layer_idx);

        // Load
        let circuit = load_block_integrated_circuit::<E>(&discovered, layer_idx)?;

        // Dry-run → d1/d2 indices
        let (d1_indices, d2_indices) = {
            let temp = load_block_integrated_circuit::<E>(&discovered, layer_idx)?;
            let v1 = Rc::clone(&temp.committed_vars);
            let v2 = Rc::clone(&temp.committed_vars_2);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            (
                crate::util::utils::extract_witness_indices(&v1.borrow()),
                crate::util::utils::extract_witness_indices(&v2.borrow()),
            )
        };
        let num_d1 = d1_indices.len() as u32;
        let num_d2 = d2_indices.len() as u32;
        println!("  Commit counts: d1={}, d2={}", num_d1, num_d2);

        // [0] Setup
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(circuit.clone(), &d1_indices, &d2_indices, &mut rng)
            .map_err(|e| anyhow::anyhow!("Setup L{} failed: {:?}", layer_idx, e))?;
        let setup_time = setup_start.elapsed();
        let pvk = prepare_verifying_key::<E>(&params.vk);

        // Witness extraction
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1: Vec<Fr> = circuit.committed_vars.borrow().iter()
            .map(|v| v.value().unwrap_or(Fr::zero())).collect();
        let num_pub = cs_witness.num_instance_variables() - 1;

        // Groth16 commit + Fiat-Shamir
        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);

        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();
        let r_fs = hash_cm_to_field::<E>(&cm1);

        // ── Compute Poseidon hashes for attention layer ─────────────────────
        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<Fr>();
        let mut circuit_with_hashes = circuit.clone();
        {
            let attn_w = &mut circuit_with_hashes.attn;
            // input_hash
            {
                let x_input = match attn_w {
                    AttnWitness::Nystrom(w) => &w.q_x_input,
                    AttnWitness::Bert(w) => &w.q_x_input,
                };
                let mut x_flat: Vec<Fr> = Vec::new();
                for batch in x_input {
                    for seq in batch {
                        x_flat.extend(seq.iter());
                    }
                }
                let ih = crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &x_flat);
                match attn_w {
                    AttnWitness::Nystrom(w) => w.input_hash = ih,
                    AttnWitness::Bert(w) => w.input_hash = ih,
                }
            }
            // output_hash
            {
                let proj_o_out = match attn_w {
                    AttnWitness::Nystrom(w) => &w.proj_o.q_out_proj,
                    AttnWitness::Bert(w) => &w.proj_o.q_out_proj,
                };
                let mut output_flat: Vec<Fr> = Vec::new();
                for batch in proj_o_out {
                    for head in batch {
                        for seq in head {
                            output_flat.extend(seq.iter());
                        }
                    }
                }
                let oh = crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &output_flat);
                match attn_w {
                    AttnWitness::Nystrom(w) => w.output_hash = oh,
                    AttnWitness::Bert(w) => w.output_hash = oh,
                }
            }
        }

        let final_circuit = BlockIntegratedCircuit {
            r_challenge: r_fs,
            ..circuit_with_hashes
        };

        // [1] Groth16 Prove
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(final_circuit.clone(), v1, v2, &params, &mut rng)
            .map_err(|e| anyhow::anyhow!("Proof L{} failed: {:?}", layer_idx, e))?;
        let g16_prove_time = g16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Serialization L{} failed: {:?}", layer_idx, e))?;
        let proof_size_bytes = proof_bytes.len();

        // [1] Groth16 Verify
        let g16_verify_start = Instant::now();
        let (input_hash, output_hash) = match &final_circuit.attn {
            AttnWitness::Nystrom(w) => (w.input_hash, w.output_hash),
            AttnWitness::Bert(w) => (w.input_hash, w.output_hash),
        };
        let result = verify_proof::<E>(&pvk, &proof, &[r_fs, input_hash, output_hash]);
        let g16_verify_time = g16_verify_start.elapsed();
        println!("  - Block L{} SNARK Verification: {:?}", layer_idx, result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("L{}: {:?}", layer_idx, e)));
        }

        let g16_report = proof_size::compute_groth16_report::<E>(&proof);

        // [2] Lasso (this block's Attn + MLP)
        proof_size::reset_lasso_aggregator();
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let mut lasso_results: Vec<LassoResult> = Vec::new();

        println!("  -- Layer {} Attention Lasso --", layer_idx);
        let (attn_res, attn_agg) = run_layer_attn_lasso(&final_circuit.attn, layer_idx, &mut rng)?;
        lasso_agg.add_aggregate(&attn_agg);
        lasso_results.extend(attn_res);

        println!("  -- Layer {} MLP Lasso --", layer_idx);
        let (mlp_res, mlp_agg) = run_layer_mlp_lasso(&final_circuit.mlp, layer_idx, &mut rng)?;
        lasso_agg.add_aggregate(&mlp_agg);
        lasso_results.extend(mlp_res);

        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        // Debug assertion
        let lasso_sp_sum: usize = lasso_results.iter().map(|(_, _, _, sp)| *sp).sum();
        debug_assert_eq!(
            committed_group1.len(), lasso_sp_sum,
            "L{}: committed_vars length ({}) must equal sum of Lasso sparsities ({})",
            layer_idx, committed_group1.len(), lasso_sp_sum
        );

        // [3] CP-Link
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
            return Err(anyhow::anyhow!("CP-Link verification failed for L{}", layer_idx));
        }
        println!("  -> [Success] Block L{} CP-Link verified", layer_idx);
        let cp_verify_ms = cp_verify_start.elapsed().as_millis();

        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        proof_size::print_full_report(&g16_report, &cplink_report);

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: format!("{}_layers", discovered.num_layers),
            circuit: format!("block_integrated_L{}", layer_idx),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: g16_commit_ms,
                prove_ms: g16_prove_time.as_millis(),
                verify_ms: g16_verify_time.as_millis(),
                proof_size_bytes,
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

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn block_integrated_per_layer_test() {
    use std::env;

    let witness_dir = env::var("BLOCK_INTEGRATED_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_m64".to_string());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] BLOCK_INTEGRATED_WITNESS_DIR={} not a directory.",
            witness_dir
        );
        return;
    }

    println!("\n=== [Block Integrated prove+verify] dir={} ===", witness_dir);

    let total_start = Instant::now();
    let res = run_block_integrated_paths::<Bn254>(dir);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!("\n=== [Block Integrated] PASS | total_wall_ms={} ===", total_ms),
        Err(e) => panic!("[Block Integrated] failed after {}ms: {}", total_ms, e),
    }
}
