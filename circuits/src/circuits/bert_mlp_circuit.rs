use std::{path::Path, str::FromStr, time::Instant};
use anyhow::Result;
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ec::pairing::Pairing;
use ark_ff::{BigInteger, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::{rc::Rc, fs, UniformRand, rand::{SeedableRng, rngs::StdRng}};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
use ark_r1cs_std::R1CSVar;
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{BenchmarkResult, prove_and_verify_factored_multi_link},
        mlp_circuit::{NystromMLPCircuit, load_nystrom_circuit},
    },
    util::{
        lookup_mlp::{run_lasso_for_gelu, run_lasso_for_rmsnorm, run_mlp_remainders},
        utils::cm1_from_vk,
    },
};

type G1 = G1Projective;

/// BERT MLP circuit is structurally identical to `NystromMLPCircuit`.
pub type BertMLPCircuit<F> = NystromMLPCircuit<F>;

/// Load a BERT MLP witness NPZ.
///
/// Key naming matches the existing Nystromformer MLP loader exactly
/// (prefixes: rms1, lin1, gelu, lin2, rms2).
pub fn load_bert_mlp_circuit<E: Pairing>(path: &Path) -> Result<BertMLPCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    load_nystrom_circuit::<E>(path)
}

pub fn run_bert_mlp_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [BERT MLP] Processing: {:?} ===", file_path);

        // ── Dry-run ──────────────────────────────────────────────────────────
        let num_commit_1 = {
            let temp = load_bert_mlp_circuit::<E>(file_path)?;
            let vars_ptr = Rc::clone(&temp.committed_vars);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            vars_ptr.borrow().len() as u32
        };
        println!("Commit count d1={}", num_commit_1);

        let sample_circuit = load_bert_mlp_circuit::<E>(file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup ===");
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            num_commit_1,
            0, // no d2 group for MLP
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
            "[Debug] LegoGroth16 committed_group1 total={}",
            committed_group1.len()
        );
        let print_count = std::cmp::min(10, committed_group1.len());
        if !committed_group1.is_empty() {
            println!(
                "  Head(10): {:?}",
                committed_group1[..print_count]
                    .iter()
                    .map(|f: &Fr| f.into_bigint())
                    .collect::<Vec<_>>()
            );
        }

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng); // blind for d2 (unused, num_commit_2=0)
        let num_pub = cs_witness.num_instance_variables() - 1;
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        println!("=== [1] LegoGroth16 GenProof ===");
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            sample_circuit.clone(),
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
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &[]);
        let g16_verify_time = g16_verify_start.elapsed();

        println!("  - BERT MLP SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        // ── [2] Lasso ────────────────────────────────────────────────────────
        println!("=== [2] BERT MLP Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();

        let (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1) =
            run_lasso_for_rmsnorm(&sample_circuit.norm1, b"bert_mlp_rms1", &mut rng)?;

        let (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2) = run_lasso_for_gelu(
            &sample_circuit.gelu1,
            &sample_circuit
                .ff1
                .q_out_sim
                .iter()
                .map(|batch| {
                    batch
                        .iter()
                        .map(|seq| {
                            seq.iter()
                                .map(|val: &Fr| Fr::from(val.into_bigint().as_ref()[0] as i64))
                                .collect::<Vec<Fr>>()
                        })
                        .collect::<Vec<Vec<Fr>>>()
                })
                .collect::<Vec<Vec<Vec<Fr>>>>(),
            &sample_circuit.ff1.r_out_lin,
            sample_circuit.k_bits,
            b"bert_mlp_gelu",
            &mut rng,
        )?;

        let (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3) =
            run_lasso_for_rmsnorm(&sample_circuit.norm2, b"bert_mlp_rms2", &mut rng)?;

        let mut lasso_results = vec![
            (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1),
            (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2),
            (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3),
        ];

        let res_remainders = run_mlp_remainders::<Fr, G1>(&sample_circuit, &mut rng)?;
        lasso_results.extend(res_remainders);

        // Re-extract u_values from already-synthesized circuit (same as committed_group1)
        let u_values: Vec<Fr> = sample_circuit
            .committed_vars
            .borrow()
            .iter()
            .map(|v| v.value().unwrap_or(Fr::zero()))
            .collect();

        let lasso_time = lasso_start.elapsed();

        // ── [3] CP-Link ──────────────────────────────────────────────────────
        let cp_link_start = Instant::now();
        prove_and_verify_factored_multi_link::<E>(
            &params.vk,
            num_pub,
            v1,
            cm1,
            lasso_results,
            u_values,
            &mut rng,
        )?;
        let cp_link_time = cp_link_start.elapsed();

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "bert_mlp".to_string(),
            constraints: num_constraints,
            setup_ms: setup_time.as_millis(),
            proof_size_bytes: proof_size,
            groth16_prove_ms: g16_prove_time.as_millis(),
            lasso_ms: lasso_time.as_millis(),
            cp_link_ms: cp_link_time.as_millis(),
            groth16_verify_ms: g16_verify_time.as_millis(),
            end_to_end_ms: e2e_time.as_millis(),
        }
        .print_json();
    }
    Ok(())
}

#[test]
fn bert_mlp_test() {
    let outputs = Path::new("outputs_bert_small");
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
                .map_or(false, |n| n.ends_with(".norm.witness.npz"))
        {
            targets.push(path);
        }
    }
    targets.sort();
    println!(
        "Running BERT MLP circuit test on {} witness file(s)",
        targets.len()
    );
    let res = run_bert_mlp_paths::<Bn254>(&targets);
    if let Err(e) = res {
        panic!("BERT MLP proof failed: {}", e);
    }
}
