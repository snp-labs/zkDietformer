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
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        freivalds_matmul::nystrom_mlp_circuit::{NystromMLPCircuit, load_nystrom_circuit},
        proof_size,
    },
    util::{
        mlp_lookup::{run_lasso_for_gelu, run_lasso_for_rmsnorm, run_mlp_remainders},
        utils::cm1_from_vk,
    },
};

type G1 = G1Projective;

/// BERT MLP circuit is structurally identical to `NystromMLPCircuit`.
pub type BERTMLPCircuit<F> = NystromMLPCircuit<F>;

/// Load a BERT MLP witness NPZ.
///
/// Key naming matches the existing Nystromformer MLP loader exactly
/// (prefixes: rms1, lin1, gelu, lin2, rms2).
pub fn load_bert_mlp_circuit<E: Pairing>(path: &Path) -> Result<BERTMLPCircuit<E::ScalarField>>
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

        // ── Groth16 commit (Pedersen cm1 from VK) ───────────────────────────
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

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

        // ── [2] Lasso (commit + prove + verify bundled in helpers) ───────────
        // Reset Lasso aggregator before individual Lasso proofs
        proof_size::reset_lasso_aggregator();
        println!("=== [2] BERT MLP Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();

        let ((lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1), m1) =
            run_lasso_for_rmsnorm(&sample_circuit.norm1, b"bert_mlp_rms1", &mut rng)?;
        lasso_agg.add(&m1);

        let ((lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2), m2) = run_lasso_for_gelu(
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
        lasso_agg.add(&m2);

        let ((lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3), m3) =
            run_lasso_for_rmsnorm(&sample_circuit.norm2, b"bert_mlp_rms2", &mut rng)?;
        lasso_agg.add(&m3);

        let mut lasso_results = vec![
            (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1),
            (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2),
            (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3),
        ];

        let (res_remainders, rem_agg) = run_mlp_remainders::<Fr, G1>(&sample_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        lasso_results.extend(res_remainders);

        let u_values: Vec<Fr> = sample_circuit
            .committed_vars
            .borrow()
            .iter()
            .map(|v| v.value().unwrap_or(Fr::zero()))
            .collect();

        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        // ── [3] CP-Link (setup/keygen, prove, verify timed separately) ───────
        let u_padded = pad_lasso_u_values::<E>(&lasso_results, &u_values);

        let cp_setup_start = Instant::now();
        let (pp, crs) = factored_link_setup::<E, _>(&params.vk, num_pub, u_values.len(), &lasso_results, &mut rng);
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

        // ── Groth16 proof size report ──
        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // ── CP-Link proof size report ──
        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        cplink_report.print_log();

        // ── Grand total proof size report ──
        proof_size::print_full_report(&g16_report, &cplink_report);

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "bert_mlp".to_string(),
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


#[test]
fn bert_mlp_hf_single_layer_test() {
    use std::env;
    use std::path::Path;
    use ark_std::fs;
    use crate::circuits::freivalds_matmul::nystrom_mlp_circuit::run_paths_layernorm;
    use crate::lasso::subtables::layernorm::set_layernorm_table_dir;
    use crate::lasso::subtables::gelu::set_gelu_table_dir;

    // ── 환경 변수 ───────────────────────────────────────────────────────────
    let witness_dir = env::var("BERT_MLP_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_bert".to_string());
    let layer_idx: Option<usize> = env::var("BERT_MLP_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] BERT_MLP_WITNESS_DIR={} not a directory. \
             Set the env var to a valid HF BERT outputs_bert/ dir to run.",
            witness_dir
        );
        return;
    }

    // ── 표 경로 결정: env var 미설정 시 witness_dir 안의 tables.npz ─
    if env::var("LAYERNORM_TABLE_PATH").is_err() {
        set_layernorm_table_dir(dir);
        println!(
            "[info] LAYERNORM_TABLE_PATH not set; using {}/layernorm_tables.npz",
            witness_dir
        );
    }
    if env::var("GELU_TABLE_PATH").is_err() {
        set_gelu_table_dir(dir);
        println!(
            "[info] GELU_TABLE_PATH not set; using {}/gelu_tables.npz",
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
                .map_or(false, |n| n.ends_with(".norm.witness.npz"))
        {
            all_targets.push(path);
        }
    }
    all_targets.sort();

    if all_targets.is_empty() {
        eprintln!(
            "[skip] no '*.norm.witness.npz' files in {}. \
             Run HF BERT normalization pipeline first.",
            witness_dir
        );
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!(
                "BERT_MLP_LAYER_IDX={} out of range (have {} layers).",
                idx,
                all_targets.len()
            );
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [BERT MLP prove+verify] dir={} | files={} | mode={} ===",
        witness_dir,
        targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_paths_layernorm::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!(
            "\n=== [BERT MLP] ALL {} files PASS | total_wall_ms={} ===",
            targets.len(), total_ms,
        ),
        Err(e) => panic!(
            "[BERT MLP] proof generate/verify failed after {}ms: {}",
            total_ms, e,
        ),
    }
}
