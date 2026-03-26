use std::{fs::File, ops::{AddAssign, Mul}, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing, AffineRepr};
use ndarray::{Ix0, Ix1, Ix2, Ix3};
use ndarray_npy::NpzReader;
use ark_std::{rc::Rc, cell::RefCell, fs, UniformRand, marker::PhantomData, vec::Vec, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{fp::FpVar, FieldVar}, R1CSVar};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::common::{
        BenchmarkResult,
        factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
        pad_lasso_u_values, rss_mb,
    },
    config::{GELU_C, GELU_M, KBITS_C, KBITS_M, RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M},
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu},
        layernorm::constraints::{NormBundle, enforce_rmsnorm},
        linear::constraints::{LinearBundle, enforce_linear},
    },
    lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
        subtables::{
            gelu::GeLUSubtableStrategy, layernorm::RMSNormSubtableStrategy,
            range_check::RangeCheckSubtableStrategy,
        },
        utils::random::RandomTape,
    },
    util::{
        load::{to_field_1d_i64, to_field_2d_i32, to_field_3d_i32, to_field_3d_i64, to_field_3d_u64},
        lookup_mlp::{run_lasso_for_gelu, run_lasso_for_rmsnorm, run_mlp_remainders},
        utils::{cm1_from_vk, hash_cm_to_field_vec},
    },
};

type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromMLPCircuit<F: PrimeField> {
    pub norm1: NormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,
    pub ff2: LinearBundle<F>,
    pub norm2: NormBundle<F>,
    pub frac_bits: usize,
    pub k_bits: usize,

    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromMLPCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();

        let mut rms1_pool = Vec::new();
        let mut gelu_pool = Vec::new();
        let mut rms2_pool = Vec::new();
        let mut r_pool = Vec::new();

        let dim: usize = self.norm1.q_ri[0][0].len();
        let q_norm1_vars = enforce_rmsnorm(cs.clone(), &self.norm1, dim, self.frac_bits, self.k_bits, &mut r_pool, &mut rms1_pool)?;
        let (q_ff1_vars, r_ff1_vars) = enforce_linear(cs.clone(), &q_norm1_vars, &self.ff1, self.k_bits, &mut r_pool)?;
        let q_gelu_vars = enforce_gelu(cs.clone(), &q_ff1_vars, &r_ff1_vars, &self.gelu1, self.k_bits, &mut r_pool, &mut gelu_pool)?;
        let (q_ff2_vars, _) = enforce_linear(cs.clone(), &q_gelu_vars, &self.ff2, self.k_bits, &mut r_pool)?;
        let q_norm2_vars = enforce_rmsnorm(cs.clone(), &self.norm2, dim, self.frac_bits, self.k_bits, &mut r_pool, &mut rms2_pool)?;

        {
            let mut final_vars = self.committed_vars.borrow_mut();
            final_vars.extend(rms1_pool);  
            final_vars.extend(gelu_pool); 
            final_vars.extend(rms2_pool);  
            final_vars.extend(r_pool);
        }

        Ok(())
    }
}

pub fn load_nystrom_circuit<E: Pairing>(path: &Path) -> Result<NystromMLPCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    
    let load_norm = |npz: &mut NpzReader<File>, prefix: &str| -> Result<NormBundle<E::ScalarField>> {
        Ok(NormBundle {
            q_ri: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_ri_{}.npy", prefix))?.into_dimensionality()?),
            z_ri: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_ri_{}.npy", prefix))?.into_scalar()),
            // sum_sq: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("sum_sq_{}.npy", prefix))?.into_dimensionality()?),
            m_ms: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_ms_{}.npy", prefix))?.into_scalar()),
            m_prime_ms: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_ms_{}.npy", prefix))?.into_scalar()),
            shifted_val: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("shifted_val_{}.npy", prefix))?.into_dimensionality()?),
            r_shift: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_shift_{}.npy", prefix))?.into_dimensionality()?),
            q_ms_sim: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_ms_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_ms: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_ms_{}.npy", prefix))?.into_dimensionality()?),
            inv_rms: to_field_3d_u64(&npz.by_name::<_, Ix3>(&format!("inv_rms_total_{}.npy", prefix))?.into_dimensionality()?),
            q_gamma: to_field_1d_i64(&npz.by_name::<_, Ix1>(&format!("Q_gamma_{}.npy", prefix))?.into_dimensionality()?),
            z_gamma: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_gamma_{}.npy", prefix))?.into_scalar()),
            q_out: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_rms_output_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_rms_output_{}.npy", prefix))?.into_dimensionality()?),
            m_rms_idx: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?.into_scalar()),
            m_norm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_norm_{}.npy", prefix))?.into_scalar()),
            m_prime_norm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_norm_{}.npy", prefix))?.into_scalar()),
        })
    };
    
    let norm1 = load_norm(&mut npz, "rms1")?;

    let load_linear = |npz: &mut NpzReader<File>, prefix: &str| -> Result<LinearBundle<E::ScalarField>> {
        Ok(LinearBundle {
            z_in: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_in_{}.npy", prefix))?.into_scalar()),
            
            q_w_diff: to_field_2d_i32(&npz.by_name::<_, Ix2>(&format!("Q_w_diff_{}.npy", prefix))?.into_dimensionality()?),
            q_b_diff: to_field_1d_i64(&npz.by_name::<_, Ix1>(&format!("Q_b_diff_{}.npy", prefix))?.into_dimensionality()?),
            
            m_lin: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?.into_scalar()),
            m_prime_lin: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?.into_scalar()),
            
            q_out_sim: to_field_3d_i32(&npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out_lin: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?.into_dimensionality()?),
        })
    };

    let ff1 = load_linear(&mut npz, "lin1")?;
        
    let load_gelu = |npz: &mut NpzReader<File>, prefix: &str| -> Result<GeLUBundle<E::ScalarField>> {
        Ok(GeLUBundle {
            z_in: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_in_{}.npy", prefix))?.into_scalar()),
            sigmoid_table_val: to_field_3d_u64(&npz.by_name::<_, Ix3>(&format!("sigmoid_part_{}.npy", prefix))?.into_dimensionality()?),
            // acc_gelu: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("acc_{}.npy", prefix))?.into_dimensionality()?),
            m_gelu: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?.into_scalar()),
            m_prime_gelu: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?.into_scalar()),
            q_out_sim: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out_gelu: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?.into_dimensionality()?),
            m_gelu_idx: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_gelu_idx.npy"))?.into_scalar()),
        })
    };
    
    let gelu1 = load_gelu(&mut npz, "gelu")?;

    let ff2 = load_linear(&mut npz, "lin2")?; 
    let norm2 = load_norm(&mut npz, "rms2")?; 

    Ok(NystromMLPCircuit {
        norm1,
        ff1,
        gelu1,
        ff2,
        norm2,
        frac_bits,
        k_bits,
        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })

    
}

pub fn run_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);

        // ── Dry-run: count committed witnesses ───────────────────────────────
        let num_commit_1 = {
            let temp_circuit = load_nystrom_circuit::<E>(&file_path)?;
            let vars_ptr = Rc::clone(&temp_circuit.committed_vars);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;
            vars_ptr.borrow().len() as u32
        };
        println!("최종 측정된 커밋 개수: {:?}", num_commit_1);

        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (Num Commit: {:?}) ===", num_commit_1);
        let mem_pre_setup = rss_mb();
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            num_commit_1,
            0,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();
        let mem_post_setup = rss_mb();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction + constraint count ────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> = vars.iter()
                .map(|v| v.value().unwrap_or(Fr::zero()))
                .collect();
            if values.len() != num_commit_1 as usize {
                return Err(anyhow::anyhow!(
                    "Commitment size mismatch: expected {}, found {}",
                    num_commit_1,
                    values.len()
                ));
            }
            values
        };

        println!("\n[Debug] LegoGroth16 committed_group1 (Total: {})", committed_group1.len());
        let print_count = std::cmp::min(10, committed_group1.len());
        if committed_group1.len() > 0 {
            println!("  Head (10): {:?}", committed_group1[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  Tail (10): {:?}", committed_group1[committed_group1.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit (Pedersen cm1 from VK) ───────────────────────────
        let mem0 = rss_mb(); // baseline before commit
        let groth16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let groth16_commit_ms = groth16_commit_start.elapsed().as_millis();
        let mem1 = rss_mb(); // after commit

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        println!("=== [1] LegoGroth16 GenProof ===");
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            sample_circuit.clone(),
            v1,
            v2,
            &params,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;
        let groth16_prove_time = groth16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // ── [1] Groth16 Verify ───────────────────────────────────────────────
        let public_inputs = vec![];
        let groth16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let groth16_verify_time = groth16_verify_start.elapsed();
        let mem2 = rss_mb(); // after Groth16 prove+verify

        println!("  - LegoGroth16 SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::Error::msg(format!("{:?}", e)));
        }

        // ── [2] Lasso (commit + prove + verify bundled in helpers) ───────────
        println!("=== [2] Lasso Lookup Proofs: MLP Sections ===");
        let lasso_start = Instant::now();
        let (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1) = run_lasso_for_rmsnorm(&sample_circuit.norm1, b"lasso_rms1", &mut rng)?;
        let (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2) = run_lasso_for_gelu(
            &sample_circuit.gelu1,
            &sample_circuit.ff1.q_out_sim.iter().map(|batch| {
                batch.iter().map(|seq| {
                    seq.iter().map(|val| {
                        Fr::from((*val).into_bigint().as_ref()[0] as i64)
                    }).collect::<Vec<Fr>>()
                }).collect::<Vec<Vec<Fr>>>()
            }).collect::<Vec<Vec<Vec<Fr>>>>(),
            &sample_circuit.ff1.r_out_lin,
            sample_circuit.k_bits,
            b"lasso_gelu1",
            &mut rng
        )?;
        let (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3) = run_lasso_for_rmsnorm(&sample_circuit.norm2, b"lasso_rms2", &mut rng)?;
        let res_remainders = run_mlp_remainders::<Fr, G1>(&sample_circuit, &mut rng)?;
        let u_values: Vec<Fr> = sample_circuit.committed_vars.borrow().iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
        let lasso_time = lasso_start.elapsed();
        let mem3 = rss_mb(); // after all Lasso phases

        let mut lasso_results = vec![
            (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1),
            (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2),
            (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3),
        ];
        lasso_results.extend(res_remainders);

        // ── [3] CP-Link (setup/keygen, prove, verify timed separately) ───────
        let u_padded = pad_lasso_u_values::<E>(&lasso_results, &u_values);

        let cp_setup_start = Instant::now();
        let (pp, crs) = factored_link_setup::<E, _>(&params.vk, num_pub, u_values.len(), &lasso_results, &mut rng);
        let (ek, vk_link) = factored_link_keygen::<E, _>(&pp, crs, &mut rng);
        let cp_setup_ms = cp_setup_start.elapsed().as_millis();

        let cp_prove_start = Instant::now();
        let (proof_link, instance) = factored_link_prove::<E, _>(&pp, &ek, v1, cm1, &lasso_results, u_padded, &mut rng);
        let cp_prove_ms = cp_prove_start.elapsed().as_millis();
        let mem4 = rss_mb(); // after CP-Link prove

        let cp_verify_start = Instant::now();
        if !factored_link_verify::<E>(&pp, &vk_link, &instance, &proof_link) {
            return Err(anyhow::anyhow!("Factored link verification failed"));
        }
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        let cp_verify_ms = cp_verify_start.elapsed().as_millis();
        let mem5 = rss_mb(); // after CP-Link verify

        let e2e_time = e2e_start.elapsed();

        // ── Benchmark output ─────────────────────────────────────────────────
        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit:           "mlp".to_string(),
            constraints:       num_constraints,
            setup_ms:          setup_time.as_millis(),
            groth16_commit_ms,
            groth16_prove_ms:  groth16_prove_time.as_millis(),
            groth16_verify_ms: groth16_verify_time.as_millis(),
            lasso_ms:          lasso_time.as_millis(),
            cp_link_setup_ms:  cp_setup_ms,
            cp_link_prove_ms:  cp_prove_ms,
            cp_link_verify_ms: cp_verify_ms,
            end_to_end_ms:     e2e_time.as_millis(),
            proof_size_bytes:  proof_size,
            // mem_pre_setup → mem_post_setup: LegoGroth16 setup
            // mem0 (=mem_post_setup) → mem1: Groth16 commit (cm1_from_vk)
            // mem1 → mem2: Groth16 prove+verify
            // mem2 → mem3: all Lasso phases (commit+prove+verify bundled)
            // mem3 → mem4: CP-Link setup+prove
            // mem4 → mem5: CP-Link verify
            memory_setup_mb:   mem_post_setup - mem_pre_setup,
            memory_commit_mb:  mem1 - mem0,
            memory_prove_mb:   mem4 - mem1,
            memory_verify_mb:  mem5 - mem4,
        }
        .print_json();
    }
    Ok(())
}


#[test]
fn mlp_test() {
    let outputs = Path::new("outputs_server");
    if !outputs.is_dir() {
        return;
    }
    let mut targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(outputs).expect("reading outputs dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("norm_ff_combined.witness.npz"))
        {
            targets.push(path);
        }
    }
    targets.sort();
    println!("Running MLP circuit test on {} norm_ff_combined witness file(s)", targets.len());
    let res = run_paths::<Bn254>(&targets);
    if let Err(e) = res {
        panic!("proof generate failed: {}", e);
    }
}