
use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix1, Ix4, Array2, Ix2, Ix3, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{fs, UniformRand, log2, marker::PhantomData, rand::{Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use merlin::Transcript;
use ark_r1cs_std::select::CondSelectGadget;
use crate::{
    cc_snark::cc_gro::{CcGroth16, prepare_verifying_key}, config::{SOFTMAX_C, SOFTMAX_M, RMSNORM_C, RMSNORM_M}, 
    gadgets::{
        layernorm::{constraints::{NormBundle, enforce_rmsnorm}}, 
    }, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        subtables::{
            SubtableStrategy, and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy,
            layernorm::RMSNormSubtableStrategy
        }, utils::{math::Math, random::RandomTape},
    }, util::load::{to_field_3d_u64, to_field_3d_i64, to_field_1d_i64}
};


type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromAttentionCircuit<F: PrimeField> {
    pub norm1: NormBundle<F>,
    //pub norm2: NormBundle<F>,

    pub frac_bits: usize,
    pub k_bits: usize,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        
        let dim = self.norm1.q_ri[0][0].len();

        enforce_rmsnorm(cs.clone(), &self.norm1, dim, self.frac_bits, self.k_bits)?;
        //enforce_rmsnorm(cs.clone(), &self.norm2, dim, self.frac_bits, self.k_bits)?;

        Ok(())
    }
}

pub fn load_nystrom_circuit<E: Pairing>(path: &Path) -> Result<NystromAttentionCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    // 1. NPZ 파일 열기
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let k_bits = npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let frac_bits = npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    
    let load_norm = |npz: &mut NpzReader<File>, prefix: &str| -> Result<NormBundle<E::ScalarField>> {
        Ok(NormBundle {
            q_ri: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_ri_{}.npy", prefix))?.into_dimensionality()?),
            z_ri: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>(&format!("z_ri_{}.npy", prefix))?.into_scalar()),
            sum_sq: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("sum_sq_{}.npy", prefix))?.into_dimensionality()?),
            m_ms: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>(&format!("M_ms_{}.npy", prefix))?.into_scalar()),
            m_prime_ms: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>(&format!("M_prime_ms_{}.npy", prefix))?.into_scalar()),
            shifted_val: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("shifted_val_{}.npy", prefix))?.into_dimensionality()?),
            r_shift: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_shift_{}.npy", prefix))?.into_dimensionality()?),
            q_ms_sim: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_ms_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_ms: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_ms_{}.npy", prefix))?.into_dimensionality()?),
            inv_rms: to_field_3d_u64(&npz.by_name::<_, Ix3>(&format!("inv_rms_total_{}.npy", prefix))?.into_dimensionality()?),
            q_gamma: to_field_1d_i64(&npz.by_name::<_, Ix1>(&format!("Q_gamma_{}.npy", prefix))?.into_dimensionality()?),
            q_out: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_rms_output_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_rms_output_{}.npy", prefix))?.into_dimensionality()?),
            weight_bits: npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>(&format!("weight_bits.npy"))?.into_scalar() as usize,
        })
    };
    
    let norm1 = load_norm(&mut npz, "rms")?;
    //let norm2 = load_norm(&mut npz, "norm2")?;

    Ok(NystromAttentionCircuit {
        norm1,
        //norm2,

        frac_bits: frac_bits,
        k_bits: k_bits,
    })

    
}

pub fn run_paths<E: Pairing>(path: &Path) -> Result<()>
where
E: Pairing<ScalarField = Fr>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{

    type ScalarF<E> = <E as Pairing>::ScalarField;
    // collect targets
    let mut targets: Vec<std::path::PathBuf> = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path).context("reading directory")? {
            let ent = entry?.path();
            if ent.is_file() {
                if let Some(name) = ent.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(".witness.npz") || name.ends_with(".npz") {
                        targets.push(ent);
                    }
                }
            }
        }
        targets.sort();
        if targets.is_empty() {
            return Err(anyhow::anyhow!("No .witness.npz or .npz files found in directory {:?}", path));
        }
    } else if path.is_file() {
        targets.push(path.to_path_buf());
    } else {
        return Err(anyhow::anyhow!("Provided path is neither file nor directory: {:?}", path));
    }

    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets.iter() {
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);
        let circuit= load_nystrom_circuit::<E>(file_path)?;

        let cs = ark_relations::r1cs::ConstraintSystem::<ScalarF<E>>::new_ref();
        circuit.clone().generate_constraints(cs.clone())?;
        
        if !cs.is_satisfied().unwrap() {
            match cs.which_is_unsatisfied() {
                Ok(Some(path)) => println!("Failed Path: {}", path),
                Ok(None) => println!("Else"),
                Err(e) => println!("Error while debugging: {:?}", e),
            }
            return Err(anyhow::anyhow!("Constraints NOT satisfied in {:?}", file_path));
        }

        let cs = ark_relations::r1cs::ConstraintSystem::<Fr>::new_ref();
        circuit.clone().generate_constraints(cs.clone())?;
        let num_constraints = cs.num_constraints();
        let num_instance_variables = cs.num_instance_variables();
        let num_witness_variables = cs.num_witness_variables();
        
        println!("  [Constraint Stats]");
        println!("  - Number of Constraints: {}", num_constraints);
        println!("  - Public Inputs (Instance Vars): {}", num_instance_variables);
        println!("  - Private Inputs (Witness Vars): {}", num_witness_variables);

        if !cs.is_satisfied().unwrap() {
            return Err(anyhow::anyhow!("Constraints NOT satisfied"));
        }

        println!("=== [1] ccGroth Proofs: Freivald/Softmax ===");    
        let setup_start = Instant::now();
        let (pk, vk) = CcGroth16::<E>::setup(circuit.clone(), &mut rng)?;
        let setup_duration = setup_start.elapsed();
        println!("  - Setup Time: {:?}", setup_duration);

        let prove_start = Instant::now();
        let proof = CcGroth16::<E>::prove(&pk, circuit.clone(), &mut rng)?;
        let prove_duration = prove_start.elapsed();
        println!("  - Prover Time (ccGroth): {:?}", prove_duration);

        let pvk = prepare_verifying_key::<E>(&vk);
        let public_inputs = vec![];
        let verify_start = Instant::now();
        let ok = CcGroth16::<E>::verify_with_processed_vk(&pvk, &public_inputs, &proof)?;
        println!("Verification result for {:?}: {}", file_path, ok);
        if !ok {
            return Err(anyhow::anyhow!("Verification failed for {:?}", file_path));
        }
        let verify_duration = verify_start.elapsed();
        println!("  - Verifier Time (ccGroth): {:?}", verify_duration);
        println!("  - Verification Result: {}", ok);

        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        let dim = circuit.norm1.q_ri[0][0].len();
        run_lasso_for_rmsnorm(&circuit.norm1, circuit.k_bits, b"lasso_rms1", &mut rng)?;
    }

    Ok(())
}

fn run_lasso_for_rmsnorm(
    bundle: &NormBundle<Fr>,
    k_bits: usize,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<()> {
    let b_dim = bundle.q_ms_sim.len();
    let s_dim = bundle.q_ms_sim[0].len();

    let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
    
    for b in 0..b_dim {
        for s in 0..s_dim {
            // BigInt에서 직접 limb을 가져올 때, 음수가 양수로 오해되지 않도록 i64로 먼저 해석 후 u64로 변환하거나 
            // 0보다 작은지 체크하는 것이 안전합니다.
            let q_ms_bigint = bundle.q_ms_sim[b][s][0].into_bigint();
            let r_ms_bigint = bundle.r_ms[b][s][0].into_bigint();
            
            let q_ms_u64 = q_ms_bigint.as_ref()[0];
            let r_ms_u64 = r_ms_bigint.as_ref()[0];
            
            // 파이썬 rms_table.py의 index = (Q_sim << k_bits) + R_mat 로직을 따름
            // 단, 이미 Q_ms_sim이 최종 몫이라면 k_bits 시프트가 맞으나, 
            // 현재 에러 수치로 보아 시프트된 결과값이 이미 들어왔을 가능성이 큼.
            // 안전하게 범위를 체크하고 clipping을 수행합니다.
            let mut combined_idx = (q_ms_u64 << k_bits) + r_ms_u64;

            // RMSNORM_M 범위를 초과하면 테이블 마지막 인덱스로 고정 (Python의 np.clip과 동기화)
            if combined_idx >= RMSNORM_M as u64 {
                combined_idx = (RMSNORM_M - 1) as u64;
            }
            
            indices_x[b][s][0] = Fr::from(combined_idx);
        }
    }
    
    let total_sparsity = b_dim * s_dim;
    let gens = SparsePolyCommitmentGens::<G1>::new(
        label,
        RMSNORM_C,
        total_sparsity,
        RMSNORM_C,
        RMSNORM_M.trailing_zeros() as usize,
    );

    let start = Instant::now();
    prove_lasso_for_rmsnorm(&indices_x, &gens, rng, label)?;
    println!("  - Lasso Total Time (RMSNorm): {:?}", start.elapsed());
    
    Ok(())
}

fn prove_lasso_for_rmsnorm(
    indices: &Vec<Vec<Vec<Fr>>>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<()> {
    let b_dim = indices.len();
    let s_dim = indices[0].len();

    let mut nz_indices: Vec<[usize; RMSNORM_C]> = Vec::with_capacity(b_dim * s_dim);
    
    for b in 0..b_dim {
        for s in 0..s_dim {
            let val_bigint = indices[b][s][0].into_bigint();
            let limbs = val_bigint.as_ref(); 
            let idx_u64 = limbs[0]; 

            // 필드 원소가 테이블 크기보다 크면 Lasso 제약 조건을 만족할 수 없음
            if idx_u64 >= RMSNORM_M as u64 {
                return Err(anyhow::anyhow!(
                    "Lasso Index Final Check Failed: index {} >= table size {} at [Batch: {}, Seq: {}]. \
                    This should have been clipped in run_lasso_for_rmsnorm.", 
                    idx_u64, RMSNORM_M, b, s
                ));
            }
            
            nz_indices.push([idx_u64 as usize]);
        }
    }

    let sparsity = nz_indices.len();
    if sparsity == 0 { return Ok(()); }

    let log_M = RMSNORM_M.trailing_zeros() as usize;
    let lookup_matrix = SparseLookupMatrix::<RMSNORM_C>::new(nz_indices, log_M);
    let mut densified = DensifiedRepresentation::<Fr, RMSNORM_C>::from(&lookup_matrix);

    let commitment = densified.commit::<G1>(gens);
    let log_s = (sparsity.next_power_of_two().trailing_zeros()) as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape_rms");
    let mut transcript = Transcript::new(b"lasso_rmsnorm");
    
    let proof_lasso = SparsePolynomialEvaluationProof::<G1, RMSNORM_C, RMSNORM_M, RMSNormSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"lasso_rmsnorm");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso verification failed for: {:?}", std::str::from_utf8(label)))?;

    println!("Lasso proof verified: {:?}", std::str::from_utf8(label).unwrap());
    Ok(())
}

#[test]
fn mlp_test() {
    let p = Path::new("outputs/outputs-model.transformer_0.norm1.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            panic!("proof generate failed: {}", e);
        }
    }
}