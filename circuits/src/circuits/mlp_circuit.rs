
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
    cc_snark::cc_gro::{CcGroth16, prepare_verifying_key}, config::{RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M, GELU_M, GELU_C}, 
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu}, layernorm::constraints::{NormBundle, enforce_rmsnorm}, linear::constraints::{LinearBundle, enforce_linear}
    }, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        subtables::{
            SubtableStrategy, and::AndSubtableStrategy, 
            layernorm::RMSNormSubtableStrategy, 
            lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, 
            softmax::SoftmaxSubtableStrategy, gelu::GeLUSubtableStrategy
        }, utils::{math::Math, random::RandomTape},
    }, util::load::{to_field_1d_i64, to_field_2d_i32, to_field_3d_i32, to_field_3d_i64, to_field_3d_u64}
};


type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromAttentionCircuit<F: PrimeField> {
    pub norm1: NormBundle<F>,
    //pub norm2: NormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,

    pub frac_bits: usize,
    pub k_bits: usize,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        
        let dim: usize = self.norm1.q_ri[0][0].len();

        let q_norm1_vars = enforce_rmsnorm(cs.clone(), &self.norm1, dim, self.frac_bits, self.k_bits)?;
        /*let (q_ff1_vars, r_ff1_vars) = enforce_linear(cs.clone(),&q_norm1_vars, &self.ff1, self.k_bits)?;
        let q_gelu_vars = enforce_gelu(
            cs.clone(),
            &q_ff1_vars,  
            &r_ff1_vars,  
            &self.gelu1,
            self.k_bits,
            self.frac_bits
        )?;*/

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

    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    
    let load_norm = |npz: &mut NpzReader<File>, prefix: &str| -> Result<NormBundle<E::ScalarField>> {
        Ok(NormBundle {
            q_ri: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_ri_{}.npy", prefix))?.into_dimensionality()?),
            z_ri: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_ri_{}.npy", prefix))?.into_scalar()),
            sum_sq: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("sum_sq_{}.npy", prefix))?.into_dimensionality()?),
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
    
    let norm1 = load_norm(&mut npz, "rms")?;
    //let norm2 = load_norm(&mut npz, "norm2")?;

    // mlp_circuit.rs 또는 테스트 코드 내부에 위치
    let load_linear = |npz: &mut NpzReader<File>, prefix: &str| -> Result<LinearBundle<E::ScalarField>> {
        Ok(LinearBundle {
            // z_in: 입력(Norm 출력)의 제로포인트
            z_in: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_in_{}.npy", prefix))?.into_scalar()),
            
            // 가중치 및 바이어스 (Q_w - z_w, Q_b - z_b)
            q_w_diff: to_field_2d_i32(&npz.by_name::<_, Ix2>(&format!("Q_w_diff_{}.npy", prefix))?.into_dimensionality()?),
            q_b_diff: to_field_1d_i64(&npz.by_name::<_, Ix1>(&format!("Q_b_diff_{}.npy", prefix))?.into_dimensionality()?),
            
            // 재스케일링 상수
            m_lin: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?.into_scalar()),
            m_prime_lin: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?.into_scalar()),
            
            // 결과 Witness (enforce_equal 검증용)
            q_out_sim: to_field_3d_i32(&npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out_lin: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?.into_dimensionality()?),
        })
    };

    let ff1 = load_linear(&mut npz, "lin")?;
        
    let load_gelu = |npz: &mut NpzReader<File>, prefix: &str| -> Result<GeLUBundle<E::ScalarField>> {
        Ok(GeLUBundle {
            z_in: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_in_{}.npy", prefix))?.into_scalar()),
            sigmoid_table_val: to_field_3d_u64(&npz.by_name::<_, Ix3>(&format!("sigmoid_part_{}.npy", prefix))?.into_dimensionality()?),
            acc_gelu: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("acc_{}.npy", prefix))?.into_dimensionality()?),
            m_gelu: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?.into_scalar()),
            m_prime_gelu: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?.into_scalar()),
            q_out_sim: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out_gelu: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?.into_dimensionality()?),
            m_gelu_idx: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_gelu_idx.npy"))?.into_scalar()),
        })
    };
    
    let gelu1 = load_gelu(&mut npz, "gelu")?;


    Ok(NystromAttentionCircuit {
        norm1,
        //norm2,
        ff1,
        gelu1,

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
        run_lasso_for_gelu(
            &circuit.gelu1, 
            &circuit.ff1.q_out_sim.iter().map(|batch| {
                batch.iter().map(|seq| {
                    seq.iter().map(|val| {
                        Fr::from((*val).into_bigint().as_ref()[0] as i64)
                    }).collect::<Vec<Fr>>()
                }).collect::<Vec<Vec<Fr>>>()
            }).collect::<Vec<Vec<Vec<Fr>>>>(),
            &circuit.ff1.r_out_lin, 
            circuit.k_bits, 
            b"lasso_gelu1", 
            &mut rng
        )?;
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
    let dim = bundle.q_out[0][0].len();

    let mut indices_x = vec![vec![vec![Fr::zero(); 1]; s_dim]; b_dim];
    
    // 파이썬 로직 동기화: Index = ((shifted_val * M_rms_idx) >> k) // dim
    let m_idx = bundle.m_rms_idx.into_bigint().as_ref()[0] as u128;

    for b in 0..b_dim {
        for s in 0..s_dim {
            let val = bundle.shifted_val[b][s][0].into_bigint().as_ref()[0] as u128;
            
            let numerator = val * m_idx;
            let scaled = numerator >> k_bits;
            let mut combined_idx = (scaled / (dim as u128)) as u64;

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

fn run_lasso_for_gelu(
    bundle: &GeLUBundle<Fr>,
    q_in_sim: &Vec<Vec<Vec<Fr>>>, // Passed from Linear output (Q)
    r_in_lin: &Vec<Vec<Vec<Fr>>>, // Passed from Linear remainder (R)
    k_bits: usize,
    label: &'static [u8],
    rng: &mut StdRng,
) -> Result<()> {
    let b_dim = bundle.q_out_sim.len();
    let s_dim = bundle.q_out_sim[0].len();
    let d_dim = bundle.q_out_sim[0][0].len();

    let mut indices = Vec::new();
    let offset = 32768u128; // TABLE_SIZE // 2 for fixed-resolution 0-point
    let m_gelu_idx = bundle.m_gelu_idx.into_bigint().as_ref()[0] as u128;

    for b in 0..b_dim {
        for s in 0..s_dim {
            for d in 0..d_dim {
                // 1. Reconstruct high-precision X_diff: (Q_in - z_in) * 2^k + R_in
                let q_in_diff = (q_in_sim[b][s][d].into_bigint().as_ref()[0] as i128) 
                                - (bundle.z_in.into_bigint().as_ref()[0] as i128);
                let r_in = r_in_lin[b][s][d].into_bigint().as_ref()[0] as i128;
                
                let x_diff_full = (q_in_diff << k_bits) + r_in;

                // 2. Map to Fixed Table Index: Index = (X_diff * M_gelu_idx >> k) + Offset
                let scaled_diff = (x_diff_full * m_gelu_idx as i128) >> k_bits;
                let mut idx = scaled_diff + offset as i128;

                // 3. Clipping to TABLE_SIZE (65536)
                if idx < 0 { idx = 0; }
                if idx >= GELU_M as i128 { idx = (GELU_M - 1) as i128; }
                
                indices.push([idx as usize]);
            }
        }
    }

    let sparsity = indices.len();
    let gens = SparsePolyCommitmentGens::<G1>::new(label, GELU_C, sparsity, GELU_C, 16);
    prove_lasso_for_gelu(&indices, &gens, rng, label)
}

fn prove_lasso_for_gelu(
    nz_indices: &Vec<[usize; GELU_C]>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<()> {
    let sparsity = nz_indices.len();
    let log_M = 16;
    
    let lookup_matrix = SparseLookupMatrix::<GELU_C>::new(nz_indices.clone(), log_M);
    let mut densified = DensifiedRepresentation::<Fr, GELU_C>::from(&lookup_matrix);

    let commitment = densified.commit::<G1>(gens);
    let r_point: Vec<Fr> = (0..sparsity.next_power_of_two().trailing_zeros() as usize)
        .map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape_gelu");
    let mut transcript = Transcript::new(b"lasso_gelu");
    
    // GeLUSubtableStrategy 사용
    let proof = SparsePolynomialEvaluationProof::<G1, GELU_C, GELU_M, GeLUSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut v_transcript = Transcript::new(b"lasso_gelu");
    proof.verify(&commitment, &r_point, gens, &mut v_transcript)
        .map_err(|_| anyhow::anyhow!("GeLU Lasso verification failed"))?;

    println!("GeLU Lasso proof verified: {:?}", std::str::from_utf8(label).unwrap());
    Ok(())
}

#[test]
fn mlp_test() {
    let p = Path::new("outputs/outputs-model.transformer_0.norm1_ff1_gelu.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            panic!("proof generate failed: {}", e);
        }
    }
}