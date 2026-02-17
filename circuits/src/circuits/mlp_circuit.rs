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
    cc_snark::legogroth16::{LinkPublicGenerators, prepare_verifying_key, generate_random_parameters_incl_cp_link, create_random_proof_incl_cp_link, verify_proof_incl_cp_link},
    //cc_snark::cc_gro::{CcGroth16, prepare_verifying_key}, 
    config::{RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M, GELU_M, GELU_C}, 
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
    }, util::load::{to_field_1d_i64, to_field_2d_i32, to_field_3d_i32, to_field_3d_i64, to_field_3d_u64},
    util::utils::{cm1_from_vk, hash_cm_to_field_vec},
};
use ark_std::{
    rand::{RngCore},
    vec::Vec,
};

pub fn get_link_public_gens<R: RngCore, E: Pairing>(
    rng: &mut R,
    count: u32,
) -> LinkPublicGenerators<E> {
    let pedersen_gens = (0..count)
        .map(|_| E::G1::rand(rng).into_affine())
        .collect::<Vec<_>>();
    let g1 = E::G1::rand(rng).into_affine();
    let g2 = E::G2::rand(rng).into_affine();
    LinkPublicGenerators {
        pedersen_gens,
        g1,
        g2,
    }
}

type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromAttentionCircuit<F: PrimeField> {
    pub norm1: NormBundle<F>,
    //pub norm2: NormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,
    pub challenge_r: Vec<F>,
    pub frac_bits: usize,
    pub k_bits: usize,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let dim: usize = self.norm1.q_ri[0][0].len();
        let challenge_r_var = Vec::<FpVar<F>>::new_input(cs.clone(), || Ok(self.challenge_r.clone()))?;
        let q_norm1_vars = enforce_rmsnorm(cs.clone(), &self.norm1, dim, self.frac_bits, self.k_bits)?;
        let (q_ff1_vars, r_ff1_vars) = enforce_linear(cs.clone(),&q_norm1_vars, &self.ff1, self.k_bits)?;
        let q_gelu_vars = enforce_gelu(
            cs.clone(),
            &q_ff1_vars,  
            &r_ff1_vars,  
            &self.gelu1,
            self.k_bits,
        )?;

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
    let n_len = norm1.q_ri.len() * norm1.q_ri[0].len();

    Ok(NystromAttentionCircuit {
        norm1,
        //norm2,
        ff1,
        gelu1,
        challenge_r: vec![E::ScalarField::zero(); n_len],
        frac_bits: frac_bits,
        k_bits: k_bits,
    })

    
}
/* 
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
        // --- [STEP 1] LegoGroth16 Setup (첫 번째 샘플 파일 기준) ---
        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        let cs = ark_relations::r1cs::ConstraintSystem::<ScalarF<E>>::new_ref();
        sample_circuit.clone().generate_constraints(cs.clone())?;
        
        if !cs.is_satisfied().unwrap() {
            match cs.which_is_unsatisfied() {
                Ok(Some(path)) => println!("Failed Path: {}", path),
                Ok(None) => println!("Else"),
                Err(e) => println!("Error while debugging: {:?}", e),
            }
            return Err(anyhow::anyhow!("Constraints NOT satisfied in {:?}", file_path));
        }

        let cs = ark_relations::r1cs::ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs.clone())?;
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
        
        // CP-link용 공용 생성기 (norm1.q_out 전체를 커밋하므로 그 길이에 맞춤)
        let num_commit_1 = sample_circuit.norm1.q_out.iter().flatten().flatten().count() as u32;
        let link_gens = get_link_public_gens::<_, E>(&mut rng, num_commit_1 + 1);

        println!("=== [0] LegoGroth16 Setup (Num Commit: {}) ===", num_commit_1);
        let params = generate_random_parameters_incl_cp_link::<E, _, _>(
            sample_circuit.clone(), link_gens, num_commit_1, 0, &mut rng
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        
        let pvk = prepare_verifying_key::<E>(&params.vk.groth16_vk);
        println!("\n=== Processing: {:?} ===", file_path);
        let circuit_data = load_nystrom_circuit::<E>(file_path)?;

        // 1. 실제 Public Input 개수 동적 분석 (Variable::One 제외)
        let cs_analyze = ConstraintSystem::<Fr>::new_ref();
        circuit_data.clone().generate_constraints(cs_analyze.clone())?;
        let actual_num_pub = cs_analyze.num_instance_variables() - 1;

        // 2. 커밋 데이터 수집 (RMSNorm의 출력값 q_out을 평탄화)
        let mut committed_group1 = Vec::new();
        for batch in &circuit_data.norm1.q_out {
            for seq in batch {
                for val in seq { committed_group1.push(*val); }
            }
        }

        // 3. Fiat-Shamir: 커밋 기반 챌린지 생성
        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let link_v = Fr::rand(&mut rng);

        let cm1 = cm1_from_vk::<E>(&params.vk.groth16_vk, actual_num_pub, &committed_group1, &v1);
        let r_vec = hash_cm_to_field_vec::<E>(&cm1, actual_num_pub);

        // 4. 최종 회로 빌드 및 CP-link 증명 생성
        let final_circuit = NystromAttentionCircuit { challenge_r: r_vec, ..circuit_data };
        let proof = create_random_proof_incl_cp_link(
            final_circuit.clone(), v1, v2, link_v, &params, &mut rng
        ).map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;

        // 5. 검증 (LegoGroth16 + CP-link)
        let r_verifier = hash_cm_to_field_vec::<E>(&proof.groth16_proof.d1, actual_num_pub);
        let mut public_inputs = Vec::new();
        public_inputs.extend_from_slice(&r_verifier);

        let ok = verify_proof_incl_cp_link(&pvk, &params.vk, &proof, &public_inputs);
        println!("  - LegoGroth16 + CP-link Verification: {:?}", ok);

        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        /*let dim = final_circuit.norm1.q_ri[0][0].len();
        run_lasso_for_rmsnorm(&final_circuit.norm1, final_circuit.k_bits, b"lasso_rms1", &mut rng)?;
        run_lasso_for_gelu(
            &final_circuit.gelu1, 
            &final_circuit.ff1.q_out_sim.iter().map(|batch| {
                batch.iter().map(|seq| {
                    seq.iter().map(|val| {
                        Fr::from((*val).into_bigint().as_ref()[0] as i64)
                    }).collect::<Vec<Fr>>()
                }).collect::<Vec<Vec<Fr>>>()
            }).collect::<Vec<Vec<Vec<Fr>>>>(),
            &final_circuit.ff1.r_out_lin, 
            final_circuit.k_bits, 
            b"lasso_gelu1", 
            &mut rng
        )?;*/
    }

    Ok(())
}*/

pub fn run_paths<E: Pairing>(path: &Path) -> Result<()>
where
    E: Pairing<ScalarField = Fr>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    type ScalarF<E> = <E as Pairing>::ScalarField;
    let mut targets: Vec<std::path::PathBuf> = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path).context("reading directory")? {
            let ent = entry?.path();
            if ent.is_file() && ent.extension().map_or(false, |s| s == "npz") {
                targets.push(ent);
            }
        }
        targets.sort();
    } else {
        targets.push(path.to_path_buf());
    }

    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets.iter() {
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);
        
        // 1. 초기 회로 로드 및 구조 분석
        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;
        let cs_analyze = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_analyze.clone())?;
        cs_analyze.finalize(); // 할당 순서 확정

        let num_pub = cs_analyze.num_instance_variables() - 1; 
        let num_commit_1 = sample_circuit.norm1.q_out.iter().flatten().flatten().count() as u32;

        // 2. LegoGroth16 Setup
        let link_gens = get_link_public_gens::<_, E>(&mut rng, num_commit_1 + 1);
        println!("=== [0] LegoGroth16 Setup (Num Commit: {}) ===", num_commit_1);
        let params = generate_random_parameters_incl_cp_link::<E, _, _>(
            sample_circuit.clone(), link_gens, num_commit_1, 0, &mut rng
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        
        let pvk = prepare_verifying_key::<E>(&params.vk.groth16_vk);

        // 3. [핵심 수정] 서킷 할당 순서와 100% 일치하는 Witness 추출
        // 수동으로 q_out을 모으는 대신, CS가 생성한 할당 리스트에서 앞부분을 가져옵니다.
        let committed_group1 = {
            let cs_inner = cs_analyze.borrow().ok_or_else(|| anyhow::anyhow!("CS borrow failed"))?;
            let full_witness = &cs_inner.witness_assignment;
            
            if full_witness.len() < num_commit_1 as usize {
                return Err(anyhow::anyhow!("Not enough witness variables: expected {}, found {}", num_commit_1, full_witness.len()));
            }
            // 정확한 개수만큼 슬라이싱하여 복사
            full_witness[0..num_commit_1 as usize].to_vec()
        };

        // 4. Fiat-Shamir: 정확한 Witness 그룹으로 커밋 및 챌린지 생성
        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let link_v = Fr::rand(&mut rng);

        let cm1 = cm1_from_vk::<E>(&params.vk.groth16_vk, num_pub, &committed_group1, &v1);
        let r_vec = hash_cm_to_field_vec::<E>(&cm1, num_pub);

        // 5. 최종 증명 생성
        println!("=== [0] LegoGroth16 GenProof ===");
        let final_circuit = NystromAttentionCircuit { challenge_r: r_vec, ..sample_circuit };
        let proof = create_random_proof_incl_cp_link(
            final_circuit, v1, v2, link_v, &params, &mut rng
        ).map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;

        // 6. 검증 (Public Input은 challenge_r 벡터 포함)
        let r_verifier = hash_cm_to_field_vec::<E>(&proof.groth16_proof.d1, num_pub);
        let mut public_inputs = Vec::new();
        public_inputs.extend_from_slice(&r_verifier);

        let result = verify_proof_incl_cp_link(&pvk, &params.vk, &proof, &public_inputs);
        println!("  - LegoGroth16 + CP-link Verification: {:?}", result);
        
        if let Err(e) = result {
            println!("    Reason: {:?}", e);
        }
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