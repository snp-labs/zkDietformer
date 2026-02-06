use crate::{config::{SOFTMAX_C, SOFTMAX_M}, gadgets::matmul::quantized::MatmulData, util::load::{to_field_5d_i64, to_field_5d_u64}}; 
use std::{fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix4, Array2, Ix2, Ix3, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{UniformRand, log2, marker::PhantomData, rand::{Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use crate::{
    cc_snark::cc_gro::{CcGroth16, prepare_verifying_key},
    gadgets::{
        matmul::{
            constraints::{new_matrix_witness, MatrixVar}, 
            quantized::{
                enforce_freivald, 
                ZenParams, 
                new_matrix_witness_4d, 
                transpose_matrix, 
                gen_challenge_vector,
                compute_circuit_landmarks
            },},
        softmax::{
            constraints::{get_softmax_witness_at, 
                enforce_softmax, 
                new_matrix_witness_3d,
                enforce_softmax_without_sum
            },
        }
    }, util::{
        load::{read_array_i64, to_field_4d_i64, to_field_4d_u64, to_field_3d_u64},
    }, 
};
use ark_std::fs;
use crate::lasso::{
    lasso::{
        densified::DensifiedRepresentation,
        surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
    },
    subtables::{
        and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy,
        softmax::SoftmaxSubtableStrategy,
        SubtableStrategy,
    },
    utils::math::Math,
    utils::random::RandomTape,
};
use merlin::Transcript;
use std::time::Instant;
use std::error::Error;

type G1 = G1Projective;
#[derive(Clone)]
pub struct NystromAttentionCircuit<F: PrimeField> {
    // 공통 입력 (한 번만 로드)
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,       
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,       
    pub z_q: F,      
    pub z_k: F,      
    
    // Kernel 1
    pub r_mean_k1: Vec<Vec<Vec<Vec<F>>>>, 
    pub q_sim_k1: Vec<Vec<Vec<Vec<F>>>>,
    pub r_matmul_k1: Vec<Vec<Vec<Vec<F>>>>,
    pub m_k1: F,
    pub m_prime_k1: F,
    pub z_sim_k1: F,

    // Kernel 2
    pub r_mean_k2: Vec<Vec<Vec<Vec<F>>>>,
    pub q_sim_k2: Vec<Vec<Vec<Vec<F>>>>,
    pub r_matmul_k2: Vec<Vec<Vec<Vec<F>>>>,
    pub m_k2: F,
    pub m_prime_k2: F,
    pub z_sim_k2: F,

    // Kernel 3
    pub q_sim_k3: Vec<Vec<Vec<Vec<F>>>>,
    pub r_matmul_k3: Vec<Vec<Vec<Vec<F>>>>,
    pub m_k3: F,
    pub m_prime_k3: F,
    pub z_sim_k3: F,

    // softmax(Kernel1)
    pub limbs_k1: Vec<Vec<Vec<Vec<Vec<F>>>>>,     // [K_SEGMENTS, B, H, S, L]
    pub exp_limbs_k1: Vec<Vec<Vec<Vec<Vec<F>>>>>, // [K_SEGMENTS, B, H, S, L]
    pub q_exp_k1: Vec<Vec<Vec<Vec<Vec<F>>>>>,     // [K_SEGMENTS-2, B, H, S, L] (중간 몫)
    pub r_exp_k1: Vec<Vec<Vec<Vec<Vec<F>>>>>,     // [K_SEGMENTS-1, B, H, S, L] (중간 나머지)
    pub exp_total_k1: Vec<Vec<Vec<Vec<F>>>>,      // 소프트맥스 분자 (최종 지수 합)
    pub exp_sum_k1: Vec<Vec<Vec<F>>>,            // 소프트맥스 분모 (L 방향 합계)
    pub p_k1: Vec<Vec<Vec<Vec<F>>>>,              // 최종 확률 P (몫)
    pub r_div_k1: Vec<Vec<Vec<Vec<F>>>>,          // 나눗셈 나머지 R


    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub r_challenge: F, 
    pub seg_size: usize,
    pub k_bits: usize,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k = FpVar::Constant(F::from(1u64 << 22));

        // 1. 공통 입력 및 위트니스 할당 ([B][H][S][D])
        let q_proc = new_matrix_witness_4d(cs.clone(), &self.q_q);
        let k_proc = new_matrix_witness_4d(cs.clone(), &self.q_k);
        let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &self.r_mean_k1);
        let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &self.r_mean_k2);

        // 2. Kernel 결과 데이터 할당
        let k1_data = MatmulData {
            q_sim: new_matrix_witness_4d(cs.clone(), &self.q_sim_k1),
            r_matmul: new_matrix_witness_4d(cs.clone(), &self.r_matmul_k1),
        };
        let k2_data = MatmulData {
            q_sim: new_matrix_witness_4d(cs.clone(), &self.q_sim_k2),
            r_matmul: new_matrix_witness_4d(cs.clone(), &self.r_matmul_k2),
        };
        let k3_data = MatmulData {
            q_sim: new_matrix_witness_4d(cs.clone(), &self.q_sim_k3),
            r_matmul: new_matrix_witness_4d(cs.clone(), &self.r_matmul_k3),
        };

        // 3. 프리발드 파라미터 설정
        let params_k1 = ZenParams {
            m: FpVar::Constant(self.m_k1), m_prime: FpVar::Constant(self.m_prime_k1),
            z_w: FpVar::Constant(self.z_q), z_x: FpVar::Constant(self.z_k), z_y: FpVar::Constant(self.z_sim_k1),
            two_k: two_k.clone(),
        };
        let params_k2 = ZenParams {
            m: FpVar::Constant(self.m_k2), m_prime: FpVar::Constant(self.m_prime_k2),
            z_w: FpVar::Constant(self.z_q), z_x: FpVar::Constant(self.z_k), z_y: FpVar::Constant(self.z_sim_k2),
            two_k: two_k.clone(),
        };
        let params_k3 = ZenParams {
            m: FpVar::Constant(self.m_k3), m_prime: FpVar::Constant(self.m_prime_k3),
            z_w: FpVar::Constant(self.z_q), z_x: FpVar::Constant(self.z_k), z_y: FpVar::Constant(self.z_sim_k3),
            two_k: two_k.clone(),
        };

        // 4. 소프트맥스 결과 데이터 할당
        let limbs_vars = self.limbs_k1.iter().map(|l| new_matrix_witness_4d(cs.clone(), l));
        let exp_limbs_vars = self.exp_limbs_k1.iter().map(|e| new_matrix_witness_4d(cs.clone(), e));
        let q_exp_vars = self.q_exp_k1.iter().map(|q| new_matrix_witness_4d(cs.clone(), q));
        let r_exp_vars = self.r_exp_k1.iter().map(|r| new_matrix_witness_4d(cs.clone(), r));

        let exp_total_var = new_matrix_witness_4d(cs.clone(), &self.exp_total_k1);
        let p_k1_var = new_matrix_witness_4d(cs.clone(), &self.p_k1);
        let r_div_var = new_matrix_witness_4d(cs.clone(), &self.r_div_k1);
        let exp_sum_var = new_matrix_witness_3d(cs.clone(), &self.exp_sum_k1)?; // 3차원 할당 함수 사용

        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;
        
        for b in 0..q_proc.len() {
            // 해당 배치의 Landmarks 계산
            let k_landmarks = compute_circuit_landmarks(cs.clone(), &k_proc[b], &r_mean_k1[b], self.seg_size)?;
            let q_landmarks = compute_circuit_landmarks(cs.clone(), &q_proc[b], &r_mean_k2[b], self.seg_size)?;

            for h in 0..q_proc[b].len() {
                let k_l_t = transpose_matrix(&k_landmarks[h]);
                let r_pows_k1 = gen_challenge_vector(&r_var, k1_data.q_sim[b][h][0].len());
                // Kernel 1
                enforce_freivald(
                    &q_proc[b][h], 
                    &k_l_t, 
                    &k1_data.q_sim[b][h], 
                    &k1_data.r_matmul[b][h], 
                    &params_k1, 
                    &r_pows_k1
                )?;

                // Kernel 2
                let q_l_h = &q_landmarks[h];
                let r_pows_k2: Vec<FpVar<F>> = gen_challenge_vector(&r_var, k2_data.q_sim[b][h][0].len());
                enforce_freivald(
                    q_l_h, 
                    &k_l_t, 
                    &k2_data.q_sim[b][h], 
                    &k2_data.r_matmul[b][h], 
                    &params_k2, 
                    &r_pows_k2
                )?;

                let k_full_t = transpose_matrix(&k_proc[b][h]);
                let r_pows_k3 = gen_challenge_vector(&r_var, k3_data.q_sim[b][h][0].len()); 
                enforce_freivald(
                    &q_landmarks[h], 
                    &k_full_t, 
                    &k3_data.q_sim[b][h], 
                    &k3_data.r_matmul[b][h], 
                    &params_k3, 
                    &r_pows_k3
                )?;

                // Softmax Kernel 1 검증
                enforce_softmax_without_sum(
                    &k1_data.q_sim[b][h], 
                    &k1_data.r_matmul[b][h],
                    &limbs_vars.clone().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &exp_limbs_vars.clone().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &q_exp_vars.clone().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &r_exp_vars.clone().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &exp_total_var[b][h],
                    &p_k1_var[b][h],
                    &r_div_var[b][h],
                    self.frac_bits,
                    &two_k_var,
                )?;
                
            }
        }

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

    // 2. 공통 챌린지 값 로드
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let r = E::ScalarField::from(42u64); // 혹은 결정론적 해시 결과값
    
    // 3. 공통 입력 로드
    let q_q = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?.into_dimensionality()?;
    let q_k = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?.into_dimensionality()?;
    let z_q = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_q.npy")?.into_scalar();
    let z_k = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_k.npy")?.into_scalar();
    let seg= npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("seg_size.npy")?.into_scalar() as usize;

    // 4. Kernel 1 데이터 로드
    let r_mean_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_mean_k1.npy")?.into_dimensionality()?;
    let q_sim_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_k1.npy")?.into_dimensionality()?;
    let r_matmul_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_k1.npy")?.into_dimensionality()?;
    let m_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_k1.npy")?.into_scalar();
    let m_prime_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_prime_k1.npy")?.into_scalar();
    let z_sim_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_sim_k1.npy")?.into_scalar();

    // 5. Kernel 2 데이터 로드
    let r_mean_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_mean_k2.npy")?.into_dimensionality()?;
    let q_sim_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_k2.npy")?.into_dimensionality()?;
    let r_matmul_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_k2.npy")?.into_dimensionality()?;
    let m_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_k2.npy")?.into_scalar();
    let m_prime_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_prime_k2.npy")?.into_scalar();
    let z_sim_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_sim_k2.npy")?.into_scalar();

    let q_sim_k3 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_k3.npy")?.into_dimensionality()?;
    let r_matmul_k3 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_k3.npy")?.into_dimensionality()?;
    let m_k3 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_k3.npy")?.into_scalar();
    let m_prime_k3 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_prime_k3.npy")?.into_scalar();
    let z_sim_k3 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_sim_k3.npy")?.into_scalar();

    let limbs_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("limbs_k1.npy")?; // [5, B, H, S, L]
    let exp_limbs_k1 = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("exp_limbs_k1.npy")?;
    let q_exp_k1 = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("Q_exp_k1.npy")?;
    let r_exp_k1 = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("R_exp_k1.npy")?;
    let exp_total_k1 = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("exp_total_k1.npy")?.into_dimensionality()?;
    let exp_sum_k1_4d = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("exp_sum_k1.npy")?;
    // exp_sum_k1 was stored with shape (B, H, S, 1). Convert to (B, H, S) by dropping the last axis.
    let shape4: (usize, usize, usize, usize) = exp_sum_k1_4d.dim();
    let (b_dim, h_dim, s_dim, last_dim) = (shape4.0, shape4.1, shape4.2, shape4.3);
    if last_dim != 1 {
        return Err(anyhow::anyhow!("exp_sum_k1 last dimension expected to be 1, got {}", last_dim));
    }
    // into_shape will work because total elements match
    let exp_sum_k1_3d = exp_sum_k1_4d.into_shape((b_dim, h_dim, s_dim))?; // Ix3
    let p_k1 = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("P_k1.npy")?.into_dimensionality()?;
    let r_div_k1 = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("R_div_k1.npy")?.into_dimensionality()?;

    let frac_bits = npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let limb_bits = npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>("limb_bits.npy")?.into_scalar() as usize;
    let k_segments = npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>("k_segments.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<i32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    // 6. NystromAttentionCircuit 인스턴스 생성
    Ok(NystromAttentionCircuit {
        q_q: to_field_4d_i64(&q_q),
        q_k: to_field_4d_i64(&q_k),
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),

        r_mean_k1: to_field_4d_i64(&r_mean_k1),
        q_sim_k1: to_field_4d_i64(&q_sim_k1),
        r_matmul_k1: to_field_4d_i64(&r_matmul_k1),
        m_k1: E::ScalarField::from(m_k1),
        m_prime_k1: E::ScalarField::from(m_prime_k1),
        z_sim_k1: E::ScalarField::from(z_sim_k1),

        r_mean_k2: to_field_4d_i64(&r_mean_k2),
        q_sim_k2: to_field_4d_i64(&q_sim_k2),
        r_matmul_k2: to_field_4d_i64(&r_matmul_k2),
        m_k2: E::ScalarField::from(m_k2),
        m_prime_k2: E::ScalarField::from(m_prime_k2),
        z_sim_k2: E::ScalarField::from(z_sim_k2),

        q_sim_k3: to_field_4d_i64(&q_sim_k3), 
        r_matmul_k3: to_field_4d_i64(&r_matmul_k3), 
        m_k3: E::ScalarField::from(m_k3), 
        m_prime_k3: E::ScalarField::from(m_prime_k3), 
        z_sim_k3: E::ScalarField::from(z_sim_k3),

        // Softmax Kernel 1
        limbs_k1: to_field_5d_i64(&limbs_k1),
        exp_limbs_k1: to_field_5d_u64(&exp_limbs_k1),
        q_exp_k1: to_field_5d_u64(&q_exp_k1),
        r_exp_k1: to_field_5d_u64(&r_exp_k1),
        exp_total_k1: to_field_4d_u64(&exp_total_k1),
        exp_sum_k1: to_field_3d_u64(&exp_sum_k1_3d),
        p_k1: to_field_4d_u64(&p_k1),
        r_div_k1: to_field_4d_u64(&r_div_k1),

        frac_bits: frac_bits,
        limb_bits: limb_bits,
        k_segments: k_segments,
        r_challenge: r,
        seg_size: seg,
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

    // RNG for all setups/proofs (deterministic seed for reproducibility here)
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets.iter() {
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);
        let circuit= load_nystrom_circuit::<E>(file_path)?;

        let cs = ark_relations::r1cs::ConstraintSystem::<ScalarF<E>>::new_ref();
        circuit.clone().generate_constraints(cs.clone())?;
        
        if !cs.is_satisfied().unwrap() {
            match cs.which_is_unsatisfied() {
                Ok(Some(path)) => println!("Failed Path: {}", path),
                Ok(None) => println!("제약 조건이 만족되지 않으나 경로를 찾을 수 없음."),
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
        // Per-file SNARK flow: setup, prove, verify
        let setup_start = Instant::now();
        let (pk, vk) = CcGroth16::<E>::setup(circuit.clone(), &mut rng)?;
        let setup_duration = setup_start.elapsed();
        println!("  - Setup Time: {:?}", setup_duration);

        let prove_start = Instant::now();
        let proof = CcGroth16::<E>::prove(&pk, circuit.clone(), &mut rng)?;
        let prove_duration = prove_start.elapsed();
        println!("  - Prover Time (ccGroth): {:?}", prove_duration);

        // public inputs: include the Fiat-Shamir r_challenge
        let pvk = prepare_verifying_key::<E>(&vk);
        let public_inputs = vec![circuit.r_challenge];
        let verify_start = Instant::now();
        let ok = CcGroth16::<E>::verify_with_processed_vk(&pvk, &public_inputs, &proof)?;
        println!("Verification result for {:?}: {}", file_path, ok);
        if !ok {
            return Err(anyhow::anyhow!("Verification failed for {:?}", file_path));
        }
        let verify_duration = verify_start.elapsed();
        println!("  - Verifier Time (ccGroth): {:?}", verify_duration);
        println!("  - Verification Result: {}", ok);

    }

    Ok(())
}

#[test]
fn combined_full_witness_test() {
    // let cs = ConstraintSystem::<Fr>::new_ref();
    // cs.set_mode(ark_relations::r1cs::SynthesisMode::Setup); 

    let p = Path::new("outputs/outputs-model.transformer_0.mha.attn.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            panic!("proof generate failed: {}", e);
        }
    }
}