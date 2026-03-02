use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix1, Ix4, Array2, Ix2, Ix3, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{rc::Rc, fs, UniformRand, log2, marker::PhantomData, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}, cell::RefCell};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{R1CSVar, alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use merlin::Transcript;
use ark_r1cs_std::select::CondSelectGadget;
use crate::{
    cc_snark::legogroth16::{LinkPublicGenerators, prepare_verifying_key, generate_random_parameters_incl_cp_link, create_random_proof_incl_cp_link, verify_proof_incl_cp_link},
    // cc_snark::cc_gro::{CcGroth16, prepare_verifying_key}, config::{FRACBITS_C, FRACBITS_M, KBITS_C, KBITS_M, SOFTMAX_C, SOFTMAX_M}, 
    gadgets::{
        matmul::{constraints::{MatrixVar, new_matrix_witness}, 
    quantized::{MatmulBundle, MatmulData, PseudoinverseBundle, ZenParams, compute_circuit_landmarks, enforce_freivald, gen_challenge_vector, new_matrix_witness_4d, transpose_matrix}}, softmax::constraints::{SoftmaxBundle, enforce_softmax}}, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        subtables::{
            SubtableStrategy, and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
        }, utils::{math::Math, random::RandomTape},
    }, util::{
        utils::{cm1_from_vk, hash_cm_to_field},
        load::{read_array_i64, to_field_2d_i64, to_field_3d_u64, to_field_4d_i64, to_field_4d_u64, to_field_5d_i64, to_field_5d_u64},
        lookup_attention::{verify_all_remainders, prove_lasso_for_softmax, run_lasso_bundle},
    }
};
use crate::cc_snark::legogroth16::{
    ProvingKey,          
    create_random_proof, 
    verify_proof,       
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
    // 공통 입력 (한 번만 로드)
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,       
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,     
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,  
    pub z_q: F,      
    pub z_k: F, 
    pub z_v: F,

    pub matmul1: MatmulBundle<F>,
    pub matmul2: MatmulBundle<F>,
    pub matmul3: MatmulBundle<F>,


    pub sm1: SoftmaxBundle<F>,
    pub sm2: SoftmaxBundle<F>,
    pub sm3: SoftmaxBundle<F>,

    pub pseudo_inv: PseudoinverseBundle<F>,

    pub matmul_temp: MatmulBundle<F>,
    pub matmul_mid: MatmulBundle<F>,
    pub matmul_x: MatmulBundle<F>,

    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub r_challenge: F, 
    pub seg_size: usize,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,

    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    // pub challenge_r: Vec<F>,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        let mut committed_pool = self.committed_vars.borrow_mut();
        
        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        // 1. 공통 입력 및 위트니스 할당 ([B][H][S][D])
        let q_proc = new_matrix_witness_4d(cs.clone(), &self.q_q);
        let k_proc = new_matrix_witness_4d(cs.clone(), &self.q_k);
        let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &self.matmul1.r_mean);
        let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &self.matmul2.r_mean);

        // 2. Kernel 결과 데이터 할당
        let (k1_data, params_k1) = self.matmul1.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (k2_data, params_k2) = self.matmul2.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (k3_data, params_k3) = self.matmul3.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm1_vars = self.sm1.alloc_vars(cs.clone())?;
        let sm2_vars = self.sm2.alloc_vars(cs.clone())?;
        let sm3_vars = self.sm3.alloc_vars(cs.clone())?;

        // temp
        let v_proc = new_matrix_witness_4d(cs.clone(), &self.q_v); 
        let (temp_data, params_temp) = self.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);
        let (mid_data, params_mid) = self.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), self.pseudo_inv.z_inv, &two_k_var);
        let (x_data, params_x) = self.matmul_x.alloc_and_setup(cs.clone(), self.matmul_mid.z_sim, self.matmul_temp.z_sim, &two_k_var);
        
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
                    &r_pows_k1,
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
                println!("DEBUG: Current pool size before softmax: {}", committed_pool.len());
                enforce_softmax(
                    &k1_data.q_sim[b][h], 
                    &k1_data.r_matmul[b][h],
                    &sm1_vars.r_sm[b][h],        
                    &sm1_vars.m_sm,
                    &sm1_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm1_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm1_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    //&sm1_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    //&sm1_vars.exp_total[b][h],
                    &sm1_vars.p[b][h],
                    &sm1_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut committed_pool
                )?;
                println!("DEBUG: Current pool size after softmax: {}", committed_pool.len());

                enforce_softmax(
                    &k2_data.q_sim[b][h], 
                    &k2_data.r_matmul[b][h],
                    &sm2_vars.r_sm[b][h],        
                    &sm2_vars.m_sm,
                    &sm2_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm2_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm2_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    // &sm2_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    // &sm2_vars.exp_total[b][h],
                    &sm2_vars.p[b][h],
                    &sm2_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut committed_pool
                )?;
                println!("DEBUG: Current pool size after softmax: {}", committed_pool.len());

                // Kernel 3 Matmul 결과인 k3_data.q_sim[b][h]는 [L][S] 형태입니다.
                let zero_var = FpVar::zero();
                let num_landmarks = k3_data.q_sim[b][h].len();
                let seq_len = k3_data.q_sim[b][h][0].len();

                // 몫(Q_sim) 변수 자체를 마스킹된 버전으로 교체
                let mut masked_q_sim_k3 = k3_data.q_sim[b][h].clone();
                for l in 0..num_landmarks {
                    for s in 0..seq_len {
                        // mask_var[b][s] (FpVar)를 BooleanVar로 변환
                        let is_not_padding = mask_var[b][s].is_one()?; 
                        
                        // 정적 메서드 호출로 문법 오류 해결
                        masked_q_sim_k3[l][s] = FpVar::conditionally_select(
                            &is_not_padding,
                            &k3_data.q_sim[b][h][l][s], 
                            &zero_var
                        )?;
                    }
                }

                enforce_softmax(
                    &masked_q_sim_k3, // 마스킹 적용됨
                    &k3_data.r_matmul[b][h],
                    &sm3_vars.r_sm[b][h],        
                    &sm3_vars.m_sm,
                    &sm3_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm3_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm3_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    //&sm3_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    //&sm3_vars.exp_total[b][h],
                    &sm3_vars.p[b][h],
                    &sm3_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut committed_pool
                )?;

                // moore-penrose
                let q_inv_var = new_matrix_witness_4d(cs.clone(), &self.pseudo_inv.q_inv);
                let q_y_vars = self.pseudo_inv.q_y_intermediates.iter()
                    .map(|q_step| new_matrix_witness_4d(cs.clone(), q_step))
                    .collect::<Vec<_>>();
                let r_inv_vars = self.pseudo_inv.r_inv_remainders.iter()
                    .map(|r_step| new_matrix_witness_4d(cs.clone(), r_step))
                    .collect::<Vec<_>>();

                let z_zero = FpVar::zero();
                let z_inv = FpVar::Constant(self.pseudo_inv.z_inv);

                let mut inv_params = Vec::new();
                for i in 0..4 {
                    let (zw, zx, zy) = match i {
                        0 => (&z_zero, &z_inv, &z_zero), 
                        1 => (&z_zero, &z_zero, &z_zero), 
                        2 => (&z_inv, &z_zero, &z_inv),  
                        3 => (&z_inv, &z_inv, &z_inv),   
                        _ => unreachable!(),
                    };

                    inv_params.push(ZenParams {
                        m: FpVar::Constant(self.pseudo_inv.m_inv[i]),
                        m_prime: FpVar::Constant(self.pseudo_inv.m_prime_inv[i]),
                        z_w: zw.clone(),
                        z_x: zx.clone(),
                        z_y: zy.clone(),
                        two_k: two_k_var.clone(),
                    });
                }

                let r_pows_inv = gen_challenge_vector(&r_var, q_inv_var[b][h].len());

                // A * A_inv * A = A
                // Step 1: P_k2 * Q_inv_k2 = Q_y1
                enforce_freivald(
                    &sm2_vars.p[b][h], 
                    &q_inv_var[b][h], 
                    &q_y_vars[0][b][h], 
                    &r_inv_vars[0][b][h], 
                    &inv_params[0], 
                    &r_pows_inv
                )?;

                // Step 2: Q_y1 * P_k2 = P_k2
                enforce_freivald(
                    &q_y_vars[0][b][h], // 이거를 다시
                    &sm2_vars.p[b][h], 
                    &q_y_vars[1][b][h],
                    &r_inv_vars[1][b][h], 
                    &inv_params[1], 
                    &r_pows_inv
                )?;
                
                // A_inv * A * A_inv = A_inv
                // Step 2-1: Q_inv (A+) * P_k2 (A) = Q_y3
                enforce_freivald(
                    &q_inv_var[b][h],             
                    &sm2_vars.p[b][h], 
                    &q_y_vars[2][b][h],             
                    &r_inv_vars[2][b][h],        
                    &inv_params[2],              
                    &r_pows_inv
                )?;

                // Step 2-2: Q_y3 * Q_inv (A+) = Q_y4
                enforce_freivald(
                    &q_y_vars[2][b][h],            
                    &q_inv_var[b][h],
                    &q_y_vars[3][b][h],             
                    &r_inv_vars[3][b][h],       
                    &inv_params[3],             
                    &r_pows_inv
                )?;

                let r_pows_temp = gen_challenge_vector(&r_var, temp_data.q_sim[b][h][0].len());
                enforce_freivald(
                    &sm3_vars.p[b][h],            
                    &v_proc[b][h],                
                    &temp_data.q_sim[b][h],       
                    &temp_data.r_matmul[b][h],    
                    &params_temp, 
                    &r_pows_temp                  
                )?;

                let r_pows_mid = gen_challenge_vector(&r_var, mid_data.q_sim[b][h][0].len()); // 길이 L
                enforce_freivald(
                    &sm1_vars.p[b][h],        
                    &q_inv_var[b][h],         
                    &mid_data.q_sim[b][h],    
                    &mid_data.r_matmul[b][h], 
                    &params_mid, 
                    &r_pows_mid
                )?;


                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len()); // 길이 D
                enforce_freivald(
                    &mid_data.q_sim[b][h],    
                    &temp_data.q_sim[b][h],   
                    &x_data.q_sim[b][h],      
                    &x_data.r_matmul[b][h],  
                    &params_x, 
                    &r_pows_x
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
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let r = E::ScalarField::zero(); // 혹은 결정론적 해시 결과값
    
    let q_q = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?.into_dimensionality()?;
    let q_k = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?.into_dimensionality()?;
    let z_q = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar();
    let z_k = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar();
    let seg= npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("seg_size.npy")?.into_scalar() as usize;

    let load_matmul = |npz: &mut NpzReader<File>, suffix: &str, has_rmean: bool| -> Result<MatmulBundle<E::ScalarField>> {
        Ok(MatmulBundle {
            // 커널 3처럼 r_mean이 없는 경우 빈 벡터 할당
            r_mean: if has_rmean {
                to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_mean_{}.npy", suffix))?.into_dimensionality()?)
            } else {
                vec![] 
            },
            q_sim: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_sim_{}.npy", suffix))?.into_dimensionality()?),
            r_matmul: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_matmul_{}.npy", suffix))?.into_dimensionality()?),
            m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", suffix))?.into_scalar()),
            m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", suffix))?.into_scalar()),
            z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_sim_{}.npy", suffix))?.into_scalar()),
        })
    };

    let matmul1 = load_matmul(&mut npz, "k1", true)?;
    let matmul2 = load_matmul(&mut npz, "k2", true)?;
    let matmul3 = load_matmul(&mut npz, "k3", false)?;
    
    let load_softmax = |npz: &mut NpzReader<File>, suffix: &str| -> Result<SoftmaxBundle<E::ScalarField>> {
        Ok(SoftmaxBundle {
            limbs: to_field_5d_i64(&npz.by_name::<_, IxDyn>(&format!("limbs_{}.npy", suffix))?),
            exp_limbs: to_field_5d_u64(&npz.by_name::<_, IxDyn>(&format!("exp_limbs_{}.npy", suffix))?),
            q_exp: to_field_5d_u64(&npz.by_name::<_, IxDyn>(&format!("Q_exp_{}.npy", suffix))?),
            r_exp: to_field_5d_u64(&npz.by_name::<_, IxDyn>(&format!("R_exp_{}.npy", suffix))?),
            exp_total: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("exp_total_{}.npy", suffix))?.into_dimensionality()?),
            p: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("P_{}.npy", suffix))?.into_dimensionality()?),
            r_div: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("R_div_{}.npy", suffix))?.into_dimensionality()?),
            r_sm: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("r_sm_{}.npy", suffix))?.into_dimensionality()?),
            m_sm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar()), // 또는 커널별 suffix 적용
        })
    };

    let sm1 = load_softmax(&mut npz, "k1")?;
    let sm2 = load_softmax(&mut npz, "k2")?;
    let sm3 = load_softmax(&mut npz, "k3")?;

    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let limb_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("limb_bits.npy")?.into_scalar() as usize;
    let k_segments = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_segments.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    // Pseudoinverse 관련 데이터 로드
    let q_inv_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_inv_k2.npy")?.into_dimensionality()?;
    let z_inv_k2 = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_inv_k2.npy")?.into_scalar();

    // M_inv_all (4개), M_prime_inv_all (4개)
    let m_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_inv_all.npy")?;
    let m_prime_inv_all = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("M_prime_inv_all.npy")?;
    let q_y_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("Q_y_inv_intermediates.npy")?;
    let r_inv_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("R_inv_remainders.npy")?;

    let pseudo_inv = PseudoinverseBundle {
        q_inv: to_field_4d_i64(&q_inv_k2),
        z_inv: E::ScalarField::from(z_inv_k2),
        m_inv: [
            E::ScalarField::from(m_inv_all[0]),
            E::ScalarField::from(m_inv_all[1]),
            E::ScalarField::from(m_inv_all[2]),
            E::ScalarField::from(m_inv_all[3]),
        ],
        m_prime_inv: [
            E::ScalarField::from(m_prime_inv_all[0]),
            E::ScalarField::from(m_prime_inv_all[1]),
            E::ScalarField::from(m_prime_inv_all[2]),
            E::ScalarField::from(m_prime_inv_all[3]),
        ],
        q_y_intermediates: to_field_5d_i64(&q_y_inv_raw),
        r_inv_remainders: to_field_5d_i64(&r_inv_raw),
    };

    let q_v = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_v.npy")?.into_dimensionality()?;
    let z_v = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar();
    let matmul_temp = load_matmul(&mut npz, "temp", false)?;
    let matmul_mid = load_matmul(&mut npz, "mid", false)?; // P_k1 @ Q_inv_k2
    let matmul_x = load_matmul(&mut npz, "x", false)?;     // mid @ temp

    Ok(NystromAttentionCircuit {
        q_q: to_field_4d_i64(&q_q),
        q_k: to_field_4d_i64(&q_k),
        q_v: to_field_4d_i64(&q_v),
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),
        z_v: E::ScalarField::from(z_v),

        matmul1: matmul1,
        matmul2: matmul2,
        matmul3: matmul3,

        sm1: sm1,
        sm2: sm2,
        sm3: sm3,

        pseudo_inv,
        matmul_temp,
        matmul_mid,
        matmul_x,

        frac_bits: frac_bits,
        limb_bits: limb_bits,
        k_segments: k_segments,
        r_challenge: r,
        seg_size: seg,
        k_bits: k_bits,
        mask: to_field_2d_i64(&mask_raw),

        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })

    
}

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
    
        let num_commit_1 = {
            let temp_circuit = load_nystrom_circuit::<E>(&file_path)?;

            let vars_ptr = Rc::clone(&temp_circuit.committed_vars);

            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;

            vars_ptr.borrow().len() as u32
        };

        
        println!("최종 측정된 커밋 개수: {:?}", num_commit_1);

        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        // LegoGroth16 Setup
        let link_gens = get_link_public_gens::<_, E>(&mut rng, num_commit_1 + 1);
        println!("=== [0] LegoGroth16 Setup (Num Commit: {:?}) ===", num_commit_1);
        
        let params = generate_random_parameters_incl_cp_link::<E, _, _>(
            sample_circuit.clone(), 
            link_gens, 
            num_commit_1, 
            0, 
            &mut rng
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        
        let pvk = prepare_verifying_key::<E>(&params.vk.groth16_vk);

        // Witness 값 추출을 위해 sample_circuit을 한 번 실행 (Synthesize)
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;

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

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let link_v = Fr::rand(&mut rng);
        
        // 서킷의 공개 입력 개수
        let num_pub = cs_witness.num_instance_variables() - 1; 

        let cm1 = cm1_from_vk::<E>(&params.vk.groth16_vk, num_pub, &committed_group1, &v1);
        let r_fs = hash_cm_to_field::<E>(&cm1);

        println!("=== [1] LegoGroth16 GenProof ===");

        let groth16_pk = ProvingKey {
            vk: params.vk.groth16_vk.clone(),
            common: params.common.clone(),
        };

        println!("=== [1] LegoGroth16 GenProof (SNARK Logic Only) ===");

        let final_circuit = NystromAttentionCircuit {
            r_challenge: r_fs,
            ..sample_circuit.clone()
        };

        // 1. 증명 생성: CP-Link 없이 LegoGroth16 SNARK 본체만 생성
        // v1, v2는 위트니스 그룹 1, 2에 대한 blinding factor입니다.
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(), 
            v1, 
            v2, 
            &groth16_pk, 
            &mut rng
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;

        // 2. 증명 검증: CP-Link 검증을 제외하고 SNARK 제약 조건 및 커밋먼트 정당성만 검증
        let public_inputs = vec![final_circuit.r_challenge]; 
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);

        println!("  - LegoGroth16 SNARK Verification: {:?}", result);

        if let Err(e) = result {
            println!("    Reason: {:?}", e);
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        run_lasso_bundle(&final_circuit.sm1, b"lasso_k1", &mut rng)?;
        run_lasso_bundle(&final_circuit.sm2, b"lasso_k2", &mut rng)?;
        run_lasso_bundle(&final_circuit.sm3, b"lasso_k3", &mut rng)?;
        verify_all_remainders::<Fr, G1>(&final_circuit, &mut rng)?;
    }
    Ok(())
}

#[test]
fn attention_test() {
    let p = Path::new("outputs/outputs-model.transformer_0.mha.attn.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            panic!("proof generate failed: {}", e);
        }
    }
}