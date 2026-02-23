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
    config::{RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M, GELU_M, GELU_C, KBITS_C, KBITS_M}, 
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
    }, util::{
        load::{to_field_1d_i64, to_field_2d_i32, to_field_3d_i32, to_field_3d_i64, to_field_3d_u64},
        utils::{cm1_from_vk, hash_cm_to_field_vec},
        lookup_mlp::{run_mlp_remainders, run_lasso_for_gelu, run_lasso_for_rmsnorm}
    }, 
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
pub struct NystromMLPCircuit<F: PrimeField> {
    pub norm1: NormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,
    pub ff2: LinearBundle<F>,
    pub norm2: NormBundle<F>,
    pub challenge_r: Vec<F>,
    pub frac_bits: usize,
    pub k_bits: usize,
}
impl<F: PrimeField> ConstraintSynthesizer<F> for NystromMLPCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let dim: usize = self.norm1.q_ri[0][0].len();
        let challenge_r_var = Vec::<FpVar<F>>::new_input(cs.clone(), || Ok(self.challenge_r.clone()))?;
        let q_norm1_vars = enforce_rmsnorm(cs.clone(), &self.norm1, dim, self.frac_bits, self.k_bits)?;
        let (q_ff1_vars, r_ff1_vars) = enforce_linear(cs.clone(), &q_norm1_vars, &self.ff1, self.k_bits)?;
        let q_gelu_vars = enforce_gelu(cs.clone(), &q_ff1_vars, &r_ff1_vars, &self.gelu1, self.k_bits)?;
        let (q_ff2_vars, _) = enforce_linear(cs.clone(), &q_gelu_vars, &self.ff2, self.k_bits)?;
        let q_norm2_vars = enforce_rmsnorm(cs.clone(), &self.norm2, dim, self.frac_bits, self.k_bits)?;

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
    let n_len = norm1.q_ri.len() * norm1.q_ri[0].len();

    let ff2 = load_linear(&mut npz, "lin2")?; 
    let norm2 = load_norm(&mut npz, "rms2")?; 

    Ok(NystromMLPCircuit {
        norm1,
        ff1,
        gelu1,
        ff2,
        norm2,
        challenge_r: vec![E::ScalarField::zero(); n_len],
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
        
        // 1. 할당 순서 중요함
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

        // 3. 서킷 할당 순서와 일치하는 Witness 추출
        // 수동으로 q_out을 모으는 대신, CS가 생성한 할당 리스트에서 앞부분을 가져와야 일치함. 위치 선정 중요(TODO)
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
        let final_circuit = NystromMLPCircuit { challenge_r: r_vec, ..sample_circuit };
        let proof = create_random_proof_incl_cp_link(
            final_circuit.clone(), v1, v2, link_v, &params, &mut rng
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

        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        let dim = final_circuit.norm1.q_ri[0][0].len();
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
        )?;
        run_mlp_remainders::<Fr, G1>(&final_circuit, &mut rng)?;
    }
    Ok(())
}


#[test]
fn mlp_test() {
    let p = Path::new("outputs/outputs-model.transformer_0.norm_ff_combined.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            panic!("proof generate failed: {}", e);
        }
    }
}