use std::{time::Instant, fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing, AffineRepr};
use ndarray::{ArrayD, IxDyn, Ix0, Ix1, Ix4, Array2, Ix2, Ix3, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{rc::Rc, cell::RefCell, fs, UniformRand, log2, marker::PhantomData, vec::Vec, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{emulated_fp::params, fp::FpVar, FieldVar}};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use merlin::Transcript;
use ark_r1cs_std::{select::CondSelectGadget, R1CSVar};
use crate::{
    cc_snark::{legogroth16::{ProvingKey, VerifyingKey, LinkPublicGenerators, create_random_proof, create_random_proof_incl_cp_link, generate_random_parameters_incl_cp_link, prepare_verifying_key, verify_proof, verify_proof_incl_cp_link}, 
    linker::{Linker, snark::LinkSnark, relation_generator::generate_cp_arithm_relation,relation_generator::{generate_cp_factored_instance, generate_cp_factored_relation, generate_cp_factored_witness}, snark::PP}},
    config::{GELU_C, GELU_M, KBITS_C, KBITS_M, RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M}, 
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu}, layernorm::constraints::{NormBundle, enforce_rmsnorm}, linear::constraints::{LinearBundle, enforce_linear}
    }, lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        }, poly::{commitments::{Commitments, MultiCommitGens}, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}}, subtables::{
            SubtableStrategy, and::AndSubtableStrategy, gelu::GeLUSubtableStrategy, layernorm::RMSNormSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
        }, utils::{math::Math, random::RandomTape}
    }, util::{
        load::{to_field_1d_i64, to_field_2d_i32, to_field_3d_i32, to_field_3d_i64, to_field_3d_u64}, lookup_mlp::{run_lasso_for_gelu, run_lasso_for_rmsnorm, run_mlp_remainders}, utils::{cm1_from_vk, hash_cm_to_field_vec}
    }, 
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
        frac_bits: frac_bits,
        k_bits: k_bits,
        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })

    
}

pub fn prove_and_verify_factored_multi_link<E: Pairing>(
    vk: &VerifyingKey<E>,
    num_pub: usize,
    v_groth: E::ScalarField,
    cm_groth: E::G1Affine,
    lasso_results: Vec<(PolyCommitment<E::G1>, PolyCommitmentBlinds<E::ScalarField>, MultiCommitGens<E::G1>)>,
    u_total: Vec<E::ScalarField>,
    rng: &mut impl Rng,
) -> Result<()> {
    let mut l_rows_vec = Vec::new();
    let mut r_cols_vec = Vec::new();
    let mut cks_lasso_vec = Vec::new();
    let mut rs = vec![v_groth];
    let mut lasso_cms_all = Vec::new();

    for (cm, blinds, gens) in &lasso_results {
        l_rows_vec.push(cm.C.len()); 
        r_cols_vec.push(gens.n);    
        
        let mut ck = vec![gens.h.into_affine()];
        ck.extend(gens.G.iter().map(|g| g.into_affine()));
        cks_lasso_vec.push(ck);
        
        rs.extend(&blinds.blinds); 
        lasso_cms_all.extend(cm.C.iter().map(|c| c.into_affine()));
    }

    let mut ck_joint = vec![vk.eta1_gamma_inv_g1];
    let start = 1 + num_pub;
    ck_joint.extend(vk.gamma_abc_g1[start..start + u_total.len()].iter().cloned());

    let crs = generate_cp_factored_relation::<E>(
        lasso_results.len(), l_rows_vec, r_cols_vec, ck_joint, cks_lasso_vec
    );

    let pp = PP::new(crs.nr, crs.nc, &E::G1::rand(rng).into_affine(), &E::G2::rand(rng).into_affine());
    let (ek, vk_link) = LinkSnark::<E>::keygen(&pp, crs, rng);

    let witness = generate_cp_factored_witness::<E>(rs, u_total);
    let instance = generate_cp_factored_instance::<E>(cm_groth, lasso_cms_all);

    let (proof, _) = LinkSnark::<E>::prove(&pp, &ek, witness, rng);
    if LinkSnark::<E>::verify(&pp, &vk_link, &instance, &proof) {
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Factored link verification failed"))
    }
}

pub fn run_paths<E: Pairing>(path: &Path) -> Result<()>
where
E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>, // 이 제약을 추가
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

        // Witness 값 추출
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

        println!("\n[Debug] LegoGroth16 committed_group1 (Total: {})", committed_group1.len());
        let print_count = std::cmp::min(10, committed_group1.len());
        if committed_group1.len() > 0 {
            println!("  Head (10): {:?}", committed_group1[..print_count].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
            println!("  Tail (10): {:?}", committed_group1[committed_group1.len()-print_count..].iter().map(|f| f.into_bigint()).collect::<Vec<_>>());
        }

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let link_v = Fr::rand(&mut rng);
        
        // 서킷의 공개 입력 개수
        let num_pub = cs_witness.num_instance_variables() - 1; 

        let cm1 = cm1_from_vk::<E>(&params.vk.groth16_vk, num_pub, &committed_group1, &v1);

        println!("=== [1] LegoGroth16 GenProof ===");

        let groth16_pk = ProvingKey {
            vk: params.vk.groth16_vk.clone(),
            common: params.common.clone(),
        };

        let proof = create_random_proof::<E, _, _>(
            sample_circuit.clone(), 
            v1, 
            v2, 
            &groth16_pk, 
            &mut rng
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;

        let public_inputs = vec![]; // 공개 입력이 없는 경우
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);

        println!("  - LegoGroth16 SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::Error::msg(format!("{:?}", e)));
        }

        println!("=== [2] Lasso Lookup Proofs: Softmax Sections ===");
        let dim = sample_circuit.norm1.q_ri[0][0].len();
        let (lasso_cm1, lasso_blinds1, lasso_gens1) = run_lasso_for_rmsnorm(&sample_circuit.norm1, sample_circuit.k_bits, b"lasso_rms1", &mut rng)?;
        let (lasso_cm2, lasso_blinds2, lasso_gens2) = run_lasso_for_gelu(
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
        let (lasso_cm3, lasso_blinds3, lasso_gens3) = run_lasso_for_rmsnorm(&sample_circuit.norm2, sample_circuit.k_bits, b"lasso_rms2", &mut rng)?;
        let res_remainders = run_mlp_remainders::<Fr, G1>(&sample_circuit, &mut rng)?;
        let u_values: Vec<Fr> = sample_circuit.committed_vars.borrow().iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();


        let mut lasso_results = vec![
        (lasso_cm1, lasso_blinds1, lasso_gens1), // res_rms1
        (lasso_cm2, lasso_blinds2, lasso_gens2), // res_gelu
        (lasso_cm3, lasso_blinds3, lasso_gens3), // res_rms2
        ];
        lasso_results.extend(res_remainders);

        prove_and_verify_factored_multi_link::<E>(
            &params.vk.groth16_vk,
            num_pub,
            v1,
            cm1,
            lasso_results,
            u_values,
            &mut rng,
        )?;
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