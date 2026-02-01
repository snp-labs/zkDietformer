use std::{fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix4, Array2, Ix2, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{UniformRand, log2, marker::PhantomData, rand::{Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use crate::{
    cc_snark::cc_gro::{CcGroth16, prepare_verifying_key},
    circuits::{
        freivalds_circuit::FreivaldsCircuit, 
        softmax_circuit::{K_SEGMENTS, SoftmaxCircuit}, 
    },
    gadgets::{
        matmul::constraints::{new_matrix_witness, enforce_quantized_freivalds, MatmulBundle},
        softmax::constraints::{SoftmaxBundle, enforce_softmax}
    
    }, util::{load::{read_array_u64},freivalds_load::load_freivalds_circuit, softmax_load::{load_softmax_circuit, load_softmax_metadata}}, 
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
// Lasso에 필요한 타입 정의: Bn254::Fr에 맞춰 Bn254::G1Projective 사용
type G1 = G1Projective;

/// 단순화된 통합 서킷 구조체
#[derive(Clone)]
pub struct CombinedCircuit<F: PrimeField> {
    // 1) Kernel Matmuls
    pub k1_data: MatmulBundle<F>,
    pub k2_data: MatmulBundle<F>,
    pub k3_data: MatmulBundle<F>,
    
    // 2) Moore-Penrose Inverse Property
    pub aaplus_data: MatmulBundle<F>,
    pub aplusa_data: MatmulBundle<F>,
    pub aaplusa_data: MatmulBundle<F>,
    pub aplusaaplus_data: MatmulBundle<F>,

    // 3) Final Output Reconstruction
    pub temp_data: MatmulBundle<F>,
    pub mid_data: MatmulBundle<F>,
    pub final_data: MatmulBundle<F>,

    // 4) Softmax Sub-circuits
    pub k1_softmax: SoftmaxBundle<F>,
    pub k2_softmax: SoftmaxBundle<F>,
    pub k3_softmax: SoftmaxBundle<F>,

    // 공통 챌린지 및 배율
    pub r_challenge: F,
    pub multiplier: F,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for CombinedCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // 공통 변수 할당
        let r_var = FpVar::new_input(ark_relations::ns!(cs, "r_challenge"), || Ok(self.r_challenge))?;
        let m_var = FpVar::new_witness(ark_relations::ns!(cs, "multiplier"), || Ok(self.multiplier))?;

        // 검증할 모든 행렬 묶음 리스트 생성
        let bundles = vec![
            &self.k1_data, &self.k2_data, &self.k3_data,
            &self.aaplus_data, &self.aplusa_data, &self.aaplusa_data, &self.aplusaaplus_data,
            &self.temp_data, &self.mid_data, &self.final_data,
        ];

        // 각 번들에 대해 루프를 돌며 제약 조건 생성
        for bundle in bundles {
            let batch_size = bundle.a.len();
            if batch_size == 0 { continue; }
            let head_size = bundle.a[0].len();

            // 번들별 개별 양자화 상수를 FpVar로 변환 (상수로 처리)
            let a0_var = FpVar::Constant(bundle.input_0);
            let c0_var = FpVar::Constant(bundle.c_0_converted);

            for n in 0..batch_size {
                for h in 0..head_size {
                    // 증인 할당 (MatrixVar 생성)
                    let a_v = new_matrix_witness(cs.clone(), &bundle.a[n][h]);
                    let b_v = new_matrix_witness(cs.clone(), &bundle.b[n][h]);
                    let c_v = new_matrix_witness(cs.clone(), &bundle.c[n][h]);
                    let d_v = new_matrix_witness(cs.clone(), &bundle.div[n][h]);
                    let rem_v = new_matrix_witness(cs.clone(), &bundle.rem[n][h]);

                    // 가젯 함수를 사용하여 프리발드 제약 조건 강제
                    enforce_quantized_freivalds(
                        &a_v, &b_v, &c_v, &d_v, &rem_v,
                        &a0_var, &c0_var, &r_var, &m_var
                    )?;
                }
            }
        }

        // 3. 모든 Softmax 번들에 대해 가젯 실행
        enforce_softmax(cs.clone(), &self.k1_softmax)?;
        enforce_softmax(cs.clone(), &self.k2_softmax)?;
        enforce_softmax(cs.clone(), &self.k3_softmax)?;

        Ok(())
    }
}

pub fn load_single_circuit<E: Pairing>(path: &Path) -> Result<CombinedCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    // 1. NPZ 파일 열기
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    // 2. 공통 챌린지 값 로드
    let r_val = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("r_challenge.npy")
        .map(|a| a.into_scalar())
        .context("reading r_val")?;
    let r = E::ScalarField::from(r_val); 

    let mult_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("multiplier.npy")
        .context("reading multiplier")?;
    let multiplier = E::ScalarField::from(mult_arr.iter().next().copied().unwrap_or(1u64));

    // 3. Matmul Bundles 로드
    let load_matmul_bundle = |npz: &mut NpzReader<File>, suffix: &str, r: E::ScalarField, m: E::ScalarField| -> Result<MatmulBundle<E::ScalarField>> {
        let circuit = load_freivalds_circuit(npz, suffix, r, m)?;
        Ok(MatmulBundle {
            a: circuit.a,
            b: circuit.b,
            c: circuit.c,
            div: circuit.div,
            rem: circuit.remainder,
            input_0: circuit.input_0,
            c_0_converted: circuit.c_0_converted,
        })
    };

    let k1_data = load_matmul_bundle(&mut npz, "_k1", r, multiplier)?;
    let k2_data = load_matmul_bundle(&mut npz, "_k2", r, multiplier)?;
    let k3_data = load_matmul_bundle(&mut npz, "_k3", r, multiplier)?;

    let aaplus_data = load_matmul_bundle(&mut npz, "_A_AP", r, multiplier)?;
    let aplusa_data = load_matmul_bundle(&mut npz, "_P_A", r, multiplier)?;
    let aaplusa_data = load_matmul_bundle(&mut npz, "_AAplus_A", r, multiplier)?;
    let aplusaaplus_data = load_matmul_bundle(&mut npz, "_PAm_P", r, multiplier)?;

    let temp_data = load_matmul_bundle(&mut npz, "_k3V", r, multiplier)?;
    let mid_data = load_matmul_bundle(&mut npz, "_k1_inv_k2", r, multiplier)?;
    let final_data = load_matmul_bundle(&mut npz, "_mt", r, multiplier)?;

    // 4. Softmax Tables 로드 및 메타데이터 처리
    let tables_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::Ix2>("softmax_tables.npy")?;
    let (soft_frac_bits, soft_limb_bits, _, _, _, _) = load_softmax_metadata(&mut npz);
    
    let mut tables_field: Vec<Vec<E::ScalarField>> = Vec::new();
    for row in tables_arr.rows() {
        tables_field.push(row.iter().map(|&v| E::ScalarField::from(v)).collect());
    }

    // 5. Softmax Bundles 로드 (softmax_load.rs의 load_softmax_circuit 활용)
    let load_soft_bundle = |npz: &mut NpzReader<File>, suffix: &str| -> Result<SoftmaxBundle<E::ScalarField>> {
        let (circuit, _, _, _) = load_softmax_circuit(
            npz, suffix, &tables_field, &tables_arr, soft_limb_bits, soft_frac_bits
        )?;
        Ok(SoftmaxBundle {
            matmul_outputs: circuit.matmul_outputs,
            limbs: circuit.limbs,
            exp_limb_vals: circuit.exp_limb_vals,
            softmax_outputs: circuit.softmax_outputs,
        })
    };

    let k1_softmax = load_soft_bundle(&mut npz, "_k1")?;
    let k2_softmax = load_soft_bundle(&mut npz, "_k2")?;
    let k3_softmax = load_soft_bundle(&mut npz, "_k3")?;

    // 소프트맥스 메타데이터
    // 테이블값을 2차원으로 강제함
    let tables_2d = tables_arr.into_dimensionality::<Ix2>().context("softmax_tables must be 2D (K, table_width)")?;
    let lookup_npz_path = "tmp_tables/embedded_softmax_for_lookup.npz";
    {
        let _ = std::fs::create_dir_all("tmp_tables");
        let f = File::create(lookup_npz_path).context("creating temporary lookup npz file")?;
        let mut writer = NpzWriter::new(f);
        writer.add_array("tables.npy", &tables_2d).context("writing tables to temp npz")?;
        let frac = arr0(soft_frac_bits as i32);
        let limb = arr0(soft_limb_bits as i32);
        writer.add_array("frac_bits.npy", &frac).context("writing frac_bits")?;
        writer.add_array("limb_bits.npy", &limb).context("writing limb_bits")?;
        writer.finish().context("finalizing temp npz write")?;
    }

    // 6. 최종 통합 서킷 반환
    Ok(CombinedCircuit {
        k1_data,
        k2_data,
        k3_data,
        aaplus_data,
        aplusa_data,
        aaplusa_data,
        aplusaaplus_data,
        temp_data,
        mid_data,
        final_data,
        k1_softmax,
        k2_softmax,
        k3_softmax,
        r_challenge: r,
        multiplier,
    })
}

// [주의] 내가 만든 서브테이블 SPEC과 동일하게 맞춰 주어야 함.
const SOFTMAX_C: usize = 2;   // number of segments (C)
const SOFTMAX_M: usize = 256; // table width (M) - must be a power of two/ limb bits = 8

fn prove_lasso_for_softmax(
    bundle: &SoftmaxBundle<Fr>,
    gens: &SparsePolyCommitmentGens<G1>,
    rng: &mut StdRng,
    label: &[u8],
) -> Result<()> 
{
    use ark_ff::{BigInteger, PrimeField};

    // Build nz_indices
    let mut nz_indices: Vec<[usize; SOFTMAX_C]> = Vec::new();
    for batch in &bundle.limbs {
        for head in batch {
            let idx_arr: [usize; SOFTMAX_C] = std::array::from_fn(|seg| {
                let val = head[seg][0][0];
                let bytes = val.into_bigint().to_bytes_le();
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[0..8]);
                u64::from_le_bytes(b) as usize
            });
            nz_indices.push(idx_arr);
        }
    }

    let sparsity = nz_indices.len();
    if sparsity == 0 { return Ok(()); } 

    assert!(SOFTMAX_M.is_power_of_two(), "SOFTMAX_M must be a power of two");
    let log_M = SOFTMAX_M.trailing_zeros() as usize;

    let lookup_matrix = SparseLookupMatrix::<SOFTMAX_C>::new(nz_indices, log_M);
    let mut densified = DensifiedRepresentation::<Fr, SOFTMAX_C>::from(&lookup_matrix);


    let commitment = densified.commit::<G1>(gens);


    let log_s = (sparsity.next_power_of_two().trailing_zeros()) as usize;
    let r_point: Vec<Fr> = (0..log_s).map(|_| Fr::rand(rng)).collect();

    let mut random_tape = RandomTape::new(b"lasso_tape");
    let mut transcript = Transcript::new(b"lasso_softmax");

    let proof_lasso = SparsePolynomialEvaluationProof::<G1, SOFTMAX_C, SOFTMAX_M, SoftmaxSubtableStrategy>::prove(
        &mut densified, &r_point, gens, &mut transcript, &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"lasso_softmax");
    proof_lasso.verify(&commitment, &r_point, gens, &mut verifier_transcript)
        .map_err(|_| anyhow::anyhow!("Lasso verification failed for label: {:?}", std::str::from_utf8(label)))?;

    println!("Lasso proof verified for: {:?}", std::str::from_utf8(label).unwrap());
    Ok(())
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
        let circuit= load_single_circuit::<E>(file_path)?;

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

        println!("=== [1] ccGroth Proofs: Freivald/Softmax ===");    
        // Per-file SNARK flow: setup, prove, verify
        let (pk, vk) = CcGroth16::<E>::setup(circuit.clone(), &mut rng)?;
        let pvk = prepare_verifying_key::<E>(&vk);
        let proof = CcGroth16::<E>::prove(&pk, circuit.clone(), &mut rng)?;

        // public inputs: include the Fiat-Shamir r_challenge
        let public_inputs = vec![circuit.r_challenge];
        let ok = CcGroth16::<E>::verify_with_processed_vk(&pvk, &public_inputs, &proof)?;
        println!("Verification result for {:?}: {}", file_path, ok);
        if !ok {
            return Err(anyhow::anyhow!("Verification failed for {:?}", file_path));
        }

        println!("=== [2] Lasso Lookup Proofs: Softmax K1, K2, K3 ===");    
        
        let bundles = [
            (&circuit.k1_softmax, b"lasso_k1"),
            (&circuit.k2_softmax, b"lasso_k2"),
            (&circuit.k3_softmax, b"lasso_k3"),
        ];

        for (bundle, label) in bundles {
            // non-zero값 세기
            let sparsity = bundle.limbs.iter().map(|b| b.len()).sum::<usize>();
            assert!(SOFTMAX_M.is_power_of_two(), "SOFTMAX_M must be a power of two");
            let log_M = SOFTMAX_M.trailing_zeros() as usize;
            let gens = SparsePolyCommitmentGens::<G1>::new(b"gens_lasso", SOFTMAX_C, sparsity, SOFTMAX_C, log_M);
            prove_lasso_for_softmax(bundle, &gens, &mut rng, label)?;
        }

    }

    Ok(())
}

#[test]
fn combined_full_witness_test() {
    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_mode(ark_relations::r1cs::SynthesisMode::Setup); 

    let p = Path::new("attn_debug_outputs/attn_debug-model.transformer_0.mha.attn.witness.npz");
    if p.exists() {
        println!("Running combined circuit test on {:?}", p);
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            println!("failed constraint: {:?}", cs.which_is_unsatisfied());
            panic!("proof generate failed: {}", e);
        }
    }
}