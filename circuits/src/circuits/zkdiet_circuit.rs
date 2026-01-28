use std::{fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix0, Ix4, Array2, Ix2, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{marker::PhantomData, rand::{rngs::StdRng, SeedableRng, Rng}, UniformRand};
use ark_bn254::{Bn254, Fr};
use ark_ff::{PrimeField, One, Zero};
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
        matmul::constraints::{ MatrixVar, mat_vec_mul, new_matrix_witness},
        softmax::constraints::build_softmax_witnesses,
    }, util::{load::{read_array_u64},freivalds_load::load_freivalds_circuit, softmax_load::load_softmax_circuit}, 
};
use ark_std::fs;

/// 모든 서브 서킷을 하나로 합친 통합 서킷
#[derive(Clone)]
pub struct ZKDietCircuit<F: PrimeField> {
    // 1) Kernel Matmuls (Q*K_L, Q_L*K_L, Q_L*K)
    pub k1_matmul: FreivaldsCircuit<F>,
    pub k2_matmul: FreivaldsCircuit<F>,
    pub k3_matmul: FreivaldsCircuit<F>,
    
    // 2) Softmax (for K1, K2, K3)
    pub k1_softmax: SoftmaxCircuit<F>,
    pub k2_softmax: SoftmaxCircuit<F>,
    pub k3_softmax: SoftmaxCircuit<F>,

    // 3) Moore-Penrose Inverse Property
    // A: k2_softmax_output, P: inv_q AA+A=A+
    pub aaplus_matmul: FreivaldsCircuit<F>,
    pub aplusa_matmul: FreivaldsCircuit<F>,
    pub aaplusa_matmul: FreivaldsCircuit<F>,
    pub aplusaaplus_matmul: FreivaldsCircuit<F>,

    // 4) Final Output Reconstruction (Left_num * Right_num)
    pub temp_matmul: FreivaldsCircuit<F>,
    pub mid_matmul: FreivaldsCircuit<F>,
    pub final_matmul: FreivaldsCircuit<F>,
    
    pub r_challenge: F,
}

impl <F: PrimeField> ConstraintSynthesizer<F> for ZKDietCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // 모든 서브 서킷의 제약 조건을 순차적으로 등록
        self.k1_matmul.generate_constraints(cs.clone())?;
        self.k2_matmul.generate_constraints(cs.clone())?;
        self.k3_matmul.generate_constraints(cs.clone())?;
        
        self.k1_softmax.generate_constraints(cs.clone())?;
        self.k2_softmax.generate_constraints(cs.clone())?;
        self.k3_softmax.generate_constraints(cs.clone())?;

        self.aaplus_matmul.generate_constraints(cs.clone())?;
        self.aplusa_matmul.generate_constraints(cs.clone())?;
        self.aaplusa_matmul.generate_constraints(cs.clone())?;
        self.aplusaaplus_matmul.generate_constraints(cs.clone())?;

        self.temp_matmul.generate_constraints(cs.clone())?;
        self.mid_matmul.generate_constraints(cs.clone())?;
        self.final_matmul.generate_constraints(cs.clone())?;
        
        Ok(())
    }
}

// 소프트맥스 메타데이터 가져오기
fn load_softmax_metadata(reader: &mut NpzReader<File>) -> (i32, usize, usize, i64, i64, i32) {
    let frac_bits = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("softmax_tables_frac_bits.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(32);
    let limb_bits = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("softmax_limb_bits.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(4) as usize;
    let k_segments = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("softmax_k_segments.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(2) as usize;
    let scale_input = reader.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("softmax_scale_input.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or((1i64<<8) as i64);
    let multiplier = reader.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("softmax_multiplier.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(1i64);
    let base_scale_exp = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("softmax_base_scale_exp.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(16i32);
    (frac_bits, limb_bits, k_segments, scale_input, multiplier, base_scale_exp)
}

/// Load a single witness NPZ and build the circuit; requires softmax_tables either inside NPZ or provided `external_tables_path`.
pub fn load_single_circuit<E: Pairing>(path: &Path) -> Result<ZKDietCircuit<E::ScalarField>>
where
<E::ScalarField as FromStr>::Err: core::fmt::Debug,
{

    // Open witness npz
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let r_val = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("r_challenge.npy")
    .map(|a| a.into_scalar())
    .context("reading r_val")?;
    let r = E::ScalarField::from(r_val); 
    let mult_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("multiplier.npy").context("reading multiplier")?;
    let multiplier = E::ScalarField::from(mult_arr.iter().next().copied().unwrap_or(1u64));

    let k1_matmul = load_freivalds_circuit(&mut npz, "_k1", r, multiplier)?;
    let k2_matmul = load_freivalds_circuit(&mut npz, "_k2", r, multiplier)?;
    let k3_matmul = load_freivalds_circuit(&mut npz, "_k3", r, multiplier)?;

    // 소프트맥스 테이블 로드
    let tables_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("softmax_tables.npy").context("softmax_tables not found in integrated npz; ensure whole_pipeline embedded it")?;
    // 테이블값을 2차원으로 강제함
    let tables_2d = tables_arr.into_dimensionality::<Ix2>().context("softmax_tables must be 2D (K, table_width)")?;
    
    // 소프트맥스 메타데이터
    let (soft_frac_bits, soft_limb_bits, soft_k_segments, _, _, _) = load_softmax_metadata(&mut npz);
    // 테이블 또한 벡터로 변환
    let mut tables_field: Vec<Vec<E::ScalarField>> = Vec::with_capacity(tables_2d.shape()[0]);
    for k in 0..tables_2d.shape()[0] {
        let row = tables_2d.row(k);
        let mut v = Vec::with_capacity(row.len());
        for val in row.iter() {
            v.push(E::ScalarField::from(*val as u64));
        }
        tables_field.push(v);
    }

    // (추가) softmax lookup 테이블과 그 메타데이터를 .npz 파일로 임시 저장하는 코드 // todo: fix
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
    
    let (k1_softmax,k1_combined_matmul_ints, k1_limbs_bigints_flat, k1_cols) = load_softmax_circuit(&mut npz, "_k1", &tables_field, &tables_2d, soft_limb_bits as usize, soft_frac_bits as u32)?;

    
    let (k2_softmax, k2_combined_matmul_ints, k2_prod_bigints_flat, k2_cols) = load_softmax_circuit(&mut npz, "_k2", &tables_field, &tables_2d, soft_limb_bits as usize, soft_frac_bits as u32)?;
    let (k3_softmax, k3_combined_matmul_ints, k3_prod_bigints_flat , k3_cols)= load_softmax_circuit(&mut npz, "_k3", &tables_field, &tables_2d, soft_limb_bits as usize, soft_frac_bits as u32)?;
    
    // 최종곱
    let temp_matmul = load_freivalds_circuit(&mut npz, "_k3V", r, multiplier)?;
    let mid_matmul = load_freivalds_circuit(&mut npz, "_k1_inv_k2", r, multiplier)?;
    let final_matmul = load_freivalds_circuit(&mut npz, "_mt", r, multiplier)?;

    // 무어펜로즈 역행렬 속성 검증
    let aaplus_matmul = load_freivalds_circuit(&mut npz, "_A_AP", r, multiplier)?;
    let aplusa_matmul = load_freivalds_circuit(&mut npz, "_P_A", r, multiplier)?;
    let aaplusa_matmul = load_freivalds_circuit(&mut npz, "_AAplus_A", r, multiplier)?;
    let aplusaaplus_matmul = load_freivalds_circuit(&mut npz, "_PAm_P", r, multiplier)?;


    let zkdiet_circuit = ZKDietCircuit {
        k1_matmul,
        k2_matmul,
        k3_matmul,
        k1_softmax,
        k2_softmax,
        k3_softmax,
        aaplus_matmul,
        aplusa_matmul,
        aaplusa_matmul,
        aplusaaplus_matmul,
        temp_matmul,
        mid_matmul,
        final_matmul,
        r_challenge: r,
    };
    
    Ok(zkdiet_circuit)
}

pub fn run_paths<E: Pairing>(path: &Path) -> Result<()>
where
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

        // Per-file SNARK flow: setup, prove, verify
        let (pk, vk) = CcGroth16::<E>::setup(circuit.clone(), &mut rng)?;
        let pvk = prepare_verifying_key::<E>(&vk);
        let proof = CcGroth16::<E>::prove(&pk, circuit.clone(), &mut rng)?;

        // public inputs: include the Fiat-Shamir r_challenge
        // let public_inputs = vec![circuit.r_challenge]; // 프리발드 때
        let public_inputs = vec![]; // 소프트맥스 때
        let ok = CcGroth16::<E>::verify_with_processed_vk(&pvk, &public_inputs, &proof)?;
        println!("Verification result for {:?}: {}", file_path, ok);
        if !ok {
            return Err(anyhow::anyhow!("Verification failed for {:?}", file_path));
        }

    }

    Ok(())
}

#[test]
fn zkdiet_full_witness_test() {
    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_mode(ark_relations::r1cs::SynthesisMode::Setup); 

    let p = Path::new("attn_debug_outputs/attn_debug-model.transformer_0.mha.attn.witness.npz");
    if p.exists() {
        let res = run_paths::<Bn254>(p);
        if let Err(e) = res {
            println!("failed constraint: {:?}", cs.which_is_unsatisfied());
            panic!("proof generate failed: {}", e);
        }
    }
}