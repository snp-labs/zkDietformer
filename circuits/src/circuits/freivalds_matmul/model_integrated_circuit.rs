//! Version A: Model-wide Integrated Circuit
//!
//! All N transformer layers (Attention + MLP) are proven in a single Groth16 proof.
//! Supports both Nystrom/BERT attention and RMSNorm/LayerNorm MLP variants.
//!
//! **Constraint**: Existing files in `freivalds_matmul/` are NOT modified.
//! This file copies the `generate_constraints` logic from the existing circuits
//! into standalone helper functions.

use std::{cell::RefCell, fs::File, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::pairing::Pairing;
use ndarray::{Ix0, Ix4, Ix2, IxDyn};
use ndarray_npy::NpzReader;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{SeedableRng, rngs::StdRng},
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero, BigInteger};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{
    ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError,
};
use ark_r1cs_std::{
    alloc::AllocVar,
    fields::{fp::FpVar, FieldVar},
    select::CondSelectGadget,
    R1CSVar,
};
use crate::{
    cc_snark::legogroth16::{
        create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    },
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics,
            LassoMetricsAggregate, factored_link_setup, factored_link_keygen,
            factored_link_prove, factored_link_verify, pad_lasso_u_values,
            groth16_commitment_size,
        },
        proof_size,
    },
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu},
        layernorm::constraints::{NormBundle, LayerNormBundle, enforce_rmsnorm, enforce_layernorm},
        linear::constraints::{LinearBundle, enforce_linear},
        matmul::{
            constraints::new_matrix_witness,
            quantized::{
                MatmulBundle, PseudoinverseBundle, ZenParams,
                compute_circuit_landmarks, enforce_freivalds,
                gen_challenge_vector, new_matrix_witness_4d, transpose_matrix,
            },
        },
        softmax::constraints::{SoftmaxBundle, enforce_softmax},
    },
    lasso::poly::{
        commitments::MultiCommitGens,
        dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds},
    },
    util::{
        load::{
            read_array_i64, to_field_1d_i64, to_field_2d_i32, to_field_2d_i64,
            to_field_3d_i32, to_field_3d_i64, to_field_3d_u64,
            to_field_4d_i64, to_field_4d_u64, to_field_5d_i64, to_field_5d_u64,
        },
        norm_detect::{NormType, detect_norm_type},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;
use crate::lasso::subtables::layernorm::set_layernorm_table_dir;
use crate::lasso::subtables::gelu::set_gelu_table_dir;

type G1 = G1Projective;
pub type LassoResult = (PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize);

// ═══════════════════════════════════════════════════════════════════════════════
// Common Types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionVariant {
    Nystrom,
    Bert,
}

/// Pure witness data for Nystrom attention (no circuit scaffolding).
#[derive(Clone)]
pub struct NystromAttnWitness<F: PrimeField> {
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
    pub seg_size: usize,
    pub mask: Vec<Vec<F>>,
    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub k_bits: usize,
}

/// Pure witness data for BERT attention (no circuit scaffolding).
#[derive(Clone)]
pub struct BertAttnWitness<F: PrimeField> {
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,
    pub matmul_attn: MatmulBundle<F>,
    pub sm: SoftmaxBundle<F>,
    pub matmul_x: MatmulBundle<F>,
    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,
}

#[derive(Clone)]
pub enum AttnWitness<F: PrimeField> {
    Nystrom(NystromAttnWitness<F>),
    Bert(BertAttnWitness<F>),
}

/// MLP witness variant.
#[derive(Clone)]
pub enum MlpWitness<F: PrimeField> {
    LayerNorm(NystromMLPLayerNormData<F>),
    RMSNorm(NystromMLPRMSNormData<F>),
}

/// Pure RMSNorm MLP data (mirrors NystromMLPCircuit fields minus scaffolding).
#[derive(Clone)]
pub struct NystromMLPRMSNormData<F: PrimeField> {
    pub norm1: NormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,
    pub ff2: LinearBundle<F>,
    pub norm2: NormBundle<F>,
    pub frac_bits: usize,
    pub k_bits: usize,
}

/// Pure LayerNorm MLP data (re-export from nystrom_mlp_circuit).
pub use crate::circuits::freivalds_matmul::nystrom_mlp_circuit::NystromMLPLayerNormData;

/// Discovered witness file pairs for all layers.
pub struct DiscoveredWitnesses {
    pub attn_files: Vec<std::path::PathBuf>,
    pub mlp_files: Vec<std::path::PathBuf>,
    pub attn_variant: AttentionVariant,
    pub norm_variant: NormType,
    pub num_layers: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Witness Discovery & Loading
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect attention variant from an NPZ file.
/// `seg_size.npy` present → Nystrom; absent → BERT.
pub fn detect_attention_variant(path: &Path) -> Result<AttentionVariant> {
    let f = File::open(path)?;
    let mut npz = NpzReader::new(f)?;
    match npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("seg_size.npy") {
        Ok(_) => Ok(AttentionVariant::Nystrom),
        Err(_) => Ok(AttentionVariant::Bert),
    }
}

/// Discover and pair attention/MLP witness files in a directory.
pub fn discover_witnesses(dir: &Path) -> Result<DiscoveredWitnesses> {
    let mut attn_files = Vec::new();
    let mut mlp_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with("attention.self.witness.npz") {
            attn_files.push(path);
        } else if name.ends_with(".norm.witness.npz") {
            mlp_files.push(path);
        }
    }

    attn_files.sort();
    mlp_files.sort();

    if attn_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No '*attention.self.witness.npz' files found in {:?}", dir
        ));
    }
    if mlp_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No '*.norm.witness.npz' files found in {:?}", dir
        ));
    }
    if attn_files.len() != mlp_files.len() {
        return Err(anyhow::anyhow!(
            "Attention/MLP file count mismatch: {} attn vs {} mlp",
            attn_files.len(), mlp_files.len()
        ));
    }

    let attn_variant = detect_attention_variant(&attn_files[0])?;
    let norm_variant = detect_norm_type(&mlp_files[0])?;
    let num_layers = attn_files.len();

    println!(
        "[discover] dir={:?} | layers={} | attn={:?} | norm={:?}",
        dir, num_layers, attn_variant, norm_variant
    );

    Ok(DiscoveredWitnesses {
        attn_files,
        mlp_files,
        attn_variant,
        norm_variant,
        num_layers,
    })
}

/// Load a Nystrom attention witness from NPZ.
pub fn load_nystrom_attn_witness<E: Pairing>(path: &Path) -> Result<NystromAttnWitness<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let q_q = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?.into_dimensionality()?;
    let q_k = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?.into_dimensionality()?;
    let z_q = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar();
    let z_k = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar();
    let seg = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("seg_size.npy")?.into_scalar() as usize;

    let load_matmul = |npz: &mut NpzReader<File>, suffix: &str, has_rmean: bool| -> Result<MatmulBundle<E::ScalarField>> {
        Ok(MatmulBundle {
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
            m_sm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar()),
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

    let q_inv_k2 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_inv_k2.npy")?.into_dimensionality()?;
    let z_inv_k2 = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_inv_k2.npy")?.into_scalar();
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
    let matmul_mid = load_matmul(&mut npz, "mid", false)?;
    let matmul_x = load_matmul(&mut npz, "x", false)?;

    Ok(NystromAttnWitness {
        q_q: to_field_4d_i64(&q_q),
        q_k: to_field_4d_i64(&q_k),
        q_v: to_field_4d_i64(&q_v),
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),
        z_v: E::ScalarField::from(z_v),
        matmul1, matmul2, matmul3,
        sm1, sm2, sm3,
        pseudo_inv,
        matmul_temp, matmul_mid, matmul_x,
        frac_bits, limb_bits, k_segments, k_bits,
        seg_size: seg,
        mask: to_field_2d_i64(&mask_raw),
    })
}

/// Load a BERT attention witness from NPZ.
pub fn load_bert_attn_witness<E: Pairing>(path: &Path) -> Result<BertAttnWitness<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening BERT witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let q_q_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?.into_dimensionality()?;
    let q_k_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?.into_dimensionality()?;
    let q_v_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_v.npy")?.into_dimensionality()?;
    let z_q = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar());
    let z_k = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar());
    let z_v = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar());

    let q_sim_attn = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_attn.npy")?.into_dimensionality()?;
    let r_matmul_attn = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_attn.npy")?.into_dimensionality()?;
    let matmul_attn = MatmulBundle {
        r_mean: vec![],
        q_sim: to_field_4d_i64(&q_sim_attn),
        r_matmul: to_field_4d_i64(&r_matmul_attn),
        m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_attn.npy")?.into_scalar()),
        m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_attn.npy")?.into_scalar()),
        z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_attn.npy")?.into_scalar()),
    };

    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let limb_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("limb_bits.npy")?.into_scalar() as usize;
    let k_segments = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_segments.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;

    let limbs_raw = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("limbs.npy")?;
    let exp_limbs_raw = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("exp_limbs.npy")?;
    let q_exp_raw = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("Q_exp.npy")?;
    let r_exp_raw = npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("R_exp.npy")?;
    let exp_total_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("exp_total.npy")?.into_dimensionality()?;
    let p_attn_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("P_attn.npy")?.into_dimensionality()?;
    let r_div_attn_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("R_div_attn.npy")?.into_dimensionality()?;
    let r_sm_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("r_sm.npy")?.into_dimensionality()?;

    let sm = SoftmaxBundle {
        limbs: to_field_5d_i64(&limbs_raw),
        exp_limbs: to_field_5d_u64(&exp_limbs_raw),
        q_exp: to_field_5d_u64(&q_exp_raw),
        r_exp: to_field_5d_u64(&r_exp_raw),
        exp_total: to_field_4d_u64(&exp_total_raw),
        p: to_field_4d_u64(&p_attn_raw),
        r_div: to_field_4d_u64(&r_div_attn_raw),
        r_sm: to_field_4d_i64(&r_sm_raw),
        m_sm: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar()),
    };

    let q_sim_x = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_x.npy")?.into_dimensionality()?;
    let r_matmul_x = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_x.npy")?.into_dimensionality()?;
    let matmul_x = MatmulBundle {
        r_mean: vec![],
        q_sim: to_field_4d_i64(&q_sim_x),
        r_matmul: to_field_4d_i64(&r_matmul_x),
        m: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_x.npy")?.into_scalar()),
        m_prime: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_x.npy")?.into_scalar()),
        z_sim: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_x.npy")?.into_scalar()),
    };

    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    Ok(BertAttnWitness {
        q_q: to_field_4d_i64(&q_q_raw),
        q_k: to_field_4d_i64(&q_k_raw),
        q_v: to_field_4d_i64(&q_v_raw),
        z_q, z_k, z_v,
        matmul_attn, sm, matmul_x,
        frac_bits, limb_bits, k_segments, k_bits,
        mask: to_field_2d_i64(&mask_raw),
    })
}

/// Load RMSNorm MLP data from NPZ.
pub fn load_rmsnorm_mlp_data<E: Pairing>(path: &Path) -> Result<NystromMLPRMSNormData<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    use ndarray::{Ix1, Ix2, Ix3};
    let f = File::open(path).context(format!("opening witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;
    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;

    let load_norm = |npz: &mut NpzReader<File>, prefix: &str| -> Result<NormBundle<E::ScalarField>> {
        Ok(NormBundle {
            q_ri: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_ri_{}.npy", prefix))?.into_dimensionality()?),
            z_ri: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_ri_{}.npy", prefix))?.into_scalar()),
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
            m_gelu: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?.into_scalar()),
            m_prime_gelu: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?.into_scalar()),
            q_out_sim: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?.into_dimensionality()?),
            r_out_gelu: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?.into_dimensionality()?),
            m_gelu_idx: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_gelu_idx.npy")?.into_scalar()),
        })
    };

    let gelu1 = load_gelu(&mut npz, "gelu")?;
    let ff2 = load_linear(&mut npz, "lin2")?;
    let norm2 = load_norm(&mut npz, "rms2")?;

    Ok(NystromMLPRMSNormData { norm1, ff1, gelu1, ff2, norm2, frac_bits, k_bits })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constraint Generation Helpers (copied from existing circuits)
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate Nystrom attention constraints.
/// Copied from NystromAttentionCircuit::generate_constraints.
pub fn generate_nystrom_attn_constraints<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    w: &NystromAttnWitness<F>,
    r_var: &FpVar<F>,
    two_k_var: &FpVar<F>,
    committed_vars: &Rc<RefCell<Vec<FpVar<F>>>>,
    committed_vars_2: &Rc<RefCell<Vec<FpVar<F>>>>,
) -> Result<(), SynthesisError> {
    let mut r2_pool = Vec::new();
    let mut r3_pool = Vec::new();

    let q_proc = new_matrix_witness_4d(cs.clone(), &w.q_q);
    let k_proc = new_matrix_witness_4d(cs.clone(), &w.q_k);
    let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &w.matmul1.r_mean);
    let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &w.matmul2.r_mean);

    let (k1_data, params_k1) = w.matmul1.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);
    let (k2_data, params_k2) = w.matmul2.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);
    let (k3_data, params_k3) = w.matmul3.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);

    let mask_var = new_matrix_witness(cs.clone(), &w.mask);
    let sm1_vars = w.sm1.alloc_vars(cs.clone())?;
    let sm2_vars = w.sm2.alloc_vars(cs.clone())?;
    let sm3_vars = w.sm3.alloc_vars(cs.clone())?;

    let v_proc = new_matrix_witness_4d(cs.clone(), &w.q_v);

    // d2: Q, K, V
    {
        let mut vars_2 = committed_vars_2.borrow_mut();
        for batch in &q_proc { for head in batch { for seq in head { vars_2.extend(seq.iter().cloned()); } } }
        for batch in &k_proc { for head in batch { for seq in head { vars_2.extend(seq.iter().cloned()); } } }
        for batch in &v_proc { for head in batch { for seq in head { vars_2.extend(seq.iter().cloned()); } } }
    }

    let (temp_data, params_temp) = w.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), w.z_v, two_k_var);
    let (mid_data, params_mid) = w.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), w.pseudo_inv.z_inv, two_k_var);
    let (x_data, params_x) = w.matmul_x.alloc_and_setup(cs.clone(), w.matmul_mid.z_sim, w.matmul_temp.z_sim, two_k_var);

    let mut global_k1_lows = Vec::new(); let mut global_k1_highs = Vec::new();
    let mut global_k2_lows = Vec::new(); let mut global_k2_highs = Vec::new();
    let mut global_k3_lows = Vec::new(); let mut global_k3_highs = Vec::new();
    let mut global_sm1_idx = vec![Vec::new(); 5];
    let mut global_sm1_r_sm = vec![Vec::new(); 2];
    let mut global_sm2_idx = vec![Vec::new(); 5];
    let mut global_sm2_r_sm = vec![Vec::new(); 2];
    let mut global_sm3_idx = vec![Vec::new(); 5];
    let mut global_sm3_r_sm = vec![Vec::new(); 2];
    let mut global_inv0_lows = Vec::new(); let mut global_inv0_highs = Vec::new();
    let mut global_inv1_lows = Vec::new(); let mut global_inv1_highs = Vec::new();
    let mut global_inv2_lows = Vec::new(); let mut global_inv2_highs = Vec::new();
    let mut global_inv3_lows = Vec::new(); let mut global_inv3_highs = Vec::new();
    let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
    let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
    let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();

    for b in 0..q_proc.len() {
        let k_landmarks = compute_circuit_landmarks(cs.clone(), &k_proc[b], &r_mean_k1[b], w.seg_size)?;
        let q_landmarks = compute_circuit_landmarks(cs.clone(), &q_proc[b], &r_mean_k2[b], w.seg_size)?;

        for h in 0..q_proc[b].len() {
            let k_l_t = transpose_matrix(&k_landmarks[h]);
            let r_pows_k1 = gen_challenge_vector(r_var, k1_data.q_sim[b][h][0].len());

            // Kernel 1
            let mut h_k1_lows = Vec::new(); let mut h_k1_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_proc[b][h], &k_l_t, &k1_data.q_sim[b][h], &k1_data.r_matmul[b][h], &w.matmul1.r_matmul[b][h], &params_k1, &r_pows_k1, &mut h_k1_lows, &mut h_k1_highs)?;
            global_k1_lows.extend(h_k1_lows); global_k1_highs.extend(h_k1_highs);

            // Kernel 2
            let mut h_k2_lows = Vec::new(); let mut h_k2_highs = Vec::new();
            let q_l_h = &q_landmarks[h];
            let r_pows_k2 = gen_challenge_vector(r_var, k2_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), q_l_h, &k_l_t, &k2_data.q_sim[b][h], &k2_data.r_matmul[b][h], &w.matmul2.r_matmul[b][h], &params_k2, &r_pows_k2, &mut h_k2_lows, &mut h_k2_highs)?;
            global_k2_lows.extend(h_k2_lows); global_k2_highs.extend(h_k2_highs);

            // Kernel 3
            let mut h_k3_lows = Vec::new(); let mut h_k3_highs = Vec::new();
            let k_full_t = transpose_matrix(&k_proc[b][h]);
            let r_pows_k3 = gen_challenge_vector(r_var, k3_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &q_landmarks[h], &k_full_t, &k3_data.q_sim[b][h], &k3_data.r_matmul[b][h], &w.matmul3.r_matmul[b][h], &params_k3, &r_pows_k3, &mut h_k3_lows, &mut h_k3_highs)?;
            global_k3_lows.extend(h_k3_lows); global_k3_highs.extend(h_k3_highs);

            // Softmax 1
            let mut h_sm1_idx = [const { Vec::new() }; 5];
            let mut h_sm1_r_sm = [const { Vec::new() }; 2];
            enforce_softmax(cs.clone(), &k1_data.q_sim[b][h], &k1_data.r_matmul[b][h], &sm1_vars.r_sm[b][h], &sm1_vars.m_sm, &sm1_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm1_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm1_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm1_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm1_vars.p[b][h], &sm1_vars.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm1_idx, &mut h_sm1_r_sm, &mut r2_pool, &mut r3_pool)?;
            for k in 0..5 { global_sm1_idx[k].extend(h_sm1_idx[k].clone()); }
            for k in 0..2 { global_sm1_r_sm[k].extend(h_sm1_r_sm[k].clone()); }

            // Softmax 2
            let mut h_sm2_idx = [const { Vec::new() }; 5];
            let mut h_sm2_r_sm = [const { Vec::new() }; 2];
            enforce_softmax(cs.clone(), &k2_data.q_sim[b][h], &k2_data.r_matmul[b][h], &sm2_vars.r_sm[b][h], &sm2_vars.m_sm, &sm2_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm2_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm2_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm2_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm2_vars.p[b][h], &sm2_vars.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm2_idx, &mut h_sm2_r_sm, &mut r2_pool, &mut r3_pool)?;
            for k in 0..5 { global_sm2_idx[k].extend(h_sm2_idx[k].clone()); }
            for k in 0..2 { global_sm2_r_sm[k].extend(h_sm2_r_sm[k].clone()); }

            // Softmax 3 (with mask)
            let zero_var = FpVar::zero();
            let num_landmarks = k3_data.q_sim[b][h].len();
            let seq_len = k3_data.q_sim[b][h][0].len();
            let mut masked_q_sim_k3 = k3_data.q_sim[b][h].clone();
            for l in 0..num_landmarks {
                for s in 0..seq_len {
                    let is_not_padding = mask_var[b][s].is_one()?;
                    masked_q_sim_k3[l][s] = FpVar::conditionally_select(&is_not_padding, &k3_data.q_sim[b][h][l][s], &zero_var)?;
                }
            }
            let mut h_sm3_idx = [const { Vec::new() }; 5];
            let mut h_sm3_r_sm = [const { Vec::new() }; 2];
            enforce_softmax(cs.clone(), &masked_q_sim_k3, &k3_data.r_matmul[b][h], &sm3_vars.r_sm[b][h], &sm3_vars.m_sm, &sm3_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm3_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm3_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm3_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm3_vars.p[b][h], &sm3_vars.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm3_idx, &mut h_sm3_r_sm, &mut r2_pool, &mut r3_pool)?;
            for k in 0..5 { global_sm3_idx[k].extend(h_sm3_idx[k].clone()); }
            for k in 0..2 { global_sm3_r_sm[k].extend(h_sm3_r_sm[k].clone()); }

            // Moore-Penrose pseudoinverse (4 steps)
            let q_inv_var = new_matrix_witness_4d(cs.clone(), &w.pseudo_inv.q_inv);
            let q_y_vars: Vec<_> = w.pseudo_inv.q_y_intermediates.iter().map(|q_step| new_matrix_witness_4d(cs.clone(), q_step)).collect();
            let r_inv_vars: Vec<_> = w.pseudo_inv.r_inv_remainders.iter().map(|r_step| new_matrix_witness_4d(cs.clone(), r_step)).collect();
            let z_zero = FpVar::zero();
            let z_inv = FpVar::Constant(w.pseudo_inv.z_inv);

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
                    m: FpVar::Constant(w.pseudo_inv.m_inv[i]),
                    m_prime: FpVar::Constant(w.pseudo_inv.m_prime_inv[i]),
                    z_w: zw.clone(), z_x: zx.clone(), z_y: zy.clone(),
                    two_k: two_k_var.clone(),
                });
            }

            let r_pows_inv = gen_challenge_vector(r_var, q_inv_var[b][h].len());

            // Step 1
            let mut h_inv0_lows = Vec::new(); let mut h_inv0_highs = Vec::new();
            enforce_freivalds(cs.clone(), &sm2_vars.p[b][h], &q_inv_var[b][h], &q_y_vars[0][b][h], &r_inv_vars[0][b][h], &w.pseudo_inv.r_inv_remainders[0][b][h], &inv_params[0], &r_pows_inv, &mut h_inv0_lows, &mut h_inv0_highs)?;
            global_inv0_lows.extend(h_inv0_lows); global_inv0_highs.extend(h_inv0_highs);

            // Step 2
            let mut h_inv1_lows = Vec::new(); let mut h_inv1_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_y_vars[0][b][h], &sm2_vars.p[b][h], &q_y_vars[1][b][h], &r_inv_vars[1][b][h], &w.pseudo_inv.r_inv_remainders[1][b][h], &inv_params[1], &r_pows_inv, &mut h_inv1_lows, &mut h_inv1_highs)?;
            global_inv1_lows.extend(h_inv1_lows); global_inv1_highs.extend(h_inv1_highs);

            // Step 3
            let mut h_inv2_lows = Vec::new(); let mut h_inv2_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_inv_var[b][h], &sm2_vars.p[b][h], &q_y_vars[2][b][h], &r_inv_vars[2][b][h], &w.pseudo_inv.r_inv_remainders[2][b][h], &inv_params[2], &r_pows_inv, &mut h_inv2_lows, &mut h_inv2_highs)?;
            global_inv2_lows.extend(h_inv2_lows); global_inv2_highs.extend(h_inv2_highs);

            // Step 4
            let mut h_inv3_lows = Vec::new(); let mut h_inv3_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_y_vars[2][b][h], &q_inv_var[b][h], &q_y_vars[3][b][h], &r_inv_vars[3][b][h], &w.pseudo_inv.r_inv_remainders[3][b][h], &inv_params[3], &r_pows_inv, &mut h_inv3_lows, &mut h_inv3_highs)?;
            global_inv3_lows.extend(h_inv3_lows); global_inv3_highs.extend(h_inv3_highs);

            // temp matmul
            let mut h_temp_lows = Vec::new(); let mut h_temp_highs = Vec::new();
            let r_pows_temp = gen_challenge_vector(r_var, temp_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &sm3_vars.p[b][h], &v_proc[b][h], &temp_data.q_sim[b][h], &temp_data.r_matmul[b][h], &w.matmul_temp.r_matmul[b][h], &params_temp, &r_pows_temp, &mut h_temp_lows, &mut h_temp_highs)?;
            global_temp_lows.extend(h_temp_lows); global_temp_highs.extend(h_temp_highs);

            // mid matmul
            let mut h_mid_lows = Vec::new(); let mut h_mid_highs = Vec::new();
            let r_pows_mid = gen_challenge_vector(r_var, mid_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &sm1_vars.p[b][h], &q_inv_var[b][h], &mid_data.q_sim[b][h], &mid_data.r_matmul[b][h], &w.matmul_mid.r_matmul[b][h], &params_mid, &r_pows_mid, &mut h_mid_lows, &mut h_mid_highs)?;
            global_mid_lows.extend(h_mid_lows); global_mid_highs.extend(h_mid_highs);

            // X matmul
            let mut h_x_lows = Vec::new(); let mut h_x_highs = Vec::new();
            let r_pows_x = gen_challenge_vector(r_var, x_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &mid_data.q_sim[b][h], &temp_data.q_sim[b][h], &x_data.q_sim[b][h], &x_data.r_matmul[b][h], &w.matmul_x.r_matmul[b][h], &params_x, &r_pows_x, &mut h_x_lows, &mut h_x_highs)?;
            global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);
        }
    }

    // Push to committed_vars in the same order as the original
    {
        let mut final_vars = committed_vars.borrow_mut();
        for seg in 0..5 { final_vars.extend(global_sm1_idx[seg].clone()); }
        for seg in 0..5 { final_vars.extend(global_sm2_idx[seg].clone()); }
        for seg in 0..5 { final_vars.extend(global_sm3_idx[seg].clone()); }
        final_vars.extend(global_k1_lows); final_vars.extend(global_k1_highs);
        final_vars.extend(global_k2_lows); final_vars.extend(global_k2_highs);
        final_vars.extend(global_k3_lows); final_vars.extend(global_k3_highs);
        for k in 0..2 { final_vars.extend(global_sm1_r_sm[k].clone()); }
        for k in 0..2 { final_vars.extend(global_sm2_r_sm[k].clone()); }
        for k in 0..2 { final_vars.extend(global_sm3_r_sm[k].clone()); }
        final_vars.extend(global_inv0_lows); final_vars.extend(global_inv0_highs);
        final_vars.extend(global_inv1_lows); final_vars.extend(global_inv1_highs);
        final_vars.extend(global_inv2_lows); final_vars.extend(global_inv2_highs);
        final_vars.extend(global_inv3_lows); final_vars.extend(global_inv3_highs);
        final_vars.extend(global_temp_lows); final_vars.extend(global_temp_highs);
        final_vars.extend(global_mid_lows); final_vars.extend(global_mid_highs);
        final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
        final_vars.extend(r2_pool);
        final_vars.extend(r3_pool);
    }

    Ok(())
}

/// Generate BERT attention constraints.
/// Copied from BERTAttentionCircuit::generate_constraints.
pub fn generate_bert_attn_constraints<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    w: &BertAttnWitness<F>,
    r_var: &FpVar<F>,
    two_k_var: &FpVar<F>,
    committed_vars: &Rc<RefCell<Vec<FpVar<F>>>>,
    committed_vars_2: &Rc<RefCell<Vec<FpVar<F>>>>,
) -> Result<(), SynthesisError> {
    let mut r2_pool = Vec::new();
    let mut r3_pool = Vec::new();

    let q_proc = new_matrix_witness_4d(cs.clone(), &w.q_q);
    let k_proc = new_matrix_witness_4d(cs.clone(), &w.q_k);
    let v_proc = new_matrix_witness_4d(cs.clone(), &w.q_v);

    // d2: Q, K, V
    {
        let mut vars_2 = committed_vars_2.borrow_mut();
        for batch in &q_proc { for head in batch { for seq in head { vars_2.extend(seq.iter().cloned()); } } }
        for batch in &k_proc { for head in batch { for seq in head { vars_2.extend(seq.iter().cloned()); } } }
        for batch in &v_proc { for head in batch { for seq in head { vars_2.extend(seq.iter().cloned()); } } }
    }

    let (attn_data, params_attn) = w.matmul_attn.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);
    let (x_data, params_x) = w.matmul_x.alloc_and_setup(cs.clone(), F::zero(), w.z_v, two_k_var);
    let mask_var = new_matrix_witness(cs.clone(), &w.mask);
    let sm_vars = w.sm.alloc_vars(cs.clone())?;

    let mut global_attn_lows = Vec::new(); let mut global_attn_highs = Vec::new();
    let mut global_sm_idx = vec![Vec::new(); 5];
    let mut global_sm_r_sm = vec![Vec::new(); 2];
    let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();
    let zero_var = FpVar::zero();

    for b in 0..q_proc.len() {
        for h in 0..q_proc[b].len() {
            // QK^T
            let k_t = transpose_matrix(&k_proc[b][h]);
            let r_pows_attn = gen_challenge_vector(r_var, attn_data.q_sim[b][h][0].len());
            let mut h_attn_lows = Vec::new(); let mut h_attn_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_proc[b][h], &k_t, &attn_data.q_sim[b][h], &attn_data.r_matmul[b][h], &w.matmul_attn.r_matmul[b][h], &params_attn, &r_pows_attn, &mut h_attn_lows, &mut h_attn_highs)?;
            global_attn_lows.extend(h_attn_lows); global_attn_highs.extend(h_attn_highs);

            // Mask
            let seq_len = attn_data.q_sim[b][h].len();
            let col_len = attn_data.q_sim[b][h][0].len();
            let mut masked_q_sim = attn_data.q_sim[b][h].clone();
            for row in 0..seq_len {
                for col in 0..col_len {
                    let is_valid = mask_var[b][col].is_one()?;
                    masked_q_sim[row][col] = FpVar::conditionally_select(&is_valid, &attn_data.q_sim[b][h][row][col], &zero_var)?;
                }
            }

            // Full-sequence softmax
            let mut h_sm_idx = [const { Vec::new() }; 5];
            let mut h_sm_r_sm = [const { Vec::new() }; 2];
            enforce_softmax(cs.clone(), &masked_q_sim, &attn_data.r_matmul[b][h], &sm_vars.r_sm[b][h], &sm_vars.m_sm, &sm_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(), &sm_vars.p[b][h], &sm_vars.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm_idx, &mut h_sm_r_sm, &mut r2_pool, &mut r3_pool)?;
            for k in 0..5 { global_sm_idx[k].extend(h_sm_idx[k].clone()); }
            for k in 0..2 { global_sm_r_sm[k].extend(h_sm_r_sm[k].clone()); }

            // P * V
            let r_pows_x = gen_challenge_vector(r_var, x_data.q_sim[b][h][0].len());
            let mut h_x_lows = Vec::new(); let mut h_x_highs = Vec::new();
            enforce_freivalds(cs.clone(), &sm_vars.p[b][h], &v_proc[b][h], &x_data.q_sim[b][h], &x_data.r_matmul[b][h], &w.matmul_x.r_matmul[b][h], &params_x, &r_pows_x, &mut h_x_lows, &mut h_x_highs)?;
            global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);
        }
    }

    // Push to committed_vars
    {
        let mut final_vars = committed_vars.borrow_mut();
        for seg in 0..5 { final_vars.extend(global_sm_idx[seg].clone()); }
        final_vars.extend(global_attn_lows); final_vars.extend(global_attn_highs);
        for k in 0..2 { final_vars.extend(global_sm_r_sm[k].clone()); }
        final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
        final_vars.extend(r2_pool);
        final_vars.extend(r3_pool);
    }

    Ok(())
}

/// Generate RMSNorm MLP constraints.
pub fn generate_mlp_rmsnorm_constraints<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    data: &NystromMLPRMSNormData<F>,
    committed_vars: &Rc<RefCell<Vec<FpVar<F>>>>,
) -> Result<(), SynthesisError> {
    let mut rms1_pool = Vec::new();
    let mut gelu_pool = Vec::new();
    let mut rms2_pool = Vec::new();
    let mut r_pool = Vec::new();

    let dim = data.norm1.q_ri[0][0].len();
    let q_norm1_vars = enforce_rmsnorm(cs.clone(), &data.norm1, dim, data.frac_bits, data.k_bits, &mut r_pool, &mut rms1_pool)?;
    let (q_ff1_vars, r_ff1_vars) = enforce_linear(cs.clone(), &q_norm1_vars, &data.ff1, data.k_bits, &mut r_pool)?;
    let q_gelu_vars = enforce_gelu(cs.clone(), &q_ff1_vars, &r_ff1_vars, &data.gelu1, data.k_bits, &mut r_pool, &mut gelu_pool)?;
    let (q_ff2_vars, _) = enforce_linear(cs.clone(), &q_gelu_vars, &data.ff2, data.k_bits, &mut r_pool)?;
    let _q_norm2_vars = enforce_rmsnorm(cs.clone(), &data.norm2, dim, data.frac_bits, data.k_bits, &mut r_pool, &mut rms2_pool)?;

    {
        let mut final_vars = committed_vars.borrow_mut();
        final_vars.extend(rms1_pool);
        final_vars.extend(gelu_pool);
        final_vars.extend(rms2_pool);
        final_vars.extend(r_pool);
    }
    Ok(())
}

/// Generate LayerNorm MLP constraints.
pub fn generate_mlp_layernorm_constraints<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    data: &NystromMLPLayerNormData<F>,
    committed_vars: &Rc<RefCell<Vec<FpVar<F>>>>,
) -> Result<(), SynthesisError> {
    let mut ln1_pool = Vec::new();
    let mut gelu_pool = Vec::new();
    let mut ln2_pool = Vec::new();
    let mut r_pool = Vec::new();

    let dim = data.norm1.q_ri[0][0].len();
    let q_norm1_vars = enforce_layernorm(cs.clone(), &data.norm1, dim, data.frac_bits, data.k_bits, &mut r_pool, &mut ln1_pool)?;
    let (q_ff1_vars, r_ff1_vars) = enforce_linear(cs.clone(), &q_norm1_vars, &data.ff1, data.k_bits, &mut r_pool)?;
    let q_gelu_vars = enforce_gelu(cs.clone(), &q_ff1_vars, &r_ff1_vars, &data.gelu1, data.k_bits, &mut r_pool, &mut gelu_pool)?;
    let (q_ff2_vars, _) = enforce_linear(cs.clone(), &q_gelu_vars, &data.ff2, data.k_bits, &mut r_pool)?;
    let _q_norm2_vars = enforce_layernorm(cs.clone(), &data.norm2, dim, data.frac_bits, data.k_bits, &mut r_pool, &mut ln2_pool)?;

    {
        let mut final_vars = committed_vars.borrow_mut();
        final_vars.extend(ln1_pool);
        final_vars.extend(gelu_pool);
        final_vars.extend(ln2_pool);
        final_vars.extend(r_pool);
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Version A: ModelIntegratedCircuit
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct ModelIntegratedCircuit<F: PrimeField> {
    pub attn_layers: Vec<AttnWitness<F>>,
    pub mlp_layers: Vec<MlpWitness<F>>,
    pub frac_bits: usize,
    pub k_bits: usize,
    pub r_challenge: F,
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for ModelIntegratedCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        for layer_idx in 0..self.attn_layers.len() {
            // Attention constraints
            match &self.attn_layers[layer_idx] {
                AttnWitness::Nystrom(w) => {
                    generate_nystrom_attn_constraints(
                        cs.clone(), w, &r_var, &two_k_var,
                        &self.committed_vars, &self.committed_vars_2,
                    )?;
                }
                AttnWitness::Bert(w) => {
                    generate_bert_attn_constraints(
                        cs.clone(), w, &r_var, &two_k_var,
                        &self.committed_vars, &self.committed_vars_2,
                    )?;
                }
            }

            // MLP constraints
            match &self.mlp_layers[layer_idx] {
                MlpWitness::LayerNorm(data) => {
                    generate_mlp_layernorm_constraints(
                        cs.clone(), data, &self.committed_vars,
                    )?;
                }
                MlpWitness::RMSNorm(data) => {
                    generate_mlp_rmsnorm_constraints(
                        cs.clone(), data, &self.committed_vars,
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// Load all layer witnesses and build a ModelIntegratedCircuit.
pub fn load_model_integrated_circuit<E: Pairing>(
    discovered: &DiscoveredWitnesses,
) -> Result<ModelIntegratedCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut attn_layers = Vec::with_capacity(discovered.num_layers);
    let mut mlp_layers = Vec::with_capacity(discovered.num_layers);
    let mut frac_bits = 0;
    let mut k_bits = 0;

    for i in 0..discovered.num_layers {
        let attn = match discovered.attn_variant {
            AttentionVariant::Nystrom => {
                let w = load_nystrom_attn_witness::<E>(&discovered.attn_files[i])?;
                if i == 0 { frac_bits = w.frac_bits; k_bits = w.k_bits; }
                AttnWitness::Nystrom(w)
            }
            AttentionVariant::Bert => {
                let w = load_bert_attn_witness::<E>(&discovered.attn_files[i])?;
                if i == 0 { frac_bits = w.frac_bits; k_bits = w.k_bits; }
                AttnWitness::Bert(w)
            }
        };
        attn_layers.push(attn);

        let mlp = match discovered.norm_variant {
            NormType::LayerNorm => {
                use crate::circuits::freivalds_matmul::nystrom_mlp_circuit::load_nystrom_layernorm_data;
                let data = load_nystrom_layernorm_data::<E>(&discovered.mlp_files[i])?;
                MlpWitness::LayerNorm(data)
            }
            NormType::RMSNorm => {
                let data = load_rmsnorm_mlp_data::<E>(&discovered.mlp_files[i])?;
                MlpWitness::RMSNorm(data)
            }
        };
        mlp_layers.push(mlp);
    }

    Ok(ModelIntegratedCircuit {
        attn_layers,
        mlp_layers,
        frac_bits,
        k_bits,
        r_challenge: E::ScalarField::zero(),
        committed_vars: Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
    })
}

/// Full proving pipeline for the model-integrated circuit.
pub fn run_model_integrated_paths<E: Pairing>(
    witness_dir: &Path,
) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    use crate::util::model_integrated_lookup::{run_layer_attn_lasso, run_layer_mlp_lasso};

    let mut rng = StdRng::seed_from_u64(42u64);
    let e2e_start = Instant::now();

    // Initialize all lookup tables
    if std::env::var("SOFTMAX_TABLE_PATH").is_err() { set_softmax_table_dir(witness_dir); }
    if std::env::var("LAYERNORM_TABLE_PATH").is_err() { set_layernorm_table_dir(witness_dir); }
    if std::env::var("GELU_TABLE_PATH").is_err() { set_gelu_table_dir(witness_dir); }

    let discovered = discover_witnesses(witness_dir)?;
    println!("\n=== [Model Integrated] {} layers, {:?} attn, {:?} norm ===",
        discovered.num_layers, discovered.attn_variant, discovered.norm_variant);

    // Dry-run → d1/d2 count
    let (num_d1, num_d2) = {
        let temp = load_model_integrated_circuit::<E>(&discovered)?;
        let v1 = Rc::clone(&temp.committed_vars);
        let v2 = Rc::clone(&temp.committed_vars_2);
        let cs = ConstraintSystem::<Fr>::new_ref();
        temp.generate_constraints(cs)?;
        (v1.borrow().len() as u32, v2.borrow().len() as u32)
    };
    println!("Commit counts: d1={}, d2={}", num_d1, num_d2);

    let circuit = load_model_integrated_circuit::<E>(&discovered)?;

    // [0] Setup
    println!("=== [0] LegoGroth16 Setup ===");
    let setup_start = Instant::now();
    let params = generate_random_parameters::<E, _, _>(circuit.clone(), num_d1, num_d2, &mut rng)
        .map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
    let setup_time = setup_start.elapsed();
    let pvk = prepare_verifying_key::<E>(&params.vk);

    // Witness extraction
    let cs_witness = ConstraintSystem::<Fr>::new_ref();
    circuit.clone().generate_constraints(cs_witness.clone())?;
    let num_constraints = cs_witness.num_constraints();

    let committed_group1: Vec<Fr> = circuit.committed_vars.borrow().iter()
        .map(|v| v.value().unwrap_or(Fr::zero())).collect();
    let num_pub = cs_witness.num_instance_variables() - 1;

    // Groth16 commit + Fiat-Shamir
    let v1 = Fr::rand(&mut rng);
    let v2 = Fr::rand(&mut rng);

    let g16_commit_start = Instant::now();
    let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
    let g16_commit_ms = g16_commit_start.elapsed().as_millis();
    let r_fs = hash_cm_to_field::<E>(&cm1);

    let final_circuit = ModelIntegratedCircuit {
        r_challenge: r_fs,
        ..circuit.clone()
    };

    // [1] Groth16 Prove
    println!("=== [1] LegoGroth16 Prove ===");
    let g16_prove_start = Instant::now();
    let proof = create_random_proof::<E, _, _>(final_circuit.clone(), v1, v2, &params, &mut rng)
        .map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
    let g16_prove_time = g16_prove_start.elapsed();

    let mut proof_bytes = Vec::new();
    proof.serialize_uncompressed(&mut proof_bytes)
        .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
    let proof_size_bytes = proof_bytes.len();

    // [1] Groth16 Verify
    let g16_verify_start = Instant::now();
    let result = verify_proof::<E>(&pvk, &proof, &[r_fs]);
    let g16_verify_time = g16_verify_start.elapsed();
    println!("  - Model Integrated SNARK Verification: {:?}", result);
    if let Err(e) = result {
        return Err(anyhow::anyhow!(format!("{:?}", e)));
    }

    let g16_report = proof_size::compute_groth16_report::<E>(&proof);
    g16_report.print_log();

    // [2] Lasso (all layers)
    proof_size::reset_lasso_aggregator();
    println!("=== [2] Model Integrated Lasso Lookup Proofs ===");
    let lasso_start = Instant::now();
    let mut lasso_agg = LassoMetricsAggregate::default();
    let mut lasso_results: Vec<LassoResult> = Vec::new();

    for i in 0..discovered.num_layers {
        println!("  -- Layer {} Attention Lasso --", i);
        let (attn_res, attn_agg) = run_layer_attn_lasso(&final_circuit.attn_layers[i], i, &mut rng)?;
        lasso_agg.add_aggregate(&attn_agg);
        lasso_results.extend(attn_res);

        println!("  -- Layer {} MLP Lasso --", i);
        let (mlp_res, mlp_agg) = run_layer_mlp_lasso(&final_circuit.mlp_layers[i], i, &mut rng)?;
        lasso_agg.add_aggregate(&mlp_agg);
        lasso_results.extend(mlp_res);
    }
    let lasso_time = lasso_start.elapsed();
    lasso_agg.total_ms = lasso_time.as_millis();

    // Debug assertion: committed_vars length == sum of Lasso sparsities
    let lasso_sp_sum: usize = lasso_results.iter().map(|(_, _, _, sp)| *sp).sum();
    debug_assert_eq!(
        committed_group1.len(), lasso_sp_sum,
        "committed_vars length ({}) must equal sum of all Lasso sparsities ({})",
        committed_group1.len(), lasso_sp_sum
    );

    // [3] CP-Link
    println!("=== [3] CP-Link ===");
    let u_padded = pad_lasso_u_values::<E>(&lasso_results, &committed_group1);

    let cp_setup_start = Instant::now();
    let (pp, crs) = factored_link_setup::<E, _>(&params.vk, num_pub, committed_group1.len(), &lasso_results, &mut rng);
    let (ek, vk_link) = factored_link_keygen::<E, _>(&pp, crs, &mut rng);
    let cp_setup_ms = cp_setup_start.elapsed().as_millis();

    let cp_prove_start = Instant::now();
    let (proof_link, instance) = factored_link_prove::<E, _>(&pp, &ek, v1, cm1, &lasso_results, u_padded, &mut rng);
    let cp_prove_ms = cp_prove_start.elapsed().as_millis();

    let cp_verify_start = Instant::now();
    if !factored_link_verify::<E>(&pp, &vk_link, &instance, &proof_link) {
        return Err(anyhow::anyhow!("Factored link verification failed"));
    }
    println!("  -> [Success] Model Integrated CP-Link verified");
    let cp_verify_ms = cp_verify_start.elapsed().as_millis();

    let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
    cplink_report.print_log();
    proof_size::print_full_report(&g16_report, &cplink_report);

    let e2e_time = e2e_start.elapsed();

    BenchmarkResult {
        file: format!("{}_layers", discovered.num_layers),
        circuit: "model_integrated".to_string(),
        constraints: num_constraints,
        groth16: Groth16Metrics {
            setup_ms: setup_time.as_millis(),
            commit_ms: g16_commit_ms,
            prove_ms: g16_prove_time.as_millis(),
            verify_ms: g16_verify_time.as_millis(),
            proof_size_bytes: proof_size_bytes,
            commitment_size_bytes: groth16_commitment_size::<E>(&cm1),
        },
        lasso: LassoBenchmarkMetrics::from_aggregate(&lasso_agg),
        cp_link: CpLinkMetrics {
            setup_ms: cp_setup_ms,
            prove_ms: cp_prove_ms,
            verify_ms: cp_verify_ms,
            proof_size_bytes: cplink_report.proof_total,
        },
        end_to_end_ms: e2e_time.as_millis(),
    }
    .print_json();

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn model_integrated_all_layers_test() {
    use std::env;

    let witness_dir = env::var("MODEL_INTEGRATED_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_m64".to_string());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] MODEL_INTEGRATED_WITNESS_DIR={} not a directory.",
            witness_dir
        );
        return;
    }

    println!("\n=== [Model Integrated prove+verify] dir={} ===", witness_dir);

    let total_start = Instant::now();
    let res = run_model_integrated_paths::<Bn254>(dir);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!("\n=== [Model Integrated] PASS | total_wall_ms={} ===", total_ms),
        Err(e) => panic!("[Model Integrated] failed after {}ms: {}", total_ms, e),
    }
}
