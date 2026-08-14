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
use ndarray::{Ix0, Ix3, Ix4, Ix2, IxDyn};
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
    eq::EqGadget,
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
            global_link_setup, global_link_keygen, global_link_prove, global_link_verify,
            groth16_commitment_size,
        },
        proof_size,
    },
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu, split_remainder_and_push},
        layernorm::constraints::{NormBundle, LayerNormBundle, enforce_rmsnorm, enforce_layernorm},
        linear::constraints::{LinearBundle, enforce_linear},
        matmul::{
            constraints::new_matrix_witness,
            quantized::{
                MatmulBundle, PseudoinverseBundle, ProjectionBundle, ScalingBundle, ZenParams,
                compute_circuit_landmarks, enforce_freivalds, enforce_scaling,
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
            to_field_3d_i64, to_field_3d_u64,
            to_field_4d_i64, to_field_4d_u64, to_field_5d_i64,
        },
        norm_detect::{NormType, detect_norm_type},
        utils::{cm1_from_vk, cm2_from_vk, cm_aux_from_vk, generate_global_ck, commit_global, hash_commitments_to_field_v3},
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

/// Witness data for Toeplitz conv (Nystrom-specific: X_final = Ỹ + T(W_conv) @ V).
#[derive(Clone)]
pub struct ConvBundle<F: PrimeField> {
    pub q_t_conv: Vec<Vec<Vec<F>>>,      // [H][S][S] — quantized Toeplitz matrix
    pub z_t_conv: F,                      // zero-point for Toeplitz
    pub q_conv: Vec<Vec<Vec<Vec<F>>>>,    // [B][H][S][D] — conv matmul output
    pub r_conv: Vec<Vec<Vec<Vec<F>>>>,    // [B][H][S][D] — conv matmul remainder
    pub m_conv: F,                        // ZEN M constant
    pub m_prime_conv: F,                  // ZEN M' constant
    pub z_conv_out: F,                    // zero-point for conv output
    pub q_xf: Vec<Vec<Vec<Vec<F>>>>,      // [B][H][S][D] — post-conv quantized result
    pub r_xf: Vec<Vec<Vec<Vec<F>>>>,      // [B][H][S][D] — post-conv remainder
    pub m_add_x: F,                       // rescaled addition multiplier for X
    pub m_add_conv: F,                    // rescaled addition multiplier for conv
    pub z_xf: F,                          // zero-point for X_final
}

/// Pure witness data for Nystrom attention (no circuit scaffolding).
#[derive(Clone)]
pub struct NystromAttnWitness<F: PrimeField> {
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,
    // Projection inputs
    pub q_x_input: Vec<Vec<Vec<F>>>,      // [B][S][H*D]
    pub z_x_input: F,
    // Q/K/V Projection
    pub proj_q: ProjectionBundle<F>,
    pub proj_k: ProjectionBundle<F>,
    pub proj_v: ProjectionBundle<F>,
    // Q/K Scaling
    pub scale_q: ScalingBundle<F>,
    pub scale_k: ScalingBundle<F>,
    // Output projection
    pub z_attn_output: F,
    pub q_attn_output: Vec<Vec<Vec<F>>>,  // [B][S][HD] — O-proj input (post-conv)
    pub proj_o: ProjectionBundle<F>,
    // Core matmuls
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
    pub input_hash: F,
    pub output_hash: F,
    /// Toeplitz conv witness (Some when W_conv exists, None otherwise)
    pub conv: Option<ConvBundle<F>>,
}

/// Pure witness data for BERT attention (no circuit scaffolding).
#[derive(Clone)]
pub struct BertAttnWitness<F: PrimeField> {
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,
    // Projection inputs
    pub q_x_input: Vec<Vec<Vec<F>>>,      // [B][S][H*D]
    pub z_x_input: F,
    // Q/K/V Projection
    pub proj_q: ProjectionBundle<F>,
    pub proj_k: ProjectionBundle<F>,
    pub proj_v: ProjectionBundle<F>,
    // Q/K Scaling
    pub scale_q: ScalingBundle<F>,
    pub scale_k: ScalingBundle<F>,
    // Output projection
    pub z_attn_output: F,
    pub q_attn_output: Vec<Vec<Vec<F>>>,  // [B][S][HD] — O-proj input (post-conv)
    pub proj_o: ProjectionBundle<F>,
    // Core matmuls
    pub matmul_attn: MatmulBundle<F>,
    pub sm: SoftmaxBundle<F>,
    pub matmul_x: MatmulBundle<F>,
    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,
    pub input_hash: F,
    pub output_hash: F,
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
            table_idx: to_field_4d_u64(&npz.by_name::<_, Ix4>(&format!("table_idx_{}.npy", suffix))?.into_dimensionality()?),
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

    let z_v = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar();
    let matmul_temp = load_matmul(&mut npz, "temp", false)?;
    let matmul_mid = load_matmul(&mut npz, "mid", false)?;
    let matmul_x = load_matmul(&mut npz, "x", false)?;

    // === Projection witness loading ===
    let q_x_input = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_x_input.npy")?.into_dimensionality()?);
    let z_x_input = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_x_input.npy")?.into_scalar());

    let load_projection = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ProjectionBundle<E::ScalarField>> {
        Ok(ProjectionBundle {
            q_w_diff: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_w_diff_{}.npy", suffix))?.into_dimensionality()?),
            q_b_diff: to_field_2d_i64(&npz.by_name::<_, Ix2>(&format!("Q_b_diff_{}.npy", suffix))?.into_dimensionality()?),
            z_w: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_w_{}.npy", suffix))?.into_scalar()),
            m_proj: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_proj_{}.npy", suffix))?.into_scalar()),
            m_prime_proj: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_proj_{}.npy", suffix))?.into_scalar()),
            q_out_proj: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_raw_proj_{}.npy", suffix))?.into_dimensionality()?),
            r_out_proj: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_proj_{}.npy", suffix))?.into_dimensionality()?),
            z_raw: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_raw_{}.npy", suffix))?.into_scalar()),
        })
    };
    let proj_q = load_projection(&mut npz, "q")?;
    let proj_k = load_projection(&mut npz, "k")?;
    let proj_v = load_projection(&mut npz, "v")?;

    let load_scaling = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ScalingBundle<E::ScalarField>> {
        Ok(ScalingBundle {
            q_scaled: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_scaled_{}.npy", suffix))?.into_dimensionality()?),
            r_scale: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_scale_{}.npy", suffix))?.into_dimensionality()?),
            m_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_scale_{}.npy", suffix))?.into_scalar()),
            m_prime_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_scale_{}.npy", suffix))?.into_scalar()),
        })
    };
    let scale_q = load_scaling(&mut npz, "q")?;
    let scale_k = load_scaling(&mut npz, "k")?;

    let z_attn_output = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_attn_output.npy")?.into_scalar());
    let q_attn_output = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_attn_output.npy")?.into_dimensionality()?);
    let proj_o = load_projection(&mut npz, "o")?;

    // === Conv witness (optional — only present for Nystrom with conv) ===
    let conv = match npz.by_name::<ndarray::OwnedRepr<i64>, Ix3>("Q_t_conv.npy") {
        Ok(q_t_conv_raw) => {
            let q_t_conv_raw = q_t_conv_raw.into_dimensionality()?;
            let z_t_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_t_conv.npy")?.into_scalar();
            let m_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_conv.npy")?.into_scalar();
            let m_prime_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_conv.npy")?.into_scalar();
            let z_conv_out = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_conv_out.npy")?.into_scalar();
            let m_add_x = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_add_x.npy")?.into_scalar();
            let m_add_conv = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_add_conv.npy")?.into_scalar();
            let z_xf = npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_xf.npy")?.into_scalar();
            Some(ConvBundle {
                q_t_conv: to_field_3d_i64(&q_t_conv_raw),
                z_t_conv: E::ScalarField::from(z_t_conv),
                q_conv: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_conv.npy")?.into_dimensionality()?),
                r_conv: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_conv.npy")?.into_dimensionality()?),
                m_conv: E::ScalarField::from(m_conv),
                m_prime_conv: E::ScalarField::from(m_prime_conv),
                z_conv_out: E::ScalarField::from(z_conv_out),
                q_xf: to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_xf.npy")?.into_dimensionality()?),
                r_xf: to_field_4d_i64(&npz.by_name::<_, Ix4>("R_xf.npy")?.into_dimensionality()?),
                m_add_x: E::ScalarField::from(m_add_x),
                m_add_conv: E::ScalarField::from(m_add_conv),
                z_xf: E::ScalarField::from(z_xf),
            })
        }
        Err(_) => None,
    };

    Ok(NystromAttnWitness {
        z_q: E::ScalarField::from(z_q),
        z_k: E::ScalarField::from(z_k),
        z_v: E::ScalarField::from(z_v),
        q_x_input, z_x_input,
        proj_q, proj_k, proj_v,
        scale_q, scale_k,
        z_attn_output, q_attn_output, proj_o,
        matmul1, matmul2, matmul3,
        sm1, sm2, sm3,
        pseudo_inv,
        matmul_temp, matmul_mid, matmul_x,
        frac_bits, limb_bits, k_segments, k_bits,
        seg_size: seg,
        mask: to_field_2d_i64(&mask_raw),
        input_hash: E::ScalarField::zero(),
        output_hash: E::ScalarField::zero(),
        conv,
    })
}

/// Load a BERT attention witness from NPZ.
pub fn load_bert_attn_witness<E: Pairing>(path: &Path) -> Result<BertAttnWitness<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening BERT witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let q_q_raw: ndarray::Array4<i64> = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?;
    let q_k_raw: ndarray::Array4<i64> = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?;
    let q_v_raw: ndarray::Array4<i64> = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_v.npy")?;
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

    let table_idx_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("table_idx.npy")?.into_dimensionality()?;
    let exp_total_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("exp_total.npy")?.into_dimensionality()?;
    let p_attn_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("P_attn.npy")?.into_dimensionality()?;
    let r_div_attn_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("R_div_attn.npy")?.into_dimensionality()?;
    let r_sm_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("r_sm.npy")?.into_dimensionality()?;

    let sm = SoftmaxBundle {
        table_idx: to_field_4d_u64(&table_idx_raw),
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

    // === Projection witness loading ===
    let q_x_input = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_x_input.npy")?.into_dimensionality()?);
    let z_x_input = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_x_input.npy")?.into_scalar());

    let load_projection = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ProjectionBundle<E::ScalarField>> {
        Ok(ProjectionBundle {
            q_w_diff: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_w_diff_{}.npy", suffix))?.into_dimensionality()?),
            q_b_diff: to_field_2d_i64(&npz.by_name::<_, Ix2>(&format!("Q_b_diff_{}.npy", suffix))?.into_dimensionality()?),
            z_w: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_w_{}.npy", suffix))?.into_scalar()),
            m_proj: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_proj_{}.npy", suffix))?.into_scalar()),
            m_prime_proj: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_proj_{}.npy", suffix))?.into_scalar()),
            q_out_proj: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_raw_proj_{}.npy", suffix))?.into_dimensionality()?),
            r_out_proj: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_proj_{}.npy", suffix))?.into_dimensionality()?),
            z_raw: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_raw_{}.npy", suffix))?.into_scalar()),
        })
    };
    let proj_q = load_projection(&mut npz, "q")?;
    let proj_k = load_projection(&mut npz, "k")?;
    let proj_v = load_projection(&mut npz, "v")?;

    let load_scaling = |npz: &mut NpzReader<File>, suffix: &str| -> Result<ScalingBundle<E::ScalarField>> {
        Ok(ScalingBundle {
            q_scaled: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_scaled_{}.npy", suffix))?.into_dimensionality()?),
            r_scale: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_scale_{}.npy", suffix))?.into_dimensionality()?),
            m_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_scale_{}.npy", suffix))?.into_scalar()),
            m_prime_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_scale_{}.npy", suffix))?.into_scalar()),
        })
    };
    let scale_q = load_scaling(&mut npz, "q")?;
    let scale_k = load_scaling(&mut npz, "k")?;

    let z_attn_output = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_attn_output.npy")?.into_scalar());
    let q_attn_output = to_field_3d_i64(&npz.by_name::<_, Ix3>("Q_attn_output.npy")?.into_dimensionality()?);
    let proj_o = load_projection(&mut npz, "o")?;

    Ok(BertAttnWitness {
        z_q, z_k, z_v,
        q_x_input, z_x_input,
        proj_q, proj_k, proj_v,
        scale_q, scale_k,
        z_attn_output, q_attn_output, proj_o,
        matmul_attn, sm, matmul_x,
        frac_bits, limb_bits, k_segments, k_bits,
        mask: to_field_2d_i64(&mask_raw),
        input_hash: E::ScalarField::zero(),
        output_hash: E::ScalarField::zero(),
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
            q_out_sim: to_field_3d_i64(&npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?.into_dimensionality()?),
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
    committed_vars_aux: &Rc<RefCell<Vec<FpVar<F>>>>,
) -> Result<(), SynthesisError> {
    // r_div: 26-bit split (low 16 + high 10) — per-softmax pools
    let mut r_div_k1_lows = Vec::new();
    let mut r_div_k1_highs = Vec::new();
    let mut r_div_k2_lows = Vec::new();
    let mut r_div_k2_highs = Vec::new();
    let mut r_div_k3_lows = Vec::new();
    let mut r_div_k3_highs = Vec::new();

    let r_mean_k1 = new_matrix_witness_4d(cs.clone(), &w.matmul1.r_mean);
    let r_mean_k2 = new_matrix_witness_4d(cs.clone(), &w.matmul2.r_mean);

    let (k1_data, params_k1) = w.matmul1.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);
    let (k2_data, params_k2) = w.matmul2.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);
    let (k3_data, params_k3) = w.matmul3.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);

    let mask_var = new_matrix_witness(cs.clone(), &w.mask);
    let sm1_vars = w.sm1.alloc_vars(cs.clone())?;
    let sm2_vars = w.sm2.alloc_vars(cs.clone())?;
    let sm3_vars = w.sm3.alloc_vars(cs.clone())?;

    // === Projection input allocation (witness for x_input) ===
    let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = w.q_x_input.iter().map(|batch|
        batch.iter().map(|seq| seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()).collect()
    ).collect();
    let proj_q_out = new_matrix_witness_4d(cs.clone(), &w.proj_q.q_out_proj);
    let proj_q_r = new_matrix_witness_4d(cs.clone(), &w.proj_q.r_out_proj);
    let proj_k_out = new_matrix_witness_4d(cs.clone(), &w.proj_k.q_out_proj);
    let proj_k_r = new_matrix_witness_4d(cs.clone(), &w.proj_k.r_out_proj);
    let proj_v_out = new_matrix_witness_4d(cs.clone(), &w.proj_v.q_out_proj);
    let proj_v_r = new_matrix_witness_4d(cs.clone(), &w.proj_v.r_out_proj);
    let scale_q_out = new_matrix_witness_4d(cs.clone(), &w.scale_q.q_scaled);
    let scale_q_r = new_matrix_witness_4d(cs.clone(), &w.scale_q.r_scale);
    let scale_k_out = new_matrix_witness_4d(cs.clone(), &w.scale_k.q_scaled);
    let scale_k_r = new_matrix_witness_4d(cs.clone(), &w.scale_k.r_scale);
    let proj_o_out = new_matrix_witness_4d(cs.clone(), &w.proj_o.q_out_proj);
    let proj_o_r = new_matrix_witness_4d(cs.clone(), &w.proj_o.r_out_proj);

    // d3: commit Freivalds-constrained intermediates (Q, K, V)
    {
        let mut vars_aux = committed_vars_aux.borrow_mut();
        for b in &proj_q_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
        for b in &proj_k_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
        for b in &proj_v_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
    }

    // Allocate W_Q, W_K, W_V weights as witnesses (transposed form: [H][HD][D])
    let num_heads = w.proj_q.q_w_diff.len();
    let w_q_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_q.q_w_diff[h].len();
        let hd_dim = w.proj_q.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_q.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();
    let w_k_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_k.q_w_diff[h].len();
        let hd_dim = w.proj_k.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_k.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();
    let w_v_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_v.q_w_diff[h].len();
        let hd_dim = w.proj_v.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_v.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();

    let w_o_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_o.q_w_diff[h].len();
        let hd_dim = w.proj_o.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_o.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();

    // d2: commit W_Q, W_K, W_V, W_O weights
    {
        let mut vars_2 = committed_vars_2.borrow_mut();
        for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        for h_w in &w_k_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
    }

    let (temp_data, params_temp) = w.matmul_temp.alloc_and_setup(cs.clone(), F::zero(), w.z_v, two_k_var);
    let (mid_data, params_mid) = w.matmul_mid.alloc_and_setup(cs.clone(), F::zero(), w.pseudo_inv.z_inv, two_k_var);
    let (x_data, params_x) = w.matmul_x.alloc_and_setup(cs.clone(), w.matmul_mid.z_sim, w.matmul_temp.z_sim, two_k_var);

    // d3: commit chain intermediates (temp=B̃V, mid=F̃Ã⁺, x=Ỹ)
    {
        let mut vars_aux = committed_vars_aux.borrow_mut();
        for b in &temp_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
        for b in &mid_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
        for b in &x_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
    }

    // === Conv allocation & O-proj input derivation ===
    // If conv exists: allocate Toeplitz, Q_conv, R_conv, Q_xf, R_xf.
    //   attn_output_vars = reshape(Q_xf) — bound via FpVar::clone (zero constraints).
    // If no conv:  attn_output_vars = reshape(x_data.q_sim) — same binding as BERT.
    let (q_t_conv_vars, q_conv_vars, r_conv_vars, q_xf_vars, r_xf_vars, attn_output_vars) =
    if let Some(ref conv) = w.conv {
        // Allocate quantized Toeplitz [H][S][S] — shared across batches
        let q_t_conv_vars: Vec<Vec<Vec<FpVar<F>>>> = conv.q_t_conv.iter().map(|head| {
            head.iter().map(|row| {
                row.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()
            }).collect()
        }).collect();
        // d2: commit Toeplitz weights (model-derived, reusable)
        {
            let mut vars_2 = committed_vars_2.borrow_mut();
            for head in &q_t_conv_vars { for row in head { vars_2.extend(row.iter().cloned()); } }
        }
        // Allocate conv matmul output/remainder [B][H][S][D]
        let q_conv_vars = new_matrix_witness_4d(cs.clone(), &conv.q_conv);
        let r_conv_vars = new_matrix_witness_4d(cs.clone(), &conv.r_conv);
        // Allocate post-conv result [B][H][S][D]
        let q_xf_vars = new_matrix_witness_4d(cs.clone(), &conv.q_xf);
        let r_xf_vars = new_matrix_witness_4d(cs.clone(), &conv.r_xf);
        // d3: commit Q_conv and Q_xf (Freivalds-constrained intermediates)
        {
            let mut vars_aux = committed_vars_aux.borrow_mut();
            for b in &q_conv_vars { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
            for b in &q_xf_vars { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
        }
        // Derive attn_output_vars from Q_xf via reshape: [B][H][S][D] → [B][S][H*D]
        let attn_output: Vec<Vec<Vec<FpVar<F>>>> = {
            let num_h = q_xf_vars[0].len();
            let head_dim = q_xf_vars[0][0][0].len();
            q_xf_vars.iter().map(|batch| {
                let seq_len = batch[0].len();
                (0..seq_len).map(|s| {
                    (0..num_h * head_dim).map(|hd| {
                        batch[hd / head_dim][s][hd % head_dim].clone()
                    }).collect()
                }).collect()
            }).collect()
        };
        (Some(q_t_conv_vars), Some(q_conv_vars), Some(r_conv_vars), Some(q_xf_vars), Some(r_xf_vars), attn_output)
    } else {
        // No conv: bind attn_output_vars directly to x_data.q_sim reshape (like BERT)
        let attn_output: Vec<Vec<Vec<FpVar<F>>>> = {
            let num_h = x_data.q_sim[0].len();
            let head_dim = x_data.q_sim[0][0][0].len();
            x_data.q_sim.iter().map(|batch| {
                let seq_len = batch[0].len();
                (0..seq_len).map(|s| {
                    (0..num_h * head_dim).map(|hd| {
                        batch[hd / head_dim][s][hd % head_dim].clone()
                    }).collect()
                }).collect()
            }).collect()
        };
        (None, None, None, None, None, attn_output)
    };

    // Collector for Z* (q_inv_var) — allocated inside b/h loop, pushed to aux after loop closes
    let mut q_inv_committed: Vec<FpVar<F>> = Vec::new();

    let mut global_k1_lows = Vec::new(); let mut global_k1_highs = Vec::new();
    let mut global_k2_lows = Vec::new(); let mut global_k2_highs = Vec::new();
    let mut global_k3_lows = Vec::new(); let mut global_k3_highs = Vec::new();
    // Conv remainder pools (only used when conv exists)
    let mut global_conv_lows = Vec::new(); let mut global_conv_highs = Vec::new();
    let mut global_xf_lows = Vec::new(); let mut global_xf_highs = Vec::new();
    let mut global_sm1_idx = Vec::new();
    let mut global_sm1_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut global_sm2_idx = Vec::new();
    let mut global_sm2_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut global_sm3_idx = Vec::new();
    let mut global_sm3_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut global_inv0_lows = Vec::new(); let mut global_inv0_highs = Vec::new();
    let mut global_inv1_lows = Vec::new(); let mut global_inv1_highs = Vec::new();
    let mut global_inv2_lows = Vec::new(); let mut global_inv2_highs = Vec::new();
    let mut global_inv3_lows = Vec::new(); let mut global_inv3_highs = Vec::new();
    // Moore-Penrose bounded-difference pools (Lasso range check, C=1)
    let mut mp_cond1_pool: Vec<FpVar<F>> = Vec::new();
    let mut mp_cond3_pool: Vec<FpVar<F>> = Vec::new();
    let mut mp_cond4_pool: Vec<FpVar<F>> = Vec::new();
    let mut global_temp_lows = Vec::new(); let mut global_temp_highs = Vec::new();
    let mut global_mid_lows = Vec::new(); let mut global_mid_highs = Vec::new();
    let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();
    // Projection/scaling remainder accumulators
    let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
    let mut global_proj_k_lows = Vec::new(); let mut global_proj_k_highs = Vec::new();
    let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
    let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
    let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
    let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

    for b in 0..scale_q_out.len() {
        let k_landmarks = compute_circuit_landmarks(cs.clone(), &scale_k_out[b], &r_mean_k1[b], w.seg_size)?;
        let q_landmarks = compute_circuit_landmarks(cs.clone(), &scale_q_out[b], &r_mean_k2[b], w.seg_size)?;

        for h in 0..scale_q_out[b].len() {
            // === Q Projection: Q_raw_q = X_input @ W_q_h^T + b_q_h ===
            {
                let d_dim = w.proj_q.q_w_diff[h].len();
                let proj_q_params = ZenParams {
                    m: FpVar::Constant(w.proj_q.m_proj),
                    m_prime: FpVar::Constant(w.proj_q.m_prime_proj),
                    z_w: FpVar::Constant(w.z_x_input),
                    z_x: FpVar::zero(),
                    z_y: FpVar::zero(),
                    two_k: two_k_var.clone(),
                };
                let r_pows_pq = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &x_input_vars[b], &w_q_vars[h], &proj_q_out[b][h], &proj_q_r[b][h], &w.proj_q.r_out_proj[b][h], &proj_q_params, &r_pows_pq, &mut global_proj_q_lows, &mut global_proj_q_highs, Some(&w.proj_q.q_b_diff[h]))?;
            }
            // === K Projection ===
            {
                let d_dim = w.proj_k.q_w_diff[h].len();
                let proj_k_params = ZenParams {
                    m: FpVar::Constant(w.proj_k.m_proj),
                    m_prime: FpVar::Constant(w.proj_k.m_prime_proj),
                    z_w: FpVar::Constant(w.z_x_input),
                    z_x: FpVar::zero(),
                    z_y: FpVar::zero(),
                    two_k: two_k_var.clone(),
                };
                let r_pows_pk = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &x_input_vars[b], &w_k_vars[h], &proj_k_out[b][h], &proj_k_r[b][h], &w.proj_k.r_out_proj[b][h], &proj_k_params, &r_pows_pk, &mut global_proj_k_lows, &mut global_proj_k_highs, Some(&w.proj_k.q_b_diff[h]))?;
            }
            // === V Projection ===
            {
                let d_dim = w.proj_v.q_w_diff[h].len();
                let proj_v_params = ZenParams {
                    m: FpVar::Constant(w.proj_v.m_proj),
                    m_prime: FpVar::Constant(w.proj_v.m_prime_proj),
                    z_w: FpVar::Constant(w.z_x_input),
                    z_x: FpVar::zero(),
                    z_y: FpVar::zero(),
                    two_k: two_k_var.clone(),
                };
                let r_pows_pv = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &x_input_vars[b], &w_v_vars[h], &proj_v_out[b][h], &proj_v_r[b][h], &w.proj_v.r_out_proj[b][h], &proj_v_params, &r_pows_pv, &mut global_proj_v_lows, &mut global_proj_v_highs, Some(&w.proj_v.q_b_diff[h]))?;
            }
            // === Q Scaling: Q_proc = Q_raw_q / scale ===
            {
                let z_raw_q = FpVar::Constant(w.proj_q.z_raw);
                let m_scale_q = FpVar::Constant(w.scale_q.m_scale);
                let m_prime_q = FpVar::Constant(w.scale_q.m_prime_scale);
                enforce_scaling(cs.clone(), &proj_q_out[b][h], &scale_q_out[b][h], &scale_q_r[b][h], &w.scale_q.r_scale[b][h], &z_raw_q, &m_scale_q, &m_prime_q, two_k_var, &mut global_scale_q_lows, &mut global_scale_q_highs)?;
            }
            // === K Scaling ===
            {
                let z_raw_k = FpVar::Constant(w.proj_k.z_raw);
                let m_scale_k = FpVar::Constant(w.scale_k.m_scale);
                let m_prime_k = FpVar::Constant(w.scale_k.m_prime_scale);
                enforce_scaling(cs.clone(), &proj_k_out[b][h], &scale_k_out[b][h], &scale_k_r[b][h], &w.scale_k.r_scale[b][h], &z_raw_k, &m_scale_k, &m_prime_k, two_k_var, &mut global_scale_k_lows, &mut global_scale_k_highs)?;
            }

            let k_l_t = transpose_matrix(&k_landmarks[h]);
            let r_pows_k1 = gen_challenge_vector(r_var, k1_data.q_sim[b][h][0].len());

            // Kernel 1
            let mut h_k1_lows = Vec::new(); let mut h_k1_highs = Vec::new();
            enforce_freivalds(cs.clone(), &scale_q_out[b][h], &k_l_t, &k1_data.q_sim[b][h], &k1_data.r_matmul[b][h], &w.matmul1.r_matmul[b][h], &params_k1, &r_pows_k1, &mut h_k1_lows, &mut h_k1_highs, None)?;
            global_k1_lows.extend(h_k1_lows); global_k1_highs.extend(h_k1_highs);

            // Kernel 2
            let mut h_k2_lows = Vec::new(); let mut h_k2_highs = Vec::new();
            let q_l_h = &q_landmarks[h];
            let r_pows_k2 = gen_challenge_vector(r_var, k2_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), q_l_h, &k_l_t, &k2_data.q_sim[b][h], &k2_data.r_matmul[b][h], &w.matmul2.r_matmul[b][h], &params_k2, &r_pows_k2, &mut h_k2_lows, &mut h_k2_highs, None)?;
            global_k2_lows.extend(h_k2_lows); global_k2_highs.extend(h_k2_highs);

            // Kernel 3
            let mut h_k3_lows = Vec::new(); let mut h_k3_highs = Vec::new();
            let k_full_t = transpose_matrix(&scale_k_out[b][h]);
            let r_pows_k3 = gen_challenge_vector(r_var, k3_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &q_landmarks[h], &k_full_t, &k3_data.q_sim[b][h], &k3_data.r_matmul[b][h], &w.matmul3.r_matmul[b][h], &params_k3, &r_pows_k3, &mut h_k3_lows, &mut h_k3_highs, None)?;
            global_k3_lows.extend(h_k3_lows); global_k3_highs.extend(h_k3_highs);

            // Softmax 1
            let mut h_sm1_idx = Vec::new();
            let mut h_sm1_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            enforce_softmax(cs.clone(), &k1_data.q_sim[b][h], &k1_data.r_matmul[b][h], &sm1_vars.r_sm[b][h], &sm1_vars.m_sm, &sm1_vars.table_idx[b][h], &sm1_vars.exp_total[b][h], &sm1_vars.p[b][h], &sm1_vars.r_div[b][h], &w.sm1.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm1_idx, &mut h_sm1_r_sm, &mut r_div_k1_lows, &mut r_div_k1_highs)?;
            global_sm1_idx.extend(h_sm1_idx);
            for k in 0..4 { global_sm1_r_sm[k].extend(h_sm1_r_sm[k].clone()); }

            // Softmax 2
            let mut h_sm2_idx = Vec::new();
            let mut h_sm2_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            enforce_softmax(cs.clone(), &k2_data.q_sim[b][h], &k2_data.r_matmul[b][h], &sm2_vars.r_sm[b][h], &sm2_vars.m_sm, &sm2_vars.table_idx[b][h], &sm2_vars.exp_total[b][h], &sm2_vars.p[b][h], &sm2_vars.r_div[b][h], &w.sm2.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm2_idx, &mut h_sm2_r_sm, &mut r_div_k2_lows, &mut r_div_k2_highs)?;
            global_sm2_idx.extend(h_sm2_idx);
            for k in 0..4 { global_sm2_r_sm[k].extend(h_sm2_r_sm[k].clone()); }

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
            let mut h_sm3_idx = Vec::new();
            let mut h_sm3_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            enforce_softmax(cs.clone(), &masked_q_sim_k3, &k3_data.r_matmul[b][h], &sm3_vars.r_sm[b][h], &sm3_vars.m_sm, &sm3_vars.table_idx[b][h], &sm3_vars.exp_total[b][h], &sm3_vars.p[b][h], &sm3_vars.r_div[b][h], &w.sm3.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm3_idx, &mut h_sm3_r_sm, &mut r_div_k3_lows, &mut r_div_k3_highs)?;
            global_sm3_idx.extend(h_sm3_idx);
            for k in 0..4 { global_sm3_r_sm[k].extend(h_sm3_r_sm[k].clone()); }

            // Moore-Penrose pseudoinverse (4 steps)
            let q_inv_var = new_matrix_witness_4d(cs.clone(), &w.pseudo_inv.q_inv);
            // Collect Z* (q_inv_var) for d3 commitment — Freivalds-only constrained
            for row in &q_inv_var[b][h] { q_inv_committed.extend(row.iter().cloned()); }
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
            enforce_freivalds(cs.clone(), &sm2_vars.p[b][h], &q_inv_var[b][h], &q_y_vars[0][b][h], &r_inv_vars[0][b][h], &w.pseudo_inv.r_inv_remainders[0][b][h], &inv_params[0], &r_pows_inv, &mut h_inv0_lows, &mut h_inv0_highs, None)?;
            global_inv0_lows.extend(h_inv0_lows); global_inv0_highs.extend(h_inv0_highs);

            // Step 2
            let mut h_inv1_lows = Vec::new(); let mut h_inv1_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_y_vars[0][b][h], &sm2_vars.p[b][h], &q_y_vars[1][b][h], &r_inv_vars[1][b][h], &w.pseudo_inv.r_inv_remainders[1][b][h], &inv_params[1], &r_pows_inv, &mut h_inv1_lows, &mut h_inv1_highs, None)?;
            global_inv1_lows.extend(h_inv1_lows); global_inv1_highs.extend(h_inv1_highs);

            // Step 3
            let mut h_inv2_lows = Vec::new(); let mut h_inv2_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_inv_var[b][h], &sm2_vars.p[b][h], &q_y_vars[2][b][h], &r_inv_vars[2][b][h], &w.pseudo_inv.r_inv_remainders[2][b][h], &inv_params[2], &r_pows_inv, &mut h_inv2_lows, &mut h_inv2_highs, None)?;
            global_inv2_lows.extend(h_inv2_lows); global_inv2_highs.extend(h_inv2_highs);

            // Step 4
            let mut h_inv3_lows = Vec::new(); let mut h_inv3_highs = Vec::new();
            enforce_freivalds(cs.clone(), &q_y_vars[2][b][h], &q_inv_var[b][h], &q_y_vars[3][b][h], &r_inv_vars[3][b][h], &w.pseudo_inv.r_inv_remainders[3][b][h], &inv_params[3], &r_pows_inv, &mut h_inv3_lows, &mut h_inv3_highs, None)?;
            global_inv3_lows.extend(h_inv3_lows); global_inv3_highs.extend(h_inv3_highs);

            // ── Moore-Penrose condition enforcement (bounded-difference via Lasso) ──
            let m_dim = q_y_vars[0][b][h].len();

            // Condition 1: |Q_y2 - P_k2| < 4096 → shifted ∈ [0, 8192)
            for i in 0..m_dim {
                for j in 0..q_y_vars[1][b][h][i].len() {
                    let shifted_expr = &q_y_vars[1][b][h][i][j] - &sm2_vars.p[b][h][i][j]
                        + FpVar::Constant(F::from(4096u64));
                    let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                    shifted_expr.enforce_equal(&shifted)?;
                    mp_cond1_pool.push(shifted);
                }
            }

            // Condition 3: |Q_y1[i][j] - Q_y1[j][i]| < 512 → shifted ∈ [0, 1024)
            for i in 0..m_dim {
                for j in (i+1)..m_dim {
                    let shifted_expr = &q_y_vars[0][b][h][i][j] - &q_y_vars[0][b][h][j][i]
                        + FpVar::Constant(F::from(512u64));
                    let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                    shifted_expr.enforce_equal(&shifted)?;
                    mp_cond3_pool.push(shifted);
                }
            }

            // Condition 4: |Q_y3[i][j] - Q_y3[j][i]| < 32 → shifted ∈ [0, 64)
            for i in 0..m_dim {
                for j in (i+1)..m_dim {
                    let shifted_expr = &q_y_vars[2][b][h][i][j] - &q_y_vars[2][b][h][j][i]
                        + FpVar::Constant(F::from(32u64));
                    let shifted = FpVar::new_witness(cs.clone(), || shifted_expr.value())?;
                    shifted_expr.enforce_equal(&shifted)?;
                    mp_cond4_pool.push(shifted);
                }
            }

            // temp matmul
            let mut h_temp_lows = Vec::new(); let mut h_temp_highs = Vec::new();
            let r_pows_temp = gen_challenge_vector(r_var, temp_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &sm3_vars.p[b][h], &proj_v_out[b][h], &temp_data.q_sim[b][h], &temp_data.r_matmul[b][h], &w.matmul_temp.r_matmul[b][h], &params_temp, &r_pows_temp, &mut h_temp_lows, &mut h_temp_highs, None)?;
            global_temp_lows.extend(h_temp_lows); global_temp_highs.extend(h_temp_highs);

            // mid matmul
            let mut h_mid_lows = Vec::new(); let mut h_mid_highs = Vec::new();
            let r_pows_mid = gen_challenge_vector(r_var, mid_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &sm1_vars.p[b][h], &q_inv_var[b][h], &mid_data.q_sim[b][h], &mid_data.r_matmul[b][h], &w.matmul_mid.r_matmul[b][h], &params_mid, &r_pows_mid, &mut h_mid_lows, &mut h_mid_highs, None)?;
            global_mid_lows.extend(h_mid_lows); global_mid_highs.extend(h_mid_highs);

            // X matmul
            let mut h_x_lows = Vec::new(); let mut h_x_highs = Vec::new();
            let r_pows_x = gen_challenge_vector(r_var, x_data.q_sim[b][h][0].len());
            enforce_freivalds(cs.clone(), &mid_data.q_sim[b][h], &temp_data.q_sim[b][h], &x_data.q_sim[b][h], &x_data.r_matmul[b][h], &w.matmul_x.r_matmul[b][h], &params_x, &r_pows_x, &mut h_x_lows, &mut h_x_highs, None)?;
            global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);

            // === Conv: Q_conv = Toeplitz(W_conv_h) @ V_h, then rescaled addition ===
            if let Some(ref conv) = w.conv {
                let q_t_h = q_t_conv_vars.as_ref().unwrap();
                let q_conv_v = q_conv_vars.as_ref().unwrap();
                let r_conv_v = r_conv_vars.as_ref().unwrap();
                let q_xf_v = q_xf_vars.as_ref().unwrap();
                let r_xf_v = r_xf_vars.as_ref().unwrap();

                // Conv Freivalds: T_h[S×S] @ V[S×D] = Q_conv[S×D]
                let conv_params = ZenParams {
                    m: FpVar::Constant(conv.m_conv),
                    m_prime: FpVar::Constant(conv.m_prime_conv),
                    z_w: FpVar::Constant(conv.z_t_conv),
                    z_x: FpVar::Constant(w.z_v),
                    z_y: FpVar::Constant(conv.z_conv_out),
                    two_k: two_k_var.clone(),
                };
                let r_pows_conv = gen_challenge_vector(r_var, q_conv_v[b][h][0].len());
                let mut h_conv_lows = Vec::new(); let mut h_conv_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(), &q_t_h[h], &proj_v_out[b][h],
                    &q_conv_v[b][h], &r_conv_v[b][h],
                    &conv.r_conv[b][h],
                    &conv_params, &r_pows_conv,
                    &mut h_conv_lows, &mut h_conv_highs,
                    None,
                )?;
                global_conv_lows.extend(h_conv_lows); global_conv_highs.extend(h_conv_highs);

                // Rescaled addition: Q_xf * 2^k + R_xf = M_add_x*(Ỹ - z_x) + M_add_conv*(Q_conv - z_conv) + z_xf*2^k
                let m_add_x_const = FpVar::Constant(conv.m_add_x);
                let m_add_conv_const = FpVar::Constant(conv.m_add_conv);
                let z_sim_x_const = FpVar::Constant(w.matmul_x.z_sim);
                let z_conv_out_const = FpVar::Constant(conv.z_conv_out);
                let z_xf_two_k = FpVar::Constant(conv.z_xf) * two_k_var;
                for s in 0..q_xf_v[b][h].len() {
                    for d in 0..q_xf_v[b][h][s].len() {
                        let lhs = &q_xf_v[b][h][s][d] * two_k_var + &r_xf_v[b][h][s][d];
                        let rhs = &m_add_x_const * (&x_data.q_sim[b][h][s][d] - &z_sim_x_const)
                                + &m_add_conv_const * (&q_conv_v[b][h][s][d] - &z_conv_out_const)
                                + &z_xf_two_k;
                        lhs.enforce_equal(&rhs)?;

                        // Range check R_xf
                        let (low, high) = split_remainder_and_push(
                            cs.clone(), &r_xf_v[b][h][s][d], conv.r_xf[b][h][s][d]
                        )?;
                        global_xf_lows.push(low);
                        global_xf_highs.push(high);
                    }
                }
            }

            // === O Projection: O_h = attn_output @ W_o_h^T + b_o_h ===
            // Uses attn_output_vars (conv-bound Q_xf or x_data reshape) as input.
            {
                let o_input = &attn_output_vars[b];

                let w_o_h_t = &w_o_vars[h];
                let d_dim = w.proj_o.q_w_diff[h].len();
                let proj_o_params = ZenParams {
                    m: FpVar::Constant(w.proj_o.m_proj),
                    m_prime: FpVar::Constant(w.proj_o.m_prime_proj),
                    z_w: FpVar::Constant(w.z_attn_output),
                    z_x: FpVar::zero(),
                    z_y: FpVar::zero(),
                    two_k: two_k_var.clone(),
                };
                let r_pows_po = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &o_input, w_o_h_t, &proj_o_out[b][h], &proj_o_r[b][h], &w.proj_o.r_out_proj[b][h], &proj_o_params, &r_pows_po, &mut global_proj_o_lows, &mut global_proj_o_highs, Some(&w.proj_o.q_b_diff[h]))?;
            }
        }
    }

    // d3: commit Z* (q_inv_var) — collected during b/h loop, pushed after loop closes
    {
        committed_vars_aux.borrow_mut().extend(q_inv_committed);
    }

    // Push to committed_vars in CP-Link ordering
    {
        let mut final_vars = committed_vars.borrow_mut();

        // === Softmax index lookup vars (C=1, single pool) ===
        final_vars.extend(global_sm1_idx);
        final_vars.extend(global_sm2_idx);
        final_vars.extend(global_sm3_idx);

        // === 26-bit remainders (C=2: 16+10) ===
        // Order must match run_all_remainders_with_link target_26bit
        final_vars.extend(global_k1_lows); final_vars.extend(global_k1_highs);
        final_vars.extend(global_k2_lows); final_vars.extend(global_k2_highs);
        final_vars.extend(global_k3_lows); final_vars.extend(global_k3_highs);
        // Moore-Penrose Steps
        final_vars.extend(global_inv0_lows); final_vars.extend(global_inv0_highs);
        final_vars.extend(global_inv1_lows); final_vars.extend(global_inv1_highs);
        final_vars.extend(global_inv2_lows); final_vars.extend(global_inv2_highs);
        final_vars.extend(global_inv3_lows); final_vars.extend(global_inv3_highs);
        // Final MatMuls
        final_vars.extend(global_temp_lows); final_vars.extend(global_temp_highs);
        final_vars.extend(global_mid_lows); final_vars.extend(global_mid_highs);
        final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
        // Conv remainders (only non-empty when conv exists)
        final_vars.extend(global_conv_lows); final_vars.extend(global_conv_highs);
        final_vars.extend(global_xf_lows); final_vars.extend(global_xf_highs);
        // Projection/scaling remainders
        final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
        final_vars.extend(global_proj_k_lows); final_vars.extend(global_proj_k_highs);
        final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
        final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);
        final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
        final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

        // === 52-bit r_sm (C=4: 16+16+16+4) ===
        for k in 0..4 { final_vars.extend(global_sm1_r_sm[k].clone()); }
        for k in 0..4 { final_vars.extend(global_sm2_r_sm[k].clone()); }
        for k in 0..4 { final_vars.extend(global_sm3_r_sm[k].clone()); }

        // === R_div per-softmax (26-bit C=2: 16+10) ===
        // Order must match Lasso results: sm1[low,high], sm2[low,high], sm3[low,high]
        final_vars.extend(r_div_k1_lows);
        final_vars.extend(r_div_k1_highs);
        final_vars.extend(r_div_k2_lows);
        final_vars.extend(r_div_k2_highs);
        final_vars.extend(r_div_k3_lows);
        final_vars.extend(r_div_k3_highs);

        // Moore-Penrose bounded-difference pools (C=1, Lasso range check)
        final_vars.extend(mp_cond1_pool);
        final_vars.extend(mp_cond3_pool);
        final_vars.extend(mp_cond4_pool);
    }

    // === Poseidon hash of x_input → public input ===
    {
        use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
        use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();
        let declared_input_hash = FpVar::new_input(cs.clone(), || Ok(w.input_hash))?;

        let mut x_flat: Vec<FpVar<F>> = Vec::new();
        for batch in &x_input_vars {
            for seq in batch {
                x_flat.extend(seq.iter().cloned());
            }
        }
        let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
        sponge_var.absorb(&x_flat)?;
        let computed_hash = sponge_var.squeeze_field_elements(1)?;
        computed_hash[0].enforce_equal(&declared_input_hash)?;
    }

    // === Poseidon hash of output → public input ===
    {
        use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
        use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();

        let declared_hash = FpVar::new_input(cs.clone(), || Ok(w.output_hash))?;

        let mut output_flat: Vec<FpVar<F>> = Vec::new();
        for batch in &proj_o_out {
            for head in batch {
                for seq in head {
                    output_flat.extend(seq.iter().cloned());
                }
            }
        }
        let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
        sponge_var.absorb(&output_flat)?;
        let computed_hash = sponge_var.squeeze_field_elements(1)?;

        computed_hash[0].enforce_equal(&declared_hash)?;
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
    committed_vars_aux: &Rc<RefCell<Vec<FpVar<F>>>>,
) -> Result<(), SynthesisError> {
    let mut r_div_lows = Vec::new();
    let mut r_div_highs = Vec::new();

    // === Projection input allocation (witness for x_input) ===
    let x_input_vars: Vec<Vec<Vec<FpVar<F>>>> = w.q_x_input.iter().map(|batch|
        batch.iter().map(|seq| seq.iter().map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)).unwrap()).collect()).collect()
    ).collect();
    let proj_q_out = new_matrix_witness_4d(cs.clone(), &w.proj_q.q_out_proj);
    let proj_q_r = new_matrix_witness_4d(cs.clone(), &w.proj_q.r_out_proj);
    let proj_k_out = new_matrix_witness_4d(cs.clone(), &w.proj_k.q_out_proj);
    let proj_k_r = new_matrix_witness_4d(cs.clone(), &w.proj_k.r_out_proj);
    let proj_v_out = new_matrix_witness_4d(cs.clone(), &w.proj_v.q_out_proj);
    let proj_v_r = new_matrix_witness_4d(cs.clone(), &w.proj_v.r_out_proj);
    let scale_q_out = new_matrix_witness_4d(cs.clone(), &w.scale_q.q_scaled);
    let scale_q_r = new_matrix_witness_4d(cs.clone(), &w.scale_q.r_scale);
    let scale_k_out = new_matrix_witness_4d(cs.clone(), &w.scale_k.q_scaled);
    let scale_k_r = new_matrix_witness_4d(cs.clone(), &w.scale_k.r_scale);
    let proj_o_out = new_matrix_witness_4d(cs.clone(), &w.proj_o.q_out_proj);
    let proj_o_r = new_matrix_witness_4d(cs.clone(), &w.proj_o.r_out_proj);

    // d3: commit Freivalds-constrained intermediates (Q, K, V)
    {
        let mut vars_aux = committed_vars_aux.borrow_mut();
        for b in &proj_q_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
        for b in &proj_k_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
        for b in &proj_v_out { for h in b { for s in h { vars_aux.extend(s.iter().cloned()); } } }
    }

    // Allocate W_Q, W_K, W_V weights as witnesses (transposed form: [H][HD][D])
    let num_heads = w.proj_q.q_w_diff.len();
    let w_q_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_q.q_w_diff[h].len();
        let hd_dim = w.proj_q.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_q.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();
    let w_k_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_k.q_w_diff[h].len();
        let hd_dim = w.proj_k.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_k.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();
    let w_v_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_v.q_w_diff[h].len();
        let hd_dim = w.proj_v.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_v.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();

    let w_o_vars: Vec<Vec<Vec<FpVar<F>>>> = (0..num_heads).map(|h| {
        let d_dim = w.proj_o.q_w_diff[h].len();
        let hd_dim = w.proj_o.q_w_diff[h][0].len();
        (0..hd_dim).map(|hd| {
            (0..d_dim).map(|d| FpVar::new_witness(cs.clone(), || Ok(w.proj_o.q_w_diff[h][d][hd])).unwrap()).collect()
        }).collect()
    }).collect();

    // d2: commit W_Q, W_K, W_V, W_O weights
    {
        let mut vars_2 = committed_vars_2.borrow_mut();
        for h_w in &w_q_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        for h_w in &w_k_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        for h_w in &w_v_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
        for h_w in &w_o_vars { for row in h_w { vars_2.extend(row.iter().cloned()); } }
    }

    let (attn_data, params_attn) = w.matmul_attn.alloc_and_setup(cs.clone(), w.z_q, w.z_k, two_k_var);
    let (x_data, params_x) = w.matmul_x.alloc_and_setup(cs.clone(), F::zero(), w.z_v, two_k_var);

    // d3: commit chain output x=P*V [B][H][S][D] — Freivalds-only constrained
    {
        let mut vars_aux = committed_vars_aux.borrow_mut();
        for b in &x_data.q_sim { for h in b { for row in h { vars_aux.extend(row.iter().cloned()); } } }
    }

    // O-proj input: rearranged reference to x_data.q_sim (conv = arithmetic-free reshape)
    // Y[b][s][h*D+d] = Ỹ[b][h][s][d] — shares same R1CS variables, no separate witness
    let attn_output_vars: Vec<Vec<Vec<FpVar<F>>>> = {
        let num_h = x_data.q_sim[0].len();
        let head_dim = x_data.q_sim[0][0][0].len();
        x_data.q_sim.iter().map(|batch| {
            let seq_len = batch[0].len();
            (0..seq_len).map(|s| {
                (0..num_h * head_dim).map(|hd| {
                    batch[hd / head_dim][s][hd % head_dim].clone()
                }).collect()
            }).collect()
        }).collect()
    };

    let mask_var = new_matrix_witness(cs.clone(), &w.mask);
    let sm_vars = w.sm.alloc_vars(cs.clone())?;

    let mut global_attn_lows = Vec::new(); let mut global_attn_highs = Vec::new();
    let mut global_sm_idx = Vec::new();
    let mut global_sm_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut global_x_lows = Vec::new(); let mut global_x_highs = Vec::new();
    // Projection/scaling remainder accumulators
    let mut global_proj_q_lows = Vec::new(); let mut global_proj_q_highs = Vec::new();
    let mut global_proj_k_lows = Vec::new(); let mut global_proj_k_highs = Vec::new();
    let mut global_proj_v_lows = Vec::new(); let mut global_proj_v_highs = Vec::new();
    let mut global_proj_o_lows = Vec::new(); let mut global_proj_o_highs = Vec::new();
    let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
    let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();
    let zero_var = FpVar::zero();

    for b in 0..scale_q_out.len() {
        for h in 0..scale_q_out[b].len() {
            // === Q Projection ===
            {
                let d_dim = w.proj_q.q_w_diff[h].len();
                let proj_q_params = ZenParams {
                    m: FpVar::Constant(w.proj_q.m_proj), m_prime: FpVar::Constant(w.proj_q.m_prime_proj),
                    z_w: FpVar::Constant(w.z_x_input), z_x: FpVar::zero(), z_y: FpVar::zero(), two_k: two_k_var.clone(),
                };
                let r_pows_pq = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &x_input_vars[b], &w_q_vars[h], &proj_q_out[b][h], &proj_q_r[b][h], &w.proj_q.r_out_proj[b][h], &proj_q_params, &r_pows_pq, &mut global_proj_q_lows, &mut global_proj_q_highs, Some(&w.proj_q.q_b_diff[h]))?;
            }
            // === K Projection ===
            {
                let d_dim = w.proj_k.q_w_diff[h].len();
                let proj_k_params = ZenParams {
                    m: FpVar::Constant(w.proj_k.m_proj), m_prime: FpVar::Constant(w.proj_k.m_prime_proj),
                    z_w: FpVar::Constant(w.z_x_input), z_x: FpVar::zero(), z_y: FpVar::zero(), two_k: two_k_var.clone(),
                };
                let r_pows_pk = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &x_input_vars[b], &w_k_vars[h], &proj_k_out[b][h], &proj_k_r[b][h], &w.proj_k.r_out_proj[b][h], &proj_k_params, &r_pows_pk, &mut global_proj_k_lows, &mut global_proj_k_highs, Some(&w.proj_k.q_b_diff[h]))?;
            }
            // === V Projection ===
            {
                let d_dim = w.proj_v.q_w_diff[h].len();
                let proj_v_params = ZenParams {
                    m: FpVar::Constant(w.proj_v.m_proj), m_prime: FpVar::Constant(w.proj_v.m_prime_proj),
                    z_w: FpVar::Constant(w.z_x_input), z_x: FpVar::zero(), z_y: FpVar::zero(), two_k: two_k_var.clone(),
                };
                let r_pows_pv = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &x_input_vars[b], &w_v_vars[h], &proj_v_out[b][h], &proj_v_r[b][h], &w.proj_v.r_out_proj[b][h], &proj_v_params, &r_pows_pv, &mut global_proj_v_lows, &mut global_proj_v_highs, Some(&w.proj_v.q_b_diff[h]))?;
            }
            // === Q Scaling ===
            {
                let z_raw_q = FpVar::Constant(w.proj_q.z_raw);
                let m_scale_q = FpVar::Constant(w.scale_q.m_scale);
                let m_prime_q = FpVar::Constant(w.scale_q.m_prime_scale);
                enforce_scaling(cs.clone(), &proj_q_out[b][h], &scale_q_out[b][h], &scale_q_r[b][h], &w.scale_q.r_scale[b][h], &z_raw_q, &m_scale_q, &m_prime_q, two_k_var, &mut global_scale_q_lows, &mut global_scale_q_highs)?;
            }
            // === K Scaling ===
            {
                let z_raw_k = FpVar::Constant(w.proj_k.z_raw);
                let m_scale_k = FpVar::Constant(w.scale_k.m_scale);
                let m_prime_k = FpVar::Constant(w.scale_k.m_prime_scale);
                enforce_scaling(cs.clone(), &proj_k_out[b][h], &scale_k_out[b][h], &scale_k_r[b][h], &w.scale_k.r_scale[b][h], &z_raw_k, &m_scale_k, &m_prime_k, two_k_var, &mut global_scale_k_lows, &mut global_scale_k_highs)?;
            }

            // QK^T
            let k_t = transpose_matrix(&scale_k_out[b][h]);
            let r_pows_attn = gen_challenge_vector(r_var, attn_data.q_sim[b][h][0].len());
            let mut h_attn_lows = Vec::new(); let mut h_attn_highs = Vec::new();
            enforce_freivalds(cs.clone(), &scale_q_out[b][h], &k_t, &attn_data.q_sim[b][h], &attn_data.r_matmul[b][h], &w.matmul_attn.r_matmul[b][h], &params_attn, &r_pows_attn, &mut h_attn_lows, &mut h_attn_highs, None)?;
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
            let mut h_sm_idx = Vec::new();
            let mut h_sm_r_sm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            enforce_softmax(cs.clone(), &masked_q_sim, &attn_data.r_matmul[b][h], &sm_vars.r_sm[b][h], &sm_vars.m_sm, &sm_vars.table_idx[b][h], &sm_vars.exp_total[b][h], &sm_vars.p[b][h], &sm_vars.r_div[b][h], &w.sm.r_div[b][h], w.frac_bits, two_k_var, &mut h_sm_idx, &mut h_sm_r_sm, &mut r_div_lows, &mut r_div_highs)?;
            global_sm_idx.extend(h_sm_idx);
            for k in 0..4 { global_sm_r_sm[k].extend(h_sm_r_sm[k].clone()); }

            // P * V
            let r_pows_x = gen_challenge_vector(r_var, x_data.q_sim[b][h][0].len());
            let mut h_x_lows = Vec::new(); let mut h_x_highs = Vec::new();
            enforce_freivalds(cs.clone(), &sm_vars.p[b][h], &proj_v_out[b][h], &x_data.q_sim[b][h], &x_data.r_matmul[b][h], &w.matmul_x.r_matmul[b][h], &params_x, &r_pows_x, &mut h_x_lows, &mut h_x_highs, None)?;
            global_x_lows.extend(h_x_lows); global_x_highs.extend(h_x_highs);

            // === O Projection ===
            // Uses q_attn_output witness (post-conv if applicable) as input.
            {
                let o_input = &attn_output_vars[b];

                let w_o_h_t = &w_o_vars[h];
                let d_dim = w.proj_o.q_w_diff[h].len();
                let proj_o_params = ZenParams {
                    m: FpVar::Constant(w.proj_o.m_proj), m_prime: FpVar::Constant(w.proj_o.m_prime_proj),
                    z_w: FpVar::Constant(w.z_attn_output), z_x: FpVar::zero(), z_y: FpVar::zero(), two_k: two_k_var.clone(),
                };
                let r_pows_po = gen_challenge_vector(r_var, d_dim);
                enforce_freivalds(cs.clone(), &o_input, w_o_h_t, &proj_o_out[b][h], &proj_o_r[b][h], &w.proj_o.r_out_proj[b][h], &proj_o_params, &r_pows_po, &mut global_proj_o_lows, &mut global_proj_o_highs, Some(&w.proj_o.q_b_diff[h]))?;
            }
        }
    }

    // Push to committed_vars in CP-Link ordering
    {
        let mut final_vars = committed_vars.borrow_mut();

        // === Softmax index lookup vars (C=1, single pool) ===
        final_vars.extend(global_sm_idx);

        // === 26-bit remainders (C=2: 16+10) ===
        // Order must match run_all_bert_remainders_with_link target_26bit
        final_vars.extend(global_attn_lows); final_vars.extend(global_attn_highs);
        final_vars.extend(global_x_lows); final_vars.extend(global_x_highs);
        // Projection/scaling remainders
        final_vars.extend(global_proj_q_lows); final_vars.extend(global_proj_q_highs);
        final_vars.extend(global_proj_k_lows); final_vars.extend(global_proj_k_highs);
        final_vars.extend(global_proj_v_lows); final_vars.extend(global_proj_v_highs);
        final_vars.extend(global_proj_o_lows); final_vars.extend(global_proj_o_highs);
        final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
        final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

        // === 52-bit r_sm (C=4: 16+16+16+4) ===
        for k in 0..4 { final_vars.extend(global_sm_r_sm[k].clone()); }

        // === R_div per-softmax (26-bit C=2: 16+10) ===
        final_vars.extend(r_div_lows);
        final_vars.extend(r_div_highs);
    }

    // === Poseidon hash of x_input → public input ===
    {
        use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
        use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();
        let declared_input_hash = FpVar::new_input(cs.clone(), || Ok(w.input_hash))?;

        let mut x_flat: Vec<FpVar<F>> = Vec::new();
        for batch in &x_input_vars {
            for seq in batch {
                x_flat.extend(seq.iter().cloned());
            }
        }
        let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
        sponge_var.absorb(&x_flat)?;
        let computed_hash = sponge_var.squeeze_field_elements(1)?;
        computed_hash[0].enforce_equal(&declared_input_hash)?;
    }

    // === Poseidon hash of output → public input ===
    {
        use ark_crypto_primitives::sponge::poseidon::constraints::PoseidonSpongeVar;
        use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;

        let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<F>();

        let declared_hash = FpVar::new_input(cs.clone(), || Ok(w.output_hash))?;

        let mut output_flat: Vec<FpVar<F>> = Vec::new();
        for batch in &proj_o_out {
            for head in batch {
                for seq in head {
                    output_flat.extend(seq.iter().cloned());
                }
            }
        }
        let mut sponge_var = PoseidonSpongeVar::new(cs.clone(), &poseidon_config);
        sponge_var.absorb(&output_flat)?;
        let computed_hash = sponge_var.squeeze_field_elements(1)?;

        computed_hash[0].enforce_equal(&declared_hash)?;
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
    /// d3: Freivalds-constrained intermediates (Q, K, V, Z*, temp, mid, x) —
    /// committed as cm_aux and included in FS hash v3.
    pub committed_vars_aux: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for ModelIntegratedCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();
        self.committed_vars_aux.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        for layer_idx in 0..self.attn_layers.len() {
            // Attention constraints
            match &self.attn_layers[layer_idx] {
                AttnWitness::Nystrom(w) => {
                    generate_nystrom_attn_constraints(
                        cs.clone(), w, &r_var, &two_k_var,
                        &self.committed_vars, &self.committed_vars_2,
                        &self.committed_vars_aux,
                    )?;
                }
                AttnWitness::Bert(w) => {
                    generate_bert_attn_constraints(
                        cs.clone(), w, &r_var, &two_k_var,
                        &self.committed_vars, &self.committed_vars_2,
                        &self.committed_vars_aux,
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
        committed_vars_aux: Rc::new(RefCell::new(Vec::new())),
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

    // Dry-run → d1/d2/d3 indices
    let (d1_indices, d2_indices, d3_indices) = {
        let temp = load_model_integrated_circuit::<E>(&discovered)?;
        let v1 = Rc::clone(&temp.committed_vars);
        let v2 = Rc::clone(&temp.committed_vars_2);
        let v3 = Rc::clone(&temp.committed_vars_aux);
        let cs = ConstraintSystem::<Fr>::new_ref();
        temp.generate_constraints(cs)?;
        (
            crate::util::utils::extract_witness_indices(&v1.borrow()),
            crate::util::utils::extract_witness_indices(&v2.borrow()),
            crate::util::utils::extract_witness_indices(&v3.borrow()),
        )
    };
    let num_d1 = d1_indices.len() as u32;
    let num_d2 = d2_indices.len() as u32;
    let num_d3 = d3_indices.len() as u32;
    println!("Commit counts: d1={}, d2={}, d3={}", num_d1, num_d2, num_d3);

    let circuit = load_model_integrated_circuit::<E>(&discovered)?;

    // [0] Setup
    println!("=== [0] LegoGroth16 Setup ===");
    let setup_start = Instant::now();
    let params = generate_random_parameters::<E, _, _>(circuit.clone(), &d1_indices, &d2_indices, &d3_indices, &mut rng)
        .map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
    let setup_time = setup_start.elapsed();
    let pvk = prepare_verifying_key::<E>(&params.vk);

    // Witness extraction
    let cs_witness = ConstraintSystem::<Fr>::new_ref();
    circuit.clone().generate_constraints(cs_witness.clone())?;
    let num_constraints = cs_witness.num_constraints();

    let committed_group1: Vec<Fr> = circuit.committed_vars.borrow().iter()
        .map(|v| v.value().unwrap_or(Fr::zero())).collect();
    let committed_group2: Vec<Fr> = circuit.committed_vars_2.borrow().iter()
        .map(|v| v.value().unwrap_or(Fr::zero())).collect();
    let committed_aux: Vec<Fr> = circuit.committed_vars_aux.borrow().iter()
        .map(|v| v.value().unwrap_or(Fr::zero())).collect();
    println!("[Debug] committed_group2 total={}, committed_aux total={}", committed_group2.len(), committed_aux.len());
    let num_pub = cs_witness.num_instance_variables() - 1;

    // Groth16 commit + Fiat-Shamir
    let v1 = Fr::rand(&mut rng);
    let v2 = Fr::rand(&mut rng);
    let v3 = Fr::rand(&mut rng);

    let g16_commit_start = Instant::now();

    let cm1_start = Instant::now();
    let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
    let cm1_ms = cm1_start.elapsed().as_millis();

    let cm2_start = Instant::now();
    let cm2 = cm2_from_vk::<E>(&params.vk, num_pub, &committed_group2, &v2);
    let cm2_ms = cm2_start.elapsed().as_millis();

    let cm_aux_start = Instant::now();
    let cm_aux = cm_aux_from_vk::<E>(&params.vk, num_pub, &committed_aux, &v3);
    let cm_aux_ms = cm_aux_start.elapsed().as_millis();

    let g16_commit_ms = g16_commit_start.elapsed().as_millis();

    // ── Global weight commitment c_W ─────────────────────────────────────
    let global_commit_start = Instant::now();
    let ck_glob = generate_global_ck::<E, _>(committed_group2.len(), &mut rng);
    let o_glob = Fr::rand(&mut rng);
    let c_w = commit_global::<E>(&ck_glob, &committed_group2, &o_glob);
    let global_commit_ms = global_commit_start.elapsed().as_millis();

    // ── Compute Poseidon hashes for each attention layer ─────────────────────
    let poseidon_config = crate::gadgets::poseidon_helper::default_poseidon_config::<Fr>();
    let mut circuit_with_hashes = circuit.clone();
    for attn_w in &mut circuit_with_hashes.attn_layers {
        // input_hash = Poseidon(flatten(q_x_input))
        {
            let x_input = match attn_w {
                AttnWitness::Nystrom(w) => &w.q_x_input,
                AttnWitness::Bert(w) => &w.q_x_input,
            };
            let mut x_flat: Vec<Fr> = Vec::new();
            for batch in x_input {
                for seq in batch {
                    x_flat.extend(seq.iter());
                }
            }
            let ih = crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &x_flat);
            match attn_w {
                AttnWitness::Nystrom(w) => w.input_hash = ih,
                AttnWitness::Bert(w) => w.input_hash = ih,
            }
        }
        // output_hash = Poseidon(flatten(proj_o_out))
        let proj_o_out = match attn_w {
            AttnWitness::Nystrom(w) => &w.proj_o.q_out_proj,
            AttnWitness::Bert(w) => &w.proj_o.q_out_proj,
        };
        let mut output_flat: Vec<Fr> = Vec::new();
        for batch in proj_o_out {
            for head in batch {
                for seq in head {
                    output_flat.extend(seq.iter());
                }
            }
        }
        let h = crate::gadgets::poseidon_helper::poseidon_hash(&poseidon_config, &output_flat);
        match attn_w {
            AttnWitness::Nystrom(w) => w.output_hash = h,
            AttnWitness::Bert(w) => w.output_hash = h,
        }
    }

    // ── Multi-commitment Fiat-Shamir challenge (v3: includes cm_aux) ────
    // Use first layer's hashes for the Fiat-Shamir challenge
    let (input_hash, output_hash) = match &circuit_with_hashes.attn_layers[0] {
        AttnWitness::Nystrom(w) => (w.input_hash, w.output_hash),
        AttnWitness::Bert(w) => (w.input_hash, w.output_hash),
    };
    let r_fs = hash_commitments_to_field_v3::<E>(&c_w, &cm1, &cm2, &cm_aux, &input_hash, &output_hash);

    let final_circuit = ModelIntegratedCircuit {
        r_challenge: r_fs,
        ..circuit_with_hashes
    };

    // [1] Groth16 Prove
    println!("=== [1] LegoGroth16 Prove ===");
    let g16_prove_start = Instant::now();
    let proof = create_random_proof::<E, _, _>(final_circuit.clone(), v1, v2, v3, &params, &mut rng)
        .map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
    let g16_prove_time = g16_prove_start.elapsed();

    let mut proof_bytes = Vec::new();
    proof.serialize_uncompressed(&mut proof_bytes)
        .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
    let proof_size_bytes = proof_bytes.len();

    // [1] Groth16 Verify
    let g16_verify_start = Instant::now();
    let mut public_inputs = vec![r_fs];
    for attn_w in &final_circuit.attn_layers {
        let (input_hash, output_hash) = match attn_w {
            AttnWitness::Nystrom(w) => (w.input_hash, w.output_hash),
            AttnWitness::Bert(w) => (w.input_hash, w.output_hash),
        };
        public_inputs.push(input_hash);
        public_inputs.push(output_hash);
    }
    let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
    let g16_verify_time = g16_verify_start.elapsed();
    println!("  - Model Integrated SNARK Verification: {:?}", result);
    if let Err(e) = result {
        return Err(anyhow::anyhow!(format!("{:?}", e)));
    }

    // ── Log circuit output (proj_o_out) per layer ────────────────────────
    for (layer_idx, attn_w) in final_circuit.attn_layers.iter().enumerate() {
        let (proj_o_out, output_hash) = match attn_w {
            AttnWitness::Nystrom(w) => (&w.proj_o.q_out_proj, w.output_hash),
            AttnWitness::Bert(w) => (&w.proj_o.q_out_proj, w.output_hash),
        };
        /*println!("\n=== Circuit Output (proj_o_out) Layer {} ===", layer_idx);
        for (b, batch) in proj_o_out.iter().enumerate() {
            for (h, head) in batch.iter().enumerate() {
                println!("  [batch={}, head={}] shape=[{}][{}]", b, h, head.len(), head[0].len());
                for (s, seq) in head.iter().enumerate() {
                    let vals: Vec<_> = seq.iter()
                        .map(|v| v.into_bigint())
                        .collect();
                    println!("    seq[{}]: {:?}", s, vals);
                }
            }
        }*/
        println!("  output_hash = {:?}", output_hash.into_bigint());
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

    // ── [3] CP-Link #1: d1 ↔ Lasso (setup/keygen, prove, verify) ────────
    println!("=== [3] CP-Link #1: d1 ↔ Lasso ===");
    let u_padded = pad_lasso_u_values::<E>(&lasso_results, &committed_group1);

    let cpl_setup_start = Instant::now();
    let (pp_lasso, crs_lasso) = factored_link_setup::<E, _>(&params.vk, num_pub, committed_group1.len(), &lasso_results, &mut rng);
    let (ek_lasso, vk_link_lasso) = factored_link_keygen::<E, _>(&pp_lasso, crs_lasso, &mut rng);
    let cpl_setup_ms = cpl_setup_start.elapsed().as_millis();

    let cpl_prove_start = Instant::now();
    let (proof_link_lasso, instance_lasso) = factored_link_prove::<E, _>(&pp_lasso, &ek_lasso, v1, cm1, &lasso_results, u_padded, &mut rng);
    let cpl_prove_ms = cpl_prove_start.elapsed().as_millis();

    let cpl_verify_start = Instant::now();
    if !factored_link_verify::<E>(&pp_lasso, &vk_link_lasso, &instance_lasso, &proof_link_lasso) {
        return Err(anyhow::anyhow!("CP-Link #1 (d1 ↔ Lasso) verification failed"));
    }
    println!("  -> [Success] CP-Link #1 (d1 ↔ Lasso) verified");
    let cpl_verify_ms = cpl_verify_start.elapsed().as_millis();

    let cplink_lasso_report = proof_size::compute_cplink_report::<E>(&proof_link_lasso, &instance_lasso);
    cplink_lasso_report.print_log();

    // ── CP-Link #2: c_W ↔ d2 (global weight commitment link) ────────
    let cpg_setup_start = Instant::now();
    let (pp_glob, crs_glob) = global_link_setup::<E, _>(&params.vk, num_pub, &ck_glob, &mut rng);
    let (ek_glob, vk_link_glob) = global_link_keygen::<E, _>(&pp_glob, crs_glob, &mut rng);
    let cpg_setup_ms = cpg_setup_start.elapsed().as_millis();

    let cpg_prove_start = Instant::now();
    let (proof_link_glob, instance_glob) = global_link_prove::<E, _>(
        &pp_glob, &ek_glob, o_glob, v2, &committed_group2, c_w, proof.d2, &mut rng,
    );
    let cpg_prove_ms = cpg_prove_start.elapsed().as_millis();

    let cpg_verify_start = Instant::now();
    if !global_link_verify::<E>(&pp_glob, &vk_link_glob, &instance_glob, &proof_link_glob) {
        return Err(anyhow::anyhow!("CP-Link #2 (c_W ↔ d2) verification failed"));
    }
    println!("  -> [Success] CP-Link #2 (c_W ↔ d2) verified");
    let cpg_verify_ms = cpg_verify_start.elapsed().as_millis();

    let cplink_global_report = proof_size::compute_cplink_report::<E>(&proof_link_glob, &instance_glob);
    cplink_global_report.print_log();

    // ── Fiat-Shamir verification (v3: includes d3) ─────────────────────
    let r_fs_verify = hash_commitments_to_field_v3::<E>(
        &c_w, &proof.d1, &proof.d2, &proof.d3,
        &public_inputs[1], &public_inputs[2],
    );
    assert_eq!(r_fs, r_fs_verify, "Fiat-Shamir recomputation mismatch");
    println!("  -> [Success] Fiat-Shamir recomputation verified");

    // ── Grand total proof size report ──
    proof_size::print_full_report(&g16_report, &cplink_lasso_report, &cplink_global_report);

    let e2e_time = e2e_start.elapsed();

    BenchmarkResult {
        file: format!("{}_layers", discovered.num_layers),
        circuit: "model_integrated".to_string(),
        constraints: num_constraints,
        groth16: Groth16Metrics {
            setup_ms: setup_time.as_millis(),
            commit_ms: g16_commit_ms,
            cm1_ms,
            cm2_ms,
            cm_aux_ms,
            prove_ms: g16_prove_time.as_millis(),
            verify_ms: g16_verify_time.as_millis(),
            proof_size_bytes: proof_size_bytes,
            commitment_size_bytes: groth16_commitment_size::<E>(&cm1),
        },
        lasso: LassoBenchmarkMetrics::from_aggregate(&lasso_agg),
        cp_link_lasso: CpLinkMetrics {
            commit_ms: 0,
            setup_ms: cpl_setup_ms,
            prove_ms: cpl_prove_ms,
            verify_ms: cpl_verify_ms,
            proof_size_bytes: cplink_lasso_report.proof_total,
        },
        cp_link_global: CpLinkMetrics {
            commit_ms: global_commit_ms,
            setup_ms: cpg_setup_ms,
            prove_ms: cpg_prove_ms,
            verify_ms: cpg_verify_ms,
            proof_size_bytes: cplink_global_report.proof_total,
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
