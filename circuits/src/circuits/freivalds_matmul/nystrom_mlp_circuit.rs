use std::{fs::File, ops::{AddAssign, Mul}, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing, AffineRepr};
use ndarray::{Ix0, Ix1, Ix2, Ix3};
use ndarray_npy::NpzReader;
use ark_std::{rc::Rc, cell::RefCell, fs, UniformRand, marker::PhantomData, vec::Vec, rand::{RngCore, Rng, SeedableRng, rngs::StdRng}};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, One, Zero, BigInteger};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{fp::FpVar, FieldVar}, R1CSVar};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        proof_size,
    },
    config::{GELU_C, GELU_M, KBITS_C, KBITS_M, RMSNORM_C, RMSNORM_M, SOFTMAX_C, SOFTMAX_M},
    gadgets::{
        activation::gelu::{GeLUBundle, enforce_gelu},
        layernorm::constraints::{NormBundle, LayerNormBundle, enforce_rmsnorm, enforce_layernorm},
        linear::constraints::{LinearBundle, enforce_linear},
    },
    lasso::{
        lasso::{
            densified::DensifiedRepresentation,
            surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
        },
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
        subtables::{
            gelu::GeLUSubtableStrategy, layernorm::RMSNormSubtableStrategy,
            range_check::RangeCheckSubtableStrategy,
        },
        utils::random::RandomTape,
    },
    util::{
        load::{to_field_1d_i64, to_field_2d_i32, to_field_3d_i32, to_field_3d_i64, to_field_3d_u64},
        mlp_lookup::{
            run_lasso_for_gelu, run_lasso_for_layernorm, run_lasso_for_rmsnorm,
            run_mlp_remainders, run_mlp_remainders_layernorm,
        },
        norm_detect::{NormType, detect_norm_type},
        utils::{cm1_from_vk, hash_cm_to_field_vec},
    },
};
use crate::lasso::subtables::layernorm::set_layernorm_table_dir;
use crate::lasso::subtables::gelu::set_gelu_table_dir;

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
        frac_bits,
        k_bits,
        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })


}

// ── Phase 5-B B3-2: LayerNorm 데이터 loader ─────────────────────────────────
//
// 자체학습 RMSNorm 측 `load_nystrom_circuit` (위) 보존. 본 함수는 HF LayerNorm
// witness npz 를 읽어 LayerNormBundle 두 개 + 기존 LIN/GELU 데이터를 묶음.
// `enforce_layernorm` 가젯 (B4) + 회로 wrap (B4) 진입 전 단계라 `ConstraintSynthesizer`
// 를 구현한 회로 struct 가 아니라 *데이터 묶음* 만 반환.
//
// 자동 분기는 `load_nystrom_circuit_auto` (아래) 가 `detect_norm_type` 로 판별.

/// HF LayerNorm 측 데이터 묶음 (B3 단계 — 데이터만, ConstraintSynthesizer 없음).
/// B4-2 에서 `NystromMLPLayerNormCircuit` 으로 wrap 됨.
#[derive(Clone)]
pub struct NystromMLPLayerNormData<F: PrimeField> {
    pub norm1: LayerNormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,
    pub ff2: LinearBundle<F>,
    pub norm2: LayerNormBundle<F>,
    pub frac_bits: usize,
    pub k_bits: usize,
}

// ── Phase 5-B B4-2: HF LayerNorm 측 회로 (ConstraintSynthesizer impl 포함) ──
//
// 자체학습 RMSNorm 측 `NystromMLPCircuit<F>` (위) 와 평행. LayerNorm 측 데이터
// (`NystromMLPLayerNormData`) 를 받아 `enforce_layernorm` 가젯으로 R1CS 검증.
//
// `BERTMLPCircuit` type alias (RMSNorm) 와 같은 의미로 BERT/Nystrom 측 LayerNorm
// 회로가 같은 struct 를 공유. (B4-2 단계에서 alias 추가하지 않고 단일 struct 만.)
#[derive(Clone)]
pub struct NystromMLPLayerNormCircuit<F: PrimeField> {
    pub norm1: LayerNormBundle<F>,
    pub ff1: LinearBundle<F>,
    pub gelu1: GeLUBundle<F>,
    pub ff2: LinearBundle<F>,
    pub norm2: LayerNormBundle<F>,
    pub frac_bits: usize,
    pub k_bits: usize,
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for NystromMLPLayerNormCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();

        let mut ln1_pool = Vec::new();
        let mut gelu_pool = Vec::new();
        let mut ln2_pool = Vec::new();
        let mut r_pool = Vec::new();

        let dim: usize = self.norm1.q_ri[0][0].len();

        // 1. norm1 → enforce_layernorm  (Python: _process_layernorm_block, ln1)
        let q_norm1_vars = enforce_layernorm(
            cs.clone(),
            &self.norm1,
            dim,
            self.frac_bits,
            self.k_bits,
            &mut r_pool,
            &mut ln1_pool,
        )?;

        // 2. ff1 → enforce_linear (RMSNorm 측과 동일)
        let (q_ff1_vars, r_ff1_vars) =
            enforce_linear(cs.clone(), &q_norm1_vars, &self.ff1, self.k_bits, &mut r_pool)?;

        // 3. gelu1 → enforce_gelu (RMSNorm 측과 동일 — sigmoid GELU)
        let q_gelu_vars = enforce_gelu(
            cs.clone(),
            &q_ff1_vars,
            &r_ff1_vars,
            &self.gelu1,
            self.k_bits,
            &mut r_pool,
            &mut gelu_pool,
        )?;

        // 4. ff2 → enforce_linear (RMSNorm 측과 동일)
        let (q_ff2_vars, _) =
            enforce_linear(cs.clone(), &q_gelu_vars, &self.ff2, self.k_bits, &mut r_pool)?;
        let _ = q_ff2_vars; // 다음 LayerNorm 의 입력은 norm2.q_ri (witness 측에서 제공) — 회로 외부에서 정합

        // 5. norm2 → enforce_layernorm (Python: ln2)
        let _q_norm2_vars = enforce_layernorm(
            cs.clone(),
            &self.norm2,
            dim,
            self.frac_bits,
            self.k_bits,
            &mut r_pool,
            &mut ln2_pool,
        )?;

        // 6. committed_vars 에 push (Phase 5-D 측정용 CP-Link commitment)
        {
            let mut final_vars = self.committed_vars.borrow_mut();
            final_vars.extend(ln1_pool);
            final_vars.extend(gelu_pool);
            final_vars.extend(ln2_pool);
            final_vars.extend(r_pool);
        }

        Ok(())
    }
}

/// `load_nystrom_layernorm_data` 결과를 `NystromMLPLayerNormCircuit` 으로 wrap.
/// Phase 5-D 측정 시 ConstraintSynthesizer 인터페이스가 필요한 경우 사용.
pub fn load_nystrom_layernorm_circuit<E: Pairing>(
    path: &Path,
) -> Result<NystromMLPLayerNormCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let data = load_nystrom_layernorm_data::<E>(path)?;
    Ok(NystromMLPLayerNormCircuit {
        norm1: data.norm1,
        ff1: data.ff1,
        gelu1: data.gelu1,
        ff2: data.ff2,
        norm2: data.norm2,
        frac_bits: data.frac_bits,
        k_bits: data.k_bits,
        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })
}

/// LayerNorm 측 npz 로더 (`var_scale=1000`, `eps=1e-12 또는 1e-5`, prefix=ln1/ln2).
///
/// Python pipeline `bert_normalization_pipeline.py` 의 출력과 1:1 매핑.
/// 키 매핑 reference: `docs/migration/20-b3-keymap-analysis.md` §4.
pub fn load_nystrom_layernorm_data<E: Pairing>(
    path: &Path,
) -> Result<NystromMLPLayerNormData<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening layernorm witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let k_bits = npz
        .by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?
        .into_scalar() as usize;
    let frac_bits = npz
        .by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?
        .into_scalar() as usize;

    // ── load_layernorm_block: 26 키 read → LayerNormBundle ──────────────────
    let load_layernorm = |npz: &mut NpzReader<File>, prefix: &str|
        -> Result<LayerNormBundle<E::ScalarField>>
    {
        Ok(LayerNormBundle {
            // ── 입력 ──
            q_ri: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("Q_ri_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            z_ri: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_ri_{}.npy", prefix))?
                    .into_scalar(),
            ),
            // ── 평균 (LayerNorm 신규) ──
            sum_x: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("sum_x_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            q_mean: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("Q_mean_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_mean: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("R_mean_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            // ── 차이 ──
            q_diff: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("Q_diff_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            // ── 분산 (var) ──
            sum_sq: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("sum_sq_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            m_var: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_var_{}.npy", prefix))?
                    .into_scalar(),
            ),
            m_prime_var: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_var_{}.npy", prefix))?
                    .into_scalar(),
            ),
            shifted_val_var: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("shifted_val_var_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_shift_var: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("R_shift_var_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            q_var_sim: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("Q_var_sim_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_var: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("R_var_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            // ── inv_std lookup ──
            inv_std: to_field_3d_u64(
                &npz.by_name::<_, Ix3>(&format!("inv_std_total_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            m_ln_idx: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_ln_{}.npy", prefix))?
                    .into_scalar(),
            ),
            // ── γ ──
            q_gamma: to_field_1d_i64(
                &npz.by_name::<_, Ix1>(&format!("Q_gamma_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            z_gamma: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_gamma_{}.npy", prefix))?
                    .into_scalar(),
            ),
            // ── β + lifting (LayerNorm 신규) ──
            q_beta: to_field_1d_i64(
                &npz.by_name::<_, Ix1>(&format!("Q_beta_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            z_beta: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_beta_{}.npy", prefix))?
                    .into_scalar(),
            ),
            m_beta_lift: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_beta_lift_{}.npy", prefix))?
                    .into_scalar(),
            ),
            q_beta_lifted: to_field_1d_i64(
                &npz.by_name::<_, Ix1>(&format!("Q_beta_lifted_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_beta_lift: to_field_1d_i64(
                &npz.by_name::<_, Ix1>(&format!("R_beta_lift_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            // ── 출력 rescale ──
            m_norm: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_norm_{}.npy", prefix))?
                    .into_scalar(),
            ),
            m_prime_norm: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_norm_{}.npy", prefix))?
                    .into_scalar(),
            ),
            q_out: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("Q_ln_output_sim_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_out: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("R_ln_output_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
        })
    };

    let norm1 = load_layernorm(&mut npz, "ln1")?;

    // ── LIN/GELU 는 RMSNorm 측과 동일 — 기존 클로저 그대로 재사용 ─────────
    let load_linear = |npz: &mut NpzReader<File>, prefix: &str| -> Result<LinearBundle<E::ScalarField>> {
        Ok(LinearBundle {
            z_in: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_in_{}.npy", prefix))?
                    .into_scalar(),
            ),
            q_w_diff: to_field_2d_i32(
                &npz.by_name::<_, Ix2>(&format!("Q_w_diff_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            q_b_diff: to_field_1d_i64(
                &npz.by_name::<_, Ix1>(&format!("Q_b_diff_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            m_lin: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?
                    .into_scalar(),
            ),
            m_prime_lin: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?
                    .into_scalar(),
            ),
            q_out_sim: to_field_3d_i32(
                &npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_out_lin: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
        })
    };

    let ff1 = load_linear(&mut npz, "lin1")?;

    let load_gelu = |npz: &mut NpzReader<File>, prefix: &str| -> Result<GeLUBundle<E::ScalarField>> {
        Ok(GeLUBundle {
            z_in: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("z_in_{}.npy", prefix))?
                    .into_scalar(),
            ),
            sigmoid_table_val: to_field_3d_u64(
                &npz.by_name::<_, Ix3>(&format!("sigmoid_part_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            m_gelu: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_{}.npy", prefix))?
                    .into_scalar(),
            ),
            m_prime_gelu: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_{}.npy", prefix))?
                    .into_scalar(),
            ),
            q_out_sim: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("Q_out_sim_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            r_out_gelu: to_field_3d_i64(
                &npz.by_name::<_, Ix3>(&format!("R_out_{}.npy", prefix))?
                    .into_dimensionality()?,
            ),
            m_gelu_idx: E::ScalarField::from(
                npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_gelu_idx.npy")?
                    .into_scalar(),
            ),
        })
    };

    let gelu1 = load_gelu(&mut npz, "gelu")?;
    let ff2 = load_linear(&mut npz, "lin2")?;
    let norm2 = load_layernorm(&mut npz, "ln2")?;

    Ok(NystromMLPLayerNormData {
        norm1,
        ff1,
        gelu1,
        ff2,
        norm2,
        frac_bits,
        k_bits,
    })
}

// ── Phase 5-B B3-2: 자동 분기 enum + helper ────────────────────────────────

/// 회로 측이 받을 수 있는 두 가지 데이터 변형. `load_nystrom_circuit_auto` 가
/// npz 메타로 자동 결정 후 반환. 양 variant 모두 `ConstraintSynthesizer` 구현
/// (B4-2 후) — 호출 측에서 match 로 분기 후 prove/verify 동일 인터페이스.
pub enum NystromMLPLoaded<F: PrimeField> {
    /// 자체학습 시절 RMSNorm 회로 (`enforce_rmsnorm` + `NormBundle`).
    RMSNorm(NystromMLPCircuit<F>),
    /// HF LayerNorm 회로 (`enforce_layernorm` + `LayerNormBundle`).
    LayerNorm(NystromMLPLayerNormCircuit<F>),
}

/// npz 메타 (var_scale.npy 존재 여부) 기반 자동 분기.
/// 결과는 `NystromMLPLoaded` enum — 호출 측에서 match 로 처리.
pub fn load_nystrom_circuit_auto<E: Pairing>(
    path: &Path,
) -> Result<NystromMLPLoaded<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    match detect_norm_type(path)? {
        NormType::RMSNorm => Ok(NystromMLPLoaded::RMSNorm(load_nystrom_circuit::<E>(path)?)),
        NormType::LayerNorm => Ok(NystromMLPLoaded::LayerNorm(
            load_nystrom_layernorm_circuit::<E>(path)?,
        )),
    }
}

pub fn run_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== Processing witness NPZ: {:?} ===", file_path);

        // ── Dry-run: extract committed witness indices ───────────────────────
        let d1_indices = {
            let temp_circuit = load_nystrom_circuit::<E>(&file_path)?;
            let vars_ptr = Rc::clone(&temp_circuit.committed_vars);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;
            crate::util::utils::extract_witness_indices(&vars_ptr.borrow())
        };
        let num_commit_1 = d1_indices.len() as u32;
        println!("최종 측정된 커밋 개수: {:?}", num_commit_1);

        let sample_circuit = load_nystrom_circuit::<E>(&file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (Num Commit: {:?}) ===", num_commit_1);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            &d1_indices,
            &[],
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction + constraint count ────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

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
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit (Pedersen cm1 from VK) ───────────────────────────
        let groth16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let groth16_commit_ms = groth16_commit_start.elapsed().as_millis();

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        println!("=== [1] LegoGroth16 GenProof ===");
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            sample_circuit.clone(),
            v1,
            v2,
            &params,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;
        let groth16_prove_time = groth16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // ── [1] Groth16 Verify ───────────────────────────────────────────────
        let public_inputs = vec![];
        let groth16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let groth16_verify_time = groth16_verify_start.elapsed();

        println!("  - LegoGroth16 SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::Error::msg(format!("{:?}", e)));
        }

        // ── [2] Lasso (commit + prove + verify bundled in helpers) ───────────
        // Reset Lasso aggregator before individual Lasso proofs
        proof_size::reset_lasso_aggregator();
        println!("=== [2] Lasso Lookup Proofs: MLP Sections ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let ((lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1), m1) = run_lasso_for_rmsnorm(&sample_circuit.norm1, b"lasso_rms1", &mut rng)?;
        lasso_agg.add(&m1);
        let ((lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2), m2) = run_lasso_for_gelu(
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
        lasso_agg.add(&m2);
        let ((lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3), m3) = run_lasso_for_rmsnorm(&sample_circuit.norm2, b"lasso_rms2", &mut rng)?;
        lasso_agg.add(&m3);
        let (res_remainders, rem_agg) = run_mlp_remainders::<Fr, G1>(&sample_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let u_values: Vec<Fr> = sample_circuit.committed_vars.borrow().iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = vec![
            (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1),
            (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2),
            (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3),
        ];
        lasso_results.extend(res_remainders);

        // ── [3] CP-Link (setup/keygen, prove, verify timed separately) ───────
        let u_padded = pad_lasso_u_values::<E>(&lasso_results, &u_values);

        let cp_setup_start = Instant::now();
        let (pp, crs) = factored_link_setup::<E, _>(&params.vk, num_pub, u_values.len(), &lasso_results, &mut rng);
        let (ek, vk_link) = factored_link_keygen::<E, _>(&pp, crs, &mut rng);
        let cp_setup_ms = cp_setup_start.elapsed().as_millis();

        let cp_prove_start = Instant::now();
        let (proof_link, instance) = factored_link_prove::<E, _>(&pp, &ek, v1, cm1, &lasso_results, u_padded, &mut rng);
        let cp_prove_ms = cp_prove_start.elapsed().as_millis();

        let cp_verify_start = Instant::now();
        if !factored_link_verify::<E>(&pp, &vk_link, &instance, &proof_link) {
            return Err(anyhow::anyhow!("Factored link verification failed"));
        }
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        let cp_verify_ms = cp_verify_start.elapsed().as_millis();

        // ── Groth16 proof size report ──
        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // ── CP-Link proof size report ──
        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        cplink_report.print_log();

        // ── Grand total proof size report ──
        proof_size::print_full_report(&g16_report, &cplink_report);

        let e2e_time = e2e_start.elapsed();

        // ── Benchmark output ─────────────────────────────────────────────────
        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "mlp".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: groth16_commit_ms,
                prove_ms: groth16_prove_time.as_millis(),
                verify_ms: groth16_verify_time.as_millis(),
                proof_size_bytes: proof_size,
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
    }
    Ok(())
}



pub fn run_paths_layernorm<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [LayerNorm] Processing witness NPZ: {:?} ===", file_path);

        // ── Dry-run: extract committed witness indices ───────────────────────
        let dry_run_start = Instant::now();
        let d1_indices = {
            let temp_circuit = load_nystrom_layernorm_circuit::<E>(&file_path)?;
            let vars_ptr = Rc::clone(&temp_circuit.committed_vars);
            let cs_analyze = ConstraintSystem::<Fr>::new_ref();
            temp_circuit.generate_constraints(cs_analyze)?;
            crate::util::utils::extract_witness_indices(&vars_ptr.borrow())
        };
        let num_commit_1 = d1_indices.len() as u32;
        println!("  - Dry-run commit count: {} ({:?})", num_commit_1, dry_run_start.elapsed());

        let load_start = Instant::now();
        let sample_circuit = load_nystrom_layernorm_circuit::<E>(&file_path)?;
        println!("  - Witness load time: {:?}", load_start.elapsed());

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (Num Commit: {}) ===", num_commit_1);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            &d1_indices,
            &[],
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();
        println!("  - Setup time: {:?}", setup_time);

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction + constraint count ────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();
        println!("  - Num constraints: {}", num_constraints);

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
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit (Pedersen cm1 from VK) ───────────────────────────
        let groth16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let groth16_commit_ms = groth16_commit_start.elapsed().as_millis();

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        println!("=== [1] LegoGroth16 GenProof ===");
        let groth16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            sample_circuit.clone(),
            v1,
            v2,
            &params,
            &mut rng,
        ).map_err(|e| anyhow::anyhow!("Groth16 proof generation failed: {:?}", e))?;
        let groth16_prove_time = groth16_prove_start.elapsed();
        println!("  - Groth16 prove time: {:?}", groth16_prove_time);

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // ── [1] Groth16 Verify ───────────────────────────────────────────────
        let public_inputs = vec![];
        let groth16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let groth16_verify_time = groth16_verify_start.elapsed();

        println!("  - LegoGroth16 SNARK Verification: {:?} ({:?})", result, groth16_verify_time);
        if let Err(e) = result {
            return Err(anyhow::Error::msg(format!("{:?}", e)));
        }

        // ── [2] Lasso (commit + prove + verify bundled in helpers) ───────────
        // Reset Lasso aggregator before individual Lasso proofs
        proof_size::reset_lasso_aggregator();
        println!("=== [2] Lasso Lookup Proofs: MLP Sections (LayerNorm) ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let ((lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1), m1) =
            run_lasso_for_layernorm(&sample_circuit.norm1, b"lasso_ln1", &mut rng)?;
        lasso_agg.add(&m1);
        let ((lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2), m2) = run_lasso_for_gelu(
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
            &mut rng,
        )?;
        lasso_agg.add(&m2);
        let ((lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3), m3) =
            run_lasso_for_layernorm(&sample_circuit.norm2, b"lasso_ln2", &mut rng)?;
        lasso_agg.add(&m3);
        let (res_remainders, rem_agg) = run_mlp_remainders_layernorm::<Fr, G1>(&sample_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let u_values: Vec<Fr> = sample_circuit.committed_vars.borrow().iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();
        println!("  - Lasso total time: {:?}", lasso_time);

        let mut lasso_results = vec![
            (lasso_cm1, lasso_blinds1, lasso_gens1, lasso_sp1),
            (lasso_cm2, lasso_blinds2, lasso_gens2, lasso_sp2),
            (lasso_cm3, lasso_blinds3, lasso_gens3, lasso_sp3),
        ];
        lasso_results.extend(res_remainders);

        // ── [3] CP-Link (setup/keygen, prove, verify timed separately) ───────
        let u_padded = pad_lasso_u_values::<E>(&lasso_results, &u_values);

        let cp_setup_start = Instant::now();
        let (pp, crs) = factored_link_setup::<E, _>(&params.vk, num_pub, u_values.len(), &lasso_results, &mut rng);
        let (ek, vk_link) = factored_link_keygen::<E, _>(&pp, crs, &mut rng);
        let cp_setup_ms = cp_setup_start.elapsed().as_millis();

        let cp_prove_start = Instant::now();
        let (proof_link, instance) = factored_link_prove::<E, _>(&pp, &ek, v1, cm1, &lasso_results, u_padded, &mut rng);
        let cp_prove_ms = cp_prove_start.elapsed().as_millis();

        let cp_verify_start = Instant::now();
        if !factored_link_verify::<E>(&pp, &vk_link, &instance, &proof_link) {
            return Err(anyhow::anyhow!("Factored link verification failed"));
        }
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        let cp_verify_ms = cp_verify_start.elapsed().as_millis();

        // ── Groth16 proof size report ──
        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // ── CP-Link proof size report ──
        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        cplink_report.print_log();

        // ── Grand total proof size report ──
        proof_size::print_full_report(&g16_report, &cplink_report);

        let e2e_time = e2e_start.elapsed();
        println!("\n=== [LayerNorm] e2e total: {:?} ===", e2e_time);

        // ── Benchmark output ─────────────────────────────────────────────────
        BenchmarkResult {
            file: file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit: "mlp-layernorm".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: groth16_commit_ms,
                prove_ms: groth16_prove_time.as_millis(),
                verify_ms: groth16_verify_time.as_millis(),
                proof_size_bytes: proof_size,
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
    }
    Ok(())
}



// 환경 변수 (모두 optional):
//   LAYERNORM_WITNESS_DIR : witness npz 디렉터리 (예: outputs_m64)
//                           기본값: "outputs_m64"
//   LAYERNORM_TABLE_PATH  : layernorm_tables.npz 경로 (subtables/layernorm.rs 가 직접 read)
//                           기본값: <WITNESS_DIR>/layernorm_tables.npz (set_layernorm_table_dir)
//   LAYERNORM_LAYER_IDX   : 처리할 layer index (단일 정수, 기본 0 — 첫 번째 layer)
//
// 실행:
//   LAYERNORM_WITNESS_DIR=/path/to/outputs_m64 \
//     cargo test --release nystrom_mlp_hf_single_layer_test -- --nocapture
//
// peak memory 측정: macOS `/usr/bin/time -l ./target/release/...` 또는
// linux `/usr/bin/time -v ...` (회로 측 측정 안 함, 사용자 시스템 측 측정 권장).
#[test]
fn nystrom_mlp_hf_single_layer_test() {
    use std::env;

    // ── 환경 변수 ───────────────────────────────────────────────────────────
    let witness_dir = env::var("LAYERNORM_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_m64".to_string());
    let layer_idx: Option<usize> = env::var("LAYERNORM_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] LAYERNORM_WITNESS_DIR={} not a directory. \
             Set the env var to a valid Phase 4 outputs_m{{N}}/ dir to run.",
            witness_dir
        );
        return;
    }

    // ── 표 경로 결정: env var 미설정 시 witness_dir 안의 tables.npz ─
    if env::var("LAYERNORM_TABLE_PATH").is_err() {
        set_layernorm_table_dir(dir);
        println!(
            "[info] LAYERNORM_TABLE_PATH not set; using {}/layernorm_tables.npz",
            witness_dir
        );
    }
    if env::var("GELU_TABLE_PATH").is_err() {
        set_gelu_table_dir(dir);
        println!(
            "[info] GELU_TABLE_PATH not set; using {}/gelu_tables.npz",
            witness_dir
        );
    }

    // ── npz 수집 + 정렬 + layer_idx 만 단일 추출 ───────────────────────────
    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("norm.witness.npz"))
        {
            all_targets.push(path);
        }
    }
    all_targets.sort();

    if all_targets.is_empty() {
        eprintln!(
            "[skip] no '*norm.witness.npz' files in {}. \
             Run Phase 4 export pipeline first.",
            witness_dir
        );
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!(
                "LAYERNORM_LAYER_IDX={} out of range (have {} layers).",
                idx,
                all_targets.len()
            );
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [LayerNorm prove+verify] dir={} | files={} | mode={} ===",
        witness_dir,
        targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_paths_layernorm::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!(
            "\n=== [LayerNorm] ALL {} files PASS | total_wall_ms={} ===",
            targets.len(), total_ms,
        ),
        Err(e) => panic!(
            "[LayerNorm] proof generate/verify failed after {}ms: {}",
            total_ms, e,
        ),
    }
}