use std::{fs::File, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::pairing::Pairing;
use ndarray::{Ix0, Ix2, Ix3, Ix4};
use ndarray_npy::NpzReader;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{Rng, SeedableRng, rngs::StdRng},
    cell::RefCell,
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero, BigInteger};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::{fp::FpVar, FieldVar}, R1CSVar, select::CondSelectGadget};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult, Groth16Metrics, LassoBenchmarkMetrics, CpLinkMetrics, LassoMetricsAggregate,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size,
        },
        proof_size,
    },
    config::{KBITS_C, KBITS_M, SM_KBITS_C, SM_KBITS_M},
    gadgets::{
        matmul::{
            constraints::new_matrix_witness,
            quantized::{
                MatmulBundle, ScalingBundle, ZenParams,
                enforce_freivalds, enforce_scaling, gen_challenge_vector,
                new_matrix_witness_4d, transpose_matrix,
            },
        },
        softmax::constraints::{SoftmaxBundle, SoftmaxVarBundle, enforce_softmax},
    },
    lasso::{
        lasso::surge::SparsePolyCommitmentGens,
        poly::{commitments::MultiCommitGens, dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds}},
    },
    util::{
        load::{to_field_2d_i64, to_field_4d_i64, to_field_4d_u64},
        bert_attn_lookup::run_bert_lasso_bundle,
        nystrom_attn_lookup::{prove_lasso_for_range_22, prove_lasso_for_range_52},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;

type G1 = G1Projective;
type LassoResult = (PolyCommitment<G1>, PolyCommitmentBlinds<Fr>, MultiCommitGens<G1>, usize);

/// BERT attention circuit proving ONLY the core attention equation.
/// Excludes: input projections (XW_Q/K/V), output projection (XW_O),
/// weight matrices, Poseidon hashing, global weight commitment (c_W),
/// CP-Link #2, d2 group.
#[derive(Clone)]
pub struct BERTAttentionOnlyeqCircuit<F: PrimeField> {
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,

    /// Raw Q [B][H][S][D] (before scaling) — loaded from Q_raw_proj_q.npy
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,
    /// Raw K [B][H][S][D] (before scaling) — loaded from Q_raw_proj_k.npy
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,
    /// V [B][H][S][D] — loaded from Q_raw_proj_v.npy
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,
    pub z_raw_q: F,
    pub z_raw_k: F,

    pub scale_q: ScalingBundle<F>,
    pub scale_k: ScalingBundle<F>,

    /// QK^T matmul
    pub matmul_attn: MatmulBundle<F>,
    /// Full-sequence softmax
    pub sm: SoftmaxBundle<F>,
    /// P * V output
    pub matmul_x: MatmulBundle<F>,

    pub frac_bits: usize,
    pub r_challenge: F,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>,

    /// d1 committed group only
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for BERTAttentionOnlyeqCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();

        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        let mut r_div_lows = Vec::new();
        let mut r_div_highs = Vec::new();

        // Raw Q/K witnesses (replaces proj_q_out, proj_k_out)
        let q_q_vars = new_matrix_witness_4d(cs.clone(), &self.q_q);
        let q_k_vars = new_matrix_witness_4d(cs.clone(), &self.q_k);
        // V witness (replaces proj_v_out)
        let q_v_vars = new_matrix_witness_4d(cs.clone(), &self.q_v);

        // Scaling witnesses
        let scale_q_out = new_matrix_witness_4d(cs.clone(), &self.scale_q.q_scaled);
        let scale_q_r = new_matrix_witness_4d(cs.clone(), &self.scale_q.r_scale);
        let scale_k_out = new_matrix_witness_4d(cs.clone(), &self.scale_k.q_scaled);
        let scale_k_r = new_matrix_witness_4d(cs.clone(), &self.scale_k.r_scale);

        let mut global_scale_q_lows = Vec::new(); let mut global_scale_q_highs = Vec::new();
        let mut global_scale_k_lows = Vec::new(); let mut global_scale_k_highs = Vec::new();

        // QK^T and P*V matmul allocation
        let (attn_data, params_attn) =
            self.matmul_attn.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        let (x_data, params_x) =
            self.matmul_x.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);

        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm_vars = self.sm.alloc_vars(cs.clone())?;

        let mut global_attn_lows  = Vec::new();
        let mut global_attn_highs = Vec::new();
        let mut global_sm_idx: Vec<FpVar<F>> = Vec::new();
        let mut global_sm_r_sm: [Vec<FpVar<F>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut global_x_lows   = Vec::new();
        let mut global_x_highs  = Vec::new();

        let zero_var = FpVar::zero();

        for b in 0..scale_q_out.len() {
            for h in 0..scale_q_out[b].len() {
                // Q Scaling: raw_Q → scaled_Q
                {
                    let z_raw_q = FpVar::Constant(self.z_raw_q);
                    let m_scale_q = FpVar::Constant(self.scale_q.m_scale);
                    let m_prime_q = FpVar::Constant(self.scale_q.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &q_q_vars[b][h], &scale_q_out[b][h], &scale_q_r[b][h],
                        &self.scale_q.r_scale[b][h],
                        &z_raw_q, &m_scale_q, &m_prime_q, &two_k_var,
                        &mut global_scale_q_lows, &mut global_scale_q_highs,
                    )?;
                }
                // K Scaling: raw_K → scaled_K
                {
                    let z_raw_k = FpVar::Constant(self.z_raw_k);
                    let m_scale_k = FpVar::Constant(self.scale_k.m_scale);
                    let m_prime_k = FpVar::Constant(self.scale_k.m_prime_scale);
                    enforce_scaling(
                        cs.clone(),
                        &q_k_vars[b][h], &scale_k_out[b][h], &scale_k_r[b][h],
                        &self.scale_k.r_scale[b][h],
                        &z_raw_k, &m_scale_k, &m_prime_k, &two_k_var,
                        &mut global_scale_k_lows, &mut global_scale_k_highs,
                    )?;
                }

                // [1] QK^T: scale_q_out × scale_k_out^T → [S×S]
                let k_t = transpose_matrix(&scale_k_out[b][h]);
                let r_pows_attn = gen_challenge_vector(&r_var, attn_data.q_sim[b][h][0].len());
                let mut h_attn_lows  = Vec::new();
                let mut h_attn_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &scale_q_out[b][h], &k_t,
                    &attn_data.q_sim[b][h], &attn_data.r_matmul[b][h],
                    &self.matmul_attn.r_matmul[b][h],
                    &params_attn, &r_pows_attn,
                    &mut h_attn_lows, &mut h_attn_highs,
                    None,
                )?;
                global_attn_lows.extend(h_attn_lows);
                global_attn_highs.extend(h_attn_highs);

                // [2] Softmax
                enforce_softmax(
                    cs.clone(),
                    &attn_data.q_sim[b][h], &attn_data.r_matmul[b][h],
                    &sm_vars.r_sm[b][h], &sm_vars.m_sm,
                    &sm_vars.table_idx[b][h], &sm_vars.exp_total[b][h],
                    &sm_vars.p[b][h], &sm_vars.r_div[b][h],
                    &self.sm.r_div[b][h],
                    self.frac_bits, &two_k_var,
                    &mut global_sm_idx, &mut global_sm_r_sm,
                    &mut r_div_lows, &mut r_div_highs,
                )?;

                // v3 mask: verify exp_total=0 at padding positions
                {
                    let seq_len = sm_vars.exp_total[b][h].len();
                    let col_len = sm_vars.exp_total[b][h][0].len();
                    for row in 0..seq_len {
                        for col in 0..col_len {
                            let is_valid = mask_var[b][col].is_one()?;
                            let should_be_zero = FpVar::conditionally_select(
                                &is_valid,
                                &zero_var,
                                &sm_vars.exp_total[b][h][row][col],
                            )?;
                            should_be_zero.enforce_equal(&zero_var)?;
                        }
                    }
                }

                // [3] P * V: sm.p × V → [S×D]
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len());
                let mut h_x_lows  = Vec::new();
                let mut h_x_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &sm_vars.p[b][h], &q_v_vars[b][h],
                    &x_data.q_sim[b][h], &x_data.r_matmul[b][h],
                    &self.matmul_x.r_matmul[b][h],
                    &params_x, &r_pows_x,
                    &mut h_x_lows, &mut h_x_highs,
                    None,
                )?;
                global_x_lows.extend(h_x_lows);
                global_x_highs.extend(h_x_highs);
            }
        }

        // Populate d1 committed_vars (no projection remainders, no Poseidon hash)
        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // Softmax index lookup vars (C=1, single pool)
            final_vars.extend(global_sm_idx);

            // 26-bit remainders (C=2: 16+10) — NO projection remainders
            final_vars.extend(global_attn_lows);
            final_vars.extend(global_attn_highs);
            final_vars.extend(global_x_lows);
            final_vars.extend(global_x_highs);
            // Scaling remainders (KEPT)
            final_vars.extend(global_scale_q_lows); final_vars.extend(global_scale_q_highs);
            final_vars.extend(global_scale_k_lows); final_vars.extend(global_scale_k_highs);

            // 52-bit r_sm (C=4: 16+16+16+4)
            for k in 0..4 {
                final_vars.extend(global_sm_r_sm[k].clone());
            }

            // R_div per-softmax (26-bit C=2: 16+10)
            final_vars.extend(r_div_lows);
            final_vars.extend(r_div_highs);
        }

        Ok(())
    }
}

pub fn load_bert_attention_onlyeq_circuit<E: Pairing>(
    path: &Path,
) -> Result<BERTAttentionOnlyeqCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening BERT witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    let z_q = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar(),
    );
    let z_k = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar(),
    );
    let z_v = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar(),
    );

    // Raw Q/K/V (projection outputs)
    let q_q = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_raw_proj_q.npy")?.into_dimensionality()?);
    let q_k = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_raw_proj_k.npy")?.into_dimensionality()?);
    let q_v = to_field_4d_i64(&npz.by_name::<_, Ix4>("Q_raw_proj_v.npy")?.into_dimensionality()?);
    let z_raw_q = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_raw_q.npy")?.into_scalar());
    let z_raw_k = E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_raw_k.npy")?.into_scalar());

    // QK^T matmul
    let q_sim_attn = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_attn.npy")?.into_dimensionality()?;
    let r_matmul_attn = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_attn.npy")?.into_dimensionality()?;
    let matmul_attn = MatmulBundle {
        r_mean:   vec![],
        q_sim:    to_field_4d_i64(&q_sim_attn),
        r_matmul: to_field_4d_i64(&r_matmul_attn),
        m:        E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_attn.npy")?.into_scalar()),
        m_prime:  E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_attn.npy")?.into_scalar()),
        z_sim:    E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_attn.npy")?.into_scalar()),
    };

    let frac_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let k_bits = npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;

    // Softmax v3
    let table_idx_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("table_idx.npy")?.into_dimensionality()?;
    let exp_total_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("exp_total.npy")?.into_dimensionality()?;
    let p_attn_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("P_attn.npy")?.into_dimensionality()?;
    let r_div_attn_raw = npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("R_div_attn.npy")?.into_dimensionality()?;
    let r_sm_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("r_sm.npy")?.into_dimensionality()?;

    let sm = SoftmaxBundle {
        table_idx: to_field_4d_i64(&table_idx_raw),
        exp_total: to_field_4d_u64(&exp_total_raw),
        p:         to_field_4d_u64(&p_attn_raw),
        r_div:     to_field_4d_u64(&r_div_attn_raw),
        r_sm:      to_field_4d_i64(&r_sm_raw),
        m_sm:      E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar()),
    };

    // P*V output matmul
    let q_sim_x = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_x.npy")?.into_dimensionality()?;
    let r_matmul_x = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_x.npy")?.into_dimensionality()?;
    let matmul_x = MatmulBundle {
        r_mean:   vec![],
        q_sim:    to_field_4d_i64(&q_sim_x),
        r_matmul: to_field_4d_i64(&r_matmul_x),
        m:        E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_x.npy")?.into_scalar()),
        m_prime:  E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_x.npy")?.into_scalar()),
        z_sim:    E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_x.npy")?.into_scalar()),
    };

    let mask_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    let load_scaling = |npz: &mut NpzReader<std::fs::File>, suffix: &str| -> Result<ScalingBundle<E::ScalarField>> {
        Ok(ScalingBundle {
            q_scaled: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("Q_scaled_{}.npy", suffix))?.into_dimensionality()?),
            r_scale: to_field_4d_i64(&npz.by_name::<_, Ix4>(&format!("R_scale_{}.npy", suffix))?.into_dimensionality()?),
            m_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_scale_{}.npy", suffix))?.into_scalar()),
            m_prime_scale: E::ScalarField::from(npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>(&format!("M_prime_scale_{}.npy", suffix))?.into_scalar()),
        })
    };
    let scale_q = load_scaling(&mut npz, "q")?;
    let scale_k = load_scaling(&mut npz, "k")?;

    Ok(BERTAttentionOnlyeqCircuit {
        z_q, z_k, z_v,
        q_q, q_k, q_v,
        z_raw_q, z_raw_k,
        scale_q, scale_k,
        matmul_attn, sm, matmul_x,
        frac_bits,
        r_challenge: E::ScalarField::zero(),
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        committed_vars: Rc::new(RefCell::new(Vec::new())),
    })
}

// ── Lasso remainder function (inline, referencing onlyeq circuit) ────────────

fn run_onlyeq_bert_remainders_with_link(
    circuit: &BERTAttentionOnlyeqCircuit<Fr>,
    rng: &mut StdRng,
) -> Result<(Vec<LassoResult>, LassoMetricsAggregate)> {
    println!("\n=== [Lasso Lookup] BERT Onlyeq: Verifying All Remainders ===");

    let mut all_results = Vec::new();
    let mut metrics_agg = LassoMetricsAggregate::default();

    // 26-bit range checks — NO projection remainders
    let target_26bit: &[(&Vec<Vec<Vec<Vec<Fr>>>>, &str)] = &[
        (&circuit.matmul_attn.r_matmul, "bert_attn_remainder"),
        (&circuit.matmul_x.r_matmul,   "bert_x_remainder"),
        (&circuit.scale_q.r_scale,      "bert_scale_q_remainder"),
        (&circuit.scale_k.r_scale,      "bert_scale_k_remainder"),
    ];

    for (r_matrix, label) in target_26bit {
        let b_dim   = r_matrix.len();
        let h_dim   = r_matrix[0].len();
        let row_dim = r_matrix[0][0].len();
        let col_dim = r_matrix[0][0][0].len();
        let total_sparsity = b_dim * h_dim * row_dim * col_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for r in 0..row_dim {
                    for c in 0..col_dim {
                        let val_u64 = r_matrix[b][h][r][c].into_bigint().as_ref()[0];
                        nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3FF) as usize]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), KBITS_C,
            total_sparsity.next_power_of_two(), KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    // 52-bit r_sm range check
    {
        let r_sm_4d = &circuit.sm.r_sm;
        let label = "bert_sm_r_sm_52bit";

        let b_dim = r_sm_4d.len();
        let h_dim = r_sm_4d[0].len();
        let s_dim = r_sm_4d[0][0].len();
        let l_dim = r_sm_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz: Vec<[usize; SM_KBITS_C]> = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = r_sm_4d[b][h][s][l].into_bigint().as_ref()[0];
                        nz.push([
                            (val_u64 & 0xFFFF) as usize,
                            ((val_u64 >> 16) & 0xFFFF) as usize,
                            ((val_u64 >> 32) & 0xFFFF) as usize,
                            ((val_u64 >> 48) & 0xF) as usize,
                        ]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), SM_KBITS_C,
            total_sparsity.next_power_of_two(), SM_KBITS_C,
            SM_KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_52(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    // R_div 26-bit
    {
        let matrix_4d = &circuit.sm.r_div;
        let label = "bert_sm_r_div";

        let b_dim = matrix_4d.len();
        let h_dim = matrix_4d[0].len();
        let s_dim = matrix_4d[0][0].len();
        let l_dim = matrix_4d[0][0][0].len();
        let total_sparsity = b_dim * h_dim * s_dim * l_dim;

        let mut nz = Vec::with_capacity(total_sparsity);
        for b in 0..b_dim {
            for h in 0..h_dim {
                for s in 0..s_dim {
                    for l in 0..l_dim {
                        let val_u64 = matrix_4d[b][h][s][l].into_bigint().as_ref()[0];
                        nz.push([(val_u64 & 0xFFFF) as usize, ((val_u64 >> 16) & 0x3FF) as usize]);
                    }
                }
            }
        }

        let gens = SparsePolyCommitmentGens::<G1>::new(
            label.as_bytes(), KBITS_C,
            total_sparsity.next_power_of_two(), KBITS_C,
            KBITS_M.trailing_zeros() as usize,
        );

        let start = Instant::now();
        let (link_data, metrics) = prove_lasso_for_range_22(nz, &gens, rng, label.as_bytes())?;
        all_results.extend(link_data);
        metrics_agg.add(&metrics);
        println!("  - {} verified | Count: {} | Time: {:?}", label, total_sparsity, start.elapsed());
    }

    Ok((all_results, metrics_agg))
}

// ── Proving pipeline ─────────────────────────────────────────────────────────

pub fn run_bert_attention_onlyeq_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [BERT Onlyeq] Processing: {:?} ===", file_path);

        // Dry-run: extract d1 indices only (no d2)
        let d1_indices = {
            let temp = load_bert_attention_onlyeq_circuit::<E>(file_path)?;
            let v1_ptr = Rc::clone(&temp.committed_vars);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            crate::util::utils::extract_witness_indices(&v1_ptr.borrow())
        };
        let num_commit_1 = d1_indices.len() as u32;
        let d2_indices: Vec<u32> = vec![];
        println!("Commit counts: d1={}, d2=0", num_commit_1);

        let sample_circuit = load_bert_attention_onlyeq_circuit::<E>(file_path)?;

        // [0] Setup — d1 only
        println!("=== [0] LegoGroth16 Setup (d1={}, d2=0) ===", num_commit_1);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(), &d1_indices, &d2_indices, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();
        let pvk = prepare_verifying_key::<E>(&params.vk);

        // Witness extraction
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> = vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
            assert_eq!(values.len(), num_commit_1 as usize, "d1 size mismatch");
            values
        };

        let v1 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // cm1 only
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        // Simple Fiat-Shamir: hash(cm1)
        let r_fs = hash_cm_to_field::<E>(&cm1);

        println!("=== [1] Building circuit with Fiat-Shamir challenge ===");
        let final_circuit = BERTAttentionOnlyeqCircuit {
            r_challenge: r_fs,
            ..sample_circuit.clone()
        };

        // [1] Groth16 Prove (v2=Fr::zero(), no d2)
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(), v1, Fr::zero(), &params, &mut rng,
        ).map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
        let g16_prove_time = g16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof.serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // [1] Groth16 Verify — public_inputs = [r_challenge] only
        let public_inputs = vec![final_circuit.r_challenge];
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let g16_verify_time = g16_verify_start.elapsed();
        println!("  - BERT Onlyeq SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // [2] Lasso proofs
        proof_size::reset_lasso_aggregator();
        println!("=== [2] BERT Onlyeq Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let mut lasso_agg = LassoMetricsAggregate::default();
        let (lasso_sm, sm_agg) =
            run_bert_lasso_bundle(&final_circuit.sm, b"bert_lasso_sm", &mut rng)?;
        lasso_agg.add_aggregate(&sm_agg);
        let (lasso_remainders, rem_agg) =
            run_onlyeq_bert_remainders_with_link(&final_circuit, &mut rng)?;
        lasso_agg.add_aggregate(&rem_agg);
        let lasso_time = lasso_start.elapsed();
        lasso_agg.total_ms = lasso_time.as_millis();

        let mut lasso_results = Vec::new();
        lasso_results.extend(lasso_sm);
        lasso_results.extend(lasso_remainders);

        // [3] CP-Link #1: d1 ↔ Lasso
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

        // Fiat-Shamir verification
        let r_fs_verify = hash_cm_to_field::<E>(&proof.d1);
        assert_eq!(r_fs, r_fs_verify, "Fiat-Shamir recomputation mismatch");
        println!("  -> [Success] Fiat-Shamir recomputation verified");

        // Proof size report (no global link)
        let empty_cplink = proof_size::CpLinkSizeReport {
            proof_g_count: 0, proof_g_bytes: 0,
            proof_f_count: 0, proof_f_bytes: 0,
            proof_total: 0,
            instance_g_count: 0, instance_g_bytes: 0,
        };
        proof_size::print_full_report(&g16_report, &cplink_lasso_report, &empty_cplink);

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
            circuit: "bert_attention_onlyeq".to_string(),
            constraints: num_constraints,
            groth16: Groth16Metrics {
                setup_ms: setup_time.as_millis(),
                commit_ms: g16_commit_ms,
                prove_ms: g16_prove_time.as_millis(),
                verify_ms: g16_verify_time.as_millis(),
                proof_size_bytes: proof_size,
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
                commit_ms: 0, setup_ms: 0, prove_ms: 0, verify_ms: 0, proof_size_bytes: 0,
            },
            end_to_end_ms: e2e_time.as_millis(),
        }
        .print_json();
    }
    Ok(())
}

#[test]
fn bert_attn_onlyeq_single_layer_test() {
    use std::env;

    let witness_dir = env::var("BERT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_bert".to_string());
    let layer_idx: Option<usize> = env::var("BERT_ATTN_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] BERT_ATTN_WITNESS_DIR={} not a directory.",
            witness_dir
        );
        return;
    }

    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
        println!("[info] SOFTMAX_TABLE_PATH not set; using {}/softmax_tables.npz", witness_dir);
    }

    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path.file_name().and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention.self.witness.npz"))
        {
            all_targets.push(path);
        }
    }
    all_targets.sort();

    if all_targets.is_empty() {
        eprintln!("[skip] no '*attention.self.witness.npz' files in {}.", witness_dir);
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        assert!(idx < all_targets.len(), "BERT_ATTN_LAYER_IDX={} out of range", idx);
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [BERT Onlyeq prove+verify] dir={} | files={} | mode={} ===",
        witness_dir, targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_bert_attention_onlyeq_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!("\n=== [BERT Onlyeq] ALL {} files PASS | total_wall_ms={} ===", targets.len(), total_ms),
        Err(e) => panic!("[BERT Onlyeq] proof failed after {}ms: {}", total_ms, e),
    }
}
