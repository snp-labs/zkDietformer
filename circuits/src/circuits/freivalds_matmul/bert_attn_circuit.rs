use std::{fs::File, path::Path, str::FromStr, time::Instant};
use anyhow::{Context, Result};
use ark_ec::pairing::Pairing;
use ndarray::{IxDyn, Ix0, Ix2, Ix4};
use ndarray_npy::NpzReader;
use ark_std::{
    rc::Rc, fs, UniformRand,
    rand::{Rng, SeedableRng, rngs::StdRng},
    cell::RefCell,
};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ff::{PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, fields::{fp::FpVar, FieldVar}, R1CSVar, select::CondSelectGadget};
use crate::{
    cc_snark::legogroth16::{create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof},
    circuits::{
        common::{
            BenchmarkResult,
            factored_link_setup, factored_link_keygen, factored_link_prove, factored_link_verify,
            pad_lasso_u_values, groth16_commitment_size, lasso_commitment_size,
        },
        proof_size,
    },
    gadgets::{
        matmul::{
            constraints::new_matrix_witness,
            quantized::{
                MatmulBundle, enforce_freivalds, gen_challenge_vector,
                new_matrix_witness_4d, transpose_matrix,
            },
        },
        softmax::constraints::{SoftmaxBundle, enforce_softmax},
    },
    lasso::poly::{
        commitments::MultiCommitGens,
        dense_mlpoly::{PolyCommitment, PolyCommitmentBlinds},
    },
    util::{
        load::{to_field_2d_i64, to_field_4d_i64, to_field_4d_u64, to_field_5d_i64, to_field_5d_u64},
        bert_attn_lookup::{run_bert_lasso_bundle, run_all_bert_remainders_with_link},
        utils::{cm1_from_vk, hash_cm_to_field},
    },
};
use crate::lasso::subtables::softmax::set_softmax_table_dir;

type G1 = G1Projective;

#[derive(Clone)]
pub struct BERTAttentionCircuit<F: PrimeField> {
    /// Quantized Q, K, V: [batch][head][seq][head_dim]
    pub q_q: Vec<Vec<Vec<Vec<F>>>>,
    pub q_k: Vec<Vec<Vec<Vec<F>>>>,
    pub q_v: Vec<Vec<Vec<Vec<F>>>>,
    pub z_q: F,
    pub z_k: F,
    pub z_v: F,

    /// QK^T matmul: q_sim/r_matmul shape [batch][head][seq][seq], r_mean empty
    pub matmul_attn: MatmulBundle<F>,

    /// Full-sequence softmax over [batch][head][seq][seq]
    pub sm: SoftmaxBundle<F>,

    /// P * V output: q_sim/r_matmul shape [batch][head][seq][head_dim], r_mean empty
    pub matmul_x: MatmulBundle<F>,

    pub frac_bits: usize,
    pub limb_bits: usize,
    pub k_segments: usize,
    pub r_challenge: F,
    pub k_bits: usize,
    pub mask: Vec<Vec<F>>, 

    /// d1 committed group: Lasso-linked wires (remainder splits, softmax indices)
    pub committed_vars: Rc<RefCell<Vec<FpVar<F>>>>,
    /// d2 committed group: Q/K/V witnesses committed via v2, NOT CP-linked to Lasso
    pub committed_vars_2: Rc<RefCell<Vec<FpVar<F>>>>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for BERTAttentionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.committed_vars.borrow_mut().clear();
        self.committed_vars_2.borrow_mut().clear();

        // Public input: Fiat-Shamir challenge r
        let r_var = FpVar::new_input(cs.clone(), || Ok(self.r_challenge))?;
        let two_k_var = FpVar::new_constant(cs.clone(), F::from(1u128 << self.k_bits))?;

        // 16-bit remainder pools accumulated across all (b, h)
        let mut r2_pool = Vec::new(); // from sm.r_exp (5D)
        let mut r3_pool = Vec::new(); // from sm.r_div (4D)

        // Q/K/V: [B][H][S][D]
        let q_proc = new_matrix_witness_4d(cs.clone(), &self.q_q);
        let k_proc = new_matrix_witness_4d(cs.clone(), &self.q_k);
        let v_proc = new_matrix_witness_4d(cs.clone(), &self.q_v);

        // d2: commit Q, K, V (not CP-linked to Lasso)
        {
            let mut vars_2 = self.committed_vars_2.borrow_mut();
            for batch in &q_proc {
                for head in batch {
                    for seq in head {
                        vars_2.extend(seq.iter().cloned());
                    }
                }
            }
            for batch in &k_proc {
                for head in batch {
                    for seq in head {
                        vars_2.extend(seq.iter().cloned());
                    }
                }
            }
            for batch in &v_proc {
                for head in batch {
                    for seq in head {
                        vars_2.extend(seq.iter().cloned());
                    }
                }
            }
        }

        // Allocate QK^T output witnesses; ZenParams for QK^T (z_left=z_q, z_right=z_k)
        let (attn_data, params_attn) =
            self.matmul_attn.alloc_and_setup(cs.clone(), self.z_q, self.z_k, &two_k_var);
        // Allocate P*V output witnesses; ZenParams (z_left=0 for softmax output, z_right=z_v)
        let (x_data, params_x) =
            self.matmul_x.alloc_and_setup(cs.clone(), F::zero(), self.z_v, &two_k_var);

        // Softmax: mask and bundle vars
        let mask_var = new_matrix_witness(cs.clone(), &self.mask);
        let sm_vars = self.sm.alloc_vars(cs.clone())?;

        // ── Per-(batch, head) loop ────────────────────────────────────────────
        let mut global_attn_lows  = Vec::new();
        let mut global_attn_highs = Vec::new();
        let mut global_sm_idx   = vec![Vec::new(); 5];
        let mut global_sm_r_sm  = vec![Vec::new(); 2];
        let mut global_x_lows   = Vec::new();
        let mut global_x_highs  = Vec::new();

        let zero_var = FpVar::zero();

        for b in 0..q_proc.len() {
            for h in 0..q_proc[b].len() {
                // [1] QK^T: q_proc[B][H] [S×D] × k_proc[B][H]^T [D×S] → [S×S]
                let k_t = transpose_matrix(&k_proc[b][h]);
                let r_pows_attn =
                    gen_challenge_vector(&r_var, attn_data.q_sim[b][h][0].len());
                let mut h_attn_lows  = Vec::new();
                let mut h_attn_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &q_proc[b][h],
                    &k_t,
                    &attn_data.q_sim[b][h],
                    &attn_data.r_matmul[b][h],
                    &self.matmul_attn.r_matmul[b][h],
                    &params_attn,
                    &r_pows_attn,
                    &mut h_attn_lows,
                    &mut h_attn_highs,
                )?;
                global_attn_lows.extend(h_attn_lows);
                global_attn_highs.extend(h_attn_highs);

                // [2] Mask padding key positions in attention scores [S×S]
                let seq_len = attn_data.q_sim[b][h].len();
                let col_len = attn_data.q_sim[b][h][0].len();
                let mut masked_q_sim = attn_data.q_sim[b][h].clone();
                for row in 0..seq_len {
                    for col in 0..col_len {
                        let is_valid = mask_var[b][col].is_one()?;
                        masked_q_sim[row][col] = FpVar::conditionally_select(
                            &is_valid,
                            &attn_data.q_sim[b][h][row][col],
                            &zero_var,
                        )?;
                    }
                }

                // [3] Full-sequence softmax on masked [S×S] scores
                let mut h_sm_idx = [const { Vec::new() }; 5];
                let mut h_sm_r_sm = [const { Vec::new() }; 2];
                enforce_softmax(
                    cs.clone(),
                    &masked_q_sim,
                    &attn_data.r_matmul[b][h],
                    &sm_vars.r_sm[b][h],
                    &sm_vars.m_sm,
                    &sm_vars.limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.exp_limbs.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.q_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.r_exp.iter().map(|v| v[b][h].clone()).collect::<Vec<_>>(),
                    &sm_vars.p[b][h],
                    &sm_vars.r_div[b][h],
                    self.frac_bits,
                    &two_k_var,
                    &mut h_sm_idx,
                    &mut h_sm_r_sm,
                    &mut r2_pool,
                    &mut r3_pool,
                )?;
                for k in 0..5 { global_sm_idx[k].extend(h_sm_idx[k].clone()); }
                for k in 0..2 { global_sm_r_sm[k].extend(h_sm_r_sm[k].clone()); }

                // [4] P * V: sm_vars.p[B][H] [S×S] × v_proc[B][H] [S×D] → [S×D]
                let r_pows_x = gen_challenge_vector(&r_var, x_data.q_sim[b][h][0].len());
                let mut h_x_lows  = Vec::new();
                let mut h_x_highs = Vec::new();
                enforce_freivalds(
                    cs.clone(),
                    &sm_vars.p[b][h],
                    &v_proc[b][h],
                    &x_data.q_sim[b][h],
                    &x_data.r_matmul[b][h],
                    &self.matmul_x.r_matmul[b][h],
                    &params_x,
                    &r_pows_x,
                    &mut h_x_lows,
                    &mut h_x_highs,
                )?;
                global_x_lows.extend(h_x_lows);
                global_x_highs.extend(h_x_highs);
            }
        }

        // ── Populate d1 committed_vars ────────────────────────────────────────
        {
            let mut final_vars = self.committed_vars.borrow_mut();

            // Softmax index lookup vars (C=5 segments)
            for seg in 0..5 {
                final_vars.extend(global_sm_idx[seg].clone());
            }

            // QK^T matmul remainder splits (low / high)
            final_vars.extend(global_attn_lows);
            final_vars.extend(global_attn_highs);

            // Softmax r_sm (K=2 chunks)
            for k in 0..2 {
                final_vars.extend(global_sm_r_sm[k].clone());
            }

            // P*V matmul remainder splits
            final_vars.extend(global_x_lows);
            final_vars.extend(global_x_highs);

            // 16-bit remainder pools
            final_vars.extend(r2_pool); // from r_exp (5D)
            final_vars.extend(r3_pool); // from r_div (4D)
        }

        Ok(())
    }
}


pub fn load_bert_attention_circuit<E: Pairing>(
    path: &Path,
) -> Result<BERTAttentionCircuit<E::ScalarField>>
where
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let f = File::open(path).context(format!("opening BERT witness npz {:?}", path))?;
    let mut npz = NpzReader::new(f)?;

    // ── Inputs ───────────────────────────────────────────────────────────────
    let q_q_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_q.npy")?.into_dimensionality()?;
    let q_k_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_k.npy")?.into_dimensionality()?;
    let q_v_raw = npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_v.npy")?.into_dimensionality()?;
    let z_q = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_q.npy")?.into_scalar(),
    );
    let z_k = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_k.npy")?.into_scalar(),
    );
    let z_v = E::ScalarField::from(
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_v.npy")?.into_scalar(),
    );

    // Shape: Q/K/V must be [B, H, S, D]
    assert_eq!(q_q_raw.ndim(), 4, "Q_q must be 4D [B, H, S, D]");
    assert_eq!(q_k_raw.ndim(), 4, "Q_k must be 4D [B, H, S, D]");
    assert_eq!(q_v_raw.ndim(), 4, "Q_v must be 4D [B, H, S, D]");
    assert_eq!(
        q_q_raw.shape()[1], q_k_raw.shape()[1],
        "Q and K must have the same number of heads"
    );
    assert_eq!(
        q_q_raw.shape()[2], q_k_raw.shape()[2],
        "Q and K must have the same seq_len"
    );
    assert_eq!(
        q_q_raw.shape()[3], q_k_raw.shape()[3],
        "Q and K must have the same head_dim"
    );
    assert_eq!(
        q_q_raw.shape()[2], q_v_raw.shape()[2],
        "Q and V must have the same seq_len"
    );
    assert_eq!(
        q_q_raw.shape()[3], q_v_raw.shape()[3],
        "Q and V must have the same head_dim"
    );

    let b = q_q_raw.shape()[0];
    let h = q_q_raw.shape()[1];
    let s = q_q_raw.shape()[2];
    let d = q_q_raw.shape()[3];

    // ── QK^T matmul ──────────────────────────────────────────────────────────
    let q_sim_attn =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_attn.npy")?.into_dimensionality()?;
    let r_matmul_attn =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_attn.npy")?.into_dimensionality()?;

    // Shape: [B, H, S, S]
    assert_eq!(
        q_sim_attn.shape(), &[b, h, s, s],
        "Q_sim_attn must be [B={b}, H={h}, S={s}, S={s}]"
    );
    assert_eq!(
        r_matmul_attn.shape(), q_sim_attn.shape(),
        "R_matmul_attn must match Q_sim_attn shape"
    );

    let matmul_attn = MatmulBundle {
        r_mean:   vec![],
        q_sim:    to_field_4d_i64(&q_sim_attn),
        r_matmul: to_field_4d_i64(&r_matmul_attn),
        m:        E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_attn.npy")?.into_scalar(),
        ),
        m_prime:  E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_attn.npy")?.into_scalar(),
        ),
        z_sim:    E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_attn.npy")?.into_scalar(),
        ),
    };

    // ── Metadata ─────────────────────────────────────────────────────────────
    let frac_bits =
        npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("frac_bits.npy")?.into_scalar() as usize;
    let limb_bits =
        npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("limb_bits.npy")?.into_scalar() as usize;
    let k_segments =
        npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_segments.npy")?.into_scalar() as usize;
    let k_bits =
        npz.by_name::<ndarray::OwnedRepr<u32>, Ix0>("k_bits.npy")?.into_scalar() as usize;

    // ── Softmax ───────────────────────────────────────────────────────────────
    // 5D arrays: [K_SEGMENTS=5, B, H, S, S]
    let limbs_raw =
        npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("limbs.npy")?;
    let exp_limbs_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("exp_limbs.npy")?;
    let q_exp_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("Q_exp.npy")?;
    let r_exp_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("R_exp.npy")?;

    // limbs / exp_limbs: [K_SEGMENTS=5, B, H, S, S]
    assert_eq!(limbs_raw.ndim(), 5, "limbs must be 5D [C, B, H, S, S]");
    assert_eq!(
        &limbs_raw.shape()[1..], &[b, h, s, s],
        "limbs inner dims must be [B={b}, H={h}, S={s}, S={s}]"
    );
    assert_eq!(
        exp_limbs_raw.shape(), limbs_raw.shape(),
        "exp_limbs shape must match limbs [C, B, H, S, S]"
    );

    // Q_exp / R_exp: [K_SEGMENTS-1=4, B, H, S, S]
    // combine_exp_limbs produces one fewer segment than the raw limb count.
    assert_eq!(q_exp_raw.ndim(), 5, "Q_exp must be 5D");
    assert_eq!(
        &q_exp_raw.shape()[1..], &[b, h, s, s],
        "Q_exp inner dims must be [B={b}, H={h}, S={s}, S={s}]"
    );
    assert_eq!(
        r_exp_raw.shape(), q_exp_raw.shape(),
        "R_exp shape must match Q_exp shape"
    );

    // 4D softmax arrays: [B, H, S, S]
    let exp_total_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("exp_total.npy")?.into_dimensionality()?;
    let p_attn_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("P_attn.npy")?.into_dimensionality()?;
    let r_div_attn_raw =
        npz.by_name::<ndarray::OwnedRepr<u64>, Ix4>("R_div_attn.npy")?.into_dimensionality()?;
    let r_sm_raw =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("r_sm.npy")?.into_dimensionality()?;

    // Shape: P_attn must be [B, H, S, S]
    assert_eq!(
        p_attn_raw.shape(), &[b, h, s, s],
        "P_attn must be [B={b}, H={h}, S={s}, S={s}]"
    );
    assert_eq!(exp_total_raw.shape(), p_attn_raw.shape(), "exp_total shape must match P_attn");
    assert_eq!(r_div_attn_raw.shape(), p_attn_raw.shape(), "R_div_attn shape must match P_attn");
    assert_eq!(r_sm_raw.shape(), p_attn_raw.shape(), "r_sm shape must match P_attn");

    let sm = SoftmaxBundle {
        limbs:     to_field_5d_i64(&limbs_raw),
        exp_limbs: to_field_5d_u64(&exp_limbs_raw),
        q_exp:     to_field_5d_u64(&q_exp_raw),
        r_exp:     to_field_5d_u64(&r_exp_raw),
        exp_total: to_field_4d_u64(&exp_total_raw),
        p:         to_field_4d_u64(&p_attn_raw),
        r_div:     to_field_4d_u64(&r_div_attn_raw),
        r_sm:      to_field_4d_i64(&r_sm_raw),
        m_sm:      E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_sm.npy")?.into_scalar(),
        ),
    };

    // ── P*V output matmul ─────────────────────────────────────────────────────
    let q_sim_x =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("Q_sim_x.npy")?.into_dimensionality()?;
    let r_matmul_x =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix4>("R_matmul_x.npy")?.into_dimensionality()?;

    // Shape: [B, H, S, D]
    assert_eq!(
        q_sim_x.shape(), &[b, h, s, d],
        "Q_sim_x must be [B={b}, H={h}, S={s}, D={d}]"
    );
    assert_eq!(
        r_matmul_x.shape(), q_sim_x.shape(),
        "R_matmul_x must match Q_sim_x shape"
    );

    let matmul_x = MatmulBundle {
        r_mean:   vec![],
        q_sim:    to_field_4d_i64(&q_sim_x),
        r_matmul: to_field_4d_i64(&r_matmul_x),
        m:        E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_x.npy")?.into_scalar(),
        ),
        m_prime:  E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("M_prime_x.npy")?.into_scalar(),
        ),
        z_sim:    E::ScalarField::from(
            npz.by_name::<ndarray::OwnedRepr<u64>, Ix0>("z_sim_x.npy")?.into_scalar(),
        ),
    };

    // ── Mask ─────────────────────────────────────────────────────────────────
    let mask_raw =
        npz.by_name::<ndarray::OwnedRepr<i64>, Ix2>("mask.npy")?.into_dimensionality()?;

    // Shape: [B, S]
    assert_eq!(
        mask_raw.shape(), &[b, s],
        "mask must be [B={b}, S={s}]"
    );

    Ok(BERTAttentionCircuit {
        q_q: to_field_4d_i64(&q_q_raw),
        q_k: to_field_4d_i64(&q_k_raw),
        q_v: to_field_4d_i64(&q_v_raw),
        z_q,
        z_k,
        z_v,
        matmul_attn,
        sm,
        matmul_x,
        frac_bits,
        limb_bits,
        k_segments,
        r_challenge: E::ScalarField::zero(),
        k_bits,
        mask: to_field_2d_i64(&mask_raw),
        committed_vars:   Rc::new(RefCell::new(Vec::new())),
        committed_vars_2: Rc::new(RefCell::new(Vec::new())),
    })
}

// ── Proving pipeline ─────────────────────────────────────────────────────────

pub fn run_bert_attention_paths<E: Pairing>(targets: &[std::path::PathBuf]) -> Result<()>
where
    E: Pairing<ScalarField = Fr, G1 = ark_bn254::G1Projective, G1Affine = ark_bn254::G1Affine>,
    <E::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = StdRng::seed_from_u64(42u64);

    for file_path in targets {
        let e2e_start = Instant::now();
        println!("\n=== [BERT Attention] Processing: {:?} ===", file_path);

        // ── Dry-run: count committed witnesses ───────────────────────────────
        let (num_commit_1, num_commit_2) = {
            let temp = load_bert_attention_circuit::<E>(file_path)?;
            let v1_ptr = Rc::clone(&temp.committed_vars);
            let v2_ptr = Rc::clone(&temp.committed_vars_2);
            let cs = ConstraintSystem::<Fr>::new_ref();
            temp.generate_constraints(cs)?;
            (v1_ptr.borrow().len() as u32, v2_ptr.borrow().len() as u32)
        };
        println!("Commit counts: d1={}, d2={}", num_commit_1, num_commit_2);

        let sample_circuit = load_bert_attention_circuit::<E>(file_path)?;

        // ── [0] Setup ────────────────────────────────────────────────────────
        println!("=== [0] LegoGroth16 Setup (d1={}, d2={}) ===", num_commit_1, num_commit_2);
        let setup_start = Instant::now();
        let params = generate_random_parameters::<E, _, _>(
            sample_circuit.clone(),
            num_commit_1,
            num_commit_2,
            &mut rng,
        )
        .map_err(|e| anyhow::anyhow!("Setup failed: {:?}", e))?;
        let setup_time = setup_start.elapsed();

        let pvk = prepare_verifying_key::<E>(&params.vk);

        // ── Witness extraction ────────────────────────────────────────────────
        let cs_witness = ConstraintSystem::<Fr>::new_ref();
        sample_circuit.clone().generate_constraints(cs_witness.clone())?;
        let num_constraints = cs_witness.num_constraints();

        let committed_group1 = {
            let vars = sample_circuit.committed_vars.borrow();
            let values: Vec<Fr> =
                vars.iter().map(|v| v.value().unwrap_or(Fr::zero())).collect();
            if values.len() != num_commit_1 as usize {
                return Err(anyhow::anyhow!(
                    "d1 size mismatch: expected {}, got {}",
                    num_commit_1,
                    values.len()
                ));
            }
            values
        };

        println!(
            "[Debug] committed_group1 total={}",
            committed_group1.len()
        );

        let v1 = Fr::rand(&mut rng);
        let v2 = Fr::rand(&mut rng);
        let num_pub = cs_witness.num_instance_variables() - 1;

        // ── Groth16 commit (Pedersen cm1 from VK) ───────────────────────────
        let g16_commit_start = Instant::now();
        let cm1 = cm1_from_vk::<E>(&params.vk, num_pub, &committed_group1, &v1);
        let g16_commit_ms = g16_commit_start.elapsed().as_millis();

        let r_fs = hash_cm_to_field::<E>(&cm1);

        println!("=== [1] Rebuilding circuit with Fiat-Shamir challenge ===");
        let final_circuit = BERTAttentionCircuit {
            r_challenge: r_fs,
            ..sample_circuit.clone()
        };

        // ── [1] Groth16 Prove ────────────────────────────────────────────────
        let g16_prove_start = Instant::now();
        let proof = create_random_proof::<E, _, _>(
            final_circuit.clone(),
            v1,
            v2,
            &params,
            &mut rng,
        )
        .map_err(|e| anyhow::anyhow!("Proof generation failed: {:?}", e))?;
        let g16_prove_time = g16_prove_start.elapsed();

        let mut proof_bytes = Vec::new();
        proof
            .serialize_uncompressed(&mut proof_bytes)
            .map_err(|e| anyhow::anyhow!("Proof serialization failed: {:?}", e))?;
        let proof_size = proof_bytes.len();

        // ── [1] Groth16 Verify ───────────────────────────────────────────────
        let public_inputs = vec![final_circuit.r_challenge];
        let g16_verify_start = Instant::now();
        let result = verify_proof::<E>(&pvk, &proof, &public_inputs);
        let g16_verify_time = g16_verify_start.elapsed();

        println!("  - BERT Attention SNARK Verification: {:?}", result);
        if let Err(e) = result {
            return Err(anyhow::anyhow!(format!("{:?}", e)));
        }

        // ── Groth16 proof size report ──
        let g16_report = proof_size::compute_groth16_report::<E>(&proof);
        g16_report.print_log();

        // ── [2] Lasso (commit + prove + verify bundled in helpers) ───────────
        proof_size::reset_lasso_aggregator();
        println!("=== [2] BERT Attention Lasso Lookup Proofs ===");
        let lasso_start = Instant::now();
        let lasso_sm =
            run_bert_lasso_bundle(&final_circuit.sm, b"bert_lasso_sm", &mut rng)?;
        let lasso_remainders =
            run_all_bert_remainders_with_link(&final_circuit, &mut rng)?;
        let lasso_time = lasso_start.elapsed();

        let mut lasso_results = Vec::new();
        lasso_results.extend(lasso_sm);
        lasso_results.extend(lasso_remainders);

        // ── [3] CP-Link (setup/keygen, prove, verify timed separately) ───────
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
        println!("  -> [Success] Factored Multi-Lasso CP-Link verified");
        let cp_verify_ms = cp_verify_start.elapsed().as_millis();

        // ── CP-Link proof size report ──
        let cplink_report = proof_size::compute_cplink_report::<E>(&proof_link, &instance);
        cplink_report.print_log();

        // ── Grand total proof size report ──
        proof_size::print_full_report(&g16_report, &cplink_report);

        let (_, lasso_agg_bytes) = proof_size::get_lasso_aggregate();

        let e2e_time = e2e_start.elapsed();

        BenchmarkResult {
            file: file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            circuit:           "bert_attention".to_string(),
            constraints:       num_constraints,
            setup_ms:          setup_time.as_millis(),
            groth16_commit_ms: g16_commit_ms,
            groth16_prove_ms:  g16_prove_time.as_millis(),
            groth16_verify_ms: g16_verify_time.as_millis(),
            lasso_ms:          lasso_time.as_millis(),
            cp_link_setup_ms:  cp_setup_ms,
            cp_link_prove_ms:  cp_prove_ms,
            cp_link_verify_ms: cp_verify_ms,
            end_to_end_ms:     e2e_time.as_millis(),
            proof_size_bytes:  proof_size,
            groth16_commitment_size_bytes: groth16_commitment_size::<E>(&cm1),
            lasso_commitment_size_bytes:   lasso_commitment_size::<E>(&lasso_results),
            lasso_proof_size_bytes:  lasso_agg_bytes,
            cplink_proof_size_bytes: cplink_report.proof_total,
        }
        .print_json();
    }
    Ok(())
}


#[test]
fn bert_attn_hf_single_layer_test() {
    use std::env;

    // ── 환경 변수 ───────────────────────────────────────────────────────────
    let witness_dir = env::var("BERT_ATTN_WITNESS_DIR")
        .unwrap_or_else(|_| "outputs_bert".to_string());
    let layer_idx: Option<usize> = env::var("BERT_ATTN_LAYER_IDX")
        .ok()
        .and_then(|s| s.parse().ok());

    let dir = Path::new(&witness_dir);
    if !dir.is_dir() {
        eprintln!(
            "[skip] BERT_ATTN_WITNESS_DIR={} not a directory. \
             Set the env var to a valid HF BERT outputs_bert/ dir to run.",
            witness_dir
        );
        return;
    }

    // ── softmax 표 경로 결정: env var 미설정 시 witness_dir 안의 softmax_tables.npz ─
    if env::var("SOFTMAX_TABLE_PATH").is_err() {
        set_softmax_table_dir(dir);
        println!(
            "[info] SOFTMAX_TABLE_PATH not set; using {}/softmax_tables.npz",
            witness_dir
        );
    }

    // ── npz 수집 + 정렬 + layer_idx 단일 추출 ──────────────────────────────
    let mut all_targets: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("reading witness dir") {
        let path = entry.expect("dir entry").path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.ends_with("attention.self.witness.npz"))
        {
            all_targets.push(path);
        }
    }
    all_targets.sort();

    if all_targets.is_empty() {
        eprintln!(
            "[skip] no '*attention.self.witness.npz' files in {}. \
             Run HF BERT attention pipeline first.",
            witness_dir
        );
        return;
    }

    let targets = if let Some(idx) = layer_idx {
        if idx >= all_targets.len() {
            panic!(
                "BERT_ATTN_LAYER_IDX={} out of range (have {} layers).",
                idx,
                all_targets.len()
            );
        }
        vec![all_targets[idx].clone()]
    } else {
        all_targets
    };

    println!(
        "\n=== [BERT Attention prove+verify] dir={} | files={} | mode={} ===",
        witness_dir,
        targets.len(),
        if layer_idx.is_some() { "single-layer" } else { "all-layers" },
    );

    let total_start = std::time::Instant::now();
    let res = run_bert_attention_paths::<Bn254>(&targets);
    let total_ms = total_start.elapsed().as_millis();

    match res {
        Ok(()) => println!(
            "\n=== [BERT Attention] ALL {} files PASS | total_wall_ms={} ===",
            targets.len(), total_ms,
        ),
        Err(e) => panic!(
            "[BERT Attention] proof generate/verify failed after {}ms: {}",
            total_ms, e,
        ),
    }
}
