use crate::cc_snark::legogroth16::{
    Proof, ProofWithLink, ProvingKey, ProvingKeyCommon, ProvingKeyWithLink, VerifyingKey,
    VerifyingKeyWithLink,
    link::{PESubspaceSnark, SubspaceSnark},
    r1cs_to_qap::LibsnarkReduction,
};
use crate::cc_snark::legogroth16::{Result, error::Error, r1cs_to_qap::R1CStoQAP};
use ark_ec::{
    PrimeGroup,
    AffineRepr, CurveGroup, VariableBaseMSM, pairing::Pairing,
};
use ark_ff::{Field, PrimeField, UniformRand, Zero};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, OptimizationGoal, SynthesisError,
};
use ark_std::{
    cfg_into_iter, cfg_iter, end_timer,
    ops::{AddAssign, Mul},
    rand::Rng,
    start_timer,
    vec::Vec,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Same as `create_random_proof` but returns the CP_link and its corresponding proof as well.
/// `link_v` is the blinding in CP_link.
/// `v1` is blinding for d1 (linked), `v2` is blinding for d2.
#[inline]
pub fn create_random_proof_incl_cp_link<E, C, R>(
    circuit: C,
    v1: E::ScalarField,
    v2: E::ScalarField,
    link_v: E::ScalarField,
    pk: &ProvingKeyWithLink<E>,
    rng: &mut R,
) -> Result<ProofWithLink<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
{
    let r = E::ScalarField::rand(rng);
    let s = E::ScalarField::rand(rng);
    create_proof_incl_cp_link::<E, C>(circuit, pk, r, s, v1, v2, link_v)
}

/// Create a LegoGroth16 proof that is zero-knowledge.
/// `v1` and `v2` are blinkings used in the commitment to the witness groups 1 and 2.
#[inline]
pub fn create_random_proof<E, C, R>(
    circuit: C,
    v1: E::ScalarField,
    v2: E::ScalarField,
    pk: &ProvingKey<E>,
    rng: &mut R,
) -> Result<Proof<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
{
    let r = E::ScalarField::rand(rng);
    let s = E::ScalarField::rand(rng);
    create_proof::<E, C>(circuit, pk, r, s, v1, v2)
}

#[inline]
/// Create a LegoGroth16 proof using randomness `r`, `s`, `v1`, `v2` and `link_v`.
pub fn create_proof_incl_cp_link<E, C>(
    circuit: C,
    pk: &ProvingKeyWithLink<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
    link_v: E::ScalarField,
) -> Result<ProofWithLink<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
{
    create_proof_incl_cp_link_with_reduction::<E, C, LibsnarkReduction>(
        circuit, pk, r, s, v1, v2, link_v,
    )
}

#[inline]
/// Create a LegoGroth16 proof using randomness `r`, `s`, `v1`, `v2`.
pub fn create_proof<E, C>(
    circuit: C,
    pk: &ProvingKey<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
) -> Result<Proof<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
{
    create_proof_with_reduction::<E, C, LibsnarkReduction>(circuit, pk, r, s, v1, v2)
}

#[inline]
pub fn create_proof_incl_cp_link_with_reduction<E, C, QAP>(
    circuit: C,
    pk: &ProvingKeyWithLink<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
    link_v: E::ScalarField,
) -> Result<ProofWithLink<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    QAP: R1CStoQAP,
{
    let prover_time = start_timer!(|| "Groth16::Prover");
    let (cs, h) = synthesize_circuit::<E, C, QAP>(circuit)?;
    let prover = cs.borrow().unwrap();
    let proof = create_proof_incl_cp_link_with_assignment::<E, QAP>(
        pk,
        r,
        s,
        v1,
        v2,
        link_v,
        &h,
        &prover.instance_assignment,
        &prover.witness_assignment,
    )?;
    drop(prover);
    drop(cs);
    end_timer!(prover_time);
    Ok(proof)
}

#[inline]
pub fn create_proof_with_reduction<E, C, QAP>(
    circuit: C,
    pk: &ProvingKey<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
) -> Result<Proof<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    QAP: R1CStoQAP,
{
    let prover_time = start_timer!(|| "Groth16::Prover");
    let (cs, h) = synthesize_circuit::<E, C, QAP>(circuit)?;
    let prover = cs.borrow().unwrap();
    let proof = create_proof_with_assignment::<E, QAP>(
        pk,
        r,
        s,
        v1,
        v2,
        &h,
        &prover.instance_assignment,
        &prover.witness_assignment,
    )?;
    drop(prover);
    drop(cs);
    end_timer!(prover_time);
    Ok(proof)
}

#[inline]
fn create_proof_incl_cp_link_with_assignment<E, QAP>(
    pk: &ProvingKeyWithLink<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
    link_v: E::ScalarField,
    h: &[E::ScalarField],
    input_assignment: &[E::ScalarField],
    witness_assignment: &[E::ScalarField],
) -> Result<ProofWithLink<E>>
where
    E: Pairing,
    QAP: R1CStoQAP,
{
    // Returns proof and (witnesses_1, witnesses_2)
    let (proof, comm_wits_1, _comm_wits_2) =
        create_proof_and_committed_witnesses_with_assignment::<E, QAP>(
            &pk.common,
            &pk.vk.groth16_vk,
            r,
            s,
            v1,
            v2,
            &h,
            input_assignment,
            witness_assignment,
        )?;

    // Link logic: Currently linking Group 1 (d1)
    let mut comm_wits_with_link_hider = cfg_iter!(comm_wits_1)
        .map(|w| w.into_bigint())
        .collect::<Vec<_>>();
    comm_wits_with_link_hider.push(link_v.into_bigint());

    let g_d_link = E::G1::msm_bigint(&pk.vk.link_bases, &comm_wits_with_link_hider);

    let mut ss_snark_witness = comm_wits_1;
    ss_snark_witness.push(link_v);
    ss_snark_witness.push(v1); // Linking to d1's randomness

    let link_time = start_timer!(|| "Compute CP_{link}");
    let link_pi = PESubspaceSnark::<E>::prove(&pk.vk.link_pp, &pk.link_ek, &ss_snark_witness)?;
    end_timer!(link_time);

    drop(comm_wits_with_link_hider);
    drop(ss_snark_witness);

    Ok(ProofWithLink {
        groth16_proof: proof,
        link_d: g_d_link.into_affine(),
        link_pi,
    })
}

#[inline]
fn create_proof_with_assignment<E, QAP>(
    pk: &ProvingKey<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
    h: &[E::ScalarField],
    input_assignment: &[E::ScalarField],
    witness_assignment: &[E::ScalarField],
) -> Result<Proof<E>>
where
    E: Pairing,
    QAP: R1CStoQAP,
{
    let (proof, _, _) = create_proof_and_committed_witnesses_with_assignment::<E, QAP>(
        &pk.common,
        &pk.vk,
        r,
        s,
        v1,
        v2,
        &h,
        input_assignment,
        witness_assignment,
    )?;
    Ok(proof)
}

/// Returns the proof and the committed witnesses (Group 1, Group 2).
#[inline]
fn create_proof_and_committed_witnesses_with_assignment<E, QAP>(
    pk_common: &ProvingKeyCommon<E>,
    vk: &VerifyingKey<E>,
    r: E::ScalarField,
    s: E::ScalarField,
    v1: E::ScalarField,
    v2: E::ScalarField,
    h: &[E::ScalarField],
    input_assignment: &[E::ScalarField],
    witness_assignment: &[E::ScalarField],
) -> Result<(Proof<E>, Vec<E::ScalarField>, Vec<E::ScalarField>)>
where
    E: Pairing,
    QAP: R1CStoQAP,
{
    let h_assignment = cfg_into_iter!(h)
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();
    let c_acc_time = start_timer!(|| "Compute C");
    let h_acc = E::G1::msm_bigint(&pk_common.h_query, &h_assignment);
    drop(h_assignment);

    let v1_repr = v1.into_bigint();
    let v2_repr = v2.into_bigint();

    // Compute C
    // witness assignment = [committed_1 | committed_2 | uncommitted]
    let aux_assignment_time = start_timer!(|| "Process witness assignment");
    let aux_assignment = cfg_iter!(witness_assignment)
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();
    end_timer!(aux_assignment_time);

    let count_1 = vk.commit_witness_count_1 as usize;
    let count_2 = vk.commit_witness_count_2 as usize;

    let committed_witnesses_1 = &aux_assignment[..count_1];
    let committed_witnesses_2 = &aux_assignment[count_1..count_1 + count_2];
    let uncommitted_witnesses = &aux_assignment[count_1 + count_2..];

    let l_aux_acc_time = start_timer!(|| "Compute L for aux");
    let l_aux_acc = E::G1::msm_bigint(&pk_common.l_query, uncommitted_witnesses);
    end_timer!(l_aux_acc_time);

    let v_eta_delta_inv_time = start_timer!(|| "Compute (v1*eta1/delta + v2*eta2/delta) * G1");
    // C calculation must subtract both hiding factors scaled by eta/delta
    let mut v_eta_delta_inv: <E as Pairing>::G1 = pk_common.eta1_delta_inv_g1.mul_bigint(v1_repr);
    v_eta_delta_inv += pk_common.eta2_delta_inv_g1.mul_bigint(v2_repr);
    end_timer!(v_eta_delta_inv_time);
    end_timer!(c_acc_time);

    let s_repr = s.into_bigint();
    let delta_g1_proj = pk_common.delta_g1.into_group();

    let window_size = 3;
    let scalar_size =
        <<E as Pairing>::G1Affine as AffineRepr>::ScalarField::MODULUS_BIT_SIZE as usize;
    let outerc = (scalar_size + window_size - 1) / window_size;
    //let delta_g1_table = FixedBase::get_window_table(scalar_size, window_size, delta_g1_proj);

    let input_assignment_wth_one = cfg_iter!(input_assignment)
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();

    let assignment = [&input_assignment_wth_one[1..], &aux_assignment].concat();

    // Compute A
    let a_acc_time = start_timer!(|| "Compute A");
    // FixedBase::windowed_mul 대신 단순 곱셈 사용 (0.5.0에서는 g * s가 최적화됨)
    let r_g1 = pk_common.delta_g1 * r; 
    let g_a = calculate_coeff(r_g1, &pk_common.a_query, vk.alpha_g1, &assignment);
    end_timer!(a_acc_time);

    // Compute B in G1 if needed
    let g1_b = if !r.is_zero() {
        let b_g1_acc_time = start_timer!(|| "Compute B in G1");
        let s_g1 = pk_common.delta_g1 * s;
        let g1_b = calculate_coeff(s_g1, &pk_common.b_g1_query, pk_common.beta_g1, &assignment);
        end_timer!(b_g1_acc_time);
        g1_b
    } else {
        E::G1::zero()
    };

    // Compute B in G2
    let b_g2_acc_time = start_timer!(|| "Compute B in G2");
    let s_g2 = vk.delta_g2.into_group().mul_bigint(s_repr);
    let g2_b = calculate_coeff(s_g2, &pk_common.b_g2_query, vk.beta_g2, &assignment);
    drop(assignment);
    end_timer!(b_g2_acc_time);

    let c_time = start_timer!(|| "Finish C");
    let mut g_c = g_a.mul_bigint(s.into_bigint());
    g_c += &g1_b.mul_bigint(r.into_bigint());
    // FixedBase::windowed_mul(..., &(r * s)) 대신 직접 곱셈
    g_c -= &(pk_common.delta_g1 * (r * s)); 
    g_c += &l_aux_acc;
    g_c += &h_acc;
    g_c -= &v_eta_delta_inv;
    end_timer!(c_time);

    // Compute D1 (Group 1)
    let d1_acc_time = start_timer!(|| "Compute D1");
    // Offset for Group 1 in gamma_abc is just after public inputs
    let offset_1 = input_assignment_wth_one.len();
    let gamma_abc_inputs_source_1 = &vk.gamma_abc_g1[offset_1..offset_1 + count_1];
    let gamma_abc_inputs_acc_1 =
        E::G1::msm_bigint(gamma_abc_inputs_source_1, &committed_witnesses_1);

    let v1_eta_gamma_inv = vk.eta1_gamma_inv_g1.into_group().mul_bigint(v1_repr);
    let mut g_d1 = gamma_abc_inputs_acc_1;
    g_d1 += &v1_eta_gamma_inv;
    end_timer!(d1_acc_time);

    // Compute D2 (Group 2)
    let d2_acc_time = start_timer!(|| "Compute D2");
    // Offset for Group 2 in gamma_abc is after public + Group 1
    let offset_2 = offset_1 + count_1;
    let gamma_abc_inputs_source_2 = &vk.gamma_abc_g1[offset_2..offset_2 + count_2];
    let gamma_abc_inputs_acc_2 =
        E::G1::msm_bigint(gamma_abc_inputs_source_2, &committed_witnesses_2);

    let v2_eta_gamma_inv = vk.eta2_gamma_inv_g1.into_group().mul_bigint(v2_repr);
    let mut g_d2 = gamma_abc_inputs_acc_2;
    g_d2 += &v2_eta_gamma_inv;
    end_timer!(d2_acc_time);

    let committed_wits_vec_1 = witness_assignment[..count_1].to_vec();
    let committed_wits_vec_2 = witness_assignment[count_1..count_1 + count_2].to_vec();
    drop(aux_assignment);

    Ok((
        Proof {
            a: g_a.into_affine(),
            b: g2_b.into_affine(),
            c: g_c.into_affine(),
            d1: g_d1.into_affine(),
            d2: g_d2.into_affine(),
        },
        committed_wits_vec_1,
        committed_wits_vec_2,
    ))
}

/// Check the opening of cp_link.
pub fn verify_link_commitment<E: Pairing>(
    cp_link_bases: &[E::G1Affine],
    link_d: &E::G1Affine,
    witnesses_expected_in_commitment: &[E::ScalarField],
    link_v: &E::ScalarField,
) -> Result<()> {
    if (witnesses_expected_in_commitment.len() + 1) > cp_link_bases.len() {
        return Err(Error::VectorLongerThanExpected(
            witnesses_expected_in_commitment.len() + 1,
            cp_link_bases.len(),
        ));
    }
    let mut committed = cfg_iter!(witnesses_expected_in_commitment)
        .map(|p| p.into_bigint())
        .collect::<Vec<_>>();
    committed.push(link_v.into_bigint());
    if *link_d != E::G1::msm_bigint(cp_link_bases, &committed).into_affine() {
        return Err(Error::InvalidLinkCommitment);
    }
    Ok(())
}

/// Check that the commitments in the proof open correctly.
/// `witnesses_expected_in_commitment`: Group 1 witnesses.
/// `v1`: Blinding for Group 1.
pub fn verify_commitments_group_1<E: Pairing>(
    vk: &VerifyingKeyWithLink<E>,
    proof: &ProofWithLink<E>,
    public_inputs_count: usize,
    witnesses_expected_in_commitment: &[E::ScalarField],
    v1: &E::ScalarField,
    link_v: &E::ScalarField,
) -> Result<()> {
    verify_link_commitment::<E>(
        &vk.link_bases,
        &proof.link_d,
        witnesses_expected_in_commitment,
        link_v,
    )?;
    verify_witness_commitment_1::<E>(
        &vk.groth16_vk,
        &proof.groth16_proof,
        public_inputs_count,
        witnesses_expected_in_commitment,
        v1,
    )
}

/// Given the proof, verify that `proof.d1` commits to the Group 1 witnesses.
pub fn verify_witness_commitment_1<E: Pairing>(
    vk: &VerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs_count: usize,
    witnesses_1: &[E::ScalarField],
    v1: &E::ScalarField,
) -> Result<()> {
    if (public_inputs_count + witnesses_1.len()) > vk.gamma_abc_g1.len() {
        return Err(Error::VectorLongerThanExpected(
            public_inputs_count + witnesses_1.len(),
            vk.gamma_abc_g1.len(),
        ));
    }
    let committed = cfg_iter!(witnesses_1)
        .map(|p| p.into_bigint())
        .collect::<Vec<_>>();

    // Group 1 starts after Public Inputs
    let start = 1 + public_inputs_count;
    let end = start + committed.len();

    let mut d = E::G1::msm_bigint(&vk.gamma_abc_g1[start..end], &committed);
    d.add_assign(&vk.eta1_gamma_inv_g1.mul_bigint(v1.into_bigint()));
    if proof.d1 != d.into_affine() {
        return Err(Error::InvalidWitnessCommitment);
    }
    Ok(())
}

/// Given the proof, verify that `proof.d2` commits to the Group 2 witnesses.
pub fn verify_witness_commitment_2<E: Pairing>(
    vk: &VerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs_count: usize,
    witnesses_2: &[E::ScalarField],
    v2: &E::ScalarField,
) -> Result<()> {
    let offset_1 = vk.commit_witness_count_1 as usize;
    if (public_inputs_count + offset_1 + witnesses_2.len()) > vk.gamma_abc_g1.len() {
        return Err(Error::VectorLongerThanExpected(
            public_inputs_count + offset_1 + witnesses_2.len(),
            vk.gamma_abc_g1.len(),
        ));
    }
    let committed = cfg_iter!(witnesses_2)
        .map(|p| p.into_bigint())
        .collect::<Vec<_>>();

    // Group 2 starts after Public Inputs + Group 1
    let start = 1 + public_inputs_count + offset_1;
    let end = start + committed.len();

    let mut d = E::G1::msm_bigint(&vk.gamma_abc_g1[start..end], &committed);
    d.add_assign(&vk.eta2_gamma_inv_g1.mul_bigint(v2.into_bigint()));
    if proof.d2 != d.into_affine() {
        return Err(Error::InvalidWitnessCommitment);
    }
    Ok(())
}

/// Given a LegoGroth16 proof, returns a fresh proof of the same statement.
/// This breaks the property that `proof.d1` and `proof.d2` are valid commitments to the witnesses,
/// so it is useful only when unlinkability is required but commitment reuse is not.
pub fn rerandomize_proof<E, R>(proof: &Proof<E>, vk: &VerifyingKey<E>, rng: &mut R) -> Proof<E>
where
    E: Pairing,
    R: Rng,
{
    // These are our rerandomization factors. They must be nonzero and uniformly sampled.
    let (mut r1, mut r2) = (E::ScalarField::zero(), E::ScalarField::zero());
    while r1.is_zero() || r2.is_zero() {
        r1 = E::ScalarField::rand(rng);
        r2 = E::ScalarField::rand(rng);
    }

    // We split r2 into r3 and r4 such that r3 + r4 = r2.
    // This ensures that the total addition to D (d1 + d2) corresponds to r2,
    // while randomizing both d1 and d2 individually to ensure unlinkability.
    let r3 = E::ScalarField::rand(rng);
    let r4 = r2 - r3;

    //   A' = (1/r₁)A
    //   B' = r₁B + r₁r₂(δG₂) + r₁r₂(γG₂)
    //   C' = C + r₂A
    //   D1' = D1 + r₃A  \
    //                    > Sum adds r₂A
    //   D2' = D2 + r₄A  /

    let new_a = proof.a.mul(r1.inverse().unwrap());

    // Update B with terms for both delta and gamma
    let new_b = proof.b.mul(r1) + (vk.delta_g2 + vk.gamma_g2).mul(r1 * &r2);

    let a_r2 = proof.a.mul(r2).into_affine();
    let new_c = proof.c + a_r2;

    // Distribute r2*A across d1 and d2
    let a_r3 = proof.a.mul(r3).into_affine();
    let a_r4 = proof.a.mul(r4).into_affine();

    let new_d1 = proof.d1 + a_r3;
    let new_d2 = proof.d2 + a_r4;

    Proof {
        a: new_a.into_affine(),
        b: new_b.into_affine(),
        c: new_c.into_affine(),
        d1: new_d1.into_affine(),
        d2: new_d2.into_affine(),
    }
}

/// Rerandomize proof. Keeps D1, D2 valid commitments by adjusting based on v changes.
pub fn rerandomize_proof_1<E, R>(
    proof: &Proof<E>,
    old_v1: E::ScalarField,
    new_v1: E::ScalarField,
    old_v2: E::ScalarField,
    new_v2: E::ScalarField,
    vk: &VerifyingKey<E>,
    eta1_delta_inv_g1: &E::G1Affine,
    eta2_delta_inv_g1: &E::G1Affine,
    rng: &mut R,
) -> Proof<E>
where
    E: Pairing,
    R: Rng,
{
    let (mut r1, mut r2) = (E::ScalarField::zero(), E::ScalarField::zero());
    while r1.is_zero() || r2.is_zero() {
        r1 = E::ScalarField::rand(rng);
        r2 = E::ScalarField::rand(rng);
    }

    let new_a = proof.a.mul(r1.inverse().unwrap());
    let new_b = proof.b.mul(r1) + vk.delta_g2.mul(r1 * &r2);
    let a_r2 = proof.a.mul(r2).into_affine();

    // Update C with both v1 and v2 diffs
    let mut new_c_group = proof.c + a_r2;
    new_c_group += eta1_delta_inv_g1.mul(old_v1 - new_v1);
    new_c_group += eta2_delta_inv_g1.mul(old_v2 - new_v2);
    let new_c = new_c_group.into_affine();

    // Update D1
    let new_d1 = (proof.d1 + vk.eta1_gamma_inv_g1.mul(new_v1 - old_v1)).into_affine();

    // Update D2
    let new_d2 = (proof.d2 + vk.eta2_gamma_inv_g1.mul(new_v2 - old_v2)).into_affine();

    Proof {
        a: new_a.into_affine(),
        b: new_b.into_affine(),
        c: new_c,
        d1: new_d1,
        d2: new_d2,
    }
}

// synthesize_circuit and calculate_coeff remain largely unchanged
#[inline]
pub fn synthesize_circuit<E, C, QAP>(
    circuit: C,
) -> Result<(ConstraintSystemRef<E::ScalarField>, Vec<E::ScalarField>)>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    QAP: R1CStoQAP,
{
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Weight);

    let synthesis_time = start_timer!(|| "Constraint synthesis");
    circuit.generate_constraints(cs.clone())?;
    #[cfg(debug_assertions)]
    {
        if !cs.is_satisfied()? {
            return Err(Error::SynthesisError(SynthesisError::Unsatisfiable));
        }
    }
    end_timer!(synthesis_time);

    let lc_time = start_timer!(|| "Inlining LCs");
    cs.finalize();
    end_timer!(lc_time);

    let witness_map_time = start_timer!(|| "R1CS to QAP witness map");
    let h =
        QAP::witness_map::<E::ScalarField, GeneralEvaluationDomain<E::ScalarField>>(cs.clone())?;
    end_timer!(witness_map_time);
    Ok((cs, h))
}

fn calculate_coeff<G: AffineRepr>(
    initial: G::Group,
    query: &[G],
    vk_param: G,
    assignment: &[<G::ScalarField as PrimeField>::BigInt],
) -> G::Group {
    let el = query[0];
    let acc = G::Group::msm_bigint(&query[1..], assignment);
    initial + el + acc + vk_param
}
