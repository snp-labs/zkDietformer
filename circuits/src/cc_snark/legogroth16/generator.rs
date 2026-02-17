use crate::cc_snark::legogroth16::{
    LinkPublicGenerators, ProvingKey, ProvingKeyCommon, ProvingKeyWithLink, Result, Vec,
    VerifyingKey, VerifyingKeyWithLink,
    link::{PESubspaceSnark, PP, SparseMatrix, SubspaceSnark},
    r1cs_to_qap::LibsnarkReduction,
};
use ark_ec::{CurveGroup, pairing::Pairing, PrimeGroup, VariableBaseMSM};
use ark_ff::{Field, PrimeField, UniformRand, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisError, SynthesisMode,
};
use ark_std::{cfg_into_iter, cfg_iter, end_timer, rand::Rng, start_timer, vec};

use crate::cc_snark::legogroth16::r1cs_to_qap::R1CStoQAP;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline]
/// Generates random parameters including CP_link.
/// `commit_witness_count_1`: number of witnesses in Group 1 (linked/committed).
/// `commit_witness_count_2`: number of witnesses in Group 2 (committed).
pub fn generate_random_parameters_incl_cp_link<E, C, R>(
    circuit: C,
    link_gens: LinkPublicGenerators<E>,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<ProvingKeyWithLink<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
{
    generate_random_parameters_incl_cp_link_with_reduction::<E, C, R, LibsnarkReduction>(
        circuit,
        link_gens,
        commit_witness_count_1,
        commit_witness_count_2,
        rng,
    )
}

#[inline]
/// Generates random parameters.
pub fn generate_random_parameters<E, C, R>(
    circuit: C,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<ProvingKey<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
{
    generate_random_parameters_with_reduction::<E, C, R, LibsnarkReduction>(
        circuit,
        commit_witness_count_1,
        commit_witness_count_2,
        rng,
    )
}

#[inline]
pub fn generate_random_parameters_with_reduction<E, C, R, QAP>(
    circuit: C,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<ProvingKey<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
    QAP: R1CStoQAP,
{
    // Modified: Generate eta1, eta2
    let (alpha, beta, gamma, delta, eta1, eta2, g1_generator, g2_generator) =
        generate_randomness::<E, R>(rng);

    generate_parameters_with_qap::<E, C, R, QAP>(
        circuit,
        alpha,
        beta,
        gamma,
        delta,
        eta1,
        eta2,
        g1_generator,
        g2_generator,
        commit_witness_count_1,
        commit_witness_count_2,
        rng,
    )
}

#[inline]
pub fn generate_random_parameters_incl_cp_link_with_reduction<E, C, R, QAP>(
    circuit: C,
    link_gens: LinkPublicGenerators<E>,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<ProvingKeyWithLink<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
    QAP: R1CStoQAP,
{
    // Modified: Generate eta1, eta2
    let (alpha, beta, gamma, delta, eta1, eta2, g1_generator, g2_generator) =
        generate_randomness::<E, R>(rng);

    generate_parameters_incl_cp_link_with_qap::<E, C, R, QAP>(
        circuit,
        alpha,
        beta,
        gamma,
        delta,
        eta1,
        eta2,
        g1_generator,
        g2_generator,
        link_gens,
        commit_witness_count_1,
        commit_witness_count_2,
        rng,
    )
}

#[inline]
pub fn generate_parameters_incl_cp_link_with_qap<E, C, R, QAP>(
    circuit: C,
    alpha: E::ScalarField,
    beta: E::ScalarField,
    gamma: E::ScalarField,
    delta: E::ScalarField,
    eta1: E::ScalarField, // Modified
    eta2: E::ScalarField, // Modified
    g1_generator: E::G1,
    g2_generator: E::G2,
    link_gens: LinkPublicGenerators<E>,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<ProvingKeyWithLink<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
    QAP: R1CStoQAP,
{
    let (groth16_pk, num_instance_variables) =
        generate_parameters_and_extra_info_with_qap::<E, C, R, QAP>(
            circuit,
            alpha,
            beta,
            gamma,
            delta,
            eta1,
            eta2,
            g1_generator,
            g2_generator,
            commit_witness_count_1,
            commit_witness_count_2,
            rng,
        )?;

    // Setup public params for the Subspace Snark
    // NOTE: This implementation assumes CP_Link is connected to Group 1 witnesses (d1).
    // If you need to link Group 2 or both, the matrix construction below needs adjustment.

    let link_rows = 2; // comparing d1 and link_d
    let link_cols = commit_witness_count_1 as u32 + 2; // Group 1 witnesses + 1 hiding factor
    let link_pp = PP::<E::G1Affine, E::G2Affine> {
        l: link_rows,
        t: link_cols,
        g1: link_gens.g1,
        g2: link_gens.g2,
    };

    let mut link_m = SparseMatrix::<E::G1Affine>::new(link_rows as usize, link_cols as usize);
    link_m.insert_row_slice(0, 0, link_gens.pedersen_gens.clone())?;

    // Insert Gamma ABC corresponding to Group 1
    link_m.insert_row_slice(
        1,
        0,
        groth16_pk.vk.gamma_abc_g1
            [num_instance_variables..num_instance_variables + commit_witness_count_1 as usize]
            .to_vec(),
    )?;
    // Insert Eta1 for Group 1
    link_m.insert_row_slice(
        1,
        commit_witness_count_1 as usize + 1,
        vec![groth16_pk.vk.eta1_gamma_inv_g1],
    )?;

    let (link_ek, link_vk) = PESubspaceSnark::<E>::keygen(rng, &link_pp, &link_m)?;

    let vk = VerifyingKeyWithLink::<E> {
        groth16_vk: groth16_pk.vk,
        link_pp,
        link_bases: link_gens.pedersen_gens,
        link_vk,
    };

    Ok(ProvingKeyWithLink {
        vk,
        common: groth16_pk.common,
        link_ek,
    })
}

#[inline]
pub fn generate_parameters_with_qap<E, C, R, QAP>(
    circuit: C,
    alpha: E::ScalarField,
    beta: E::ScalarField,
    gamma: E::ScalarField,
    delta: E::ScalarField,
    eta1: E::ScalarField,
    eta2: E::ScalarField,
    g1_generator: E::G1,
    g2_generator: E::G2,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<ProvingKey<E>>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
    QAP: R1CStoQAP,
{
    let (pk, _) = generate_parameters_and_extra_info_with_qap::<E, C, R, QAP>(
        circuit,
        alpha,
        beta,
        gamma,
        delta,
        eta1,
        eta2,
        g1_generator,
        g2_generator,
        commit_witness_count_1,
        commit_witness_count_2,
        rng,
    )?;
    Ok(pk)
}

/// Create parameters for a circuit, given some toxic waste, R1CS to QAP calculator and group generators.
#[inline]
fn generate_parameters_and_extra_info_with_qap<E, C, R, QAP>(
    circuit: C,
    alpha: E::ScalarField,
    beta: E::ScalarField,
    gamma: E::ScalarField,
    delta: E::ScalarField,
    eta1: E::ScalarField, // For Group 1
    eta2: E::ScalarField, // For Group 2
    g1_generator: E::G1,
    g2_generator: E::G2,
    commit_witness_count_1: u32,
    commit_witness_count_2: u32,
    rng: &mut R,
) -> Result<(ProvingKey<E>, usize)>
where
    E: Pairing,
    C: ConstraintSynthesizer<E::ScalarField>,
    R: Rng,
    QAP: R1CStoQAP,
{
    type D<F> = GeneralEvaluationDomain<F>;

    let setup_time = start_timer!(|| "Groth16::Generator");
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Weight);
    cs.set_mode(SynthesisMode::Setup);

    let synthesis_time = start_timer!(|| "Constraint synthesis");
    circuit.generate_constraints(cs.clone())?;
    end_timer!(synthesis_time);

    let lc_time = start_timer!(|| "Inlining LCs");
    cs.finalize();
    end_timer!(lc_time);

    let domain_time = start_timer!(|| "Constructing evaluation domain");
    let domain_size = cs.num_constraints() + cs.num_instance_variables();
    let domain = D::new(domain_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    let t = domain.sample_element_outside_domain(rng);
    end_timer!(domain_time);

    let num_instance_variables = cs.num_instance_variables();
    // Check if we have enough witnesses for both groups
    if cs.num_witness_variables() < (commit_witness_count_1 + commit_witness_count_2) as usize {
        // You might want to update the Error enum to reflect 2 args,
        // passing total sum for now to match original Error struct
        return Err(
            crate::cc_snark::legogroth16::error::Error::InsufficientWitnessesForCommitment(
                cs.num_witness_variables(),
                (commit_witness_count_1 + commit_witness_count_2) as usize,
            ),
        );
    }

    // n = Public + Group 1 + Group 2
    let n =
        num_instance_variables + commit_witness_count_1 as usize + commit_witness_count_2 as usize;

    let reduction_time = start_timer!(|| "R1CS to QAP Instance Map with Evaluation");
    let (a, b, c, zt, qap_num_variables, m_raw) = LibsnarkReduction::instance_map_with_evaluation::<
        E::ScalarField,
        D<E::ScalarField>,
    >(cs, &t)?;
    end_timer!(reduction_time);

    // Compute query densities
    let non_zero_a: usize = cfg_into_iter!(0..qap_num_variables)
        .map(|i| usize::from(!a[i].is_zero()))
        .sum();

    let non_zero_b: usize = cfg_into_iter!(0..qap_num_variables)
        .map(|i| usize::from(!b[i].is_zero()))
        .sum();

    let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;

    let gamma_inverse = gamma.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;
    let delta_inverse = delta.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;

    // Calculate Gamma ABC
    // This includes Public Inputs, then Group 1 Witnesses, then Group 2 Witnesses
    let gamma_abc = cfg_iter!(a[..n])
        .zip(&b[..n])
        .zip(&c[..n])
        .map(|((a, b), c)| (beta * a + &(alpha * b) + c) * &gamma_inverse)
        .collect::<Vec<_>>();

    let l = cfg_iter!(a)
        .zip(&b)
        .zip(&c)
        .map(|((a, b), c)| (beta * a + &(alpha * b) + c) * &delta_inverse)
        .collect::<Vec<_>>();

    drop(c);

    // Compute B window table
    /*let g2_time = start_timer!(|| "Compute G2 table");
    let g2_window = FixedBase::get_mul_window_size(non_zero_b);
    let g2_table = FixedBase::get_window_table::<E::G2>(scalar_bits, g2_window, g2_generator);
    end_timer!(g2_time);*/

    // Compute the B-query in G2
    // 1. B-query 계산 (G2)
    let b_g2_time = start_timer!(|| "Calculate B G2");
    // FixedBase::msm 대신 VariableBaseMSM의 msm 사용
    // let b_g2_query = E::G2::msm(&g2_generator.into_group().prepare_bases(&[g2_generator])[..0], &b)?; // 이 부분은 로직에 따라 아래처럼 바뀝니다.
    // 실제로는 단순히 베이스가 하나인 경우이므로 아래처럼 처리하는 것이 0.5.0 방식입니다.
    let b_g2_query = cfg_iter!(b).map(|s| g2_generator * s).collect::<Vec<_>>();
    let b_g2_query = E::G2::normalize_batch(&b_g2_query);
    end_timer!(b_g2_time);

    // Compute G window table
    /*let g1_window_time = start_timer!(|| "Compute G1 window table");
    let g1_window =
        FixedBase::get_mul_window_size(non_zero_a + non_zero_b + qap_num_variables + m_raw + 1);
    let g1_table = FixedBase::get_window_table::<E::G1>(scalar_bits, g1_window, g1_generator);
    end_timer!(g1_window_time);*/

    // Generate the R1CS proving key
    let proving_key_time = start_timer!(|| "Generate the R1CS proving key");

    let beta_repr = beta.into_bigint();
    let delta_repr = delta.into_bigint();

    let alpha_g1 = g1_generator.mul_bigint(alpha.into_bigint());
    let beta_g1 = g1_generator.mul_bigint(beta_repr);
    let beta_g2 = g2_generator.mul_bigint(beta_repr);
    let delta_g1 = g1_generator.mul_bigint(delta_repr);
    let delta_g2 = g2_generator.mul_bigint(delta_repr);

    // Compute the A-query
    // 2. A-query 계산 (G1)
    let a_time = start_timer!(|| "Calculate A");
    let a_query_proj: Vec<_> = cfg_iter!(a).map(|s| g1_generator * s).collect();
    let a_query = E::G1::normalize_batch(&a_query_proj);
    end_timer!(a_time);

    // Compute the B-query in G1
    // 3. B-query 계산 (G1)
    let b_g1_time = start_timer!(|| "Calculate B G1");
    let b_g1_proj: Vec<_> = cfg_iter!(b).map(|s| g1_generator * s).collect();
    let b_g1_query = E::G1::normalize_batch(&b_g1_proj);
    end_timer!(b_g1_time);
    // 4. H-query 계산
    let h_time = start_timer!(|| "Calculate H");
    let h_scalars = QAP::h_query_scalars::<_, D<E::ScalarField>>(m_raw - 1, t, zt, delta_inverse)?;
    let h_query_proj: Vec<_> = cfg_into_iter!(h_scalars).map(|s| g1_generator * s).collect();
    let h_query = E::G1::normalize_batch(&h_query_proj);
    end_timer!(h_time);

    // Compute the L-query
    // 5. L-query 계산
    let l_time = start_timer!(|| "Calculate L");
    let l_query_proj: Vec<_> = cfg_iter!(l[n..]).map(|s| g1_generator * s).collect();
    let l_query = E::G1::normalize_batch(&l_query_proj);
    end_timer!(l_time);

    end_timer!(proving_key_time);

    // Generate R1CS verification key
    // 6. Gamma ABC 계산
    let verifying_key_time = start_timer!(|| "Generate the R1CS verification key");
    let gamma_abc_proj: Vec<_> = cfg_iter!(gamma_abc).map(|s| g1_generator * s).collect();
    let gamma_abc_g1 = E::G1::normalize_batch(&gamma_abc_proj);
    let gamma_g2 = g2_generator.mul_bigint(gamma.into_bigint());
    end_timer!(verifying_key_time);

    // --- Modified Section: Calculate eta*gamma^-1 for both groups ---
    let eta1_gamma_inv_g1 = g1_generator.mul_bigint((eta1 * &gamma_inverse).into_bigint());
    let eta2_gamma_inv_g1 = g1_generator.mul_bigint((eta2 * &gamma_inverse).into_bigint());

    let gamma_abc_g1_affine = gamma_abc_g1;
    let eta1_gamma_inv_g1_affine = eta1_gamma_inv_g1.into_affine();
    let eta2_gamma_inv_g1_affine = eta2_gamma_inv_g1.into_affine();

    let vk = VerifyingKey::<E> {
        alpha_g1: alpha_g1.into_affine(),
        beta_g2: beta_g2.into_affine(),
        gamma_g2: gamma_g2.into_affine(),
        delta_g2: delta_g2.into_affine(),
        gamma_abc_g1: gamma_abc_g1_affine,
        eta1_gamma_inv_g1: eta1_gamma_inv_g1_affine, // Set
        eta2_gamma_inv_g1: eta2_gamma_inv_g1_affine, // Set
        commit_witness_count_1,                      // Set
        commit_witness_count_2,                      // Set
    };

    let batch_normalization_time = start_timer!(|| "Convert proving key elements to affine");
    //let a_query = E::G1::normalize_batch(&a_query);
    //let b_g1_query = E::G1::normalize_batch(&b_g1_query);
    //let b_g2_query = E::G2::normalize_batch(&b_g2_query);
    //let h_query = E::G1::normalize_batch(&h_query);
    //let l_query = E::G1::normalize_batch(&l_query);
    end_timer!(batch_normalization_time);
    end_timer!(setup_time);

    // --- Modified Section: Calculate eta*delta^-1 for both groups ---
    let eta1_delta_inv_g1 = g1_generator.mul_bigint((eta1 * &delta_inverse).into_bigint());
    let eta2_delta_inv_g1 = g1_generator.mul_bigint((eta2 * &delta_inverse).into_bigint());

    let common = ProvingKeyCommon {
        beta_g1: beta_g1.into_affine(),
        delta_g1: delta_g1.into_affine(),
        eta1_delta_inv_g1: eta1_delta_inv_g1.into_affine(), // Set
        eta2_delta_inv_g1: eta2_delta_inv_g1.into_affine(), // Set
        a_query,
        b_g1_query,
        b_g2_query,
        h_query,
        l_query,
    };
    Ok((ProvingKey { vk, common }, num_instance_variables))
}

#[inline]
fn generate_randomness<E, R>(
    rng: &mut R,
) -> (
    E::ScalarField,
    E::ScalarField,
    E::ScalarField,
    E::ScalarField,
    E::ScalarField,
    E::ScalarField, // Added eta2
    E::G1,
    E::G2,
)
where
    E: Pairing,
    R: Rng,
{
    let alpha = E::ScalarField::rand(rng);
    let beta = E::ScalarField::rand(rng);
    let gamma = E::ScalarField::rand(rng);
    let delta = E::ScalarField::rand(rng);
    let eta1 = E::ScalarField::rand(rng); // eta1
    let eta2 = E::ScalarField::rand(rng); // eta2

    let g1_generator = E::G1::rand(rng);
    let g2_generator = E::G2::rand(rng);

    (
        alpha,
        beta,
        gamma,
        delta,
        eta1,
        eta2,
        g1_generator,
        g2_generator,
    )
}
