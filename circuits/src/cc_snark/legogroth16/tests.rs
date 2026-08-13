use crate::cc_snark::legogroth16::{
    LinkPublicGenerators, create_random_proof, create_random_proof_incl_cp_link,
    generate_random_parameters, generate_random_parameters_incl_cp_link, prepare_verifying_key,
    rerandomize_proof, rerandomize_proof_1, verify_proof, verify_proof_incl_cp_link,
    verify_witness_commitment_1, verify_witness_commitment_2,
};
use ark_ec::{CurveGroup, pairing::Pairing};
use ark_ff::Field;
use ark_std::{
    UniformRand,
    rand::{RngCore, SeedableRng, rngs::StdRng},
    vec::Vec,
};

use core::ops::MulAssign;

use crate::cc_snark::legogroth16::{error::Error, prover::verify_commitments_group_1};
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable},
};

/// Circuit for computation a * b
#[derive(Clone)]
struct MySillyCircuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
}

/// Circuit for computation a * b + c * d
#[derive(Clone)]
struct MyLessSillyCircuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    d: Option<F>,
}

/// Circuit with 2 public inputs, 1st should be equal to a * b and 2nd to c * d
#[derive(Clone)]
struct MyLessSillyCircuit1<F: Field> {
    a: Option<F>,
    b: Option<F>,
    c: Option<F>,
    d: Option<F>,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for MySillyCircuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a.mul_assign(&b);
            Ok(a)
        })?;

        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        Ok(())
    }
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for MyLessSillyCircuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_witness_variable(|| self.c.ok_or(SynthesisError::AssignmentMissing))?;
        let d = cs.new_witness_variable(|| self.d.ok_or(SynthesisError::AssignmentMissing))?;

        // a * b
        let e_value = if self.a.is_some() && self.b.is_some() {
            Some(self.a.unwrap().mul(self.b.unwrap()))
        } else {
            None
        };
        let e = cs.new_witness_variable(|| e_value.ok_or(SynthesisError::AssignmentMissing))?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + e)?;

        // c * d
        let f_value = if self.c.is_some() && self.d.is_some() {
            Some(self.c.unwrap().mul(self.d.unwrap()))
        } else {
            None
        };
        let f = cs.new_witness_variable(|| f_value.ok_or(SynthesisError::AssignmentMissing))?;
        cs.enforce_constraint(lc!() + c, lc!() + d, lc!() + f)?;

        let y = cs.new_input_variable(|| {
            let e = e_value.ok_or(SynthesisError::AssignmentMissing)?;
            let f = f_value.ok_or(SynthesisError::AssignmentMissing)?;

            Ok(e + f)
        })?;

        cs.enforce_constraint(lc!() + e + f, lc!() + Variable::One, lc!() + y)?;

        Ok(())
    }
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for MyLessSillyCircuit1<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_witness_variable(|| self.c.ok_or(SynthesisError::AssignmentMissing))?;
        let d = cs.new_witness_variable(|| self.d.ok_or(SynthesisError::AssignmentMissing))?;

        let e = cs.new_input_variable(|| Ok(self.a.unwrap().mul(self.b.unwrap())))?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + e)?;

        let f = cs.new_input_variable(|| Ok(self.c.unwrap().mul(self.d.unwrap())))?;
        cs.enforce_constraint(lc!() + c, lc!() + d, lc!() + f)?;

        Ok(())
    }
}

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

fn test_prove_and_verify<E>(n_iters: usize)
where
    E: Pairing,
{
    // c1: witnesses for Group 1 (linked), c2: witnesses for Group 2
    fn run<E: Pairing>(n_iters: usize, c1: u32, c2: u32) {
        let mut rng = StdRng::seed_from_u64(0u64);
        let circuit = MySillyCircuit { a: None, b: None };

        // Build index arrays from counts (test circuits allocate witnesses sequentially)
        let idx1: Vec<u32> = (0..c1).collect();
        let idx2: Vec<u32> = (c1..c1 + c2).collect();

        // CP_link is connected to Group 1, so we need generators for c1 + 1 (hiding)
        let link_gens = get_link_public_gens(&mut rng, c1 + 1);

        // Test error when not enough witnesses
        let bad_idx1: Vec<u32> = (0..2).collect();
        let bad_idx2: Vec<u32> = (2..3).collect();
        assert_eq!(
            generate_random_parameters_incl_cp_link::<E, _, _>(
                circuit.clone(),
                link_gens.clone(),
                &bad_idx1,
                &bad_idx2,
                &[],
                &mut rng,
            )
            .unwrap_err(),
            Error::InsufficientWitnessesForCommitment(2, 3)
        );

        assert_eq!(
            generate_random_parameters::<E, _, _>(circuit.clone(), &bad_idx1, &bad_idx2, &[], &mut rng).unwrap_err(),
            Error::InsufficientWitnessesForCommitment(2, 3)
        );

        // Parameters for generating proof containing CP_link (Group 1 linked)
        let params_link = generate_random_parameters_incl_cp_link::<E, _, _>(
            circuit.clone(),
            link_gens.clone(),
            &idx1,
            &idx2,
            &[],
            &mut rng,
        )
        .unwrap();

        // Parameters for generating proof without CP_link
        let params =
            generate_random_parameters::<E, _, _>(circuit.clone(), &idx1, &idx2, &[], &mut rng).unwrap();

        let pvk_link = prepare_verifying_key::<E>(&params_link.vk.groth16_vk);
        let pvk = prepare_verifying_key::<E>(&params.vk);

        for _ in 0..n_iters {
            let a = E::ScalarField::rand(&mut rng);
            let b = E::ScalarField::rand(&mut rng);
            let mut c = a;
            c.mul_assign(&b);

            // Randomness for d1
            let v1 = E::ScalarField::rand(&mut rng);
            // Randomness for d2
            let v2 = E::ScalarField::rand(&mut rng);
            // Randomness for CP_link (linked to d1)
            let link_v = E::ScalarField::rand(&mut rng);

            let circuit = MySillyCircuit {
                a: Some(a),
                b: Some(b),
            };

            // Create proof with link
            let v3 = E::ScalarField::rand(&mut rng);
            let proof_link = create_random_proof_incl_cp_link(
                circuit.clone(),
                v1,
                v2,
                v3,
                link_v,
                &params_link,
                &mut rng,
            )
            .unwrap();

            // Create proof without link
            let proof = create_random_proof(circuit, v1, v2, v3, &params, &mut rng).unwrap();

            // -- Verify Commitments Logic --

            // Note: In MySillyCircuit, witnesses are allocated as [a, b].
            // If c1=1, c2=1: Group1=[a], Group2=[b]
            // If c1=2, c2=0: Group1=[a, b], Group2=[]
            // If c1=1, c2=0: Group1=[a], Group2=[] (b is uncommitted)

            // 1. Verify CP_Link and Group 1 commitments
            if c1 > 0 {
                let witnesses_g1 = if c1 == 1 { vec![a] } else { vec![a, b] };

                // Should pass
                verify_commitments_group_1(
                    &params_link.vk,
                    &proof_link,
                    1, // public input count
                    &witnesses_g1,
                    &v1,
                    &link_v,
                )
                .unwrap();

                // Should fail with wrong witness
                let wrong_witnesses = if c1 == 1 { vec![b] } else { vec![b, a] };
                assert!(
                    verify_commitments_group_1(
                        &params_link.vk,
                        &proof_link,
                        1,
                        &wrong_witnesses,
                        &v1,
                        &link_v
                    )
                    .is_err()
                );
            }

            // 2. Verify Witness Commitment Group 1 (Standalone)
            if c1 > 0 {
                let witnesses_g1 = if c1 == 1 { vec![a] } else { vec![a, b] };
                verify_witness_commitment_1(&params.vk, &proof, 1, &witnesses_g1, &v1).unwrap();
            }

            // 3. Verify Witness Commitment Group 2
            if c2 > 0 {
                // If c1=1, c2=1 -> Group2=[b]
                // If c1=0, c2=1 -> Group2=[a]
                let witnesses_g2 = if c1 == 1 { vec![b] } else { vec![a] };

                verify_witness_commitment_2(&params.vk, &proof, 1, &witnesses_g2, &v2).unwrap();

                // Should fail with wrong witness
                assert!(
                    verify_witness_commitment_2(
                        &params.vk,
                        &proof,
                        1,
                        &[c], // random wrong val
                        &v2
                    )
                    .is_err()
                );
            }

            // Verify LegoGroth16 proof and CP_link proof
            verify_proof_incl_cp_link(&pvk_link, &params_link.vk, &proof_link, &[c]).unwrap();
            assert!(
                verify_proof_incl_cp_link(&pvk_link, &params_link.vk, &proof_link, &[]).is_err()
            );

            // Verify LegoGroth16 proof
            verify_proof(&pvk, &proof, &[c]).unwrap();
            assert!(verify_proof(&pvk, &proof, &[]).is_err());

            // Rerandomize (without updating v/d)
            let re_rand_proof = rerandomize_proof(&proof, &params.vk, &mut rng);
            verify_proof(&pvk, &re_rand_proof, &[c]).unwrap();

            // rerandomize_proof does not keep D1/D2 valid commitments
            if c1 > 0 {
                let w = if c1 == 1 { vec![a] } else { vec![a, b] };
                assert!(
                    verify_witness_commitment_1(&params.vk, &re_rand_proof, 1, &w, &v1).is_err()
                );
            }

            // Rerandomize 1 (updating v/d)
            let new_v1 = E::ScalarField::rand(&mut rng);
            let new_v2 = E::ScalarField::rand(&mut rng);
            let new_v3 = E::ScalarField::rand(&mut rng);

            let re_rand_proof_1 = rerandomize_proof_1(
                &proof,
                v1,
                new_v1,
                v2,
                new_v2,
                v3,
                new_v3,
                &params.vk,
                &params.common.eta1_delta_inv_g1,
                &params.common.eta2_delta_inv_g1,
                &params.common.eta3_delta_inv_g1,
                &mut rng,
            );

            verify_proof(&pvk, &re_rand_proof_1, &[c]).unwrap();

            // Check commitments persist after rerandomization
            if c1 > 0 {
                let w = if c1 == 1 { vec![a] } else { vec![a, b] };
                verify_witness_commitment_1(&params.vk, &re_rand_proof_1, 1, &w, &new_v1).unwrap();
            }
            if c2 > 0 {
                let w = if c1 == 1 { vec![b] } else { vec![a] };
                verify_witness_commitment_2(&params.vk, &re_rand_proof_1, 1, &w, &new_v2).unwrap();
            }
        }
    }

    // Split witnesses: Group1=[a], Group2=[b]
    run::<E>(n_iters, 1, 1);
    // Only Group1=[a]
    run::<E>(n_iters, 1, 0);
    // No committed witnesses
    run::<E>(n_iters, 0, 0);
}

fn test_prove_and_verify_1<E>(n_iters: usize)
where
    E: Pairing,
{
    let mut rng = StdRng::seed_from_u64(0u64);

    // Commit to 4 witnesses. Let's split them: 2 in Group 1, 2 in Group 2.
    // MyLessSillyCircuit witnesses order: a, b, c, d, e, f.
    // Group 1: [a, b]
    // Group 2: [c, d]
    let c1 = 2;
    let c2 = 2;

    // Generators for CP_link (connected to Group 1, so size c1 + 1)
    let link_gens = get_link_public_gens(&mut rng, c1 + 1);

    let circuit = MyLessSillyCircuit {
        a: None,
        b: None,
        c: None,
        d: None,
    };

    let idx1: Vec<u32> = (0..c1).collect();
    let idx2: Vec<u32> = (c1..c1 + c2).collect();

    let params_link = generate_random_parameters_incl_cp_link::<E, _, _>(
        circuit.clone(),
        link_gens.clone(),
        &idx1,
        &idx2,
        &[],
        &mut rng,
    )
    .unwrap();

    let params = generate_random_parameters::<E, _, _>(circuit, &idx1, &idx2, &[], &mut rng).unwrap();

    let pvk_link = prepare_verifying_key::<E>(&params_link.vk.groth16_vk);
    let pvk = prepare_verifying_key::<E>(&params.vk);

    for _ in 0..n_iters {
        let a = E::ScalarField::rand(&mut rng);
        let b = E::ScalarField::rand(&mut rng);
        let c = E::ScalarField::rand(&mut rng);
        let d = E::ScalarField::rand(&mut rng);

        let e = a * b;
        let f = c * d;
        let y = e + f;

        let v1 = E::ScalarField::rand(&mut rng);
        let v2 = E::ScalarField::rand(&mut rng);
        let v3 = E::ScalarField::rand(&mut rng);
        let link_v = E::ScalarField::rand(&mut rng);

        let circuit = MyLessSillyCircuit {
            a: Some(a),
            b: Some(b),
            c: Some(c),
            d: Some(d),
        };

        let proof_link = create_random_proof_incl_cp_link(
            circuit.clone(),
            v1,
            v2,
            v3,
            link_v,
            &params_link,
            &mut rng,
        )
        .unwrap();

        let proof = create_random_proof(circuit.clone(), v1, v2, v3, &params, &mut rng).unwrap();

        // 1. Verify Group 1 Commitments [a, b] (Linked)
        verify_commitments_group_1(&params_link.vk, &proof_link, 1, &[a, b], &v1, &link_v).unwrap();

        // Negative test
        assert!(
            verify_commitments_group_1(&params_link.vk, &proof_link, 1, &[a], &v1, &link_v)
                .is_err()
        );

        // 2. Verify Group 1 Commitments [a, b] (Standalone)
        verify_witness_commitment_1(&params.vk, &proof, 1, &[a, b], &v1).unwrap();

        // 3. Verify Group 2 Commitments [c, d]
        verify_witness_commitment_2(&params.vk, &proof, 1, &[c, d], &v2).unwrap();

        // Verify proofs
        verify_proof_incl_cp_link(&pvk_link, &params_link.vk, &proof_link, &[y]).unwrap();
        verify_proof(&pvk, &proof, &[y]).unwrap();

        // Rerandomize
        let new_v1 = E::ScalarField::rand(&mut rng);
        let new_v2 = E::ScalarField::rand(&mut rng);
        let new_v3 = E::ScalarField::rand(&mut rng);

        let re_rand_proof_1 = rerandomize_proof_1(
            &proof,
            v1,
            new_v1,
            v2,
            new_v2,
            v3,
            new_v3,
            &params.vk,
            &params.common.eta1_delta_inv_g1,
            &params.common.eta2_delta_inv_g1,
            &params.common.eta3_delta_inv_g1,
            &mut rng,
        );

        verify_proof(&pvk, &re_rand_proof_1, &[y]).unwrap();
        verify_witness_commitment_1(&params.vk, &re_rand_proof_1, 1, &[a, b], &new_v1).unwrap();
        verify_witness_commitment_2(&params.vk, &re_rand_proof_1, 1, &[c, d], &new_v2).unwrap();
    }
}

fn test_prove_and_verify_2<E>(n_iters: usize)
where
    E: Pairing,
{
    let mut rng = StdRng::seed_from_u64(0u64);

    // MyLessSillyCircuit1 has 4 witnesses (a,b,c,d).
    // Split: 2 in G1, 2 in G2.
    let c1 = 2;
    let c2 = 2;

    let link_gens = get_link_public_gens(&mut rng, c1 + 1);

    let circuit = MyLessSillyCircuit1 {
        a: None,
        b: None,
        c: None,
        d: None,
    };

    let idx1: Vec<u32> = (0..c1).collect();
    let idx2: Vec<u32> = (c1..c1 + c2).collect();

    let params_link = generate_random_parameters_incl_cp_link::<E, _, _>(
        circuit.clone(),
        link_gens.clone(),
        &idx1,
        &idx2,
        &[],
        &mut rng,
    )
    .unwrap();

    let params = generate_random_parameters::<E, _, _>(circuit, &idx1, &idx2, &[], &mut rng).unwrap();

    let pvk_link = prepare_verifying_key::<E>(&params_link.vk.groth16_vk);
    let pvk = prepare_verifying_key::<E>(&params.vk);

    for _ in 0..n_iters {
        let a = E::ScalarField::rand(&mut rng);
        let b = E::ScalarField::rand(&mut rng);
        let c = E::ScalarField::rand(&mut rng);
        let d = E::ScalarField::rand(&mut rng);

        let mut e = a;
        e.mul_assign(&b);
        let mut f = c;
        f.mul_assign(&d);

        let v1 = E::ScalarField::rand(&mut rng);
        let v2 = E::ScalarField::rand(&mut rng);
        let v3 = E::ScalarField::rand(&mut rng);
        let link_v = E::ScalarField::rand(&mut rng);

        let circuit = MyLessSillyCircuit1 {
            a: Some(a),
            b: Some(b),
            c: Some(c),
            d: Some(d),
        };

        let proof_link = create_random_proof_incl_cp_link(
            circuit.clone(),
            v1,
            v2,
            v3,
            link_v,
            &params_link,
            &mut rng,
        )
        .unwrap();

        let proof = create_random_proof(circuit.clone(), v1, v2, v3, &params, &mut rng).unwrap();

        // 2 Public Inputs for this circuit
        let public_input_count = 2;

        // Verify G1 [a, b]
        verify_commitments_group_1(
            &params_link.vk,
            &proof_link,
            public_input_count,
            &[a, b],
            &v1,
            &link_v,
        )
        .unwrap();

        // Verify G2 [c, d]
        verify_witness_commitment_2(&params.vk, &proof, public_input_count, &[c, d], &v2).unwrap();

        // Verify Proofs
        verify_proof_incl_cp_link(&pvk_link, &params_link.vk, &proof_link, &[e, f]).unwrap();
        verify_proof(&pvk, &proof, &[e, f]).unwrap();

        // Rerandomize
        let new_v1 = E::ScalarField::rand(&mut rng);
        let new_v2 = E::ScalarField::rand(&mut rng);
        let new_v3 = E::ScalarField::rand(&mut rng);
        let re_rand_proof_1 = rerandomize_proof_1(
            &proof,
            v1,
            new_v1,
            v2,
            new_v2,
            v3,
            new_v3,
            &params.vk,
            &params.common.eta1_delta_inv_g1,
            &params.common.eta2_delta_inv_g1,
            &params.common.eta3_delta_inv_g1,
            &mut rng,
        );
        verify_proof(&pvk, &re_rand_proof_1, &[e, f]).unwrap();
        verify_witness_commitment_1(
            &params.vk,
            &re_rand_proof_1,
            public_input_count,
            &[a, b],
            &new_v1,
        )
        .unwrap();
    }
}

mod bls12_377 {
    use super::*;
    use ark_bls12_377::Bls12_377;

    #[test]
    fn prove_and_verify() {
        test_prove_and_verify::<Bls12_377>(10);
    }

    #[test]
    fn prove_and_verify_1() {
        test_prove_and_verify_1::<Bls12_377>(10);
    }

    #[test]
    fn prove_and_verify_2() {
        test_prove_and_verify_2::<Bls12_377>(10);
    }
}

mod bls12_381 {
    use super::*;
    use ark_bls12_381::Bls12_381;

    #[test]
    fn prove_and_verify() {
        test_prove_and_verify::<Bls12_381>(10);
    }

    #[test]
    fn prove_and_verify_1() {
        test_prove_and_verify_1::<Bls12_381>(10);
    }

    #[test]
    fn prove_and_verify_2() {
        test_prove_and_verify_2::<Bls12_381>(10);
    }
}

mod bn254 {
    use super::*;
    use ark_bn254::Bn254;

    #[test]
    fn prove_and_verify() {
        test_prove_and_verify::<Bn254>(10);
    }

    #[test]
    fn prove_and_verify_1() {
        test_prove_and_verify_1::<Bn254>(10);
    }

    #[test]
    fn prove_and_verify_2() {
        test_prove_and_verify_2::<Bn254>(10);
    }
}
