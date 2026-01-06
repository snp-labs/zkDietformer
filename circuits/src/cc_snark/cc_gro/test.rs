use crate::cc_snark::cc_gro::{prepare_verifying_key, CcGroth16};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError},
};
use ark_std::marker::PhantomData;
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng, UniformRand,
};

struct TestCircuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
    test: [Option<F>; 2],
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for TestCircuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a *= &b;
            Ok(a)
        })?;

        for (_, t) in self.test.iter().enumerate() {
            cs.new_input_variable(|| t.ok_or(SynthesisError::AssignmentMissing))?;
        }

        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;

        Ok(())
    }
}

fn test_prove_and_verify<E>(n_iters: usize)
where
    E: Pairing,
{
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    let (pk, vk) = CcGroth16::<E>::setup(
        TestCircuit {
            a: None,
            b: None,
            test: [None; 2],
        },
        &mut rng,
    )
    .unwrap();
    let pvk = prepare_verifying_key::<E>(&vk);

    for _ in 0..n_iters {
        let a = E::ScalarField::rand(&mut rng);
        let b = E::ScalarField::rand(&mut rng);
        let test = [Some(E::ScalarField::rand(&mut rng)); 2];
        let mut c = a;
        c *= b;

        let proof = CcGroth16::<E>::prove(
            &pk,
            TestCircuit {
                a: Some(a),
                b: Some(b),
                test: test,
            },
            &mut rng,
        )
        .unwrap();

        assert!(CcGroth16::<E>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}

mod bls12_381 {
    use super::test_prove_and_verify;
    use ark_bls12_381::Bls12_381;

    #[test]
    fn prove_and_verify() {
        test_prove_and_verify::<Bls12_381>(1);
    }
}
