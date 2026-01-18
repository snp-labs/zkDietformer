use crate::ConstraintF;

use ark_ec::CurveGroup;
use ark_ff::{One, PrimeField};
use ark_r1cs_std::{fields::fp::FpVar, prelude::*};
use ark_relations::{
    ns,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, SynthesisMode},
};
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul, MulAssign},
};

#[derive(Clone)]
pub struct BoundCircuit<F: PrimeField> {
    pub p: Option<F>,
    pub u: Option<Vec<F>>,
}

impl<F: PrimeField> BoundCircuit<F> {
    pub fn new(p: F, u: Vec<F>) -> Self {
        Self {
            p: Some(p),
            u: Some(u),
        }
    }

    pub fn mock(len: usize) -> Self {
        Self {
            p: Some(F::zero()),
            u: Some(vec![F::zero(); len]),
        }
    }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for BoundCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let u = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.u.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let p = FpVar::new_input(cs.clone(), || {
            self.p.ok_or(SynthesisError::AssignmentMissing)
        })?;

        for u_i in u.clone().iter() {
            u_i.enforce_cmp(&p.clone(), std::cmp::Ordering::Greater, false)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod bound {
    use super::BoundCircuit;
    use crate::cc_snark::{prepare_verifying_key, CcGroth16};
    use crate::harisa::constants::*;
    use crate::ConstraintF;
    use ark_crypto_primitives::snark::SNARK;
    use ark_ec::pairing::Pairing;
    use ark_ec::CurveGroup;
    use ark_ff::PrimeField;
    use ark_r1cs_std::pairing::PairingVar;
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
    use ark_std::{
        rand::{Rng, RngCore, SeedableRng},
        test_rng, One, UniformRand,
    };

    fn test_cp_bound<F: PrimeField + UniformRand>(u_len: usize) -> (Option<F>, Option<Vec<F>>) {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let mut u = Vec::new();

        for _ in 0..u_len {
            // let u_i = F::from(rng.gen::<u128>());
            let u_i: F = F::rand(&mut rng);
            u.push(u_i);
        }

        (Some(F::one()), Some(u))
    }

    const U_LEN: usize = 32;

    #[test]
    fn test_cp_bound_cc_groth16_bn254() {
        use ark_bn254::{Bn254, Fr as F};
        use ark_ed_on_bn254::{constraints::EdwardsVar as GG, EdwardsProjective as C};

        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let (p, u) = test_cp_bound::<F>(U_LEN);

        let circuit = BoundCircuit::<F>::mock(U_LEN);

        let (ek, vk) =
            CcGroth16::<Bn254>::circuit_specific_setup(circuit.clone(), &mut rng).unwrap();
        let pvk = prepare_verifying_key::<Bn254>(&vk);

        let circuit = BoundCircuit::<F>::new(p.unwrap(), u.unwrap());

        let proof = CcGroth16::<Bn254>::prove(&ek, circuit, &mut rng).unwrap();

        assert!(CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}
