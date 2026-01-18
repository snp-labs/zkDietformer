use crate::ConstraintF;

use ark_ec::CurveGroup;
use ark_ff::{
    biginteger::{BigInteger as _, BigInteger64 as B},
    One, PrimeField,
};
use ark_r1cs_std::{fields::fp::FpVar, prelude::*};
use ark_relations::{
    ns,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, SynthesisMode},
};
use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul, MulAssign},
};

const SHIFT_SIZE: usize = 14;

#[derive(Clone)]
pub struct WTCircuit<F: PrimeField> {
    pub u: Option<Vec<F>>,
    pub a: Option<Vec<F>>,
    pub z: Option<Vec<F>>,
}

impl<F: PrimeField> WTCircuit<F> {
    pub fn new(u: Vec<F>, a: Vec<F>, z: Vec<F>) -> Self {
        Self {
            u: Some(u),
            a: Some(a),
            z: Some(z),
        }
    }

    pub fn mock(u_len: usize, a_len: usize, z_len: usize) -> Self {
        Self {
            u: Some(vec![F::zero(); u_len]),
            a: Some(vec![F::zero(); a_len]),
            z: Some(vec![F::zero(); z_len]),
        }
    }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for WTCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let u = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.u.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let a = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.a.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let z = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.z.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let mut shift = FpVar::new_constant(
            cs.clone(),
            F::from(2u128.pow(SHIFT_SIZE.try_into().unwrap())),
        )?;

        for (u_i, (a_i, z_i)) in u
            .clone()
            .iter()
            .zip(a.clone().into_iter().zip(z.clone().into_iter()))
        {
            let computed_u_i = a_i.clone() * shift.clone() + z_i.clone();

            u_i.enforce_equal(&computed_u_i)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod wt {
    use super::{WTCircuit, SHIFT_SIZE};
    use crate::cc_snark::cc_gro::{prepare_verifying_key, CcGroth16};
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

    fn test_cp_wt<F: PrimeField>(a_len: usize) -> (Option<Vec<F>>, Option<Vec<F>>, Option<Vec<F>>) {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let mut u_vec = Vec::new();
        let mut a_vec = Vec::new();
        let mut z_vec = Vec::new();

        let shift = F::from(2u128.pow(SHIFT_SIZE.try_into().unwrap()));

        for i in 0..a_len {
            let a_i = F::from(ODD_PRIME[3 * i]);
            a_vec.push(a_i);

            let z_i = F::from(ODD_PRIME[3 * i + 1]);
            z_vec.push(z_i);

            let u_i = a_i * shift + z_i;
            u_vec.push(u_i);
        }

        (Some(u_vec), Some(a_vec), Some(z_vec))
    }

    const U_LEN: usize = 32;

    #[test]
    fn test_cp_wt_cc_groth16_bn254() {
        use ark_bn254::{Bn254, Fr as F};
        use ark_ed_on_bn254::{constraints::EdwardsVar as GG, EdwardsProjective as C};

        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let (u, a, z) = test_cp_wt::<F>(U_LEN);

        let circuit = WTCircuit::<F>::mock(U_LEN, U_LEN, U_LEN);

        let (ek, vk) = CcGroth16::<Bn254>::circuit_specific_setup(circuit, &mut rng).unwrap();
        let pvk = prepare_verifying_key::<Bn254>(&vk);

        let circuit = WTCircuit::<F>::new(u.unwrap(), a.unwrap(), z.unwrap());

        let proof = CcGroth16::<Bn254>::prove(&ek, circuit, &mut rng).unwrap();

        assert!(CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}
