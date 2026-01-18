use crate::ConstraintF;

use ark_ec::CurveGroup;
use ark_ff::{
    biginteger::{BigInteger as _, BigInteger64 as B},
    One, PrimeField,
};
use ark_nonnative_field::NonNativeFieldVar;
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
pub struct ArithmCircuit<F: PrimeField> {
    pub h: Option<F>,
    pub l: Option<F>,
    pub k: Option<F>,
    pub u: Option<Vec<F>>,
    pub s: Option<F>,
    pub r: Option<F>,
}

impl<F: PrimeField> ArithmCircuit<F> {
    pub fn new(h: F, l: F, k: F, u: Vec<F>, s: F, r: F) -> Self {
        Self {
            h: Some(h),
            l: Some(l),
            k: Some(k),
            u: Some(u),
            s: Some(s),
            r: Some(r),
        }
    }

    pub fn mock(len: usize) -> Self {
        Self {
            h: Some(F::zero()),
            l: Some(F::zero()),
            k: Some(F::zero()),
            u: Some(vec![F::zero(); len]),
            s: Some(F::zero()),
            r: Some(F::zero()),
        }
    }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for ArithmCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let u = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.u.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let s = FpVar::new_input(cs.clone(), || {
            self.s.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let r = FpVar::new_input(cs.clone(), || {
            self.r.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let h = FpVar::new_input(cs.clone(), || {
            self.h.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let l = FpVar::new_input(cs.clone(), || {
            self.l.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let k = FpVar::new_input(cs.clone(), || {
            self.k.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let mut computed_k = s * h;

        for u_i in u.clone().iter() {
            computed_k *= u_i;
        }

        computed_k += r;

        k.enforce_equal(&computed_k)
    }
}

#[cfg(test)]
mod arithm {
    use super::ArithmCircuit;
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
        rand::{Rng, RngCore, SeedableRng, rngs::StdRng},
        test_rng, One, UniformRand,
    };
    

    fn test_cp_arithm<F: PrimeField + UniformRand>(
        u_len: usize,
    ) -> (
        Option<F>,
        Option<F>,
        Option<F>,
        Option<Vec<F>>,
        Option<F>,
        Option<F>,
    ) {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        // let s = F::from(rng.gen::<u128>());
        // let r = F::from(rng.gen::<u128>());

        // let h = F::from(rng.gen::<u128>());
        // let l = F::from(rng.gen::<u128>());
        let s: F = F::rand(&mut rng);
        let r: F = F::rand(&mut rng);
        let h: F = F::rand(&mut rng);
        let l: F = F::rand(&mut rng);

        let mut k = s * h;

        let mut u = Vec::new();

        for _ in 0..u_len {
            // let u_i = F::from(rng.gen::<u128>());
            let u_i: F =  F::rand(&mut rng);
            k *= u_i;
            u.push(u_i);
        }

        k += r;

        (Some(h), Some(l), Some(k), Some(u), Some(s), Some(r))
    }

    fn test_cp_arithm_random<F: PrimeField>(
        u_len: usize,
    ) -> (
        Option<F>,
        Option<F>,
        Option<F>,
        Option<Vec<F>>,
        Option<F>,
        Option<F>,
    ) {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let h = F::rand(&mut rng);
        let l = F::one();
        let s = F::rand(&mut rng);
        let r = F::rand(&mut rng);
        let mut u = Vec::new();
        let mut k = s * h;
        for _ in 0..u_len {
            let r: usize = rng.gen_range(0..256);
            let u_i = F::from(ODD_PRIME[r]);
            u.push(u_i);
            k *= u_i;
        }
        k += r;
        (Some(h), Some(l), Some(k), Some(u), Some(s), Some(r))
    }

    const U_LEN: usize = 32;

    #[test]
    fn test_cp_arithm_cc_groth16_bn254() {
        use ark_bn254::{Bn254, Fr as F};
        use ark_ed_on_bn254::{constraints::EdwardsVar as GG, EdwardsProjective as C};

        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let (h, l, k, u, s, r) = test_cp_arithm_random::<F>(U_LEN);

        let circuit = ArithmCircuit::<F>::mock(U_LEN);

        let (ek, vk) = CcGroth16::<Bn254>::circuit_specific_setup(circuit, &mut rng).unwrap();
        let pvk = prepare_verifying_key::<Bn254>(&vk);

        let circuit = ArithmCircuit::<F>::new(
            h.unwrap(),
            l.unwrap(),
            k.unwrap(),
            u.unwrap(),
            s.unwrap(),
            r.unwrap(),
        );

        let proof = CcGroth16::<Bn254>::prove(&ek, circuit, &mut rng).unwrap();

        assert!(CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}
