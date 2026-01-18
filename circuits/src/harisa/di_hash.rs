use crate::harisa::{
    constants::{MIMC7_ROUNDS, MIMC_7_91_BN254_ROUND_KEYS},
    hash_to_prime::round_keys_contants_to_vec,
};

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
pub struct HashCircuit<F: PrimeField> {
    pub u: Option<Vec<F>>,
    pub x: Option<Vec<F>>,
    pub d: Option<Vec<F>>,
    pub constants: Vec<F>,
}

impl<F: PrimeField> HashCircuit<F>
where
    F::Err: core::fmt::Debug,
{
    pub fn new(u: Vec<F>, x: Vec<F>, d: Vec<F>) -> Self {
        Self {
            u: Some(u),
            x: Some(x),
            d: Some(d),
            constants: round_keys_contants_to_vec(&MIMC_7_91_BN254_ROUND_KEYS),
        }
    }

    pub fn mock(len: usize) -> Self {
        Self {
            u: Some(vec![F::zero(); len]),
            x: Some(vec![F::zero(); len]),
            d: Some(vec![F::zero(); len]),
            constants: round_keys_contants_to_vec(&MIMC_7_91_BN254_ROUND_KEYS),
        }
    }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for HashCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let constants = Vec::<FpVar<F>>::new_constant(cs.clone(), self.constants)?;

        let u = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.u.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let x = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.x.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let d = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.d.ok_or(SynthesisError::AssignmentMissing)
        })?;

        for ((u_i, x_i), d_i) in u
            .clone()
            .iter()
            .zip(x.clone().into_iter())
            .zip(d.clone().into_iter())
        {
            // hash 들어가야함 (아마도 MiMC7이 들어가겠죠~?)
            let mut res = x_i.clone();

            for c_i in constants.clone().iter() {
                res += c_i;
                res += d_i.clone();
                let tmp = res.clone() * res.clone();
                let tmp2 = tmp.clone() * tmp.clone();
                res *= tmp;
                res *= tmp2;
            }
            res += x_i + d_i.clone() + d_i.clone();

            u_i.enforce_equal(&res)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod hash {
    use super::HashCircuit;
    use crate::cc_snark::cc_gro::{prepare_verifying_key, CcGroth16};
    use crate::harisa::constants::*;
    use crate::harisa::hash_to_prime::*;
    use crate::harisa::type_conversion::*;
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
    use num_bigint::BigInt;

    fn test_di_hash<F: PrimeField + UniformRand>(u_len: usize) -> (Option<Vec<F>>, Option<Vec<F>>, Option<Vec<F>>)
    where
        F::Err: core::fmt::Debug,
    {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let constants = round_keys_contants_to_vec::<F>(&MIMC_7_91_BN254_ROUND_KEYS);

        let mut x = Vec::new();
        let mut u = Vec::new();
        let mut d = Vec::new();

        for _ in 0..u_len {
            let x_i: F = F::rand(&mut rng);
            // let x_i = F::from(rng.gen::<u128>());
            let mut d_i = F::zero();
            let mut res = mimc7(x_i, d_i.clone(), &constants);

            while !primality_test(&fr_to_bigint(res), 20) {
                d_i += F::one();
                res = mimc7(x_i, d_i.clone(), &constants);
            }
            u.push(res);
            x.push(x_i);
            d.push(d_i);
        }

        (Some(u), Some(x), Some(d))
    }

    const U_LEN: usize = 1024;

    #[test]
    fn test_hash_bn254() {
        use ark_bn254::{Bn254, Fr as F};
        use ark_ed_on_bn254::{constraints::EdwardsVar as GG, EdwardsProjective as C};

        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let (u, x, d) = test_di_hash::<F>(U_LEN);

        let circuit = HashCircuit::<F>::mock(U_LEN);

        let (ek, vk) = CcGroth16::<Bn254>::circuit_specific_setup(circuit, &mut rng).unwrap();
        let pvk = prepare_verifying_key::<Bn254>(&vk);

        let circuit = HashCircuit::<F>::new(u.unwrap(), x.unwrap(), d.unwrap());

        let proof = CcGroth16::<Bn254>::prove(&ek, circuit, &mut rng).unwrap();

        assert!(CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}
