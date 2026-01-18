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

#[derive(Clone)]
pub struct CTTCircuit<F: PrimeField> { // 일단 두 벡터를 입력으로 받아, 특정 조건을 만족하는지 확인하는 서킷을 정의할 것임.
    pub u: Option<Vec<F>>,
    pub a: Option<Vec<F>>,
}

impl<F: PrimeField> CTTCircuit<F> {
    pub fn new(u: Vec<F>, a: Vec<F>) -> Self {
        Self {
            u: Some(u),
            a: Some(a),
        }
    }

    pub fn mock(t_len: usize, u_len: usize) -> Self {
        Self {
            u: Some(vec![F::zero(); t_len]),
            a: Some(vec![F::zero(); u_len]),
        }
    }
}

impl<F: PrimeField> ConstraintSynthesizer<F> for CTTCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let u = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.u.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let a = Vec::<FpVar<F>>::new_input(cs.clone(), || {
            self.a.ok_or(SynthesisError::AssignmentMissing)
        })?;

        let zero = FpVar::zero();

        let mut this = a.clone().first().unwrap() - u.clone().first().unwrap();
        let mut that = a.clone().first().unwrap() - zero.clone();
        let mut res = this;

        res.enforce_equal(&zero.clone())?;

        for (a_i, u_i) in a.clone().iter().skip(1).zip(u.clone().into_iter().skip(1)) {
            this = a_i - u_i;
            that -= a_i;
            res = this * that;
            res.enforce_equal(&zero.clone())?;

            that = a_i.clone();
        }

        Ok(())
    }
}

#[cfg(test)]
mod ctt {
    use super::CTTCircuit;
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

    fn test_cp_ctt<F: PrimeField>(a_len: usize) -> (Option<Vec<F>>, Option<Vec<F>>) {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let mut a_vec = Vec::new();
        let mut u_vec = Vec::new();

        // let r: usize = rng.gen::<usize>() % 256;
        let r: usize = rng.gen_range(0..256);
        let a_i = F::from(ODD_PRIME[r]);
        a_vec.push(a_i);
        u_vec.push(a_i);

        for i in 1..a_len {
            // let r: usize = rng.gen::<usize>() % 256;
            let r: usize = rng.gen_range(0..256);
            let a_i = F::from(ODD_PRIME[r]);
            a_vec.push(a_i);

            let u_i = if a_vec[i] == a_vec[i - 1] && i > 0 {
                // let r: usize = rng.gen::<usize>() % 256;
                let r: usize = rng.gen_range(0..256);
                F::from(ODD_PRIME[r])
            } else {
                a_i
            };

            u_vec.push(u_i);
        }

        (Some(u_vec), Some(a_vec))
    }

    const U_LEN: usize = 32;

    #[test]
    fn test_cp_ctt_cc_groth16_bn254() {
        use ark_bn254::{Bn254, Fr as F};
        use ark_ed_on_bn254::{constraints::EdwardsVar as GG, EdwardsProjective as C};

        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

        let (u, a) = test_cp_ctt::<F>(U_LEN);

        let circuit = CTTCircuit::<F>::mock(U_LEN, U_LEN);

        let (ek, vk) = CcGroth16::<Bn254>::circuit_specific_setup(circuit, &mut rng).unwrap();
        let pvk = prepare_verifying_key::<Bn254>(&vk);

        let circuit = CTTCircuit::<F>::new(u.unwrap(), a.unwrap());

        let proof = CcGroth16::<Bn254>::prove(&ek, circuit, &mut rng).unwrap();

        assert!(CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}
