use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use num_bigint::{BigInt, Sign, ToBigUint};
use num_traits::{One, ToBytes, Zero};

use ark_std::{cfg_into_iter, cfg_iter, ops::Mul, vec::Vec};
use rand::{distributions::Uniform, thread_rng, Rng};
use std::str::FromStr;

use super::{
    constants::{self, MIMC7_ROUNDS, MIMC_7_91_BN254_ROUND_KEYS},
    type_conversion::*,
};
use ark_relations::r1cs::SynthesisError;

pub fn gen_bigint_range(rng: &mut impl Rng, low: &BigInt, high: &BigInt) -> BigInt {
    let range = high - low;
    let (_, mut bytes) = range.to_bytes_le();
    rng.fill(&mut bytes[..]);
    let random = BigInt::from_bytes_le(Sign::Plus, &bytes);
    low + (random % &range)
}

pub fn primality_test(input: &BigInt, round: u32) -> bool {
    if input <= &BigInt::from(1) {
        return false;
    }
    if input == &BigInt::from(2) || input == &BigInt::from(3) {
        return true;
    }
    if input % BigInt::from(2) == BigInt::ZERO {
        return false;
    }

    let mut d = input - 1;
    let mut r = 0;

    while &d % BigInt::from(2) == BigInt::ZERO {
        d /= BigInt::from(2);
        r += 1;
    }

    let mut rng = thread_rng();
    'outer: for _ in 0..round {
        let a = gen_bigint_range(&mut rng, &BigInt::from(2), &(input - 2));
        let mut x = a.modpow(&d, &input);

        if x == BigInt::one() || x == input - 1 {
            continue;
        }

        for _ in 0..(r - 1) {
            let x = x.modpow(&BigInt::from(2), &input);
            if x == input - 1 {
                continue 'outer;
            }
        }

        return false;
    }

    true
}

pub fn hash_to_prime<F: PrimeField>(x_l: BigInt, x_r: BigInt, constants: &[F]) -> BigInt {
    let mut hash_prime = hash_mimc(x_l.clone(), x_r.clone(), &constants);
    let mut rng = thread_rng();
    let mut input_rnd = x_r.clone();

    while !primality_test(&hash_prime, 20) {
        input_rnd = &input_rnd + BigInt::from(1);
        hash_prime = hash_mimc(x_l.clone(), input_rnd.clone(), &constants);
    }
    hash_prime
}

pub struct MiMC7<F: PrimeField> {
    pub round: usize,
    pub constants: Vec<F>,
}

impl<F: PrimeField> MiMC7<F> {
    pub fn hash_mimc(xl_big: BigInt, xr_big: BigInt, constants: &[F]) -> BigInt {
        let xl = bigint_to_fr(xl_big);
        let xr = bigint_to_fr(xr_big);

        let output_fr = mimc7(xl, xr, &constants);

        let output = fr_to_bigint(output_fr);

        output
    }

    pub fn mimc7(xl: F, xr: F, constants: &[F]) -> F {
        assert_eq!(constants.len(), MIMC7_ROUNDS);

        let mut res = Self::mimc7_round(xl, xr, &constants[0]);
        for i in 1..MIMC7_ROUNDS {
            println!("{:?}-th rounds? {:?}", i, constants[i]);
            res = Self::mimc7_round(res, xr, &constants[i]);
        }

        res += xr + xl + xr;

        res
    }

    pub fn mimc7_round(mut msg: F, key: F, constant: &F) -> F {
        msg += key;
        msg += constant;
        let mut tmp = msg;
        let mut res = msg;
        tmp = tmp.square();
        let tmp2 = tmp.square();
        res *= &tmp;
        res *= &tmp2;

        res
    }
    pub fn round_keys_contants_to_vec(round_keys: &[&str]) -> Vec<F>
    where
        F::Err: core::fmt::Debug,
    {
        round_keys.iter().map(|e| F::from_str(e).unwrap()).collect()
    }
}

pub fn hash_mimc<F: PrimeField>(xl_big: BigInt, xr_big: BigInt, constants: &[F]) -> BigInt {
    let xl = bigint_to_fr(xl_big);
    let xr = bigint_to_fr(xr_big);

    let output_fr = mimc7(xl, xr, &constants);

    let output = fr_to_bigint(output_fr);

    output
}

pub fn mimc7<F: PrimeField>(xl: F, xr: F, constants: &[F]) -> F {
    assert_eq!(constants.len(), MIMC7_ROUNDS);

    let mut res = mimc7_round(xl, xr, &constants[0]);
    for i in 1..MIMC7_ROUNDS {
        // println!("[i]-th rounds? {:?}", constants[i]);
        res = mimc7_round(res, xr, &constants[i]);
    }

    res += xr + xl + xr;

    res
}

pub fn mimc7_round<F: PrimeField>(mut msg: F, key: F, constant: &F) -> F {
    msg += key;
    msg += constant;
    let mut tmp = msg;
    let mut res = msg;
    tmp = tmp.square();
    let tmp2 = tmp.square();
    res *= &tmp;
    res *= &tmp2;

    res
}

pub fn round_keys_contants_to_vec<F: PrimeField>(round_keys: &[&str]) -> Vec<F>
where
    F::Err: core::fmt::Debug,
{
    round_keys.iter().map(|e| F::from_str(e).unwrap()).collect()
}

#[cfg(test)]
mod mimc7 {
    use ark_ff::{One, PrimeField, Zero};
    use num_bigint::BigInt;

    use ark_bn254::Fr as F;

    use crate::harisa::{constants, precompute::rsa_setup, type_conversion::fr_to_bigint};

    use super::{round_keys_contants_to_vec, MiMC7, MIMC7_ROUNDS, MIMC_7_91_BN254_ROUND_KEYS};

    fn test_mimc<F: PrimeField>()
    where
        F::Err: core::fmt::Debug,
    {
        let constants = round_keys_contants_to_vec::<F>(&MIMC_7_91_BN254_ROUND_KEYS);

        let output = MiMC7::<F>::mimc7(F::one(), F::zero(), &constants);

        println!("Output: {:?}", output.into_bigint());
    }

    #[test]
    fn test_mimc7() {
        let constants = round_keys_contants_to_vec::<F>(&MIMC_7_91_BN254_ROUND_KEYS);

        let output = MiMC7::<F>::mimc7(F::one(), F::zero(), &constants);

        println!("Output: {:?}", output);
    }

    #[test]
    fn test_hash_mimc() {
        let constants = round_keys_contants_to_vec::<F>(&MIMC_7_91_BN254_ROUND_KEYS);

        let output = MiMC7::hash_mimc(BigInt::from(1), BigInt::zero(), &constants);

        println!("Output: {:?}", output);
    }

    #[test]
    fn test_primality_test() {
        use super::{gen_bigint_range, primality_test};
        use num_bigint::BigInt;

        let n_1 = BigInt::parse_bytes(b"17189048", 10).unwrap();
        let n_2 = BigInt::parse_bytes(b"10", 10).unwrap();
        let n_3 = BigInt::parse_bytes(b"19", 10).unwrap();

        println!("{} is prime : {}", n_1, primality_test(&n_1, 20));
        println!("{} is prime : {}", n_2, primality_test(&n_2, 20));
        println!("{} is prime : {}", n_3, primality_test(&n_3, 20));
    }

    #[test]
    fn test_prime_hash() {
        use super::{hash_mimc, hash_to_prime};
        use num_bigint::BigInt;

        let constants = round_keys_contants_to_vec::<F>(&MIMC_7_91_BN254_ROUND_KEYS);

        let output = hash_to_prime(BigInt::from(1), BigInt::zero(), &constants);

        println!("Output: {:?}", output)
    }
}
