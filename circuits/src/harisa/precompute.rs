use ark_ec::pairing::Pairing;
use ark_ff::{BigInteger, BigInteger256, Field, PrimeField, UniformRand};
use ark_serialize::CanonicalDeserialize;
use ark_std::vec::Vec;
use ark_std::{rand::Rng, One, Zero};
use num_bigint::{BigInt, Sign};
use num_traits::{self, Num, Pow, Signed};
use std::mem::swap;

use crate::harisa::constants;

use super::constants::RSA_2048;

// ext_gcd(modify)
pub fn rsa_setup() -> (BigInt, BigInt) {
    let mod_n = BigInt::from_str_radix(RSA_2048, 10);
    let g = BigInt::from(2);

    (g, mod_n.unwrap())
}

pub fn prod_set(set: Vec<BigInt>) -> BigInt {
    let res = set.iter().product();
    res
}

pub fn precompute(g: BigInt, mod_n: BigInt, set: Vec<BigInt>) -> Vec<BigInt> {
    let mut pre_tree = Vec::new();
    pre_tree.push(g.clone());

    let mut right_set = set.clone();
    let mut set_size: usize = set.len();

    if set_size > 1 {
        set_size >>= 1;
        let left_set: Vec<BigInt> = right_set.drain(set_size..).collect();
        let prod_left = prod_set(left_set.clone());
        let left_res = g.modpow(&prod_left, &mod_n);
        let left = precompute(left_res, mod_n.clone(), right_set.clone());

        let prod_right = prod_set(right_set);
        let right_res = g.modpow(&prod_right, &mod_n);
        let right: Vec<BigInt> = precompute(right_res, mod_n, left_set);
        pre_tree = [left, right].concat();
    }
    pre_tree
}

pub fn extended_gcd(a: BigInt, b: BigInt) -> (BigInt, BigInt, BigInt) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    let (mut old_t, mut t) = (BigInt::zero(), BigInt::one());

    while r != BigInt::zero() {
        let quotient = &old_r / &r;

        old_r -= &quotient * &r;
        swap(&mut old_r, &mut r);

        old_s -= &quotient * &s;
        swap(&mut old_s, &mut s);

        old_t -= &quotient * &t;
        swap(&mut old_t, &mut t);
    }

    (old_r, old_s, old_t)
}

pub fn assemble(mod_n: BigInt, a: BigInt, b: BigInt, w_a: BigInt, w_b: BigInt) -> (BigInt, BigInt) {
    let (_, x, y) = extended_gcd(a.clone(), b.clone());
    let w_a_y = modpow(w_a, y, mod_n.clone());
    let w_b_x = modpow(w_b, x, mod_n.clone());
    let res = w_a_y * w_b_x;

    let w = res % mod_n;

    (w, a * b)
}

pub fn modpow(x: BigInt, y: BigInt, mod_n: BigInt) -> BigInt {
    let mut res = x.modpow(&y.abs(), &mod_n);
    if y.is_negative() {
        res = res.modinv(&mod_n).unwrap();
    }

    res
}

#[cfg(test)]
mod precompute {
    use super::{assemble, extended_gcd, precompute, prod_set, rsa_setup};
    use crate::harisa::constants::ODD_PRIME;
    use ark_ec::pairing::Pairing;
    use ark_std::{
        rand::{CryptoRng, Rng, RngCore},
        UniformRand,
    };
    use num_bigint::BigInt;
    use std::clone;

    #[test]
    fn test_rsa_setup() {
        let (g, mod_n) = rsa_setup();
        println!("g: {:?}", g);
        println!("N: {:?}", mod_n);
    }

    #[test]
    fn test_prod_set() {
        let simple_set = vec![
            BigInt::from(1),
            BigInt::from(3),
            BigInt::from(7),
            BigInt::from(11),
        ];
        let prod_simple_set = prod_set(simple_set);

        assert_eq!(prod_simple_set, BigInt::from(231));
    }

    fn test_precompute(n: usize) {
        let (g, mod_n) = rsa_setup();
        let mut rng = ark_std::test_rng();

        let mut set: Vec<BigInt> = Vec::new();

        for i in 0..n {
            let r: usize = rng.gen_range(0..256);
            set.push(BigInt::from(ODD_PRIME[r]));
        }

        let pre_tree = precompute(g.clone(), mod_n.clone(), set.clone());

        let prod_set = prod_set(set.clone());

        for i in 0..set.len() {
            let expr = prod_set.clone() / set[i].clone();
            assert_eq!(
                pre_tree[i],
                g.clone().modpow(&expr, &mod_n),
                "Not equal in elem {:?}",
                i
            );
        }
    }

    #[test]
    fn test_precompute_bn254() {
        test_precompute(8)
    }

    #[test]
    fn test_ext_gcd() {
        use super::extended_gcd;
        use num_bigint::BigInt;

        let a = BigInt::parse_bytes(b"161", 10).unwrap();
        let b = BigInt::parse_bytes(b"17", 10).unwrap();

        let (gcd, x, y) = extended_gcd(a.clone(), b.clone());

        println!("x: {:?}", x);
        println!("y: {:?}", y);

        assert_eq!(gcd, BigInt::parse_bytes(b"1", 10).unwrap());
        assert_eq!(a * &x + b * &y, gcd);
    }

    #[test]
    fn test_big_int_to_fr() {
        use super::super::type_conversion::*;
        use ark_bn254::{Bn254 as E, Fr as F};
        use num_bigint::BigInt;

        let x = BigInt::parse_bytes(b"171", 10).unwrap();
        let y = BigInt::parse_bytes(b"16", 10).unwrap();

        let scalar_x = bigint_to_fr::<F>(x.clone());
        let scalar_y = bigint_to_fr::<F>(y.clone());

        println!("scalar_x : {:?}", x);
        println!("scalar_y : {:?}", y);
    }
}
