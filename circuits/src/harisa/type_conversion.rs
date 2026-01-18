use ark_ec::pairing::Pairing;
use ark_ff::{BigInteger, Field, PrimeField, UniformRand};
use ark_std::{rand::Rng, One, Zero};
use num_bigint::BigInt;
use num_traits::{self, Num, Pow};

pub fn bigint_to_fr<F: PrimeField>(x: BigInt) -> F {
    let (_, mut x_bytes) = x.to_bytes_be();

    if x_bytes.len() > 32 {
        (_, x_bytes) = (x % BigInt::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap())
        .to_bytes_be();
    }

    while x_bytes.len() < 32 {
        x_bytes.insert(0, 0);
    }

    assert!(
        x_bytes.len() == 32,
        "The byte representation must be 32 bytes long."
    );

    let mut bytes: [u8; 32] = [0u8; 32];
    bytes.copy_from_slice(&x_bytes);

    F::from_be_bytes_mod_order(&bytes)
}

pub fn fr_to_bigint<F: PrimeField>(x: F) -> BigInt {
    let x_bytes = x.into_bigint().to_bytes_be();

    BigInt::from_bytes_be(num_bigint::Sign::Plus, &x_bytes)
}

#[cfg(test)]
mod type_conversion {
    use super::{bigint_to_fr, fr_to_bigint};
    use ark_bls12_377::FrConfig;
    use ark_ec::pairing::Pairing;
    use ark_ff::{Field, Fp, MontBackend, PrimeField, UniformRand};
    use num_traits::{self, Num, Pow};
    use std::clone;

    #[test]
    fn test_bigint_to_fr() {
        use super::bigint_to_fr;
        use ark_bn254::{Bn254 as E, Fr as F};
        use num_bigint::BigInt;

        let x = BigInt::parse_bytes(b"171", 10).unwrap();
        let x_fr: Fp<MontBackend<FrConfig, 4>, 4> = bigint_to_fr(x.clone());

        println!("x: {:?}", x);
        println!("x_fr: {:?}", x_fr);
    }

    #[test]
    fn test_fr_to_bigint() {
        use super::fr_to_bigint;
        use ark_bn254::{Bn254 as E, Fr as F};
        use ark_ff::{PrimeField, UniformRand};
        use ark_std::rand::Rng;
        use num_bigint::BigInt;

        let mut rng = ark_std::rand::thread_rng();
        let x_fr = F::rand(&mut rng);
        // let x_fr = F::from(17189048);

        let x = fr_to_bigint(x_fr.clone());

        println!("x: {:?}", x);
        println!("x_fr: {:?}", x_fr);

        let x_fr_back = bigint_to_fr::<F>(x.clone());
        println!("x_fr_back: {:?}", x_fr_back);

        assert_eq!(
            x_fr, x_fr_back,
            "BigInt to PrimeField Conversion should be consistent"
        )
    }
}
