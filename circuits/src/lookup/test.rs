use std::{ops::Bound, str::FromStr};

use crate::{
    harisa::{
        arithm::{self, ArithmCircuit},
        constants::*,
        harisa::Harisa,
        hash_to_prime::primality_test,
        type_conversion::bigint_to_fr,
    },
    linker::snark::LinkSnark,
    lookup::{
        self, constants::PRIME, copy_this_or_that::CTTCircuit, lookup::HarisaPlus,
        well_transformed::WTCircuit,
    },
};

use ark_ec::{pairing::Pairing, ScalarMul};
use ark_ff::PrimeField;
use ark_std::{
    rand::{Rng, RngCore, SeedableRng},
    test_rng, One, Zero,
};
use digest::generic_array::functional::FunctionalSequence;
use itertools::Itertools;
use num_bigint::BigInt;
use rand::thread_rng;

fn rand_setgen(n: u32, low: u32, high: u32) -> Vec<BigInt> {
    let mut rand_set: Vec<BigInt> = Vec::new();
    let base_2: i32 = 2;
    let mut rng = thread_rng();

    let set_size = base_2.pow(n.try_into().unwrap());

    for i in 0..set_size {
        let mut tmp_elem = BigInt::from(rng.gen_range(base_2.pow(low)..base_2.pow(high)));
        rand_set.push(tmp_elem * 2);
    }

    rand_set
}

fn lookup_setgen(set: Vec<BigInt>) -> (Vec<BigInt>, Vec<BigInt>) {
    use crate::harisa::hash_to_prime::hash_to_prime;
    use rand::thread_rng;
    let base_2: i32 = 2;
    let expo = 14;

    let mut z_prime: Vec<BigInt> = Vec::new();
    let mut z_len = BigInt::from(base_2.pow(expo));

    let mut prime_table: Vec<BigInt> = Vec::new();

    for f in set {
        let mut tmp_z = BigInt::from(1627);
        let mut tmp_f = f.clone() * z_len.clone() + tmp_z.clone();

        while !(primality_test(&tmp_f.clone(), 10)
            && primality_test(&tmp_z.clone(), 10)
            && !prime_table.contains(&tmp_f.clone())
            && !z_prime.contains(&tmp_z.clone()))
        {
            tmp_z += BigInt::one();
            tmp_f += BigInt::one();
        }
        prime_table.push(tmp_f.clone());
        z_prime.push(tmp_z.clone());
    }

    (prime_table, z_prime)
}

pub fn test_lookup_arbit<E: Pairing>(set_size: u32, batch_size: usize)
where
    <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    let (mut table, mut prime_table, mut z_table) = (Vec::new(), Vec::new(), Vec::new());

    // 원본 테이블 집합
    table = rand_setgen(set_size.clone(), 8, 16); // Before Transformation 여기가 테이블 만들고
    // 위 table의 변환된 소수 집합
    (prime_table, z_table) = lookup_setgen(table.clone()); // \hat{f}, z
    
    let mut vec_table: Vec<(BigInt, BigInt, BigInt)> = prime_table
        .iter()
        .cloned()
        .zip(table.iter().cloned())
        .zip(z_table.iter().cloned())
        .map(|((x, y), z)| (x, y, z))
        .collect();
    vec_table.sort();

    z_table.sort();
    let mut set_table = Vec::new();

    // Convert p* to BigInt
    let mut small_prime = Vec::new();
    for val in ODD_PRIME {
        small_prime.push(BigInt::from(val));
    }

    let (mut hat_f, mut z_f) = (Vec::new(), Vec::new());
    let mut t = Vec::new();
    for i in 0..batch_size {
        let (x, y, z) = vec_table[i].clone();
        hat_f.push(x);
        t.push(y);
        z_f.push(z);
    }

    let base_2: i32 = 2;
    let expo = 14;

    let mut bit_shift = BigInt::from(base_2.pow(expo));

    /*
    table은 원본 데이터. > 이 테이블의 요소가 t
    prime_table은 table의 변환된 소수 집합. > 이 테이블의 요소가 hat_F
    z_table은 prime_table을 생성하는 데 사용된 보조 값 집합. > 이 테이블의 요소가 z_f
    z_len == bit_shift
    */
    assert_eq!(
        hat_f[0].clone(),
        t[0].clone() * bit_shift + z_f[0].clone(),
        "Sorting Failed"
    );

    let mut proven_elem: Vec<BigInt> = [hat_f.clone(), z_f.clone()].concat();

    //  %%%%%%%%%% For Debug %%%%%%%%%%
    let mut non_proven_elem: Vec<BigInt> = Vec::new();
    let mut non_proven_z: Vec<BigInt> = Vec::new();
    for i in batch_size..table.len() {
        non_proven_elem.push(prime_table[i].clone());
        non_proven_z.push(z_table[i].clone());
    }

    let mut prod_prime_table: BigInt = BigInt::one();
    for i in 0..batch_size {
        prod_prime_table *= hat_f[i].clone();
        prod_prime_table *= z_f[i].clone();
    }
    non_proven_elem.extend(non_proven_z.clone());
    let prod_non_proven: BigInt = non_proven_elem.clone().iter().product();

    set_table = [proven_elem.clone(), non_proven_elem.clone()].concat();
    let prod_set_table: BigInt = set_table.clone().iter().product();

    assert_eq!(
        prod_set_table.clone(),
        prod_non_proven.clone() * prod_prime_table.clone(),
        "Exponentiation Failed"
    );

    let arithm_circuit = ArithmCircuit::<E::ScalarField>::mock(2 * batch_size.clone());
    let ctt_circuit =
        CTTCircuit::<E::ScalarField>::mock(2 * batch_size.clone(), 2 * batch_size.clone());
    let wt_circuit = WTCircuit::<E::ScalarField>::mock(
        batch_size.clone(),
        batch_size.clone(),
        batch_size.clone(),
    );

    let (pp, tree) =
        HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_parameters(
            set_table.clone(),
            ctt_circuit,
            wt_circuit,
            arithm_circuit,
            &mut rng,
        )
        .unwrap();

    let mut prod_set: BigInt = set_table.clone().iter().product();

    let accum = tree[0]
        .clone()
        .modpow(&set_table[0].clone(), &pp.m_pp.mod_n);

    let mut circuit_hat_f: Vec<E::ScalarField> = Vec::new();
    let mut circuit_t: Vec<E::ScalarField> = Vec::new();
    let mut circuit_z_f: Vec<E::ScalarField> = Vec::new();

    for hat_f_i in hat_f.clone() {
        circuit_hat_f.push(bigint_to_fr(hat_f_i.clone()));
    }

    for t_i in t.clone() {
        circuit_t.push(bigint_to_fr(t_i.clone()));
    }

    for z_i in z_f.clone() {
        circuit_z_f.push(bigint_to_fr(z_i.clone()));
    }

    let ctt_elem = [circuit_hat_f.clone(), circuit_z_f.clone()].concat();
    let ctt_circuit = CTTCircuit::<E::ScalarField>::new(ctt_elem.clone(), ctt_elem.clone());

    let wt_circuit = WTCircuit::<E::ScalarField>::new(
        circuit_hat_f.clone(),
        circuit_t.clone(),
        circuit_z_f.clone(),
    );

    let proof = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_proof(
        pp.clone(),
        accum.clone(),
        tree,
        hat_f,
        t,
        z_f,
        ctt_circuit,
        wt_circuit,
        &mut rng,
        non_proven_elem.clone(),
    )
    .unwrap();

    assert!(
        HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::verify_lookup(
            pp,
            accum,
            proof.cm_f_hat,
            proof.cm_f,
            proof.cm_z,
            proof,
        )
        .unwrap(),
        "[Harisa+] Verification Failed"
    );
}

// fn test_lookup<E: Pairing>(l_size: usize)
// where
//     <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
// {
//     // %%%%%%%%%%%%%%%%%%%%%%% Existing %%%%%%%%%%%%%%%%%%%%%%%
//     let mut set_hat = Vec::new(); // \hat_t
//     for p_i in ODD_PRIME.clone() {
//         set_hat.push(BigInt::from(p_i));
//     }

//     let (mut set, mut z) = (Vec::new(), Vec::new());

//     for s_i in set_hat.clone() {
//         // 2^14 = 16384
//         let f_i = s_i.clone() / 2_i128.pow(14);
//         let z_i = s_i % 2_i128.pow(14);

//         set.push(f_i); // 원래(소수가 아닌) table
//         z.push(z_i); // 전체 z에 대한 set
//     }

//     let mut f_hat = Vec::new(); // \hat_f
//     let mut f = Vec::new(); // f
//     let mut z_f = Vec::new(); // z 중 \hat_f(lookup element와 mapping되는) z들
//     for i in 0..l_size {
//         f_hat.push(set_hat[i].clone());
//         f.push(set[i].clone());
//         z_f.push(z[i].clone());
//     }

//     // lookup
//     let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

//     let arithm_circuit = ArithmCircuit::<E::ScalarField>::mock(2 * l_size);
//     let bound_circuit = BoundCircuit::<E::ScalarField>::mock(2 * l_size);
//     let ctt_circuit = CTTCircuit::<E::ScalarField>::mock(2 * l_size, 2 * l_size);
//     let wt_circuit = WTCircuit::<E::ScalarField>::mock(l_size, l_size, l_size);

//     let mut binding = set_hat.clone();
//     let mut sorted_set: Vec<BigInt> = binding.drain(..l_size).collect();

//     sorted_set = [sorted_set, z.clone(), binding].concat();

//     let (pp, tree) =
//         HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_parameters(
//             sorted_set.clone(),
//             ctt_circuit,
//             wt_circuit,
//             arithm_circuit,
//             bound_circuit,
//             &mut rng,
//         )
//         .unwrap();

//     // let accum = tree[0].clone().modpow(&sorted_set[0], &pp.m_pp.mod_n);
//     let accum = tree[0].clone().modpow(&sorted_set.clone().iter().product(), &pp.m_pp.mod_n.clone());

//     // let mut circuit_set_hat: Vec<E::ScalarField> = Vec::new();
//     let mut circuit_set: Vec<E::ScalarField> = Vec::new();
//     let mut circuit_z: Vec<E::ScalarField> = Vec::new();
//     let mut circuit_f_hat: Vec<E::ScalarField> = Vec::new();
//     let mut circuit_f: Vec<E::ScalarField> = Vec::new();
//     let mut circuit_z_f: Vec<E::ScalarField> = Vec::new();

//     // for s_hat_i in set_hat.clone() {
//     //     circuit_set_hat.push(bigint_to_fr(s_hat_i)); //
//     // }

//     for s_i in set.clone() {
//         circuit_set.push(bigint_to_fr(s_i));
//     }

//     for z_i in z.clone() {
//         circuit_z.push(bigint_to_fr(z_i));
//     }

//     for f_hat_i in f_hat.clone() {
//         circuit_f_hat.push(bigint_to_fr(f_hat_i));
//     }

//     for i in 0..f_hat.clone().len() {
//         circuit_f.push(circuit_set[i]);
//     }

//     for i in 0..f_hat.clone().len() {
//         circuit_z_f.push(circuit_z[i]);
//     }

//     let ctt_elem = [circuit_f_hat.clone(), circuit_z_f.clone()].concat();

//     let ctt_circuit = CTTCircuit::<E::ScalarField>::new(ctt_elem.clone(), ctt_elem.clone());

//     let wt_circuit = WTCircuit::<E::ScalarField>::new(
//         circuit_f_hat.clone(),
//         circuit_f.clone(),
//         circuit_z_f.clone(),
//     );

//     let proof = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_proof(
//         pp.clone(),
//         accum.clone(),
//         tree,
//         f_hat,
//         f,
//         z_f,
//         ctt_circuit,
//         wt_circuit,
//         &mut rng,
//     )
//     .unwrap();

//     assert!(
//         HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::verify_lookup(
//             pp,
//             accum,
//             proof.cm_f_hat,
//             proof.cm_f,
//             proof.cm_z,
//             proof,
//         )
//         .unwrap(),
//         "[Harisa+] Verify Failed"
//     );
// }

const SET_SIZE: usize = 16;

#[test]
fn test_lookup_bn254() {
    use ark_bn254::{Bn254, Fr as F};

    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    // test_lookup::<Bn254>(SET_SIZE);
}

#[test]
fn test_lookup_bench() {
    use ark_bn254::{Bn254, Fr as F};

    test_lookup_arbit::<Bn254>(11, 512);
}

#[test]
fn test_lookup_gen() {
    let mut table = Vec::new();
    let mut prime_table = Vec::new();
    let mut z_table = Vec::new();

    let set_size = 8; // 2^N(Table size)
    let low = 8; // Random set element range from 2^low
    let high = 16; // Random set element range to 2^high
    table = rand_setgen(set_size.clone(), low.clone(), high.clone()); // Berfore transformation

    (prime_table, z_table) = lookup_setgen(table.clone());

    // %%%%%%%%%%%%%%%%%%%%%%%% For Debug %%%%%%%%%%%%%%%%%%%%%%%%
    // for i in 0..set_size.clone() {
    //     println!("f_{:?}: {:?}", i, prime_table[i].clone());
    //     println!("z_{:?}: {:?}", i, z_table[i].clone());
    //     assert_eq!(
    //         primality_test(&prime_table[i].clone(), 10),
    //         true,
    //         "table [{:}] is not a prime element: {:}", i, prime_table[i]
    //     );
    //     assert_eq!(
    //         primality_test(&z_table[i].clone(), 10),
    //         true,
    //         "z_{:} is not a prime element: {:}", i, z_table[i]
    //     );
    // }
}
