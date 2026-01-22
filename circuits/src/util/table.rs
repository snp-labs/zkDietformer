use std::{env, error::Error};
use std::fs::File;
use std::{ops::Bound, str::FromStr, path::Path};
use ark_std::{One};
use ark_ff::PrimeField;
use anyhow::{Context, Result};
use ndarray_npy::NpzReader;
use ndarray::{Array4, ArrayD, Ix4, Array2, Array0};
use num_bigint::BigInt;
use crate::harisa::hash_to_prime::primality_test;
use std::collections::HashMap;

// 테이블 소수테이블로 만들기
pub fn lookup_setgen(set: Vec<BigInt>) -> (Vec<BigInt>, Vec<BigInt>) {
    use crate::harisa::hash_to_prime::hash_to_prime;
    use rand::thread_rng;
    println!("lookup_setgen: generating prime and z tables for {} elements", set.len());
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

/// 사전 계산된 데이터를 사용하여 O(1) 속도로 소수/z 테이블을 구성
pub fn lookup_setgen_from_precomputed(
    u_bigints: Vec<BigInt>,
    full_table: &[BigInt],       // NPZ에서 로드한 'tables'
    full_prime_table: &[BigInt], // NPZ에서 로드한 'prime_tables'
    full_z_table: &[BigInt],     // NPZ에서 로드한 'z_tables'
) -> (Vec<BigInt>, Vec<BigInt>) {
    
    // 1. 값(Table Value)에서 인덱스로의 매핑 생성 (검색 최적화)
    let mut val_to_idx = HashMap::new();
    for (i, val) in full_table.iter().enumerate() {
        val_to_idx.insert(val.clone(), i);
    }

    let mut prime_table = Vec::with_capacity(u_bigints.len());
    let mut z_prime = Vec::with_capacity(u_bigints.len());

    // 2. 입력 u_bigints에 대해 미리 계산된 소수 매핑
    for u in u_bigints {
        let idx = val_to_idx.get(&u)
            .expect("Value not found in precomputed softmax table. Table mismatch?");
        
        prime_table.push(full_prime_table[*idx].clone());
        z_prime.push(full_z_table[*idx].clone());
    }

    (prime_table, z_prime)
}