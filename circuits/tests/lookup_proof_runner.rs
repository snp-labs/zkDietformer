use std::{str::FromStr};
use std::error::Error;
use num_bigint::BigInt;
use ark_std::{test_rng, rand::SeedableRng, One};
use rand::RngCore;
use ark_ec::pairing::Pairing;
use nystromformer::{
    harisa::{
    harisa::Harisa, 
    arithm::ArithmCircuit,
    type_conversion::bigint_to_fr,
    },
    linker::snark::LinkSnark,
    lookup::{
        copy_this_or_that::CTTCircuit,
        well_transformed::WTCircuit,
        lookup::HarisaPlus
    },
    util::{load::load_tables, table::lookup_setgen},
};

/// Duplex 룩업 증명 생성하기
pub fn create_lookup_proof<E: Pairing>(
    npz_path: &str,
    matmul_ints: Vec<Vec<i128>>,
    batch_size: usize,
) -> Result<(), Box<dyn Error>>
where
    <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    // RNG
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    // 1) 테이블 로드
    let (raw_tables, _frac_bits, limb_bits) = load_tables(npz_path)?;

    let k_segments = raw_tables.len();
    if k_segments == 0 {
        return Err("No table segments found".into());
    }
    let table_width = raw_tables[0].len();
    if table_width == 0 {
        return Err("Table width is zero".into());
    }

    // 2) flat_table 구성 (row-major: segment0[..], segment1[..], ...)
    let mut flat_table: Vec<BigInt> = Vec::with_capacity(k_segments * table_width);
    for k in 0..k_segments {
        for idx in 0..table_width {
            flat_table.push(BigInt::from(raw_tables[k][idx]));
        }
    }

    println!("Table value is : {:?}", flat_table);

    // 3) u_bigints: matmul_ints -> limb 인덱스로 찾아 flat_table 항목 수집
    let mut u_bigints: Vec<BigInt> = Vec::new();
    let limb_mask: u128 = if limb_bits >= 128 { !0u128 } else { (1u128 << limb_bits) - 1u128 };
    for i in 0..matmul_ints.len() {
        for j in 0..matmul_ints[i].len() {
            let val = matmul_ints[i][j];
            if val < 0 {
                return Err(format!("Negative matmul input not supported: ({},{}) = {}", i, j, val).into());
            }
            let unsigned = val as u128;
            for k in 0..k_segments {
                let limb_val = ((unsigned >> (k * limb_bits)) & limb_mask) as usize;
                let flat_idx = k * table_width + limb_val;
                // safety check
                if flat_idx >= flat_table.len() {
                    return Err(format!("Computed flat index out of range: {}", flat_idx).into());
                }
                u_bigints.push(flat_table[flat_idx].clone());
            }
        }
    }

    // 4) sanity check for batch_size
    if batch_size == 0 {
        return Err("batch_size must be > 0".into());
    }
    if batch_size > flat_table.len() {
        return Err(format!(
            "batch_size ({}) is larger than full table length ({}) — must be <= flat_table.len()",
            batch_size,
            flat_table.len()
        ).into());
    }
    if batch_size > u_bigints.len() {
        return Err(format!(
            "batch_size ({}) is larger than number of referenced table entries (u_bigints.len() = {}) — \
            typically batch_size should be <= rows*cols*k_segments",
            batch_size,
            u_bigints.len()
        ).into());
    }

    print!("u_bigints: {:?}", u_bigints);

    // 5) prime/z 테이블 생성 (lookup_setgen)
    let (prime_table, mut z_table) = lookup_setgen(u_bigints.clone());
    /*let (mut prime_table, mut z_table) = lookup_setgen_from_precomputed(
        u_bigints.clone(),
        &flat_table,
        &flat_prime,
        &flat_z,
    );*/

    // safety: prime_table.len() == u_bigints.len()
    if prime_table.len() != u_bigints.len() || z_table.len() != u_bigints.len() {
        return Err("prime_table/z_table length mismatch with u_bigints".into());
    }

    // 6) build vec_table, sort 및 proven/non_proven 분할 (test.rs 방식)
    let mut vec_table: Vec<(BigInt, BigInt, BigInt)> = prime_table
        .iter()
        .cloned()
        .zip(u_bigints.iter().cloned())
        .zip(z_table.iter().cloned())
        .map(|((x, y), z)| (x, y, z))
        .collect();
    vec_table.sort();

    z_table.sort();
    let mut _set_table = Vec::new();

    // Extract proven (first batch_size) into hat_f, t, z_f
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

    let bit_shift = BigInt::from(base_2.pow(expo));

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

    let proven_elem: Vec<BigInt> = [hat_f.clone(), z_f.clone()].concat();

    //  --- 디버깅 ---
    let mut non_proven_elem: Vec<BigInt> = Vec::new();
    let mut non_proven_z: Vec<BigInt> = Vec::new();
    for i in batch_size..u_bigints.len() {
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

    _set_table = [proven_elem.clone(), non_proven_elem.clone()].concat(); 

    let prod_set_table: BigInt = _set_table.clone().iter().product();

    assert_eq!(
        prod_set_table.clone(),
        prod_non_proven.clone() * prod_prime_table.clone(),
        "Exponentiation Failed"
    );



    let arithm_circuit = ArithmCircuit::<E::ScalarField>::mock(2 * batch_size);
    let ctt_circuit = CTTCircuit::<E::ScalarField>::mock(2 * batch_size, 2 * batch_size);
    let wt_circuit = WTCircuit::<E::ScalarField>::mock(batch_size, batch_size, batch_size);

    // 8) generate Harisa parameters & tree (this can be expensive)
    let (pp, tree) = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_parameters(
        _set_table.clone(),
        ctt_circuit,
        wt_circuit,
        arithm_circuit,
        &mut rng,
    ).unwrap();

    let mut _prod_set: BigInt = _set_table.clone().iter().product();

    // 9) accum as in test.rs
    let accum = tree[0].clone().modpow(&_set_table[0].clone(), &pp.m_pp.mod_n);

    // 10) Prepare circuit inputs (convert BigInt -> Field)
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

    // 11) build circuit instances used for the Harisa proof
    let ctt_elem = [circuit_hat_f.clone(), circuit_z_f.clone()].concat();
    let ctt_circuit = CTTCircuit::<E::ScalarField>::new(ctt_elem.clone(), ctt_elem.clone());
    let wt_circuit = WTCircuit::<E::ScalarField>::new(circuit_hat_f.clone(), circuit_t.clone(), circuit_z_f.clone());

    // 12) generate lookup proof (prove)
    let proof = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_proof(
        pp.clone(),
        accum.clone(),
        tree.clone(),
        hat_f.clone(),
        t.clone(),
        z_f.clone(),
        ctt_circuit,
        wt_circuit,
        &mut rng,
        non_proven_elem.clone(),
    )?;

    // 13) verify proof immediately (optional sanity-check)
    let flag = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::verify_lookup(
        pp,
        accum,
        proof.cm_f_hat,
        proof.cm_f,
        proof.cm_z,
        proof,
    )?;
    if !flag {
        return Err("Harisa lookup proof failed verification".into());
    }

    // Done
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;

    #[test]
    fn test_lookup_proof_runner() {
        let npz_path = "tmp_tables/softmax_tables_quantized.npz";

        // 행렬곱 예시
        // 짝수여야 하고, batch_size보다 크면 안됨
        // 예를 들어 batch_size (10) is larger than full table length (8)
        // let matmul_ints: Vec<Vec<i128>> = vec![vec![1i128, 1i128, 1i128, 1i128, 1i128]];
        let matmul_ints: Vec<Vec<i128>> = vec![vec![1i128, 1i128, 1i128, 1i128]];

        // 테이블 로드하기
        let (raw_tables, _frac_bits, _limb_bits) = load_tables(npz_path)
            .expect("failed to load npz in test (ensure tmp_tables/softmax_tables_quantized.npz exists)");

        let k_segments = raw_tables.len();
        assert!(k_segments > 0, "k_segments must be > 0");

        let rows = matmul_ints.len();
        let cols = if rows > 0 { matmul_ints[0].len() } else { 0 };

        let batch_size = rows * cols * k_segments;

        println!("Test info: rows={}, cols={}, k_segments={}, batch_size={}", rows, cols, k_segments, batch_size);
        // 룩업 증명 생성
        create_lookup_proof::<Bn254>(npz_path, matmul_ints.clone(), batch_size)
            .expect("create_lookup_proof failed");

        println!("create_lookup_proof completed successfully");

    }
}

