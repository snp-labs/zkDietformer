use std::{env, error::Error};
use std::fs::File;
use std::{ops::Bound, str::FromStr, path::Path};
use ark_std::{One};
use ark_ff::PrimeField;
use anyhow::{Context, Result};
use ndarray_npy::NpzReader;
use ndarray::{Array4, ArrayD, Ix4, Array2, Array0};
use num_bigint::BigInt;

//  npz 파일에서 unsigned 64비트 배열을 읽어오는 함수
pub fn read_array_u64(npz: &mut NpzReader<File>, key: &str) -> Result<ArrayD<u64>, anyhow::Error> {
    let arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&format!("{}.npy", key))?;
    Ok(arr)
}

/// npz 파일에서 데이터 로드하기[test/freivalds_proof_runner.rs 전용]
pub fn load_npz(path: &Path) -> Result<(
    Vec<Vec<Vec<Vec<u128>>>>, // a
    Vec<Vec<Vec<Vec<u128>>>>, // b
    Vec<Vec<Vec<Vec<u128>>>>, // c
    Vec<Vec<Vec<Vec<u128>>>>, // rem
    Vec<Vec<Vec<Vec<u128>>>>, // div
    u128, // r
    u128, // input_0
    u128, // c0
    u128, // multiplier
    (usize, usize, usize, usize) // batch,num_head,seq_len,head_dim
)> {
    let file = File::open(path).context("opening npz file")?;
    let mut npz = NpzReader::new(file);

    // npz 파일에서 값을 읽어오기
    let mut npz = npz?;
    let a_arr = read_array_u64(&mut npz, "a_k1")?;
    let b_arr = read_array_u64(&mut npz, "b_k1")?;
    let c_arr = read_array_u64(&mut npz, "c_k1")?;
    let div_arr = read_array_u64(&mut npz, "div_k1")?;
    let rem_arr = read_array_u64(&mut npz, "rem_k1")?;
    let r_arr = read_array_u64(&mut npz, "r")?;
    let input_0_arr = read_array_u64(&mut npz, "input_0_k1")?;
    let c0_arr = read_array_u64(&mut npz, "c_0_converted_k1")?;
    let mult_arr = read_array_u64(&mut npz, "multiplier")?;
    // AFTER reading mult_arr etc.
    let frac_bits_opt: Option<u32> = match npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::IxDyn>("frac_bits.npy") {
        Ok(arr) => {
            let arr4 = arr.into_dimensionality::<ndarray::IxDyn>().ok();
            // if single scalar, extract it; (API specifics vary)
            if let Some(a) = arr4 {
                // try to get first element
                a.iter().next().map(|v| *v as u32)
            } else { None }
        },
        Err(_) => None,
    };
    if let Some(fb) = frac_bits_opt {
        println!("RUST DEBUG: frac_bits = {}", fb);
    }

    // Convert to fixed 4D arrays
    let a4: Array4<u64> = a_arr.into_dimensionality::<Ix4>()
        .context("expected 'a' to be 4-dimensional: (batch, num_head, seq_len, head_dim)")?;
    let b4: Array4<u64> = b_arr.into_dimensionality::<Ix4>()
        .context("expected 'b' to be 4-dimensional: (batch, num_head, head_dim, seq_len)")?;
    let c4: Array4<u64> = c_arr.into_dimensionality::<Ix4>()
        .context("expected 'c' to be 4-dimensional: (batch, num_head, seq_len, seq_len)")?;
    let div4: Array4<u64> = div_arr.into_dimensionality::<Ix4>()
        .context("expected 'div' to be 4-dimensional")?;
    let rem4: Array4<u64> = rem_arr.into_dimensionality::<Ix4>()
        .context("expected 'rem' to be 4-dimensional")?;

    // 값이 맞는지 확인
    println!("RUST DEBUG: shapes: a {:?}, b {:?}, c {:?}, div {:?}, rem {:?}", a4.shape(), b4.shape(), c4.shape(), div4.shape(), rem4.shape());
    // sample elements (will panic if index out of range; only for small test files)
    println!("RUST DEBUG sample a[0,0,0,0] = {}", a4[[0,0,0,0]]);
    println!("RUST DEBUG sample b[0,0,0,0] = {}", b4[[0,0,0,0]]);
    println!("RUST DEBUG sample c[0,0,0,0] = {}", c4[[0,0,0,0]]);
    println!("RUST DEBUG sample div[0,0,0,0] = {}", div4[[0,0,0,0]]);
    println!("RUST DEBUG sample rem[0,0,0,0] = {}", rem4[[0,0,0,0]]);

    let a_shape = a4.shape();
    let (batch, num_head, seq_len, head_dim) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);

    let mut a = vec![vec![vec![vec![0u128; head_dim]; seq_len]; num_head]; batch];
    let mut b = vec![vec![vec![vec![0u128; seq_len]; head_dim]; num_head]; batch];
    let mut c = vec![vec![vec![vec![0u128; seq_len]; seq_len]; num_head]; batch];
    let mut div = vec![vec![vec![vec![0u128; seq_len]; seq_len]; num_head]; batch];
    let mut rem = vec![vec![vec![vec![0u128; seq_len]; seq_len]; num_head]; batch];

    // 4D로 읽어온 값을 u128 벡터로 복사
    for ((n, h, i, k), val) in a4.indexed_iter() {
        a[n][h][i][k] = *val as u128;
    }
    for ((n, h, k, i), val) in b4.indexed_iter() {
        b[n][h][k][i] = *val as u128;
    }
    for ((n, h, i, j), val) in c4.indexed_iter() {
        c[n][h][i][j] = *val as u128;
    }
    for ((n, h, i, j), val) in div4.indexed_iter() {
        div[n][h][i][j] = *val as u128;
    }
    for ((n, h, i, j), val) in rem4.indexed_iter() {
        rem[n][h][i][j] = *val as u128;
    }

    let r = r_arr.iter().next().copied().unwrap_or(0u64) as u128;
    let input_0 = input_0_arr.iter().next().copied().unwrap_or(0u64) as u128;
    let c0 = c0_arr.iter().next().copied().unwrap_or(0u64) as u128;
    let mult = mult_arr.iter().next().copied().unwrap_or(1u64) as u128;

    Ok((a, b, c, rem, div, r, input_0, c0, mult, (batch, num_head, seq_len, head_dim)))
}

/// 테이블 로드하기 (필드 요소로 변환)
pub fn load_tables<F: PrimeField, P: AsRef<Path>>(
    npz_path: P,
) -> Result<(Vec<Vec<F>>, u32, usize), Box<dyn Error>> {
    let f = File::open(&npz_path)?;
    let mut npz = NpzReader::new(f)?;

    // Read the tables (2D array of u64)
    let tables_arr: Array2<u64> = npz.by_name("tables.npy")?;

    // Read scalar metadata as 0-d arrays and extract
    let frac_bits_arr: Array0<i32> = npz.by_name("frac_bits.npy")?;
    let limb_bits_arr: Array0<i32> = npz.by_name("limb_bits.npy")?;
    let frac_bits: i32 = frac_bits_arr[()];
    let limb_bits: i32 = limb_bits_arr[()];

    let k = tables_arr.shape()[0];
    let row_len = tables_arr.shape()[1];

    let mut tables: Vec<Vec<F>> = Vec::with_capacity(k);
    for i in 0..k {
        let row = tables_arr.row(i);
        let mut v = Vec::with_capacity(row_len);
        for val in row.iter() {
            let fe = F::from(*val as u128);
            v.push(fe);
        }
        tables.push(v);
    }

    Ok((tables, frac_bits as u32, limb_bits as usize))
}

#[test]
fn print_npz_tables() {
    // npz 경로 설정
    let npz_path = "tmp_tables/embedded_softmax_for_lookup.npz";

    // 테이블 로드
    let (tables, frac_bits, limb_bits) =load_tables::<ark_bn254::Fr,_,>(npz_path).expect("failed to load npz");

    // 테이블 정보 출력
    println!("Loaded npz: {}", npz_path);
    println!("k_segments (rows) = {}", tables.len());
    if !tables.is_empty() {
        println!("table_width (cols) = {}", tables[0].len());
    }
    println!("frac_bits = {}", frac_bits);
    println!("limb_bits = {}", limb_bits);

    // 각 세그먼트의 처음 몇 개 항목 출력(여기서는 다섯개)
    let show = 5usize.min(if tables.is_empty() { 0 } else { tables[0].len() });
    for (k, row) in tables.iter().enumerate() {
        print!("segment {}: first {} entries = [", k, show);
        for i in 0..show {
            // convert field element to BigInt string for deterministic printing
            let s = row[i].into_bigint().to_string();
            if i + 1 < show {
                print!("{}, ", s);
            } else {
                print!("{}", s);
            }
        }
        println!("]");
    }
}