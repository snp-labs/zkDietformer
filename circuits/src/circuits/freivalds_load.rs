use anyhow::{Context, Result};
use ndarray::{Ix0, Ix4, IxDyn};
use ndarray_npy::NpzReader;
use std::fs::File;
use ark_ff::PrimeField;
use crate::circuits::freivalds_circuit::FreivaldsCircuit;
use crate::util::load::read_array_u64;

pub fn load_freivalds_circuit<F: PrimeField>(
    npz: &mut NpzReader<File>,
    suffix: &str,
    r_challenge: F,
    multiplier_global: F,
) -> Result<FreivaldsCircuit<F>> {
    // 1. 행렬 데이터 로드 (a, b, c, div, rem)
    let a_arr = read_array_u64(npz, &format!("a{}", suffix)).context(format!("reading a{}", suffix))?;
    let b_arr = read_array_u64(npz, &format!("b{}", suffix)).context(format!("reading b{}", suffix))?;
    let c_arr = read_array_u64(npz, &format!("c{}", suffix)).context(format!("reading c{}", suffix))?;
    let div_arr = read_array_u64(npz, &format!("div{}", suffix)).context(format!("reading div{}", suffix))?;
    let rem_arr = read_array_u64(npz, &format!("rem{}", suffix)).context(format!("reading rem{}", suffix))?;

    // 4차원 변환
    let a_4 = a_arr.into_dimensionality::<Ix4>().context("a must be 4D")?;
    let b_4 = b_arr.into_dimensionality::<Ix4>().context("b must be 4D")?;
    let c_4 = c_arr.into_dimensionality::<Ix4>().context("c must be 4D")?;
    let div_4 = div_arr.into_dimensionality::<Ix4>().context("div must be 4D")?;
    let rem_4 = rem_arr.into_dimensionality::<Ix4>().context("rem must be 4D")?;

    let shape = a_4.shape();
    let (batch, num_head, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let b_cols = b_4.shape()[3];
    let c_cols = c_4.shape()[3];

    // 2. 가변 배열 생성 및 채우기
    let mut a_f = vec![vec![vec![vec![F::zero(); head_dim]; seq_len]; num_head]; batch];
    let mut b_f = vec![vec![vec![vec![F::zero(); b_cols]; head_dim]; num_head]; batch];
    let mut c_f = vec![vec![vec![vec![F::zero(); c_cols]; seq_len]; num_head]; batch];
    let mut div_f = vec![vec![vec![vec![F::zero(); c_cols]; seq_len]; num_head]; batch];
    let mut rem_f = vec![vec![vec![vec![F::zero(); c_cols]; seq_len]; num_head]; batch];

    for ((n, h, i, k), val) in a_4.indexed_iter() { a_f[n][h][i][k] = F::from(*val); }
    for ((n, h, k, i), val) in b_4.indexed_iter() { b_f[n][h][k][i] = F::from(*val); }
    for ((n, h, i, j), val) in c_4.indexed_iter() { c_f[n][h][i][j] = F::from(*val); }
    for ((n, h, i, j), val) in div_4.indexed_iter() { div_f[n][h][i][j] = F::from(*val); }
    for ((n, h, i, j), val) in rem_4.indexed_iter() { rem_f[n][h][i][j] = F::from(*val); }

    // 3. 스칼라 값 로드 (input_0, c_0_converted)
    let input0_name = format!("input_0{}", suffix);
    let c0_name = format!("c_0_converted{}", suffix);

    let input_0 = F::from(npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&format!("{}.npy", input0_name))
        .or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&input0_name))
        .map(|a| a.iter().next().copied().unwrap_or(0))
        .unwrap_or(0));

    let c_0_converted = F::from(npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&format!("{}.npy", c0_name))
        .or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&c0_name))
        .map(|a| a.iter().next().copied().unwrap_or(0))
        .unwrap_or(0));

    Ok(FreivaldsCircuit {
        a: a_f,
        b: b_f,
        c: c_f,
        r: r_challenge,
        remainder: rem_f,
        div: div_f,
        input_0,
        c_0_converted,
        multiplier: multiplier_global,
    })
}