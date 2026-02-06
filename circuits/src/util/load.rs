use std::{env, error::Error};
use std::fs::File;
use std::{ops::Bound, str::FromStr, path::Path};
use ark_std::{One};
use ark_ff::PrimeField;
use anyhow::{Context, Result};
use ndarray_npy::NpzReader;
use ndarray::{Array4, ArrayD, Ix4, Array2, Array0, IxDyn};
use num_bigint::BigInt;

//  npz 파일에서 unsigned 64비트 배열을 읽어오는 함수
pub fn read_array_u64(npz: &mut NpzReader<File>, key: &str) -> Result<ArrayD<u64>, anyhow::Error> {
    let arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&format!("{}.npy", key))?;
    Ok(arr)
}

pub fn read_array_i64(npz: &mut NpzReader<File>, name: &str) -> Result<ArrayD<i64>, Box<dyn Error>> {
    let key = format!("{}.npy", name);
    let arr = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>(&key)?;
    Ok(arr.into_owned().into_dimensionality().unwrap())
}

/// ndarray::Array4<u64> 데이터를 Vec<Vec<Vec<Vec<F>>>> 형태로 변환
pub fn to_field_4d_i64<F: PrimeField>(arr: &Array4<i64>) -> Vec<Vec<Vec<Vec<F>>>> {
    let (batch, head, seq, dim) = arr.dim();
    let mut out_batch = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut out_head = Vec::with_capacity(head);
        for h in 0..head {
            let mut out_seq = Vec::with_capacity(seq);
            for s in 0..seq {
                let mut out_dim = Vec::with_capacity(dim);
                for d in 0..dim {
                    // u64 값을 필드 요소 F로 변환
                    out_dim.push(F::from(arr[[b, h, s, d]]));
                }
                out_seq.push(out_dim);
            }
            out_head.push(out_seq);
        }
        out_batch.push(out_head);
    }
    out_batch
}

pub fn to_field_4d_u64<F: PrimeField>(arr: &Array4<u64>) -> Vec<Vec<Vec<Vec<F>>>> {
    let (b, h, s, l) = arr.dim();
    let mut res_b = Vec::with_capacity(b);
    for i in 0..b {
        let mut res_h = Vec::with_capacity(h);
        for j in 0..h {
            let mut res_s = Vec::with_capacity(s);
            for k in 0..s {
                let mut res_l = Vec::with_capacity(l);
                for m in 0..l {
                    res_l.push(F::from(arr[[i, j, k, m]]));
                }
                res_s.push(res_l);
            }
            res_h.push(res_s);
        }
        res_b.push(res_h);
    }
    res_b
}

pub fn to_field_5d_i64<F: PrimeField>(arr: &ArrayD<i64>) -> Vec<Vec<Vec<Vec<Vec<F>>>>> {
    let shape = arr.shape();
    let (seg, b, h, s, l) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let mut res_seg = Vec::with_capacity(seg);

    for i in 0..seg {
        let mut res_b = Vec::with_capacity(b);
        for j in 0..b {
            let mut res_h = Vec::with_capacity(h);
            for k in 0..h {
                let mut res_s = Vec::with_capacity(s);
                for m in 0..s {
                    let mut res_l = Vec::with_capacity(l);
                    for n in 0..l {
                        res_l.push(F::from(arr[[i, j, k, m, n]]));
                    }
                    res_s.push(res_l);
                }
                res_h.push(res_s);
            }
            res_b.push(res_h);
        }
        res_seg.push(res_b);
    }
    res_seg
}

pub fn to_field_5d_u64<F: PrimeField>(arr: &ArrayD<u64>) -> Vec<Vec<Vec<Vec<Vec<F>>>>> {
    let shape = arr.shape();
    let (seg, b, h, s, l) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let mut res_seg = Vec::with_capacity(seg);

    for i in 0..seg {
        let mut res_b = Vec::with_capacity(b);
        for j in 0..b {
            let mut res_h = Vec::with_capacity(h);
            for k in 0..h {
                let mut res_s = Vec::with_capacity(s);
                for m in 0..s {
                    let mut res_l = Vec::with_capacity(l);
                    for n in 0..l {
                        res_l.push(F::from(arr[[i, j, k, m, n]]));
                    }
                    res_s.push(res_l);
                }
                res_h.push(res_s);
            }
            res_b.push(res_h);
        }
        res_seg.push(res_b);
    }
    res_seg
}

/// ndarray::Array3<u64> 데이터를 Vec<Vec<Vec<F>>> 형태로 변환 (B, H, S)
pub fn to_field_3d_u64<F: PrimeField>(arr: &ndarray::Array3<u64>) -> Vec<Vec<Vec<F>>> {
    let (batch, head, seq) = arr.dim();
    let mut out_batch = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut out_head = Vec::with_capacity(head);
        for h in 0..head {
            let mut out_seq = Vec::with_capacity(seq);
            for s in 0..seq {
                // u64 값을 필드 요소 F로 변환
                out_seq.push(F::from(arr[[b, h, s]]));
            }
            out_head.push(out_seq);
        }
        out_batch.push(out_head);
    }
    out_batch
}

/// 테이블 로드하기 (필드 요소로 변환)
pub fn load_tables<F: PrimeField, P: AsRef<Path>>(
    npz_path: P,
) -> Result<(Vec<Vec<F>>, u32, usize), Box<dyn Error>> {
    let f = File::open(&npz_path)?;
    let mut npz = NpzReader::new(f)?;

    // Read the tables (2D array of u64)
    let tables_arr: Array2<u64> = npz.by_name("tables.npy").or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::Ix2>("softmax_tables.npy"))
    .or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::Ix2>("softmax_tables.npy"))?;

    // Read scalar metadata as 0-d arrays and extract
    let frac_bits_arr: Array0<i32> = npz.by_name("frac_bits.npy")?;
    let limb_bits_arr: Array0<i32> = npz.by_name("limb_bits.npy")?;
    let frac_bits: i32 = frac_bits_arr[()];
    let limb_bits: i32 = limb_bits_arr[()];

    let k = tables_arr.shape()[0];
    let row_len = tables_arr.shape()[1];

    let mut tables: Vec<Vec<F>> = Vec::with_capacity(k);
    /*for i in 0..k {
        let row = tables_arr.row(i);
        let mut v = Vec::with_capacity(row_len);
        for val in row.iter() {
            let fe = F::from(*val as u128);
            v.push(fe);
        }
        tables.push(v);
    }*/
    for i in 0..k {
        let row = tables_arr.row(i);
        let mut v = Vec::with_capacity(row_len);
        for &val in row.iter() { v.push(F::from(val));}
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