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


pub fn to_field_1d_i64<F: PrimeField>(array: &ndarray::Array1<i64>) -> Vec<F> {
    array
        .iter()
        .map(|&val| {
            if val >= 0 {
                F::from(val as u64)
            } else {
                -F::from(val.abs() as u64)
            }
        })
        .collect()
}


pub fn to_field_2d_i64<F: PrimeField>(arr: &ndarray::Array2<i64>) -> Vec<Vec<F>> {
    let (rows, cols) = arr.dim();
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(F::from(arr[[i, j]]));
        }
        out.push(row);
    }
    out
}


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

pub fn to_field_3d_i64<F: PrimeField>(arr: &ndarray::Array3<i64>) -> Vec<Vec<Vec<F>>> {
    let (batch, head, seq) = arr.dim();
    let mut out_batch = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut out_head = Vec::with_capacity(head);
        for h in 0..head {
            let mut out_seq = Vec::with_capacity(seq);
            for s in 0..seq {
                out_seq.push(F::from(arr[[b, h, s]]));
            }
            out_head.push(out_seq);
        }
        out_batch.push(out_head);
    }
    out_batch
}

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

