use std::env;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray_npy::NpzReader;

use ark_relations::r1cs::ConstraintSynthesizer;
use ndarray::{Array4, ArrayD, Ix4};

use ark_bls12_381::{Bls12_381, Fr as F};
use ark_ff::{PrimeField, Zero, One};
use ark_relations::r1cs::ConstraintSystem;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use nystromformer::{cc_snark::cc_gro::{CcGroth16, prepare_verifying_key}, circuits::freivalds_circuit::FreivaldsCircuit};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};


fn read_u64_array(npz: &mut NpzReader<File>, name: &str) -> Result<ArrayD<u64>> {
    // npz 파일에서 저장했던 대로 읽어오기
    let arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>(&format!("{}.npy", name))
        .context(format!("missing key '{}'", name))?;
    Ok(arr)
}

fn load_npz(path: &Path) -> Result<(
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
    let a_arr = read_u64_array(&mut npz, "a")?;
    let b_arr = read_u64_array(&mut npz, "b")?;
    let c_arr = read_u64_array(&mut npz, "c")?;
    let div_arr = read_u64_array(&mut npz, "div")?;
    let rem_arr = read_u64_array(&mut npz, "rem")?;
    let r_arr = read_u64_array(&mut npz, "r")?;
    let input_0_arr = read_u64_array(&mut npz, "input_0")?;
    let c0_arr = read_u64_array(&mut npz, "c_0_converted")?;
    let mult_arr = read_u64_array(&mut npz, "multiplier")?;

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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: proof_runner path/to/inputs.npz");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);

    let (a_int, b_int, c_int, rem_int, div_int, r_int, input_0_int, c0_int, mult_int, meta) =
        load_npz(path).context("loading npz")?;
    let (batch, num_head, seq_len, head_dim) = meta;

    // convert integer matrices to field elements (Fr)
    let mut a_f = vec![vec![vec![vec![F::zero(); head_dim]; seq_len]; num_head]; batch];
    let mut b_f = vec![vec![vec![vec![F::zero(); seq_len]; head_dim]; num_head]; batch];
    let mut c_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    let mut rem_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    let mut div_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];

    for n in 0..batch {
        for h in 0..num_head {
            for i in 0..seq_len {
                for k in 0..head_dim {
                    // cast u128 -> u64 may truncate if values too large; ensure quantization keeps values small
                    a_f[n][h][i][k] = F::from(a_int[n][h][i][k] as u64);
                    b_f[n][h][k][i] = F::from(b_int[n][h][k][i] as u64);
                }
            }
            for i in 0..seq_len {
                for j in 0..seq_len {
                    c_f[n][h][i][j] = F::from(c_int[n][h][i][j] as u64);
                    div_f[n][h][i][j] = F::from(div_int[n][h][i][j] as u64);
                    rem_f[n][h][i][j] = F::from(rem_int[n][h][i][j] as u64);
                }
            }
        }
    }

    let r_f = F::from(r_int as u64);
    let input_0_f = F::from(input_0_int as u64);
    let c0_f = F::from(c0_int as u64);
    let mult_f = F::from(mult_int as u64);

    // build circuit
    let circuit = FreivaldsCircuit {
        a: a_f.clone(),
        b: b_f.clone(),
        c: c_f.clone(),
        r: r_f,
        remainder: rem_f.clone(),
        div: div_f.clone(),
        input_0: input_0_f,
        c_0_converted: c0_f,
        multiplier: mult_f,
    };

    // --- DEBUG ---
    {
        let mut found = false;

        let mut check_with = |start_with_one: bool| {
            println!("Checking with r_pows starting at {}", if start_with_one { "r^0 (1, r, r^2...)" } else { "r^1 (r, r^2...)" });
            for n in 0..batch {
                for h in 0..num_head {
                    let a_mat = &a_f[n][h];
                    let b_mat = &b_f[n][h];
                    let c_mat = &c_f[n][h];
                    let div_mat = &div_f[n][h];
                    let rem_mat = &rem_f[n][h];

                    // build r_pows
                    let mut r_pows: Vec<F> = Vec::with_capacity(seq_len);
                    if seq_len > 0 {
                        if start_with_one {
                            r_pows.push(F::one()); // r^0
                            for idx in 1..seq_len {
                                r_pows.push(r_pows[idx - 1] * r_f);
                            }
                        } else {
                            r_pows.push(r_f); // r^1
                            for idx in 1..seq_len {
                                r_pows.push(r_pows[idx - 1] * r_f);
                            }
                        }
                    }

                    // precompute common quantities
                    let mut sum_r = F::zero();
                    for rp in &r_pows { sum_r += *rp; }

                    // br = B * r_pows
                    let head_dim = a_mat[0].len();
                    let mut br: Vec<F> = vec![F::zero(); head_dim];
                    for k in 0..head_dim {
                        let mut acc = F::zero();
                        for j in 0..seq_len {
                            acc += b_mat[k][j] * r_pows[j];
                        }
                        br[k] = acc;
                    }

                    // term1 = A * br
                    let mut term1: Vec<F> = vec![F::zero(); seq_len];
                    for i in 0..seq_len {
                        let mut acc = F::zero();
                        for k in 0..head_dim {
                            acc += a_mat[i][k] * br[k];
                        }
                        term1[i] = acc;
                    }

                    // term2: row_sum(A) * b0 * sum_r
                    let b0 = input_0_f;
                    let a0 = input_0_f;
                    let mut term2: Vec<F> = vec![F::zero(); seq_len];
                    for i in 0..seq_len {
                        let mut row_sum = F::zero();
                        for k in 0..head_dim {
                            row_sum += a_mat[i][k];
                        }
                        term2[i] = row_sum * (b0 * sum_r);
                    }

                    // term3_scalar
                    let mut sum_br = F::zero();
                    for v in &br { sum_br += *v; }
                    let term3_scalar = a0 * sum_br;

                    // term4_val
                    let head_dim_fp = F::from(head_dim as u64);
                    let term4_val = a0 * b0 * sum_r * head_dim_fp;

                    // c0_corr
                    let c0_corr = c0_f * sum_r;
                    let m = mult_f;

                    // cc_mat and right side
                    let two8 = F::from(1u64 << 8);
                    let two22 = F::from(1u64 << 22);
                    let mut cc_mat: Vec<Vec<F>> = vec![vec![F::zero(); seq_len]; seq_len];
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            cc_mat[i][j] = (c_mat[i][j] + div_mat[i][j] * two8) * two22 + rem_mat[i][j];
                        }
                    }
                    let mut right_side: Vec<F> = vec![F::zero(); seq_len];
                    for i in 0..seq_len {
                        let mut acc = F::zero();
                        for j in 0..seq_len {
                            acc += cc_mat[i][j] * r_pows[j];
                        }
                        right_side[i] = acc;
                    }

                    // left side per row and compare; also print components
                    for i in 0..seq_len {
                        let left = (term1[i] - term2[i] - term3_scalar + term4_val + c0_corr) * m;
                        let right = right_side[i];
                        if left != right {
                            println!("Mismatch (start_with_one={}): batch={}, head={}, row={}", start_with_one, n, h, i);
                            println!(" left = {:?}", left);
                            println!(" right= {:?}", right);
                            println!(" left + right = {:?}", left + right);
                            println!(" term1 = {:?}", term1[i]);
                            println!(" term2 = {:?}", term2[i]);
                            println!(" term3_scalar = {:?}", term3_scalar);
                            println!(" term4_val = {:?}", term4_val);
                            println!(" c0_corr = {:?}", c0_corr);
                            println!(" a_row[0..3] = {:?}", &a_mat[i][..std::cmp::min(3, a_mat[i].len())]);
                            println!(" br[0..3] = {:?}", &br[..std::cmp::min(3, br.len())]);
                            found = true;
                            break;
                        }
                    }
                    if found { break; }
                }
                if found { break; }
            }
            if found {
                println!("Found mismatch for start_with_one={}", start_with_one);
            } else {
                println!("No mismatches for start_with_one={}", start_with_one);
            }
            found
        };

        // run both checks
        let mismatch_r1 = check_with(false); // r^1 start
        let mismatch_r0 = check_with(true);  // r^0 start

        if !mismatch_r1 && !mismatch_r0 {
            println!("Plain evaluation OK for both r^1 and r^0 conventions.");
        } else {
            println!("One or both conventions produced mismatches. Inspect outputs above.");
        }

        // 보정 다 빼고 그냥 순수하게 행렬곱이 맞는지 확인
        println!("Integer diagnostic (signed)...");
        for n in 0..batch {
          for h in 0..num_head {
            for i in 0..seq_len {
              for j in 0..seq_len {
                // compute signed s as i128
                let mut s_signed: i128 = 0;
                for k in 0..head_dim {
                  // a_int, b_int are u128; cast to i128 after subtracting input_0
                  let a_val = a_int[n][h][i][k] as i128 - input_0_int as i128;
                  let b_val = b_int[n][h][k][j] as i128 - input_0_int as i128;
                  s_signed += a_val * b_val;
                }
                let target_signed = (s_signed + c0_int as i128) * mult_int as i128;
                println!("n{} h{} i{} j{} s_signed={} target_signed={}", n,h,i,j,s_signed, target_signed);
              }
            }
          }
        }
    }
    // --- END DEBUG ---

    // 1) constraint count
    let cs = ConstraintSystem::<F>::new_ref();
    circuit.clone().generate_constraints(cs.clone()).context("generate_constraints failed")?;
    println!("Generated constraints: {}", cs.num_constraints());
    assert!(cs.num_constraints() > 0, "constraints should be greater than 0");

    // 2) Setup / Prove / Verify using CcGroth16
    let mut rng = StdRng::seed_from_u64(42u64);
    let (pk, vk) = CcGroth16::<Bls12_381>::setup(circuit.clone(), &mut rng).unwrap();
    let pvk = prepare_verifying_key::<Bls12_381>(&vk);

    let proof = CcGroth16::<Bls12_381>::prove(&pk, circuit, &mut rng).unwrap();

    // 현재 서킷에 Public Input(new_input_variable)이 없다면 빈 벡터 []를 전달.
    let public_inputs = vec![]; 
    let result = CcGroth16::<Bls12_381>::verify_with_processed_vk(&pvk, &public_inputs, &proof).unwrap();
        
    assert!(result, "cc-Groth16 proof verification failed!");

    Ok(())
}