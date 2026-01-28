#[cfg(test)]
mod tests {
    use ark_ff::{PrimeField, Zero, One};
    use nystromformer::gadgets::softmax::constraints::{build_softmax_witnesses, decompose_to_limbs_dynamic};
    use nystromformer::util::load::load_tables;
    use nystromformer::circuits::softmax_circuit::{SoftmaxCircuit, K_SEGMENTS};
    use nystromformer::cc_snark::cc_gro::{prepare_verifying_key, CcGroth16};
    use ark_crypto_primitives::snark::{SNARK};
    use ark_bn254::{Bn254, Fr as F};
    use ark_std::test_rng;
    use rand::{RngCore, SeedableRng};
    use ark_relations::r1cs::ConstraintSynthesizer;
    use std::convert::TryInto;

    fn verify_softmax_witnesses(
        matmul_ints: &Vec<Vec<i128>>,
        tables: &Vec<Vec<F>>,
        limb_bits: usize,
        matmul_outputs: &Vec<Vec<Vec<Vec<F>>>>,
        limbs: &Vec<Vec<Vec<Vec<Vec<F>>>>>,
        exp_limb_vals: &Vec<Vec<Vec<Vec<Vec<F>>>>>,
        softmax_outputs: &Vec<Vec<Vec<Vec<F>>>>,
    ) {
        let k_segments = tables.len();
        let rows = matmul_outputs[0][0].len();
        let cols = matmul_outputs[0][0][0].len();

        // 1) recomposition: limb ints -> recomposed == original integer
        for i in 0..rows {
            for j in 0..cols {
                let orig = matmul_ints[i][j];
                assert!(orig >= 0, "orig negative at ({},{}) = {}", i, j, orig);
                let unsigned = orig as u128;
                let limbs_ints = decompose_to_limbs_dynamic(unsigned, limb_bits, k_segments);
                let mut recomposed: u128 = 0;
                for k in 0..k_segments {
                    recomposed |= limbs_ints[k] << (k * limb_bits);
                }
                assert_eq!(recomposed, unsigned, "Recompose mismatch at ({},{}): recomposed={} orig={}", i, j, recomposed, unsigned);

                // limb field elements check
                for k in 0..k_segments {
                    let expected = F::from(limbs_ints[k]);
                    let got = limbs[0][0][i][j][k];
                    assert_eq!(expected, got, "Limb field mismatch at ({},{},{}): expected {:?}, got {:?}", i, j, k, expected.into_bigint(), got.into_bigint());
                }
            }
        }

        // 2) exp_limb_vals should equal tables[k][limb]
        for i in 0..rows {
            for j in 0..cols {
                let unsigned = matmul_ints[i][j] as u128;
                let limbs_ints = decompose_to_limbs_dynamic(unsigned, limb_bits, k_segments);
                for k in 0..k_segments {
                    let idx = limbs_ints[k] as usize;
                    assert!(idx < tables[k].len(), "Table index out of range at ({},{},{}): idx {} >= width {}", i, j, k, idx, tables[k].len());
                    let expected = tables[k][idx];
                    let got = exp_limb_vals[0][0][i][j][k];
                    assert_eq!(expected, got, "exp_limb_val mismatch at ({},{},{}): expected {:?}, got {:?}", i, j, k, expected.into_bigint(), got.into_bigint());
                }
            }
        }

        // 3) softmax consistency: s * denom == exp_prod, and sum s == 1
        for i in 0..rows {
            let mut exp_prods: Vec<F> = Vec::with_capacity(cols);
            for j in 0..cols {
                let mut ep = F::one();
                for k in 0..k_segments {
                    ep *= exp_limb_vals[0][0][i][j][k];
                }
                exp_prods.push(ep);
            }
            let mut denom = F::zero();
            for ep in exp_prods.iter() { denom += *ep; }

            let mut sum_s = F::zero();
            for j in 0..cols {
                let s_field = softmax_outputs[0][0][i][j];
                let lhs = s_field * denom;
                let rhs = exp_prods[j];
                assert_eq!(lhs, rhs, "Softmax equality failed at row {}, col {}: s*denom != exp_prod (lhs={:?}, rhs={:?})", i, j, lhs.into_bigint(), rhs.into_bigint());
                sum_s += s_field;
            }
            assert_eq!(sum_s, F::one(), "Softmax sum != 1 at row {}: sum={:?}", i, sum_s.into_bigint());
        }
    }

    #[test]
    fn test_softmax_lookup_runner() {
        let npz_path = "tmp_tables/embedded_softmax_for_lookup.npz";
        let (tables, _frac_bits, limb_bits) =
        load_tables::<F, _>(npz_path).expect("load tables");

        // Sanity
        let k_segments = tables.len();
        assert!(k_segments > 0, "tables has zero segments");
        let table_width = tables[0].len();
        assert!(table_width > 0, "table row length is zero");

        // tiny matmul ints (rows x cols)
        let matmul_ints = vec![vec![10i128, 2i128, 5i128]];

        // Build witnesses (uses limb_bits read from npz)
        let (matmul_outputs, limbs, exp_limb_vals, softmax_outputs) =
            build_softmax_witnesses::<F>(matmul_ints.clone(), &tables, limb_bits);

        verify_softmax_witnesses(
            &matmul_ints,
            &tables,
            limb_bits,
            &matmul_outputs,
            &limbs,
            &exp_limb_vals,
            &softmax_outputs,
        );

        // Prepare circuit mock for setup (must match types and sizes)
        let rows = matmul_outputs[0][0].len();
        let cols = matmul_outputs[0][0][0].len();

        let frac : u32 = _frac_bits;
        assert_eq!(k_segments, K_SEGMENTS, "This test assumes k_segments == seg");

        let circuit_mock = SoftmaxCircuit::<F> {
            matmul_outputs: vec![vec![vec![vec![F::zero(); cols]; rows]; 1]; 1],
            limbs: vec![vec![vec![vec![[F::zero(); K_SEGMENTS]; cols]; rows]; 1]; 1],
            exp_limb_vals: vec![vec![vec![vec![[F::one(); K_SEGMENTS]; cols]; rows]; 1]; 1],
            softmax_outputs: vec![vec![vec![vec![F::zero(); cols]; rows]; 1]; 1],
            frac_bits: frac,
        };

        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
        let (ek, vk) = CcGroth16::<Bn254>::circuit_specific_setup(circuit_mock, &mut rng).unwrap();
        let pvk = prepare_verifying_key::<Bn254>(&vk);

        // Convert dynamic limbs/exp_limb_vals to fixed [F;K_SEGMENTS] arrays expected by the circuit
        let mut limbs_fixed: Vec<Vec<Vec<Vec<[F; K_SEGMENTS]>>>> =
            vec![vec![vec![vec![[F::zero(); K_SEGMENTS]; cols]; rows]; 1]; 1];
        let mut exp_fixed: Vec<Vec<Vec<Vec<[F; K_SEGMENTS]>>>> =
            vec![vec![vec![vec![[F::one(); K_SEGMENTS]; cols]; rows]; 1]; 1];

        for b in 0..1 {
            for h in 0..1 {
                for i in 0..rows {
                    for j in 0..cols {
                        let dynamic_limbs: &Vec<F> = &limbs[b][h][i][j];
                        let arr_limbs: [F; K_SEGMENTS] = dynamic_limbs.clone().try_into()
                            .expect("k_segments must be K_SEGMENTS (cannot convert Vec<F> to [F;K_SEGMENTS])");
                        limbs_fixed[b][h][i][j] = arr_limbs;

                        let dynamic_exps: &Vec<F> = &exp_limb_vals[b][h][i][j];
                        let arr_exps: [F; K_SEGMENTS] = dynamic_exps.clone().try_into()
                            .expect("k_segments must be K_SEGMENTS (cannot convert Vec<F> to [F;K_SEGMENTS])");
                        exp_fixed[b][h][i][j] = arr_exps;
                    }
                }
            }
        }

        // Build circuit instance
        let circuit_instance = SoftmaxCircuit::<F> {
            matmul_outputs: matmul_outputs.clone(),
            limbs: limbs_fixed.clone(),
            exp_limb_vals: exp_fixed.clone(),
            softmax_outputs: softmax_outputs.clone(),
            frac_bits: frac,
        };

        // Before calling CcGroth16::prove, do an explicit cs.is_satisfied() check to get early diagnostic.
        {
            use ark_relations::r1cs::ConstraintSystem;
            // create a fresh in-memory CS and generate constraints
            let cs = ConstraintSystem::<F>::new_ref();
            // generate constraints (consumes a circuit) — clone circuit_instance because method takes self
            circuit_instance.clone().generate_constraints(cs.clone()).expect("generate_constraints failed");
            let satisfied = cs.is_satisfied().unwrap_or(false);
            // print simple diagnostics
            println!("CS satisfied?: {}", satisfied);
            println!("num constraints: {}", cs.num_constraints());
            println!("num public inputs: {}", cs.num_instance_variables());
            println!("num witness: {}", cs.num_witness_variables());
            if !satisfied {
                // If not satisfied, we still abort early with notes — avoid CcGroth16 assert panic to give clearer message.
                panic!("Constraint system not satisfied on provided witnesses. See above diagnostics.");
            }
        }

        // If CS satisfied, continue with proving
        let mut rng2 = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());
        let proof = CcGroth16::<Bn254>::prove(&ek, circuit_instance, &mut rng2).unwrap();

        // Verify
        assert!(CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[], &proof).unwrap());
    }
}