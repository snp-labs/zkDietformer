use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
use merlin::Transcript;

use crate::lasso::{
  lasso::{
    densified::DensifiedRepresentation,
    surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
  },
  subtables::{
    SubtableStrategy, and::AndSubtableStrategy, gelu::GeLUSubtableStrategy, layernorm::RMSNormSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy, softmax::SoftmaxSubtableStrategy
  },
  utils::{math::Math, random::RandomTape},
};

macro_rules! e2e_test {
  ($test_name:ident, $Strategy:ty, $G:ty, $F:ty, $C:expr, $M:expr, $sparsity:expr) => {
    #[test]
    fn $test_name() {
      use crate::lasso::utils::test::{gen_indices, gen_random_point};
      use ark_std::log2;

      const C: usize = $C;
      const M: usize = $M;

      // parameters
      const NUM_MEMORIES: usize = <$Strategy as SubtableStrategy<$F, C, M>>::NUM_MEMORIES;
      let log_M: usize = M.log_2();
      let log_s: usize = log2($sparsity) as usize;

      // generate sparse polynomial
      let nz: Vec<[usize; C]> = gen_indices($sparsity, M);

      let lookup_matrix = SparseLookupMatrix::new(nz, log_M);

      let mut dense: DensifiedRepresentation<$F, C> = DensifiedRepresentation::from(&lookup_matrix);
      let gens =
        SparsePolyCommitmentGens::<$G>::new(b"gens_sparse_poly", C, $sparsity, NUM_MEMORIES, log_M);
      let commitment = dense.commit::<$G>(&gens);

      let r: Vec<$F> = gen_random_point(log_s);

      let mut random_tape = RandomTape::new(b"proof");
      let mut prover_transcript = Transcript::new(b"example");
      let proof = SparsePolynomialEvaluationProof::<$G, C, $M, $Strategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
      );

      let mut verifier_transcript = Transcript::new(b"example");
      assert!(
        proof
          .verify(&commitment, &r, &gens, &mut verifier_transcript)
          .is_ok(),
        "Failed to verify proof."
      );
    }
  };
}

e2e_test!(
  prove_4d_lt,
  LTSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 16
);
e2e_test!(
  prove_4d_lt_big_s,
  LTSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 128
);
e2e_test!(
  prove_4d_and,
  AndSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 16
);
e2e_test!(
  prove_3d_range,
  RangeCheckSubtableStrategy::<40>,
  G1Projective,
  Fr,
  /* C= */ 3,
  /* M= */ 256,
  /* sparsity= */ 16
);
e2e_test!(
  prove_2d_softmax,
  SoftmaxSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 2,
  /* M= */ 256, // 1 << 8 (SEG_LIMB_BITS)
  /* sparsity= */ 16
);
e2e_test!(
  prove_1d_rmsnorm,
  RMSNormSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 1,
  /* M= */ 65536, // 1 << 16 (TABLE_SIZE)
  /* sparsity= */ 16
);
e2e_test!(
  prove_1d_gelu,
  GeLUSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 1,
  /* M= */ 65536, // 1 << 16 (TABLE_SIZE)
  /* sparsity= */ 16
);

/// 26-bit range check: honest witness with high limb values in [0, 1023]
#[test]
fn range26_honest_high_limb() {
    use crate::lasso::utils::test::gen_random_point;
    use ark_std::log2;

    const C: usize = 2;
    const M: usize = 1 << 16;
    type Strategy = RangeCheckSubtableStrategy<26>;

    // Build indices: 16-bit low + 10-bit high, including values > 63
    let test_values: Vec<u32> = vec![
        0,                         // min
        (1 << 16) - 1,             // max low limb only
        64 << 16,                  // high limb = 64 (was out-of-range with <22>)
        100 << 16 | 0xFFFF,        // high=100, low=65535
        500 << 16 | 12345,         // high=500
        1023 << 16 | 0xFFFF,       // max valid: high=1023, low=65535
        (1 << 26) - 1,             // max 26-bit value
        42 << 16 | 7,              // high=42 (< 64, always valid)
    ];

    let nz: Vec<[usize; C]> = test_values.iter().map(|&v| {
        [(v & 0xFFFF) as usize, ((v >> 16) & 0x3FF) as usize]
    }).collect();
    let sparsity = nz.len();

    let log_m = log2(M) as usize;
    let lookup_matrix = SparseLookupMatrix::new(nz, log_m);
    let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);

    let gens = SparsePolyCommitmentGens::<G1Projective>::new(
        b"range26_honest", C, sparsity, C, log_m,
    );
    let commitment = dense.commit::<G1Projective>(&gens);

    let log_s = log2(sparsity) as usize;
    let r: Vec<Fr> = gen_random_point(log_s);

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"range26");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, M, Strategy>::prove(
        &mut dense, &r, &gens, &mut prover_transcript, &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"range26");
    assert!(
        proof.verify(&commitment, &r, &gens, &mut verifier_transcript).is_ok(),
        "Honest 26-bit range check with high limb > 63 should PASS"
    );
}

/// 26-bit range check: verify subtable behavior for out-of-range values.
///
/// Lasso alone does NOT reject out-of-range indices — the subtable simply returns 0
/// for positions >= cutoff. The actual range rejection happens at the CP-Link level:
/// combine_lookups returns a WRONG value (0 instead of 1024*2^16), causing a mismatch
/// with the Groth16 commitment.
///
/// This test verifies that mechanism by checking:
/// (a) subtable[1023] returns 1023 (last valid)
/// (b) subtable[1024] returns 0 (out-of-range → clamped to 0)
/// (c) combine_lookups with out-of-range high limb produces wrong combined value
#[test]
fn range26_subtable_clamps_out_of_range() {
    const C: usize = 2;
    const M: usize = 1 << 16;
    type Strategy = RangeCheckSubtableStrategy<26>;

    let subtables = <Strategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    // subtable[0] = full [0..65535], subtable[1] = remainder (cutoff=1024), subtable[2] = zeros

    // (a) last valid: subtable_1[1023] == 1023
    assert_eq!(subtables[1][1023], Fr::from(1023u64), "subtable[1][1023] should be 1023");

    // (b) first invalid: subtable_1[1024] == 0 (clamped)
    assert_eq!(subtables[1][1024], Fr::from(0u64), "subtable[1][1024] should be 0 (out-of-range)");

    // (c) combine_lookups: valid high limb
    let valid_vals = [Fr::from(0xFFFFu64), Fr::from(1023u64)];
    let valid_combined = <Strategy as SubtableStrategy<Fr, C, M>>::combine_lookups(&valid_vals);
    let expected_valid = Fr::from(0xFFFFu64) + Fr::from(1u64 << 16) * Fr::from(1023u64);
    assert_eq!(valid_combined, expected_valid, "valid combine should reconstruct correctly");

    // (d) combine_lookups with clamped value: high limb 1024 → subtable returns 0
    let clamped_vals = [Fr::from(0xFFFFu64), Fr::from(0u64)]; // subtable[1][1024] = 0
    let clamped_combined = <Strategy as SubtableStrategy<Fr, C, M>>::combine_lookups(&clamped_vals);
    // This would equal just 0xFFFF, NOT 0xFFFF + 1024 * 2^16
    // CP-Link would catch this mismatch with the Groth16 commitment
    let wrong_value = Fr::from(0xFFFFu64); // only low limb, high limb clamped to 0
    assert_eq!(clamped_combined, wrong_value, "clamped combine should NOT match original value");

    let intended_value = Fr::from(0xFFFFu64) + Fr::from(1u64 << 16) * Fr::from(1024u64);
    assert_ne!(clamped_combined, intended_value, "clamped combine must differ from intended → CP-Link rejects");
}