use ark_ff::PrimeField;
use ark_std::log2;
use ndarray::{Array2, Ix2};
use ndarray_npy::NpzReader;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use super::SubtableStrategy;

pub enum GaussianSubtableStrategy {}

static GAUSSIAN_CACHE: OnceLock<Array2<u64>> = OnceLock::new();
static GAUSSIAN_TABLE_PATH: OnceLock<String> = OnceLock::new();

/// Set the directory containing `gaussian_tables.npz`.
/// OnceLock: first call wins, subsequent calls are silently ignored.
pub fn set_gaussian_table_dir(dir: impl AsRef<Path>) {
    let path: PathBuf = dir.as_ref().join("gaussian_tables.npz");
    let _ = GAUSSIAN_TABLE_PATH.set(path.to_string_lossy().into_owned());
}

fn get_gaussian_table_path() -> String {
    if let Some(p) = GAUSSIAN_TABLE_PATH.get() {
        return p.clone();
    }
    if let Ok(p) = env::var("GAUSSIAN_TABLE_PATH") {
        return p;
    }
    "outputs/gaussian_tables.npz".to_string()
}

fn get_or_init_gaussian_cache(c: usize, m: usize) -> &'static Array2<u64> {
    GAUSSIAN_CACHE.get_or_init(|| {
        let path_str = get_gaussian_table_path();
        let file = File::open(&path_str).unwrap_or_else(|e| {
            panic!(
                "Gaussian table not found at '{}': {}.\n\
                 Set via `set_gaussian_table_dir()` or env var `GAUSSIAN_TABLE_PATH`.",
                path_str, e,
            )
        });
        let mut npz = NpzReader::new(file).expect("failed to create NpzReader");
        let arr = npz
            .by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("tables.npy")
            .or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("tables"))
            .expect("npz does not contain 'tables'");
        let arr2d = arr.into_dimensionality::<Ix2>().expect("must be 2D");
        assert_eq!(arr2d.shape()[0], c, "Gaussian table C mismatch");
        assert_eq!(arr2d.shape()[1], m, "Gaussian table M mismatch");
        arr2d
    })
}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
    for GaussianSubtableStrategy
{
    const NUM_SUBTABLES: usize = C;
    const NUM_MEMORIES: usize = C;

    fn memory_to_subtable_index(memory_index: usize) -> usize {
        debug_assert!(memory_index < C);
        memory_index
    }

    fn memory_to_dimension_index(memory_index: usize) -> usize {
        debug_assert!(memory_index < C);
        memory_index
    }

    fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
        let t = get_or_init_gaussian_cache(C, M);
        std::array::from_fn(|i| t.row(i).iter().map(|&v| F::from(v)).collect())
    }

    fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F {
        let t = get_or_init_gaussian_cache(C, M);
        let b = point.len();
        let mut evals = vec![F::one(); 1 << b];
        let mut size = 1;
        for j in 0..b {
            let r = point[b - 1 - j];
            for i in 0..size {
                evals[i + size] = evals[i] * r;
                evals[i] = evals[i] * (F::one() - r);
            }
            size *= 2;
        }
        let mut acc = F::zero();
        for (idx, &basis) in evals.iter().enumerate() {
            acc += F::from(t[[subtable_index, idx]]) * basis;
        }
        acc
    }

    fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
        vals.iter().product()
    }

    fn g_poly_degree() -> usize {
        1 // C=1, identity
    }
}
