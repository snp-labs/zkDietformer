pub mod constraints;
pub mod relu;

pub use constraints::*;
pub use relu::*;

use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::SynthesisError;

pub trait ActivationFunction<F: PrimeField> {
    const NAME: &'static str;

    fn eval_native(x: F) -> F;
}

pub trait ActivationGadget<AF: ActivationFunction<F>, F: PrimeField> {
    fn apply(x: &FpVar<F>) -> Result<FpVar<F>, SynthesisError>;

    fn apply_vec(xs: &[FpVar<F>]) -> Result<Vec<FpVar<F>>, SynthesisError> {
        xs.iter().map(|x| Self::apply(x)).collect()
    }

    fn apply_matrix(m: &[Vec<FpVar<F>>]) -> Result<Vec<Vec<FpVar<F>>>, SynthesisError> {
        m.iter()
            .map(|row| Self::apply_vec(row))
            .collect::<Result<Vec<_>, _>>()
    }
}
