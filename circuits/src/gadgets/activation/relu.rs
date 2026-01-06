use ark_ff::PrimeField;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_relations::r1cs::SynthesisError;

use super::{ActivationFunction, ActivationGadget, constraints};

/// Type tag for ReLU
pub struct ReLU;

impl<F: PrimeField> ActivationFunction<F> for ReLU {
    const NAME: &'static str = "relu";

    fn eval_native(x: F) -> F {
        // lower-half(<= (p-1)/2) is treated as non-negative
        let half = constraints::modulus_minus_one_div_two::<F>();
        if x.into_bigint() <= half.into_bigint() {
            x
        } else {
            F::zero()
        }
    }
}

impl<F: PrimeField> ActivationGadget<ReLU, F> for ReLU {
    fn apply(x: &FpVar<F>) -> Result<FpVar<F>, SynthesisError> {
        // neg = (x > (p-1)/2)
        let is_negative = constraints::is_negative(x)?;
        let zero = FpVar::<F>::constant(F::zero());

        // relu(x) = neg ? 0 : x
        is_negative.select(&zero, x)
    }
}

#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_r1cs_std::{R1CSVar, alloc::AllocVar};
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_relu() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let pos = Fr::from(7u64);
        let neg = -Fr::from(3u64);
        let zero = Fr::from(0u64);

        let x_pos = FpVar::<Fr>::new_witness(cs.clone(), || Ok(pos)).unwrap();
        let x_neg = FpVar::<Fr>::new_witness(cs.clone(), || Ok(neg)).unwrap();
        let x_zero = FpVar::<Fr>::new_witness(cs.clone(), || Ok(zero)).unwrap();

        let y_pos = <ReLU as ActivationGadget<ReLU, Fr>>::apply(&x_pos).unwrap();
        let y_neg = <ReLU as ActivationGadget<ReLU, Fr>>::apply(&x_neg).unwrap();
        let y_zero = <ReLU as ActivationGadget<ReLU, Fr>>::apply(&x_zero).unwrap();

        println!("ReLU({}) = {}", pos, y_pos.value().unwrap());
        println!("ReLU({}) = {}", neg, y_neg.value().unwrap());
        println!("ReLU({}) = {}", zero, y_zero.value().unwrap());

        assert_eq!(y_pos.value().unwrap(), ReLU::eval_native(pos));
        assert_eq!(y_neg.value().unwrap(), ReLU::eval_native(neg));
        assert_eq!(y_zero.value().unwrap(), ReLU::eval_native(zero));

        println!("number of constraints: {}", cs.num_constraints());
        assert!(cs.is_satisfied().unwrap());
    }
}
