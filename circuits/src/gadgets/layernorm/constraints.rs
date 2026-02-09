use ark_ff::PrimeField;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::{FieldVar, fp::FpVar};
use ark_r1cs_std::alloc::AllocVar;
use ark_relations::r1cs::SynthesisError;
use ark_relations::r1cs::ConstraintSystemRef;