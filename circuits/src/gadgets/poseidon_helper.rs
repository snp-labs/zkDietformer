use ark_crypto_primitives::sponge::poseidon::{PoseidonConfig, PoseidonSponge, find_poseidon_ark_and_mds};
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;

/// Create a default Poseidon config for BN254-sized fields (254-bit).
/// Parameters: full_rounds=8, partial_rounds=57, alpha=5, rate=2, capacity=1.
pub fn default_poseidon_config<F: PrimeField>() -> PoseidonConfig<F> {
    let (ark, mds) = find_poseidon_ark_and_mds::<F>(254, 2, 8, 57, 0);
    PoseidonConfig::new(8, 57, 5, mds, ark, 2, 1)
}

/// Compute Poseidon hash of a slice of field elements, returning a single field element.
pub fn poseidon_hash<F: PrimeField + Absorb>(config: &PoseidonConfig<F>, inputs: &[F]) -> F {
    let mut sponge = PoseidonSponge::new(config);
    sponge.absorb(&inputs.to_vec());
    sponge.squeeze_field_elements::<F>(1)[0]
}
