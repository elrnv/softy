pub mod aref;
pub mod index;
pub mod soap;
pub mod zip;

/**
 * This crate provides various convenience functions and utilities.
 */
pub use crate::zip::*;

/// Generate a random vector of triplets.
pub fn random_vectors(n: usize) -> Vec<[f64; 3]> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n)
        .map(move |_| [rng.sample(range), rng.sample(range), rng.sample(range)])
        .collect()
}

/// This function effectively zeros out the entire mantissa of the floating point number given.
///
/// The returned value is essentially the approximate max (in absolute value) power of two
/// (including negative powers) below the given number. This works for all normal floats.
pub fn approx_power_of_two64(f: f64) -> f64 {
    f64::from_bits(f.to_bits() & 0xfff0_0000_0000_0000)
}

/// This function effectively zeros out the entire mantissa of the floating point number given.
///
/// The returned value is essentially the approximate max (in absolute value) power of two
/// (including negative powers) below the given number. This works for all normal floats.
pub fn approx_power_of_two32(f: f32) -> f32 {
    f32::from_bits(f.to_bits() & 0xff800000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_power_of_two64_test() {
        assert_eq!(approx_power_of_two64(1312.421), 1024.0);
        assert_eq!(approx_power_of_two64(0.4231), 0.25);
        assert_eq!(approx_power_of_two64(0.001), 0.0009765625);
    }

    #[test]
    fn approx_power_of_two32_test() {
        assert_eq!(approx_power_of_two32(1312.421), 1024.0);
        assert_eq!(approx_power_of_two32(0.4231), 0.25);
        assert_eq!(approx_power_of_two32(0.001), 0.0009765625);
    }
}
