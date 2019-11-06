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
