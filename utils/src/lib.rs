pub mod aref;
pub mod index;
pub mod soap;
pub mod zip;

/**
 * This crate provides various convenience functions and utilities.
 */
pub use crate::zip::*;

#[macro_export]
macro_rules! iter_vals {
    // Base case
    ($first:expr) => {
        std::iter::once($first)
    };
    // Recursive case
    ($first:expr $(,$vals:expr)*) => {
        std::iter::once($first).chain(
            iter_vals!($($vals,)*)
        )
    };
    // Accept trailing commas
    ($first:expr $(,$vals:expr)*,) => {
        iter_vals!($first $(,$vals)*)
    };
}

/// Build a matrix that smoothes values at mesh vertices with their neighbours by the given
/// weight. For `weight = 0.0`, no smoothing is performed, and this matrix is the identity.
pub fn build_mesh_laplacian(mesh: &geo::mesh::TriMesh<f64>, weight: f64) -> soap::DSMatrix {
    use geo::mesh::topology::*;
    use soap::*;
    let size = mesh.num_vertices();
    let iter = mesh
        .face_iter()
        .flat_map(|&[v0, v1, v2]| {
            iter_vals!(
                (v0, v1, weight),
                (v0, v2, weight),
                (v1, v0, weight),
                (v1, v2, weight),
                (v2, v0, weight),
                (v2, v1, weight),
            )
        })
        .chain((0..size).map(|i| (i, i, (1.0 - weight))));
    let lap = DSMatrix::from_triplets_iter_uncompressed(iter, size, size);

    // Compress and prune
    // We multiply the contributions by 0.5 only when there are overlaps. From the triangle
    // mesh topology we know that there will be at most two overlaps, and if there are none,
    // that means the edge has only one adjacent face (a boundary edge).
    let mut lap_data = lap.into_data()
        .pruned(|a, &b| *a += 0.5 * (*a + b), |_, _, &val| val != 0.0);

    // Normalize rows
    for (row_idx, mut row) in lap_data.iter_mut().enumerate() {
        let n = (row.len() - 1) as f64;
        for (col_idx, val) in row.indexed_source_iter_mut() {
            if row_idx != col_idx {
                *val /= n;
            }
        }
    }

    lap_data.into_tensor()
}

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
