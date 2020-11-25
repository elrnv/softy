pub mod aref;
//pub mod soap;

pub use geo::index;
pub use geo::index::CheckedIndex;

/**
 * This crate provides various convenience functions and utilities.
 */

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
pub fn build_mesh_laplacian(mesh: &geo::mesh::TriMesh<f64>, weight: f64) -> tensr::DSMatrix {
    use geo::mesh::topology::*;
    use tensr::*;
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
    let mut lap_data = lap
        .into_data()
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
    use rand::distributions::Uniform;
    use rand::prelude::*;
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
    f32::from_bits(f.to_bits() & 0xff80_0000)
}

/// Given an iterator over integers, compute the mode and return it along with its
/// frequency.
/// If the iterator is empty just return 0.
pub fn mode_u32<I: IntoIterator<Item = u32>>(data: I) -> (u32, usize) {
    let data_iter = data.into_iter();
    let mut bins = Vec::with_capacity(100);
    for x in data_iter {
        let i = x as usize;
        if i >= bins.len() {
            bins.resize(i + 1, 0usize);
        }
        bins[i] += 1;
    }
    bins.iter()
        .cloned()
        .enumerate()
        .max_by_key(|&(_, f)| f)
        .map(|(m, f)| (m as u32, f))
        .unwrap_or((0u32, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_u32_test() {
        let v = vec![1u32, 1, 1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1];
        assert_eq!(mode_u32(v), (1, 6));
        let v = vec![];
        assert_eq!(mode_u32(v), (0, 0));
        let v = vec![0u32, 0, 0, 1, 1, 1, 1, 2, 2, 2];
        assert_eq!(mode_u32(v), (1, 4));
    }

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
