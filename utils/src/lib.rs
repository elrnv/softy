pub mod aref;
pub mod index;
pub mod soap;
pub mod transform;
pub mod zip;

/**
 * This crate provides convenience functions for building common meshes.
 */
pub use crate::transform::*;
pub use crate::zip::*;
use geo::math::Vector3;
use geo::mesh::{PolyMesh, TetMesh, TriMesh};

/// Parameters that define a grid that lies in one of the 3 axis planes in 3D space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Grid {
    /// Number of grid cells in each column.
    pub rows: usize,
    /// Number of grid cells in each row.
    pub cols: usize,
    /// Axis orientation of the grid.
    pub orientation: AxisPlaneOrientation,
}

/// Axis plane orientation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AxisPlaneOrientation {
    XY,
    YZ,
    ZX,
}

/// Generate a [-1,1]x[-1,1] mesh grid with the given cell resolution and grid orientation. The
/// grid nodes are spcified in row major order.
pub fn make_grid(grid_params: Grid) -> PolyMesh<f64> {
    let Grid {
        rows,
        cols,
        orientation,
    } = grid_params;

    let mut positions = Vec::new();

    // iterate over vertices
    for j in 0..=cols {
        for i in 0..=rows {
            let r = -1.0 + 2.0 * (i as f64) / rows as f64;
            let c = -1.0 + 2.0 * (j as f64) / cols as f64;
            let node_pos = match orientation {
                AxisPlaneOrientation::XY => [r, c, 0.0],
                AxisPlaneOrientation::YZ => [0.0, r, c],
                AxisPlaneOrientation::ZX => [c, 0.0, r],
            };
            positions.push(node_pos);
        }
    }

    let mut indices = Vec::new();

    // iterate over faces
    for i in 0..rows {
        for j in 0..cols {
            indices.push(4);
            indices.push((rows + 1) * j + i);
            indices.push((rows + 1) * j + i + 1);
            indices.push((rows + 1) * (j + 1) + i + 1);
            indices.push((rows + 1) * (j + 1) + i);
        }
    }

    PolyMesh::new(positions, &indices)
}

/// Generate a [-1,1]x[-1,1]x[-1,1] tetmesh box with the given cell resolution per axis.
/// The tetrahedralization is a simple 6 tets per cube with a regular pattern.
pub fn make_box(res: [usize; 3]) -> TetMesh<f64> {
    let mut positions = Vec::new();
    let [nx, ny, nz] = res;

    // iterate over vertices
    for ix in 0..=nx {
        for iy in 0..=ny {
            for iz in 0..=nz {
                let x = -1.0 + 2.0 * (ix as f64) / nx as f64;
                let y = -1.0 + 2.0 * (iy as f64) / ny as f64;
                let z = -1.0 + 2.0 * (iz as f64) / nz as f64;
                positions.push([x, y, z]);
            }
        }
    }

    let mut indices = Vec::new();

    // iterate over faces
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let index = |x, y, z| ((ix + x) * (ny + 1) + (iy + y)) * (nz + 1) + (iz + z);
                // Populate tets in a star pattern
                let first = index(0, 0, 0);
                let second = index(1, 1, 1);
                // Tet 1
                indices.push(first);
                indices.push(second);
                indices.push(index(0, 1, 1));
                indices.push(index(0, 1, 0));
                // Tet 2
                indices.push(first);
                indices.push(second);
                indices.push(index(0, 1, 0));
                indices.push(index(1, 1, 0));
                // Tet 3
                indices.push(first);
                indices.push(second);
                indices.push(index(1, 1, 0));
                indices.push(index(1, 0, 0));
                // Tet 4
                indices.push(first);
                indices.push(second);
                indices.push(index(1, 0, 0));
                indices.push(index(1, 0, 1));
                // Tet 5
                indices.push(first);
                indices.push(second);
                indices.push(index(1, 0, 1));
                indices.push(index(0, 0, 1));
                // Tet 6
                indices.push(first);
                indices.push(second);
                indices.push(index(0, 0, 1));
                indices.push(index(0, 1, 1));
            }
        }
    }

    TetMesh::new(positions, indices)
}

pub fn make_sample_octahedron() -> TriMesh<f64> {
    let vertices = vec![
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, -0.5],
        [0.0, 0.0, 0.5],
    ];

    let indices = vec![
        0, 5, 3, 4, 0, 3, 1, 4, 3, 5, 1, 3, 5, 0, 2, 0, 4, 2, 4, 1, 2, 1, 5, 2,
    ];

    TriMesh::new(vertices, indices)
}

pub fn make_regular_tet() -> TetMesh<f64> {
    let sqrt_8_by_9 = f64::sqrt(8.0 / 9.0);
    let sqrt_2_by_9 = f64::sqrt(2.0 / 9.0);
    let sqrt_2_by_3 = f64::sqrt(2.0 / 3.0);
    let vertices = vec![
        [0.0, 1.0, 0.0],
        [-sqrt_8_by_9, -1.0 / 3.0, 0.0],
        [sqrt_2_by_9, -1.0 / 3.0, sqrt_2_by_3],
        [sqrt_2_by_9, -1.0 / 3.0, -sqrt_2_by_3],
    ];

    let indices = vec![3, 1, 0, 2];

    TetMesh::new(vertices, indices)
}

pub fn make_regular_icosahedron() -> TriMesh<f64> {
    let sqrt5 = 5.0_f64.sqrt();
    let a = 1.0 / sqrt5;
    let w1 = 0.25 * (sqrt5 - 1.0);
    let h1 = (0.125 * (5.0 + sqrt5)).sqrt();
    let w2 = 0.25 * (sqrt5 + 1.0);
    let h2 = (0.125 * (5.0 - sqrt5)).sqrt();
    let vertices = vec![
        // North pole
        [0.0, 0.0, 1.0],
        // Alternating ring
        [0.0, 2.0 * a, a],
        [2.0 * a * h2, 2.0 * a * w2, -a],
        [2.0 * a * h1, 2.0 * a * w1, a],
        [2.0 * a * h1, -2.0 * a * w1, -a],
        [2.0 * a * h2, -2.0 * a * w2, a],
        [0.0, -2.0 * a, -a],
        [-2.0 * a * h2, -2.0 * a * w2, a],
        [-2.0 * a * h1, -2.0 * a * w1, -a],
        [-2.0 * a * h1, 2.0 * a * w1, a],
        [-2.0 * a * h2, 2.0 * a * w2, -a],
        // South pole
        [0.0, 0.0, -1.0],
    ];

    #[rustfmt::skip]
    let indices = vec![
        // North triangles
        0, 1, 3,
        0, 3, 5,
        0, 5, 7,
        0, 7, 9,
        0, 9, 1,
        // Equatorial triangles
        1, 2, 3,
        2, 4, 3,
        3, 4, 5,
        4, 6, 5,
        5, 6, 7,
        6, 8, 7,
        7, 8, 9,
        8, 10, 9,
        9, 10, 1,
        10, 2, 1,
        // South triangles
        11, 2, 10,
        11, 4, 2,
        11, 6, 4,
        11, 8, 6,
        11, 10, 8,
    ];

    TriMesh::new(vertices, indices)
}

// TODO: Complete sphere mesh
///// Create a sphere with a given level of subdivision, where `level=1` produces a regular
///// icosahedron.
//pub fn make_sphere(level: usize) -> TriMesh<f64> {
//
//
//}

/// Generate a random vector of `Vector3`s.
pub fn random_vectors(n: usize) -> Vec<Vector3<f64>> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n)
        .map(move |_| Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the icosahedron has unit radius.
    #[test]
    fn icosahedron_unity_test() {
        use approx::assert_relative_eq;
        use geo::mesh::VertexPositions;

        let icosa = make_regular_icosahedron();
        for &v in icosa.vertex_positions() {
            assert_relative_eq!(geo::math::Vector3(v).norm(), 1.0);
        }
    }
}
