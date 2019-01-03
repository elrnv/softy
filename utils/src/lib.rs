/**
 * This crate provides convenience functions for building common meshes.
 */

pub mod transform;

pub use crate::transform::*;
use geometry::{
    mesh::{
        PolyMesh,
        TriMesh,
        TetMesh,
    }
};

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
    for i in 0..rows + 1 {
        for j in 0..cols + 1 {
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
    let sqrt_8_by_9 = f64::sqrt(8.0/9.0);
    let sqrt_2_by_9 = f64::sqrt(2.0/9.0);
    let sqrt_2_by_3 = f64::sqrt(2.0/3.0);
    let vertices = vec![
        [0.0, 1.0, 0.0],
        [-sqrt_8_by_9, -1.0/3.0, 0.0],
        [sqrt_2_by_9, -1.0/3.0, sqrt_2_by_3],
        [sqrt_2_by_9, -1.0/3.0, -sqrt_2_by_3],
    ];

    let indices = vec![1, 2, 3, 0];

    TetMesh::new(vertices, indices)
}
