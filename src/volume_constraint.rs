use constraint::*;
use geo::math::{Matrix3, Vector3};
use matrix::*;
use rayon::prelude::*;
use reinterpret::*;
use std::collections::BTreeSet;
use TetMesh;

// TODO: move to geo::mesh
#[derive(Copy, Clone, Eq)]
struct TriFace {
    pub tri: [usize; 3],
}

impl TriFace {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 2, 1],
        [2, 1, 0],
        [1, 0, 2],
    ];
}

/// A utility function to index a slice using three indices, creating a new array of 3
/// corresponding entries of the slice.
fn tri_at<T: Copy>(slice: &[T], tri: &[usize; 3]) -> [T; 3] {
    [slice[tri[0]], slice[tri[1]], slice[tri[2]]]
}

/// Consider any permutation of the triangle to be equivalent to the original.
impl PartialEq for TriFace {
    fn eq(&self, other: &TriFace) -> bool {
        for p in Self::PERMUTATIONS.iter() {
            if tri_at(&other.tri, p) == self.tri {
                return true;
            }
        }

        false
    }
}

impl PartialOrd for TriFace {
    fn partial_cmp(&self, other: &TriFace) -> Option<::std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Lexicographic ordering of the sorted indices.
impl Ord for TriFace {
    fn cmp(&self, other: &TriFace) -> ::std::cmp::Ordering {
        let mut tri = self.tri;
        tri.sort();
        let mut other_tri = other.tri;
        other_tri.sort();
        tri.cmp(&other_tri)
    }
}

/// Extract the surface of the temesh.
/// The algorithm is to iterate over every tet face and upon seeing a duplicate, remove it
/// from the list. this will leave only unique faces, which corresponds to the surface of
/// the tetmesh.
/// This function assumes that the given tetmesh is a manifold.
fn extract_surface_topo(tetmesh: &TetMesh) -> Vec<[usize; 3]> {
    let mut triangles: BTreeSet<TriFace> = BTreeSet::new();

    let tet_faces = [[0, 3, 1], [3, 2, 1], [1, 2, 0], [2, 3, 0]];

    for cell in tetmesh.cell_iter() {
        for tet_face in tet_faces.iter() {
            let indices: [usize; 4] = (*cell).clone().into();
            let face = TriFace {
                tri: tri_at(&indices, tet_face),
            };

            if !triangles.remove(&face) {
                triangles.insert(face);
            }
        }
    }

    let mut surface_topo = Vec::with_capacity(triangles.len());
    for elem in triangles.into_iter() {
        surface_topo.push(elem.tri);
    }

    surface_topo
}

/// Produce a mapping from a virtual surface vertex to its index in the tetmesh.
/// Also count the number of vertices adjacent to the given topology of triangles.
/// Identically the count is the number of valid entries in the returned vector.
fn surface_vertex_maps(topo: &[[usize; 3]]) -> (Vec<isize>, Vec<usize>) {
    if topo.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let indices: &[usize] = reinterpret_slice(topo);
    let max_vert_idx = indices.iter().max().unwrap();

    let mut surf_vtx_map = vec![-1isize; max_vert_idx + 1];
    let mut tet_vtx_map = Vec::new();
    for &i in indices.iter() {
        if surf_vtx_map[i] == -1 {
            surf_vtx_map[i] = tet_vtx_map.len() as isize;
            tet_vtx_map.push(i);
        }
    }

    (surf_vtx_map, tet_vtx_map)
}

#[derive(Debug, PartialEq)]
pub struct VolumeConstraint {
    /// The topology of the surface of a tetrahedral mesh. This is a vector of triplets of indices
    /// of tetmesh vertices. Each triplet corresponds to a triangle on the surface of the tetmesh.
    pub surface_topo: Vec<[usize; 3]>,

    surf_to_tet_vtx_map: Vec<usize>,
    tet_to_surf_vtx_map: Vec<isize>,

    /// Storage for the constraint value.
    constraint_value: [f64; 1],

    /// Constraint jacobian triplet storage.
    constraint_jac_triplets: Vec<MatrixElementTriplet<f64>>,

    /// Constraint hessian triplet storage.
    constraint_hess_triplets: Vec<MatrixElementTriplet<f64>>,

    /// Constraint hessian indices storage.
    constraint_hess_indices: Vec<MatrixElementIndex>,

    /// Constraint hessian values storage.
    constraint_hess_values: Vec<f64>,
}

impl VolumeConstraint {
    pub fn new(tetmesh: &TetMesh) -> Self {
        let surface_topo = extract_surface_topo(tetmesh);
        let (tet_to_surf_vtx_map, surf_to_tet_vtx_map) = surface_vertex_maps(&surface_topo);
        VolumeConstraint {
            surface_topo,
            surf_to_tet_vtx_map,
            tet_to_surf_vtx_map,
            constraint_value: [0.0],
            constraint_jac_triplets: Vec::new(),
            constraint_hess_triplets: Vec::new(),
            constraint_hess_indices: Vec::new(),
            constraint_hess_values: Vec::new(),
        }
    }
}

impl Constraint<f64> for VolumeConstraint {
    fn constraint_size(&self) -> usize {
        1
    }

    fn constraint_lower_bound(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn constraint_upper_bound(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn constraint(&mut self, x: &[f64]) -> &[f64] {
        let pos: &[Vector3<f64>] = reinterpret_slice(x);
        let mut total_volume = 0.0;
        for tri in self.surface_topo.iter() {
            let p = tri_at(pos, tri);
            let signed_volume = p[0].dot(p[1].cross(p[2]));
            total_volume += signed_volume;
        }
        self.constraint_value[0] = total_volume;
        &self.constraint_value
    }
}

impl ConstraintJacobianSize for VolumeConstraint {
    fn constraint_jacobian_size(&self) -> usize {
        self.surf_to_tet_vtx_map.len() * 3
    }
}

impl ConstraintJacobian<f64> for VolumeConstraint {
    fn constraint_jacobian(&mut self, x: &[f64]) -> &[MatrixElementTriplet<f64>] {
        let pos: &[Vector3<f64>] = reinterpret_slice(x);

        // Reserve memory for all the triplets
        self.constraint_jac_triplets.clear();
        let num_triplets = self.constraint_jacobian_size();
        self.constraint_jac_triplets.reserve(num_triplets);

        // Initialize the triplets with zeros.
        for idx in self.surf_to_tet_vtx_map.iter() {
            for j in 0..3 {
                self.constraint_jac_triplets
                    .push(MatrixElementTriplet::new(0, 3 * idx + j, 0.0));
            }
        }

        for tri in self.surface_topo.iter() {
            let p = tri_at(pos, tri);
            let c = [p[1].cross(p[2]), p[2].cross(p[0]), p[0].cross(p[1])];

            for vi in 0..3 {
                let v = self.tet_to_surf_vtx_map[tri[vi]];
                assert!(v >= 0);
                for j in 0..3 {
                    self.constraint_jac_triplets[3 * v as usize + j].val += c[vi][j];
                }
            }
        }
        &self.constraint_jac_triplets
    }
}

// TODO: move to geo::math
/// Convert a 3 dimensional vector `v = (x,y,z)` into a skew matrix `[v]×` given by
/// ```verbatim
/// ⎡ 0 -z  y⎤
/// ⎢ z  0 -x⎥
/// ⎣-y  x  0⎦
/// ```
fn skew(x: Vector3<f64>) -> Matrix3<f64> {
    Matrix3([[0.0, x[2], -x[1]], [-x[2], 0.0, x[0]], [x[1], -x[0], 0.0]])
}

impl ConstraintHessianSize for VolumeConstraint {
    fn constraint_hessian_size(&self) -> usize {
        6 * 3 * self.surface_topo.len()
    }
}

impl ConstraintHessianIndicesValues<f64> for VolumeConstraint {
    fn constraint_hessian_indices(&mut self) -> &[MatrixElementIndex] {
        // Reserve memory for all the indices
        self.constraint_hess_indices.clear();
        let num_indices = self.constraint_hessian_size();
        self.constraint_hess_indices.reserve(num_indices);

        for tri in self.surface_topo.iter() {
            for vi in 0..3 {
                let col_v = tri[vi];
                for off in 1..=2 {
                    let vj = (vi + off) % 3;
                    let row_v = tri[vj];
                    if row_v > col_v {
                        for c in 0..3 {
                            for r in 0..3 {
                                if r == c {
                                    continue;
                                }
                                self.constraint_hess_indices.push(MatrixElementIndex {
                                    row: 3 * row_v + r,
                                    col: 3 * col_v + c,
                                });
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            self.constraint_hess_indices.len(),
            self.constraint_hessian_size()
        );
        &self.constraint_hess_indices
    }
    fn constraint_hessian_values(&mut self, x: &[f64], lambda: &[f64]) -> &[f64] {
        let pos: &[Vector3<f64>] = reinterpret_slice(x);

        // Reserve memory for all the values
        self.constraint_hess_values.clear();
        let num_values = self.constraint_hessian_size();
        self.constraint_hess_values.reserve(num_values);

        for tri in self.surface_topo.iter() {
            let p = tri_at(pos, tri);
            let local_hess = [skew(p[0]), skew(p[1]), skew(p[2])];

            for vi in 0..3 {
                let col_v = tri[vi];
                for off in 1..=2 {
                    let vj = (vi + off) % 3;
                    let vjn = (vi + off + if off == 1 { 1 } else { 2 }) % 3;
                    let row_v = tri[vj];
                    if row_v > col_v {
                        for c in 0..3 {
                            for r in 0..3 {
                                if r == c {
                                    continue;
                                }
                                self.constraint_hess_values.push(
                                    if off == 1 { 1.0 } else { -1.0 }
                                        * lambda[0]
                                        * local_hess[vjn][c][r],
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            self.constraint_hess_values.len(),
            self.constraint_hessian_size()
        );
        &self.constraint_hess_values
    }
}

impl ConstraintHessian<f64> for VolumeConstraint {
    fn constraint_hessian(&mut self, x: &[f64], lambda: &[f64]) -> &[MatrixElementTriplet<f64>] {
        let pos: &[Vector3<f64>] = reinterpret_slice(x);

        // Reserve memory for all the triplets
        self.constraint_hess_triplets.clear();
        let num_triplets = self.constraint_hessian_size();
        self.constraint_hess_triplets.reserve(num_triplets);

        for tri in self.surface_topo.iter() {
            let p = tri_at(pos, tri);
            let local_hess = [skew(p[0]), skew(p[1]), skew(p[2])];

            for vi in 0..3 {
                let col_v = tri[vi];
                for off in 1..=2 {
                    let vj = (vi + off) % 3;
                    let vjn = (vi + off + if off == 1 { 1 } else { 2 }) % 3;
                    let row_v = tri[vj];
                    if row_v > col_v {
                        for c in 0..3 {
                            for r in 0..3 {
                                if r == c {
                                    continue;
                                }
                                self.constraint_hess_triplets
                                    .push(MatrixElementTriplet::new(
                                        3 * row_v + r,
                                        3 * col_v + c,
                                        if off == 1 { 1.0 } else { -1.0 }
                                            * lambda[0]
                                            * local_hess[vjn][c][r],
                                    ));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            self.constraint_hess_triplets.len(),
            self.constraint_hessian_size()
        );
        &self.constraint_hess_triplets
    }
}
