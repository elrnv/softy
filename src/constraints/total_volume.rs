use crate::constraint::*;
use crate::geo::math::{Matrix3, Vector3};
use crate::geo::ops::Volume;
use crate::matrix::*;
use crate::TetMesh;
use reinterpret::*;
use std::collections::BTreeSet;
use std::ops::Add;
use std::{cell::RefCell, rc::Rc};

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

fn tri_at_new_pos<T: Copy + Add>(
    pos: &[T],
    disp: &[T],
    tri: &[usize; 3],
) -> [<T as Add>::Output; 3] {
    [
        pos[tri[0]] + disp[tri[0]],
        pos[tri[1]] + disp[tri[1]],
        pos[tri[2]] + disp[tri[2]],
    ]
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

#[derive(Debug, PartialEq)]
pub struct VolumeConstraint {
    /// The topology of the surface of a tetrahedral mesh. This is a vector of triplets of indices
    /// of tetmesh vertices. Each triplet corresponds to a triangle on the surface of the tetmesh.
    pub surface_topo: Vec<[usize; 3]>,

    /// The volume of the solid at rest. The contraint is equated to this value.
    pub rest_volume: f64,
}

impl VolumeConstraint {
    pub fn new(tetmesh: &TetMesh) -> Self {
        let surface_topo = extract_surface_topo(tetmesh);
        VolumeConstraint {
            surface_topo,
            rest_volume: Self::compute_volume(tetmesh),
        }
    }

    pub fn compute_volume(tetmesh: &TetMesh) -> f64 {
        tetmesh
            .cell_iter()
            .map(|cell| crate::fem::ref_tet(tetmesh, cell).volume())
            .sum()
    }
}

impl Constraint<f64> for VolumeConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        1
    }

    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        // We don't actually need the true volume, the triple scalar product does the trick. Here
        // we scale back by 6 to equate to the real volume.
        (vec![6.0 * self.rest_volume], vec![6.0 * self.rest_volume])
    }

    fn constraint(&mut self, x: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        let disp: &[Vector3<f64>] = reinterpret_slice(x);
        let mut total_volume = 0.0;
        for tri in self.surface_topo.iter() {
            let p = tri_at(disp, tri);
            let signed_volume = p[0].dot(p[1].cross(p[2]));
            total_volume += signed_volume;
        }
        value[0] = total_volume;
    }
}

impl VolumeConstraint {
    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
    pub fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        self.surface_topo.iter().flat_map(|tri| {
            (0..3).flat_map(move |vi| {
                (0..3).map(move |j| MatrixElementIndex {
                    row: 0,
                    col: 3 * tri[vi] + j,
                })
            })
        })
    }

    /// Compute the values of the constraint Jacobian.
    pub fn constraint_jacobian_values_iter<'a>(
        &'a self,
        x: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        let disp: &[Vector3<f64>] = reinterpret_slice(x);

        self.surface_topo.iter().flat_map(move |tri| {
            let p = tri_at(disp, tri);
            let c = [p[1].cross(p[2]), p[2].cross(p[0]), p[0].cross(p[1])];

            (0..3).flat_map(move |vi| (0..3).map(move |j| c[vi][j]))
        })
    }
}

impl ConstraintJacobian<f64> for VolumeConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        3 * 3 * self.surface_topo.len()
    }
    fn constraint_jacobian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        debug_assert_eq!(indices.len(), self.constraint_jacobian_size());
        for (out, idx) in indices
            .iter_mut()
            .zip(self.constraint_jacobian_indices_iter())
        {
            *out = idx + offset;
        }
    }
    fn constraint_jacobian_values(&self, x: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.constraint_jacobian_values_iter(x))
        {
            *out = val;
        }
    }
}

// TODO: move to geo::math
/// Convert a 3 dimensional vector `v = (x,y,z)` into a skew matrix `[v]×` given by
/// ```verbatim
/// ⎡ 0 -z  y⎤
/// ⎢ z  0 -x⎥
/// ⎣-y  x  0⎦
/// ```
#[inline]
fn skew(x: Vector3<f64>) -> Matrix3<f64> {
    Matrix3([[0.0, x[2], -x[1]], [-x[2], 0.0, x[0]], [x[1], -x[0], 0.0]])
}

impl VolumeConstraint {
    /// A generic Hessian element iterator. This is used to implement iterators over indices and
    /// values of the sparse Hessian matrix enetries.
    /// Note: it is an attempt to code reuse. Ideally we should use generators here.
    fn constraint_hessian_iter<'a>(
        tri: &'a [usize; 3],
    ) -> impl Iterator<Item = ((usize, usize), (usize, usize), usize, usize)> + 'a {
        (0..3).flat_map(move |vi| {
            let col_v = tri[vi];
            let row_v = move |off| tri[(vi + off) % 3];
            (1..=2)
                .filter(move |&off| row_v(off) > col_v)
                .flat_map(move |off| {
                    (0..3).flat_map(move |c| {
                        (0..3)
                            .filter(move |&r| r != c)
                            .map(move |r| ((row_v(off), col_v), (r, c), vi, off))
                    })
                })
        })
    }

    pub fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        self.surface_topo.iter().flat_map(move |tri| {
            Self::constraint_hessian_iter(tri).map(|((row_v, col_v), (r, c), _, _)| {
                MatrixElementIndex {
                    row: 3 * row_v + r,
                    col: 3 * col_v + c,
                }
            })
        })
    }

    pub fn constraint_hessian_values_iter<'a>(
        &'a self,
        x: &'a [f64],
        lambda: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        let disp: &[Vector3<f64>] = reinterpret_slice(x);

        self.surface_topo.iter().flat_map(move |tri| {
            let p = tri_at(disp, tri);
            let local_hess = [skew(p[0]), skew(p[1]), skew(p[2])];
            Self::constraint_hessian_iter(tri).map(move |(_, (r, c), vi, off)| {
                let vjn = (vi + off + off) % 3;
                let factor = if off == 1 { 1.0 } else { -1.0 };
                factor * lambda[0] * local_hess[vjn][c][r]
            })
        })
    }
}

impl ConstraintHessian<f64> for VolumeConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        6 * 3 * self.surface_topo.len()
    }
    fn constraint_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        debug_assert_eq!(indices.len(), self.constraint_hessian_size());
        for (out, idx) in indices
            .iter_mut()
            .zip(self.constraint_hessian_indices_iter())
        {
            *out = idx + offset;
        }
    }
    fn constraint_hessian_values(&self, x: &[f64], lambda: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_hessian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.constraint_hessian_values_iter(x, lambda))
        {
            *out = val;
        }
    }
}
