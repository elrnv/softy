use crate::constraint::*;
use geo::math::{Matrix3, Vector3};
use geo::ops::Volume;
use crate::matrix::*;
use crate::TetMesh;
use reinterpret::*;

#[derive(Clone, Debug, PartialEq)]
pub struct VolumeConstraint {
    /// The topology of the surface of a tetrahedral mesh. This is a vector of triplets of indices
    /// of tetmesh vertices. Each triplet corresponds to a triangle on the surface of the tetmesh.
    pub surface_topo: Vec<[usize; 3]>,

    /// The volume of the solid at rest. The contraint is equated to this value.
    pub rest_volume: f64,
}

impl VolumeConstraint {
    pub fn new(tetmesh: &TetMesh) -> Self {
        let surface_topo = tetmesh.surface_topo();
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

/// A utility function to index a slice using three indices, creating a new array of 3
/// corresponding entries of the slice.
fn tri_at<T: Copy>(slice: &[T], tri: &[usize; 3]) -> [T; 3] {
    [slice[tri[0]], slice[tri[1]], slice[tri[2]]]
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

    fn constraint(&self, x: &[f64], dx: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        let pos: &[[f64; 3]] = reinterpret_slice(x);
        let disp: &[[f64; 3]] = reinterpret_slice(dx);
        let mut total_volume = 0.0;
        for tri in self.surface_topo.iter() {
            let p = Matrix3(tri_at(pos, tri)) + Matrix3(tri_at(disp, tri));
            let signed_volume = p[0].dot(p[1].cross(p[2]));
            total_volume += signed_volume;
        }
        value[0] = total_volume;
    }
}

impl VolumeConstraint {
    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
    fn constraint_jacobian_indices_iter<'a>(
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
    fn constraint_jacobian_values_iter<'a>(
        &'a self,
        x: &'a [f64],
        dx: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        let pos: &[[f64; 3]] = reinterpret_slice(x);
        let disp: &[[f64; 3]] = reinterpret_slice(dx);

        self.surface_topo.iter().flat_map(move |tri| {
            let p = Matrix3(tri_at(pos, tri)) + Matrix3(tri_at(disp, tri));
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
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        Box::new(VolumeConstraint::constraint_jacobian_indices_iter(self))
    }
    fn constraint_jacobian_values(&self, x: &[f64], dx: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.constraint_jacobian_values_iter(x, dx))
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
        dx: &'a [f64],
        lambda: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        let pos: &[[f64; 3]] = reinterpret_slice(x);
        let disp: &[[f64; 3]] = reinterpret_slice(dx);

        self.surface_topo.iter().flat_map(move |tri| {
            let p = Matrix3(tri_at(pos, tri)) + Matrix3(tri_at(disp, tri));
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
    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        Box::new(VolumeConstraint::constraint_hessian_indices_iter(self))
    }
    fn constraint_hessian_values(&self, x: &[f64], dx: &[f64], lambda: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_hessian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.constraint_hessian_values_iter(x, dx, lambda))
        {
            *out = val;
        }
    }
}
