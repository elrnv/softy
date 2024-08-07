//! Invertible Neo-Hookean energy model for tetrahedral meshes.

//use std::path::Path;
//use geo::io::save_tetmesh;
use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::Real;
use crate::TetMesh;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::Tetrahedron;
use num_traits::FromPrimitive;
use rayon::prelude::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use tensr::{Matrix3, Vector3};

/// Per-tetrahedron Invertible Neo-Hookean energy model. This struct stores conveniently
/// precomputed values for tet energy computation. It encapsulates tet specific energy computation.
/// `lambda` and `mu` are the standard Lamé parameters, while `epsilon` is the threshold for the
/// determinant of the deformation gradient (`J = det F`) to be positive. If `J` is less than
/// epsilon, then a quadratic approximation of the volume term of the standard Neo-Hookean will be
/// used. This allows the energy to have real values for inverted tets. Smaller values of `epsilon`
/// will produce a model that is more faithful to the traditional Neo-Hookean energy.
#[allow(non_snake_case)]
pub struct InvertibleNHTetEnergy<T: Real> {
    Dx: Matrix3<T>,
    DX_inv: Matrix3<T>,
    volume: T,
    lambda: T,
    mu: T,
    epsilon: T,
}

impl<T: Real> InvertibleNHTetEnergy<T> {
    /// Compute the deformation gradient `F` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    pub fn deformation_gradient(&self) -> Matrix3<T> {
        self.Dx * self.DX_inv
    }

    /// Compute the deformation gradient differential `dF` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        // Build differential dDx
        let dDx = tet_dx.shape_matrix();
        dDx * self.DX_inv
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    pub fn elastic_energy(&self) -> T {
        let InvertibleNHTetEnergy {
            volume,
            mu,
            lambda,
            epsilon: eps,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.norm_squared(); // tr(F^TF)
        let J = F.determinant();
        let half = T::from(0.5).unwrap();

        let E = if J <= eps {
            let log_eps = eps.ln();
            let two = T::from(2.0).unwrap();
            let J_eps = (J - eps) / eps;
            let logJ_approx = log_eps + J_eps - half * J_eps * J_eps;
            let logJ2_approx =
                log_eps * log_eps + two * log_eps * J_eps + (T::one() - log_eps) * J_eps * J_eps;
            half * lambda * logJ2_approx - mu * logJ_approx
        } else {
            let logJ = J.ln();
            half * lambda * logJ * logJ - mu * logJ
        };

        volume * (E + half * mu * (I - T::from(3.0).unwrap()))
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    pub fn elastic_energy_gradient(&self) -> [Vector3<T>; 4] {
        let InvertibleNHTetEnergy {
            DX_inv,
            volume,
            mu,
            lambda,
            epsilon: eps,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let J = F.determinant();
        let H = if J <= eps {
            Matrix3::zeros()
        } else {
            let F_inv_tr = F.inverse_transpose().unwrap();
            let logJ = J.ln();
            (F * mu + F_inv_tr * (lambda * logJ - mu)) * DX_inv.transpose() * volume
        };
        [H[0], H[1], H[2], -H[0] - H[1] - H[2]]
    }

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in column-major
    /// order to be consistent with the 3x3 Matrices.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian(&self) -> [[Matrix3<T>; 4]; 4] {
        let InvertibleNHTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;

        let mut local_hessians = [[Matrix3::zeros(); 4]; 4];

        let F = self.deformation_gradient();
        let J = F.determinant();
        if J <= T::zero() {
            return local_hessians;
        }

        let A = DX_inv * DX_inv.transpose();

        // Theoretically we known Dx is invertible since F is, but it could have
        // numerical differences, so we check anyways.
        let Dx_inv_tr = match self.Dx.inverse_transpose() {
            Some(inv) => inv,
            None => return local_hessians,
        };

        let alpha = mu - lambda * J.ln();

        // Off-diagonal elements
        for col in 0..3 {
            for row in 0..3 {
                let mut last_hess = T::zero();
                for k in 0..3 {
                    // which vertex
                    let mut last_wrt_hess = T::zero();
                    for n in 0..3 {
                        // with respect to which vertex
                        let c_lambda = lambda * Dx_inv_tr[n][row] * Dx_inv_tr[k][col];
                        let c_alpha = alpha * Dx_inv_tr[n][col] * Dx_inv_tr[k][row];
                        let mut h = volume * (c_alpha + c_lambda);
                        if col == row {
                            h += volume * mu * A[k][n];
                        }
                        last_wrt_hess -= h;

                        // skip upper trianglar part
                        if (n == k && row >= col) || n > k {
                            local_hessians[k][n][col][row] = h;
                        }
                    }

                    // with respect to last vertex
                    last_hess -= last_wrt_hess;
                    local_hessians[k][3][col][row] = last_wrt_hess;
                }

                // last vertex
                if row >= col {
                    local_hessians[3][3][col][row] = last_hess;
                }
            }
        }

        local_hessians
    }
}

impl<T: Real> TetEnergy<T> for InvertibleNHTetEnergy<T> {
    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<T>, DX_inv: Matrix3<T>, volume: T, lambda: T, mu: T) -> Self {
        InvertibleNHTetEnergy {
            Dx,
            DX_inv,
            volume,
            lambda,
            mu,
            epsilon: T::from(0.2).unwrap(),
        }
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where column `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the columns.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian_product(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        let InvertibleNHTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(tet_dx);
        let J = F.determinant();
        if J > T::zero() {
            let alpha = mu - lambda * J.ln();

            let F_inv_tr = F.inverse_transpose().unwrap();
            let F_inv = F_inv_tr.transpose();

            let dP = dF * mu
                + F_inv_tr * dF.transpose() * F_inv_tr * alpha
                + F_inv_tr * ((F_inv * dF).trace() * lambda);

            dP * DX_inv.transpose() * volume
        } else {
            Matrix3::zeros()
        }
    }
}

/// A possibly non-linear elastic energy for tetrahedral meshes.
#[derive(Debug, PartialEq)]
pub struct ElasticTetMeshEnergy {
    /// The discretization of the solid domain using a tetrahedral mesh. This field should never be
    /// borrowed mutably.
    pub tetmesh: Rc<RefCell<TetMesh>>,
    /// Material parameters.
    material: MaterialModel,
}

/// A builder for the `ElasticTetMeshEnergy` struct. The only required field is the `tetmesh` which
/// is specified in the `new` method.
#[derive(Debug, PartialEq)]
pub struct ElasticTetMeshEnergyBuilder {
    tetmesh: Rc<RefCell<TetMesh>>,
    material: Option<MaterialModel>,
}

/// The material model including elasticity Lame parameters as well as dynamics specific parameter
/// like the damping coefficient.
#[derive(Copy, Clone, Debug, PartialEq)]
struct MaterialModel {
    /// First Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub lambda: f64,
    /// Second Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub mu: f64,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model. This coefficient should incorporate the timestep.
    pub damping: f64,
}

impl Default for MaterialModel {
    fn default() -> Self {
        MaterialModel {
            lambda: 5.4,
            mu: 263.1,
            damping: 0.0,
        }
    }
}

impl ElasticTetMeshEnergyBuilder {
    /// Create a new Neo-Hookean energy model defining a non-linear problem that can be solved
    /// using a non-linear solver like Ipopt.
    /// This function takes a tetrahedron mesh specifying a discretization of the solid domain
    pub fn new(tetmesh: Rc<RefCell<TetMesh>>) -> Self {
        ElasticTetMeshEnergyBuilder {
            tetmesh,
            material: None,
        }
    }

    /// Set the elastic material properties of the volumetric solid discretized by the tetmesh.
    pub fn material(mut self, lambda: f64, mu: f64, damping: f64) -> Self {
        self.material = Some(MaterialModel {
            lambda,
            mu,
            damping,
        });
        self
    }

    pub fn build(&self) -> ElasticTetMeshEnergy {
        let ElasticTetMeshEnergyBuilder {
            ref tetmesh,
            ref material,
        } = self;

        ElasticTetMeshEnergy {
            tetmesh: tetmesh.clone(),
            material: material.unwrap_or_default(),
        }
    }
}

impl ElasticTetMeshEnergy {
    pub const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 78; // There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)

    /// Helper for distributing local Hessian entries into the global Hessian matrix.
    /// This function provides the order of Hessian matrix non-zeros.
    /// `indices` is a map from the local tet vertex indices to their position in the global
    /// tetmesh
    /// `local_hess` is the function that computes a local hessian matrix for a pair of vertex
    /// indices
    /// `value` is the mapping function that would compute the hessian value. In particular
    /// `value` takes 3 pairs of (row, col) indices in this order:
    ///     - vertex indices
    ///     - local matrix indices
    /// as well as the local hessian matrix computed by `local_hess`.
    #[inline]
    fn hessian_for_each<H, L, F>(mut local_hess: L, mut value: F)
    where
        L: FnMut(usize, usize) -> H,
        F: FnMut(usize, (usize, usize), (usize, usize), &mut H),
    {
        let mut i = 0; // triplet index for the tet. there should be 78 in total
        for k in 0..4 {
            for n in k..4 {
                let mut h = local_hess(n, k);
                for col in 0..3 {
                    let start = if n == k { col } else { 0 };
                    for row in start..3 {
                        value(i, (n, k), (row, col), &mut h);
                        i += 1;
                    }
                }
            }
        }

        assert_eq!(i, Self::NUM_HESSIAN_TRIPLETS_PER_TET)
    }
}

/// Define energy for ElasticTetMeshEnergy materials.
impl<T: Real> Energy<T> for ElasticTetMeshEnergy {
    #[allow(non_snake_case)]
    fn energy(&self, x: &[T], dx: &[T]) -> T {
        let ElasticTetMeshEnergy {
            ref tetmesh,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    damping,
                },
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        let prev_pos: &[Vector3<T>] = reinterpret_slice(x);
        let disp: &[Vector3<T>] = reinterpret_slice(dx);

        tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                tetmesh
                    .attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                    .unwrap(),
            )
            .zip(tetmesh.cell_iter())
            .map(|((&vol, &DX_inv), cell)| {
                let tet_dx = Tetrahedron::from_indexed_slice(cell.get(), disp);
                let tet = Tetrahedron::from_indexed_slice(cell.get(), prev_pos) + tet_dx.clone();
                let Dx = tet.shape_matrix();
                let DX_inv = DX_inv.map(|x| T::from(x).unwrap());
                let vol = T::from(vol).unwrap();
                let lambda = T::from(lambda).unwrap();
                let mu = T::from(mu).unwrap();
                let half = T::from(0.5).unwrap();
                let damping = T::from(damping).unwrap();
                let tet_energy = InvertibleNHTetEnergy::new(Dx, DX_inv, vol, lambda, mu);
                // elasticity
                tet_energy.elastic_energy()
                    + half * damping * {
                        let dH = tet_energy.elastic_energy_hessian_product(&tet_dx);
                        // damping (viscosity)
                        (dH[0].dot(tet_dx.0) + dH[1].dot(tet_dx.1) + dH[2].dot(tet_dx.2)
                            - (tet_dx.3.transpose() * dH).sum())
                    }
            })
            .sum()
    }
}

impl<T: Real> EnergyGradient<T> for ElasticTetMeshEnergy {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, x: &[T], dx: &[T], grad_f: &mut [T]) {
        let ElasticTetMeshEnergy {
            ref tetmesh,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    damping,
                },
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        debug_assert_eq!(grad_f.len(), x.len());
        debug_assert_eq!(grad_f.len(), dx.len());

        let prev_pos: &[Vector3<T>] = reinterpret_slice(x);
        let disp: &[Vector3<T>] = reinterpret_slice(dx);

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themeselves
        for ((&vol, &DX_inv), cell) in tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                tetmesh
                    .attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                    .unwrap(),
            )
            .zip(tetmesh.cell_iter())
        {
            // Make tet displacement.
            let tet_dx = Tetrahedron::from_indexed_slice(cell.get(), disp);

            // Make deformed tet.
            let tet = Tetrahedron::from_indexed_slice(cell.get(), prev_pos) + tet_dx.clone();

            let DX_inv = DX_inv.map(|x| T::from(x).unwrap());
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let damping = T::from(damping).unwrap();

            let tet_energy =
                InvertibleNHTetEnergy::new(tet.shape_matrix(), DX_inv, vol, lambda, mu);

            let grad = tet_energy.elastic_energy_gradient();

            for i in 0..4 {
                gradient[cell[i]] += grad[i];
            }

            // Needed for damping.
            let dH = tet_energy.elastic_energy_hessian_product(&tet_dx);
            for i in 0..3 {
                // Damping
                gradient[cell[i]] += dH[i] * damping;
                gradient[cell[3]] -= dH[i] * damping;
            }
        }
    }
}

impl EnergyHessian for ElasticTetMeshEnergy {
    fn energy_hessian_size(&self) -> usize {
        Self::NUM_HESSIAN_TRIPLETS_PER_TET * self.tetmesh.borrow().num_cells()
    }
    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tetmesh = self.tetmesh.borrow();

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_row_chunks: &mut [[I; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
                reinterpret_mut_slice(rows);
            let hess_col_chunks: &mut [[I; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
                reinterpret_mut_slice(cols);

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(tetmesh.cells().par_iter());

            hess_iter.for_each(|((tet_hess_rows, tet_hess_cols), cell)| {
                Self::hessian_for_each(
                    |_, _| (),
                    |i, (n, k), (row, col), _| {
                        let mut global_row = 3 * cell[n] + row;
                        let mut global_col = 3 * cell[k] + col;
                        if cell[n] < cell[k] {
                            // In the upper triangular part of the global matrix, transpose
                            ::std::mem::swap(&mut global_row, &mut global_col);
                        }
                        tet_hess_rows[i] = I::from_usize(global_row + offset.row).unwrap();
                        tet_hess_cols[i] = I::from_usize(global_col + offset.col).unwrap();
                    },
                );
            });
        }
    }

    fn energy_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        assert_eq!(indices.len(), self.energy_hessian_size());

        let tetmesh = self.tetmesh.borrow();

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementIndex; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
                reinterpret_mut_slice(indices);

            let hess_iter = hess_chunks.par_iter_mut().zip(tetmesh.cells().par_iter());

            hess_iter.for_each(|(tet_hess, cell)| {
                Self::hessian_for_each(
                    |_, _| (),
                    |i, (n, k), (row, col), _| {
                        let mut global_row = 3 * cell[n] + row;
                        let mut global_col = 3 * cell[k] + col;
                        if cell[n] < cell[k] {
                            // In the upper triangular part of the global matrix, transpose
                            ::std::mem::swap(&mut global_row, &mut global_col);
                        }
                        tet_hess[i] = MatrixElementIndex {
                            row: global_row + offset.row,
                            col: global_col + offset.col,
                        };
                    },
                );
            });
        }
    }

    #[allow(non_snake_case)]
    fn energy_hessian_values<T: Real + Send + Sync>(
        &self,
        x: &[T],
        dx: &[T],
        scale: T,
        values: &mut [T],
    ) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let ElasticTetMeshEnergy {
            ref tetmesh,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    damping,
                },
            ..
        } = *self;

        let tetmesh = &*tetmesh.borrow();

        let prev_pos: &[Vector3<T>] = reinterpret_slice(x);
        let disp: &[Vector3<T>] = reinterpret_slice(dx);

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[T; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
                reinterpret_mut_slice(values);

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(
                    tetmesh
                        .attrib_as_slice::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                        .unwrap()
                        .par_iter(),
                )
                .zip(
                    tetmesh
                        .attrib_as_slice::<RefShapeMtxInvType, CellIndex>(
                            REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                        )
                        .unwrap()
                        .par_iter(),
                )
                .zip(tetmesh.cells().par_iter());

            hess_iter.for_each(|(((tet_hess, &vol), &DX_inv), cell)| {
                // Make tet displacement.
                let tet_dx = Tetrahedron::from_indexed_slice(cell.get(), disp);
                // Make deformed tet.
                let tet = Tetrahedron::from_indexed_slice(cell.get(), prev_pos) + tet_dx;

                let Dx = tet.shape_matrix();

                let DX_inv = DX_inv.map(|x| T::from(x).unwrap());
                let vol = T::from(vol).unwrap();
                let lambda = T::from(lambda).unwrap();
                let mu = T::from(mu).unwrap();

                let tet_energy = InvertibleNHTetEnergy::new(Dx, DX_inv, vol, lambda, mu);

                let factor = T::from(1.0 + damping).unwrap() * scale;

                let local_hessians = tet_energy.elastic_energy_hessian();

                Self::hessian_for_each(
                    |n, k| local_hessians[k][n] * factor,
                    |i, _, (row, col), h| tet_hess[i] = h[col][row],
                );
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::test_utils::*;

    #[test]
    fn gradient() {
        gradient_tester(
            |mesh| ElasticTetMeshEnergyBuilder::new(Rc::new(RefCell::new(mesh))).build(),
            EnergyType::Position,
        );
    }

    #[test]
    fn hessian() {
        hessian_tester(
            |mesh| ElasticTetMeshEnergyBuilder::new(Rc::new(RefCell::new(mesh))).build(),
            EnergyType::Position,
        );
    }
}
