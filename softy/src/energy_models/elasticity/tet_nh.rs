//! Neo-Hookean energy model for tetrahedral meshes.

//use std::path::Path;
//use geo::io::save_tetmesh;
use super::TetEnergy;
use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::*;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::Tetrahedron;
use num_traits::FromPrimitive;
use num_traits::Zero;
use rayon::prelude::*;
use reinterpret::*;
use utils::soap::*;
use utils::zip;
use unroll::unroll_for_loops;

/// Per-tetrahedron Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for tet energy computation. It encapsulates tet specific energy computation.
#[allow(non_snake_case)]
pub struct NeoHookeanTetEnergy<T: Real> {
    Dx: Matrix3<T>,
    DX_inv: Matrix3<T>,
    volume: T,
    lambda: T,
    mu: T,
}

impl<T: Real> TetEnergy<T> for NeoHookeanTetEnergy<T> {
    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<T>, DX_inv: Matrix3<T>, volume: T, lambda: T, mu: T) -> Self {
        NeoHookeanTetEnergy {
            Dx,
            DX_inv,
            volume,
            lambda,
            mu,
        }
    }

    /// Compute the deformation gradient `F` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient(&self) -> Matrix3<T> {
        self.Dx * self.DX_inv
    }

    /// Compute the deformation gradient differential `dF` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        // Build differential dDx
        let dDx = Matrix3::new(tet_dx.shape_matrix()).transpose();
        dDx * self.DX_inv
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy(&self) -> T {
        let NeoHookeanTetEnergy {
            volume, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.frob_norm_squared(); // tr(F^TF)
        let J = F.determinant();
        if J <= T::zero() {
            T::infinity()
        } else {
            let logJ = J.ln();
            let half = T::from(0.5).unwrap();
            volume
                * (half * mu * (I - T::from(3.0).unwrap()) - mu * logJ
                    + half * lambda * logJ * logJ)
        }
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_gradient(&self) -> [Vector3<T>; 4] {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let J = F.determinant();
        if J <= T::zero() {
            [Vector3::zero(); 4]
        } else {
            let F_inv = F.inverse().unwrap();
            let logJ = J.ln();
            let H = DX_inv * (F.transpose() * mu + F_inv * (lambda * logJ - mu)) * volume;
            [H[0], H[1], H[2], -H[0] - H[1] - H[2]]
        }
    }

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in row-major
    /// order to be consistent with the 3x3 Matrices.
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    #[inline]
    fn elastic_energy_hessian(&self) -> [[Matrix3<T>; 4]; 4] {
        let NeoHookeanTetEnergy {
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
        for row in 0..3 {
            for col in 0..3 {
                let mut last_hess = T::zero();
                for n in 0..3 {
                    // which vertex
                    for k in 0..3 {
                        // with respect to which vertex
                        let c_lambda = lambda * Dx_inv_tr[row][n] * Dx_inv_tr[col][k];
                        let c_alpha = alpha * Dx_inv_tr[col][n] * Dx_inv_tr[row][k];
                        let mut h = volume * (c_alpha + c_lambda);
                        if col == row {
                            h += volume * mu * A[n][k];
                        }

                        // skip upper trianglar part
                        if (n == k && row >= col) || n > k {
                            local_hessians[n][k][row][col] = h;
                        }
                        // with respect to last vertex
                        local_hessians[3][k][row][col] -= h;
                        last_hess += h;
                    }
                }

                // last vertex
                if row >= col {
                    local_hessians[3][3][row][col] = last_hess;
                }
            }
        }

        local_hessians
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where row `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the rows.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian_product_transpose(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        let NeoHookeanTetEnergy {
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

            DX_inv * dP.transpose() * volume
        } else {
            Matrix3::zero()
        }
    }
}

/// A possibly non-linear elastic energy for tetrahedral meshes.
/// This type wraps a `TetMeshSolid` to provide an interfce for computing a hyperelastic energy.
pub struct TetMeshElasticity<'a, E>(pub &'a TetMeshSolid, std::marker::PhantomData<E>);

/// NeoHookean elasticity model.
pub type TetMeshNeoHookean<'a, T> = TetMeshElasticity<'a, NeoHookeanTetEnergy<T>>;

impl<'a, E> TetMeshElasticity<'a, E> {
    const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 78; // There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)

    pub fn new(solid: &'a TetMeshSolid) -> Self {
        TetMeshElasticity(solid, std::marker::PhantomData)
    }

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
                for row in 0..3 {
                    let end = if n == k { row + 1 } else { 3 };
                    for col in 0..end {
                        value(i, (n, k), (row, col), &mut h);
                        i += 1;
                    }
                }
            }
        }

        assert_eq!(i, Self::NUM_HESSIAN_TRIPLETS_PER_TET)
    }
}

/// Define a hyperelastic energy model for `TetMeshSolid`s.
impl<T: Real, E: TetEnergy<T>> Energy<T> for TetMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy(&self, x0: &[T], x1: &[T]) -> T {
        let TetMeshSolid {
            ref tetmesh,
            material,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        let pos0: &[Vector3<T>] = reinterpret_slice(x0);
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);

        zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                .unwrap(),
            tetmesh.cell_iter(),
            tetmesh
                .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                .unwrap(),
            tetmesh.attrib_iter::<MuType, CellIndex>(MU_ATTRIB).unwrap()
        )
        .map(|(&vol, &DX_inv, cell, &lambda, &mu)| {
            let tet_x1 = Tetrahedron::from_indexed_slice(cell, pos1);
            let tet_dx = &tet_x1 - &Tetrahedron::from_indexed_slice(cell, pos0);
            let Dx = Matrix3::new(tet_x1.shape_matrix()).transpose();
            let DX_inv = DX_inv.map_inner(|x| T::from(x).unwrap());
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let half = T::from(0.5).unwrap();
            let damping = T::from(damping).unwrap();
            let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);
            // elasticity
            tet_energy.elastic_energy()
                + half * damping * {
                    let dH = tet_energy.elastic_energy_hessian_product_transpose(&tet_dx);
                    // damping (viscosity)
                    dH[0].dot(Vector3::new(tet_dx.0.into()))
                        + dH[1].dot(Vector3::new(tet_dx.1.into()))
                        + dH[2].dot(Vector3::new(tet_dx.2.into()))
                        - (dH * Vector3::new(tet_dx.3.into())).sum()
                }
        })
        .sum()
    }
}

impl<T: Real, E: TetEnergy<T>> EnergyGradient<T> for TetMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, x0: &[T], x1: &[T], grad_f: &mut [T]) {
        let TetMeshSolid {
            ref tetmesh,
            material,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        debug_assert_eq!(grad_f.len(), x0.len());
        debug_assert_eq!(grad_f.len(), x1.len());

        let pos0: &[Vector3<T>] = reinterpret_slice(x0);
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, &DX_inv, cell, &lambda, &mu) in zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                .unwrap(),
            tetmesh.cell_iter(),
            tetmesh
                .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                .unwrap(),
            tetmesh.attrib_iter::<MuType, CellIndex>(MU_ATTRIB).unwrap()
        ) {
            // Make deformed tet.
            let tet_x1 = Tetrahedron::from_indexed_slice(cell, pos1);
            // Make tet displacement.
            let tet_dx = &tet_x1 - &Tetrahedron::from_indexed_slice(cell, pos0);

            let DX_inv = DX_inv.map_inner(|x| T::from(x).unwrap());
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let damping = T::from(damping).unwrap();

            let tet_energy = E::new(
                Matrix3::new(tet_x1.shape_matrix()).transpose(),
                DX_inv,
                vol,
                lambda,
                mu,
            );

            let grad = tet_energy.elastic_energy_gradient();

            for i in 0..4 {
                gradient[cell[i]] += grad[i];
            }

            // Needed for damping.
            let dH = tet_energy.elastic_energy_hessian_product_transpose(&tet_dx);
            for i in 0..3 {
                // Damping
                gradient[cell[i]] += dH[i] * damping;
                gradient[cell[3]] -= dH[i] * damping;
            }
        }
    }
}

impl<E> EnergyHessianTopology for TetMeshElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        Self::NUM_HESSIAN_TRIPLETS_PER_TET * self.0.tetmesh.num_cells()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tetmesh = &self.0.tetmesh;

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_row_chunks: &mut [[I; 78]] = reinterpret_mut_slice(rows);
            let hess_col_chunks: &mut [[I; 78]] = reinterpret_mut_slice(cols);

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
                            std::mem::swap(&mut global_row, &mut global_col);
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

        let tetmesh = &self.0.tetmesh;

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementIndex; 78]] = reinterpret_mut_slice(indices);

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
}

impl<T: Real + Send + Sync, E: TetEnergy<T>> EnergyHessian<T> for TetMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(
        &self,
        _: &[T],
        x1: &[T],
        scale: T,
        values: &mut [T],
    ) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let TetMeshSolid {
            ref tetmesh,
            material,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        let pos1: &[Vector3<T>] = reinterpret_slice(x1);

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[T; 78]] = reinterpret_mut_slice(values);

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                tetmesh
                    .attrib_as_slice::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap()
                    .par_iter(),
                tetmesh
                    .attrib_as_slice::<RefShapeMtxInvType, CellIndex>(
                        REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                    )
                    .unwrap()
                    .par_iter(),
                tetmesh.cells().par_iter(),
                tetmesh
                    .attrib_as_slice::<LambdaType, CellIndex>(LAMBDA_ATTRIB,)
                    .unwrap()
                    .par_iter(),
                tetmesh
                    .attrib_as_slice::<MuType, CellIndex>(MU_ATTRIB,)
                    .unwrap()
                    .par_iter(),
            ));

            hess_iter.for_each(|(tet_hess, (&vol, &DX_inv, cell, &lambda, &mu))| {
                // Make deformed tet.
                let tet_x1 = Tetrahedron::from_indexed_slice(cell, pos1);

                let Dx = Matrix3::new(tet_x1.shape_matrix()).transpose();

                let DX_inv = DX_inv.map_inner(|x| T::from(x).unwrap());
                let vol = T::from(vol).unwrap();
                let lambda = T::from(lambda).unwrap();
                let mu = T::from(mu).unwrap();

                let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);

                let factor = T::from(1.0 + damping).unwrap() * scale;

                let local_hessians = tet_energy.elastic_energy_hessian();

                Self::hessian_for_each(
                    |n, k| local_hessians[n][k] * factor,
                    |i, _, (row, col), h| tet_hess[i] = h[row][col],
                );
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::elasticity::test_utils::*;
    use crate::energy_models::test_utils::*;
    use crate::fem::SolverBuilder;
    use crate::objects::TetMeshSolid;
    use geo::mesh::VertexPositions;

    fn material() -> SolidMaterial {
        SolidMaterial::new(0).with_elasticity(ElasticityParameters {
            lambda: 5.4,
            mu: 263.1,
        })
    }

    fn test_solids() -> Vec<TetMeshSolid> {
        let material = material();

        test_meshes()
            .into_iter()
            .map(|mut tetmesh| {
                // Prepare attributes relevant for elasticity computations.
                SolverBuilder::prepare_deformable_mesh_vertex_attributes(&mut tetmesh).unwrap();
                SolverBuilder::prepare_deformable_tetmesh_attributes(&mut tetmesh).unwrap();
                let mut solid = TetMeshSolid::new(tetmesh, material);
                SolverBuilder::prepare_elasticity_attributes(&mut solid).unwrap();
                solid
            })
            .collect()
    }

    fn build_energies(solids: &[TetMeshSolid]) -> Vec<(TetMeshNeoHookean<autodiff::F>, Vec<[f64; 3]>)> {
        solids
            .iter()
            .map(|solid| {
                (
                    TetMeshNeoHookean::new(solid),
                    solid.tetmesh.vertex_positions().to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn tet_energy_gradient() {
        tet_energy_gradient_tester::<NeoHookeanTetEnergy<autodiff::F>>();
    }

    #[test]
    fn tet_energy_hessian() {
        tet_energy_hessian_tester::<NeoHookeanTetEnergy<autodiff::F>>();
    }

    #[test]
    fn tet_energy_hessian_product() {
        tet_energy_hessian_product_tester::<NeoHookeanTetEnergy<f64>>();
    }

    #[test]
    fn gradient() {
        let solids = test_solids();
        gradient_tester(build_energies(&solids), EnergyType::Position);
    }

    #[test]
    fn hessian() {
        let solids = test_solids();
        hessian_tester(build_energies(&solids), EnergyType::Position);
    }
}
