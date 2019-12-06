//! Neo-Hookean energy model for tetrahedral meshes.

//use std::path::Path;
//use geo::io::save_tetmesh;
use super::{LinearElementEnergy, TriEnergy};
use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::*;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::Triangle;
use num_traits::FromPrimitive;
use num_traits::Zero;
use rayon::prelude::*;
use reinterpret::*;
use unroll::unroll_for_loops;
use utils::soap::*;
use utils::zip;

/// Per-tetrahedron Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for tet energy computation. It encapsulates tet specific energy computation.
#[allow(non_snake_case)]
pub struct NeoHookeanTriEnergy<T> {
    Dx: Matrix2x3<T>,
    DX_inv: Matrix2<T>,
    area: T,
    lambda: T,
    mu: T,
}

impl<T: Real> NeoHookeanTriEnergy<T> {
    /// Compute the deformation gradient `F` for this triangle.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient(&self) -> Matrix2x3<T> {
        self.DX_inv * self.Dx
    }

    /// Compute the deformation gradient differential `dF` for this triangle.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, tri_dx: &Triangle<T>) -> Matrix2x3<T> {
        // Build differential dDx
        let dDx = Matrix2x3::new(tri_dx.shape_matrix());
        self.DX_inv * dDx
    }
}

impl<T: Real> LinearElementEnergy<T> for NeoHookeanTriEnergy<T> {
    type Element = Triangle<T>;
    type ShapeMatrix = Matrix2x3<T>;
    type RefShapeMatrix = Matrix2<T>;
    type Gradient = [Vector3<T>; 3];
    type Hessian = [[Matrix3<T>; 3]; 3];

    #[allow(non_snake_case)]
    fn new(
        Dx: Self::ShapeMatrix,
        DX_inv: Self::RefShapeMatrix,
        area: T,
        lambda: T,
        mu: T,
    ) -> Self {
        NeoHookeanTriEnergy {
            Dx,
            DX_inv,
            area,
            lambda,
            mu,
        }
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn energy(&self) -> T {
        let NeoHookeanTriEnergy {
            area, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let C = F * F.transpose();
        let C_det = C.determinant();
        let I = C[0][0] + C[1][1]; // trace
        if C_det <= T::zero() {
            T::infinity()
        } else {
            let half = T::from(0.5).unwrap();
            let log_C_det = C_det.ln();
            let _2 = T::from(2.0).unwrap();
            area
                * half * (mu * (I - _2 - log_C_det)
                    + T::from(0.25).unwrap() * lambda * log_C_det * log_C_det)
        }
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a triangle and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn energy_gradient(&self) -> [Vector3<T>; 3] {
        let NeoHookeanTriEnergy {
            DX_inv,
            area,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let C = F * F.transpose();
        let C_det = C.determinant();
        if C_det <= T::zero() {
            [Vector3::zero(); 3]
        } else {
            let F_inv_tr = C.inverse().unwrap() * F;
            let logJ = T::from(0.5).unwrap() * C_det.ln();
            let P = F * mu + F_inv_tr * (lambda * logJ - mu);
            let H = DX_inv.transpose() * P * area;
            [H[0], H[1], -H[0] - H[1]]
        }
    }

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in row-major
    /// order to be consistent with the 3x3 Matrices.
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    #[inline]
    fn energy_hessian(&self) -> [[Matrix3<T>; 3]; 3] {
        let mut hess = [[Matrix3::zeros(); 3]; 3];

        let mut tri_dx = Triangle([T::zero(); 3].into(), [T::zero(); 3].into(), [T::zero(); 3].into());

        for i in 0..3 { // vertex
            for row in 0..3 { // component
                tri_dx[i][row] = T::one();
                let h = self.energy_hessian_product_transpose(&tri_dx);
                for j in 0..2 { // vertex
                    for col in 0..3 { // component
                        if i > j || (i == j && row >= col) {
                            hess[i][j][row][col] += h[j][col];
                            if i == 2 {
                                hess[i][2][row][col] -= h[j][col];
                            }
                        }
                    }
                }
                tri_dx[i][row] = T::zero();
            }
        }

        hess
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where row `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the rows.
    #[allow(non_snake_case)]
    #[inline]
    fn energy_hessian_product_transpose(&self, tri_dx: &Triangle<T>) -> Matrix2x3<T> {
        let NeoHookeanTriEnergy {
            DX_inv,
            area,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(tri_dx);
        let C = F * F.transpose();
        let C_det = C.determinant();
        if C_det > T::zero() {
            let alpha = mu - lambda * T::from(0.5).unwrap() * C_det.ln();

            let C_inv = C.inverse().unwrap();
            let F_inv: Matrix3x2<_> = (C_inv * F).transpose();
            let dF_F_inv: Matrix2<_> = dF * F_inv;
            let n = F[0].cross(F[1]).normalized();

            let dP_tr: Matrix3x2<_> = dF.transpose() * mu
                + F_inv * dF_F_inv * alpha
                + F_inv * (dF_F_inv.trace() * lambda)
                - n * (n.transpose() * dF.transpose() * C_inv) * alpha;

            DX_inv.transpose() * dP_tr.transpose() * area
        } else {
            Matrix2x3::zero()
        }
    }
}

/// A possibly non-linear elastic energy for tetrahedral meshes.
/// This type wraps a `TriMeshSolid` to provide an interfce for computing a hyperelastic energy.
pub struct TriMeshElasticity<'a, E>(pub &'a TriMeshShell, std::marker::PhantomData<E>);

/// NeoHookean elasticity model.
pub type TriMeshNeoHookean<'a, T> = TriMeshElasticity<'a, NeoHookeanTriEnergy<T>>;

impl<'a, E> TriMeshElasticity<'a, E> {
    const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 45; // There are 3*6 + 3*9 = 45 triplets per triangle (overestimate)

    pub fn new(shell: &'a TriMeshShell) -> Self {
        TriMeshElasticity(shell, std::marker::PhantomData)
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
        let mut i = 0; // triplet index for the tri. there should be 45 in total
        for k in 0..3 {
            for n in k..3 {
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

        assert_eq!(i, Self::NUM_HESSIAN_TRIPLETS_PER_TRI)
    }
}

/// Define a hyperelastic energy model for `TriMeshSolid`s.
impl<T: Real, E: TriEnergy<T>> Energy<T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy(&self, x0: &[T], x1: &[T]) -> T {
        let TriMeshShell {
            ref trimesh,
            material,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        let pos0: &[Vector3<T>] = reinterpret_slice(x0);
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);

        zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<RefTriShapeMtxInvType, FaceIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                .unwrap(),
            trimesh.face_iter(),
            trimesh
                .attrib_iter::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)
                .unwrap(),
            trimesh.attrib_iter::<MuType, FaceIndex>(MU_ATTRIB).unwrap()
        )
        .map(|(&area, &DX_inv, face, &lambda, &mu)| {
            let tri_x1 = Triangle::from_indexed_slice(face, pos1);
            let tri_x0 = Triangle::from_indexed_slice(face, pos0);
            let tri_dx = Triangle::new(
                (*tri_x1.as_array().as_tensor() - tri_x0.into_array().into_tensor()).into());
            let Dx = Matrix2x3::new(tri_x1.shape_matrix());
            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let area = T::from(area).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let half = T::from(0.5).unwrap();
            let damping = T::from(damping).unwrap();
            let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);
            // elasticity
            tri_energy.energy()
                + half * damping * {
                    let dH = tri_energy.energy_hessian_product_transpose(&tri_dx);
                    // damping (viscosity)
                    dH[0].dot(Vector3::new(tri_dx.0.into()))
                        + dH[1].dot(Vector3::new(tri_dx.1.into()))
                        - (dH * Vector3::new(tri_dx.2.into())).sum()
                }
        })
        .sum()
    }
}

impl<T: Real, E: TriEnergy<T>> EnergyGradient<T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, x0: &[T], x1: &[T], grad_f: &mut [T]) {
        let TriMeshShell {
            ref trimesh,
            material,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        debug_assert_eq!(grad_f.len(), x0.len());
        debug_assert_eq!(grad_f.len(), x1.len());

        let pos0: &[Vector3<T>] = reinterpret_slice(x0);
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from face-vertices to vertices themeselves
        for (&area, &DX_inv, face, &lambda, &mu) in zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<RefTriShapeMtxInvType, FaceIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                .unwrap(),
            trimesh.face_iter(),
            trimesh
                .attrib_iter::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)
                .unwrap(),
            trimesh.attrib_iter::<MuType, FaceIndex>(MU_ATTRIB).unwrap()
        ) {
            // Make deformed tri.
            let tri_x1 = Triangle::from_indexed_slice(face, pos1);
            // Make tri displacement.
            let tri_x0 = Triangle::from_indexed_slice(face, pos0);
            let tri_dx = Triangle::new(
                (*tri_x1.as_array().as_tensor() - tri_x0.into_array().into_tensor()).into());

            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let area = T::from(area).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let damping = T::from(damping).unwrap();

            let tri_energy = E::new(
                Matrix2x3::new(tri_x1.shape_matrix()),
                DX_inv,
                area,
                lambda,
                mu,
            );

            let grad = tri_energy.energy_gradient();

            for i in 0..3 {
                gradient[face[i]] += grad[i];
            }

            // Needed for damping.
            let dH = tri_energy.energy_hessian_product_transpose(&tri_dx);
            for i in 0..2 {
                // Damping
                gradient[face[i]] += dH[i] * damping;
                gradient[face[2]] -= dH[i] * damping;
            }
        }
    }
}

impl<E> EnergyHessianTopology for TriMeshElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        Self::NUM_HESSIAN_TRIPLETS_PER_TRI * self.0.trimesh.num_faces()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let trimesh = &self.0.trimesh;

        {
            // Break up the hessian indices into chunks of elements for each tri.
            let hess_row_chunks: &mut [[I; 45]] = reinterpret_mut_slice(rows);
            let hess_col_chunks: &mut [[I; 45]] = reinterpret_mut_slice(cols);

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(trimesh.faces().par_iter());

            hess_iter.for_each(|((tri_hess_rows, tri_hess_cols), face)| {
                Self::hessian_for_each(
                    |_, _| (),
                    |i, (n, k), (row, col), _| {
                        let mut global_row = 3 * face[n] + row;
                        let mut global_col = 3 * face[k] + col;
                        if face[n] < face[k] {
                            // In the upper triangular part of the global matrix, transpose
                            std::mem::swap(&mut global_row, &mut global_col);
                        }
                        tri_hess_rows[i] = I::from_usize(global_row + offset.row).unwrap();
                        tri_hess_cols[i] = I::from_usize(global_col + offset.col).unwrap();
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

        let trimesh = &self.0.trimesh;

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementIndex; 45]] = reinterpret_mut_slice(indices);

            let hess_iter = hess_chunks.par_iter_mut().zip(trimesh.faces().par_iter());

            hess_iter.for_each(|(tri_hess, face)| {
                Self::hessian_for_each(
                    |_, _| (),
                    |i, (n, k), (row, col), _| {
                        let mut global_row = 3 * face[n] + row;
                        let mut global_col = 3 * face[k] + col;
                        if face[n] < face[k] {
                            // In the upper triangular part of the global matrix, transpose
                            std::mem::swap(&mut global_row, &mut global_col);
                        }
                        tri_hess[i] = MatrixElementIndex {
                            row: global_row + offset.row,
                            col: global_col + offset.col,
                        };
                    },
                );
            });
        }
    }
}

impl<T: Real + Send + Sync, E: TriEnergy<T>> EnergyHessian<T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _: &[T], x1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let TriMeshShell {
            ref trimesh,
            material,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        let pos1: &[Vector3<T>] = reinterpret_slice(x1);

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[T; 45]] = reinterpret_mut_slice(values);

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                trimesh
                    .attrib_as_slice::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                    .unwrap()
                    .par_iter(),
                trimesh
                    .attrib_as_slice::<RefTriShapeMtxInvType, FaceIndex>(
                        REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                    )
                    .unwrap()
                    .par_iter(),
                trimesh.faces().par_iter(),
                trimesh
                    .attrib_as_slice::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)
                    .unwrap()
                    .par_iter(),
                trimesh
                    .attrib_as_slice::<MuType, FaceIndex>(MU_ATTRIB)
                    .unwrap()
                    .par_iter(),
            ));

            hess_iter.for_each(|(tri_hess, (&area, &DX_inv, face, &lambda, &mu))| {
                // Make deformed tet.
                let tri_x1 = Triangle::from_indexed_slice(face, pos1);

                let Dx = Matrix2x3::new(tri_x1.shape_matrix());

                let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                let area = T::from(area).unwrap();
                let lambda = T::from(lambda).unwrap();
                let mu = T::from(mu).unwrap();

                let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

                let factor = T::from(1.0 + damping).unwrap() * scale;

                let local_hessians = tri_energy.energy_hessian();

                Self::hessian_for_each(
                    |n, k| local_hessians[n][k] * factor,
                    |i, _, (row, col), h| tri_hess[i] = h[row][col],
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
    use crate::objects::TriMeshShell;
    use geo::mesh::VertexPositions;

    fn material() -> ShellMaterial {
        ShellMaterial::new(0).with_elasticity(ElasticityParameters {
            lambda: 5.4,
            mu: 263.1,
            model: ElasticityModel::NeoHookean,
        })
    }

    fn test_shells() -> Vec<TriMeshShell> {
        let material = material();

        test_trimeshes()
            .into_iter()
            .map(|mut trimesh| {
                // Prepare attributes relevant for elasticity computations.
                SolverBuilder::prepare_deformable_mesh_vertex_attributes(&mut trimesh).unwrap();
                SolverBuilder::prepare_deformable_trimesh_attributes(&mut trimesh).unwrap();
                let mut shell = TriMeshShell::new(trimesh, material);
                SolverBuilder::prepare_elasticity_attributes(&mut shell).unwrap();
                shell
            })
            .collect()
    }

    fn build_energies(
        shells: &[TriMeshShell],
    ) -> Vec<(TriMeshNeoHookean<autodiff::F>, Vec<[f64; 3]>)> {
        shells
            .iter()
            .map(|shell| {
                (
                    TriMeshNeoHookean::new(shell),
                    shell.trimesh.vertex_positions().to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn tri_energy_gradient() {
        tri_energy_gradient_tester::<NeoHookeanTriEnergy<autodiff::F>>();
    }

    #[test]
    fn tri_energy_hessian() {
        tri_energy_hessian_tester::<NeoHookeanTriEnergy<autodiff::F>>();
    }

    #[test]
    fn tri_energy_hessian_product() {
        tri_energy_hessian_product_tester::<NeoHookeanTriEnergy<f64>>();
    }

    #[test]
    fn gradient() {
        let shells = test_shells();
        gradient_tester(build_energies(&shells), EnergyType::Position);
    }

    #[test]
    fn hessian() {
        let shells = test_shells();
        hessian_tester(build_energies(&shells), EnergyType::Position);
    }
}
