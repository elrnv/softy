//! Neo-Hookean energy model for triangle meshes.

use num_traits::FromPrimitive;
use num_traits::Zero;
use rayon::prelude::*;
use reinterpret::*;
use unroll::unroll_for_loops;

use geo::mesh::{Attrib, topology::*};
use geo::ops::*;
use geo::prim::Triangle;
use utils::soap::*;
use utils::zip;

use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::*;

use super::{LinearElementEnergy, TriEnergy};

/// Per-triangle Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for triangle energy computation. It encapsulates triangle specific energy computation.
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
    /// be obtained from a triangle and its reference configuration.
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

/// A possibly non-linear elastic energy for triangle meshes.
///
/// This type wraps a `TriMeshShell` to provide an interfce for computing a membrane and bending
/// elastic energies. `E` specifies the per element membrane energy model.
pub struct TriMeshElasticity<'a, E>(pub &'a TriMeshShell, std::marker::PhantomData<E>);

/// NeoHookean elasticity model.
pub type TriMeshNeoHookean<'a, T> = TriMeshElasticity<'a, NeoHookeanTriEnergy<T>>;

const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 45; // There are 3*6 + 3*9 = 45 triplets per triangle

// There are 4*6 = 24 triplets on the diagonal blocks per interior edge.
const NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG: usize = 24;
// There are 4*6 + 3*3*5 = 69 triplets per interior edge in total.
const NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE: usize = 69;

impl<'a, E> TriMeshElasticity<'a, E> {
    pub fn new(shell: &'a TriMeshShell) -> Self {
        TriMeshElasticity(shell, std::marker::PhantomData)
    }

    /// Helper for distributing local membrane Hessian entries into the global Hessian matrix.
    ///
    /// This function provides the order of Hessian matrix non-zeros.
    /// `local_hess` computes a local hessian matrix for a pair of vertex indices.
    /// `value` is the mapping function that would compute the hessian value. In particular
    /// `value` takes 3 pairs of (row, col) indices in this order:
    ///     - vertex indices
    ///     - local matrix indices
    /// as well as the local hessian matrix computed by `local_hess`.
    #[inline]
    fn tri_hessian_for_each<H, L, F>(mut local_hess: L, mut value: F)
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

        assert_eq!(i, NUM_HESSIAN_TRIPLETS_PER_TRI)
    }

    /// Helper for distributing bending Hessian entries into the global Hessian matrix.
    ///
    /// This function is similar to `tri_hessian_for_each` but for the bending energy.
    ///
    /// Each edge neighbourhood is described in [`InteriorEdge`].
    #[inline]
    fn edge_hessian_for_each<D, L, DH, LH, DF, LF>(mut diag_hess: D, mut lower_hess: L, mut diag_value: DF, mut lower_value: LF)
        where
            D: FnMut(usize) -> DH,
            L: FnMut(usize) -> LH,
            DF: FnMut(usize, usize, (usize, usize), usize, &mut DH),
            LF: FnMut(usize, (usize, usize), (usize, usize), &mut LH),
    {
        let mut triplet_idx = 0; // Triplet index for the edge. there should be 69 in total.

        // Diagonal part
        for vtx in 0..4 {
            let mut i = 0; // x,y,z component counter.
            let mut h = diag_hess(vtx);
            for row in 0..3 {
                for col in 0..row + 1 {
                    diag_value(triplet_idx, vtx, (row, col), i, &mut h);
                    triplet_idx += 1;
                    i += 1;
                }
            }
        }
        assert_eq!(triplet_idx, NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);

        // Reset index since this function should be writing to disjoint arrays anyways
        triplet_idx = 0;

        // Lower triangular off-diagonal part
        let mut vtx = 0; // Vertex counter (only lower triangular vertices are counted).
        for row_vtx in 1..4 {
            for col_vtx in 0..row_vtx {
                if row_vtx + col_vtx == 5 { // There is a known zero block at row: 3, col: 2
                    continue;
                }
                let mut h = lower_hess(vtx);
                for row in 0..3 {
                    for col in 0..3 {
                        lower_value(triplet_idx, (row_vtx, col_vtx), (row, col), &mut h);
                        triplet_idx += 1;
                    }
                }
                vtx += 1;
            }
        }

        assert_eq!(triplet_idx, NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE - NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
    }

    #[inline]
    fn proj_angle<T: Real>(angle: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let mut out = (angle + pi) % two_pi - pi;
        if out < -pi {
            out += two_pi;
        }
        out
    }

    fn incremental_angle<T: Real>(angle: T, prev_angle: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let prev_angle_proj = Self::proj_angle(prev_angle);
        let mut angle_diff = angle - prev_angle_proj;
        if angle_diff < -pi {
            angle_diff += two_pi;
        } else if angle_diff > pi {
            angle_diff -= two_pi;
        }
        prev_angle + angle_diff
    }
}

/// Define a hyperelastic energy model for `TriMeshShell`s.
impl<T: Real, E: TriEnergy<T>> Energy<T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy(&self, x0: &[T], x1: &[T]) -> T {
        let TriMeshShell {
            ref trimesh,
            material,
            ref interior_edges,
            ref interior_edge_angles,
            ref interior_edge_ref_angles,
            ref interior_edge_ref_length,
            ref interior_edge_bending_stiffness,
        } = *self.0;

        let damping = material.scaled_damping();

        let pos1 = Chunked3::from_flat(x1).into_arrays();
        let pos0 = Chunked3::from_flat(x0).into_arrays();

        // Bending energy
        let bending: T = zip!(
            interior_edges.iter(),
            interior_edge_angles.iter(),
            interior_edge_ref_angles.iter(),
            interior_edge_ref_length.iter(),
            interior_edge_bending_stiffness.iter(),
        ).map(|(e, &prev_theta, &ref_theta, &ref_length, &k)| {
            let mut theta = e.edge_angle(pos1, trimesh.faces());
            let prev_theta = T::from(prev_theta).unwrap();
            theta = Self::incremental_angle(theta, prev_theta);
            let theta_strain = theta - T::from(ref_theta).unwrap();
            T::from(0.5 * ref_length * k).unwrap() * theta_strain * theta_strain
        }).sum();

        // Membrane energy
        let membrane: T = zip!(
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
        .sum();

        membrane + bending
    }
}

impl<T: Real, E: TriEnergy<T>> EnergyGradient<T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, x0: &[T], x1: &[T], grad_f: &mut [T]) {
        let TriMeshShell {
            ref trimesh,
            material,
            ref interior_edges,
            ref interior_edge_angles,
            ref interior_edge_ref_angles,
            ref interior_edge_ref_length,
            ref interior_edge_bending_stiffness,
            ..
        } = *self.0;

        let damping = material.scaled_damping();

        debug_assert_eq!(grad_f.len(), x0.len());
        debug_assert_eq!(grad_f.len(), x1.len());

        let pos1 = Chunked3::from_flat(x1).into_arrays();
        let pos0 = Chunked3::from_flat(x0).into_arrays();

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Gradient of bending energy.
        for (e, &prev_theta, &ref_theta, &ref_length, &k) in zip!(
            interior_edges.iter(),
            interior_edge_angles.iter(),
            interior_edge_ref_angles.iter(),
            interior_edge_ref_length.iter(),
            interior_edge_bending_stiffness.iter(),
        ) {
            // Compute energy derivative with respect to theta.
            let mut theta = e.edge_angle(pos1, trimesh.faces());
            let prev_theta = T::from(prev_theta).unwrap();
            theta = Self::incremental_angle(theta, prev_theta);
            let dE_dTh = T::from(ref_length * k).unwrap() * (theta - T::from(ref_theta).unwrap());

            // Theta derivative with respect to x.
            let dTh_dx = e.edge_angle_gradient(pos1, trimesh.faces());

            // Distribute gradient.
            let edge_verts = e.edge_verts(trimesh.faces());
            let tangent_verts = e.tangent_verts(trimesh.faces());
            gradient[edge_verts[0]] += Vector3::new(dTh_dx[0]) * dE_dTh;
            gradient[edge_verts[1]] += Vector3::new(dTh_dx[1]) * dE_dTh;
            gradient[tangent_verts[0]] += Vector3::new(dTh_dx[2]) * dE_dTh;
            gradient[tangent_verts[1]] += Vector3::new(dTh_dx[3]) * dE_dTh;
        }

        // Gradient of membrane energy.
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
                gradient[face[i]] += dH[i] * damping;
                gradient[face[2]] -= dH[i] * damping;
            }
        }
    }
}

impl<E> EnergyHessianTopology for TriMeshElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        NUM_HESSIAN_TRIPLETS_PER_TRI * self.0.trimesh.num_faces()
        + NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE * self.0.interior_edges.len()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let shell = &self.0;
        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * shell.trimesh.num_faces();

        // Membrane Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each tri.
            let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                reinterpret_mut_slice(&mut rows[..tri_entries]);
            let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                reinterpret_mut_slice(&mut cols[..tri_entries]);

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(shell.trimesh.faces().par_iter());

            hess_iter.for_each(|((tri_hess_rows, tri_hess_cols), face)| {
                Self::tri_hessian_for_each(
                    |_, _| (),
                    |i, (n, k), (row, col), _| {
                        let mut global_row = 3 * face[n] + row;
                        let mut global_col = 3 * face[k] + col;
                        if face[n] < face[k] {
                            // In the upper triangular part of the global matrix, transpose
                            // This is necessary because even though local matrix may be lower
                            // triangular, its global configuration may not be.
                            std::mem::swap(&mut global_row, &mut global_col);
                        }
                        tri_hess_rows[i] = I::from_usize(global_row + offset.row).unwrap();
                        tri_hess_cols[i] = I::from_usize(global_col + offset.col).unwrap();
                    },
                );
            });
        }

        // Bending Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each edge.
            let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] = reinterpret_mut_slice(&mut rows[tri_entries..]);
            let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] = reinterpret_mut_slice(&mut cols[tri_entries..]);

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(shell.interior_edges.par_iter());

            hess_iter.for_each(|((edge_hess_rows, edge_hess_cols), edge)| {
                let x = edge.verts(shell.trimesh.faces());
                let (diag_rows, lower_rows) = edge_hess_rows.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
                let (diag_cols, lower_cols) = edge_hess_cols.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
                Self::edge_hessian_for_each(
                    |_| (),
                    |_| (),
                    |triplet_idx, vtx, (row, col), _, _| {
                        let global_row = 3 * x[vtx] + row;
                        let global_col = 3 * x[vtx] + col;
                        diag_rows[triplet_idx] = I::from_usize(global_row + offset.row).unwrap();
                        diag_cols[triplet_idx] = I::from_usize(global_col + offset.col).unwrap();
                    },
                    |triplet_idx, (row_vtx, col_vtx), (row, col), _| {
                        let mut global_row = 3 * x[row_vtx] + row;
                        let mut global_col = 3 * x[col_vtx] + col;
                        if x[row_vtx] < x[col_vtx] {
                            // In the upper triangular part of the global matrix, transpose
                            std::mem::swap(&mut global_row, &mut global_col);
                        }
                        lower_rows[triplet_idx] = I::from_usize(global_row + offset.row).unwrap();
                        lower_cols[triplet_idx] = I::from_usize(global_col + offset.col).unwrap();
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

        let shell = &self.0;
        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * shell.trimesh.num_faces();

        // Membrane Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each triangle.
            let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                reinterpret_mut_slice(&mut indices[..tri_entries]);

            let hess_iter = hess_chunks.par_iter_mut().zip(shell.trimesh.faces().par_iter());

            hess_iter.for_each(|(tri_hess, face)| {
                Self::tri_hessian_for_each(
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

        // Bending Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each edge.
            let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] =
                reinterpret_mut_slice(&mut indices[tri_entries..]);

            let hess_iter = hess_chunks.par_iter_mut().zip(shell.interior_edges.par_iter());

            hess_iter.for_each(|(edge_hess, edge)| {
                let x = edge.verts(shell.trimesh.faces());
                let (diag_hess, lower_hess) = edge_hess.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
                Self::edge_hessian_for_each(
                    |_| (),
                    |_| (),
                    |triplet_idx, vtx, (row, col), _, _| {
                        let global_row = 3 * x[vtx] + row;
                        let global_col = 3 * x[vtx] + col;
                        diag_hess[triplet_idx] = MatrixElementIndex {
                            row: global_row + offset.row,
                            col: global_col + offset.col,
                        }
                    },
                    |triplet_idx, (row_vtx, col_vtx), (row, col), _| {
                        let mut global_row = 3 * x[row_vtx] + row;
                        let mut global_col = 3 * x[col_vtx] + col;
                        if x[row_vtx] < x[col_vtx] {
                            // In the upper triangular part of the global matrix, transpose
                            std::mem::swap(&mut global_row, &mut global_col);
                        }
                        lower_hess[triplet_idx] = MatrixElementIndex {
                            row: global_row + offset.row,
                            col: global_col + offset.col,
                        }
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
            ref interior_edges,
            ref interior_edge_ref_length,
            ref interior_edge_bending_stiffness,
            ..
        } = *self.0;

        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * trimesh.num_faces();
        let damping = material.scaled_damping();

        let pos1 = Chunked3::from_flat(x1).into_arrays();

        // Membrane Hessian
        {
            // Break up the hessian triplets into chunks of elements for each triangle.
            let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TRI]] = reinterpret_mut_slice(&mut values[..tri_entries]);

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
                // Make deformed triangle.
                let tri_x1 = Triangle::from_indexed_slice(face, pos1);

                let Dx = Matrix2x3::new(tri_x1.shape_matrix());

                let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                let area = T::from(area).unwrap();
                let lambda = T::from(lambda).unwrap();
                let mu = T::from(mu).unwrap();

                let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

                let factor = T::from(1.0 + damping).unwrap() * scale;

                let local_hessians = tri_energy.energy_hessian();

                Self::tri_hessian_for_each(
                    |n, k| local_hessians[n][k] * factor,
                    |i, _, (row, col), h| tri_hess[i] = h[row][col],
                );
            });
        }

        // Bending Hessian
        {
            // Break up the hessian triplets into chunks of elements for each triangle.
            let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] =
                reinterpret_mut_slice(&mut values[tri_entries..]);

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                interior_edges.par_iter(),
                interior_edge_ref_length.par_iter(),
                interior_edge_bending_stiffness.par_iter(),
            ));

            hess_iter.for_each(|(edge_hess, (e, &ref_length, &k))| {
                let local_hessians = e.edge_angle_hessian(pos1, trimesh.faces());
                let (diag_hess, lower_hess) = edge_hess.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);

                Self::edge_hessian_for_each(
                    |vtx| Vector6::new(local_hessians.0[vtx]) * T::from(k * ref_length).unwrap(),
                    |vtx| Matrix3::new(local_hessians.1[vtx]) * T::from(k * ref_length).unwrap(),
                    |triplet_idx, _, _, i, h| {
                        diag_hess[triplet_idx] = h[i];
                    },
                    |triplet_idx, _, (row, col), h| {
                        lower_hess[triplet_idx] = h[row][col];
                    }
                );
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use geo::mesh::VertexPositions;

    use crate::energy_models::elasticity::test_utils::*;
    use crate::energy_models::test_utils::*;
    use crate::objects::TriMeshShell;

    use super::*;

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
            .map(|trimesh| {
                // Prepare attributes relevant for elasticity computations.
                let mut shell = TriMeshShell::new(trimesh, material);
                shell.init_deformable_vertex_attributes().unwrap();
                shell.init_deformable_attributes().unwrap();
                shell.init_elasticity_attributes().unwrap();
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
