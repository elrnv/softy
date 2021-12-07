#![allow(clippy::just_underscores_and_digits)]
//! Neo-Hookean energy model for triangle meshes.

use num_traits::FromPrimitive;
use rayon::prelude::*;
use unroll::unroll_for_loops;

use crate::TriMesh;
use flatk::zip;
use geo::attrib::Attrib;
use geo::mesh::topology::*;
use geo::ops::*;
use geo::prim::Triangle;
use rayon::iter::Either;
use tensr::*;

use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::shell::*;
use crate::objects::*;
use crate::Real;

use super::{tri_nh::*, TriEnergy};

/// A possibly non-linear elastic energy for triangle meshes.
///
/// This type wraps a `TriMeshShell` to provide an interfce for computing a membrane and bending
/// elastic energies. `E` specifies the per element membrane energy model.
pub struct TriMeshElasticity<'a, E> {
    trimesh: &'a TriMesh,
    damping: f32,
    interior_edges: &'a [InteriorEdge],
    interior_edge_ref_angles: &'a [f64],
    interior_edge_angles: &'a [f64],
    interior_edge_ref_length: &'a [f64],
    interior_edge_bending_stiffness: &'a [f64],
    energy: std::marker::PhantomData<E>,
}

/// NeoHookean elasticity model.
pub type TriMeshNeoHookean<'a, T> = TriMeshElasticity<'a, NeoHookeanTriEnergy<T>>;

const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 45; // There are 3*6 + 3*9 = 45 triplets per triangle

// There are 4*6 = 24 triplets on the diagonal blocks per interior edge.
const NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG: usize = 24;
// There are 4*6 + 3*3*6 = 78 triplets per interior edge in total.
const NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE: usize = 78;
// There are 4*3 = 12 triplets on the diagonal per interior edge.
const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_INTERIOR_EDGE: usize = 12;
// There are 3*3 = 9 triplets on the diagonal per tri.
const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TRI: usize = 9;

impl<'a, E> TriMeshElasticity<'a, E> {
    /// Construct a new elasticity model from the given `TriMeshShell`. Since `TriMeshShell`s can
    /// be fixed or rigid, this may produce a `None` energy which effectively yields zeros for
    /// all energy functions.
    pub fn new(shell: &'a TriMeshShell) -> Option<Self> {
        match &shell.data {
            ShellData::Soft {
                material,
                interior_edges,
                interior_edge_ref_angles,
                interior_edge_angles,
                interior_edge_ref_length,
                interior_edge_bending_stiffness,
            } => Some(TriMeshElasticity {
                trimesh: &shell.trimesh,
                damping: material.damping(),
                interior_edges: interior_edges.as_slice(),
                interior_edge_ref_angles: interior_edge_ref_angles.as_slice(),
                interior_edge_angles: interior_edge_angles.as_slice(),
                interior_edge_ref_length: interior_edge_ref_length.as_slice(),
                interior_edge_bending_stiffness: interior_edge_bending_stiffness.as_slice(),
                energy: std::marker::PhantomData,
            }),
            _ => None,
        }
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
    #[unroll_for_loops]
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
    /// Each edge neighborhood is described in [`InteriorEdge`].
    #[inline]
    #[unroll_for_loops]
    fn edge_hessian_for_each<D, L, DH, LH, DF, LF>(
        mut diag_hess: D,
        mut lower_hess: L,
        mut diag_value: DF,
        mut lower_value: LF,
    ) where
        D: FnMut(usize) -> DH,
        L: FnMut((usize, usize), usize) -> LH,
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
                let mut h = lower_hess((row_vtx, col_vtx), vtx);
                for row in 0..3 {
                    for col in 0..3 {
                        lower_value(triplet_idx, (row_vtx, col_vtx), (row, col), &mut h);
                        triplet_idx += 1;
                    }
                }
                vtx += 1;
            }
        }

        assert_eq!(
            triplet_idx,
            NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE - NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG
        );
    }
}

/// Define a hyperelastic energy model for `TriMeshShell`s.
impl<T: Real, E: TriEnergy<T>> Energy<T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy(&self, x0: &[T], x1: &[T]) -> T {
        let TriMeshElasticity {
            ref trimesh,
            damping,
            ref interior_edges,
            ref interior_edge_angles,
            ref interior_edge_ref_angles,
            ref interior_edge_ref_length,
            ref interior_edge_bending_stiffness,
            ..
        } = *self;

        let pos1 = Chunked3::from_flat(x1).into_arrays();
        let pos0 = Chunked3::from_flat(x0).into_arrays();

        // Membrane energy
        let membrane: T = zip!(
            Either::from(
                trimesh
                    .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                    .map(|i| i.cloned())
                    .map_err(|_| std::iter::repeat(0.0f32))
            ),
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
        .map(|(density, &area, &DX_inv, face, &lambda, &mu)| {
            let tri_x1 = Triangle::from_indexed_slice(face, pos1);
            let tri_x0 = Triangle::from_indexed_slice(face, pos0);
            let tri_dx = Triangle::new(
                (*tri_x1.as_array().as_tensor() - tri_x0.into_array().into_tensor()).into(),
            );
            let Dx = Matrix2x3::new(tri_x1.shape_matrix());
            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let area = T::from(area).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let half = T::from(0.5).unwrap();
            let damping = T::from(damping).unwrap();
            let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);
            let dF = tri_energy.deformation_gradient_differential(&tri_dx);
            let dFTdF_tr = dF[0].dot(dF[0]) + dF[1].dot(dF[1]); // trace
            tri_energy.energy() + {
                // damping (viscosity)
                // Note: damping is already scaled by dt
                if density != 0.0 {
                    area * dFTdF_tr * half * T::from(density).unwrap() * damping
                } else {
                    T::zero()
                }
                //let dH = tri_energy.energy_hessian_product_transpose(&tri_dx);
                //half * damping * (dH[0].dot(Vector3::new(tri_dx.0.into()))
                //    + dH[1].dot(Vector3::new(tri_dx.1.into()))
                //    - (dH * Vector3::new(tri_dx.2.into())).sum())
            }
        })
        .sum();

        // Bending energy
        let bending: T = zip!(
            interior_edges.iter(),
            interior_edge_angles.iter(),
            interior_edge_ref_angles.iter(),
            interior_edge_ref_length.iter(),
            interior_edge_bending_stiffness.iter(),
        )
        .map(|(&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness)| {
            DiscreteShellBendingEnergy {
                cur_pos: pos1,
                faces: trimesh.faces(),
                edge,
                prev_theta: T::from(prev_theta).unwrap(),
                ref_theta: T::from(ref_theta).unwrap(),
                ref_shape: T::from(ref_shape).unwrap(),
                stiffness: T::from(stiffness).unwrap(),
            }
            .energy()
        })
        .sum();

        membrane + bending
    }
}

impl<X: Real, T: Real, E: TriEnergy<T>> EnergyGradient<X, T> for TriMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    fn add_energy_gradient(&self, x0: &[X], x1: &[T], grad_f: &mut [T]) {
        let TriMeshElasticity {
            ref trimesh,
            damping,
            ref interior_edges,
            ref interior_edge_angles,
            ref interior_edge_ref_angles,
            ref interior_edge_ref_length,
            ref interior_edge_bending_stiffness,
            ..
        } = *self;

        debug_assert_eq!(grad_f.len(), x0.len());
        debug_assert_eq!(grad_f.len(), x1.len());

        let pos1 = Chunked3::from_flat(x1).into_arrays();
        let pos0 = Chunked3::from_flat(x0).into_arrays();

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        // Gradient of membrane energy.
        for (density, &area, &DX_inv, face, &lambda, &mu) in zip!(
            Either::from(
                trimesh
                    .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                    .map(|i| i.cloned())
                    .map_err(|_| std::iter::repeat(0.0f32))
            ),
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
                (*tri_x1.as_array().as_tensor()
                    - tri_x0.into_array().into_tensor().cast_inner::<T>())
                .into(),
            );

            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let area = T::from(area).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();

            let Dx = Matrix2x3::new(tri_x1.shape_matrix());

            let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

            let grad = tri_energy.energy_gradient();

            for i in 0..3 {
                gradient[face[i]] += grad[i];
            }

            // Damping
            // Note: damping is already scaled by dt
            if density != 0.0 {
                let damping = T::from(damping).unwrap();
                let density = T::from(density).unwrap();

                let dF = tri_energy.deformation_gradient_differential(&tri_dx);
                let damp = DX_inv.transpose() * dF * area * density * damping;
                for i in 0..2 {
                    gradient[face[i]] += damp[i];
                    gradient[face[2]] -= damp[i];
                }

                //let dH = tri_energy.energy_hessian_product_transpose(&tri_dx);
                //for i in 0..2 {
                //    gradient[face[i]] += dH[i] * damping;
                //    gradient[face[2]] -= dH[i] * damping;
                //}
            }
        }

        // Gradient of bending energy.
        for (&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness) in zip!(
            interior_edges.iter(),
            interior_edge_angles.iter(),
            interior_edge_ref_angles.iter(),
            interior_edge_ref_length.iter(),
            interior_edge_bending_stiffness.iter(),
        ) {
            let dW_dx = DiscreteShellBendingEnergy {
                cur_pos: pos1,
                faces: trimesh.faces(),
                edge,
                prev_theta: T::from(prev_theta).unwrap(),
                ref_theta: T::from(ref_theta).unwrap(),
                ref_shape: T::from(ref_shape).unwrap(),
                stiffness: T::from(stiffness).unwrap(),
            }
            .energy_gradient();
            // Distribute gradient.
            let verts = edge.verts(|f, i| trimesh.face_to_vertex(f, i).unwrap().into());
            for i in 0..4 {
                gradient[verts[i]] += dW_dx[i];
            }
        }
    }
}

impl<E: Send + Sync> EnergyHessianTopology for TriMeshElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        NUM_HESSIAN_TRIPLETS_PER_TRI * self.trimesh.num_faces()
            + NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE * self.interior_edges.len()
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TRI * self.trimesh.num_faces()
            + NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_INTERIOR_EDGE * self.interior_edges.len()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * self.trimesh.num_faces();

        // Membrane Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each tri.
            let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut rows[..tri_entries]) };
            let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut cols[..tri_entries]) };

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(self.trimesh.faces().par_iter());

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
            let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut rows[tri_entries..]) };
            let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut cols[tri_entries..]) };

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(self.interior_edges.par_iter());

            hess_iter.for_each(|((edge_hess_rows, edge_hess_cols), edge)| {
                let verts = edge.verts(|f, i| self.trimesh.face_to_vertex(f, i).unwrap().into());
                let (diag_rows, lower_rows) =
                    edge_hess_rows.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
                let (diag_cols, lower_cols) =
                    edge_hess_cols.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
                Self::edge_hessian_for_each(
                    |_| (),
                    |_, _| (),
                    |triplet_idx, vtx, (row, col), _, _| {
                        let global_row = 3 * verts[vtx] + row;
                        let global_col = 3 * verts[vtx] + col;
                        diag_rows[triplet_idx] = I::from_usize(global_row + offset.row).unwrap();
                        diag_cols[triplet_idx] = I::from_usize(global_col + offset.col).unwrap();
                    },
                    |triplet_idx, (row_vtx, col_vtx), (row, col), _| {
                        let mut global_row = 3 * verts[row_vtx] + row;
                        let mut global_col = 3 * verts[col_vtx] + col;
                        if verts[row_vtx] < verts[col_vtx] {
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

        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * self.trimesh.num_faces();

        // Membrane Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each triangle.
            let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut indices[..tri_entries]) };

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(self.trimesh.faces().par_iter());

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
                unsafe { reinterpret::reinterpret_mut_slice(&mut indices[tri_entries..]) };

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(self.interior_edges.par_iter());

            hess_iter.for_each(|(edge_hess, edge)| {
                let verts = edge.verts(|f, i| self.trimesh.face_to_vertex(f, i).unwrap().into());
                let (diag_hess, lower_hess) =
                    edge_hess.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);
                Self::edge_hessian_for_each(
                    |_| (),
                    |_, _| (),
                    |triplet_idx, vtx, (row, col), _, _| {
                        let global_row = 3 * verts[vtx] + row;
                        let global_col = 3 * verts[vtx] + col;
                        diag_hess[triplet_idx] = MatrixElementIndex {
                            row: global_row + offset.row,
                            col: global_col + offset.col,
                        }
                    },
                    |triplet_idx, (row_vtx, col_vtx), (row, col), _| {
                        let mut global_row = 3 * verts[row_vtx] + row;
                        let mut global_col = 3 * verts[col_vtx] + col;
                        if verts[row_vtx] < verts[col_vtx] {
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

impl<T: Real + Send + Sync, E: TriEnergy<T> + Send + Sync> EnergyHessian<T>
    for TriMeshElasticity<'_, E>
{
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _: &[T], x1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let TriMeshElasticity {
            ref trimesh,
            damping,
            ref interior_edges,
            ref interior_edge_angles,
            ref interior_edge_ref_angles,
            ref interior_edge_ref_length,
            ref interior_edge_bending_stiffness,
            ..
        } = *self;

        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * trimesh.num_faces();

        let pos1 = Chunked3::from_flat(x1).into_arrays();

        // Membrane Hessian
        {
            // Break up the hessian triplets into chunks of elements for each triangle.
            let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut values[..tri_entries]) };

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                Either::from(
                    trimesh
                        .attrib_as_slice::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                        .map(|slice| slice.par_iter().cloned())
                        .map_err(|_| rayon::iter::repeatn(0.0f32, trimesh.num_faces()))
                ),
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

            hess_iter.for_each(
                |(tri_hess, (density, &area, &DX_inv, face, &lambda, &mu))| {
                    // Make deformed triangle.
                    let tri_x1 = Triangle::from_indexed_slice(face, pos1);

                    let Dx = Matrix2x3::new(tri_x1.shape_matrix());

                    let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                    let area = T::from(area).unwrap();
                    let lambda = T::from(lambda).unwrap();
                    let mu = T::from(mu).unwrap();

                    let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

                    //let factor = T::from(1.0 + damping).unwrap() * scale;
                    let factor = scale;

                    let local_hessians = tri_energy.energy_hessian();

                    // Damping
                    let damping = T::from(damping).unwrap();
                    let density = T::from(density).unwrap();
                    // Note: damping is already scaled by dt
                    let ddF = DX_inv.transpose() * DX_inv * (area * density * damping);
                    let id = Matrix3::identity();

                    Self::tri_hessian_for_each(
                        |n, k| {
                            (local_hessians[n][k]
                                + id * if n == 2 && k == 2 {
                                    ddF.sum_inner()
                                } else if k == 2 {
                                    -ddF[n].sum()
                                } else if n == 2 {
                                    -ddF[k].sum() // ddF should be symmetric
                                } else {
                                    ddF[n][k]
                                })
                                * factor
                        },
                        |i, _, (row, col), h| tri_hess[i] = h[row][col],
                    );
                },
            );
        }

        // Bending Hessian
        {
            // Break up the Hessian triplets into chunks of elements for each triangle.
            let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut values[tri_entries..]) };

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                interior_edges.par_iter(),
                interior_edge_angles.par_iter(),
                interior_edge_ref_angles.par_iter(),
                interior_edge_ref_length.par_iter(),
                interior_edge_bending_stiffness.par_iter(),
            ));

            hess_iter.for_each(
                |(edge_hess, (&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness))| {
                    let (dth_dx, d2w_dth2, dw_dth, d2th_dx2) = DiscreteShellBendingEnergy {
                        cur_pos: pos1,
                        faces: trimesh.faces(),
                        edge,
                        prev_theta: T::from(prev_theta).unwrap(),
                        ref_theta: T::from(ref_theta).unwrap(),
                        ref_shape: T::from(ref_shape).unwrap(),
                        stiffness: T::from(stiffness).unwrap(),
                    }
                    .energy_hessian();

                    let (diag_hess, lower_hess) =
                        edge_hess.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_INTERIOR_EDGE_DIAG);

                    Self::edge_hessian_for_each(
                        |vtx| {
                            (dth_dx[vtx] * (dth_dx[vtx].transpose() * d2w_dth2))
                                .lower_triangular_vec()
                                + d2th_dx2.0[vtx].into_tensor() * dw_dth
                        },
                        |(row_vtx, col_vtx), vtx| {
                            let mut out =
                                dth_dx[row_vtx] * (dth_dx[col_vtx].transpose() * d2w_dth2);
                            if vtx != 5 {
                                // One off-diagonal block is known to be zero.
                                out += Matrix3::new(d2th_dx2.1[vtx]) * dw_dth;
                            }
                            out
                        },
                        |triplet_idx, _, _, i, h| {
                            diag_hess[triplet_idx] = scale * h[i];
                        },
                        |triplet_idx, _, (row, col), h| {
                            lower_hess[triplet_idx] = scale * h[row][col];
                        },
                    );
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use geo::mesh::VertexPositions;

    use crate::energy_models::elasticity::test_utils::*;
    use crate::energy_models::test_utils::*;
    use crate::objects::shell::TriMeshShell;

    use super::*;

    fn membrane_only_material() -> SoftShellMaterial {
        SoftShellMaterial::new(0)
            .with_elasticity(ElasticityParameters {
                lambda: 5.4,
                mu: 263.1,
                model: ElasticityModel::NeoHookean,
            })
            .with_density(1.0)
            .with_damping(1.0, 0.01)
    }

    fn bend_only_material() -> SoftShellMaterial {
        SoftShellMaterial::new(0)
            .with_elasticity(ElasticityParameters {
                lambda: 0.0,
                mu: 0.0,
                model: ElasticityModel::NeoHookean,
            })
            .with_density(1.0)
            .with_damping(1.0, 0.01)
            .with_bending_stiffness(2.0)
    }

    fn material() -> SoftShellMaterial {
        membrane_only_material().with_bending_stiffness(2.0)
    }

    fn test_shells() -> Vec<TriMeshShell> {
        test_trimeshes()
            .into_iter()
            .map(|trimesh| {
                let mut shell = TriMeshShell::soft(trimesh, membrane_only_material());
                shell.init_deformable_vertex_attributes().unwrap();
                shell.init_deformable_attributes().unwrap();
                shell.init_elasticity_attributes().unwrap();
                shell.init_density_attribute().unwrap();
                shell
            })
            .chain(test_trimeshes().into_iter().map(|trimesh| {
                let mut shell = TriMeshShell::soft(trimesh, bend_only_material());
                shell.init_deformable_vertex_attributes().unwrap();
                shell.init_deformable_attributes().unwrap();
                shell.init_elasticity_attributes().unwrap();
                shell.init_density_attribute().unwrap();
                shell
            }))
            .chain(test_trimeshes().into_iter().map(|trimesh| {
                let mut shell = TriMeshShell::soft(trimesh, material());
                shell.init_deformable_vertex_attributes().unwrap();
                shell.init_deformable_attributes().unwrap();
                shell.init_elasticity_attributes().unwrap();
                shell.init_density_attribute().unwrap();
                shell
            }))
            .collect()
    }

    fn build_energies(
        shells: &[TriMeshShell],
    ) -> Vec<(TriMeshNeoHookean<autodiff::F1>, &[[f64; 3]])> {
        shells
            .iter()
            .map(|shell| {
                (
                    TriMeshNeoHookean::new(shell).unwrap(),
                    shell.trimesh.vertex_positions(),
                )
            })
            .collect()
    }

    #[test]
    fn tri_energy_gradient() {
        tri_energy_gradient_tester::<NeoHookeanTriEnergy<autodiff::F1>>();
    }

    #[test]
    fn tri_energy_hessian() {
        tri_energy_hessian_tester::<NeoHookeanTriEnergy<autodiff::F1>>();
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
