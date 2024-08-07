#![allow(clippy::just_underscores_and_digits)]
//! Neo-Hookean energy model for triangle shells.

use num_traits::FromPrimitive;
use rayon::prelude::*;
use unroll::unroll_for_loops;

use geo::ops::*;
use geo::prim::Triangle;
use tensr::*;

use crate::energy::*;
use crate::matrix::*;
use crate::objects::trishell::*;
use crate::Real;

use super::{tri_nh::*, TriEnergy};

/// A possibly non-linear elastic energy for triangle meshes.
///
/// This type wraps a `TriShell` to provide an interfce for computing a membrane and bending
/// elastic energies. `E` specifies the per element membrane energy model.
pub struct TriShellElasticity<'a, E> {
    shell: &'a TriShell,
    energy: std::marker::PhantomData<E>,
}

/// NeoHookean elasticity model.
pub type TriShellNeoHookean<'a, T> = TriShellElasticity<'a, NeoHookeanTriEnergy<T>>;

const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 45; // There are 3*6 + 3*9 = 45 triplets per triangle

// There are 4*6 = 24 triplets on the diagonal blocks per interior edge.
const NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG: usize = 24;
// There are 4*6 + 3*3*6 = 78 triplets per interior edge in total.
const NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL: usize = 78;
// There are 4*3 = 12 triplets on the diagonal per interior edge.
const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_DIHEDRAL: usize = 12;
// There are 3*3 = 9 triplets on the diagonal per tri.
const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TRI: usize = 9;

impl<'a, E> TriShellElasticity<'a, E> {
    /// Construct a new elasticity model from the given `TriShell`.
    pub fn new(shell: &'a TriShell) -> Self {
        TriShellElasticity {
            shell,
            energy: std::marker::PhantomData,
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
            let mut i = 0; // lower triangular entry counter.
            let mut h = diag_hess(vtx);
            for row in 0..3 {
                for col in 0..row + 1 {
                    diag_value(triplet_idx, vtx, (row, col), i, &mut h);
                    triplet_idx += 1;
                    i += 1;
                }
            }
        }
        assert_eq!(triplet_idx, NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);

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
            NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL - NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG
        );
    }
}

/// Define a hyperelastic energy model for `TriMeshShell`s.
impl<T: Real, E: TriEnergy<T>> Energy<T> for TriShellElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy(&self, x: &[T], v: &[T], dqdv: T) -> T {
        let tri_elems = &self.shell.triangle_elements;

        let x = Chunked3::from_flat(x).into_arrays();
        let v = Chunked3::from_flat(v).into_arrays();

        let dqdv2 = dqdv * dqdv;

        // Membrane energy
        let membrane: T = zip!(
            tri_elems.damping.iter().map(|&x| T::from(x).unwrap()),
            // tri_elems.density.iter().map(|&x| T::from(x).unwrap()),
            tri_elems.ref_area.iter().map(|&x| T::from(x).unwrap()),
            tri_elems.ref_tri_shape_mtx_inv.iter(),
            tri_elems.triangles.iter(),
            tri_elems.lambda.iter().map(|&x| T::from(x).unwrap()),
            tri_elems.mu.iter().map(|&x| T::from(x).unwrap()),
        )
        .map(|(damping, area, &DX_inv, face, lambda, mu)| {
            let tri_x = Triangle::from_indexed_slice(face, x);
            let tri_v = Triangle::from_indexed_slice(face, v);
            let Dx = Matrix2x3::new(tri_x.shape_matrix());
            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let half = T::from(0.5).unwrap();
            let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);
            let dF = tri_energy.deformation_gradient_differential(&tri_v);
            let dFTdF_tr = dF[0].dot(dF[0]) + dF[1].dot(dF[1]); // trace
            tri_energy.energy() + {
                // damping (viscosity)
                mu * area * dFTdF_tr * half * damping * dqdv2
                //let dH = tri_energy.energy_hessian_product_transpose(&tri_dx);
                //half * damping * (dH[0].dot(Vector3::new(tri_dx.0.into()))
                //    + dH[1].dot(Vector3::new(tri_dx.1.into()))
                //    - (dH * Vector3::new(tri_dx.2.into())).sum())
            }
        })
        .sum();

        let di_elems = &self.shell.dihedral_elements;

        // Bending energy
        let bending: T = zip!(
            di_elems.dihedrals.iter(),
            di_elems.angles.iter(),
            di_elems.ref_angles.iter(),
            di_elems.ref_length.iter(),
            di_elems.bending_stiffness.iter(),
        )
        .map(|(&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness)| {
            DiscreteShellBendingEnergy {
                cur_pos: x,
                faces: di_elems.triangles.as_slice(),
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

impl<T: Real, E: TriEnergy<T>> EnergyGradient<T, T> for TriShellElasticity<'_, E> {
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    fn add_energy_gradient(&self, x: &[T], v: &[T], grad_f: &mut [T], dqdv: T) {
        let tri_elems = &self.shell.triangle_elements;

        debug_assert_eq!(grad_f.len(), x.len());
        debug_assert_eq!(grad_f.len(), v.len());

        let pos = Chunked3::from_flat(x).into_arrays();
        let vel = Chunked3::from_flat(v).into_arrays();

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        // Gradient of membrane energy.
        for (&damping, &area, &DX_inv, face, &lambda, &mu) in zip!(
            tri_elems.damping.iter(),
            // tri_elems.density.iter(),
            tri_elems.ref_area.iter(),
            tri_elems.ref_tri_shape_mtx_inv.iter(),
            tri_elems.triangles.iter(),
            tri_elems.lambda.iter(),
            tri_elems.mu.iter(),
        ) {
            // Make deformed tri.
            let tri_x = Triangle::from_indexed_slice(face, pos);
            // Make tri displacement.
            let tri_v = Triangle::from_indexed_slice(face, vel);

            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let area = T::from(area).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();

            let Dx = Matrix2x3::new(tri_x.shape_matrix());

            let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

            let grad = tri_energy.energy_gradient();

            for i in 0..3 {
                gradient[face[i]] += grad[i];
            }

            // Damping
            // if density != 0.0 {
            let damping = T::from(damping).unwrap();
            // let density = T::from(density).unwrap();

            let dF = tri_energy.deformation_gradient_differential(&tri_v) * dqdv;
            let damp = DX_inv.transpose() * dF * (mu * area * damping);
            for i in 0..2 {
                gradient[face[i]] += damp[i];
                gradient[face[2]] -= damp[i];
            }

            //let dH = tri_energy.energy_hessian_product_transpose(&tri_dx);
            //for i in 0..2 {
            //    gradient[face[i]] += dH[i] * damping;
            //    gradient[face[2]] -= dH[i] * damping;
            //}
            // }
        }

        let di_elems = &self.shell.dihedral_elements;

        // let mut gradient_dbg = vec![Vector3::zeros(); gradient.len()];

        // Gradient of bending energy.
        for (&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness) in zip!(
            di_elems.dihedrals.iter(),
            di_elems.angles.iter(),
            di_elems.ref_angles.iter(),
            di_elems.ref_length.iter(),
            di_elems.bending_stiffness.iter(),
        ) {
            let dihedral = DiscreteShellBendingEnergy {
                cur_pos: pos,
                faces: di_elems.triangles.as_slice(),
                edge,
                prev_theta: T::from(prev_theta).unwrap(),
                ref_theta: T::from(ref_theta).unwrap(),
                ref_shape: T::from(ref_shape).unwrap(),
                stiffness: T::from(stiffness).unwrap(),
            };
            let dW_dx = dihedral.energy_gradient();
            // Distribute gradient.
            let verts = edge.verts(|f, i| di_elems.triangles[f][i]);
            for i in 0..4 {
                gradient[verts[i]] += dW_dx[i];
                // eprintln!("g({:?}) = {:?}", verts[i], dW_dx[i]);
                // gradient_dbg[verts[i]] += dW_dx[i];
            }
        }

        // dbg!(gradient_dbg);
    }
}

impl<E: Send + Sync> EnergyHessianTopology for TriShellElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        // dbg!(self.shell.triangle_elements.num_elements());
        // dbg!(self.shell.dihedral_elements.num_elements());
        NUM_HESSIAN_TRIPLETS_PER_TRI * self.shell.triangle_elements.num_elements()
            + NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL * self.shell.dihedral_elements.num_elements()
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TRI * self.shell.triangle_elements.num_elements()
            + NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_DIHEDRAL
                * self.shell.dihedral_elements.num_elements()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        // dbg!(offset);
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tri_entries =
            NUM_HESSIAN_TRIPLETS_PER_TRI * self.shell.triangle_elements.num_elements();

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
                .zip(self.shell.triangle_elements.triangles.par_iter());

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
            let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut rows[tri_entries..]) };
            let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut cols[tri_entries..]) };

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(self.shell.dihedral_elements.dihedrals.par_iter());

            hess_iter.for_each(|((edge_hess_rows, edge_hess_cols), edge)| {
                let verts = edge.verts(|f, i| self.shell.dihedral_elements.triangles[f][i]);
                let (diag_rows, lower_rows) =
                    edge_hess_rows.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);
                let (diag_cols, lower_cols) =
                    edge_hess_cols.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);
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

        let tri_entries =
            NUM_HESSIAN_TRIPLETS_PER_TRI * self.shell.triangle_elements.num_elements();

        // Membrane Hessian indices
        {
            // Break up the hessian indices into chunks of elements for each triangle.
            let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut indices[..tri_entries]) };

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(self.shell.triangle_elements.triangles.par_iter());

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
            // Break up the Hessian indices into chunks of elements for each edge.
            let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut indices[tri_entries..]) };

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(self.shell.dihedral_elements.dihedrals.par_iter());

            hess_iter.for_each(|(edge_hess, edge)| {
                let verts = edge.verts(|f, i| self.shell.dihedral_elements.triangles[f][i]);
                let (diag_hess, lower_hess) =
                    edge_hess.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);
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
    for TriShellElasticity<'_, E>
{
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, x: &[T], _: &[T], scale: T, values: &mut [T], _dqdv: T) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let tri_elems = &self.shell.triangle_elements;
        let tri_entries = NUM_HESSIAN_TRIPLETS_PER_TRI * tri_elems.num_elements();

        let pos = Chunked3::from_flat(x).into_arrays();

        // Membrane Hessian
        {
            // Break up the hessian triplets into chunks of elements for each triangle.
            let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut values[..tri_entries]) };

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                tri_elems.damping.par_iter().map(|&x| T::from(x).unwrap()),
                // tri_elems.density.par_iter().map(|&x| T::from(x).unwrap()),
                tri_elems.ref_area.par_iter().map(|&x| T::from(x).unwrap()),
                tri_elems.ref_tri_shape_mtx_inv.par_iter(),
                tri_elems.triangles.par_iter(),
                tri_elems.lambda.par_iter().map(|&x| T::from(x).unwrap()),
                tri_elems.mu.par_iter().map(|&x| T::from(x).unwrap()),
            ));

            hess_iter.for_each(|(tri_hess, (damping, area, &DX_inv, face, lambda, mu))| {
                // Make deformed triangle.
                let tri_x = Triangle::from_indexed_slice(face, pos);
                let Dx = Matrix2x3::new(tri_x.shape_matrix());
                let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

                //let factor = T::from(1.0 + damping).unwrap() * scale;
                let factor = scale;
                let local_hessians = tri_energy.energy_hessian();

                // Damping
                let ddF = DX_inv.transpose() * DX_inv * (area * mu * damping);
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
            });
        }

        let di_elems = &self.shell.dihedral_elements;

        // Bending Hessian
        {
            // let len = values.len() - tri_entries;
            // Break up the Hessian triplets into chunks of elements for each triangle.
            let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL]] =
                unsafe { reinterpret::reinterpret_mut_slice(&mut values[tri_entries..]) };

            let hess_iter = hess_chunks.par_iter_mut().zip(zip!(
                di_elems.dihedrals.par_iter(),
                di_elems.angles.par_iter(),
                di_elems.ref_angles.par_iter(),
                di_elems.ref_length.par_iter(),
                di_elems.bending_stiffness.par_iter(),
            ));

            hess_iter.for_each(
                |(edge_hess, (&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness))| {
                    let (dth_dx, d2w_dth2, dw_dth, d2th_dx2) = DiscreteShellBendingEnergy {
                        cur_pos: pos,
                        faces: di_elems.triangles.as_slice(),
                        edge,
                        prev_theta: T::from(prev_theta).unwrap(),
                        ref_theta: T::from(ref_theta).unwrap(),
                        ref_shape: T::from(ref_shape).unwrap(),
                        stiffness: T::from(stiffness).unwrap(),
                    }
                    .energy_hessian();

                    let (diag_hess, lower_hess) =
                        edge_hess.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);

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

            // let mut rows = vec![0usize; len];
            // let mut cols = vec![0usize; len];
            //
            // let hess_row_chunks_usize: &mut [[usize; NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL]] =
            //     unsafe { reinterpret::reinterpret_mut_slice(&mut rows) };
            // let hess_col_chunks_usize: &mut [[usize; NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL]] =
            //     unsafe { reinterpret::reinterpret_mut_slice(&mut cols) };
            // assert_eq!(hess_col_chunks_usize.len(), hess_row_chunks_usize.len());
            // assert_eq!(hess_col_chunks_usize.len(), hess_chunks.len());
            //
            // let hess_iter = hess_row_chunks_usize
            //     .iter_mut()
            //     .zip(hess_col_chunks_usize.iter_mut())
            //     .zip(self.shell.dihedral_elements.dihedrals.iter());
            //
            // hess_iter.for_each(|((edge_hess_rows, edge_hess_cols), edge)| {
            //     let verts = edge.verts(|f, i| self.shell.dihedral_elements.triangles[f][i]);
            //     let (diag_rows, lower_rows) =
            //         edge_hess_rows.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);
            //     let (diag_cols, lower_cols) =
            //         edge_hess_cols.split_at_mut(NUM_HESSIAN_TRIPLETS_PER_DIHEDRAL_DIAG);
            //     Self::edge_hessian_for_each(
            //         |_| (),
            //         |_, _| (),
            //         |triplet_idx, vtx, (row, col), _, _| {
            //             let global_row = 3 * verts[vtx] + row;
            //             let global_col = 3 * verts[vtx] + col;
            //             diag_rows[triplet_idx] = global_row;
            //             diag_cols[triplet_idx] = global_col;
            //         },
            //         |triplet_idx, (row_vtx, col_vtx), (row, col), _| {
            //             let mut global_row = 3 * verts[row_vtx] + row;
            //             let mut global_col = 3 * verts[col_vtx] + col;
            //             if verts[row_vtx] < verts[col_vtx] {
            //                 // In the upper triangular part of the global matrix, transpose
            //                 std::mem::swap(&mut global_row, &mut global_col);
            //             }
            //             lower_rows[triplet_idx] = global_row;
            //             lower_cols[triplet_idx] = global_col;
            //         },
            //     );
            // });
            //
            // let mut mtx = vec![vec![0.0; pos1.len()*3]; pos1.len()*3];
            // for (v, (row, col)) in hess_chunks.iter().zip(hess_row_chunks_usize.iter().zip(hess_col_chunks_usize.iter())) {
            //     assert_eq!(v.len(), row.len());
            //     assert_eq!(v.len(), col.len());
            //     for (v, (row, col)) in v.iter().zip(row.iter().zip(col.iter())) {
            //         if *row == 2 && *col == 2 {
            //             eprintln!("({}, {}) += {}", row, col, v);
            //         }
            //         mtx[*row][*col] += v.to_f64().unwrap();
            //         // eprintln!("({}, {}) = {}", row, col, v);
            //     }
            //     // eprintln!("");
            // }
            //
            // // eprintln!("dihedral_values = {:?}", values);
            //
            // eprintln!("dihedral_hess = [");
            // for r in mtx.iter() {
            //     for v in r.iter() {
            //         eprint!("{} ", v);
            //     }
            //     eprintln!();
            // }
            // eprintln!("]");
        }
    }

    #[allow(non_snake_case)]
    //#[unroll_for_loops]
    fn add_energy_hessian_diagonal(&self, x: &[T], _: &[T], scale: T, diag: &mut [T], _dqdv: T) {
        let tri_elems = &self.shell.triangle_elements;

        let pos = Chunked3::from_flat(x).into_arrays();

        let diag: &mut [Vector3<T>] = bytemuck::cast_slice_mut(diag);

        // Membrane Hessian
        let membrane_diag = {
            zip!(
                tri_elems.damping.par_iter().map(|&x| T::from(x).unwrap()),
                // tri_elems.density.par_iter().map(|&x| T::from(x).unwrap()),
                tri_elems.ref_area.par_iter().map(|&x| T::from(x).unwrap()),
                tri_elems.ref_tri_shape_mtx_inv.par_iter(),
                tri_elems.triangles.par_iter(),
                tri_elems.lambda.par_iter().map(|&x| T::from(x).unwrap()),
                tri_elems.mu.par_iter().map(|&x| T::from(x).unwrap()),
            )
            .map(|(damping, area, &DX_inv, face, lambda, mu)| {
                // Make deformed triangle.
                let tri_x = Triangle::from_indexed_slice(face, pos);
                let Dx = Matrix2x3::new(tri_x.shape_matrix());
                let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                let tri_energy = E::new(Dx, DX_inv, area, lambda, mu);

                //let factor = T::from(1.0 + damping).unwrap() * scale;
                let factor = scale;
                let local_hessians = tri_energy.energy_hessian();

                let mut diag = [Vector3::zeros(); 3];
                for i in 0..3 {
                    diag[i] =
                        utils::get_diag3(local_hessians[i][i].as_data()).into_tensor() * factor;
                }

                // Damping
                let ddF = DX_inv.transpose() * DX_inv * (area * mu * damping);
                let id = Vector3::from([T::one(); 3]);
                for k in 0..2 {
                    diag[k] += id * ddF[k][k] * factor;
                }
                diag[2] += id * ddF.sum_inner() * factor;

                diag
            })
            .collect::<Vec<_>>()
        };

        let di_elems = &self.shell.dihedral_elements;

        // Bending Hessian
        let bending_diag = {
            zip!(
                di_elems.dihedrals.par_iter(),
                di_elems.angles.par_iter(),
                di_elems.ref_angles.par_iter(),
                di_elems.ref_length.par_iter(),
                di_elems.bending_stiffness.par_iter(),
            )
            .map(|(&edge, &prev_theta, &ref_theta, &ref_shape, &stiffness)| {
                let (dth_dx, d2w_dth2, dw_dth, d2th_dx2) = DiscreteShellBendingEnergy {
                    cur_pos: pos,
                    faces: di_elems.triangles.as_slice(),
                    edge,
                    prev_theta: T::from(prev_theta).unwrap(),
                    ref_theta: T::from(ref_theta).unwrap(),
                    ref_shape: T::from(ref_shape).unwrap(),
                    stiffness: T::from(stiffness).unwrap(),
                }
                .energy_hessian();

                let mut diag = [Vector3::zeros(); 4];

                // Diagonal part
                for vtx in 0..4 {
                    let d2th_dx2_diag =
                        [d2th_dx2.0[vtx][0], d2th_dx2.0[vtx][2], d2th_dx2.0[vtx][5]].into_tensor();
                    let h = utils::get_diag3(
                        (dth_dx[vtx] * (dth_dx[vtx].transpose() * d2w_dth2)).as_data(),
                    )
                    .into_tensor()
                        + d2th_dx2_diag * dw_dth;
                    diag[vtx] = h * scale;
                }

                diag
            })
            .collect::<Vec<_>>()
        };

        // Transfer local values to global vector.
        membrane_diag
            .iter()
            .zip(self.shell.triangle_elements.triangles.iter())
            .for_each(|(local_diag, cell)| {
                for (&c, &g) in cell.iter().zip(local_diag.iter()) {
                    if c < diag.len() {
                        diag[c] += g;
                    }
                }
            });

        bending_diag
            .iter()
            .zip(self.shell.dihedral_elements.dihedrals.iter())
            .for_each(|(local_diag, di)| {
                for (&c, &g) in di
                    .verts(|f, i| self.shell.dihedral_elements.triangles[f][i])
                    .iter()
                    .zip(local_diag.iter())
                {
                    if c < diag.len() {
                        diag[c] += g;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::Mesh;
    use geo::mesh::VertexPositions;

    use crate::energy_models::elasticity::test_utils::*;
    use crate::energy_models::test_utils::*;
    use crate::fem::nl::solver::SolverBuilder;
    use crate::objects::trishell::TriShell;
    use crate::objects::*;

    use super::*;

    fn membrane_only_material() -> SoftShellMaterial {
        SoftShellMaterial::new(0)
            .with_elasticity(Elasticity::from_lame(
                5.4,
                263.1,
                ElasticityModel::NeoHookean,
            ))
            .with_density(1.0)
            .with_damping(1.0)
    }

    fn bend_only_material() -> SoftShellMaterial {
        SoftShellMaterial::new(0)
            .with_elasticity(Elasticity::from_lame(0.0, 0.0, ElasticityModel::NeoHookean))
            .with_density(1.0)
            .with_damping(1.0)
            .with_bending_stiffness(2.0)
    }

    fn material() -> SoftShellMaterial {
        membrane_only_material().with_bending_stiffness(2.0)
    }

    fn test_shells_for_material<'a>(
        materials: &'a [Material],
    ) -> impl Iterator<Item = (TriShell, Vec<[f64; 3]>)> + 'a {
        test_trimeshes().into_iter().map(move |mesh| {
            let mut mesh = Mesh::from(mesh);
            SolverBuilder::init_cell_vertex_ref_pos_attribute(&mut mesh).unwrap();
            let vertex_types =
                crate::fem::nl::state::sort_mesh_vertices_by_type(&mut mesh, &materials);

            (
                TriShell::try_from_mesh_and_materials(&mesh, &materials, &vertex_types).unwrap(),
                mesh.vertex_positions().to_vec(),
            )
        })
    }

    fn test_shells() -> Vec<(TriShell, Vec<[f64; 3]>)> {
        let membrane_only_materials = vec![membrane_only_material().into()];
        let bend_only_materials = vec![bend_only_material().into()];
        let materials = vec![material().into()];
        test_shells_for_material(&membrane_only_materials)
            .chain(test_shells_for_material(&bend_only_materials))
            .chain(test_shells_for_material(&materials))
            .collect()
    }

    fn build_energies(
        shells: &[(TriShell, Vec<[f64; 3]>)],
    ) -> Vec<(TriShellNeoHookean<autodiff::F1>, &[[f64; 3]])> {
        shells
            .iter()
            .map(|(shell, pos)| (TriShellNeoHookean::new(shell), pos.as_slice()))
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
