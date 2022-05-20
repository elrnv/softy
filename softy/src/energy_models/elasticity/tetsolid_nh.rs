//! Neo-Hookean energy model for tetrahedral meshes.

//use std::path::Path;
//use geo::io::save_tetmesh;
use super::tet_nh::NeoHookeanTetEnergy;
use super::TetEnergy;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::tetsolid::TetElements;
use crate::Real;
use flatk::zip;
use geo::ops::*;
use geo::prim::Tetrahedron;
use num_traits::FromPrimitive;
use rayon::prelude::*;
use reinterpret::*;
use tensr::*;
use unroll::unroll_for_loops;

/// A possibly non-linear elastic energy for tetrahedral meshes.
///
/// This type wraps a `TetElements` to provide an interfce for computing a hyperelastic energy.
pub struct TetSolidElasticity<'a, E>(pub &'a TetElements, std::marker::PhantomData<E>);

/// NeoHookean elasticity model.
pub type TetSolidNeoHookean<'a, T> = TetSolidElasticity<'a, NeoHookeanTetEnergy<T>>;

impl<'a, E> TetSolidElasticity<'a, E> {
    /// Number of Hessian triplets per tetrahedron.
    ///
    /// There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)
    const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 78;
    /// Number of Hessian triplets per tetrahedron on the diagonal.
    ///
    /// There are 4*3 = 12.
    const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET: usize = 12;

    pub fn new(solid: &'a TetElements) -> Self {
        TetSolidElasticity(solid, std::marker::PhantomData)
    }

    /// Helper for distributing local Hessian entries into the global Hessian matrix.
    ///
    /// This function provides the order of Hessian matrix non-zeros.
    /// `indices` is a map from the local tet vertex indices to their position in the global
    /// mesh
    /// `local_hess` is the function that computes a local hessian matrix for a pair of vertex
    /// indices
    /// `value` is the mapping function that would compute the hessian value. In particular
    /// `value` takes 3 pairs of (row, col) indices in this order:
    ///     - vertex indices
    ///     - local matrix indices
    /// as well as the local hessian matrix computed by `local_hess`.
    #[inline]
    #[unroll_for_loops]
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

/// Define a hyperelastic energy model for `TetElements`s.
impl<T: Real, E: TetEnergy<T>> Energy<T> for TetSolidElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy(&self, x: &[T], v: &[T], dqdv: T) -> T {
        let x: &[Vector3<T>] = bytemuck::cast_slice(x);
        let v: &[Vector3<T>] = bytemuck::cast_slice(v);

        let dqdv2 = dqdv * dqdv;

        zip!(
            self.0.damping.iter().map(|&x| f64::from(x)),
            // self.0.density.iter().map(|&x| f64::from(x)),
            self.0.ref_volume.iter(),
            self.0.ref_tet_shape_mtx_inv.iter(),
            self.0.tets.iter(),
            self.0.lambda.iter(),
            self.0.mu.iter(),
        )
        .map(|(damping, &vol, &DX_inv, cell, &lambda, &mu)| {
            let tet_x = Tetrahedron::from_indexed_slice(cell, x);
            let tet_v = Tetrahedron::from_indexed_slice(cell, v);
            let Dx = Matrix3::new(tet_x.shape_matrix());
            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let half = T::from(0.5).unwrap();
            let damping = T::from(damping).unwrap();
            let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);
            let dF = tet_energy.deformation_gradient_differential(&tet_v);
            // elasticity
            tet_energy.energy()
                // damping (viscosity)
                + mu * vol * dF.norm_squared() * half * damping * dqdv2
            //+ half * damping * {
            //    let dH = tet_energy.energy_hessian_product_transpose(&tet_dx);
            //    dH[0].dot(Vector3::new(tet_dx.0.into()))
            //        + dH[1].dot(Vector3::new(tet_dx.1.into()))
            //        + dH[2].dot(Vector3::new(tet_dx.2.into()))
            //        - (dH * Vector3::new(tet_dx.3.into())).sum()
            //}
        })
        .sum()
    }
}

impl<T: Real, E: TetEnergy<T>> EnergyGradient<T, T> for TetSolidElasticity<'_, E> {
    /// Derivative of the energy wrt `x`.
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, x: &[T], v: &[T], grad_f: &mut [T], dqdv: T) {
        debug_assert_eq!(grad_f.len(), x.len());
        debug_assert_eq!(grad_f.len(), v.len());

        let x: &[Vector3<T>] = bytemuck::cast_slice(x);
        let v: &[Vector3<T>] = bytemuck::cast_slice(v);

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        // Compute the gradient on cell-vertices in parallel first.
        let local_gradient: Vec<_> = zip!(
            self.0.damping.par_iter().map(|&x| f64::from(x)),
            // self.0.density.par_iter().map(|&x| f64::from(x)),
            self.0.ref_volume.par_iter(),
            self.0.ref_tet_shape_mtx_inv.par_iter(),
            self.0.tets.par_iter(),
            self.0.lambda.par_iter(),
            self.0.mu.par_iter(),
        )
        .map(|(damping, &vol, &DX_inv, cell, &lambda, &mu)| {
            // Make deformed tet.
            let tet_x = Tetrahedron::from_indexed_slice(cell, x);
            let tet_v = Tetrahedron::from_indexed_slice(cell, v);

            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();

            let tet_energy = E::new(Matrix3::new(tet_x.shape_matrix()), DX_inv, vol, lambda, mu);

            let mut grad = tet_energy.energy_gradient();

            // Damping
            let damping = T::from(damping).unwrap();
            let dF = tet_energy.deformation_gradient_differential(&tet_v) * dqdv;

            let damp = DX_inv.transpose() * dF * (vol * mu * damping);
            for i in 0..3 {
                grad[i] += damp[i];
                grad[3] -= damp[i];
            }

            // let damping = T::from(damping).unwrap();
            // let dH = tet_energy.energy_hessian_product_transpose(&tet_v);
            // for i in 0..3 {
            //    grad[i] += dH[i] * damping;
            //    grad[3] -= dH[i] * damping;
            // }

            grad
        })
        .collect();

        // Transfer forces from cell-vertices to vertices themselves.
        local_gradient
            .iter()
            .zip(self.0.tets.iter())
            .for_each(|(local_grad, cell)| {
                for (&c, &g) in cell.iter().zip(local_grad.iter()) {
                    gradient[c] += g;
                }
            });
    }
}

impl<E> EnergyHessianTopology for TetSolidElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        Self::NUM_HESSIAN_TRIPLETS_PER_TET * self.0.num_elements()
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        Self::NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET * self.0.num_elements()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tet_elems = &self.0;

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_row_chunks: &mut [[I; 78]] = unsafe { reinterpret_mut_slice(rows) };
            let hess_col_chunks: &mut [[I; 78]] = unsafe { reinterpret_mut_slice(cols) };

            let hess_iter = hess_row_chunks
                .par_iter_mut()
                .zip(hess_col_chunks.par_iter_mut())
                .zip(tet_elems.tets.par_iter());

            hess_iter.for_each(|((tet_hess_rows, tet_hess_cols), cell)| {
                Self::hessian_for_each(
                    |_, _| {},
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

        let tet_elems = &self.0;

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementIndex; 78]] =
                unsafe { reinterpret_mut_slice(indices) };

            let hess_iter = hess_chunks.par_iter_mut().zip(tet_elems.tets.par_iter());

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

impl<T: Real + Send + Sync, E: TetEnergy<T>> EnergyHessian<T> for TetSolidElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, x: &[T], _: &[T], scale: T, values: &mut [T], _dqdv: T) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let pos: &[Vector3<T>] = bytemuck::cast_slice(x);

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[T; 78]] = unsafe { reinterpret_mut_slice(values) };

        let project_local_hessians = self.0.projected_hessian;

        zip!(
            hess_chunks.par_iter_mut(),
            self.0.damping.par_iter().map(|&x| f64::from(x)),
            // self.0.density.par_iter().map(|&x| f64::from(x)),
            self.0.ref_volume.par_iter(),
            self.0.ref_tet_shape_mtx_inv.par_iter(),
            self.0.tets.par_iter(),
            self.0.lambda.par_iter(),
            self.0.mu.par_iter(),
        )
        .for_each(|(tet_hess, damping, &vol, &DX_inv, cell, &lambda, &mu)| {
            // Make deformed tet.
            let tet_x = Tetrahedron::from_indexed_slice(cell, pos);

            let Dx = Matrix3::new(tet_x.shape_matrix());

            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            // let size = T::from(vol.cbrt()).unwrap();
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();

            let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);

            //let factor = T::from(1.0 + damping).unwrap() * scale;
            let factor = scale;

            let mut local_hessians = tet_energy.energy_hessian();

            // Project local hessians to be positive semi-definite.
            if project_local_hessians {
                let mut local_hess_mtx =
                    na::Matrix::<T, na::U12, na::U12, na::ArrayStorage<T, 12, 12>>::from_fn(
                        |r, c| local_hessians[r / 3][c / 3][r % 3][c % 3],
                    );

                let mut eigen = local_hess_mtx.symmetric_eigen();
                for eigenval in &mut eigen.eigenvalues {
                    if *eigenval < T::zero() {
                        *eigenval = T::zero();
                    }
                }
                local_hess_mtx.copy_from(&eigen.recompose());
                for n in 0..4 {
                    for k in 0..4 {
                        for r in 0..3 {
                            for c in 0..3 {
                                local_hessians[n][k][r][c] = local_hess_mtx[(3 * n + r, 3 * k + c)];
                            }
                        }
                    }
                }
            }

            // Damping
            let damping = T::from(damping).unwrap();
            // let density = T::from(density).unwrap();
            let ddF = DX_inv.transpose() * DX_inv * (vol * mu * damping);

            let id = Matrix3::identity();

            Self::hessian_for_each(
                |n, k| {
                    let damping_hess = id
                        * if n == 3 && k == 3 {
                            ddF.sum_inner()
                        } else if k == 3 {
                            -ddF[n].sum()
                        } else if n == 3 {
                            -ddF[k].sum() // ddF should be symmetric.
                        } else {
                            ddF[n][k]
                        };
                    (local_hessians[n][k] + damping_hess) * factor
                },
                |i, _, (row, col), h| {
                    // // DEBUG CODE
                    // let mut global_row = 3 * cell[n] + row;
                    // let mut global_col = 3 * cell[k] + col;
                    // if cell[n] < cell[k] {
                    //     // In the upper triangular part of the global matrix, transpose
                    //     std::mem::swap(&mut global_row, &mut global_col);
                    // }
                    // eprintln!("({:?}, {:?}): {:?}", global_row, global_col, damp_hess[row][col]);
                    // // END OF DEBUG CODE
                    tet_hess[i] = h[row][col]
                },
            );
        });
    }

    #[allow(non_snake_case)]
    #[unroll_for_loops]
    fn add_energy_hessian_diagonal(&self, x: &[T], _: &[T], scale: T, diag: &mut [T], _dqdv: T) {
        let pos: &[Vector3<T>] = bytemuck::cast_slice(x);

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_diag: &mut [Vector3<T>] = bytemuck::cast_slice_mut(diag);

        let local_diag = zip!(
            self.0.damping.par_iter().map(|&x| f64::from(x)),
            // self.0.density.par_iter().map(|&x| f64::from(x)),
            self.0.ref_volume.par_iter(),
            self.0.ref_tet_shape_mtx_inv.par_iter(),
            self.0.tets.par_iter(),
            self.0.lambda.par_iter(),
            self.0.mu.par_iter(),
        )
        .map(|(damping, &vol, &DX_inv, cell, &lambda, &mu)| {
            // Make deformed tet.
            let tet_x = Tetrahedron::from_indexed_slice(cell, pos);

            let Dx = Matrix3::new(tet_x.shape_matrix());

            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            // let size = T::from(vol.cbrt()).unwrap();
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();

            let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);

            //let factor = T::from(1.0 + damping).unwrap() * scale;
            let factor = scale;

            let local_hessians = tet_energy.energy_hessian();

            let mut diag = [Vector3::zeros(); 4];
            for i in 0..4 {
                diag[i] = utils::get_diag3(local_hessians[i][i].as_data()).into_tensor() * factor;
            }

            // Damping
            let damping = T::from(damping).unwrap();
            let ddF = DX_inv.transpose() * DX_inv * (vol * mu * damping);

            let id = Vector3::from([T::one(); 3]);

            for i in 0..3 {
                diag[i] += id * ddF[i][i] * factor;
            }
            diag[3] += id * ddF.sum_inner() * factor;

            diag
        })
        .collect::<Vec<_>>();

        // Transfer forces from cell-vertices to vertices themselves.
        local_diag
            .iter()
            .zip(self.0.tets.iter())
            .for_each(|(local_diag, cell)| {
                for (&c, &g) in cell.iter().zip(local_diag.iter()) {
                    if c < hess_diag.len() {
                        hess_diag[c] += g;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::{elasticity::test_utils::*, test_utils::*};
    use crate::fem::nl::solver::SolverBuilder;
    use crate::objects::{tetsolid::TetElements, *};
    use geo::mesh::{Mesh, VertexPositions};

    fn material() -> SolidMaterial {
        SolidMaterial::new(0)
            .with_elasticity(Elasticity::from_lame(
                5.4,
                263.1,
                ElasticityModel::NeoHookean,
            ))
            .with_density(10.0)
            .with_damping(1.0)
    }

    fn test_solids() -> Vec<(TetElements, Vec<[f64; 3]>)> {
        let material = material();

        test_tetmeshes()
            .into_iter()
            .map(|tetmesh| Mesh::from(tetmesh))
            .flat_map(|mut mesh| {
                SolverBuilder::init_cell_vertex_ref_pos_attribute(&mut mesh).unwrap();
                let materials = vec![material.into()];
                let vertex_types =
                    crate::fem::nl::state::sort_mesh_vertices_by_type(&mut mesh, &materials);
                std::iter::once((
                    // Prepare attributes relevant for elasticity computations.
                    TetElements::try_from_mesh_and_materials(
                        ElasticityModel::NeoHookean,
                        &mesh,
                        &materials,
                        vertex_types.as_slice(),
                        false,
                    )
                    .unwrap(),
                    mesh.vertex_positions().to_vec(),
                ))
                .chain(std::iter::once((
                    // Prepare attributes relevant for elasticity computations.
                    TetElements::try_from_mesh_and_materials(
                        ElasticityModel::StableNeoHookean,
                        &mesh,
                        &materials,
                        vertex_types.as_slice(),
                        false,
                    )
                    .unwrap(),
                    mesh.vertex_positions().to_vec(),
                )))
            })
            .collect()
    }

    fn build_energies(
        solids: &[(TetElements, Vec<[f64; 3]>)],
    ) -> Vec<(TetSolidNeoHookean<autodiff::F1>, &[[f64; 3]])> {
        solids
            .iter()
            .map(|(solid, pos)| (TetSolidNeoHookean::new(solid), pos.as_slice()))
            .collect()
    }

    #[test]
    fn tet_energy_gradient() {
        tet_energy_gradient_tester::<NeoHookeanTetEnergy<autodiff::F1>>();
    }

    #[test]
    fn tet_energy_hessian() {
        tet_energy_hessian_tester::<NeoHookeanTetEnergy<autodiff::F1>>();
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
