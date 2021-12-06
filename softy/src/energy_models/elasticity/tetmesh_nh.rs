//! Neo-Hookean energy model for tetrahedral meshes.

//use std::path::Path;
//use geo::io::save_tetmesh;
use super::tet_nh::NeoHookeanTetEnergy;
use super::TetEnergy;
use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::solid::TetMeshSolid;
use crate::objects::*;
use crate::Real;
use flatk::zip;
use geo::attrib::Attrib;
use geo::mesh::topology::*;
use geo::ops::*;
use geo::prim::Tetrahedron;
use num_traits::FromPrimitive;
use rayon::iter::Either;
use rayon::prelude::*;
use reinterpret::*;
use tensr::*;

/// A possibly non-linear elastic energy for tetrahedral meshes.
/// This type wraps a `TetMeshSolid` to provide an interfce for computing a hyperelastic energy.
pub struct TetMeshElasticity<'a, E>(pub &'a TetMeshSolid, std::marker::PhantomData<E>);

/// NeoHookean elasticity model.
pub type TetMeshNeoHookean<'a, T> = TetMeshElasticity<'a, NeoHookeanTetEnergy<T>>;

impl<'a, E> TetMeshElasticity<'a, E> {
    /// Number of Hessian triplets per tetrahedron.
    ///
    /// There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)
    const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 78;
    /// Number of Hessian triplets per tetrahedron on the diagonal.
    ///
    /// There are 4*3 = 12.
    const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET: usize = 12;

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

        let damping = material.damping();

        let pos0: &[Vector3<T>] = bytemuck::cast_slice(x0);
        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x1);

        tetmesh
            .attrib_iter::<FixedIntType, CellIndex>(FIXED_ATTRIB)
            .unwrap().zip(
                zip!(
                    Either::from(tetmesh
                        .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                        .map(|i| i.cloned())
                        .map_err(|_| std::iter::repeat(0.0f32))),
                    tetmesh
                        .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                        .unwrap(),
                    tetmesh
                        .attrib_iter::<RefTetShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                        .unwrap(),
                    tetmesh.cell_iter(),
                    tetmesh
                        .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                        .unwrap(),
                    tetmesh.attrib_iter::<MuType, CellIndex>(MU_ATTRIB).unwrap()
                )
        )
        .filter(|(&fixed, _)| fixed == 0)
        .map(|(_, (density, &vol, &DX_inv, cell, &lambda, &mu))| {
            let tet_x1 = Tetrahedron::from_indexed_slice(cell, pos1);
            let tet_dx = &tet_x1 - &Tetrahedron::from_indexed_slice(cell, pos0);
            let Dx = Matrix3::new(tet_x1.shape_matrix());
            let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
            let vol = T::from(vol).unwrap();
            let lambda = T::from(lambda).unwrap();
            let mu = T::from(mu).unwrap();
            let half = T::from(0.5).unwrap();
            let damping = T::from(damping).unwrap();
            let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);
            let dF = tet_energy.deformation_gradient_differential(&tet_dx);
            // elasticity
            tet_energy.energy()
                // damping (viscosity)
                // Note: damping is already scaled by dt
                + if density != 0.0 { vol * dF.norm_squared() * half * T::from(density).unwrap() * damping } else { T::zero() }
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

impl<X: Real, T: Real, E: TetEnergy<T>> EnergyGradient<X, T> for TetMeshElasticity<'_, E> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, x0: &[X], x1: &[T], grad_f: &mut [T]) {
        let TetMeshSolid {
            ref tetmesh,
            material,
            ..
        } = *self.0;

        let damping = material.damping();

        debug_assert_eq!(grad_f.len(), x0.len());
        debug_assert_eq!(grad_f.len(), x1.len());

        let pos0: &[Vector3<X>] = bytemuck::cast_slice(x0);
        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x1);

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        let fixed_attrib = tetmesh
            .attrib_as_slice::<FixedIntType, CellIndex>(FIXED_ATTRIB)
            .unwrap();
        // Compute the gradient on cell-vertices in parallel first.
        let local_gradient: Vec<_> = fixed_attrib
            .par_iter()
            .zip(zip!(
                Either::from(
                    tetmesh
                        .attrib_as_slice::<DensityType, CellIndex>(DENSITY_ATTRIB)
                        .map(|i| i.par_iter().cloned())
                        .map_err(|_| rayon::iter::repeatn(0.0f32, tetmesh.num_cells()))
                ),
                tetmesh
                    .attrib_as_slice::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap()
                    .par_iter(),
                tetmesh
                    .attrib_as_slice::<RefTetShapeMtxInvType, CellIndex>(
                        REFERENCE_SHAPE_MATRIX_INV_ATTRIB
                    )
                    .unwrap()
                    .par_iter(),
                tetmesh.cells().par_iter(),
                tetmesh
                    .attrib_as_slice::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                    .unwrap()
                    .par_iter(),
                tetmesh
                    .attrib_as_slice::<MuType, CellIndex>(MU_ATTRIB)
                    .unwrap()
                    .par_iter()
            ))
            .filter(|(&fixed, _)| fixed == 0)
            .map(|(_, (density, &vol, &DX_inv, cell, &lambda, &mu))| {
                // Make deformed tet.
                let tet_x1 = Tetrahedron::from_indexed_slice(cell, pos1);
                let tet_x0 = Tetrahedron::from_indexed_slice(cell, pos0);
                let tet_x0_t = Tetrahedron::<T>(
                    tensr::Vector3::new(tet_x0.0.into())
                        .cast::<T>()
                        .into_data()
                        .into(),
                    tensr::Vector3::new(tet_x0.1.into())
                        .cast::<T>()
                        .into_data()
                        .into(),
                    tensr::Vector3::new(tet_x0.2.into())
                        .cast::<T>()
                        .into_data()
                        .into(),
                    tensr::Vector3::new(tet_x0.3.into())
                        .cast::<T>()
                        .into_data()
                        .into(),
                );
                // Make tet displacement.
                let tet_dx = &tet_x1 - &tet_x0_t;

                let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                let vol = T::from(vol).unwrap();
                let lambda = T::from(lambda).unwrap();
                let mu = T::from(mu).unwrap();

                let tet_energy =
                    E::new(Matrix3::new(tet_x1.shape_matrix()), DX_inv, vol, lambda, mu);

                let mut grad = tet_energy.energy_gradient();

                // Damping
                if density != 0.0 {
                    let density = T::from(density).unwrap();
                    let damping = T::from(damping).unwrap();
                    let dF = tet_energy.deformation_gradient_differential(&tet_dx);

                    // Note: damping is already scaled by dt
                    let damp = DX_inv.transpose() * dF * vol * density * damping;
                    for i in 0..3 {
                        grad[i] += damp[i];
                        grad[3] -= damp[i];
                    }
                }

                grad

                //let dH = tet_energy.energy_hessian_product_transpose(&tet_dx);
                //for i in 0..3 {
                //    gradient[cell[i]] += dH[i] * damping;
                //    gradient[cell[3]] -= dH[i] * damping;
                //}
            })
            .collect();

        // Transfer forces from cell-vertices to vertices themselves.
        local_gradient
            .iter()
            .zip(
                tetmesh
                    .cell_iter()
                    .zip(fixed_attrib.iter())
                    .filter(|(_, &fixed)| fixed == 0),
            )
            .for_each(|(local_grad, (cell, _))| {
                for (&c, &g) in cell.iter().zip(local_grad.iter()) {
                    gradient[c] += g;
                }
            });
    }
}

impl<E> EnergyHessianTopology for TetMeshElasticity<'_, E> {
    fn energy_hessian_size(&self) -> usize {
        Self::NUM_HESSIAN_TRIPLETS_PER_TET * self.0.tetmesh.num_cells()
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        Self::NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET * self.0.tetmesh.num_cells()
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
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
            //let hess_row_chunks: &mut [[I; 78]] = unsafe { reinterpret_mut_slice(rows) };
            let hess_row_chunks: &mut [[I; 78]] = unsafe { reinterpret_mut_slice(rows) };
            let hess_col_chunks: &mut [[I; 78]] = unsafe { reinterpret_mut_slice(cols) };

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
            let hess_chunks: &mut [[MatrixElementIndex; 78]] =
                unsafe { reinterpret_mut_slice(indices) };

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
    fn energy_hessian_values(&self, _: &[T], x1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());
        let TetMeshSolid {
            ref tetmesh,
            material,
            ..
        } = *self.0;

        let damping = material.damping();

        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x1);

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[T; 78]] = unsafe { reinterpret_mut_slice(values) };

            hess_chunks
                .par_iter_mut()
                .zip(
                    tetmesh
                        .attrib_as_slice::<FixedIntType, CellIndex>(FIXED_ATTRIB)
                        .unwrap()
                        .par_iter()
                        .zip(zip!(
                            Either::from(
                                tetmesh
                                    .attrib_as_slice::<DensityType, CellIndex>(DENSITY_ATTRIB)
                                    .map(|slice| slice.par_iter().cloned())
                                    .map_err(|_| rayon::iter::repeatn(0.0f32, tetmesh.num_cells()))
                            ),
                            tetmesh
                                .attrib_as_slice::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                                .unwrap()
                                .par_iter(),
                            tetmesh
                                .attrib_as_slice::<RefTetShapeMtxInvType, CellIndex>(
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
                        )),
                )
                .filter(|(_, (&fixed, _))| fixed == 0)
                .for_each(
                    |(tet_hess, (_, (density, &vol, &DX_inv, cell, &lambda, &mu)))| {
                        // Make deformed tet.
                        let tet_x1 = Tetrahedron::from_indexed_slice(cell, pos1);

                        let Dx = Matrix3::new(tet_x1.shape_matrix());

                        let DX_inv = DX_inv.mapd_inner(|x| T::from(x).unwrap());
                        let vol = T::from(vol).unwrap();
                        let lambda = T::from(lambda).unwrap();
                        let mu = T::from(mu).unwrap();

                        let tet_energy = E::new(Dx, DX_inv, vol, lambda, mu);

                        //let factor = T::from(1.0 + damping).unwrap() * scale;
                        let factor = scale;

                        let local_hessians = tet_energy.energy_hessian();

                        // Damping
                        let damping = T::from(damping).unwrap();
                        let density = T::from(density).unwrap();
                        // Note: damping is already scaled by dt
                        let ddF = DX_inv.transpose() * DX_inv * (vol * density * damping);
                        let id = Matrix3::identity();

                        Self::hessian_for_each(
                            |n, k| {
                                (local_hessians[n][k]
                                    + id * if n == 3 && k == 3 {
                                        ddF.sum_inner()
                                    } else if k == 3 {
                                        -ddF[n].sum()
                                    } else if n == 3 {
                                        -ddF[k].sum() // ddF should be symmetric.
                                    } else {
                                        ddF[n][k]
                                    })
                                    * factor
                            },
                            |i, _, (row, col), h| tet_hess[i] = h[row][col],
                        );
                    },
                );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::elasticity::test_utils::*;
    use crate::energy_models::test_utils::*;
    use crate::fem::opt::SolverBuilder;
    use crate::objects::solid::TetMeshSolid;
    use geo::mesh::VertexPositions;

    fn material() -> SolidMaterial {
        SolidMaterial::new(0)
            .with_elasticity(ElasticityParameters {
                lambda: 5.4,
                mu: 263.1,
                model: ElasticityModel::NeoHookean,
            })
            .with_density(10.0)
            .with_damping(1.0, 0.01)
    }

    fn test_solids() -> Vec<TetMeshSolid> {
        let material = material();

        test_tetmeshes()
            .into_iter()
            .map(|tetmesh| {
                // Prepare attributes relevant for elasticity computations.
                let mut solid = TetMeshSolid::new(tetmesh, material);
                solid.init_deformable_vertex_attributes().unwrap();
                SolverBuilder::prepare_deformable_tetmesh_attributes(&mut solid.tetmesh).unwrap();
                solid.init_elasticity_attributes().unwrap();
                solid.init_density_attribute().unwrap();
                solid.init_fixed_element_attribute().unwrap();
                solid
            })
            .collect()
    }

    fn build_energies(
        solids: &[TetMeshSolid],
    ) -> Vec<(TetMeshNeoHookean<autodiff::F1>, &[[f64; 3]])> {
        solids
            .iter()
            .map(|solid| {
                (
                    TetMeshNeoHookean::new(solid),
                    solid.tetmesh.vertex_positions(),
                )
            })
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
