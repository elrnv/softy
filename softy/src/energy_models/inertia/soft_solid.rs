use num_traits::FromPrimitive;
//use rayon::prelude::*;

use flatk::zip;

#[cfg(feature = "optsolver")]
use crate::attrib_defines::*;
#[cfg(feature = "optsolver")]
use crate::objects::solid::*;
#[cfg(feature = "optsolver")]
use geo::attrib::Attrib;
#[cfg(feature = "optsolver")]
use geo::mesh::topology::*;

use geo::prim::Tetrahedron;
use tensr::{IntoData, Vector3};

use crate::energy::*;
use crate::matrix::*;
use crate::objects::tetsolid::*;
use crate::Real;

const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 12;
const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET: usize = 12;

#[cfg(feature = "optsolver")]
pub(crate) struct TetMeshInertia<'a>(pub &'a TetMeshSolid);

#[cfg(feature = "optsolver")]
impl<T: Real> Energy<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let tetmesh = &self.0.tetmesh;

        let vel0: &[Vector3<T>] = bytemuck::cast_slice(v0);
        let vel1: &[Vector3<T>] = bytemuck::cast_slice(v1);

        zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tetmesh.cell_iter(),
        )
        .map(|(&vol, density, cell)| {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell, vel0);
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = tet_v1 - tet_v0;

            T::from(0.5).unwrap() * {
                let dvTdv: T = tet_dv
                    .into_array()
                    .iter()
                    .map(|&dv| Vector3::new(dv).norm_squared())
                    .sum();
                // momentum
                T::from(0.25 * vol * density).unwrap() * dvTdv
            }
        })
        .sum()
    }
}

#[cfg(feature = "optsolver")]
impl<X: Real, T: Real> EnergyGradient<X, T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[X], v1: &[T], grad_f: &mut [T]) {
        let tetmesh = &self.0.tetmesh;

        let vel0: &[Vector3<X>] = bytemuck::cast_slice(v0);
        let vel1: &[Vector3<T>] = bytemuck::cast_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        // Transfer forces from cell-vertices to vertices themselves
        for (&vol, density, cell) in zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tetmesh.cell_iter()
        ) {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell, vel0);
            let tet_v0_t = Tetrahedron::<T>(
                Vector3::new(tet_v0.0.into()).cast::<T>().into_data().into(),
                Vector3::new(tet_v0.1.into()).cast::<T>().into_data().into(),
                Vector3::new(tet_v0.2.into()).cast::<T>().into_data().into(),
                Vector3::new(tet_v0.3.into()).cast::<T>().into_data().into(),
            );
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = (tet_v1 - tet_v0_t).into_array();

            for i in 0..4 {
                gradient[cell[i]] +=
                    Vector3::new(tet_dv[i]) * (T::from(0.25 * vol * density).unwrap());
            }
        }
    }
}

#[cfg(feature = "optsolver")]
impl EnergyHessianTopology for TetMeshInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.tetmesh.num_cells() * NUM_HESSIAN_TRIPLETS_PER_TET
    }

    fn num_hessian_diagonal_nnz(&self) -> usize {
        self.0.tetmesh.num_cells() * NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET
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

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(rows) };
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(cols) };

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .iter_mut()
            .zip(hess_col_chunks.iter_mut())
            .zip(tetmesh.cells().iter())
            .for_each(|((tet_hess_rows, tet_hess_cols), cell)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess_rows[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.row).unwrap();
                        tet_hess_cols[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.col).unwrap();
                    }
                }
            });
    }

    fn energy_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        assert_eq!(indices.len(), self.energy_hessian_size());

        let tetmesh = &self.0.tetmesh;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(indices) };

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(tetmesh.cells().iter())
            .for_each(|(tet_hess, cell)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] = MatrixElementIndex {
                            row: 3 * cell[vi] + j + offset.row,
                            col: 3 * cell[vi] + j + offset.col,
                        };
                    }
                }
            });
    }
}

#[cfg(feature = "optsolver")]
impl<T: Real> EnergyHessian<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _v0: &[T], _v1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let TetMeshInertia(ref solid) = *self;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(values) };

        let vol_iter = solid
            .tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap();

        let density_iter = solid
            .tetmesh
            .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
            .unwrap()
            .map(|&x| f64::from(x));

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(vol_iter.zip(density_iter))
            .for_each(|(tet_hess, (&vol, density))| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] = T::from(0.25 * vol * density).unwrap() * scale;
                    }
                }
            });
    }
}

pub(crate) struct TetSolidInertia<'a>(pub &'a TetElements);

impl<T: Real> Energy<T> for TetSolidInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let tet_elems = &self.0;

        let vel0: &[Vector3<T>] = bytemuck::cast_slice(v0);
        let vel1: &[Vector3<T>] = bytemuck::cast_slice(v1);

        zip!(
            tet_elems.ref_volume.iter(),
            tet_elems.density.iter().map(|&x| f64::from(x)),
            tet_elems.tets.iter(),
        )
        .map(|(&vol, density, cell)| {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell, vel0);
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = tet_v1 - tet_v0;

            T::from(0.5).unwrap() * {
                let dvTdv: T = tet_dv
                    .into_array()
                    .iter()
                    .map(|&dv| Vector3::new(dv).norm_squared())
                    .sum();
                // momentum
                T::from(0.25 * vol * density).unwrap() * dvTdv
            }
        })
        .sum()
    }
}

impl<X: Real, T: Real> EnergyGradient<X, T> for TetSolidInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[X], v1: &[T], grad_f: &mut [T]) {
        let tet_elems = &self.0;

        let vel0: &[Vector3<X>] = bytemuck::cast_slice(v0);
        let vel1: &[Vector3<T>] = bytemuck::cast_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        // Transfer forces from cell-vertices to vertices themselves
        for (&vol, density, cell) in zip!(
            tet_elems.ref_volume.iter(),
            tet_elems.density.iter().map(|&x| f64::from(x)),
            tet_elems.tets.iter()
        ) {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell, vel0);
            let tet_v0_t = Tetrahedron::<T>(
                Vector3::new(tet_v0.0.into()).cast::<T>().into_data().into(),
                Vector3::new(tet_v0.1.into()).cast::<T>().into_data().into(),
                Vector3::new(tet_v0.2.into()).cast::<T>().into_data().into(),
                Vector3::new(tet_v0.3.into()).cast::<T>().into_data().into(),
            );
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = (tet_v1 - tet_v0_t).into_array();

            for i in 0..4 {
                gradient[cell[i]] +=
                    Vector3::new(tet_dv[i]) * (T::from(0.25 * vol * density).unwrap());
            }
        }
    }
}

impl EnergyHessianTopology for TetSolidInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.num_elements() * NUM_HESSIAN_TRIPLETS_PER_TET
    }

    fn num_hessian_diagonal_nnz(&self) -> usize {
        self.0.num_elements() * NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TET
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

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(rows) };
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(cols) };

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .iter_mut()
            .zip(hess_col_chunks.iter_mut())
            .zip(tet_elems.tets.iter())
            .for_each(|((tet_hess_rows, tet_hess_cols), cell)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess_rows[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.row).unwrap();
                        tet_hess_cols[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.col).unwrap();
                    }
                }
            });
    }

    fn energy_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        assert_eq!(indices.len(), self.energy_hessian_size());

        let tet_elems = &self.0;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(indices) };

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(tet_elems.tets.iter())
            .for_each(|(tet_hess, cell)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] = MatrixElementIndex {
                            row: 3 * cell[vi] + j + offset.row,
                            col: 3 * cell[vi] + j + offset.col,
                        };
                    }
                }
            });
    }
}

impl<T: Real> EnergyHessian<T> for TetSolidInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _v0: &[T], _v1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let tet_elems = &self.0;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            unsafe { reinterpret::reinterpret_mut_slice(values) };

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(
                tet_elems
                    .ref_volume
                    .iter()
                    .zip(tet_elems.density.iter().map(|&x| f64::from(x))),
            )
            .for_each(|(tet_hess, (&vol, density))| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] = T::from(0.25 * vol * density).unwrap() * scale;
                    }
                }
            });
    }
}

impl<T: Real> EnergyHessianProduct<T> for TetSolidInertia<'_> {
    fn energy_hessian_product(&self, _x: &[T], _dx: &[T], p: &[T], scale: T, prod: &mut [T]) {
        let tet_elems = &self.0;

        tet_elems
            .tets
            .iter()
            .zip(
                tet_elems
                    .ref_volume
                    .iter()
                    .zip(tet_elems.density.iter().map(|&x| f64::from(x))),
            )
            .for_each(|(cell, (&vol, density))| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        let row = 3 * cell[vi] + j;
                        let col = 3 * cell[vi] + j;
                        prod[row] += p[col] * T::from(0.25 * vol * density).unwrap() * scale;
                    }
                }
            });
    }
}
