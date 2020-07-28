use num_traits::FromPrimitive;
//use rayon::prelude::*;
use reinterpret::*;

use flatk::zip;
use geo::mesh::{topology::*, Attrib};
use geo::prim::Tetrahedron;
use tensr::{Real, Vector3};

use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::solid::*;

const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 12;

pub(crate) struct TetMeshInertia<'a>(pub &'a TetMeshSolid);

impl<T: Real> Energy<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let tetmesh = &self.0.tetmesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

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

impl<T: Real> EnergyGradient<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[T], v1: &[T], grad_f: &mut [T]) {
        let tetmesh = &self.0.tetmesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

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
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = (tet_v1 - tet_v0).into_array();

            for i in 0..4 {
                gradient[cell[i]] +=
                    Vector3::new(tet_dv[i]) * (T::from(0.25 * vol * density).unwrap());
            }
        }
    }
}

impl EnergyHessianTopology for TetMeshInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.tetmesh.num_cells() * NUM_HESSIAN_TRIPLETS_PER_TET
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

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] = reinterpret_mut_slice(rows);
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] = reinterpret_mut_slice(cols);

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
            reinterpret_mut_slice(indices);

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

impl<T: Real> EnergyHessian<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _v0: &[T], _v1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let TetMeshInertia(ref solid) = *self;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TET]] = reinterpret_mut_slice(values);

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
