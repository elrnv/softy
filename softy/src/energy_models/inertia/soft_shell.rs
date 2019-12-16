use num_traits::FromPrimitive;
//use rayon::prelude::*;
use reinterpret::*;

use geo::mesh::{Attrib, topology::*};
use geo::prim::Triangle;
use utils::soap::{AsTensor, IntoTensor, Real, Vector3};
use utils::zip;

use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::shell::*;

const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 9;

pub(crate) struct SoftShellInertia<'a>(pub &'a TriMeshShell);

impl<T: Real> Energy<T> for SoftShellInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let trimesh = &self.0.trimesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh.face_iter(),
        )
            .map(|(&area, density, face)| {
                let tri_v0 = Triangle::from_indexed_slice(face, vel0);
                let tri_v1 = Triangle::from_indexed_slice(face, vel1);
                let dv = tri_v1.into_array().into_tensor() - tri_v0.into_array().into_tensor();

                let third = 1.0 / 3.0;
                T::from(0.5).unwrap() * {
                    let dvTdv: T = dv
                        .map(|dv| dv.norm_squared().into_tensor())
                        .sum();
                    // momentum
                    T::from(third * area * density).unwrap() * dvTdv
                }
            })
            .sum()
    }
}

impl<T: Real> EnergyGradient<T> for SoftShellInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[T], v1: &[T], grad_f: &mut [T]) {
        let trimesh = &self.0.trimesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themselves
        for (&area, density, face) in zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh.face_iter()
        ) {
            let tri_v0 = Triangle::from_indexed_slice(face, vel0);
            let tri_v1 = Triangle::from_indexed_slice(face, vel1);
            let dv = *tri_v1.as_array().as_tensor() - *tri_v0.as_array().as_tensor();

            let third = 1.0 / 3.0;
            for i in 0..3 {
                gradient[face[i]] +=
                    dv[i] * (T::from(third * area * density).unwrap());
            }
        }
    }
}

impl EnergyHessianTopology for SoftShellInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.trimesh.num_faces() * NUM_HESSIAN_TRIPLETS_PER_TRI
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        debug_assert_eq!(rows.len(), self.energy_hessian_size());
        debug_assert_eq!(cols.len(), self.energy_hessian_size());

        let trimesh = &self.0.trimesh;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(rows);
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(cols);

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .iter_mut()
            .zip(hess_col_chunks.iter_mut())
            .zip(trimesh.faces().iter())
            .for_each(|((tri_hess_rows, tri_hess_cols), cell)| {
                for vi in 0..3 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tri_hess_rows[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.row).unwrap();
                        tri_hess_cols[3 * vi + j] =
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
        debug_assert_eq!(indices.len(), self.energy_hessian_size());

        let trimesh = &self.0.trimesh;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(indices);

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(trimesh.faces().iter())
            .for_each(|(tri_hess, face)| {
                for vi in 0..3 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tri_hess[3 * vi + j] = MatrixElementIndex {
                            row: 3 * face[vi] + j + offset.row,
                            col: 3 * face[vi] + j + offset.col,
                        };
                    }
                }
            });
    }
}

impl<T: Real> EnergyHessian<T> for SoftShellInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(
        &self,
        _v0: &[T],
        _v1: &[T],
        scale: T,
        values: &mut [T],
    ) {
        debug_assert_eq!(values.len(), self.energy_hessian_size());

        let SoftShellInertia(ref shell) = *self;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(values);

        let vol_iter = shell
            .trimesh
            .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
            .unwrap();

        let density_iter = shell
            .trimesh
            .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
            .unwrap()
            .map(|&x| f64::from(x));

        let third = 1.0 / 3.0;
        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(vol_iter.zip(density_iter))
            .for_each(|(tri_hess, (&area, density))| {
                for vi in 0..3 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tri_hess[3 * vi + j] = T::from(third * area * density).unwrap() * scale;
                    }
                }
            });
    }
}
