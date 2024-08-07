use num_traits::FromPrimitive;
//use rayon::prelude::*;

use flatk::zip;

use geo::prim::Triangle;
use tensr::{AsTensor, IntoData, IntoTensor, Vector3};

use crate::energy::*;
use crate::matrix::*;
use crate::objects::trishell::*;
use crate::Real;

const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 9;
const NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TRI: usize = 9;

pub(crate) struct SoftTriShellInertia<'a>(pub &'a TriShell);

impl<T: Real> Energy<T> for SoftTriShellInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T], _dqdv: T) -> T {
        let tri_elems = &self.0.triangle_elements;

        let vel0: &[Vector3<T>] = bytemuck::cast_slice(v0);
        let vel1: &[Vector3<T>] = bytemuck::cast_slice(v1);

        tri_elems
            .triangles
            .iter()
            .zip(
                tri_elems
                    .ref_area
                    .iter()
                    .zip(tri_elems.density.iter().map(|&x| f64::from(x))),
            )
            .map(|(face, (&area, density))| {
                let tri_v0 = Triangle::from_indexed_slice(face, vel0);
                let tri_v1 = Triangle::from_indexed_slice(face, vel1);
                let dv = tri_v1.into_array().into_tensor() - tri_v0.into_array().into_tensor();

                let third = 1.0 / 3.0;
                T::from(0.5).unwrap() * {
                    let dvTdv: T = dv.map(|dv| dv.norm_squared().into_tensor()).sum();
                    // Momentum
                    T::from(third * area * density).unwrap() * dvTdv
                }
            })
            .sum()
    }
}

impl<X: Real, T: Real> EnergyGradient<X, T> for SoftTriShellInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[X], v1: &[T], grad_f: &mut [T], _dqdv: T) {
        let tri_elems = &self.0.triangle_elements;

        let vel0: &[Vector3<X>] = bytemuck::cast_slice(v0);
        let vel1: &[Vector3<T>] = bytemuck::cast_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        // Transfer forces from cell-vertices to vertices themselves
        for (&area, density, face) in zip!(
            tri_elems.ref_area.iter(),
            tri_elems.density.iter().map(|&x| f64::from(x)),
            tri_elems.triangles.iter(),
        ) {
            let tri_v0 = Triangle::from_indexed_slice(face, vel0);
            let tri_v0_t = Triangle::<T>(
                Vector3::new(tri_v0.0.into()).cast::<T>().into_data().into(),
                Vector3::new(tri_v0.1.into()).cast::<T>().into_data().into(),
                Vector3::new(tri_v0.2.into()).cast::<T>().into_data().into(),
            );
            let tri_v1 = Triangle::from_indexed_slice(face, vel1);
            let dv = *tri_v1.as_array().as_tensor() - *tri_v0_t.as_array().as_tensor();

            let third = 1.0 / 3.0;
            for i in 0..3 {
                gradient[face[i]] += dv[i] * (T::from(third * area * density).unwrap());
            }
        }
    }
}

impl EnergyHessianTopology for SoftTriShellInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.triangle_elements.num_elements() * NUM_HESSIAN_TRIPLETS_PER_TRI
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        self.0.triangle_elements.num_elements() * NUM_HESSIAN_DIAGONAL_TRIPLETS_PER_TRI
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        debug_assert_eq!(rows.len(), self.energy_hessian_size());
        debug_assert_eq!(cols.len(), self.energy_hessian_size());

        let tri_elems = &self.0.triangle_elements;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            unsafe { reinterpret::reinterpret_mut_slice(rows) };
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            unsafe { reinterpret::reinterpret_mut_slice(cols) };

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .iter_mut()
            .zip(hess_col_chunks.iter_mut())
            .zip(tri_elems.triangles.iter())
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

        let tri_elems = &self.0.triangle_elements;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            unsafe { reinterpret::reinterpret_mut_slice(indices) };

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(tri_elems.triangles.iter())
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

impl<T: Real> EnergyHessian<T> for SoftTriShellInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _v0: &[T], _v1: &[T], scale: T, values: &mut [T], _dqdv: T) {
        debug_assert_eq!(values.len(), self.energy_hessian_size());

        let tri_elems = &self.0.triangle_elements;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            unsafe { reinterpret::reinterpret_mut_slice(values) };

        let third = 1.0 / 3.0;
        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(
                tri_elems
                    .ref_area
                    .iter()
                    .zip(tri_elems.density.iter().map(|&x| f64::from(x))),
            )
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

    #[allow(non_snake_case)]
    fn add_energy_hessian_diagonal(
        &self,
        _v0: &[T],
        _v1: &[T],
        scale: T,
        diag: &mut [T],
        _dqdv: T,
    ) {
        let tri_elems = &self.0.triangle_elements;

        let third = 1.0 / 3.0;
        // The momentum hessian is a diagonal matrix.
        tri_elems
            .triangles
            .iter()
            .zip(
                tri_elems
                    .ref_area
                    .iter()
                    .zip(tri_elems.density.iter().map(|&x| f64::from(x))),
            )
            .for_each(|(tri, (&area, density))| {
                for vi in tri.iter() {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        diag[3 * vi + j] += T::from(third * area * density).unwrap() * scale;
                    }
                }
            });
    }
}
impl<T: Real> EnergyHessianProduct<T> for SoftTriShellInertia<'_> {
    fn energy_hessian_product(&self, _x: &[T], _dx: &[T], p: &[T], scale: T, prod: &mut [T]) {
        let tri_elems = &self.0.triangle_elements;

        let third = 1.0 / 3.0;
        // The momentum hessian is a diagonal matrix.
        tri_elems
            .triangles
            .iter()
            .zip(
                tri_elems
                    .ref_area
                    .iter()
                    .zip(tri_elems.density.iter().map(|&x| f64::from(x))),
            )
            .for_each(|(face, (&area, density))| {
                for &vtx in face.iter() {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        let row = 3 * vtx + j;
                        let col = 3 * vtx + j;
                        prod[row] = p[col] * T::from(third * area * density).unwrap() * scale;
                    }
                }
            });
    }
}
