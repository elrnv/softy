use num_traits::FromPrimitive;
use unroll::unroll_for_loops;

use tensr::{AsData, Chunked3, IntoData, IntoTensor, Matrix2x3, Matrix3, Vector3};

use crate::energy::*;
use crate::matrix::*;
use crate::Real;

const NUM_HESSIAN_TRIPLETS: usize = 9;
const NUM_HESSIAN_DIAGONAL_TRIPLETS: usize = 6;

pub(crate) struct RigidShellInertia {
    pub mass: f64,
    pub inertia: Matrix3<f64>,
}

/// Extract rigid degrees of freedom from a slice.
///
/// The returned matrix has 2 rows one for translational degrees of freedom and another
/// for rotational.
#[inline]
fn rigid_dofs<T: Real>(x: &[T]) -> Matrix2x3<T> {
    let x = Chunked3::from_flat(x).into_arrays();
    debug_assert_eq!(x.len(), 2);
    Matrix2x3::new(unsafe { [*x.get_unchecked(0), *x.get_unchecked(1)] })
}

impl<T: Real> Energy<T> for RigidShellInertia {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T], _dqdv: T) -> T {
        // There are translation and rotation dofs only for rigid bodies.
        let v0 = rigid_dofs(v0);
        let v1 = rigid_dofs(v1);
        let dv = v1 - v0;

        T::from(0.5).unwrap() * {
            // Compute Momenta

            // Translational dofs
            let p: T = dv[0].norm_squared() * T::from(self.mass).unwrap();

            // Rotational dofs
            let l: T = dv[1].dot(self.inertia.cast_inner::<T>() * dv[1]);

            l + p
        }
    }
}

impl<X: Real, T: Real> EnergyGradient<X, T> for RigidShellInertia {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[X], v1: &[T], grad_f: &mut [T], _dqdv: T) {
        debug_assert_eq!(grad_f.len(), v0.len());

        // There are translation and rotation dofs only for rigid bodies.
        let v0 = rigid_dofs(v0);
        let v1 = rigid_dofs(v1);
        let dv = v1 - v0.cast_inner::<T>();

        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad_f);

        gradient[0] += dv[0] * T::from(self.mass).unwrap();
        gradient[1] += self.inertia.cast_inner::<T>() * dv[1];
    }
}

impl EnergyHessianTopology for RigidShellInertia {
    fn energy_hessian_size(&self) -> usize {
        NUM_HESSIAN_TRIPLETS
    }

    fn num_hessian_diagonal_nnz(&self) -> usize {
        NUM_HESSIAN_DIAGONAL_TRIPLETS
    }

    #[unroll_for_loops]
    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        mut offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        debug_assert_eq!(rows.len(), self.energy_hessian_size());
        debug_assert_eq!(cols.len(), self.energy_hessian_size());

        for i in 0..3 {
            // Translational degrees of freedom generate a diagonal matrix.
            rows[i] = I::from_usize(i + offset.row).unwrap();
            cols[i] = I::from_usize(i + offset.col).unwrap();
        }

        offset.row += 3;
        offset.col += 3;

        let mut i = 3;
        for row in 0..3 {
            for col in 0..=row {
                // Lower triangular part of the inertia matrix in row-major order.
                rows[i] = I::from_usize(row + offset.row).unwrap();
                cols[i] = I::from_usize(col + offset.col).unwrap();
                i += 1;
            }
        }
        assert_eq!(i, self.energy_hessian_size());
    }

    fn energy_hessian_indices_offset(
        &self,
        mut offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        debug_assert_eq!(indices.len(), self.energy_hessian_size());

        for (i, idx) in indices.iter_mut().enumerate().take(3) {
            // Translational degrees of freedom generate a diagonal matrix.
            *idx = MatrixElementIndex {
                row: i + offset.row,
                col: i + offset.col,
            };
        }

        offset.row += 3;
        offset.col += 3;

        let mut i = 3;
        for row in 0..3 {
            for col in 0..=row {
                // Lower triangular part of the inertia matrix in row-major order.
                indices[i] = MatrixElementIndex {
                    row: row + offset.row,
                    col: col + offset.col,
                };
                i += 1;
            }
        }
        assert_eq!(i, self.energy_hessian_size());
    }
}

impl<T: Real> EnergyHessian<T> for RigidShellInertia {
    #[allow(non_snake_case)]
    fn energy_hessian_values(&self, _v0: &[T], _v1: &[T], scale: T, values: &mut [T], _dqdv: T) {
        debug_assert_eq!(values.len(), self.energy_hessian_size());
        for v in values.iter_mut().take(3) {
            *v = T::from(self.mass).unwrap() * scale;
        }
        values[3..].copy_from_slice(
            &(self.inertia.lower_triangular_vec().cast::<T>() * scale).into_data()[..],
        );
    }
    fn add_energy_hessian_diagonal(
        &self,
        _v0: &[T],
        _v1: &[T],
        scale: T,
        diag: &mut [T],
        _dqdv: T,
    ) {
        for v in diag[..3].iter_mut() {
            *v += T::from(self.mass).unwrap() * scale;
        }
        diag[3..]
            .iter_mut()
            .zip(
                (utils::get_diag3(self.inertia.as_data())
                    .into_tensor()
                    .cast::<T>()
                    * scale)
                    .into_data()
                    .iter(),
            )
            .for_each(|(out, &input)| *out += input);
    }
}
