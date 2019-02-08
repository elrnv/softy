/*!
 * Any energy interface that can be implemented by any quantity that has first and
 * second order derivatives. Implementing this interface allows the quantity to be used in an
 * optimization solver. Becasue these functions may be called many times in an inner loop it is
 * advised for implementers to reuse allocated memory as much as possible. For this reason the
 * trains in this module take a mutable reference to `self` instead of an immutable one.
 */

use geo::math::{Matrix3, Scalar};
use geo::prim::Tetrahedron;
use num_traits::FromPrimitive;

use crate::matrix::{MatrixElementIndex, MatrixElementTriplet};

/// Tetrahedron energy interface. Abstracting over tet energies is useful for damping
/// implementations like Rayleigh damping which depend on the elasticity model used.
pub trait TetEnergy<T: Scalar> {
    /// Constructor accepts:
    /// `Dx`: the deformed shape matrix
    /// `DX_inv`: the undeformed shape matrix
    /// `volume`: volume of the tetrahedron
    /// `lambda` and `mu`: Lamé parameters
    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<f64>, DX_inv: Matrix3<f64>, volume: f64, lambda: f64, mu: f64) -> Self;
    /// Elasticity Hessian*displacement product per element. Represented by a 3x3 matrix where
    /// column `i` produces the hessian product contribution for the vertex `i` within the current
    /// element.
    fn elastic_energy_hessian_product(&self, dx: &Tetrahedron<f64>) -> Matrix3<T>;
}

/// Energy trait. This trait provides the energy value that, for instance, may be used in the
/// objective function for an optimization algorithm.
pub trait Energy<T: Scalar> {
    /// Compute the energy of the current configuration.
    ///
    ///   - `x` is the variable expected by the specific energy for the previous configuration. For
    /// example elastic energy expects position while momentum energy expects velocity.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    /// differential of `x` but it often is.
    fn energy(&self, x: &[T], dx: &[T]) -> T;
}

/// The energy gradient is required for optimization methods that require first order derivative
/// information, like Gradient Descent for instance.
pub trait EnergyGradient<T: Scalar> {
    /// Compute the change in energy with respect to change in configuration and add it to the
    /// given slice of global gradient values.
    ///
    ///   - `x` is the variable expected by the specific energy for the previous configuration. For
    /// example elastic energy expects position while momentum energy expects velocity.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    /// differential of `x` but it often is.
    ///
    /// This derivative is with respect to `dx`.
    fn add_energy_gradient(&self, x: &[T], dx: &[T], grad: &mut [T]);
}

/// This trait provides an interface for retrieving the energy Hessian just like
/// `EnergyHessianIndicesValues`, however the indices and values are combined together into
/// the `MatrixElementTriplet` type.
pub trait EnergyHessian {
    /// The number of non-zeros in the Hessian matrix of the energy.
    fn energy_hessian_size(&self) -> usize;
    /// Compute the Hessian row and column indices of the Hessian matrix non-zero values.
    /// The `offset` parameter positions this energy Hessian product
    /// within a global Hessian matrix specified by the user.
    fn energy_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    );

    /// Compute the Hessian matrix values corresponding to their positions in the matrix returned
    /// by `energy_hessian_indices` or `energy_hessian_indices_offset`.
    ///
    ///   - `x` is the variable expected by the specific energy for the previous configuration. For
    /// example elastic energy expects position while momentum energy expects velocity.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    /// differential of `x` but it often is.
    ///
    /// This derivative is with respect to `dx`.
    fn energy_hessian_values<T: Scalar>(&self, x: &[T], dx: &[T], values: &mut [T]);

    /// Compute the Hessian row and column indices of the Hessian matrix non-zero values.
    fn energy_hessian_indices(&self, indices: &mut [MatrixElementIndex]) {
        self.energy_hessian_indices_offset((0, 0).into(), indices)
    }

    /// Compute the Hessian row and column indices of the Hessian matrix non-zero values into two
    /// separate arrays.
    fn energy_hessian_rows_cols<I: FromPrimitive + Send>(&self, rows: &mut [I], cols: &mut [I]) {
        self.energy_hessian_rows_cols_offset((0, 0).into(), rows, cols);
    }

    /// Compute the Hessian row and column indices of the Hessian matrix non-zero values into two
    /// separate arrays.
    /// The `offset` parameter positions this energy Hessian
    /// within a global Hessian matrix specified by the user.
    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        let n = self.energy_hessian_size();
        let mut indices = unsafe { vec![::std::mem::uninitialized(); n] };
        self.energy_hessian_indices_offset(offset, indices.as_mut_slice());
        for (MatrixElementIndex { row, col }, (r, c)) in indices
            .into_iter()
            .zip(rows.iter_mut().zip(cols.iter_mut()))
        {
            *r = I::from_usize(row).unwrap();
            *c = I::from_usize(col).unwrap();
        }
    }

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the Hessian matrix triplets.
    /// The `offset` parameter positions this energy Hessian within a global Hessian matrix
    /// specified by the user.
    ///
    ///   - `x` is the variable expected by the specific energy for the previous configuration. For
    /// example elastic energy expects position while momentum energy expects velocity.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    /// differential of `x` but it often is.
    ///
    /// This derivative is with respect to `dx`.
    fn energy_hessian_offset<T: Scalar>(
        &self,
        x: &[T],
        dx: &[T],
        offset: MatrixElementIndex,
        triplets: &mut [MatrixElementTriplet<T>],
    ) {
        let n = self.energy_hessian_size();
        let mut indices = unsafe { vec![::std::mem::uninitialized(); n] };
        self.energy_hessian_indices_offset(offset, indices.as_mut_slice());
        let mut values = unsafe { vec![::std::mem::uninitialized(); n] };
        self.energy_hessian_values(x, dx, values.as_mut_slice());
        for (trip, (idx, val)) in triplets.iter_mut().zip(indices.iter().zip(values.iter())) {
            *trip = MatrixElementTriplet::new(idx.row, idx.col, *val);
        }
    }

    /// Compute the Hessian matrix triplets.
    ///
    ///   - `x` is the variable expected by the specific energy for the previous configuration. For
    /// example elastic energy expects position while momentum energy expects velocity.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    /// differential of `x` but it often is.
    ///
    /// This derivative is with respect to `dx`.
    fn energy_hessian<T: Scalar>(&self, x: &[T], dx: &[T], triplets: &mut [MatrixElementTriplet<T>]) {
        self.energy_hessian_offset(x, dx, (0, 0).into(), triplets)
    }
}
