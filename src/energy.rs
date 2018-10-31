/*!
 * Any energy interface that can be implemented by any quantity that has first and
 * second order derivatives. Implementing this interface allows the quantity to be used in an
 * optimization solver. Becasue these functions may be called many times in an inner loop it is
 * advised for implementers to reuse allocated memory as much as possible. For this reason the
 * trains in this module take a mutable reference to `self` instead of an immutable one.
 */

use crate::geo::math::Scalar;

use crate::matrix::{MatrixElementIndex, MatrixElementTriplet};

/// Energy trait. This trait provides the energy value that, for instance, may be used in the
/// objective function for an optimization algorithm.
pub trait Energy<T: Scalar> {
    /// Compute the energy of the current configuration.
    fn energy(&mut self, x: &[T]) -> T;
}

/// The energy gradient is required for optimization methods that require first order derivative
/// information, like Gradient Descent for instance.
pub trait EnergyGradient2<T: Scalar> {
    /// Compute the change in energy with respect to change in configuration and add it to the
    /// given slice of global gradient values.
    fn add_energy_gradient(&self, x: &[T], grad: &mut [T]);
}

/// The energy Hessian provides second order information for optimization methods like
/// Newton-Raphson. This trait provides the energy Hessian in terms of non-zero values provided by
/// `energy_hessian_values` and indexed by a slice of `MatrixElementIndex`es, which is provided by
/// `energy_hessian_indices`.
pub trait EnergyHessian<T: Scalar> {
    /// The number of non-zeros in the Hessian matrix of the energy.
    fn energy_hessian_size(&self) -> usize;
    /// Compute the Hessian row and column indices of the Hessian matrix values.
    fn energy_hessian_indices(&mut self) -> &[MatrixElementIndex];
    /// Compute the Hessian matrix values corresponding to their positions in the matrix returned
    /// by `energy_hessian_indices`. This means that the vector returned from this function must
    /// have the same length as the vector returned by `energy_hessian_indices`.
    fn energy_hessian_values(&mut self, x: &[T]) -> &[T];
}

/// This trait provides an interface for retrieving the energy Hessian just like
/// `EnergyHessianIndicesValues`, however the indices and values are combined together into
/// the `MatrixElementTriplet` type.
pub trait EnergyHessian2<T: Scalar> {
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
    fn energy_hessian_values(&self, x: &[T], values: &mut [T]);

    /// Compute the Hessian row and column indices of the Hessian matrix non-zero values.
    fn energy_hessian_indices(&self, indices: &mut [MatrixElementIndex]) {
        self.energy_hessian_indices_offset((0, 0).into(), indices)
    }

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the Hessian matrix triplets.
    /// The `offset` parameter positions this energy Hessian
    /// within a global Hessian matrix specified by the user.
    fn energy_hessian_offset(
        &self,
        x: &[T],
        offset: MatrixElementIndex,
        triplets: &mut [MatrixElementTriplet<T>],
    ) {
        let n = self.energy_hessian_size();
        let mut indices = unsafe { vec![::std::mem::uninitialized(); n] };
        self.energy_hessian_indices_offset(offset, indices.as_mut_slice());
        let mut values = unsafe { vec![::std::mem::uninitialized(); n] };
        self.energy_hessian_values(x, values.as_mut_slice());
        for (trip, (idx, val)) in triplets.iter_mut().zip(indices.iter().zip(values.iter())) {
            *trip = MatrixElementTriplet::new(idx.row, idx.col, *val);
        }
    }

    /// Compute the Hessian matrix triplets.
    fn energy_hessian(&self, x: &[T], triplets: &mut [MatrixElementTriplet<T>]) {
        self.energy_hessian_offset(x, (0, 0).into(), triplets)
    }
}

/// Some optimizers require only the energy Hessian product with another vector. This trait
/// provides an interface for such applications.
pub trait EnergyHessianProduct<T: Scalar> {
    /// Compute the product of Hessian and a given vector `dx`.
    fn energy_hessian_product(&mut self, x: &[T], dx: &[T]) -> &[T];
}
