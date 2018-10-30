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
pub trait EnergyGradient<T: Scalar> {
    /// Compute the change in energy with respect to change in configuration.
    fn energy_gradient(&mut self, x: &[T]) -> &[T];
}

/// This trait specifies how many non-zeros are stored in the Hessian of the energy. This is not to
/// be confused with the Hessian matrix size, which is typically smaller than the number of
/// non-zeros in the Hessian.
pub trait EnergyHessianSize {
    /// The number of non-zeros in the Hessian matrix of the energy.
    fn energy_hessian_size(&self) -> usize;
}

/// The energy Hessian provides second order information for optimization methods like
/// Newton-Raphson. This trait provides the energy Hessian in terms of non-zero values provided by
/// `energy_hessian_values` and indexed by a slice of `MatrixElementIndex`es, which is provided by
/// `energy_hessian_indices`.
pub trait EnergyHessianIndicesValues<T: Scalar>: EnergyHessianSize {
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
pub trait EnergyHessian<T: Scalar>: EnergyHessianSize {
    /// Compute the Hessian matrix in triplet form. This effectively computes the matrix row and
    /// column indices as returned by `energy_hessian_indices` as well as the corresponding values
    /// returned by `energy_hessian_values`.
    fn energy_hessian(&mut self, x: &[T]) -> &[MatrixElementTriplet<T>];
}

/// Some optimizers require only the energy Hessian product with another vector. This trait
/// provides an interface for such applications.
pub trait EnergyHessianProduct<T: Scalar> {
    /// Compute the product of Hessian and a given vector `dx`.
    fn energy_hessian_product(&mut self, x: &[T], dx: &[T]) -> &[T];
}
