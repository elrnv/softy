//! This module provides an interface for specifying constraints to an optimization solver.

use geo::math::{Scalar};

use matrix::{MatrixElementIndex, MatrixElementTriplet};

/// Constraint trait specifies the constraint function.
pub trait Constraint<T: Scalar> {
    /// The dimension of the constraint function.
    fn constraint_size(&self) -> usize;
    /// The lower bound of the constraint function. Unequal lower and upper bounds correspond to
    /// inequality constraints.
    fn constraint_lower_bound(&self) -> Vec<T>;
    /// The upper bound of the constraint function. Unequal lower and upper bounds correspond to
    /// inequality constraints.
    fn constraint_upper_bound(&self) -> Vec<T>;
    /// Compute the constraint function of the current configuration.
    fn constraint(&mut self, x: &[T]) -> &[T];
}

/// This trait specifies how many non-zeros are stored in the Jacobian of the constraint. This is not to
/// be confused with the Jacobian matrix size, which is typically smaller than the number of
/// non-zeros in the Jacobian.
pub trait ConstraintJacobianSize {
    /// The number of non-zeros in the Jacobian matrix of the constraint.
    fn constraint_jacobian_size(&self) -> usize;
}

/// The constraint jacobian. Required for optimizers that use first order derivative information.
pub trait ConstraintJacobian<T: Scalar>: ConstraintJacobianSize {
    /// Compute the change in the constraint function with respect to change in configuration.
    fn constraint_jacobian(&mut self, x: &[T]) -> &[MatrixElementTriplet<T>];
}

/// This trait specifies how many non-zeros are stored in the Hessian of the constraint after it
/// has been multiplied by a vector of Lagrange multipliers. This is not to
/// be confused with the Hessian matrix size, which is typically smaller than the number of
/// non-zeros in the Hessian.
pub trait ConstraintHessianSize {
    /// The number of non-zeros in the Hessian matrix of the energy.
    fn constraint_hessian_size(&self) -> usize;
}

/// This trait provides a way to compute the constraint Hessian multiplied by a vector of Lagrange
/// multipliers. Strictly speaking this is not the actual constraint Hessian, however if the
/// constraint function had dimension 1 and `lambda` was given to be 1, then this trait provides
/// the true Hessian of the constraint function.
/// The "Hessian" matrix is provided by a series of sparse indices and values via two separate
/// functions.
pub trait ConstraintHessianIndicesValues<T: Scalar>: ConstraintHessianSize {
    /// Compute the Hessian row and column indices of the matrix resulting from the constraint
    /// Hessian multiplied by the Lagrange multiplier vector.
    fn constraint_hessian_indices(&mut self) -> &[MatrixElementIndex];
    /// Compute the Hessian matrix values (multiplied by `lambda`) corresponding to their positions
    /// in the matrix returned by `constraint_hessian_indices`. This means that the vector returned
    /// from this function must have the same length as the vector returned by
    /// `constraint_hessian_indices`.
    fn constraint_hessian_values(&mut self, x: &[T], lambda: &[T]) -> &[T];
}

/// This trait provides a way to compute the constraint Hessian multiplied by a vector of Lagrange
/// multipliers. Strictly speaking this is not the actual constraint Hessian, however if the
/// constraint function had dimension 1 and `lambda` was given to be 1, then this trait provides
/// the true Hessian of the constraint function.
pub trait ConstraintHessian<T: Scalar>: ConstraintHessianSize {
    /// Compute the constraint Hessian matrix multiplied by Lagrange multipliers `lambda`, in
    /// triplet form. This effectively computes the matrix row and
    /// column indices as returned by `constraint_hessian_indices` as well as the corresponding values
    /// returned by `constraint_hessian_values` in the `ConstraintHessianIndicesValues` trait.
    fn constraint_hessian(&mut self, x: &[T], lambda: &[T]) -> &[MatrixElementTriplet<T>];
}
