//! This module provides an interface for specifying constraints to an optimization solver.

use crate::geo::math::Scalar;

use crate::matrix::{MatrixElementIndex, MatrixElementTriplet};

/// Constraint trait specifies the constraint function.
pub trait Constraint<T: Scalar> {
    /// The dimension of the constraint function.
    fn constraint_size(&self) -> usize;
    /// The lower and upper bounds of the constraint function in that order.
    /// Unequal lower and upper bounds correspond to inequality constraints.
    fn constraint_bounds(&self) -> (Vec<T>, Vec<T>);
    /// Compute the constraint function of the current configuration.
    fn constraint(&mut self, x: &[T], value: &mut [T]);
}

/// The constraint Jacobian. Required for optimizers that use first order derivative information.
/// This trait requires indice and values of the sparse Jacobian matrix triplets to be specified
/// separately.
pub trait ConstraintJacobian<T: Scalar> {
    /// The number of non-zeros in the Jacobian matrix of the constraint provided by the
    /// `constraint_jacobian_indices` and `constraint_jacobian_values` functions.
    fn constraint_jacobian_size(&self) -> usize;

    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
    fn constraint_jacobian_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item =  MatrixElementIndex> + 'a>;

    /// Compute the values of the constraint Jacobian.
    fn constraint_jacobian_values(&self, x: &[T], values: &mut [T]);

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the change in the constraint function with respect to change in configuration.
    fn constraint_jacobian_offset(
        &self,
        x: &[T],
        offset: MatrixElementIndex,
        triplets: &mut [MatrixElementTriplet<T>],
    ) {
        let n = self.constraint_jacobian_size();
        let indices_iter = self.constraint_jacobian_indices_iter().map(|idx| idx + offset);
        let mut values = unsafe { vec![::std::mem::uninitialized(); n] };
        self.constraint_jacobian_values(x, values.as_mut_slice());
        for (trip, (idx, val)) in triplets.iter_mut().zip(indices_iter.zip(values.iter())) {
            *trip = MatrixElementTriplet::new(idx.row, idx.col, *val);
        }
    }

    /// Compute the change in the constraint function with respect to change in configuration.
    fn constraint_jacobian(&self, x: &[T], triplets: &mut [MatrixElementTriplet<T>]) {
        self.constraint_jacobian_offset(x, (0, 0).into(), triplets)
    }
}

/// This trait provides a way to compute the constraint Hessian multiplied by a vector of Lagrange
/// multipliers. Strictly speaking this is not the actual constraint Hessian, however if the
/// constraint function had dimension 1 and `lambda` was given to be 1, then this trait provides
/// the true Hessian of the constraint function.
/// The "Hessian" matrix is provided by a series of sparse indices and values via two separate
/// functions.
pub trait ConstraintHessian<T: Scalar> {
    /// The number of non-zeros in the Hessian matrix of the constraint.
    fn constraint_hessian_size(&self) -> usize;
    /// Compute the Hessian matrix values (multiplied by `lambda`) corresponding to their positions
    /// in the matrix returned by `constraint_hessian_indices`. This means that the vector returned
    /// from this function must have the same length as the vector returned by
    /// `constraint_hessian_indices`.
    fn constraint_hessian_values(&self, x: &[T], lambda: &[T], values: &mut [T]);

    /// Compute the Hessian row and column indices of the matrix resulting from the constraint
    /// Hessian multiplied by the Lagrange multiplier vector.
    fn constraint_hessian_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>;

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the constraint Hessian matrix multiplied by Lagrange multipliers `lambda`, in
    /// triplet form. This effectively computes the matrix row and
    /// column indices as returned by `constraint_hessian_indices` as well as the corresponding values
    /// returned by `constraint_hessian_values`.
    fn constraint_hessian_offset(
        &self,
        x: &[T],
        lambda: &[T],
        offset: MatrixElementIndex,
        triplets: &mut [MatrixElementTriplet<T>],
    ) {
        let n = self.constraint_hessian_size();
        let indices_iter = self.constraint_hessian_indices_iter().map(|idx| idx + offset);
        let mut values = unsafe { vec![::std::mem::uninitialized(); n] };
        self.constraint_hessian_values(x, lambda, values.as_mut_slice());
        for (trip, (idx, val)) in triplets.iter_mut().zip(indices_iter.zip(values.iter())) {
            *trip = MatrixElementTriplet::new(idx.row, idx.col, *val);
        }
    }

    /// Compute the constraint Hessian matrix multiplied by Lagrange multipliers `lambda`, in
    /// triplet form. This effectively computes the matrix row and
    /// column indices as returned by `constraint_hessian_indices` as well as the corresponding values
    /// returned by `constraint_hessian_values`.
    fn constraint_hessian(&self, x: &[T], lambda: &[T], triplets: &mut [MatrixElementTriplet<T>]) {
        self.constraint_hessian_offset(x, lambda, (0, 0).into(), triplets)
    }
}
