//! This module provides an interface for specifying constraints to an optimization solver.

use geo::math::Scalar;

use crate::matrix::{MatrixElementIndex, MatrixElementTriplet};
use crate::Error;

/// Constraint trait specifies the constraint function.
pub trait Constraint<T: Scalar> {
    /// The dimension of the constraint function.
    fn constraint_size(&self) -> usize;
    /// The lower and upper bounds of the constraint function in that order.
    /// Unequal lower and upper bounds correspond to inequality constraints.
    fn constraint_bounds(&self) -> (Vec<T>, Vec<T>);
    /// Compute the constraint function of the current configuration.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint(&self, x: &[T], dx: &[T], value: &mut [T]);
}

/// The constraint Jacobian. Required for optimizers that use first order derivative information.
/// This trait requires indice and values of the sparse Jacobian matrix triplets to be specified
/// separately.
pub trait ConstraintJacobian<T: Scalar> {
    /// The number of non-zeros in the Jacobian matrix of the constraint provided by the
    /// `constraint_jacobian_indices` and `constraint_jacobian_values` functions.
    fn constraint_jacobian_size(&self) -> usize;

    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error>;

    /// Compute the values of the constraint Jacobian.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_jacobian_values(&self, x: &[T], dx: &[T], values: &mut [T]) -> Result<(), Error>;

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the change in the constraint function with respect to change in configuration.
    fn constraint_jacobian_offset(
        &self,
        x: &[T],
        dx: &[T],
        offset: MatrixElementIndex,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        let n = self.constraint_jacobian_size();
        let indices_iter = self
            .constraint_jacobian_indices_iter()?
            .map(|idx| idx + offset);
        let mut values = unsafe { vec![::std::mem::uninitialized(); n] };
        self.constraint_jacobian_values(x, dx, values.as_mut_slice())?;
        for (trip, (idx, val)) in triplets.iter_mut().zip(indices_iter.zip(values.iter())) {
            *trip = MatrixElementTriplet::new(idx.row, idx.col, *val);
        }
        Ok(())
    }

    /// Compute the change in the constraint function with respect to change in configuration.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_jacobian(
        &self,
        x: &[T],
        dx: &[T],
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        self.constraint_jacobian_offset(x, dx, (0, 0).into(), triplets)
    }
}

/// Trait defining a constraint Jacobian for ArrayFire.
pub trait ConstraintJacobianAF: Constraint<f64> + ConstraintJacobian<f64> {
    /// Construct ArrayFire matrix for the Constraint Jacobian.
    fn constraint_jacobian_af(&self, x: &[f64], dx: &[f64]) -> Result<af::Array<f64>, Error> {
        let nnz = self.constraint_jacobian_size();
        let mut rows = vec![0i32; nnz];
        let mut cols = vec![0i32; nnz];

        let indices_iter = self.constraint_jacobian_indices_iter()?;

        for (MatrixElementIndex { row, col }, (r, c)) in
            indices_iter.zip(rows.iter_mut().zip(cols.iter_mut()))
        {
            *r = row as i32;
            *c = col as i32;
        }

        let mut values = vec![0.0f64; nnz];
        self.constraint_jacobian_values(x, dx, &mut values)?;

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = self.constraint_size() as u64;
        let num_cols = x.len() as u64;
        af::Dim4::new(&[num_rows, num_cols, 1, 1]);

        let values = af::Array::new(&values, af::Dim4::new(&[nnz, 1, 1, 1]));
        let row_indices = af::Array::new(&rows, af::Dim4::new(&[nnz, 1, 1, 1]));
        let col_indices = af::Array::new(&cols, af::Dim4::new(&[nnz, 1, 1, 1]));

        Ok(af::sparse(
            num_rows,
            num_cols,
            &values,
            &row_indices,
            &col_indices,
            af::SparseFormat::COO,
        ))
    }
}

impl<T> ConstraintJacobianAF for T where T: Constraint<f64> + ConstraintJacobian<f64> {}

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
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_hessian_values(
        &self,
        x: &[T],
        dx: &[T],
        lambda: &[T],
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error>;

    /// Compute the Hessian row and column indices of the matrix resulting from the constraint
    /// Hessian multiplied by the Lagrange multiplier vector.
    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error>;

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the constraint Hessian matrix multiplied by Lagrange multipliers `lambda`, in
    /// triplet form. This effectively computes the matrix row and
    /// column indices as returned by `constraint_hessian_indices` as well as the corresponding values
    /// returned by `constraint_hessian_values`.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_hessian_offset(
        &self,
        x: &[T],
        dx: &[T],
        lambda: &[T],
        offset: MatrixElementIndex,
        scale: T,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        let n = self.constraint_hessian_size();
        let indices_iter = self
            .constraint_hessian_indices_iter()?
            .map(|idx| idx + offset);
        let mut values = unsafe { vec![::std::mem::uninitialized(); n] };
        self.constraint_hessian_values(x, dx, lambda, scale, values.as_mut_slice())?;
        for (trip, (idx, val)) in triplets.iter_mut().zip(indices_iter.zip(values.iter())) {
            *trip = MatrixElementTriplet::new(idx.row, idx.col, *val);
        }
        Ok(())
    }

    /// Compute the constraint Hessian matrix multiplied by Lagrange multipliers `lambda`, in
    /// triplet form. This effectively computes the matrix row and
    /// column indices as returned by `constraint_hessian_indices` as well as the corresponding values
    /// returned by `constraint_hessian_values`.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_hessian(
        &self,
        x: &[T],
        dx: &[T],
        lambda: &[T],
        scale: T,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        self.constraint_hessian_offset(x, dx, lambda, (0, 0).into(), scale, triplets)
    }
}

/// Trait implementing the constraint Hessian for ArrayFire.
trait ConstraintHessianAF: ConstraintHessian<f64> {
    /// Construct ArrayFire matrix.
    fn constraint_hessian_af(
        &self,
        x: &[f64],
        dx: &[f64],
        lambda: &[f64],
        scale: f64,
    ) -> Result<af::Array<f64>, Error> {
        let nnz = self.constraint_hessian_size();
        let mut rows = vec![0i32; nnz];
        let mut cols = vec![0i32; nnz];

        let indices_iter = self.constraint_hessian_indices_iter()?;

        for (MatrixElementIndex { row, col }, (r, c)) in
            indices_iter.zip(rows.iter_mut().zip(cols.iter_mut()))
        {
            *r = row as i32;
            *c = col as i32;
        }

        let mut values = vec![0.0; nnz];
        self.constraint_hessian_values(x, dx, lambda, scale, &mut values)?;

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = x.len() as u64;
        let num_cols = x.len() as u64;
        af::Dim4::new(&[num_rows, num_cols, 1, 1]);

        let values = af::Array::new(&values, af::Dim4::new(&[nnz, 1, 1, 1]));
        let row_indices = af::Array::new(&rows, af::Dim4::new(&[nnz, 1, 1, 1]));
        let col_indices = af::Array::new(&cols, af::Dim4::new(&[nnz, 1, 1, 1]));

        Ok(af::sparse(
            num_rows,
            num_cols,
            &values,
            &row_indices,
            &col_indices,
            af::SparseFormat::COO,
        ))
    }
}

impl<T> ConstraintHessianAF for T where T: ConstraintHessian<f64> {}
