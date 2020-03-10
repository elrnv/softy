//! This module provides an interface for specifying constraints to an optimization solver.

use tensr::Scalar;

use crate::matrix::{MatrixElementIndex, MatrixElementTriplet};
use crate::Error;

/// Constraint trait specifies the constraint function.
/// The lifetime `'a` tracks potential borrows in `Self::Input`.
pub trait Constraint<'a, T: Scalar> {
    type Input;
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
    fn constraint(&mut self, x: Self::Input, dx: Self::Input, value: &mut [T]);
}

/// The constraint Jacobian. Required for optimizers that use first order derivative information.
/// This trait requires indice and values of the sparse Jacobian matrix triplets to be specified
/// separately.
pub trait ConstraintJacobian<'a, T: Scalar>: Constraint<'a, T> {
    /// The number of non-zeros in the Jacobian matrix of the constraint provided by the
    /// `constraint_jacobian_indices` and `constraint_jacobian_values` functions.
    fn constraint_jacobian_size(&self) -> usize;

    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
    fn constraint_jacobian_indices_iter<'b>(
        &'b self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'b>, Error>;

    /// Compute the values of the constraint Jacobian.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_jacobian_values(
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        values: &mut [T],
    ) -> Result<(), Error>;

    /*
     * Below are convenience functions for auxiliary applications. Users should provide custom
     * implementations if performance is important.
     */

    /// Compute the change in the constraint function with respect to change in configuration.
    fn constraint_jacobian_offset(
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        offset: MatrixElementIndex,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        let n = self.constraint_jacobian_size();
        let mut values = vec![T::zero(); n];
        self.constraint_jacobian_values(x, dx, values.as_mut_slice())?;
        let indices_iter = self
            .constraint_jacobian_indices_iter()?
            .map(|idx| idx + offset);
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
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        self.constraint_jacobian_offset(x, dx, (0, 0).into(), triplets)
    }
}

///// Trait defining a constraint Jacobian for ArrayFire.
//pub trait ConstraintJacobianAF<'a>: Constraint<'a, f64> + ConstraintJacobian<'a, f64> {
//    /// Construct ArrayFire matrix for the Constraint Jacobian.
//    fn constraint_jacobian_af(
//        &self,
//        x: Self::Input,
//        dx: Self::Input,
//    ) -> Result<af::Array<f64>, Error> {
//        let nnz = self.constraint_jacobian_size();
//        let mut rows = vec![0i32; nnz];
//        let mut cols = vec![0i32; nnz];
//
//        let indices_iter = self.constraint_jacobian_indices_iter()?;
//
//        for (MatrixElementIndex { row, col }, (r, c)) in
//            indices_iter.zip(rows.iter_mut().zip(cols.iter_mut()))
//        {
//            *r = row as i32;
//            *c = col as i32;
//        }
//
//        let mut values = vec![0.0f64; nnz];
//        self.constraint_jacobian_values(x, dx, &mut values)?;
//
//        // Build ArrayFire matrix
//        let nnz = nnz as u64;
//        let num_rows = self.constraint_size() as u64;
//        let num_cols = x.len() as u64;
//        af::Dim4::new(&[num_rows, num_cols, 1, 1]);
//
//        let values = af::Array::new(&values, af::Dim4::new(&[nnz, 1, 1, 1]));
//        let row_indices = af::Array::new(&rows, af::Dim4::new(&[nnz, 1, 1, 1]));
//        let col_indices = af::Array::new(&cols, af::Dim4::new(&[nnz, 1, 1, 1]));
//
//        Ok(af::sparse(
//            num_rows,
//            num_cols,
//            &values,
//            &row_indices,
//            &col_indices,
//            af::SparseFormat::COO,
//        ))
//    }
//}
//
//impl<'a, T> ConstraintJacobianAF<'a> for T where T: Constraint<'a, f64> + ConstraintJacobian<'a, f64>
//{}

/// This trait provides a way to compute the constraint Hessian multiplied by a vector of Lagrange
/// multipliers. Strictly speaking this is not the actual constraint Hessian, however if the
/// constraint function had dimension 1 and `lambda` was given to be 1, then this trait provides
/// the true Hessian of the constraint function.
/// The "Hessian" matrix is provided by a series of sparse indices and values via two separate
/// functions.
pub trait ConstraintHessian<'a, T: Scalar>: ConstraintJacobian<'a, T> {
    type InputDual;
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
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        lambda: Self::InputDual,
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error>;

    /// Compute the Hessian row and column indices of the matrix resulting from the constraint
    /// Hessian multiplied by the Lagrange multiplier vector.
    fn constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'b>, Error>;

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
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        lambda: Self::InputDual,
        offset: MatrixElementIndex,
        scale: T,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        let n = self.constraint_hessian_size();
        let mut values = vec![T::zero(); n];
        self.constraint_hessian_values(x, dx, lambda, scale, values.as_mut_slice())?;
        let indices_iter = self
            .constraint_hessian_indices_iter()?
            .map(|idx| idx + offset);
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
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        lambda: Self::InputDual,
        scale: T,
        triplets: &mut [MatrixElementTriplet<T>],
    ) -> Result<(), Error> {
        self.constraint_hessian_offset(x, dx, lambda, (0, 0).into(), scale, triplets)
    }
}

///// Trait implementing the constraint Hessian for ArrayFire.
//trait ConstraintHessianMatrix<'a>:
//    ConstraintHessian<'a, f64, InputDual = &'a [f64]> + Constraint<'a, f64, Input = &'a [f64]>
//{
//    /// Construct ArrayFire matrix.
//    fn constraint_hessian_af(
//        &self,
//        x: &[f64],
//        dx: &[f64],
//        lambda: &[f64],
//        scale: f64,
//    ) -> Result<af::Array<f64>, Error> {
//        let nnz = self.constraint_hessian_size();
//        let mut rows = vec![0i32; nnz];
//        let mut cols = vec![0i32; nnz];
//
//        let indices_iter = self.constraint_hessian_indices_iter()?;
//
//        for (MatrixElementIndex { row, col }, (r, c)) in
//            indices_iter.zip(rows.iter_mut().zip(cols.iter_mut()))
//        {
//            *r = row as i32;
//            *c = col as i32;
//        }
//
//        let mut values = vec![0.0; nnz];
//        self.constraint_hessian_values(x, dx, lambda, scale, &mut values)?;
//
//        // Build ArrayFire matrix
//        let nnz = nnz as u64;
//        let num_rows = x.len() as u64;
//        let num_cols = x.len() as u64;
//
//        let values = af::Array::new(&values, af::Dim4::new(&[nnz, 1, 1, 1]));
//        let row_indices = af::Array::new(&rows, af::Dim4::new(&[nnz, 1, 1, 1]));
//        let col_indices = af::Array::new(&cols, af::Dim4::new(&[nnz, 1, 1, 1]));
//
//        Ok(af::sparse(
//            num_rows,
//            num_cols,
//            &values,
//            &row_indices,
//            &col_indices,
//            af::SparseFormat::COO,
//        ))
//    }
//
//    /// Construct ArrayFire matrix.
//    fn constraint_hessian_sprs(
//        &self,
//        x: &[f64],
//        dx: &[f64],
//        lambda: &[f64],
//        scale: f64,
//    ) -> Result<sprs::CsMat<f64>, Error> {
//        let indices_iter = self.constraint_hessian_indices_iter()?;
//        let (rows, cols) = indices_iter
//            .map(|MatrixElementIndex { row, col }| (row, col))
//            .unzip();
//
//        let mut values = vec![0.0; self.constraint_hessian_size()];
//        self.constraint_hessian_values(x, dx, lambda, scale, &mut values)?;
//
//        let num_rows = x.len();
//        let num_cols = x.len();
//        Ok(sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, values).to_csr())
//    }
//}
//
//impl<'a, T> ConstraintHessianMatrix<'a> for T where
//    T: ConstraintHessian<'a, f64, InputDual = &'a [f64]> + Constraint<'a, f64, Input = &'a [f64]>
//{
//}
