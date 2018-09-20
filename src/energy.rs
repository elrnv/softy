use geo::math::{Scalar, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct MatrixElementIndex {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatrixElementTriplet<T> {
    pub idx: MatrixElementIndex,
    pub val: T,
}

impl<T> MatrixElementTriplet<T> {
    pub fn new(row: usize, col: usize, val: T) -> Self {
        MatrixElementTriplet {
            idx: MatrixElementIndex { row, col },
            val,
        }
    }
}

/// Energy trait. Any energy interface that can be implemented by any quantity that has first and
/// second order derivatives. Implementing this interface allows the quantity to be used in an
/// optimization solver. Becasue these functions may be called many times in an inner loop it is
/// advised for implementers to reuse allocated memory as much as possible.
pub trait Energy<T: Scalar> {
    /// Compute the energy of the current configuration.
    fn energy(&mut self) -> T;
    /// Compute the change in energy with respect to change in configuration.
    fn energy_gradient(&mut self) -> &[Vector3<T>] {
        unimplemented!();
    }
    /// The number of non-zeros in the hessian matrix of the energy wrt. configuration change.
    fn energy_hessian_size(&self) -> usize {
        unimplemented!();
    }
    /// Compute the hessian row and column indices of the hessian matrix values.
    fn energy_hessian_indices(&mut self) -> &[MatrixElementIndex] {
        unimplemented!();
    }
    /// Compute the hessian matrix values corresponding to their positions in the matrix returned
    /// by `energy_hessian_indices`. This means that the vector returned from this function must
    /// have the same length as the vector returned by `energy_hessian_indices`.
    fn energy_hessian_values(&mut self) -> &[T] {
        unimplemented!();
    }
    /// Compute the hessian matrix in triplet form. This effectively computes the matrix row and
    /// column indices as returned by `energy_hessian_indices` as well as the corresponding values
    /// returned by `energy_hessian_values`.
    fn energy_hessian(&mut self) -> &[MatrixElementTriplet<T>] {
        unimplemented!();
    }
}
