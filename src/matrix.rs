//! This module provides an interface for working with sparse matrix structures such as those
//! provided by third party optimizers and linear algebra libraries.

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
