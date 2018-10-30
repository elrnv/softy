//! This module provides an interface for working with sparse matrix structures such as those
//! provided by third party optimizers and linear algebra libraries.

use std::ops::Add;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MatrixElementIndex {
    pub row: usize,
    pub col: usize,
}

impl From<(usize,usize)> for MatrixElementIndex {
    fn from((row, col): (usize,usize)) -> MatrixElementIndex {
        MatrixElementIndex { row, col }
    }
}

impl Add for MatrixElementIndex {
    type Output = MatrixElementIndex;

    fn add(mut self, other: Self) -> MatrixElementIndex {
        self.row += other.row;
        self.col += other.col;
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
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
