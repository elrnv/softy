#![cfg(feature = "sprs")]

//! A compatibility layer with the `sprs` sparse linear algebra library.

use super::*;
use std::convert::AsRef;

impl<S, I> Into<sprs::CsMat<f64>> for DSBlockMatrixBase<S, I, U3, U3>
where
    // Needed for num_cols/num_rows
    S: Set,
    I: Set + AsRef<[usize]>,
    // Needed for view
    Self: for<'a> View<'a, Type = DSBlockMatrix3View<'a>>,
{
    fn into(self) -> sprs::CsMat<f64> {
        let view = self.view();
        let num_rows = view.num_total_rows();
        let num_cols = view.num_total_cols();

        let view = view.as_data();
        let values = view.clone().into_flat().as_ref().to_vec();

        let (rows, cols) = {
            view.into_iter()
                .enumerate()
                .flat_map(move |(row_idx, row)| {
                    row.into_iter().flat_map(move |(col_idx, _)| {
                        (0..3).flat_map(move |row| {
                            (0..3).map(move |col| (3 * row_idx + row, 3 * col_idx + col))
                        })
                    })
                })
                .unzip()
        };

        sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, values).to_csr()
    }
}

impl Into<sprs::CsMat<f64>> for DSMatrix {
    fn into(self) -> sprs::CsMat<f64> {
        let num_rows = self.num_rows();
        let num_cols = self.num_cols();

        let (rows, cols) = {
            self.as_data()
                .view()
                .into_iter()
                .enumerate()
                .flat_map(move |(row_idx, row)| {
                    row.into_iter().map(move |(col_idx, _)| (row_idx, col_idx))
                })
                .unzip()
        };

        let values = self.into_data().into_flat();

        sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, values).to_csr()
    }
}
