#![cfg(feature = "sprs")]

//! A compatibility layer with the `sprs` sparse linear algebra library.

use super::*;
use std::convert::AsRef;

impl<S, I> Into<sprs::CsMat<f64>> for DSBlockMatrix3<S, I>
where
    // Needed for num_cols/num_rows
    S: Set,
    I: Set + AsRef<[usize]>,
    // Needed for view
    Self: for<'a> View<'a, Type = DSBlockMatrix3View<'a>>,
{
    fn into(self) -> sprs::CsMat<f64> {
        let view = self.view();
        let num_rows = view.num_rows();
        let num_cols = view.num_cols();

        let view = view.data;
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
