#![cfg(feature = "sprs")]

//! A compatibility layer with the `sprs` sparse linear algebra library.

use super::*;
use std::convert::AsRef;

impl<S, I> Into<sprs::CsMat<f64>> for DSMatrix3<S, I>
where
    // Needed for num_cols/num_rows
    S: Set,
    I: AsRef<[usize]>,
    // Needed for view
    DSMatrix3<S, I>: for<'a> View<'a, Type = DSMatrix3View<'a>>,
{
    fn into(self) -> sprs::CsMat<f64> {
        let num_rows = self.num_rows();
        let num_cols = self.num_cols();

        let values = self.view().data.into_flat().as_ref().to_vec();

        let (rows, cols) = {
            let view = self.view().data;
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

        sprs::TriMat::from_triplets((3 * num_rows, 3 * num_cols), rows, cols, values).to_csr()
    }
}
