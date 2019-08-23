//!
//! Common matrix types and operations.
//!
use super::*;
use std::ops::{Add, AddAssign, Mul};

pub type DiagonalMatrix3<S = Vec<f64>> = Tensor<Chunked3<S>>;
pub type DiagonalMatrix3View<'a> = DiagonalMatrix3<&'a [f64]>;

impl<S: Set> DiagonalMatrix3<S> {
    pub fn num_cols(&self) -> usize {
        self.data.len()
    }
    pub fn num_rows(&self) -> usize {
        self.data.len()
    }
}

/// Sparse-row sparse-column 3x3 block matrix.
pub type SSMatrix3<S = Vec<f64>, I = Vec<usize>> = Tensor<
    Sparse<
        Chunked<Sparse<Chunked3<Chunked3<S>>, std::ops::Range<usize>, I>, I>,
        std::ops::Range<usize>,
        I,
    >,
>;

pub type SSMatrix3View<'a> = SSMatrix3<&'a [f64], &'a [usize]>;

impl<S, I> SSMatrix3<S, I> {
    pub fn num_cols(&self) -> usize {
        self.data.data().data().selection().data.distance()
    }
    pub fn num_rows(&self) -> usize {
        self.data.selection().data.distance()
    }
}

/// Dense-row sparse-column row-major 3x3 block matrix.
pub type DSMatrix3<S = Vec<f64>, I = Vec<usize>> =
    Tensor<Chunked<Sparse<Chunked3<Chunked3<S>>, std::ops::Range<usize>, I>, I>>;

pub type DSMatrix3View<'a> = DSMatrix3<&'a [f64], &'a [usize]>;

impl<S: Set, I: std::borrow::Borrow<[usize]>> DSMatrix3<S, I> {
    pub fn num_cols(&self) -> usize {
        self.data.data().selection().data.distance()
    }
    pub fn num_rows(&self) -> usize {
        self.data.len()
    }
}

impl From<DiagonalMatrix3> for DSMatrix3 {
    fn from(diag: DiagonalMatrix3) -> DSMatrix3 {
        // Need to convert each triplet in diag into a diagonal 3x3 matrix.
        // Each block is essentially [x, 0, 0, 0, y, 0, 0, 0, z].
        let data: Chunked3<Vec<_>> = diag
            .data
            .iter()
            .map(|&[x, y, z]| [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]])
            .collect();

        let num_cols = diag.num_cols();
        Tensor::new(Chunked::from_sizes(
            vec![1; diag.num_rows()], // One element in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(data.into_flat())),
            ),
        ))
    }
}

impl Add<DiagonalMatrix3View<'_>> for SSMatrix3View<'_> {
    type Output = DSMatrix3;
    fn add(self, rhs: DiagonalMatrix3View<'_>) -> Self::Output {
        let rhs = rhs.data;
        let num_rows = self.num_rows();
        let num_cols = self.num_cols();

        let lhs_nnz = self.data.storage().len();
        let rhs_nnz = rhs.storage().len();
        let num_non_zero_blocks = lhs_nnz + rhs_nnz;

        let mut non_zero_row_offsets = vec![num_non_zero_blocks; num_rows + 1];
        non_zero_row_offsets[0] = 0;

        let mut out = Chunked::from_offsets(
            non_zero_row_offsets,
            Sparse::from_dim(
                vec![0; num_non_zero_blocks], // Pre-allocate column index vec.
                num_cols,
                Chunked3::from_flat(Chunked3::from_flat(vec![0.0; num_non_zero_blocks * 9])),
            ),
        );

        let mut rhs_iter = rhs.iter().enumerate();

        let add_diagonal_entry = |out: Chunked3<&mut [f64]>, entry: &[f64; 3]| {
            let out_mtx = out.into_arrays();
            out_mtx[0][0] += entry[0];
            out_mtx[1][1] += entry[1];
            out_mtx[2][2] += entry[2];
        };

        let mut count = 0;
        for (sparse_row_idx, row_l, _) in self.data.iter() {
            let (row_idx, rhs_entry) = loop {
                if let Some((row_idx, entry)) = rhs_iter.next() {
                    if row_idx < sparse_row_idx {
                        let out_row = out.view_mut().isolate(row_idx);
                        let (idx, out_col, _) = out_row.isolate(0);
                        *idx = count;
                        add_diagonal_entry(out_col, entry);
                        out.transfer_forward_all_but(row_idx, 1);
                        count += 1;
                    } else {
                        break (row_idx, entry);
                    }
                } else {
                    assert!(false, "RHS ran out of entries");
                }
            };

            assert_eq!(sparse_row_idx, row_idx);
            // Copy row from lhs and add or add to the diagonal entry.

            let mut count_out_cols = 0;
            let mut prev_col_idx = 0;
            for (col_idx, col, _) in row_l.iter() {
                if col_idx > row_idx && prev_col_idx < row_idx {
                    let out_row = out.view_mut().isolate(row_idx);
                    // Additional diagonal entry, add rhs entry before adding
                    // subsequent entries from lhs to preserve order of indices.
                    let (out_col_idx, out_col, _) = out_row.isolate(count_out_cols);
                    add_diagonal_entry(out_col, rhs_entry);
                    *out_col_idx = row_idx;
                    count_out_cols += 1;
                }

                let out_row = out.view_mut().isolate(row_idx);
                let (out_col_idx, mut out_col, _) = out_row.isolate(count_out_cols);
                out_col.copy_from_flat(*col.data());
                *out_col_idx = col_idx;
                if col_idx == row_idx {
                    add_diagonal_entry(out_col, rhs_entry);
                }

                prev_col_idx = col_idx;
                count_out_cols += 1;
            }

            // Truncate the current row to fit.
            out.transfer_forward_all_but(row_idx, count_out_cols);
            count += count_out_cols;
        }

        Tensor::new(out)
    }
}

impl AddAssign<Tensor<Chunked3<&[f64]>>> for Tensor<SubsetView<'_, Chunked3<&mut [f64]>>> {
    fn add_assign(&mut self, other: Tensor<Chunked3<&[f64]>>) {
        for (out, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *out = (geo::math::Vector3(*out) + geo::math::Vector3(b)).into();
        }
    }
}

impl AddAssign<Tensor<SubsetView<'_, Chunked3<&[f64]>>>>
    for Tensor<SubsetView<'_, SubsetView<'_, Chunked3<&mut [f64]>>>>
{
    fn add_assign(&mut self, other: Tensor<SubsetView<'_, Chunked3<&[f64]>>>) {
        for (out, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *out = (geo::math::Vector3(*out) + geo::math::Vector3(b)).into();
        }
    }
}

impl AddAssign<Tensor<Chunked3<&[f64]>>>
    for Tensor<SubsetView<'_, SubsetView<'_, Chunked3<&mut [f64]>>>>
{
    fn add_assign(&mut self, other: Tensor<Chunked3<&[f64]>>) {
        for (out, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *out = (geo::math::Vector3(*out) + geo::math::Vector3(b)).into();
        }
    }
}

impl Mul<Tensor<Chunked3<&[f64]>>> for DiagonalMatrix3View<'_> {
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, other: Tensor<Chunked3<&[f64]>>) -> Self::Output {
        let mut out = crate::soap::ToOwned::to_owned(other.data);
        for (&b, out) in self.data.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        Tensor::new(out)
    }
}
