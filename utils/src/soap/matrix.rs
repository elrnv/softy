//!
//! Common matrix types and operations.
//!

mod sprs_compat;
pub use sprs_compat::*;

use super::*;
use chunked::Offsets;
use geo::math::{Matrix3, Vector3};
use std::convert::AsRef;
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
        Chunked<Sparse<Chunked3<Chunked3<S>>, std::ops::Range<usize>, I>, Offsets<I>>,
        std::ops::Range<usize>,
        I,
    >,
>;

pub type SSMatrix3View<'a> = SSMatrix3<&'a [f64], &'a [usize]>;

impl<S: Set, I> SSMatrix3<S, I> {
    pub fn num_cols(&self) -> usize {
        self.data.source().data().selection().target.distance()
    }
    pub fn num_rows(&self) -> usize {
        self.data.selection().target.distance()
    }
    pub fn transpose<'a>(&'a self) -> Transpose<SSMatrix3<S::Type, &'a [usize]>>
    where
        S: View<'a>,
        I: AsRef<[usize]>,
    {
        Transpose(View::view(self))
    }
}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: Set + AsRef<[usize]>> SSMatrix3<S, I> {
    #[cfg(debug_assertions)]
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_rows() * 3;
        let ncols = self.num_cols() * 3;

        let ciel = 10.0; //jac.max();
        let floor = -10.0; //jac.min();

        let img = ImageBuffer::from_fn(ncols as u32, nrows as u32, |c, r| {
            let val = self.coeff(r as usize, c as usize);
            let color = if val > 0.0 {
                [255, (255.0 * val / ciel) as u8, 0]
            } else if val < 0.0 {
                [0, (255.0 * (1.0 + val / floor)) as u8, 255]
            } else {
                [255, 0, 255]
            };
            image::Rgb(color)
        });

        img.save(path.as_ref())
            .expect("Failed to save matrix image.");
    }

    /// Get the value in the matrix at the given coordinates.
    pub fn coeff(&'a self, r: usize, c: usize) -> f64 {
        let view = self.view();
        if let Ok(row) = view
            .data
            .selection
            .indices
            .binary_search(&(r / 3))
            .map(|idx| view.data.source.isolate(idx))
        {
            row.selection
                .indices
                .binary_search(&(c / 3))
                .map(|idx| row.source.isolate(idx).at(r % 3)[c % 3])
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }
}

impl SSMatrix3 {
    pub fn from_triplets<It: Iterator<Item = (usize, usize)>>(
        index_iter: It,
        num_rows: usize,
        num_cols: usize,
        blocks: Chunked3<Chunked3<Vec<f64>>>,
    ) -> Self {
        let num_blocks = blocks.len();
        let mut rows = Vec::with_capacity(num_blocks);
        let mut cols = Vec::with_capacity(num_blocks);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col) in index_iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                rows.push(row);
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();
        rows.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(offsets, Sparse::from_dim(cols, num_cols, blocks));

        col_data.sort_chunks_by_index();

        Tensor::new(Sparse::from_dim(rows, num_rows, col_data))
    }
}

/// Dense-row sparse-column row-major 3x3 block matrix. Block version of CSR.
pub type DSMatrix3<S = Vec<f64>, I = Vec<usize>> =
    Tensor<Chunked<Sparse<Chunked3<Chunked3<S>>, std::ops::Range<usize>, I>, Offsets<I>>>;

pub type DSMatrix3View<'a> = DSMatrix3<&'a [f64], &'a [usize]>;

impl<S: Set, I: Set> DSMatrix3<S, I> {
    pub fn num_cols(&self) -> usize {
        self.data.data().selection().target.distance()
    }
    pub fn num_rows(&self) -> usize {
        self.data.len()
    }
}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: Set + AsRef<[usize]>> DSMatrix3<S, I> {
    pub fn transpose(&'a self) -> Transpose<DSMatrix3<S::Type, &'a [usize]>> {
        Transpose(View::view(self))
    }
    #[cfg(debug_assertions)]
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_rows() * 3;
        let ncols = self.num_cols() * 3;

        let ciel = 10.0; //jac.max();
        let floor = -10.0; //jac.min();

        let img = ImageBuffer::from_fn(ncols as u32, nrows as u32, |c, r| {
            let val = self.coeff(r as usize, c as usize);
            let color = if val > 0.0 {
                [255, (255.0 * val / ciel) as u8, 0]
            } else if val < 0.0 {
                [0, (255.0 * (1.0 + val / floor)) as u8, 255]
            } else {
                [255, 0, 255]
            };
            image::Rgb(color)
        });

        img.save(path.as_ref())
            .expect("Failed to save matrix image.");
    }

    /// Get the value in the matrix at the given coordinates.
    pub fn coeff(&'a self, r: usize, c: usize) -> f64 {
        let row = self.data.view().isolate(r / 3);
        row.selection
            .indices
            .binary_search(&(c / 3))
            .map(|idx| row.source.isolate(idx).at(r % 3)[c % 3])
            .unwrap_or(0.0)
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

        let lhs_nnz = self.data.source.data.source.len();
        let rhs_nnz = rhs.data().len();
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
                        *idx = row_idx; // Diagonal entry col_idx == row_idx
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
            *out = (Vector3(*out) + Vector3(b)).into();
        }
    }
}

impl AddAssign<Tensor<SubsetView<'_, Chunked3<&[f64]>>>>
    for Tensor<SubsetView<'_, SubsetView<'_, Chunked3<&mut [f64]>>>>
{
    fn add_assign(&mut self, other: Tensor<SubsetView<'_, Chunked3<&[f64]>>>) {
        for (out, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *out = (Vector3(*out) + Vector3(b)).into();
        }
    }
}

impl AddAssign<Tensor<Chunked3<&[f64]>>>
    for Tensor<SubsetView<'_, SubsetView<'_, Chunked3<&mut [f64]>>>>
{
    fn add_assign(&mut self, other: Tensor<Chunked3<&[f64]>>) {
        for (out, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *out = (Vector3(*out) + Vector3(b)).into();
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

impl<'a, Rhs> std::ops::Mul<Rhs> for SSMatrix3View<'_>
where
    Rhs: Into<Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>>,
{
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        assert_eq!(rhs.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_rows()]);
        for (row_idx, row, _) in self.iter() {
            for (col_idx, block, _) in row.iter() {
                let out =
                    Vector3(res[row_idx]) + Matrix3(*block.into_arrays()) * Vector3(rhs[col_idx]);
                res[row_idx] = out.into();
            }
        }

        Tensor::new(res)
    }
}

impl<'a, Rhs> std::ops::Mul<Rhs> for Transpose<SSMatrix3View<'_>>
where
    Rhs: Into<Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>>,
{
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        assert_eq!(rhs.len(), self.0.num_rows());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.0.num_cols()]);
        for (row_idx, row, _) in self.0.iter() {
            for (col_idx, block, _) in row.iter() {
                let out =
                    Vector3(res[col_idx]) + Matrix3(*block.into_arrays()) * Vector3(rhs[row_idx]);
                res[col_idx] = out.into();
            }
        }

        Tensor::new(res)
    }
}

impl<S> std::ops::MulAssign<DiagonalMatrix3<S>> for SSMatrix3
where
    S: Set + for<'a> View<'a, Type = &'a [f64]>,
{
    fn mul_assign(&mut self, rhs: DiagonalMatrix3<S>) {
        let rhs = View::view(&rhs);
        assert_eq!(rhs.data.len(), self.num_cols());
        for (_, mut row) in self.view_mut().iter_mut() {
            for (col_idx, mut block) in row.iter_mut() {
                let mass_vec = *rhs.data.at(*col_idx);
                for (block_row, &mass) in block.iter_mut().zip(mass_vec.iter()) {
                    *block_row = (geo::math::Vector3(*block_row) * mass).into();
                }
            }
        }
    }
}

impl std::ops::Mul<Transpose<SSMatrix3View<'_>>> for SSMatrix3View<'_> {
    type Output = SSMatrix3;
    fn mul(self, rhs: Transpose<SSMatrix3View>) -> Self::Output {
        let rhs_t = rhs.0;
        let num_rows = self.num_rows();
        let num_cols = rhs_t.num_rows();

        let lhs_nnz = self.data.source.data.source.len();
        let rhs_nnz = rhs_t.data.source.data.source.len();
        let num_non_zero_blocks = lhs_nnz + rhs_nnz;

        // Allocate enough offsets for all non-zero rows in self. and assign the
        // first row to contain all elements by setting all offsets to
        // num_non_zero_blocks except the first.
        let mut non_zero_row_offsets = vec![num_non_zero_blocks; self.len() + 1];
        non_zero_row_offsets[0] = 0;

        let mut out = Sparse::from_dim(
            self.indices().to_vec(),
            num_rows,
            Chunked::from_offsets(
                non_zero_row_offsets,
                Sparse::from_dim(
                    vec![0; num_non_zero_blocks], // Pre-allocate column index vec.
                    num_cols,
                    Chunked3::from_flat(Chunked3::from_flat(vec![0.0; num_non_zero_blocks * 9])),
                ),
            ),
        );

        let mut nz_row_idx = 0;
        for (row_idx, row_l, _) in self.iter() {
            let (_, out_row, _) = out.view_mut().isolate(nz_row_idx);
            let num_non_zero_blocks_in_row = rhs_t
                .view()
                .mul_sparse_matrix3_vector(Tensor::new(row_l), Tensor::new(out_row));

            // Truncate resulting row. This makes space for the next row in the output.
            if num_non_zero_blocks_in_row > 0 {
                // This row is non-zero, set the row index in the output.
                out.indices_mut()[nz_row_idx] = row_idx;
                // Truncate the current row to fit.
                out.source_mut()
                    .transfer_forward_all_but(nz_row_idx, num_non_zero_blocks_in_row);
                nz_row_idx += 1;
            }
        }

        // There may be fewer non-zero rows than in self. Truncate those.
        out.indices_mut().truncate(nz_row_idx);
        // Also truncate the entries in storage we didn't use.
        out.source_mut().trim();

        Tensor::new(out)
    }
}

// A row vector of row-major 3x3 matrix blocks.
// This can also be interpreted as a column vector of column-major 3x3 matrix blocks.
pub type SparseVectorMatrix3<S = Vec<f64>, I = Offsets> =
    Tensor<Sparse<Chunked3<Chunked3<S>>, std::ops::Range<usize>, I>>;
pub type SparseVectorMatrix3View<'a> = SparseVectorMatrix3<&'a [f64], &'a [usize]>;

impl SSMatrix3View<'_> {
    /// Multiply `self` by the given `rhs` vector into the given `out` view.
    /// Note that the output vector `out` may be more sparse than the number of
    /// rows in `self`, however it is assumed that enough elements is allocated
    /// in `out` to ensure that the result fits. Entries are packed towards the
    /// beginning of out, and the number of non-zeros produced is returned so it
    /// can be simply truncated to fit at the end of this function.
    fn mul_sparse_matrix3_vector<S, I>(
        self,
        rhs: SparseVectorMatrix3<S, I>,
        mut out: SparseVectorMatrix3<&mut [f64], &mut [usize]>,
    ) -> usize
    where
        SparseVectorMatrix3<S, I>: for<'a> View<'a, Type = SparseVectorMatrix3View<'a>>,
    {
        let rhs = rhs.view();
        // The output iterator will advance when we see a non-zero result.
        let mut out_iter_mut = out.iter_mut();
        let mut num_non_zeros = 0;

        for (row_idx, row, _) in self.iter() {
            // Initialize output
            let mut sum_mtx = geo::math::Matrix3::zeros();
            let mut row_nnz = 0;

            // Compute the dot product of the two sparse vectors.
            let mut row_iter = row.iter();
            let mut rhs_iter = rhs.iter();

            let mut col_mb = row_iter.next();
            let mut rhs_mb = rhs_iter.next();
            if col_mb.is_some() && rhs_mb.is_some() {
                loop {
                    if col_mb.is_none() || rhs_mb.is_none() {
                        break;
                    }
                    let (col_idx, col_block, _) = col_mb.unwrap();
                    let (rhs_idx, rhs_block, _) = rhs_mb.unwrap();

                    if rhs_idx < col_idx {
                        rhs_mb = rhs_iter.next();
                        continue;
                    } else if rhs_idx > col_idx {
                        col_mb = row_iter.next();
                        continue;
                    } else {
                        // rhs_idx == row_idx
                        // col here is transposed because geo::matrix::Matrix3 is interpreted as col major.
                        sum_mtx += geo::math::Matrix3(*col_block.into_arrays()).transpose()
                            * geo::math::Matrix3(*rhs_block.into_arrays());
                        row_nnz += 1;
                        rhs_mb = rhs_iter.next();
                        col_mb = row_iter.next();
                    }
                }
            }

            if row_nnz > 0 {
                let (index, out_block) = out_iter_mut.next().unwrap();
                *index = row_idx;
                *(out_block.into_arrays()) = sum_mtx.into();
                num_non_zeros += 1;
            }
        }

        num_non_zeros
    }
}

/// A transpose of a matrix.
pub struct Transpose<M>(pub M);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn sparse_sparse_mul_diag() {
        let blocks = vec![
            // Block 1
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            // Block 2
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 1), (3, 2)];
        let mtx = SSMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);

        let sym = mtx.view() * mtx.transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 16.94, 38.72, 60.5, 38.72,
            93.17, 147.62, 60.5, 147.62, 234.74,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    //#[test]
    //fn sparse_diag_add() {
    //    let blocks = vec![
    //        // Block 1
    //        [1.0, 2.0, 3.0],
    //        [4.0, 5.0, 6.0],
    //        [7.0, 8.0, 9.0],
    //        // Block 2
    //        [1.1, 2.2, 3.3],
    //        [4.4, 5.5, 6.6],
    //        [7.7, 8.8, 9.9],
    //    ];
    //    let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
    //    let indices = vec![(1, 1), (3, 2)];
    //    let mtx = SSMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);
    //    let diag = SSMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);

    //    let sym = mtx.view() * mtx.transpose();
    //    sym.write_img("out/sym.png");

    //    let exp_vec = vec![
    //        14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 16.94, 38.72, 60.5, 38.72,
    //        93.17, 147.62, 60.5, 147.62, 234.74,
    //    ];

    //    let val_vec = sym.storage();
    //    for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
    //        assert_relative_eq!(val, exp);
    //    }
    //}

    #[test]
    fn sparse_sparse_mul_non_diag() {
        let blocks = vec![
            // Block 1
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            // Block 2
            [10.0, 11.0, 12.0],
            [16.0, 17.0, 18.0],
            [22.0, 23.0, 24.0],
            // Block 3
            [13.0, 14.0, 15.0],
            [19.0, 20.0, 21.0],
            [25.0, 26.0, 27.0],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 0), (2, 0), (2, 1)];
        let mtx = SSMatrix3::from_triplets(indices.iter().cloned(), 3, 2, chunked_blocks);

        let sym = mtx.view() * mtx.transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 68.0, 104.0, 140.0, 167.0,
            257.0, 347.0, 266.0, 410.0, 554.0, 68.0, 167.0, 266.0, 104.0, 257.0, 410.0, 140.0,
            347.0, 554.0, 955.0, 1405.0, 1855.0, 1405.0, 2071.0, 2737.0, 1855.0, 2737.0, 3619.0,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

}
