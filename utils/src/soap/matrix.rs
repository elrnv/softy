//!
//! Common matrix types and operations.
//!

mod sprs_compat;
pub use sprs_compat::*;

use super::*;
use chunked::Offsets;
use geo::math::{Matrix3, Vector3};
use std::convert::AsRef;
use std::ops::{Add, Mul, MulAssign};

pub trait SparseMatrix {
    fn num_non_zeros(&self) -> usize;
}

pub trait SparseBlockMatrix {
    fn num_non_zero_blocks(&self) -> usize;
}

/// This trait defines information provided by any matrx type.
pub trait Matrix {
    type Transpose;
    fn transpose(self) -> Self::Transpose;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
}

/// A block matrix is a matrix of of smaller matrices organized in blocks. It can also be
/// interpreted as a fourth order tensor.
pub trait BlockMatrix {
    fn num_rows_per_block(&self) -> usize;
    fn num_cols_per_block(&self) -> usize;
    fn num_chunked_rows(&self) -> usize;
    fn num_chunked_cols(&self) -> usize;
}

/*
 * One-dimentional vectors
 */

impl<T, I> Matrix for Tensor<Vec<T>, I> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        1
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

impl<T, I> Matrix for Tensor<&[T], I> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        1
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

impl<T, I> Matrix for Tensor<&mut [T], I> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        1
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

/*
 * Matrices
 */

/// Row-major dense matrix with dynamic number of rows and N columns, where N can be `usize` or a
/// constant.
pub type DMatrix<N = usize, S = Vec<f64>> = Tensor<UniChunked<S, N>>;
pub type DMatrixView<'a, N = usize> = DMatrix<&'a [f64], N>;

impl<N, S> Matrix for DMatrix<N, S>
where
    N: Dimension,
    UniChunked<S, N>: Set,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.data.chunk_size()
    }
    fn num_rows(&self) -> usize {
        self.data.len()
    }
}

/// Row-major dense matrix of row-major NxM blocks where N is the number of rows an M number of
/// columns.
pub type DBlockMatrix<N = usize, M = usize, S = Vec<f64>> =
    DMatrix<UniChunked<UniChunked<S, M>, N>>;
pub type DBlockMatrixView<'a, N = usize, M = usize> = DBlockMatrix<N, M, &'a [f64]>;

/// Row-major dense matrix of row-major 3x3 blocks.
pub type DBlockMatrix3<S = Vec<f64>> = DBlockMatrix<Chunked3<Chunked3<S>>>;
pub type DBlockMatrix3View<'a> = DBlockMatrix3<&'a [f64]>;

/*
 * A diagonal matrix has the same structure as a vector, so it needs a newtype to distinguish it
 * from such.
 */

/// A diagonal matrix of `N` sized chunks. this is not to be confused with block diagonal matrix,
/// which may contain off-diagonal elements in each block. This is a purely diagonal matrix, whose
/// diagonal elements are grouped into `N` sized chunks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagonalBlockMatrix<N = usize, S = Vec<f64>, I = Box<[usize]>>(
    Subset<UniChunked<S, N>, I>,
);
pub type DiagonalBlockMatrixView<'a, N = usize> = DiagonalBlockMatrix<N, &'a [f64], &'a [usize]>;
pub type DiagonalBlockMatrixViewMut<'a, N = usize> =
    DiagonalBlockMatrix<N, &'a mut [f64], &'a [usize]>;

pub type DiagonalBlockMatrix3<S = Vec<f64>, I = Box<[usize]>> = DiagonalBlockMatrix<U3, S, I>;
pub type DiagonalBlockMatrix3View<'a> = DiagonalBlockMatrix3<&'a [f64], &'a [usize]>;

impl<S, N: Dimension> DiagonalBlockMatrix<N, S, Box<[usize]>>
where
    UniChunked<S, N>: Set,
{
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new<T: Into<Subset<UniChunked<S, N>, Box<[usize]>>>>(chunks: T) -> Self {
        DiagonalBlockMatrix(chunks.into())
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> DiagonalBlockMatrix<N, S, I>
where
    UniChunked<S, N>: Set,
{
    /// Explicit constructor from subsets.
    pub fn from_subset(chunks: Subset<UniChunked<S, N>, I>) -> Self {
        DiagonalBlockMatrix(chunks.into())
    }
    /// Produce a mutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view_mut<'a>(&'a mut self) -> Tensor<SubsetView<'a, UniChunked<S::Type, N>>>
    where
        S: Set + ViewMut<'a>,
    {
        Tensor::new(ViewMut::view_mut(&mut self.0))
    }

    /// Produce an immutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view<'a>(&'a self) -> Tensor<SubsetView<'a, UniChunked<S::Type, N>>>
    where
        S: Set + View<'a>,
    {
        Tensor::new(self.0.view())
    }
}

impl<S, N: Dimension> DiagonalBlockMatrix<N, S, Box<[usize]>>
where
    UniChunked<S, N>: Set,
{
    /// Explicit constructor from uniformly chunked collections.
    pub fn from_uniform(chunks: UniChunked<S, N>) -> Self {
        DiagonalBlockMatrix(Subset::all(chunks))
    }
}
impl<S, N> DiagonalBlockMatrix<U<N>, S, Box<[usize]>>
where
    UniChunked<S, U<N>>: Set,
    N: Unsigned + Default,
    S: Set,
{
    pub fn from_flat(chunks: S) -> Self {
        DiagonalBlockMatrix(Subset::all(UniChunked::from_flat(chunks)))
    }
}

impl<S, N: Dimension> BlockMatrix for DiagonalBlockMatrix<N, S>
where
    UniChunked<S, N>: Set,
{
    fn num_cols_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_chunked_cols(&self) -> usize {
        self.0.len()
    }
    fn num_chunked_rows(&self) -> usize {
        self.0.len()
    }
}

impl<S, N: Dimension> Matrix for DiagonalBlockMatrix<N, S>
where
    UniChunked<S, N>: Set,
{
    type Transpose = Self;
    fn transpose(self) -> Self {
        self
    }
    fn num_cols(&self) -> usize {
        self.num_chunked_cols() * self.num_cols_per_block()
    }
    fn num_rows(&self) -> usize {
        self.num_chunked_rows() * self.num_rows_per_block()
    }
}

impl<S, N: Dimension> SparseMatrix for DiagonalBlockMatrix<N, S>
where
    UniChunked<S, N>: Set,
{
    fn num_non_zeros(&self) -> usize {
        self.num_rows()
    }
}

impl<S, N: Dimension> SparseBlockMatrix for DiagonalBlockMatrix<N, S>
where
    UniChunked<S, N>: Set,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.num_chunked_rows()
    }
}

impl<S: Viewed, I, N> Viewed for DiagonalBlockMatrix<N, S, I> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>, N: Copy> View<'a>
    for DiagonalBlockMatrix<N, S, I>
where
    UniChunked<S, N>: Set,
{
    type Type = DiagonalBlockMatrixView<'a, N>;
    fn view(&'a self) -> Self::Type {
        DiagonalBlockMatrix(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>, N: Copy> ViewMut<'a>
    for DiagonalBlockMatrix<N, S, I>
where
    UniChunked<S, N>: Set,
{
    type Type = DiagonalBlockMatrixViewMut<'a, N>;
    fn view_mut(&'a mut self) -> Self::Type {
        DiagonalBlockMatrix(ViewMut::view_mut(&mut self.0))
    }
}

/*
 * Block Diagonal matrices
 */

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockDiagonalMatrix<N = usize, M = usize, S = Vec<f64>, I = Box<[usize]>>(
    Subset<UniChunked<UniChunked<S, M>, N>, I>,
);
pub type BlockDiagonalMatrixView<'a, N = usize, M = usize> =
    BlockDiagonalMatrix<N, M, &'a [f64], &'a [usize]>;
pub type BlockDiagonalMatrixViewMut<'a, N = usize, M = usize> =
    BlockDiagonalMatrix<N, M, &'a mut [f64], &'a [usize]>;

pub type BlockDiagonalMatrix3x2<S = Vec<f64>, I = Box<[usize]>> = BlockDiagonalMatrix<U3, U2, S, I>;
pub type BlockDiagonalMatrix3x2View<'a> = BlockDiagonalMatrix3x2<&'a [f64], &'a [usize]>;

impl<S, N: Dimension, M: Dimension> BlockDiagonalMatrix<N, M, S, Box<[usize]>>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new<T: Into<Subset<UniChunked<UniChunked<S, M>, N>, Box<[usize]>>>>(chunks: T) -> Self {
        BlockDiagonalMatrix(chunks.into())
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension, M: Dimension> BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// Explicit constructor from subsets.
    pub fn from_subset(chunks: Subset<UniChunked<UniChunked<S, M>, N>, I>) -> Self {
        BlockDiagonalMatrix(chunks.into())
    }
}

impl<S, N: Dimension, M: Dimension> BlockDiagonalMatrix<N, M, S, Box<[usize]>>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// Explicit constructor from uniformly chunked collections.
    pub fn from_uniform(chunks: UniChunked<UniChunked<S, M>, N>) -> Self {
        BlockDiagonalMatrix(Subset::all(chunks))
    }
}
impl<S, N, M> BlockDiagonalMatrix<U<N>, U<M>, S, Box<[usize]>>
where
    UniChunked<UniChunked<S, U<M>>, U<N>>: Set,
    UniChunked<S, U<M>>: Set,
    N: Unsigned + Default,
    M: Unsigned + Default,
    S: Set,
{
    pub fn from_flat(chunks: S) -> Self {
        BlockDiagonalMatrix(Subset::all(UniChunked::from_flat(UniChunked::from_flat(
            chunks,
        ))))
    }
}

impl<S, I, N: Dimension, M: Dimension> BlockMatrix for BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_cols_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_chunked_cols(&self) -> usize {
        self.0.len()
    }
    fn num_chunked_rows(&self) -> usize {
        self.0.len()
    }
}

impl<S, I, N: Dimension, M: Dimension> Matrix for BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    type Transpose = Self;
    fn transpose(self) -> Self {
        self
    }
    fn num_cols(&self) -> usize {
        self.num_chunked_cols() * self.num_cols_per_block()
    }
    fn num_rows(&self) -> usize {
        self.num_chunked_rows() * self.num_rows_per_block()
    }
}

impl<S, I, N: Dimension, M: Dimension> SparseMatrix for BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_non_zeros(&self) -> usize {
        self.num_rows()
    }
}

impl<S, I, N: Dimension, M: Dimension> SparseBlockMatrix for BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.num_chunked_rows()
    }
}

impl<S: Viewed, I, N, M> Viewed for BlockDiagonalMatrix<N, M, S, I> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>, N: Copy, M: Copy> View<'a>
    for BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Type = BlockDiagonalMatrixView<'a, N, M>;
    fn view(&'a self) -> Self::Type {
        BlockDiagonalMatrix(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>, N: Copy, M: Copy>
    ViewMut<'a> for BlockDiagonalMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Type = BlockDiagonalMatrixViewMut<'a, N, M>;
    fn view_mut(&'a mut self) -> Self::Type {
        BlockDiagonalMatrix(ViewMut::view_mut(&mut self.0))
    }
}

/// Sparse-row sparse-column 3x3 block matrix.
pub type SSBlockMatrix<N = usize, M = usize, S = Vec<f64>, I = Vec<usize>> = Tensor<
    Sparse<
        Chunked<Sparse<UniChunked<UniChunked<S, M>, N>, std::ops::Range<usize>, I>, Offsets<I>>,
        std::ops::Range<usize>,
        I,
    >,
>;

pub type SSBlockMatrixView<'a, N = usize, M = usize> = SSBlockMatrix<N, M, &'a [f64], &'a [usize]>;

pub type SSBlockMatrix3<S = Vec<f64>, I = Vec<usize>> = SSBlockMatrix<U3, U3, S, I>;

pub type SSBlockMatrix3View<'a> = SSBlockMatrix3<&'a [f64], &'a [usize]>;

impl<S: Set, I, N: Dimension, M: Dimension> BlockMatrix for SSBlockMatrix<N, M, S, I> {
    fn num_cols_per_block(&self) -> usize {
        self.data.source().data().source().data().chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.data.source().data().source().chunk_size()
    }
    fn num_chunked_cols(&self) -> usize {
        self.data.source().data().selection().target.distance()
    }
    fn num_chunked_rows(&self) -> usize {
        self.data.selection().target.distance()
    }
}

impl<S: Set, I, N: Dimension, M: Dimension> Matrix for SSBlockMatrix<N, M, S, I> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.num_chunked_cols() * self.num_cols_per_block()
    }
    fn num_rows(&self) -> usize {
        self.num_chunked_rows() * self.num_rows_per_block()
    }
}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: Set + AsRef<[usize]>> SSBlockMatrix3<S, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_rows();
        let ncols = self.num_cols();
        if nrows == 0 || ncols == 0 {
            return;
        }

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

impl SSBlockMatrix3 {
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
pub type DSBlockMatrix<N = usize, M = usize, S = Vec<f64>, I = Vec<usize>> =
    Tensor<Chunked<Sparse<UniChunked<UniChunked<S, M>, N>, std::ops::Range<usize>, I>, Offsets<I>>>;
pub type DSBlockMatrixView<'a, N = usize, M = usize> = DSBlockMatrix<N, M, &'a [f64], &'a [usize]>;

pub type DSBlockMatrix2<S = Vec<f64>, I = Vec<usize>> = DSBlockMatrix<U2, U2, S, I>;
pub type DSBlockMatrix2View<'a> = DSBlockMatrix2<&'a [f64], &'a [usize]>;

pub type DSBlockMatrix3<S = Vec<f64>, I = Vec<usize>> = DSBlockMatrix<U3, U3, S, I>;
pub type DSBlockMatrix3View<'a> = DSBlockMatrix3<&'a [f64], &'a [usize]>;

impl<S: Set, I: Set, N: Dimension, M: Dimension> BlockMatrix for DSBlockMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    fn num_cols_per_block(&self) -> usize {
        self.data.data().source().data().chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.data.data().source().chunk_size()
    }
    fn num_chunked_cols(&self) -> usize {
        self.data.data().selection().target.distance()
    }
    fn num_chunked_rows(&self) -> usize {
        self.data.len()
    }
}

impl<S: Set, I: Set, N: Dimension, M: Dimension> Matrix for DSBlockMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.num_chunked_cols() * self.num_cols_per_block()
    }
    fn num_rows(&self) -> usize {
        self.num_chunked_rows() * self.num_rows_per_block()
    }
}

impl<S, I, N, M> SparseMatrix for DSBlockMatrix<N, M, S, I>
where
    Self: Storage,
    <Self as Storage>::Storage: Set,
{
    fn num_non_zeros(&self) -> usize {
        self.storage().len()
    }
}

impl<S, I, N, M> SparseBlockMatrix for DSBlockMatrix<N, M, S, I>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.data.data().source().len()
    }
}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: Set + AsRef<[usize]>> DSBlockMatrix3<S, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_rows();
        let ncols = self.num_cols();

        if nrows == 0 || ncols == 0 {
            return;
        }

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

impl From<DiagonalBlockMatrix3> for DSBlockMatrix3 {
    fn from(diag: DiagonalBlockMatrix3) -> DSBlockMatrix3 {
        // Need to convert each triplet in diag into a diagonal 3x3 matrix.
        // Each block is essentially [x, 0, 0, 0, y, 0, 0, 0, z].
        let data: Chunked3<Vec<_>> = diag
            .view()
            .0
            .iter()
            .map(|&[x, y, z]| [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]])
            .collect();

        let num_cols = diag.num_chunked_cols();
        Tensor::new(Chunked::from_sizes(
            vec![1; diag.num_chunked_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(data.into_flat())),
            ),
        ))
    }
}

/*
 * The following is an attempt at generic implementation of the function below
impl<'a, N> DSBlockMatrixView<'a, U<N>, U<N>>
where
    N: Copy + Default + Unsigned + Array<f64> + Array<<N as Array<f64>>::Array> + std::ops::Mul<N>,
    <N as Mul<N>>::Output: Copy + Default + Unsigned + Array<f64>,
{
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    fn diagonal_congruence_transform<M>(self, p: BlockDiagonalMatrixView<U<N>, U<M>>) -> DSBlockMatrix<U<M>, U<M>>
        where
            N: Array<<M as Array<f64>>::Array> + std::ops::Mul<M>,
            M: Copy + Default + Unsigned + Array<f64> + Array<<M as Array<f64>>::Array> + std::ops::Mul<M>,
            <M as Mul<M>>::Output: Copy + Default + Unsigned + Array<f64>,
            StaticRange<<N as std::ops::Mul<M>>::Output>: for<'b> IsolateIndex<&'b [f64]>,
    {
        let mut out = Chunked::from_offsets(
            (*self.data.offsets()).into_owned().into_inner(),
            Sparse::new(
                self.data.data().selection().clone().into_owned(),
                UniChunked::from_flat(UniChunked::from_flat(vec![0.0; self.num_non_zero_blocks() * 4]))
            ),
        );

        for (out_row, row) in Iterator::zip(out.iter_mut(), self.iter()) {
            for ((col_idx, out_entry), orig_entry) in out_row.indexed_source_iter_mut().zip(IntoIterator::into_iter(row.source().view())) {
                *out_entry.as_mut_tensor() = p.0.view().isolate(col_idx).as_tensor().transpose() * orig_entry.as_tensor() * p.0.view().isolate(col_idx).as_tensor();
            }
        }

        Tensor::new(out)
    }
}

impl<'a> DSBlockMatrix3View<'a> {
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    fn diagonal_congruence_transform(self, p: BlockDiagonalMatrix3x2View) -> DSBlockMatrix2 {
        let mut out = Chunked::from_offsets(
            (*self.data.offsets()).into_owned().into_inner(),
            Sparse::new(
                self.data.data().selection().clone().into_owned(),
                UniChunked::from_flat(UniChunked::from_flat(vec![0.0; self.num_non_zero_blocks() * 4]))
            ),
        );

        for (out_row, row) in Iterator::zip(out.iter_mut(), self.iter()) {
            for ((col_idx, out_entry), orig_entry) in out_row.indexed_source_iter_mut().zip(IntoIterator::into_iter(row.source().view())) {
                *out_entry.as_mut_tensor() = p.0.view().isolate(col_idx).as_tensor().transpose() * orig_entry.as_tensor() * p.0.view().isolate(col_idx).as_tensor();
            }
        }

        Tensor::new(out)
    }
}


*/
impl<'a> Mul<Tensor<Chunked3<&'a [f64]>>> for DSBlockMatrix3View<'_> {
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, rhs: Tensor<Chunked3<&'a [f64]>>) -> Self::Output {
        //let rhs = rhs.into();
        assert_eq!(rhs.len(), self.num_chunked_cols());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_chunked_rows()]);
        for (row, out_row) in self.iter().zip(res.iter_mut()) {
            for (col_idx, block, _) in row.iter() {
                let out = Vector3(*out_row) + Matrix3(*block.into_arrays()) * Vector3(rhs[col_idx]);
                *out_row = out.into();
            }
        }

        Tensor::new(res)
    }
}

impl Add<DiagonalBlockMatrix3View<'_>> for SSBlockMatrix3View<'_> {
    type Output = DSBlockMatrix3;
    fn add(self, rhs: DiagonalBlockMatrix3View<'_>) -> Self::Output {
        let rhs = rhs.0;
        let num_rows = self.num_chunked_rows();
        let num_cols = self.num_chunked_cols();

        let lhs_nnz = self.data.source.data.source.len();
        let rhs_nnz = rhs.len();
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

        let add_diagonal_entry = |out: Chunked3<&mut [f64; 9]>, entry: &[f64; 3]| {
            let out_mtx = out.into_arrays();
            out_mtx[0][0] += entry[0];
            out_mtx[1][1] += entry[1];
            out_mtx[2][2] += entry[2];
        };

        for (sparse_row_idx, row_l, _) in self.data.iter() {
            let (row_idx, rhs_entry) = loop {
                if let Some((row_idx, entry)) = rhs_iter.next() {
                    if row_idx < sparse_row_idx {
                        let out_row = out.view_mut().isolate(row_idx);
                        let (idx, out_col, _) = out_row.isolate(0);
                        *idx = row_idx; // Diagonal entry col_idx == row_idx
                        add_diagonal_entry(out_col, entry);
                        out.transfer_forward_all_but(row_idx, 1);
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
        }
        out.trim();

        Tensor::new(out)
    }
}

impl Mul<Tensor<Chunked3<&[f64]>>> for DiagonalBlockMatrix3View<'_> {
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, other: Tensor<Chunked3<&[f64]>>) -> Self::Output {
        let mut out = other.data.into_owned();
        for (&b, out) in self.0.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        Tensor::new(out)
    }
}

impl Mul<Tensor<Chunked3<&[f64]>>> for DiagonalBlockMatrix3<&[f64]> {
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, other: Tensor<Chunked3<&[f64]>>) -> Self::Output {
        let mut out = other.data.into_owned();
        for (&b, out) in self.0.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        Tensor::new(out)
    }
}

impl<'a, Rhs> Mul<Rhs> for SSBlockMatrix3View<'_>
where
    Rhs: Into<Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>>,
{
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        assert_eq!(rhs.len(), self.num_chunked_cols());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_chunked_rows()]);
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

impl<'a, Rhs> Mul<Rhs> for Transpose<SSBlockMatrix3View<'_>>
where
    Rhs: Into<Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>>,
{
    type Output = Tensor<Chunked3<Vec<f64>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        assert_eq!(rhs.len(), self.0.num_chunked_rows());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.0.num_chunked_cols()]);
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

impl<S, I> MulAssign<DiagonalBlockMatrix3<S, I>> for SSBlockMatrix3
where
    S: Set + for<'a> View<'a, Type = &'a [f64]>,
    I: AsRef<[usize]>,
{
    fn mul_assign(&mut self, rhs: DiagonalBlockMatrix3<S, I>) {
        let rhs = View::view(&rhs);
        assert_eq!(rhs.0.len(), self.num_chunked_cols());
        for (_, mut row) in self.view_mut().iter_mut() {
            for (col_idx, mut block) in row.iter_mut() {
                let mass_vec = *rhs.0.at(*col_idx);
                for (block_row, &mass) in block.iter_mut().zip(mass_vec.iter()) {
                    *block_row = (geo::math::Vector3(*block_row) * mass).into();
                }
            }
        }
    }
}

impl Mul<Transpose<SSBlockMatrix3View<'_>>> for SSBlockMatrix3View<'_> {
    type Output = SSBlockMatrix3;
    fn mul(self, rhs: Transpose<SSBlockMatrix3View>) -> Self::Output {
        let rhs_t = rhs.0;
        let num_rows = self.num_chunked_rows();
        let num_cols = rhs_t.num_chunked_rows();

        let lhs_nnz = self.data.source.data.source.len();
        let rhs_nnz = rhs_t.data.source.data.source.len();
        let num_non_zero_cols = rhs_t.indices().len();
        let num_non_zero_blocks = (lhs_nnz + rhs_nnz).max(num_non_zero_cols);

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

            // We may run out of memory in out. Check this and allocate space for each additional
            // row as needed.
            if nz_row_idx < out.len() {
                let num_available = out.view().isolate(nz_row_idx).1.len();
                if num_available < num_non_zero_cols {
                    // The next row has less than num_non_zero_cols available space. We should allocate
                    // additional entries.
                    // First append entries to the last chunk.
                    let num_new_elements = num_non_zero_cols - num_available;
                    Chunked::extend_last(
                        &mut out.source_mut(),
                        std::iter::repeat((0, [[0.0; 3]; 3])).take(num_new_elements),
                    );
                    // Next we transfer all elements of the last chunk into the current row.
                    for idx in (nz_row_idx + 1..out.len()).rev() {
                        out.source_mut().transfer_backward(idx, num_new_elements);
                    }
                }
            }
        }

        // There may be fewer non-zero rows than in self. Truncate those
        // and truncate the entries in storage we didn't use.
        out.trim();

        Tensor::new(out)
    }
}

// A row vector of row-major 3x3 matrix blocks.
// This can also be interpreted as a column vector of column-major 3x3 matrix blocks.
pub type SparseBlockVector<N = usize, M = usize, S = Vec<f64>, I = Offsets> =
    Tensor<Sparse<UniChunked<UniChunked<S, M>, N>, std::ops::Range<usize>, I>>;
pub type SparseBlockVectorView<'a, N = usize, M = usize> =
    SparseBlockVector<N, M, &'a [f64], &'a [usize]>;
pub type SparseBlockVector3<S = Vec<f64>, I = Offsets> = SparseBlockVector<U3, U3, S, I>;
pub type SparseBlockVector3View<'a> = SparseBlockVector3<&'a [f64], &'a [usize]>;

impl SSBlockMatrix3View<'_> {
    /// Multiply `self` by the given `rhs` vector into the given `out` view.
    /// Note that the output vector `out` may be more sparse than the number of
    /// rows in `self`, however it is assumed that enough elements is allocated
    /// in `out` to ensure that the result fits. Entries are packed towards the
    /// beginning of out, and the number of non-zeros produced is returned so it
    /// can be simply truncated to fit at the end of this function.
    fn mul_sparse_matrix3_vector<S, I>(
        self,
        rhs: SparseBlockVector3<S, I>,
        mut out: SparseBlockVector3<&mut [f64], &mut [usize]>,
    ) -> usize
    where
        SparseBlockVector3<S, I>: for<'a> View<'a, Type = SparseBlockVector3View<'a>>,
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

impl<M: BlockMatrix> BlockMatrix for Transpose<M> {
    fn num_chunked_cols(&self) -> usize {
        self.0.num_chunked_rows()
    }
    fn num_chunked_rows(&self) -> usize {
        self.0.num_chunked_cols()
    }
    fn num_cols_per_block(&self) -> usize {
        self.0.num_rows_per_block()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.num_cols_per_block()
    }
}

impl<M: Matrix> Matrix for Transpose<M> {
    type Transpose = M;
    fn transpose(self) -> Self::Transpose {
        self.0
    }
    fn num_cols(&self) -> usize {
        self.0.num_rows()
    }
    fn num_rows(&self) -> usize {
        self.0.num_cols()
    }
}

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
        let mtx = SSBlockMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 16.94, 38.72, 60.5, 38.72,
            93.17, 147.62, 60.5, 147.62, 234.74,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    #[test]
    fn sparse_diag_add() {
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
        let mtx = SSBlockMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let diagonal: Vec<_> = (1..=12).map(|i| i as f64).collect();
        let diag = DiagonalBlockMatrix::from_uniform(Chunked3::from_flat(diagonal));

        let non_singular_sym = sym.view() + diag.view();

        let exp_vec = vec![
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 18.0, 32.0, 50.0, 32.0, 82.0, 122.0, 50.0,
            122.0, 200.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 9.0, 26.94, 38.72, 60.5, 38.72,
            104.17, 147.62, 60.5, 147.62, 246.74,
        ];

        let val_vec = non_singular_sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

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
        let mtx = SSBlockMatrix3::from_triplets(indices.iter().cloned(), 3, 2, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

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
