//!
//! Common matrix types and operations.
//!

mod sprs_compat;
pub use sprs_compat::*;

use super::*;
use crate::zip;
use chunked::Offsets;
use num_traits::{Float, Zero};
use std::convert::AsRef;
use std::ops::{Add, Mul, MulAssign};

type Dim = std::ops::RangeTo<usize>;

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
    fn num_total_rows(&self) -> usize;
    fn num_total_cols(&self) -> usize;
}

/*
 * One-dimentional vectors
 */

impl<T: Scalar> Matrix for Tensor<Vec<T>> {
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

impl<T: Scalar> Matrix for &Tensor<[T]> {
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

impl<T: Scalar> Matrix for &mut Tensor<[T]> {
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
pub type DMatrixBase<T, N = usize> = Tensor<UniChunked<T, N>>;
pub type DMatrix<T = f64, N = usize> = DMatrixBase<Tensor<Vec<T>>, N>;
pub type DMatrixView<'a, T = f64, N = usize> = DMatrixBase<&'a Tensor<[T]>, N>;

impl<N, T> Matrix for DMatrixBase<T, N>
where
    N: Dimension,
    Self: Set,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.chunk_size()
    }
    fn num_rows(&self) -> usize {
        self.len()
    }
}

/// Row-major dense matrix of row-major NxM blocks where N is the number of rows an M number of
/// columns.
pub type DBlockMatrixBase<T, N, M> = DMatrixBase<Tensor<UniChunked<Tensor<UniChunked<T, M>>, N>>>;
pub type DBlockMatrix<T = f64, N = usize, M = usize> = DBlockMatrixBase<Tensor<Vec<T>>, N, M>;
pub type DBlockMatrixView<'a, T = f64, N = usize, M = usize> =
    DBlockMatrixBase<&'a Tensor<[T]>, N, M>;

/// Row-major dense matrix of row-major 3x3 blocks.
pub type DBlockMatrix3<T = f64> = DBlockMatrix<T, U3, U3>;
pub type DBlockMatrix3View<'a, T = f64> = DBlockMatrixView<'a, T, U3, U3>;

/// Dense-row sparse-column row-major matrix. AKA CSR matrix.
pub type DSMatrixBase<T, I> = Tensor<Chunked<Tensor<Sparse<T, Dim, I>>, Offsets<I>>>;
pub type DSMatrix<T = f64, I = Vec<usize>> = DSMatrixBase<Tensor<Vec<T>>, I>;
pub type DSMatrixView<'a, T = f64> = DSMatrixBase<&'a Tensor<[T]>, &'a [usize]>;

impl<S: IntoData, I: Set> Matrix for DSMatrixBase<S, I>
where
    S::Data: Set,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.as_data().data().selection().target.distance()
    }
    fn num_rows(&self) -> usize {
        self.as_data().len()
    }
}

impl<S, I> SparseMatrix for DSMatrixBase<S, I>
where
    S: Storage,
    S::Storage: Set,
{
    fn num_non_zeros(&self) -> usize {
        self.storage().len()
    }
}

impl DSMatrix {
    /// Construct a sparse matrix from a given iterator of triplets.
    pub fn from_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, f64)>,
    {
        Self::from_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }

    /// Construct a possibly uncompressed sparse matrix from a given iterator of triplets.
    /// This is useful if the caller needs to prune the matrix anyways, which will compress it in
    /// the process, thus saving an extra pass through the values.
    pub fn from_triplets_iter_uncompressed<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, f64)>,
    {
        let mut triplets: Vec<_> = iter.collect();
        triplets.sort_by_key(|&(row, _, _)| row);
        Self::from_sorted_triplets_iter_uncompressed(triplets.into_iter(), num_rows, num_cols)
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_sorted_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, f64)>,
    {
        Self::from_sorted_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_sorted_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, f64)>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut vals: Vec<f64> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, val) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            vals.push(val);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(offsets, Sparse::from_dim(cols, num_cols, vals));

        col_data.sort_chunks_by_index();

        col_data.into_tensor()
    }
}

impl<S, I> DSMatrixBase<S, I>
where
    Self: for<'a> View<'a, Type = DSMatrixView<'a>>,
{
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSMatrix {
        self.view()
            .into_data()
            .compressed(|a, &b| *a += b)
            .into_tensor()
    }
}

impl<S, I> DSMatrixBase<S, I>
where
    Self: for<'a> View<'a, Type = DSMatrixView<'a>>,
{
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(&self, keep: impl Fn(usize, usize, &f64) -> bool) -> DSMatrix {
        self.view()
            .into_data()
            .pruned(|a, &b| *a += b, keep)
            .into_tensor()
    }
}

/*
 * A diagonal matrix has the same structure as a vector, so it needs a newtype to distinguish it
 * from such.
 */

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagonalMatrixBase<T, I = Box<[usize]>>(Subset<T, I>);
pub type DiagonalMatrix<T = f64, I = Box<[usize]>> = DiagonalMatrixBase<Vec<T>, I>;
pub type DiagonalMatrixView<'a, T = f64> = DiagonalMatrixBase<&'a [T], &'a [usize]>;
pub type DiagonalMatrixViewMut<'a, T = f64> = DiagonalMatrixBase<&'a mut [T], &'a [usize]>;

impl<S: Set> DiagonalMatrixBase<S, Box<[usize]>> {
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new<T: Into<Subset<S, Box<[usize]>>>>(set: T) -> Self {
        DiagonalMatrixBase(set.into())
    }
}

impl<S: Set, I: AsRef<[usize]>> DiagonalMatrixBase<S, I> {
    /// Explicit constructor from subsets.
    pub fn from_subset(subset: Subset<S, I>) -> Self {
        DiagonalMatrixBase(subset.into())
    }
    /// Produce a mutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view_mut<'a, T>(&'a mut self) -> Tensor<SubsetView<'a, T>>
    where
        S: ViewMut<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        self.0.view_mut().into_tensor()
    }

    /// Produce an immutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view<'a, T>(&'a self) -> Tensor<SubsetView<'a, T>>
    where
        S: View<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        self.0.view().into_tensor()
    }
}

impl<S: Set> Matrix for DiagonalMatrixBase<S> {
    type Transpose = Self;
    fn transpose(self) -> Self {
        self
    }
    fn num_cols(&self) -> usize {
        self.0.len()
    }
    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

impl<T, S, I> Norm<T> for DiagonalMatrixBase<S, I>
where
    T: Scalar,
    Subset<S, I>: for<'a> ViewIterator<'a, Item = &'a T>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => {
                self.0.view_iter().map(|x| x.abs().powi(p)).sum::<T>().powf(
                    T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type."),
                )
            }
            LpNorm::Inf => self
                .0
                .view_iter()
                .map(|x| x.abs())
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.0.view_iter().map(|&x| x * x).sum::<T>()
    }
}

impl<S: Set> SparseMatrix for DiagonalMatrixBase<S> {
    fn num_non_zeros(&self) -> usize {
        self.num_rows()
    }
}

impl<S: Viewed, I> Viewed for DiagonalMatrixBase<S, I> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>> View<'a>
    for DiagonalMatrixBase<S, I>
{
    type Type = DiagonalMatrixView<'a>;
    fn view(&'a self) -> Self::Type {
        DiagonalMatrixBase(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>> ViewMut<'a>
    for DiagonalMatrixBase<S, I>
{
    type Type = DiagonalMatrixViewMut<'a>;
    fn view_mut(&'a mut self) -> Self::Type {
        DiagonalMatrixBase(ViewMut::view_mut(&mut self.0))
    }
}

/// A diagonal matrix of `N` sized chunks. this is not to be confused with block diagonal matrix,
/// which may contain off-diagonal elements in each block. This is a purely diagonal matrix, whose
/// diagonal elements are grouped into `N` sized chunks.
//
// TODO: Unify specialized matrix types like DiagonalBlockMatrixBase to have a similar api to
// Tensors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DiagonalBlockMatrixBase<S, I = Box<[usize]>, N = usize>(pub Subset<UniChunked<S, N>, I>);
pub type DiagonalBlockMatrix<T = f64, I = Box<[usize]>, N = usize> =
    DiagonalBlockMatrixBase<Vec<T>, I, N>;
pub type DiagonalBlockMatrixView<'a, T = f64, N = usize> =
    DiagonalBlockMatrixBase<&'a [T], &'a [usize], N>;
pub type DiagonalBlockMatrixViewMut<'a, T = f64, N = usize> =
    DiagonalBlockMatrixBase<&'a mut [T], &'a [usize], N>;

pub type DiagonalBlockMatrix3<T = f64, I = Box<[usize]>> = DiagonalBlockMatrix<T, I, U3>;
pub type DiagonalBlockMatrix3View<'a, T = f64> = DiagonalBlockMatrixView<'a, T, U3>;

impl<S, N: Dimension> DiagonalBlockMatrixBase<S, Box<[usize]>, N>
where
    UniChunked<S, N>: Set,
{
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new<T: Into<Subset<UniChunked<S, N>>>>(chunks: T) -> Self {
        DiagonalBlockMatrixBase(chunks.into())
    }
}

impl<'a, S, N: Dimension> DiagonalBlockMatrixBase<S, &'a [usize], N>
where
    UniChunked<S, N>: Set,
{
    pub fn view<T: Into<SubsetView<'a, UniChunked<S, N>>>>(chunks: T) -> Self {
        DiagonalBlockMatrixBase(chunks.into())
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    /// Explicit constructor from subsets.
    pub fn from_subset(chunks: Subset<UniChunked<S, N>, I>) -> Self {
        DiagonalBlockMatrixBase(chunks.into())
    }
    /// Produce a mutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view_mut<'a, T>(&'a mut self) -> Tensor<SubsetView<'a, Tensor<UniChunked<T, N>>>>
    where
        S: Set + ViewMut<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        ViewMut::view_mut(&mut self.0).into_tensor()
    }

    /// Produce an immutable view of this diagonal block matrix as a tensor. When interprepted as a tensor the data
    /// contained in this matrix represents a dense matrix with `self.0.len()` rows and `N` columns.
    pub fn tensor_view<'a, T>(&'a self) -> Tensor<SubsetView<'a, Tensor<UniChunked<T, N>>>>
    where
        S: Set + View<'a>,
        T: IntoData<Data = S::Type>,
        S::Type: IntoTensor<Tensor = T>,
    {
        self.0.view().into_tensor()
    }
}

impl<S, N: Dimension> DiagonalBlockMatrixBase<S, Box<[usize]>, N>
where
    UniChunked<S, N>: Set,
{
    /// Explicit constructor from uniformly chunked collections.
    pub fn from_uniform(chunks: UniChunked<S, N>) -> Self {
        DiagonalBlockMatrixBase(Subset::all(chunks))
    }
}
impl<S, N> DiagonalBlockMatrixBase<S, Box<[usize]>, U<N>>
where
    UniChunked<S, U<N>>: Set,
    N: Unsigned + Default,
    S: Set,
{
    pub fn from_flat(chunks: S) -> Self {
        DiagonalBlockMatrixBase(Subset::all(UniChunked::from_flat(chunks)))
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> BlockMatrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    fn num_cols_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.0.chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> Matrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    type Transpose = Self;
    fn transpose(self) -> Self {
        self
    }
    fn num_cols(&self) -> usize {
        self.0.len()
    }
    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

impl<T, I> Norm<T> for DiagonalBlockMatrix3<T, I>
where
    T: Scalar,
    Subset<Chunked3<Vec<T>>, I>: for<'a> ViewIterator<'a, Item = &'a [T; 3]>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => self
                .0
                .view_iter()
                .map(|v| v.as_tensor().map(|x| x.abs().powi(p)).sum())
                .sum::<T>()
                .powf(T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type.")),
            LpNorm::Inf => self
                .0
                .view_iter()
                .flat_map(|v| v.iter().map(|x| x.abs()))
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.0
            .view_iter()
            .map(|&x| x.as_tensor().norm_squared())
            .sum::<T>()
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> SparseMatrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    fn num_non_zeros(&self) -> usize {
        self.num_total_rows()
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension> SparseBlockMatrix for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.num_rows()
    }
}

impl<S: Viewed, I, N> Viewed for DiagonalBlockMatrixBase<S, I, N> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>, N: Copy> View<'a>
    for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    type Type = DiagonalBlockMatrixView<'a, f64, N>;
    fn view(&'a self) -> Self::Type {
        DiagonalBlockMatrixBase(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>, N: Copy> ViewMut<'a>
    for DiagonalBlockMatrixBase<S, I, N>
where
    UniChunked<S, N>: Set,
{
    type Type = DiagonalBlockMatrixViewMut<'a, f64, N>;
    fn view_mut(&'a mut self) -> Self::Type {
        DiagonalBlockMatrixBase(ViewMut::view_mut(&mut self.0))
    }
}

/*
 * Block Diagonal matrices
 */

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockDiagonalMatrixBase<S, I = Box<[usize]>, N = usize, M = usize>(
    pub Subset<UniChunked<UniChunked<S, M>, N>, I>,
);
pub type BlockDiagonalMatrix<T = f64, I = Box<[usize]>, N = usize, M = usize> =
    BlockDiagonalMatrixBase<Vec<T>, I, N, M>;
pub type BlockDiagonalMatrixView<'a, T = f64, N = usize, M = usize> =
    BlockDiagonalMatrixBase<&'a [T], &'a [usize], N, M>;
pub type BlockDiagonalMatrixViewMut<'a, T = f64, N = usize, M = usize> =
    BlockDiagonalMatrixBase<&'a mut [T], &'a [usize], N, M>;

pub type BlockDiagonalMatrix3x2<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U3, U2>;
pub type BlockDiagonalMatrix3x2View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U3, U2>;

pub type BlockDiagonalMatrix3x1<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U3, U1>;
pub type BlockDiagonalMatrix3x1View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U3, U1>;

pub type BlockDiagonalMatrix2<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U2, U2>;
pub type BlockDiagonalMatrix2View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U2, U2>;

pub type BlockDiagonalMatrix3<T = f64, I = Box<[usize]>> = BlockDiagonalMatrix<T, I, U3, U3>;
pub type BlockDiagonalMatrix3View<'a, T = f64> = BlockDiagonalMatrixView<'a, T, U3, U3>;

impl<S, N: Dimension, M: Dimension> BlockDiagonalMatrixBase<S, Box<[usize]>, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// A generic constructor that transforms the input into the underlying storage type. This
    /// sometimes requires additional generic parameters to be explicitly specified.
    /// This function assumes `Box<[usize]>` as a placeholder for indices where the subset is
    /// entire.
    pub fn new<T: Into<Subset<UniChunked<UniChunked<S, M>, N>, Box<[usize]>>>>(chunks: T) -> Self {
        BlockDiagonalMatrixBase(chunks.into())
    }
}

impl BlockDiagonalMatrix3x1 {
    pub fn negate(&mut self) {
        for mut x in self.0.iter_mut() {
            for x in x.iter_mut() {
                for x in x.iter_mut() {
                    *x = -*x;
                }
            }
        }
    }
}

impl<S, I: AsRef<[usize]>, N: Dimension, M: Dimension> BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// Explicit constructor from subsets.
    pub fn from_subset(chunks: Subset<UniChunked<UniChunked<S, M>, N>, I>) -> Self {
        BlockDiagonalMatrixBase(chunks.into())
    }
}

impl<S, N: Dimension, M: Dimension> BlockDiagonalMatrixBase<S, Box<[usize]>, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    /// Explicit constructor from uniformly chunked collections.
    pub fn from_uniform(chunks: UniChunked<UniChunked<S, M>, N>) -> Self {
        BlockDiagonalMatrixBase(Subset::all(chunks))
    }
}
impl<S, N, M> BlockDiagonalMatrixBase<S, Box<[usize]>, U<N>, U<M>>
where
    UniChunked<UniChunked<S, U<M>>, U<N>>: Set,
    UniChunked<S, U<M>>: Set,
    N: Unsigned + Default,
    M: Unsigned + Default,
    S: Set,
{
    pub fn from_flat(chunks: S) -> Self {
        BlockDiagonalMatrixBase(Subset::all(UniChunked::from_flat(UniChunked::from_flat(
            chunks,
        ))))
    }
}

impl<S, I, N: Dimension, M: Dimension> BlockMatrix for BlockDiagonalMatrixBase<S, I, N, M>
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
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S, I, N: Dimension, M: Dimension> Matrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.0.len()
    }
    fn num_rows(&self) -> usize {
        self.0.len()
    }
}

impl<T, I> Norm<T> for BlockDiagonalMatrix3<T, I>
where
    T: Scalar,
    Subset<Chunked3<Chunked3<Vec<T>>>, I>: for<'a> ViewIterator<'a, Item = &'a [[T; 3]; 3]>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => self
                .0
                .view_iter()
                .map(|v| v.as_tensor().map_inner(|x| x.abs().powi(p)).sum_inner())
                .sum::<T>()
                .powf(T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type.")),
            LpNorm::Inf => self
                .0
                .view_iter()
                .flat_map(|v| v.iter().flat_map(|v| v.iter()))
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .cloned()
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.0
            .view_iter()
            .map(|&x| x.as_tensor().frob_norm_squared())
            .sum::<T>()
    }
}

impl<S, I, N: Dimension, M: Dimension> SparseMatrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_non_zeros(&self) -> usize {
        self.num_total_rows()
    }
}

impl<S, I, N: Dimension, M: Dimension> SparseBlockMatrix for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
    I: AsRef<[usize]>,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.num_rows()
    }
}

impl<S: Viewed, I, N, M> Viewed for BlockDiagonalMatrixBase<S, I, N, M> {}

impl<'a, S: Set + View<'a, Type = &'a [f64]>, I: AsRef<[usize]>, N: Copy, M: Copy> View<'a>
    for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Type = BlockDiagonalMatrixView<'a, f64, N, M>;
    fn view(&'a self) -> Self::Type {
        BlockDiagonalMatrixBase(View::view(&self.0))
    }
}

impl<'a, S: Set + ViewMut<'a, Type = &'a mut [f64]>, I: AsRef<[usize]>, N: Copy, M: Copy>
    ViewMut<'a> for BlockDiagonalMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S, M>, N>: Set,
    UniChunked<S, M>: Set,
{
    type Type = BlockDiagonalMatrixViewMut<'a, f64, N, M>;
    fn view_mut(&'a mut self) -> Self::Type {
        BlockDiagonalMatrixBase(ViewMut::view_mut(&mut self.0))
    }
}

// TODO: make this generic over the number
impl<I: AsRef<[usize]>, J: AsRef<[usize]>> Mul<BlockDiagonalMatrix3x2<f64, J>>
    for Transpose<BlockDiagonalMatrix3x2<f64, I>>
{
    type Output = BlockDiagonalMatrix2;
    fn mul(self, other: BlockDiagonalMatrix3x2<f64, J>) -> Self::Output {
        let ref self_data = (self.0).0;
        let ref other_data = other.0;
        assert_eq!(Set::len(self_data), other_data.len());
        assert_eq!(2, self_data.inner_chunk_size());
        let mut out = BlockDiagonalMatrix::from_flat(vec![0.0; 4 * self_data.len()]);
        for (out_block, lhs_block, rhs_block) in
            zip!(out.0.iter_mut(), self_data.iter(), other_data.iter())
        {
            let out_mtx: &mut Matrix2<f64> = out_block.as_matrix();
            *out_mtx = lhs_block.as_matrix().transpose() * *rhs_block.as_matrix();
        }
        out
    }
}

impl<I: AsRef<[usize]>, J: AsRef<[usize]>> Mul<BlockDiagonalMatrix3x1<f64, J>>
    for Transpose<BlockDiagonalMatrix3x1<f64, I>>
{
    type Output = DiagonalMatrix;
    fn mul(self, other: BlockDiagonalMatrix3x1<f64, J>) -> Self::Output {
        let ref self_data = (self.0).0;
        let ref other_data = other.0;
        assert_eq!(Set::len(self_data), other_data.len());
        assert_eq!(self_data.inner_chunk_size(), 1);
        let mut out = DiagonalMatrixBase::new(vec![0.0; self_data.len()]);
        for (out_entry, lhs_block, rhs_block) in
            zip!(out.0.iter_mut(), self_data.iter(), other_data.iter())
        {
            *out_entry = (lhs_block.as_matrix().transpose() * *rhs_block.as_matrix()).data[0][0];
        }
        out
    }
}

/// Sparse-row sparse-column 3x3 block matrix.
pub type SSBlockMatrixBase<S = Vec<f64>, I = Vec<usize>, N = usize, M = usize> = Tensor<
    Sparse<
        Tensor<
            Chunked<
                Tensor<Sparse<Tensor<UniChunked<Tensor<UniChunked<S, M>>, N>>, Dim, I>>,
                Offsets<I>,
            >,
        >,
        Dim,
        I,
    >,
>;
pub type SSBlockMatrix<T = f64, I = Vec<usize>, N = usize, M = usize> =
    SSBlockMatrixBase<Tensor<Vec<T>>, I, N, M>;
pub type SSBlockMatrixView<'a, T = f64, N = usize, M = usize> =
    SSBlockMatrixBase<&'a Tensor<[T]>, &'a [usize], N, M>;
pub type SSBlockMatrix3<T = f64, I = Vec<usize>> = SSBlockMatrix<T, I, U3, U3>;
pub type SSBlockMatrix3View<'a, T = f64> = SSBlockMatrixView<'a, T, U3, U3>;

impl<S: Set + IntoData, I, N: Dimension, M: Dimension> BlockMatrix
    for SSBlockMatrixBase<S, I, N, M>
{
    fn num_cols_per_block(&self) -> usize {
        self.as_data().source().data().source().data().chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.as_data().source().data().source().chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S: Set + IntoData, I, N: Dimension, M: Dimension> Matrix for SSBlockMatrixBase<S, I, N, M> {
    type Transpose = Transpose<Self>;
    fn transpose(self) -> Self::Transpose {
        Transpose(self)
    }
    fn num_cols(&self) -> usize {
        self.as_data().source().data().selection().target.distance()
    }
    fn num_rows(&self) -> usize {
        self.as_data().selection().target.distance()
    }
}

impl<'a, I: Set + AsRef<[usize]>> SSBlockMatrix3<f64, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = self.num_total_rows();
        let ncols = self.num_total_cols();
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
        let view = self.view().into_data();
        if let Ok(row) = view
            .selection
            .indices
            .binary_search(&(r / 3))
            .map(|idx| view.source.isolate(idx))
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
    pub fn from_index_iter_and_data<It: Iterator<Item = (usize, usize)>>(
        index_iter: It,
        num_rows: usize,
        num_cols: usize,
        blocks: Chunked3<Chunked3<Vec<f64>>>,
    ) -> Self {
        Self::from_block_triplets_iter(
            index_iter
                .zip(blocks.iter())
                .map(|((i, j), x)| (i, j, *x.into_arrays())),
            num_rows,
            num_cols,
        )
    }

    /// Assume that rows are monotonically increasing in the iterator.
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [[f64; 3]; 3])>,
    {
        let cap = iter.size_hint().0;
        let mut rows = Vec::with_capacity(cap);
        let mut cols = Vec::with_capacity(cap);
        let mut blocks: Vec<[f64; 3]> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                rows.push(row);
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[f64; 3]>
            blocks.push(block[0]);
            blocks.push(block[1]);
            blocks.push(block[2]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();
        rows.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(blocks)),
            ),
        );

        col_data.sort_chunks_by_index();

        Sparse::from_dim(rows, num_rows, col_data)
            .into_tensor()
            .compressed()
    }
}

fn is_unique(indices: &[usize]) -> bool {
    if indices.is_empty() {
        return true;
    }
    let mut prev_index = indices[0];
    for &index in indices[1..].iter() {
        if index == prev_index {
            return false;
        }
        prev_index = index;
    }
    true
}

impl<I: AsRef<[usize]>> SSBlockMatrix3<f64, I>
where
    Self: for<'a> View<'a, Type = SSBlockMatrix3View<'a>>,
    I: IntoOwned<Owned = Vec<usize>>,
{
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> SSBlockMatrix3 {
        let data = self.as_data();
        // Check that there are no duplicate rows. This should not happen when crating from
        // triplets.
        assert!(is_unique(data.selection.indices.as_ref()));
        Sparse::new(
            data.selection.view().into_owned(),
            data.view().source.compressed(|a, b| {
                *AsMutTensor::as_mut_tensor(a.as_mut_arrays()) += b.into_arrays().as_tensor()
            }),
        )
        .into_tensor()
    }
}

impl<I: AsRef<[usize]>> SSBlockMatrix3<f64, I>
where
    Self: for<'a> View<'a, Type = SSBlockMatrix3View<'a>>,
    I: IntoOwned<Owned = Vec<usize>>,
{
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(&self, keep: impl Fn(usize, usize, &Matrix3<f64>) -> bool) -> SSBlockMatrix3 {
        let data = self.as_data();
        // Check that there are no duplicate rows. This should not happen when crating from
        // triplets.
        assert!(is_unique(data.selection.indices.as_ref()));
        Sparse::new(
            data.selection.view().into_owned(),
            data.view().source.pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
            ),
        )
        .into_tensor()
    }
}

/// Dense-row sparse-column row-major 3x3 block matrix. Block version of CSR.
pub type DSBlockMatrixBase<S, I = Vec<usize>, N = usize, M = usize> = Tensor<
    Chunked<Tensor<Sparse<Tensor<UniChunked<Tensor<UniChunked<S, M>>, N>>, Dim, I>>, Offsets<I>>,
>;
pub type DSBlockMatrix<T = f64, I = Vec<usize>, N = usize, M = usize> =
    DSBlockMatrixBase<Tensor<Vec<T>>, I, N, M>;
pub type DSBlockMatrixView<'a, T = f64, N = usize, M = usize> =
    DSBlockMatrixBase<&'a Tensor<[T]>, &'a [usize], N, M>;

pub type DSBlockMatrix2<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U2, U2>;
pub type DSBlockMatrix2View<'a, T = f64> = DSBlockMatrixView<'a, T, U2, U2>;

pub type DSBlockMatrix3<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U3, U3>;
pub type DSBlockMatrix3View<'a, T = f64> = DSBlockMatrixView<'a, T, U3, U3>;

pub type DSBlockMatrix1x3<T = f64, I = Vec<usize>> = DSBlockMatrix<T, I, U1, U3>;
pub type DSBlockMatrix1x3View<'a, T = f64> = DSBlockMatrixView<'a, T, U1, U3>;

impl<S: Set + IntoData, I: Set, N: Dimension, M: Dimension> BlockMatrix
    for DSBlockMatrixBase<S, I, N, M>
where
    Self: Matrix,
{
    fn num_cols_per_block(&self) -> usize {
        self.as_data().data().source().data().chunk_size()
    }
    fn num_rows_per_block(&self) -> usize {
        self.as_data().data().source().chunk_size()
    }
    fn num_total_cols(&self) -> usize {
        self.num_cols() * self.num_cols_per_block()
    }
    fn num_total_rows(&self) -> usize {
        self.num_rows() * self.num_rows_per_block()
    }
}

impl<S: Set + IntoData, I, N, M> SparseBlockMatrix for DSBlockMatrixBase<S, I, N, M>
where
    UniChunked<UniChunked<S::Data, M>, N>: Set,
{
    fn num_non_zero_blocks(&self) -> usize {
        self.as_data().data().source().len()
    }
}

impl DSBlockMatrix3 {
    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [[f64; 3]; 3])>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut blocks: Vec<[f64; 3]> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[f64; 3]>
            blocks.push(block[0]);
            blocks.push(block[1]);
            blocks.push(block[2]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(blocks)),
            ),
        );

        col_data.sort_chunks_by_index();

        col_data.into_tensor().compressed()
    }
}

impl<T: Scalar, I: AsRef<[usize]>> DSBlockMatrix3<T, I> {
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSBlockMatrix3<T> {
        self.as_data()
            .view()
            .compressed(|a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor())
            .into_tensor()
    }
}

impl<T: Scalar, I: AsRef<[usize]>> DSBlockMatrix3<T, I> {
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(&self, keep: impl Fn(usize, usize, &Matrix3<T>) -> bool) -> DSBlockMatrix3<T> {
        self.as_data()
            .view()
            .pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
            )
            .into_tensor()
    }
}

impl DSBlockMatrix1x3 {
    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter_uncompressed<I>(
        iter: I,
        num_rows: usize,
        num_cols: usize,
    ) -> Self
    where
        I: Iterator<Item = (usize, usize, [[f64; 3]; 1])>,
    {
        let cap = iter.size_hint().0;
        let mut cols = Vec::with_capacity(cap);
        let mut blocks: Vec<[f64; 3]> = Vec::with_capacity(cap);
        let mut offsets = Vec::with_capacity(num_rows);

        let mut prev_row = 0; // offset by +1 so we don't have to convert between isize.
        for (row, col, block) in iter {
            assert!(row + 1 >= prev_row); // We assume that rows are monotonically increasing.

            if row + 1 != prev_row {
                prev_row = row + 1;
                offsets.push(cols.len());
            }

            cols.push(col);
            // Push each row at a time to produce a Vec<[f64; 3]>
            blocks.push(block[0]);
        }
        offsets.push(cols.len());
        offsets.shrink_to_fit();

        let mut col_data = Chunked::from_offsets(
            offsets,
            Sparse::from_dim(
                cols,
                num_cols,
                Chunked1::from_flat(Chunked3::from_array_vec(blocks)),
            ),
        );

        col_data.sort_chunks_by_index();

        col_data.into_tensor()
    }

    /// Assume that rows are monotonically increasing in the iterator. Columns don't have an order
    /// restriction.
    pub fn from_block_triplets_iter<I>(iter: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: Iterator<Item = (usize, usize, [[f64; 3]; 1])>,
    {
        Self::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols).compressed()
    }
}

impl<T: Scalar, I: AsRef<[usize]>> DSBlockMatrix1x3<T, I> {
    /// Compress the matrix representation by consolidating duplicate entries.
    pub fn compressed(&self) -> DSBlockMatrix1x3<T> {
        self.as_data()
            .view()
            .compressed(|a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor())
            .into_tensor()
    }
}

impl<T: Scalar, I: AsRef<[usize]>> DSBlockMatrix1x3<T, I> {
    /// Remove all elements that do not satisfy the given predicate and compress the resulting matrix.
    pub fn pruned(
        &self,
        keep: impl Fn(usize, usize, &Matrix1x3<T>) -> bool,
    ) -> DSBlockMatrix1x3<T> {
        self.as_data()
            .view()
            .pruned(
                |a, b| *a.as_mut_arrays().as_mut_tensor() += b.into_arrays().as_tensor(),
                |i, j, e| keep(i, j, e.as_arrays().as_tensor()),
            )
            .into_tensor()
    }
}

impl<'a, T: Scalar, I: Set + AsRef<[usize]>> DSBlockMatrix3<T, I> {
    pub fn write_img<P: AsRef<std::path::Path>>(&'a self, path: P) {
        use image::ImageBuffer;

        let nrows = BlockMatrix::num_total_rows(&self.view());
        let ncols = self.view().num_total_cols();

        if nrows == 0 || ncols == 0 {
            return;
        }

        let ciel = 10.0; //jac.max();
        let floor = -10.0; //jac.min();

        let img = ImageBuffer::from_fn(ncols as u32, nrows as u32, |c, r| {
            let val = self.coeff(r as usize, c as usize).to_f64().unwrap();
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
    pub fn coeff(&'a self, r: usize, c: usize) -> T {
        let row = self.as_data().view().isolate(r / 3);
        row.selection
            .indices
            .binary_search(&(c / 3))
            .map(|idx| row.source.isolate(idx).at(r % 3)[c % 3])
            .unwrap_or(T::zero())
    }
}

impl From<DBlockMatrix3> for DSBlockMatrix3 {
    fn from(dense: DBlockMatrix3) -> DSBlockMatrix3 {
        let num_rows = dense.num_rows();
        let num_cols = dense.num_cols();
        Chunked::from_sizes(
            vec![num_cols; num_rows], // num_cols blocks for every row
            Sparse::from_dim(
                (0..num_cols).cycle().take(num_cols*num_rows).collect(), // No sparsity
                num_cols,
                dense.into_data().data,
            ),
        )
        .into_tensor()
    }
}

impl<I: AsRef<[usize]>> From<DiagonalBlockMatrix3<f64, I>> for DSBlockMatrix3 {
    fn from(diag: DiagonalBlockMatrix3<f64, I>) -> DSBlockMatrix3 {
        // Need to convert each triplet in diag into a diagonal 3x3 matrix.
        // Each block is essentially [x, 0, 0, 0, y, 0, 0, 0, z].
        let data: Chunked3<Vec<_>> = diag
            .view()
            .0
            .iter()
            .map(|&[x, y, z]| [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]])
            .collect();

        let num_cols = diag.num_cols();
        Chunked::from_sizes(
            vec![1; diag.num_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                Chunked3::from_flat(Chunked3::from_array_vec(data.into_flat())),
            ),
        )
        .into_tensor()
    }
}

impl From<DiagonalMatrix> for DSMatrix {
    fn from(diag: DiagonalMatrix) -> DSMatrix {
        let mut out_data = vec![0.0; diag.0.len()];
        Subset::clone_into_other(&diag.0.view(), &mut out_data);

        let num_cols = diag.num_cols();
        Chunked::from_sizes(
            vec![1; diag.num_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                out_data,
            ),
        )
        .into_tensor()
    }
}

impl From<BlockDiagonalMatrix2> for DSBlockMatrix2 {
    fn from(diag: BlockDiagonalMatrix2) -> DSBlockMatrix2 {
        let mut out_data = Chunked2::from_flat(Chunked2::from_flat(vec![0.0; diag.0.len() * 4]));
        Subset::clone_into_other(&diag.0.view(), &mut out_data);

        let num_cols = diag.num_cols();
        Chunked::from_sizes(
            vec![1; diag.num_rows()], // One block in every row.
            Sparse::from_dim(
                (0..num_cols).collect(), // Diagonal sparsity pattern
                num_cols,
                out_data,
            ),
        )
        .into_tensor()
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

        out.into_tensor()
    }
}
*/

impl<'a> DSBlockMatrix3View<'a> {
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    pub fn diagonal_congruence_transform(self, p: BlockDiagonalMatrix3x2View) -> DSBlockMatrix2 {
        let mut out = Chunked::from_offsets(
            (*self.as_data().offsets()).into_owned().into_inner(),
            Sparse::new(
                self.as_data().data().selection().clone().into_owned(),
                Chunked2::from_flat(Chunked2::from_flat(vec![
                    0.0;
                    self.num_non_zero_blocks() * 4
                ])),
            ),
        );

        // TODO: It is annoying to always have to call into_arrays to construct small matrices.
        // We should implement array math on UniChunked<Array> types or find another solution.
        for (row_idx, (mut out_row, row)) in
            Iterator::zip(out.iter_mut(), self.as_data().iter()).enumerate()
        {
            for ((col_idx, out_entry), orig_entry) in out_row
                .indexed_source_iter_mut()
                .zip(IntoIterator::into_iter(row.source().view()))
            {
                let basis_lhs =
                    *p.0.as_data()
                        .view()
                        .isolate(row_idx)
                        .into_arrays()
                        .as_tensor();
                let basis_rhs =
                    *p.0.as_data()
                        .view()
                        .isolate(col_idx)
                        .into_arrays()
                        .as_tensor();
                let basis_lhs_tr = basis_lhs.transpose();
                let orig_block = *orig_entry.into_arrays().as_tensor();
                *out_entry.into_arrays().as_mut_tensor() = basis_lhs_tr * orig_block * basis_rhs;
            }
        }

        out.into_tensor()
    }
}

impl<'a> DSBlockMatrix3View<'a> {
    /// A special congruence tranform of the form `A -> P^T A P` where `P` is a block diagonal matrix.
    /// This is useful for instance as a change of basis transformation.
    pub fn diagonal_congruence_transform3x1(self, p: BlockDiagonalMatrix3x1View) -> DSMatrix {
        let mut out = Chunked::from_offsets(
            (*self.as_data().offsets()).into_owned().into_inner(),
            Sparse::new(
                self.as_data().data().selection().clone().into_owned(),
                vec![0.0; self.num_non_zero_blocks()],
            ),
        );

        // TODO: It is annoying to always have to call into_arrays to construct small matrices.
        // We should implement array math on UniChunked<Array> types or find another solution.
        for (row_idx, (mut out_row, row)) in out.iter_mut().zip(self.as_data().iter()).enumerate() {
            for ((col_idx, out_entry), orig_entry) in out_row
                .indexed_source_iter_mut()
                .zip(IntoIterator::into_iter(row.source().view()))
            {
                let basis_lhs = *p.0.view().isolate(row_idx).into_arrays().as_tensor();
                let basis_rhs = *p.0.view().isolate(col_idx).into_arrays().as_tensor();
                let basis_lhs_tr = basis_lhs.transpose();
                let orig_block = *orig_entry.into_arrays().as_tensor();
                *out_entry = (basis_lhs_tr * orig_block * basis_rhs)[0][0];
            }
        }

        out.into_tensor()
    }
}

impl<'a> Mul<&'a Tensor<[f64]>> for DSMatrixView<'_> {
    type Output = Tensor<Vec<f64>>;
    fn mul(self, rhs: &'a Tensor<[f64]>) -> Self::Output {
        let mut res = vec![0.0; self.num_rows()].into_tensor();
        self.mul_into(rhs, res.view_mut());
        res
    }
}

impl<'a> DSMatrixView<'_> {
    fn mul_into(&self, rhs: &Tensor<[f64]>, out: &mut Tensor<[f64]>) {
        assert_eq!(rhs.len(), self.num_cols());
        assert_eq!(out.len(), self.num_rows());
        let view = self.as_data();

        for (row, out_row) in view.iter().zip(out.data.iter_mut()) {
            for (col_idx, entry, _) in row.iter() {
                *out_row += *entry * rhs.data[col_idx];
            }
        }
    }
}

impl<'a> Mul<Tensor<Chunked3<&'a Tensor<[f64]>>>> for DSBlockMatrix3View<'_> {
    type Output = Tensor<Chunked3<Tensor<Vec<f64>>>>;
    fn mul(self, rhs: Tensor<Chunked3<&'a Tensor<[f64]>>>) -> Self::Output {
        let rhs = rhs.as_data();
        assert_eq!(rhs.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_rows()]);
        for (row, out_row) in self.as_data().iter().zip(res.iter_mut()) {
            for (col_idx, block, _) in row.iter() {
                *out_row.as_mut_tensor() +=
                    *block.into_arrays().as_tensor() * rhs[col_idx].into_tensor();
            }
        }
        res.into_tensor()
    }
}

impl Mul<DSBlockMatrix1x3View<'_>> for DSMatrixView<'_> {
    type Output = DSBlockMatrix1x3;
    fn mul(self, rhs: DSBlockMatrix1x3View<'_>) -> Self::Output {
        let mut blocks = Chunked1::from_flat(Chunked3::from_flat(Vec::new()));
        let mut offsets = vec![0];
        offsets.reserve(self.data.len());
        let mut indices = Vec::new();
        indices.reserve(5 * rhs.as_data().len());
        let mut workspace_blocks = Vec::new();

        for row in self.as_data().iter() {
            let mut out_expr = row.expr().cwise_mul(rhs.as_data().expr());

            // Write out block sums into a dense vector of blocks
            if let Some(next) = out_expr.next() {
                workspace_blocks.resize(next.expr.target_size(), Matrix1x3::new([[0.0; 3]; 1]));
                for elem in next.expr {
                    workspace_blocks[elem.index] += elem.expr;
                }
                for next in out_expr {
                    for elem in next.expr {
                        workspace_blocks[elem.index] += elem.expr;
                    }
                }
            }

            // Condense the dense blocks into a sparse vector
            for (i, block) in workspace_blocks
                .iter_mut()
                .enumerate()
                .filter(|(_, b)| !b.is_zero())
            {
                indices.push(i);
                blocks.eval_extend(std::mem::replace(block, Matrix1x3::new([[0.0; 3]; 1])));
            }

            offsets.push(blocks.len());
        }
        let data = Sparse::from_dim(indices, rhs.num_cols(), blocks);
        Chunked::from_offsets(offsets, data).into_tensor()
    }
}

impl<'a, Rhs> Mul<Rhs> for DSBlockMatrix1x3View<'_>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[f64]>>>>>>,
{
    type Output = Tensor<Vec<f64>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.into_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = vec![0.0; self.num_rows()];
        for (row, out_row) in self.as_data().iter().zip(res.iter_mut()) {
            for (col_idx, block, _) in row.iter() {
                *out_row.as_mut_tensor() +=
                    (*block.into_arrays().as_tensor() * Vector3::new(rhs_data[col_idx])).data[0];
            }
        }
        res.into_tensor()
    }
}

impl Add<DiagonalBlockMatrix3View<'_>> for SSBlockMatrix3View<'_> {
    type Output = DSBlockMatrix3;
    fn add(self, rhs: DiagonalBlockMatrix3View<'_>) -> Self::Output {
        let rhs = rhs.0;
        let num_rows = self.num_rows();
        let num_cols = self.num_cols();

        let lhs_nnz = self.as_data().source.data.source.len();
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

        for (sparse_row_idx, row_l, _) in self.as_data().iter() {
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

        out.into_tensor()
    }
}

impl Mul<Tensor<Chunked3<&Tensor<[f64]>>>> for DiagonalBlockMatrix3View<'_> {
    type Output = Tensor<Chunked3<Tensor<Vec<f64>>>>;
    fn mul(self, other: Tensor<Chunked3<&Tensor<[f64]>>>) -> Self::Output {
        let mut out = other.into_data().into_owned();
        for (&b, out) in self.0.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        out.into_tensor()
    }
}

impl Mul<Tensor<Chunked3<&Tensor<[f64]>>>> for DiagonalBlockMatrix3<f64> {
    type Output = Tensor<Chunked3<Tensor<Vec<f64>>>>;
    fn mul(self, other: Tensor<Chunked3<&Tensor<[f64]>>>) -> Self::Output {
        let mut out = other.into_data().into_owned();
        for (&b, out) in self.0.iter().zip(out.iter_mut()) {
            for j in 0..3 {
                out[j] *= b[j];
            }
        }
        out.into_tensor()
    }
}

impl<'a, Rhs> Mul<Rhs> for SSBlockMatrix3View<'_>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[f64]>>>>>>,
{
    type Output = Tensor<Chunked3<Tensor<Vec<f64>>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_rows()]);
        for (row_idx, row, _) in self.as_data().iter() {
            for (col_idx, block, _) in row.iter() {
                *res[row_idx].as_mut_tensor() +=
                    *block.into_arrays().as_tensor() * rhs_data[col_idx].into_tensor();
            }
        }

        res.into_tensor()
    }
}

impl<'a, Rhs> Mul<Rhs> for Transpose<SSBlockMatrix3View<'_>>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[f64]>>>>>>,
{
    type Output = Tensor<Chunked3<Tensor<Vec<f64>>>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.num_cols());

        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_rows()]);
        for (col_idx, col, _) in self.0.as_data().iter() {
            let rhs = rhs_data[col_idx].into_tensor();
            for (row_idx, block, _) in col.iter() {
                *res[row_idx].as_mut_tensor() +=
                    (rhs.transpose() * *block.into_arrays().as_tensor())[0];
            }
        }

        res.into_tensor()
    }
}

impl<'a, Rhs> Mul<Rhs> for Transpose<BlockDiagonalMatrix3x1View<'a>>
where
    Rhs: Into<Tensor<SubsetView<'a, Tensor<Chunked3<&'a Tensor<[f64]>>>>>>,
{
    type Output = Tensor<Vec<f64>>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        let rhs = rhs.into();
        let rhs_data = rhs.as_data();
        assert_eq!(rhs_data.len(), self.0.num_rows());

        let mut res = vec![0.0; self.0.num_cols()];
        for (idx, block) in (self.0).0.iter().enumerate() {
            res[idx] +=
                ((*block.into_arrays().as_tensor()).transpose() * Vector3::new(rhs_data[idx]))[0];
        }

        res.into_tensor()
    }
}

impl MulAssign<DiagonalBlockMatrix3> for SSBlockMatrix3 {
    fn mul_assign(&mut self, rhs: DiagonalBlockMatrix3) {
        let rhs = View::view(&rhs);
        self.mul_assign(rhs);
    }
}
impl MulAssign<DiagonalBlockMatrix3View<'_>> for SSBlockMatrix3 {
    fn mul_assign(&mut self, rhs: DiagonalBlockMatrix3View<'_>) {
        assert_eq!(rhs.0.len(), self.num_cols());
        for (_, mut row) in self.view_mut().as_mut_data().iter_mut() {
            for (col_idx, mut block) in row.iter_mut() {
                let mass_vec = *rhs.0.at(*col_idx);
                for (block_row, &mass) in block.iter_mut().zip(mass_vec.iter()) {
                    *block_row = (Vector3::new(*block_row) * mass).into();
                }
            }
        }
    }
}

impl Mul<Transpose<SSBlockMatrix3View<'_>>> for SSBlockMatrix3View<'_> {
    type Output = SSBlockMatrix3;
    fn mul(self, rhs: Transpose<SSBlockMatrix3View>) -> Self::Output {
        let rhs_t = rhs.0;
        let num_rows = self.num_rows();
        let num_cols = rhs_t.num_rows();

        let lhs_nnz = self.as_data().source.data.source.len();
        let rhs_nnz = rhs_t.as_data().source.data.source.len();
        let num_non_zero_cols = rhs_t.as_data().indices().len();
        let num_non_zero_blocks = (lhs_nnz + rhs_nnz).max(num_non_zero_cols);

        // Allocate enough offsets for all non-zero rows in self. and assign the
        // first row to contain all elements by setting all offsets to
        // num_non_zero_blocks except the first.
        let mut non_zero_row_offsets = vec![num_non_zero_blocks; self.as_data().len() + 1];
        non_zero_row_offsets[0] = 0;

        let mut out = Sparse::from_dim(
            self.as_data().indices().to_vec(),
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
        for (row_idx, row_l, _) in self.as_data().iter() {
            let (_, out_row, _) = out.view_mut().isolate(nz_row_idx);
            let num_non_zero_blocks_in_row =
                rhs_t.mul_sparse_matrix3_vector(row_l.into_tensor(), out_row.into_tensor());

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
                        std::iter::repeat((0, Chunked3::from_flat([0.0; 9])))
                            .take(num_new_elements),
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

        out.into_tensor()
    }
}

// A row vector of row-major 3x3 matrix blocks.
// This can also be interpreted as a column vector of column-major 3x3 matrix blocks.
pub type SparseBlockVectorBase<S, I = Vec<usize>, N = usize, M = usize> =
    Tensor<Sparse<Tensor<UniChunked<Tensor<UniChunked<S, M>>, N>>, Dim, I>>;
pub type SparseBlockVector<T = f64, I = Vec<usize>, N = usize, M = usize> =
    SparseBlockVectorBase<Tensor<Vec<T>>, I, N, M>;
pub type SparseBlockVectorView<'a, T = f64, N = usize, M = usize> =
    SparseBlockVectorBase<&'a Tensor<[T]>, &'a [usize], N, M>;
pub type SparseBlockVectorViewMut<'a, T = f64, N = usize, M = usize> =
    SparseBlockVectorBase<&'a mut Tensor<[T]>, &'a [usize], N, M>;
pub type SparseBlockVector3<T = f64, I = Vec<usize>> = SparseBlockVector<T, I, U3, U3>;
pub type SparseBlockVector3View<'a, T = f64> = SparseBlockVectorView<'a, T, U3, U3>;
pub type SparseBlockVector3ViewMut<'a, T = f64> = SparseBlockVectorViewMut<'a, T, U3, U3>;

impl SSBlockMatrix3View<'_> {
    /// Multiply `self` by the given `rhs` vector into the given `out` view.
    /// Note that the output vector `out` may be more sparse than the number of
    /// rows in `self`, however it is assumed that enough elements is allocated
    /// in `out` to ensure that the result fits. Entries are packed towards the
    /// beginning of out, and the number of non-zeros produced is returned so it
    /// can be simply truncated to fit at the end of this function.
    fn mul_sparse_matrix3_vector(
        self,
        rhs: SparseBlockVector3View<f64>,
        mut out: SparseBlockVectorBase<&mut Tensor<[f64]>, &mut [usize], U3, U3>,
    ) -> usize {
        // The output iterator will advance when we see a non-zero result.
        let mut out_iter_mut = out.as_mut_data().iter_mut();
        let mut num_non_zeros = 0;

        for (row_idx, row, _) in self.as_data().iter() {
            // Initialize output
            let mut sum_mtx = Matrix3::new([[0.0; 3]; 3]);
            let mut row_nnz = 0;

            // Compute the dot product of the two sparse vectors.
            let mut row_iter = row.iter();
            let mut rhs_iter = rhs.as_data().iter();

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
                        sum_mtx += Matrix3::new(*rhs_block.into_arrays())
                            * Matrix3::new(*col_block.into_arrays()).transpose();
                        row_nnz += 1;
                        rhs_mb = rhs_iter.next();
                        col_mb = row_iter.next();
                    }
                }
            }

            if row_nnz > 0 {
                let (index, out_block) = out_iter_mut.next().unwrap();
                *index = row_idx;
                *out_block.into_arrays().as_mut_tensor() = sum_mtx;
                num_non_zeros += 1;
            }
        }

        num_non_zeros
    }
}

/// A transpose of a matrix.
pub struct Transpose<M>(pub M);

impl<M: BlockMatrix> BlockMatrix for Transpose<M> {
    fn num_total_cols(&self) -> usize {
        self.0.num_total_rows()
    }
    fn num_total_rows(&self) -> usize {
        self.0.num_total_cols()
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

impl<T, M> Norm<T> for Transpose<M>
where
    M: Norm<T>,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        self.0.lp_norm(norm)
    }
    fn norm_squared(&self) -> T {
        self.0.norm_squared()
    }
}

impl<'a, M: View<'a>> View<'a> for Transpose<M> {
    type Type = Transpose<M::Type>;
    fn view(&'a self) -> Self::Type {
        Transpose(self.0.view())
    }
}
impl<'a, M: ViewMut<'a>> ViewMut<'a> for Transpose<M> {
    type Type = Transpose<M::Type>;
    fn view_mut(&'a mut self) -> Self::Type {
        Transpose(self.0.view_mut())
    }
}

impl<M: Viewed> Viewed for Transpose<M> {}

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
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 4, 3, chunked_blocks);

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
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 4, 3, chunked_blocks);

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
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 3, 2, chunked_blocks);

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

    #[test]
    fn sparse_sparse_mul_non_diag_uncompressed() {
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
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            // Block 4
            [13.0, 14.0, 15.0],
            [19.0, 20.0, 21.0],
            [25.0, 26.0, 27.0],
        ];
        let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
        let indices = vec![(1, 0), (2, 0), (2, 0), (2, 1)];
        let mtx =
            SSBlockMatrix3::from_index_iter_and_data(indices.iter().cloned(), 3, 2, chunked_blocks);

        let sym = mtx.view() * mtx.view().transpose();

        let exp_vec = vec![
            14.0, 32.0, 50.0, 32.0, 77.0, 122.0, 50.0, 122.0, 194.0, 68.6, 104.6, 140.6, 168.5,
            258.5, 348.5, 268.4, 412.4, 556.4, 68.6, 168.5, 268.4, 104.6, 258.5, 412.4, 140.6,
            348.5, 556.4, 961.63, 1413.43, 1865.23, 1413.43, 2081.23, 2749.03, 1865.23, 2749.03,
            3632.83,
        ];

        let val_vec = sym.storage();
        for (&val, &exp) in val_vec.iter().zip(exp_vec.iter()) {
            assert_relative_eq!(val, exp);
        }
    }

    //#[test]
    //fn ds_mtx_mul_ds_block_mtx_1x3() {
    //    // [1 2 .]
    //    // [. . 3]
    //    // [. 2 .]
    //    let ds = Chunked::from_sizes(vec![2, 1, 1], Sparse::from_dim(vec![0, 1, 2, 1], 3, vec![1.0,2.0,3.0,2.0]));
    //    // [1 2 3 . . .]
    //    // [1 2 1 4 5 6]
    //    // [7 8 9 . . .]
    //    let blocks = Chunked1::from_flat(Chunked3::from_flat(
    //            vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    //    let ds_block_mtx_1x3 = Chunked::from_sizes(vec![1, 2, 1], Sparse::from_dim(vec![0, 0, 1, 0], 2, blocks));

    //    let out = JaggedTensor::new(ds.view()) * JaggedTensor::new(ds_block_mtx_1x3.view());

    //    // [ 3  6  5  8 10 12]
    //    // [21 24 27  .  .  .]
    //    // [ 2  4  2  8 10 12]
    //    let exp_blocks = Chunked1::from_flat(Chunked3::from_flat(
    //            vec![3.0,6.0,5.0,8.0,10.0,12.0,21.0,24.0,27.0,2.0,4.0,2.0,8.0,10.0,12.0]));
    //    assert_eq!(out.data, Chunked::from_sizes(vec![2,1,2], Sparse::from_dim(vec![0,1,0,0,1], 2, exp_blocks)));
    //}
}
