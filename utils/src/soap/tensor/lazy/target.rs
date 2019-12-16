use super::*;
use crate::soap::Set;

type Zeros = std::ops::RangeTo<usize>;

/// A trait to retrieve the target type of a sparse expression.
/// The target is the set of default values of a sparse expression.
/// Most matrix expressions assume a target of zeros.
pub trait Target {
    type Target;
    fn target(&self) -> &Self::Target;
    fn target_size(&self) -> usize;
}

impl<'a, S, T: Set> Target for SparseIterExpr<'a, S, T> {
    type Target = T;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self.target
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.target.len()
    }
}

impl<E: Target + Iterator> Target for SparseExpr<E> {
    type Target = E::Target;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self.expr.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr.target_size()
    }
}

impl<E: Target> Target for IndexedExpr<E> {
    type Target = E::Target;

    #[inline]
    fn target(&self) -> &Self::Target {
        self.expr.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr.target_size()
    }
}
impl<E: Target> Target for Enumerate<E> {
    type Target = E::Target;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self.iter.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.iter.target_size()
    }
}

impl<E: Target, F> Target for CwiseUnExpr<E, F> {
    type Target = E::Target;

    #[inline]
    fn target(&self) -> &Self::Target {
        self.expr.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr.target_size()
    }
}

// TODO: Refactor Tensor multiplication into a repeat
impl<L, R, T, F> Target for CwiseBinExpr<Tensor<L>, R, F>
where
    R: Target<Target = T>,
{
    type Target = T;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self.right.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.right.target_size()
    }
}

impl<L, R, T, F> Target for CwiseBinExpr<L, Tensor<R>, F>
where
    L: Target<Target = T>,
{
    type Target = T;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self.left.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.left.target_size()
    }
}

impl<L, R, T, F> Target for CwiseBinExpr<SparseExpr<L>, R, F>
where
    L: Iterator,
    SparseExpr<L>: Target<Target = Zeros>,
    R: DenseExpr + Target<Target = T>,
{
    type Target = T;

    #[inline]
    fn target(&self) -> &Self::Target {
        debug_assert_eq!(self.left.target_size(), self.right.target_size());
        &self.right.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.right.target_size()
    }
}

impl<L, R, T, F> Target for CwiseBinExpr<L, SparseExpr<R>, F>
where
    L: DenseExpr + Target<Target = T>,
    R: Iterator,
    SparseExpr<R>: Target<Target = Zeros>,
{
    type Target = T;

    #[inline]
    fn target(&self) -> &Self::Target {
        debug_assert_eq!(self.left.target_size(), self.right.target_size());
        &self.left.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.left.target_size()
    }
}

impl<L, R, F> Target for CwiseBinExpr<SparseExpr<L>, SparseExpr<R>, F>
where
    L: Iterator,
    R: Iterator,
    SparseExpr<L>: Target<Target = Zeros>,
    SparseExpr<R>: Target<Target = Zeros>,
{
    type Target = Zeros;

    #[inline]
    fn target(&self) -> &Self::Target {
        debug_assert_eq!(self.left.target_size(), self.right.target_size());
        &self.left.target()
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.left.target_size()
    }
}

impl<'a, T> Target for SliceIterExpr<'a, T> {
    type Target = Self;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr_size()
    }
}

impl<T> Target for VecIterExpr<T> {
    type Target = Self;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr_size()
    }
}

impl<S, N> Target for UniChunkedIterExpr<S, N>
where
    Self: ExprSize,
{
    type Target = Self;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr_size()
    }
}

impl<'a, S> Target for ChunkedIterExpr<'a, S>
where
    Self: ExprSize,
{
    type Target = Self;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr_size()
    }
}

impl<'a, S> Target for SubsetIterExpr<'a, S>
where
    Self: ExprSize,
{
    type Target = Self;

    #[inline]
    fn target(&self) -> &Self::Target {
        &self
    }
    #[inline]
    fn target_size(&self) -> usize {
        self.expr_size()
    }
}
