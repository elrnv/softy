/**
 * Lazy tensor arithmetic
 */
use super::*;
use std::ops::Add as AddOp;
use std::ops::AddAssign as AddAssignOp;
use std::ops::Mul as MulOp;
use std::ops::MulAssign as MulAssignOp;
use std::ops::Neg as NegOp;
use std::ops::Sub as SubOp;
use std::ops::SubAssign as SubAssignOp;

mod cwise_bin_expr;
mod enumerate;
mod eval;
mod expr_mut;
mod iterator;
mod sparse_expr;
mod target;
pub use cwise_bin_expr::*;
pub use enumerate::Enumerate;
pub use eval::{EvalExtend, Evaluate};
pub use expr_mut::*;
pub use iterator::*;
pub use sparse_expr::SparseExpr;
pub use target::*;

/// Marker trait for indicating that an expression is unoptimized. This is a workaround for
/// specialization.
pub trait Unoptimized {}
impl<L, R> Unoptimized for CwiseBinExpr<L, R, Subtraction> {}
impl<L, R> Unoptimized for CwiseBinExpr<L, R, Addition> {}
impl<E, F> Unoptimized for CwiseUnExpr<E, F> {}
impl<E: Iterator> Unoptimized for SparseExpr<E> {}
impl<E> Unoptimized for Enumerate<E> {}
impl<E> Unoptimized for Repeat<E> {}
impl<'a, E> Unoptimized for SliceIterExpr<'a, E> {}
impl<'a, E> Unoptimized for SliceIterExprMut<'a, E> {}
impl<T> Unoptimized for VecIterExpr<T> {}
impl<S, N> Unoptimized for UniChunkedIterExpr<S, N> {}
impl<'a, E> Unoptimized for ChunkedIterExpr<'a, E> {}
impl<'a, E, T> Unoptimized for SparseIterExpr<'a, E, T> {}
impl<'a, S> Unoptimized for SubsetIterExpr<'a, S> {}

/// Recursive Sum operator.
///
/// This is similar to `std::iter::Sum` in purpose but it fits the better with the expression
/// and is recursive, which means it will sum all expressions of expressions.
pub trait RecursiveSumOp {
    type Output;
    fn recursive_sum(self) -> Self::Output;
}

/// Non-Recursive Sum operator.
///
/// This is similar to `std::iter::Sum` in purpose but it fits the better with the expression.
pub trait SumOp {
    type Output;
    fn sum_op(self) -> Self::Output;
}

/// Define a standard dot product operator akin to ones found in `std::ops`.
pub trait DotOp<R = Self> {
    type Output;
    fn dot_op(self, rhs: R) -> Self::Output;
}

/// Component-wise multiplication.
///
/// We reserve the standard `Mul` trait for context-sensitive multiplication (e.g. matrix multiply)
/// since these are typically more common than component-wise multiplication in application code.
pub trait CwiseMulOp<R = Self> {
    type Output;
    fn cwise_mul(self, rhs: R) -> Self::Output;
}

/// Component-wise multiplication with assignment.
///
/// We reserve the standard `MulAssign` trait for context-sensitive multiplication (e.g. matrix
/// multiply) since these are typically more common than component-wise multiplication in
/// application code.
pub trait CwiseMulAssignOp<R = Self> {
    fn cwise_mul_assign(&mut self, rhs: R);
}

/*
 * The following structs mark *what* operations need to be done.
 * They are used in conjunction with CwiseUnExpr and CwiseBinExpr, which actually execute the
 * desired behaviour.
 */

// Unary operations
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Summation;
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Negation;

// Binary operations
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Addition;
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Subtraction;
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CwiseMultiplication;
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Multiplication;

macro_rules! impl_default {
    ($type:ty) => {
        impl Default for $type {
            fn default() -> Self {
                Self
            }
        }
    };
}
impl_default!(Addition);
impl_default!(Subtraction);
impl_default!(CwiseMultiplication);
impl_default!(Multiplication);
impl_default!(Summation);
impl_default!(Negation);

///// A marker trait to identify eager expressions like small tensors.
//pub trait Eager {}
//impl<T> Eager for Tensor<T> {}

/// A marker trait to describe unary operations that reduce the underlying expression to a singleton.
pub trait Reduction {}
impl Reduction for Summation {}

/// A marker trait to describe additive binary operations.
///
/// More precisely, given an element `a` and an additive identity element `id`, implementing this
/// trait indicates that `apply(a, id) = a`.
pub trait Additive {}
impl Additive for Addition {}
impl Additive for Subtraction {}

/// A marker trait to describe multiplicative binary operations.
///
/// More precisely, given an element `a` and an identity element `id`, implementing this trait
/// indicates that `apply(a, id) = id`.
pub trait Multiplicative {}
impl Multiplicative for CwiseMultiplication {}
impl Multiplicative for Multiplication {}

pub trait UnOp<T> {
    type Output;
    fn apply(&self, val: T) -> Self::Output;
}

pub trait UnOpAssign<T> {
    fn apply_assign(&self, val: &mut T);
}

impl<T, O, F> UnOp<T> for F
where
    F: Fn(T) -> O,
{
    type Output = O;
    #[inline]
    fn apply(&self, input: T) -> Self::Output {
        self(input)
    }
}

impl<T, F> UnOpAssign<T> for F
where
    F: Fn(&mut T),
{
    #[inline]
    fn apply_assign(&self, inout: &mut T) {
        self(inout)
    }
}

impl<T, O> UnOp<T> for Summation
where
    T: SumOp<Output = O>,
{
    type Output = O;
    #[inline]
    fn apply(&self, input: T) -> Self::Output {
        input.sum_op()
    }
}

impl<T, O> UnOp<T> for Negation
where
    T: NegOp<Output = O>,
{
    type Output = O;
    #[inline]
    fn apply(&self, val: T) -> Self::Output {
        -val
    }
}
impl<T> UnOpAssign<T> for Negation
where
    T: Copy + NegOp<Output = T>,
{
    #[inline]
    fn apply_assign(&self, val: &mut T) {
        *val = -*val;
    }
}

pub trait BinOp<L, R> {
    type Output;
    fn apply(&self, lhs: L, rhs: R) -> Self::Output;
}

pub trait BinOpAssign<L: ?Sized, R> {
    fn apply_assign(&self, lhs: &mut L, rhs: R);
}

impl<L, R, O, F> BinOp<L, R> for F
where
    F: Fn(L, R) -> O,
{
    type Output = O;
    #[inline]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        self(lhs, rhs)
    }
}

impl<L: ?Sized, R, F> BinOpAssign<L, R> for F
where
    F: Fn(&mut L, R),
{
    #[inline]
    fn apply_assign(&self, lhs: &mut L, rhs: R) {
        self(lhs, rhs)
    }
}

impl<L, R, O> BinOp<L, R> for Addition
where
    L: AddOp<R, Output = O>,
{
    type Output = O;
    #[inline]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs + rhs
    }
}

impl<L: ?Sized, R> BinOpAssign<L, R> for Addition
where
    L: AddAssignOp<R>,
{
    #[inline]
    fn apply_assign(&self, lhs: &mut L, rhs: R) {
        *lhs += rhs;
    }
}

impl<L, R, O> BinOp<L, R> for Subtraction
where
    L: SubOp<R, Output = O>,
{
    type Output = O;
    #[inline]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs - rhs
    }
}
impl<L: ?Sized, R> BinOpAssign<L, R> for Subtraction
where
    L: SubAssignOp<R>,
{
    #[inline]
    fn apply_assign(&self, lhs: &mut L, rhs: R) {
        *lhs -= rhs;
    }
}

impl<L, R, O> BinOp<L, R> for CwiseMultiplication
where
    L: CwiseMulOp<R, Output = O>,
{
    type Output = O;
    #[inline]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs.cwise_mul(rhs)
    }
}
impl<L: ?Sized, R> BinOpAssign<L, R> for CwiseMultiplication
where
    L: CwiseMulAssignOp<R>,
{
    #[inline]
    fn apply_assign(&self, lhs: &mut L, rhs: R) {
        lhs.cwise_mul_assign(rhs);
    }
}

impl<L, R, O> BinOp<L, R> for Multiplication
where
    L: MulOp<R, Output = O>,
{
    type Output = O;
    #[inline]
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs.mul(rhs)
    }
}
impl<L: ?Sized, R> BinOpAssign<L, R> for Multiplication
where
    L: MulAssignOp<R>,
{
    #[inline]
    fn apply_assign(&self, lhs: &mut L, rhs: R) {
        lhs.mul_assign(rhs);
    }
}

/// A lazy component-wise unary expression to be evaluated at a later time. This is basically
/// equivalent to `std::iter::Map`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CwiseUnExpr<E, F> {
    expr: E,
    op: F,
}

impl<E, F: Default> CwiseUnExpr<E, F> {
    #[inline]
    pub fn new(expr: E) -> Self {
        Self::with_op(expr, Default::default())
    }
}

impl<E, F> CwiseUnExpr<E, F> {
    #[inline]
    pub fn with_op(expr: E, op: F) -> Self {
        CwiseUnExpr { expr, op }
    }
}

/// A lazy component-wise summation.
type CwiseSum<E> = CwiseUnExpr<E, Summation>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Repeat<E> {
    expr: E,
}

impl<E> Repeat<E> {
    #[inline]
    pub fn new(expr: E) -> Self {
        Repeat { expr }
    }
}

/// A lazy reduce expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reduce<E, F> {
    expr: E,
    op: F,
    // TODO: Add a default value: Some sparse reduce operations are empty but we need to return an
    // empty value.
    // def: ?
}

impl<E, F: Default> Reduce<E, F> {
    #[inline]
    pub fn new(expr: E) -> Self {
        Self::with_op(expr, Default::default())
    }
}
impl<E, F> Reduce<E, F> {
    #[inline]
    pub fn with_op(expr: E, op: F) -> Self {
        Reduce { expr, op }
    }
}

/// A lazy sum expression.
type Sum<E> = Reduce<E, Addition>;

// Convert common containers into nested iterators for lazy processing.

/// A wrapper around a cloned iterator over a slice.
#[derive(Clone, Debug)]
pub struct SliceIterExpr<'a, T>(std::slice::Iter<'a, T>);

#[derive(Clone, Debug)]
pub struct VecIterExpr<T>(std::vec::IntoIter<T>);

#[derive(Clone, Debug, PartialEq)]
pub struct UniChunkedIterExpr<S, N> {
    data: S,
    chunk_size: N,
}

pub type ChunkedNIterExpr<S> = UniChunkedIterExpr<S, usize>;

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkedIterExpr<'a, S> {
    offsets: Offsets<&'a [usize]>,
    data: S,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseIterExpr<'a, S, T> {
    indices: &'a [usize],
    source: S,
    target: T,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubsetIterExpr<'a, S> {
    indices: Option<&'a [usize]>,
    data: S,
}

// Trait that indicates that an iterator produces elements from a dense or
// contiguous collection as opposed to a sparse one.
pub trait DenseExpr {}
impl<'a, T> DenseExpr for SliceIterExpr<'a, T> {}
impl<T> DenseExpr for VecIterExpr<T> {}
impl<S, N> DenseExpr for UniChunkedIterExpr<S, N> {}
impl<'a, S> DenseExpr for ChunkedIterExpr<'a, S> {}
// Subset is treated like a dense expr instead of sparse. This is because the behavior of subsets
// is intended to be agnostic of the underlying superset.
impl<'a, S> DenseExpr for SubsetIterExpr<'a, S> {}
impl<'a, E: DenseExpr, F> DenseExpr for CwiseUnExpr<E, F> {}
impl<'a, A: DenseExpr, B: DenseExpr, F> DenseExpr for CwiseBinExpr<A, B, F> {}
impl<'a, A: DenseExpr, B, F> DenseExpr for CwiseBinExpr<A, Tensor<B>, F> {}
impl<'a, A, B: DenseExpr, F> DenseExpr for CwiseBinExpr<Tensor<A>, B, F> {}
impl<'a, E: DenseExpr> DenseExpr for Repeat<E> {}
impl<'a, E: DenseExpr> DenseExpr for Enumerate<E> {}
impl<'a, T> DenseExpr for Repeat<Tensor<T>> {}

/// A trait describing types that can be evaluated.
pub trait Expression: TotalExprSize {
    #[inline]
    fn eval<T>(self) -> T
    where
        Self: Sized,
        T: Evaluate<Self>,
    {
        Evaluate::eval(self)
    }

    #[inline]
    fn dot<T, R>(self, rhs: R) -> T
    where
        Self: Sized + DotOp<R>,
        T: Evaluate<Self::Output>,
    {
        Evaluate::eval(self.dot_op(rhs))
    }

    #[inline]
    fn reduce<T, F>(self, f: F) -> T
    where
        Self: Sized,
        T: Evaluate<Reduce<Self, F>>,
    {
        Evaluate::eval(Reduce::with_op(self, f))
    }
}

pub trait ExprSize {
    /// The size of this expression. In other words, the number of times this expression
    /// will iterate (similar to `size_hint`).
    /// Infinite expressions will return max int.
    fn expr_size(&self) -> usize;
}

/// Total number of elements that can be generated with this iterator
/// counting items generated by generated iterators.
///
/// A value of `None` indicates that the iterator can be infinite.
/// It used for allocating space so the estimates are expected to be conservative.
pub trait TotalExprSize: ExprSize {
    /// Get the size hint for number of elements returned by this expression. This is useful
    /// during evaluation to allocate sufficient memory.
    ///
    /// # Panics
    ///
    /// This function will panic on an expression that evaluates infinitely.
    #[inline]
    fn reserve_hint(&self) -> usize {
        self.total_size_hint(0).unwrap()
    }

    /// Similar to reserve hint but this function will compute the total size hint for this
    /// expression given all the component wise reductions in the `cwise_reduce` bitmask.
    ///
    /// This function will return `None` on unbounded expressions.
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize>;
}

/// A helper trait that implements `TotalExprSize` for most base expressions.
pub trait BaseExprTotalSizeHint {
    fn base_total_size_hint(&self, cwise_reduce: u32) -> Option<usize>;
}
impl<T> BaseExprTotalSizeHint for T
where
    T: Clone + Iterator,
    T::Item: TotalExprSize,
{
    fn base_total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        // TODO: Is it worth determining the exact size here?
        if cwise_reduce & 1 == 1 {
            self.clone().fold(Some(0), |acc, item| {
                acc.and_then(|acc| {
                    item.total_size_hint(cwise_reduce >> 1).map(|tot| {
                        acc + if item.expr_size() > 0 {
                            debug_assert!(tot > 0);
                            tot / item.expr_size()
                        } else {
                            debug_assert_eq!(tot, 0);
                            0
                        }
                    })
                })
            })
        } else {
            self.clone().fold(Some(0), |acc, item| {
                acc.and_then(|acc| item.total_size_hint(cwise_reduce >> 1).map(|tot| acc + tot))
            })
        }
    }
}

impl<T: Scalar + Expression> Expression for Tensor<T> {}
impl<T: Scalar + ExprSize> ExprSize for Tensor<T> {
    #[inline]
    fn expr_size(&self) -> usize {
        1
    }
}
impl<T: Scalar + TotalExprSize> TotalExprSize for Tensor<T> {
    fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
        Some(1)
    }
}

impl<E: Iterator + Expression, F> Expression for Reduce<E, F> {}
impl<E, F> ExprSize for Reduce<E, F> {
    #[inline]
    fn expr_size(&self) -> usize {
        1
    }
}
impl<E: Iterator + Expression, F> TotalExprSize for Reduce<E, F> {
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        Some(
            self.expr
                .total_size_hint(cwise_reduce)
                .expect("Can't reduce an infinite iterator")
                / self.expr.size_hint().0,
        )
    }
}

impl<'a, T: Clone + IntoExpr> Iterator for SliceIterExpr<'a, T> {
    type Item = T::Expr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.clone().into_expr())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<'a, T> Expression for SliceIterExpr<'a, T> {}
impl<'a, T: Clone + IntoExpr> ExactSizeIterator for SliceIterExpr<'a, T> {}
impl<'a, T> ExprSize for SliceIterExpr<'a, T> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.0.size_hint().1.unwrap_or(self.0.size_hint().0)
    }
}
impl<'a, T> TotalExprSize for SliceIterExpr<'a, T> {
    fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
        Some(self.0.size_hint().1.unwrap_or(self.0.size_hint().0))
    }
}

impl<'a, T: IntoExpr> Iterator for VecIterExpr<T> {
    type Item = T::Expr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.into_expr())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<T: IntoExpr> Expression for VecIterExpr<T> {}
impl<T: IntoExpr> ExactSizeIterator for VecIterExpr<T> {}
impl<T> ExprSize for VecIterExpr<T> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.0.size_hint().1.unwrap_or(self.0.size_hint().0)
    }
}
impl<T> TotalExprSize for VecIterExpr<T> {
    fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
        Some(self.0.size_hint().1.unwrap_or(self.0.size_hint().0))
    }
}

impl<S, N> Iterator for UniChunkedIterExpr<S, U<N>>
where
    S: Set + SplitPrefix<N> + Dummy,
    S::Prefix: IntoExpr,
    N: Unsigned,
{
    type Item = <S::Prefix as IntoExpr>::Expr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        data_slice.split_prefix().map(|(prefix, rest)| {
            self.data = rest;
            prefix.into_expr()
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.data.len() / N::to_usize();
        (n, Some(n))
    }
}

impl<S: Set, N> Expression for UniChunkedIterExpr<S, N> where Self: BaseExprTotalSizeHint {}
impl<S: Set, N> ExprSize for UniChunkedIterExpr<S, N> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.data.len()
    }
}
impl<S: Set, N> TotalExprSize for UniChunkedIterExpr<S, N>
where
    Self: BaseExprTotalSizeHint,
{
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.base_total_size_hint(cwise_reduce)
    }
}

impl<S, N> ExactSizeIterator for UniChunkedIterExpr<S, N> where Self: Iterator {}

impl<'a, S> Iterator for ChunkedNIterExpr<S>
where
    S: Set + SplitAt + Dummy + IntoExpr,
{
    type Item = S::Expr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        let (l, r) = data_slice.split_at(self.chunk_size.value());
        self.data = r;
        Some(l.into_expr())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.data.len() / self.chunk_size;
        (n, Some(n))
    }
}

impl<'a, S> Iterator for ChunkedIterExpr<'a, S>
where
    S: Set + SplitAt + Dummy + IntoExpr,
{
    type Item = S::Expr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        self.offsets.pop_offset().map(move |n| {
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            l.into_expr()
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.offsets.len() - 1;
        (n, Some(n))
    }
}

impl<'a, S: Set> Expression for ChunkedIterExpr<'a, S> where Self: BaseExprTotalSizeHint {}
impl<'a, S: Set> ExprSize for ChunkedIterExpr<'a, S> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.data.len()
    }
}
impl<'a, S: Set> TotalExprSize for ChunkedIterExpr<'a, S>
where
    Self: BaseExprTotalSizeHint,
{
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.base_total_size_hint(cwise_reduce)
    }
}
impl<'a, S> ExactSizeIterator for ChunkedIterExpr<'a, S> where Self: Iterator {}

impl<'a, S, T> Iterator for SparseIterExpr<'a, S, T>
where
    S: SplitFirst + Dummy,
    S::First: IntoExpr,
{
    type Item = IndexedExpr<<S::First as IntoExpr>::Expr>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        source_slice.split_first().map(|(first, rest)| {
            self.source = rest;
            // We know that sparse has at least one element, no need to check again
            let first_idx = unsafe { self.indices.get_unchecked(0) };
            self.indices = &self.indices[1..];
            (*first_idx, first.into_expr()).into()
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.indices.len();
        (n, Some(n))
    }
}

impl<'a, S: Set, T> Expression for SparseIterExpr<'a, S, T> where Self: BaseExprTotalSizeHint {}
impl<'a, S: Set, T> ExactSizeIterator for SparseIterExpr<'a, S, T> where
    Self: Iterator + BaseExprTotalSizeHint
{
}
impl<'a, S: Set, T> ExprSize for SparseIterExpr<'a, S, T> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.source.len()
    }
}
impl<'a, S: Set, T> TotalExprSize for SparseIterExpr<'a, S, T>
where
    Self: BaseExprTotalSizeHint,
{
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.base_total_size_hint(cwise_reduce)
    }
}

impl<'a, S> Iterator for SubsetIterExpr<'a, S>
where
    S: Set + SplitAt + SplitFirst + Dummy,
    S::First: IntoExpr,
{
    type Item = <S::First as IntoExpr>::Expr;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use std::borrow::Borrow;
        let SubsetIterExpr { indices, data } = self;
        let data_slice = std::mem::replace(data, unsafe { Dummy::dummy() });
        match indices {
            Some(ref mut indices) => {
                indices.clone().split_first().map(|(first, rest)| {
                    let (item, right) = data_slice.split_first().expect("Corrupt subset");
                    if let Some((second, _)) = rest.clone().split_first() {
                        let (_, r) = right.split_at(*second.borrow() - *first.borrow() - 1);
                        *data = r;
                    } else {
                        // No more elements, the rest is empty, just discard the rest of data.
                        // An alternative implementation simply assigns data to the empty version of S.
                        // This would require additional traits so we settle for this possibly less
                        // efficient version for now.
                        let n = right.len();
                        let (_, r) = right.split_at(n);
                        *data = r;
                    }
                    *indices = rest;
                    item.into_expr()
                })
            }
            None => data_slice.split_first().map(|(item, rest)| {
                *data = rest;
                item.into_expr()
            }),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self.indices {
            Some(indices) => indices.len(),
            None => self.data.len(),
        };
        (n, Some(n))
    }
}

impl<'a, S: Set> Expression for SubsetIterExpr<'a, S> where Self: BaseExprTotalSizeHint {}
impl<'a, S: Set> ExactSizeIterator for SubsetIterExpr<'a, S> where
    Self: Iterator + BaseExprTotalSizeHint
{
}
impl<'a, S: Set> ExprSize for SubsetIterExpr<'a, S> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.data.len()
    }
}
impl<'a, S: Set> TotalExprSize for SubsetIterExpr<'a, S>
where
    Self: BaseExprTotalSizeHint,
{
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.base_total_size_hint(cwise_reduce)
    }
}

/// An expression with an associated index into some larger set. This allows us
/// to implement operations on sparse structures.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IndexedExpr<E> {
    pub(crate) index: usize,
    pub(crate) expr: E,
}

impl<E> IndexedExpr<E> {
    #[inline]
    pub fn map_expr<G>(self, f: impl FnOnce(E) -> G) -> IndexedExpr<G> {
        IndexedExpr {
            index: self.index,
            expr: f(self.expr),
        }
    }
}

impl<E> From<(usize, E)> for IndexedExpr<E> {
    #[inline]
    fn from(pair: (usize, E)) -> Self {
        IndexedExpr {
            index: pair.0,
            expr: pair.1,
        }
    }
}

impl<E: SumOp> SumOp for IndexedExpr<E> {
    type Output = IndexedExpr<E::Output>;
    #[inline]
    fn sum_op(self) -> Self::Output {
        let IndexedExpr { index, expr } = self;
        IndexedExpr {
            index,
            expr: expr.sum_op(),
        }
    }
}

impl<E: RecursiveSumOp> RecursiveSumOp for IndexedExpr<E> {
    type Output = E::Output;
    #[inline]
    fn recursive_sum(self) -> Self::Output {
        self.expr.recursive_sum()
    }
}

impl<R, L: CwiseMulOp<R>> CwiseMulOp<R> for IndexedExpr<L> {
    type Output = IndexedExpr<L::Output>;
    #[inline]
    fn cwise_mul(self, rhs: R) -> Self::Output {
        IndexedExpr {
            index: self.index,
            expr: self.expr.cwise_mul(rhs),
        }
    }
}

impl<R, L: MulOp<R>> MulOp<R> for IndexedExpr<L> {
    type Output = IndexedExpr<L::Output>;
    #[inline]
    fn mul(self, rhs: R) -> Self::Output {
        IndexedExpr {
            index: self.index,
            expr: self.expr.mul(rhs),
        }
    }
}

impl<R, L, Out> CwiseMulOp<IndexedExpr<Tensor<R>>> for Tensor<L>
where
    Tensor<L>: CwiseMulOp<Tensor<R>, Output = Out>,
{
    type Output = IndexedExpr<Out>;
    #[inline]
    fn cwise_mul(self, rhs: IndexedExpr<Tensor<R>>) -> Self::Output {
        IndexedExpr {
            index: rhs.index,
            expr: self.cwise_mul(rhs.expr),
        }
    }
}

impl<R, L, Out> MulOp<IndexedExpr<Tensor<R>>> for Tensor<L>
where
    Tensor<L>: MulOp<Tensor<R>, Output = Out>,
{
    type Output = IndexedExpr<Out>;
    #[inline]
    fn mul(self, rhs: IndexedExpr<Tensor<R>>) -> Self::Output {
        IndexedExpr {
            index: rhs.index,
            expr: self.mul(rhs.expr),
        }
    }
}

impl<R, L> AddAssignOp<IndexedExpr<Tensor<R>>> for Tensor<[L]>
where
    Tensor<[L]>: AddAssignOp<Tensor<R>>,
{
    #[inline]
    fn add_assign(&mut self, rhs: IndexedExpr<Tensor<R>>) {
        self.add_assign(rhs.expr)
    }
}

impl<R, L> AddAssignOp<IndexedExpr<Tensor<R>>> for Tensor<L>
where
    Tensor<L>: AddAssignOp<Tensor<R>>,
{
    #[inline]
    fn add_assign(&mut self, rhs: IndexedExpr<Tensor<R>>) {
        self.add_assign(rhs.expr)
    }
}

impl<R, L: AddAssignOp<R>> AddAssignOp<R> for IndexedExpr<L> {
    #[inline]
    fn add_assign(&mut self, rhs: R) {
        self.expr.add_assign(rhs);
    }
}

impl<R, L: AddOp<R>> AddOp<R> for IndexedExpr<L> {
    type Output = IndexedExpr<L::Output>;
    #[inline]
    fn add(self, rhs: R) -> Self::Output {
        IndexedExpr {
            index: self.index,
            expr: self.expr.add(rhs),
        }
    }
}

impl<R, L> SubAssignOp<IndexedExpr<Tensor<R>>> for Tensor<[L]>
where
    Tensor<[L]>: SubAssignOp<Tensor<R>>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: IndexedExpr<Tensor<R>>) {
        self.sub_assign(rhs.expr)
    }
}

impl<R, L> SubAssignOp<IndexedExpr<Tensor<R>>> for Tensor<L>
where
    Tensor<L>: SubAssignOp<Tensor<R>>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: IndexedExpr<Tensor<R>>) {
        self.sub_assign(rhs.expr)
    }
}

impl<R, L: SubAssignOp<R>> SubAssignOp<R> for IndexedExpr<L> {
    #[inline]
    fn sub_assign(&mut self, rhs: R) {
        self.expr.sub_assign(rhs);
    }
}

impl<R, L: SubOp<R>> SubOp<R> for IndexedExpr<L> {
    type Output = IndexedExpr<L::Output>;
    #[inline]
    fn sub(self, rhs: R) -> Self::Output {
        IndexedExpr {
            index: self.index,
            expr: self.expr.sub(rhs),
        }
    }
}

impl<E: Expression> Expression for IndexedExpr<E> {}
impl<E: ExprSize> ExprSize for IndexedExpr<E> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.expr.expr_size()
    }
}
impl<E: TotalExprSize> TotalExprSize for IndexedExpr<E> {
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.expr.total_size_hint(cwise_reduce)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SparseAddResult<L, R, E> {
    Left(L),
    Right(R),
    Expr(E),
}
impl<T> From<SparseAddResult<Tensor<T>, Tensor<T>, Tensor<T>>> for Tensor<T> {
    #[inline]
    fn from(res: SparseAddResult<Tensor<T>, Tensor<T>, Tensor<T>>) -> Tensor<T> {
        match res {
            SparseAddResult::Left(val) => val,
            SparseAddResult::Right(val) => val,
            SparseAddResult::Expr(val) => val,
        }
    }
}

impl<R, A: CwiseMulOp<R>, B: CwiseMulOp<R>, C: CwiseMulOp<R>> CwiseMulOp<R>
    for SparseAddResult<A, B, C>
{
    type Output = SparseAddResult<A::Output, B::Output, C::Output>;
    #[inline]
    fn cwise_mul(self, rhs: R) -> Self::Output {
        match self {
            SparseAddResult::Left(val) => SparseAddResult::Left(val.cwise_mul(rhs)),
            SparseAddResult::Right(val) => SparseAddResult::Right(val.cwise_mul(rhs)),
            SparseAddResult::Expr(val) => SparseAddResult::Expr(val.cwise_mul(rhs)),
        }
    }
}

impl<R, A: AddOp<R>, B: AddOp<R>, C: AddOp<R>> AddOp<R> for SparseAddResult<A, B, C> {
    type Output = SparseAddResult<A::Output, B::Output, C::Output>;
    #[inline]
    fn add(self, rhs: R) -> Self::Output {
        match self {
            SparseAddResult::Left(val) => SparseAddResult::Left(val.add(rhs)),
            SparseAddResult::Right(val) => SparseAddResult::Right(val.add(rhs)),
            SparseAddResult::Expr(val) => SparseAddResult::Expr(val.add(rhs)),
        }
    }
}

pub trait IntoExpr {
    type Expr;
    fn into_expr(self) -> Self::Expr;
}

impl<S: DynamicCollection, N> IntoExpr for UniChunked<S, N> {
    type Expr = UniChunkedIterExpr<S, N>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        UniChunkedIterExpr {
            data: self.data,
            chunk_size: self.chunk_size,
        }
    }
}

impl<'a, S> IntoExpr for ChunkedView<'a, S> {
    type Expr = ChunkedIterExpr<'a, S>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        ChunkedIterExpr {
            data: self.data,
            offsets: self.chunks,
        }
    }
}

impl<'a, S, T> IntoExpr for SparseView<'a, S, T>
where
    SparseIterExpr<'a, S, T>: Iterator,
{
    type Expr = SparseExpr<SparseIterExpr<'a, S, T>>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        SparseExpr::new(SparseIterExpr {
            indices: self.selection.indices,
            source: self.source,
            target: self.selection.target,
        })
    }
}

impl<'a, S> IntoExpr for SubsetView<'a, S> {
    type Expr = SubsetIterExpr<'a, S>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        let Subset { indices, data } = self;
        SubsetIterExpr { indices, data }
    }
}

impl<'a, T: Clone> IntoExpr for &'a [T] {
    type Expr = SliceIterExpr<'a, T>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        self.expr()
    }
}

impl<'a, T> IntoExpr for &'a mut [T] {
    type Expr = SliceIterExprMut<'a, T>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        self.expr_mut()
    }
}

impl<T: Clone> IntoExpr for Vec<T> {
    type Expr = VecIterExpr<T>;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        VecIterExpr(self.into_iter())
    }
}

pub trait Expr<'a> {
    type Output;
    fn expr(&'a self) -> Self::Output;
}

impl<'a, T: 'a + Clone> Expr<'a> for [T] {
    type Output = SliceIterExpr<'a, T>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter())
    }
}

impl<'a, T: 'a + Clone> Expr<'a> for &'a [T] {
    type Output = SliceIterExpr<'a, T>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter())
    }
}

impl<'a, T: 'a + Clone> Expr<'a> for Vec<T> {
    type Output = SliceIterExpr<'a, T>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter())
    }
}

impl<'a, S: View<'a>, N: Copy> Expr<'a> for UniChunked<S, N> {
    type Output = UniChunkedIterExpr<S::Type, N>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        UniChunkedIterExpr {
            data: self.data.view(),
            chunk_size: self.chunk_size,
        }
    }
}

impl<'a, S, O> Expr<'a> for Chunked<S, O>
where
    S: View<'a>,
    O: View<'a, Type = Offsets<&'a [usize]>>,
{
    type Output = ChunkedIterExpr<'a, S::Type>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        ChunkedIterExpr {
            data: self.data.view(),
            offsets: self.chunks.view(),
        }
    }
}

impl<'a, S, T, I> Expr<'a> for Sparse<S, T, I>
where
    S: View<'a>,
    T: View<'a>,
    I: View<'a, Type = &'a [usize]>,
    SparseIterExpr<'a, S::Type, T::Type>: Iterator,
{
    type Output = SparseExpr<SparseIterExpr<'a, S::Type, T::Type>>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        SparseExpr::new(SparseIterExpr {
            indices: self.selection.indices.view(),
            source: self.source.view(),
            target: self.selection.target.view(),
        })
    }
}

impl<'a, S, I> Expr<'a> for Subset<S, I>
where
    S: View<'a>,
    I: View<'a, Type = &'a [usize]>,
{
    type Output = SubsetIterExpr<'a, S::Type>;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        SubsetIterExpr {
            indices: self.indices.as_ref().map(|i| i.view()),
            data: self.data.view(),
        }
    }
}

impl<'a, T: ?Sized, D: 'a + ?Sized> Expr<'a> for Tensor<T>
where
    Self: AsData<Data = D>,
    D: Expr<'a>,
{
    type Output = D::Output;
    #[inline]
    fn expr(&'a self) -> Self::Output {
        self.as_data().expr()
    }
}

macro_rules! impl_bin_op {
    (common impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl<$($type_vars),*, R> $op_trait<IndexedExpr<R>> for $type {
            type Output = IndexedExpr<$op_type<Self, R>>;
            #[inline]
            fn $op_fn(self, rhs: IndexedExpr<R>) -> Self::Output {
                IndexedExpr {
                    index: rhs.index,
                    expr: $op_type::new(self, rhs.expr),
                }
            }
        }
        impl<$($type_vars),*, R: DenseExpr> $op_trait<R> for $type {
            type Output = $op_type<Self, R>;
            #[inline]
            fn $op_fn(self, rhs: R) -> Self::Output {
                $op_type::new(self, rhs)
            }
        }
    };
    (impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl_bin_op!(common impl<$($type_vars),*> $op_trait for $type { $op_type::$op_fn });
        impl<$($type_vars),*, R> $op_trait<Tensor<R>> for $type {
            type Output = $op_type<Self, Tensor<R>>;
            #[inline]
            fn $op_fn(self, rhs: Tensor<R>) -> Self::Output {
                $op_type::new(self, rhs)
            }
        }

        impl<$($type_vars),*, R> $op_trait<SparseExpr<R>> for $type
            where R: Iterator,
                  $op_type<Self, SparseExpr<R>>: Iterator
        {
            type Output = SparseExpr<$op_type<Self, SparseExpr<R>>>;
            #[inline]
            fn $op_fn(self, rhs: SparseExpr<R>) -> Self::Output {
                SparseExpr::new($op_type::new(self, rhs))
            }
        }
    };
    // For multiplication we must enumerate when multiplying against sparse expressions
    (mul impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl_bin_op!(common impl<$($type_vars),*> $op_trait for $type { $op_type::$op_fn });

        impl<$($type_vars),*, R> $op_trait<Tensor<R>> for $type {
            type Output = CwiseBinExpr<Self, Tensor<R>, Multiplication>;
            #[inline]
            fn $op_fn(self, rhs: Tensor<R>) -> Self::Output {
                CwiseBinExpr::new(self, rhs)
            }
        }

        impl<$($type_vars),*, R> $op_trait<SparseExpr<R>> for $type
            where R: Iterator,
                  $op_type<Enumerate<Self>, SparseExpr<R>>: Iterator
        {
            type Output = SparseExpr<$op_type<Enumerate<Self>, SparseExpr<R>>>;
            #[inline]
            fn $op_fn(self, rhs: SparseExpr<R>) -> Self::Output {
                SparseExpr::new($op_type::new(Enumerate::new(self), rhs))
            }
        }
    };
    (sparse impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl<$($type_vars),*, R> $op_trait<R> for $type
            where E: Iterator,
                  $op_type<Self, R>: Iterator,
        {
            type Output = SparseExpr<$op_type<Self, R>>;
            #[inline]
            fn $op_fn(self, rhs: R) -> Self::Output {
                SparseExpr::new($op_type::new(self, rhs))
            }
        }
    };
    (mul sparse impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl<$($type_vars),*, R> $op_trait<Tensor<R>> for $type
            where E: Iterator,
                  CwiseBinExpr<Self, Tensor<R>, Multiplication>: Iterator,
        {
            type Output = SparseExpr<CwiseBinExpr<Self, Tensor<R>, Multiplication>>;
            #[inline]
            fn $op_fn(self, rhs: Tensor<R>) -> Self::Output {
                SparseExpr::new(CwiseBinExpr::new(self, rhs))
            }
        }
        impl<$($type_vars),*, R: DenseExpr> $op_trait<R> for $type
            where E: Iterator,
                  $op_type<Self, Enumerate<R>>: Iterator,
        {
            type Output = SparseExpr<$op_type<Self, Enumerate<R>>>;
            #[inline]
            fn $op_fn(self, rhs: R) -> Self::Output {
                SparseExpr::new($op_type::new(self, Enumerate::new(rhs)))
            }
        }
        impl<$($type_vars),*, R: Iterator> $op_trait<SparseExpr<R>> for $type
            where E: Iterator,
                  $op_type<Self, SparseExpr<R>>: Iterator,
        {
            type Output = SparseExpr<$op_type<Self, SparseExpr<R>>>;
            #[inline]
            fn $op_fn(self, rhs: SparseExpr<R>) -> Self::Output {
                SparseExpr::new($op_type::new(self, rhs))
            }
        }
    }
}

impl_bin_op!(mul impl<T> CwiseMulOp for VecIterExpr<T> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<'a, T> CwiseMulOp for SliceIterExpr<'a, T> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<S, N> CwiseMulOp for UniChunkedIterExpr<S, N> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<'a, S> CwiseMulOp for ChunkedIterExpr<'a, S> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul sparse impl<E> CwiseMulOp for SparseExpr<E> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<'a, S> CwiseMulOp for SubsetIterExpr<'a, S> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<E, F> CwiseMulOp for CwiseUnExpr<E, F> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<A, B, F> CwiseMulOp for CwiseBinExpr<A, B, F> { CwiseMulExpr::cwise_mul });
impl_bin_op!(mul impl<A, B> CwiseMulOp for Reduce<A, B> { CwiseMulExpr::cwise_mul });

impl_bin_op!(impl<T> AddOp for VecIterExpr<T> { AddExpr::add });
impl_bin_op!(impl<'a, T> AddOp for SliceIterExpr<'a, T> { AddExpr::add });
impl_bin_op!(impl<S, N> AddOp for UniChunkedIterExpr<S, N> { AddExpr::add });
impl_bin_op!(impl<'a, S> AddOp for ChunkedIterExpr<'a, S> { AddExpr::add });
impl_bin_op!(sparse impl<E> AddOp for SparseExpr<E> { AddExpr::add });
impl_bin_op!(impl<'a, S> AddOp for SubsetIterExpr<'a, S> { AddExpr::add });
impl_bin_op!(impl<E, F> AddOp for CwiseUnExpr<E, F> { AddExpr::add });
impl_bin_op!(impl<A, B, F> AddOp for CwiseBinExpr<A, B, F> { AddExpr::add });
impl_bin_op!(impl<A, B> AddOp for Reduce<A, B> { AddExpr::add });

impl_bin_op!(impl<T> SubOp for VecIterExpr<T> { SubExpr::sub });
impl_bin_op!(impl<'a, T> SubOp for SliceIterExpr<'a, T> { SubExpr::sub });
impl_bin_op!(impl<S, N> SubOp for UniChunkedIterExpr<S, N> { SubExpr::sub });
impl_bin_op!(impl<'a, S> SubOp for ChunkedIterExpr<'a, S> { SubExpr::sub });
impl_bin_op!(sparse impl<E> SubOp for SparseExpr<E> { SubExpr::sub });
impl_bin_op!(impl<'a, S> SubOp for SubsetIterExpr<'a, S> { SubExpr::sub });
impl_bin_op!(impl<E, F> SubOp for CwiseUnExpr<E, F> { SubExpr::sub });
impl_bin_op!(impl<A, B, F> SubOp for CwiseBinExpr<A, B, F> { SubExpr::sub });
impl_bin_op!(impl<A, B> SubOp for Reduce<A, B> { SubExpr::sub });

macro_rules! impl_scalar_mul {
    (impl<$($type_vars:tt),*> for $type:ty) => {
        impl_scalar_mul!(impl<$($type_vars),*> for $type where);
    };
    (impl<$($type_vars:tt),*> for $type:ty where $($type_constraints:tt)*) => {
        impl<$($type_vars),*> MulOp<T> for $type where T: Scalar, $($type_constraints)* {
            type Output = CwiseBinExpr<Self, Tensor<T>, Multiplication>;
            #[inline]
            fn mul(self, rhs: T) -> Self::Output {
                CwiseBinExpr::new(self, Tensor { data: rhs })
            }
        }
        impl<$($type_vars),*> MulOp<Tensor<T>> for $type where T: Scalar, $($type_constraints)* {
            type Output = CwiseBinExpr<Self, Tensor<T>, Multiplication>;
            #[inline]
            fn mul(self, rhs: Tensor<T>) -> Self::Output {
                CwiseBinExpr::new(self, rhs)
            }
        }
        impl<$($type_vars),*> MulOp<$type> for Tensor<T> where T: Scalar, $($type_constraints)* {
            type Output = CwiseBinExpr<Tensor<T>, $type, Multiplication>;
            #[inline]
            fn mul(self, rhs: $type) -> Self::Output {
                CwiseBinExpr::new(self, rhs)
            }
        }
        impl<$($type_vars),*> CwiseMulOp<$type> for Tensor<T> where $($type_constraints)* {
            type Output = CwiseBinExpr<Tensor<T>, $type, Multiplication>;
            #[inline]
            fn cwise_mul(self, rhs: $type) -> Self::Output {
                CwiseBinExpr::new(self, rhs)
            }
        }
    }
}

impl_scalar_mul!(impl<T> for VecIterExpr<T>);
impl_scalar_mul!(impl<'a, T> for SliceIterExpr<'a, T>);
impl_scalar_mul!(impl<S, N, T> for UniChunkedIterExpr<S, N>);
impl_scalar_mul!(impl<'a, S, T> for ChunkedIterExpr<'a, S>);
impl_scalar_mul!(impl<'a, S, T> for SubsetIterExpr<'a, S>);
impl_scalar_mul!(impl<E, T, F> for CwiseUnExpr<E, F>);
impl_scalar_mul!(impl<A, B, T, F> for CwiseBinExpr<A, B, F>);

//impl<T, E> MulOp<T> for SparseExpr<E> where T: Scalar, E: Iterator {
//    type Output = SparseExpr<CwiseBinExpr<Self, Tensor<T>, Multiplication>>;
//    #[inline]
//    fn mul(self, rhs: T) -> Self::Output {
//        SparseExpr::new(CwiseBinExpr::new(self, Tensor::new(rhs)))
//    }
//}
//
//impl<T, E> MulOp<Tensor<T>> for SparseExpr<E> where T: Scalar, E: Iterator {
//    type Output = SparseExpr<CwiseBinExpr<Self, Tensor<T>, Multiplication>>;
//    #[inline]
//    fn mul(self, rhs: Tensor<T>) -> Self::Output {
//        SparseExpr::new(CwiseBinExpr::new(self, rhs))
//    }
//}
//
//impl<T, E, A> MulOp<SparseExpr<E>> for Tensor<T>
//where T: Scalar,
//      E: Iterator<Item = IndexedExpr<A>>,
//      Tensor<T>: MulOp<A>,
//      CwiseBinExpr<Tensor<T>, SparseExpr<E>, Multiplication>: Iterator,
//{
//    type Output = SparseExpr<CwiseBinExpr<Tensor<T>, SparseExpr<E>, Multiplication>>;
//    #[inline]
//    fn mul(self, rhs: SparseExpr<E>) -> Self::Output {
//        SparseExpr::new(CwiseBinExpr::new(self, rhs))
//    }
//}
//
impl<T, E> CwiseMulOp<SparseExpr<E>> for Tensor<T>
where
    E: Iterator,
    CwiseBinExpr<Tensor<T>, SparseExpr<E>, Multiplication>: Iterator,
{
    type Output = SparseExpr<CwiseBinExpr<Tensor<T>, SparseExpr<E>, Multiplication>>;
    #[inline]
    fn cwise_mul(self, rhs: SparseExpr<E>) -> Self::Output {
        SparseExpr::new(CwiseBinExpr::new(self, rhs))
    }
}

// Tensor multiplication
// Note that CwiseSum doesn't care if its expression is dense or sparse, so there is not special
// case here for sparse expressions (except for requiring Iterator on R)
macro_rules! impl_mul_op {
    (impl<$($type_vars:tt),*> $l:ty {$r:ty} ) => {
        impl<$($type_vars),*> MulOp<$r> for $l {
            type Output = CwiseSum<CwiseMulExpr<$l, Repeat<$r>>>;
            #[inline]
            fn mul(self, rhs: $r) -> Self::Output {
                CwiseSum::new(CwiseMulExpr::new(self, Repeat::new(rhs)))
            }
        }
    };
    (sparse impl<$($type_vars:tt),*> $l:ty {$r:ty} ) => {
        impl<$($type_vars),*> MulOp<$r> for $l
        where R: Iterator,
        {
            type Output = CwiseSum<CwiseMulExpr<$l, Repeat<$r>>>;
            #[inline]
            fn mul(self, rhs: $r) -> Self::Output {
                CwiseSum::new(CwiseMulExpr::new(self, Repeat::new(rhs)))
            }
        }
    }
}

impl_mul_op!(impl<L, R>        VecIterExpr<L> {VecIterExpr<R>});
impl_mul_op!(impl<'r, L, R>    VecIterExpr<L> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<L, R, N>     VecIterExpr<L> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'r, L, R>    VecIterExpr<L> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<L, R> VecIterExpr<L> {SparseExpr<R>});
impl_mul_op!(impl<'r, L, R>    VecIterExpr<L> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<E, F, L>     VecIterExpr<L> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<A, B, F, L>  VecIterExpr<L> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<'l, L, R>         SliceIterExpr<'l, L> {VecIterExpr<R>});
impl_mul_op!(impl<'l, 'r, L, R>     SliceIterExpr<'l, L> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<'l, L, R, N>      SliceIterExpr<'l, L> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'l, 'r, L, R>     SliceIterExpr<'l, L> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<'l, L, R>  SliceIterExpr<'l, L> {SparseExpr<R>});
impl_mul_op!(impl<'l, 'r, L, R>     SliceIterExpr<'l, L> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<'l, E, F, L>      SliceIterExpr<'l, L> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<'l, A, B, F, L>   SliceIterExpr<'l, L> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<L, M, R>        UniChunkedIterExpr<L, M> {VecIterExpr<R>});
impl_mul_op!(impl<'r, L, M, R>    UniChunkedIterExpr<L, M> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<L, M, R, N>     UniChunkedIterExpr<L, M> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'r, L, M, R>    UniChunkedIterExpr<L, M> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<L, M, R> UniChunkedIterExpr<L, M> {SparseExpr<R>});
impl_mul_op!(impl<'r, L, M, R>    UniChunkedIterExpr<L, M> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<E, F, L, M>     UniChunkedIterExpr<L, M> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<A, B, F, L, M>  UniChunkedIterExpr<L, M> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<'l, L, R>        ChunkedIterExpr<'l, L> {VecIterExpr<R>});
impl_mul_op!(impl<'l, 'r, L, R>    ChunkedIterExpr<'l, L> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<'l, L, R, N>     ChunkedIterExpr<'l, L> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'l, 'r, L, R>    ChunkedIterExpr<'l, L> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<'l, L, R> ChunkedIterExpr<'l, L> {SparseExpr<R>});
impl_mul_op!(impl<'l, 'r, L, R>    ChunkedIterExpr<'l, L> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<'l, E, F, L>     ChunkedIterExpr<'l, L> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<'l, A, B, F, L>  ChunkedIterExpr<'l, L> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<'l, L, V, R>        SparseIterExpr<'l, L, V> {VecIterExpr<R>});
impl_mul_op!(impl<'l, 'r, L,V,R>      SparseIterExpr<'l, L, V> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<'l, L, V, R, N>     SparseIterExpr<'l, L, V> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'l, 'r, L,V,  R>    SparseIterExpr<'l, L, V> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<'l, L, V, R> SparseIterExpr<'l, L, V> {SparseExpr<R>});
impl_mul_op!(impl<'l, 'r, L, V, R>    SparseIterExpr<'l, L, V> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<'l, E, F, L, V>     SparseIterExpr<'l, L, V> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<'l, A, B, F, L, V>  SparseIterExpr<'l, L, V> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<'l, L, R>        SubsetIterExpr<'l, L> {VecIterExpr<R>});
impl_mul_op!(impl<'l, 'r, L, R>    SubsetIterExpr<'l, L> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<'l, L, R, N>     SubsetIterExpr<'l, L> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'l, 'r, L, R>    SubsetIterExpr<'l, L> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<'l, L, R> SubsetIterExpr<'l, L> {SparseExpr<R>});
impl_mul_op!(impl<'l, 'r, L, R>    SubsetIterExpr<'l, L> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<'l, E, F, L>     SubsetIterExpr<'l, L> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<'l, A, B, F, L>  SubsetIterExpr<'l, L> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<L, G, R>        CwiseUnExpr<L, G> {VecIterExpr<R>});
impl_mul_op!(impl<'r, L, G, R>    CwiseUnExpr<L, G> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<L, G, R, N>     CwiseUnExpr<L, G> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'r, L, G, R>    CwiseUnExpr<L, G> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<L, G, R> CwiseUnExpr<L, G> {SparseExpr<R>});
impl_mul_op!(impl<'r, L, G, R>    CwiseUnExpr<L, G> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<E, F, L, G>     CwiseUnExpr<L, G> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<A, B, F, L, G>  CwiseUnExpr<L, G> {CwiseBinExpr<A, B, F>});

impl_mul_op!(impl<C, D, G, R>        CwiseBinExpr<C, D, G> {VecIterExpr<R>});
impl_mul_op!(impl<'r, C, D, G, R>    CwiseBinExpr<C, D, G> {SliceIterExpr<'r, R>});
impl_mul_op!(impl<C, D, G, R, N>     CwiseBinExpr<C, D, G> {UniChunkedIterExpr<R, N>});
impl_mul_op!(impl<'r, C, D, G, R>    CwiseBinExpr<C, D, G> {ChunkedIterExpr<'r, R>});
impl_mul_op!(sparse impl<C, D, G, R> CwiseBinExpr<C, D, G> {SparseExpr<R>});
impl_mul_op!(impl<'r, C, D, G, R>    CwiseBinExpr<C, D, G> {SubsetIterExpr<'r, R>});
impl_mul_op!(impl<E, F, C, D, G>     CwiseBinExpr<C, D, G> {CwiseUnExpr<E, F>});
impl_mul_op!(impl<A, B, F, C, D, G>  CwiseBinExpr<C, D, G> {CwiseBinExpr<A, B, F>});

/*
 * Repeat impls
 */

impl<E: Clone> Iterator for Repeat<E> {
    type Item = E;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.expr.clone())
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (std::usize::MAX, None)
    }
}
impl<E: Clone> DoubleEndedIterator for Repeat<E> {
    #[inline]
    fn next_back(&mut self) -> Option<E> {
        Some(self.expr.clone())
    }
}
impl<E: Clone> std::iter::FusedIterator for Repeat<E> {}

impl<E: Expression> Expression for Repeat<E> {}
impl<E> ExprSize for Repeat<E> {
    #[inline]
    fn expr_size(&self) -> usize {
        std::usize::MAX
    }
}
impl<E> TotalExprSize for Repeat<E> {
    fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
        None
    }
}

/*
 * CwiseUnExpr impls
 */

impl<E: Iterator, F, Out> Iterator for CwiseUnExpr<E, F>
where
    F: UnOp<E::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.expr.next().map(|expr| self.op.apply(expr))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.expr.size_hint()
    }
}

impl<E: Iterator + Expression, F: Reduction> Expression for CwiseUnExpr<E, F> {}
impl<E: ExprSize, F: Reduction> ExprSize for CwiseUnExpr<E, F> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.expr.expr_size()
    }
}
impl<E: TotalExprSize, F: Reduction> TotalExprSize for CwiseUnExpr<E, F> {
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.expr.total_size_hint((cwise_reduce << 1) | 1)
    }
}
impl<E: Iterator + Expression> Expression for CwiseUnExpr<E, Negation> {}
impl<E: ExprSize> ExprSize for CwiseUnExpr<E, Negation> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.expr.expr_size()
    }
}
impl<E: TotalExprSize> TotalExprSize for CwiseUnExpr<E, Negation> {
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.expr.total_size_hint(cwise_reduce)
    }
}

impl<E: ExactSizeIterator, F> ExactSizeIterator for CwiseUnExpr<E, F> where Self: Iterator {}

// Teach `Vec` types to extend themselves with small tensors.
macro_rules! impl_array_vector_traits {
    () => {};
    ($n:expr) => { // Allow optional trailing comma
        impl_array_vector_traits!($n,);
    };
    ($n:expr, $($ns:tt)*) => {
        impl<T: Scalar> IntoExpr for [T; $n] {
            type Expr = Tensor<[Tensor<T>; $n]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                self.into_tensor()
            }
        }
        impl<T: Scalar> IntoExpr for &[T; $n] {
            type Expr = Tensor<[Tensor<T>; $n]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                (*self).into_tensor()
            }
        }
        impl<'a, T: Scalar> IntoExpr for &'a mut [T; $n] {
            type Expr = &'a mut Tensor<[Tensor<T>; $n]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                self.as_mut_tensor()
            }
        }
        impl<'a, T: Scalar> Expr<'a> for [T; $n] {
            type Output = Tensor<[Tensor<T>; $n]>;
            #[inline]
            fn expr(&'a self) -> Self::Output {
                (*self).into_tensor()
            }
        }
        impl<'a, T> IntoExpr for ChunkedN<&'a [T; $n]> {
            type Expr = ChunkedNIterExpr<&'a [T]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                UniChunkedIterExpr {
                    data: self.data.view(),
                    chunk_size: self.chunk_size,
                }
            }
        }
        impl<'a, T> IntoExpr for ChunkedN<&'a mut [T; $n]> {
            type Expr = ChunkedNIterExpr<&'a mut [T]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                UniChunkedIterExpr {
                    data: self.data.view_mut(),
                    chunk_size: self.chunk_size,
                }
            }
        }
        impl<'a, T, N: Unsigned, A: Copy + 'a> IntoExpr for UniChunked<&'a [T; $n], U<N>>
            where Self: AsMatrix<Matrix = &'a Tensor<A>>,
        {
            type Expr = Tensor<A>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                *self.as_matrix()
            }
        }
        impl<'a, T, N: Unsigned, A: 'a> IntoExpr for UniChunked<&'a mut [T; $n], U<N>>
            where Self: AsMatrix<Matrix = &'a mut Tensor<A>>,
        {
            type Expr = &'a mut Tensor<A>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                self.as_matrix()
            }
        }
        impl<T: Copy + TotalExprSize> Expression for Tensor<[T; $n]> {}
        impl<T> ExprSize for Tensor<[T; $n]> {
            #[inline]
            fn expr_size(&self) -> usize {
                $n
            }
        }
        impl<T: Copy + TotalExprSize> TotalExprSize for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
                if cwise_reduce & 1 == 1 {
                    self.fold(Some(0), |acc, elem| {
                        acc.and_then(|acc| {
                            elem.total_size_hint(cwise_reduce >> 1).map(|tot| {
                                acc + if elem.expr_size() > 0 {
                                    debug_assert!(tot > 0);
                                    tot / elem.expr_size()
                                } else {
                                    debug_assert_eq!(tot, 0);
                                    0
                                }
                            })
                        })
                    })
                } else {
                    self.fold(Some(0), |acc, item| {
                        acc.and_then(|acc| item.total_size_hint(cwise_reduce >> 1).map(|tot| acc + tot))
                    })
                }
            }
        }
        impl_array_vector_traits!($($ns)*);
    };
}

impl_array_vector_traits!(1, 2, 3, 4);

macro_rules! impl_array_matrix_traits {
    () => {};
    (($r:expr, $c:expr, $cty:ident)) => { // Allow optional trailing comma
        impl_array_matrix_traits!(($r, $c, $cty),);
    };
    (($r:expr, $c:expr, $cty:ident), $($ns:tt)*) => {
        impl<T: Scalar> IntoExpr for [[T; $c]; $r] {
            type Expr = Tensor<[Tensor<[Tensor<T>; $c]>; $r]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                self.into_tensor()
            }
        }
        impl<T: Scalar> IntoExpr for &[[T; $c]; $r] {
            type Expr = Tensor<[Tensor<[Tensor<T>; $c]>; $r]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                (*self).into_tensor()
            }
        }
        impl<'a, T: Scalar> IntoExpr for &'a mut [[T; $c]; $r] {
            type Expr = &'a mut Tensor<[Tensor<[Tensor<T>; $c]>; $r]>;
            #[inline]
            fn into_expr(self) -> Self::Expr {
                self.as_mut_tensor()
            }
        }
        impl<'a, T: Scalar> Expr<'a> for [[T; $c]; $r] {
            type Output = Tensor<[Tensor<[Tensor<T>; $c]>; $r]>;
            #[inline]
            fn expr(&'a self) -> Self::Output {
                (*self).into_tensor()
            }
        }

        //impl<T: Scalar> MulOp<UniChunkedIterExpr<I, $cty>> for Tensor<[[T; $c]; $r]> {
        //    type Output = Tensor<[T; $n]>;
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn cwise_mul(mut self, rhs: Self) -> Self::Output {
        //        for i in 0..$r {
        //            for i in 0..$c {
        //                self[i] *= rhs[i];
        //            }
        //        }
        //        self
        //    }
        //}

        //impl<T: Scalar> Expression for [[T; $c]; $r] {}
        //impl<T: Scalar> ExprSize for [[T; $c]; $r] {
        //    #[inline]
        //    fn expr_size(&self) -> usize {
        //        $r * $c
        //    }
        //}
        //impl<T: Scalar> TotalExprSize for [[T; $c]; $r] {
        //    #[inline]
        //    fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
        //        Some($r * $c)
        //    }
        //}
        impl_array_matrix_traits!($($ns)*);
    };
}

impl_array_matrix_traits!(
    (1, 1, U1),
    (1, 2, U2),
    (1, 3, U3),
    (1, 4, U4),
    (2, 1, U1),
    (2, 2, U2),
    (2, 3, U3),
    (2, 4, U4),
    (3, 1, U1),
    (3, 2, U2),
    (3, 3, U3),
    (3, 4, U4),
    (4, 1, U1),
    (4, 2, U2),
    (4, 3, U3),
    (4, 4, U4),
);

/*
 * SumOp/RecursiveSumOp impls
 */

impl<T: Scalar> RecursiveSumOp for Tensor<T> {
    type Output = Tensor<T>;
    #[inline]
    fn recursive_sum(self) -> Self::Output {
        self
    }
}

impl<I, A, Out> RecursiveSumOp for I
where
    I: Iterator<Item = A>,
    A: RecursiveSumOp<Output = Out>,
    Out: Default + AddOp<Output = Out>,
{
    type Output = Out;
    #[inline]
    fn recursive_sum(self) -> Self::Output {
        self.fold(Default::default(), |acc, x| acc + x.recursive_sum())
    }
}

impl<T: Scalar> SumOp for Tensor<T> {
    type Output = Tensor<T>;
    #[inline]
    fn sum_op(self) -> Self::Output {
        self
    }
}

impl<I: Iterator> SumOp for I {
    type Output = Sum<I>;
    #[inline]
    fn sum_op(self) -> Self::Output {
        Sum::new(self)
    }
}

/*
 * DotOp Base impls
 */

impl<L: Scalar + MulOp<R>, R: Scalar> DotOp<Tensor<R>> for Tensor<L> {
    type Output = Tensor<<L as MulOp<R>>::Output>;
    #[inline]
    fn dot_op(self, rhs: Tensor<R>) -> Self::Output {
        Tensor {
            data: self.data * rhs.data,
        }
    }
}

impl<L, R, C, D> DotOp<Tensor<R>> for L
where
    L: Iterator,
    L: CwiseMulOp<Tensor<R>, Output = C>,
    C: RecursiveSumOp<Output = Tensor<D>>,
{
    type Output = Tensor<D>;
    #[inline]
    fn dot_op(self, rhs: Tensor<R>) -> Self::Output {
        self.cwise_mul(rhs).recursive_sum()
    }
}

impl<L, R, C, D> DotOp<R> for Tensor<L>
where
    R: Iterator,
    Tensor<L>: CwiseMulOp<R, Output = C>,
    // Specifying output to be tensor here helps the compiler with type inference.
    C: RecursiveSumOp<Output = Tensor<D>>,
{
    type Output = Tensor<D>;
    #[inline]
    fn dot_op(self, rhs: R) -> Self::Output {
        self.cwise_mul(rhs).recursive_sum()
    }
}

impl<L, R, C, D> DotOp<R> for L
where
    L: Iterator,
    R: Iterator,
    L: CwiseMulOp<R, Output = C>,
    // Specifying output to be tensor here helps the compiler with type inference.
    C: RecursiveSumOp<Output = Tensor<D>>,
{
    type Output = Tensor<D>;
    #[inline]
    fn dot_op(self, rhs: R) -> Self::Output {
        self.cwise_mul(rhs).recursive_sum()
    }
}

// The following implementation of Sparse.dot(Dense) handles repeating indices in the Sparse
// type. Keeping it around for future reference. This can be still accomplished with the current
// infrastructure using CwiseMulExpr, but it would require a Cylce expr of some sort.
//impl<L: Iterator, R, A, B, Out> DotOp<R> for SparseExpr<L>
//where
//    SparseExpr<L>: Iterator<Item = IndexedExpr<A>>,
//    R: Iterator<Item = B> + DenseExpr + Clone,
//    A: DotOp<B, Output = Out>,
//    B: std::fmt::Debug,
//    A: std::fmt::Debug,
//    Out: Default + AddOp<Output = Out>,
//{
//    type Output = Out;
//    fn dot_op(self, rhs: R) -> Self::Output {
//        self.scan(
//            (0, rhs.clone()),
//            |(prev_idx, cur), IndexedExpr { index, expr }| {
//                if index <= *prev_idx {
//                    // Reset the rhs iterator
//                    *cur = rhs.clone();
//                    *prev_idx = 0;
//                }
//                let rhs_val = cur
//                    .nth(index - *prev_idx)
//                    .expect("Sparse . Dense dot product index out of bounds");
//                let dot = expr.dot_op(rhs_val);
//                *prev_idx = index + 1;
//                Some(dot)
//            },
//        )
//        .fold(Default::default(), |acc, x| acc + x)
//    }
//}

//impl<L, R: Iterator, A, B, Out> DotOp<SparseExpr<R>> for L
//where
//    L: Iterator<Item = A> + DenseExpr + Clone,
//    SparseExpr<R>: Iterator<Item = IndexedExpr<B>>,
//    B: DotOp<A, Output = Out>,
//    B: std::fmt::Debug,
//    A: std::fmt::Debug,
//    Out: Default + AddOp<Output = Out>,
//{
//    type Output = Out;
//    #[inline]
//    fn dot_op(self, rhs: SparseExpr<R>) -> Self::Output {
//        rhs.dot_op(self)
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        assert_eq!(70.into_tensor(), DotOp::dot_op(a.expr(), b.expr()));
        assert_eq!(70, a.expr().dot(b.expr()));
        assert_eq!(
            70,
            Expr::expr(a.view().as_tensor()).dot(b.view().as_tensor().expr())
        );
        assert_eq!(70, a.as_tensor().expr().dot(b.as_tensor().expr()));
    }

    #[test]
    fn unichunked_dot() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        assert_eq!(70, Evaluate::eval(DotOp::dot_op(a.expr(), b.expr())));
        assert_eq!(70, a.expr().dot(b.expr()));

        let a = Chunked2::from_flat(vec![1, 2, 3, 4]);
        let b = Chunked2::from_flat(vec![5, 6, 7, 8]);
        assert_eq!(70, Expression::dot(a.expr(), b.expr()));
    }

    #[test]
    fn chunkedn_add() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        assert_eq!(
            ChunkedN::from_flat_with_stride(vec![6, 8, 10, 12], 2),
            (a.expr() + b.expr()).eval()
        );
    }

    #[test]
    fn unichunked_add_assign() {
        let mut a = Chunked2::from_flat(vec![1, 2, 3, 4]);
        let b = Chunked2::from_flat(vec![5, 6, 7, 8]);
        AddAssign::add_assign(&mut a.expr_mut(), b.expr());
        assert_eq!(Chunked2::from_flat(vec![6, 8, 10, 12]), a);
    }

    #[test]
    fn chunkedn_add_assign() {
        let mut a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        a.expr_mut().add_assign(b.expr());
        assert_eq!(ChunkedN::from_flat_with_stride(vec![6, 8, 10, 12], 2), a);
    }

    #[test]
    fn subset_unichunked_add_assign() {
        let mut a = Subset::from_unique_ordered_indices(
            vec![1, 2],
            Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6]),
        );
        let b = Subset::from_unique_ordered_indices(
            vec![0, 2],
            Chunked2::from_flat(vec![7, 8, 9, 10, 11, 12]),
        );
        AddAssign::add_assign(&mut a.expr_mut(), b.expr());
        assert_eq!(
            Subset::from_unique_ordered_indices(
                vec![1, 2],
                Chunked2::from_flat(vec![1, 2, 10, 12, 16, 18])
            ),
            a
        );
    }

    #[test]
    fn chunkedn_unichunked_add() {
        let a =
            ChunkedN::from_flat_with_stride(Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]), 2);
        let b = ChunkedN::from_flat_with_stride(
            Chunked2::from_flat(vec![9, 10, 11, 12, 13, 14, 15, 16]),
            2,
        );
        assert_eq!(
            ChunkedN::from_flat_with_stride(
                Chunked2::from_flat(vec![10, 12, 14, 16, 18, 20, 22, 24]),
                2
            ),
            Evaluate::eval(a.expr() + b.expr())
        );
    }

    #[test]
    fn scalar_mul() {
        let a = 32u32;
        let v = vec![1u32, 2, 3];
        // The reason why the expression is not directly in the assert, is because
        // The rust compiler can't figure out the correct lifetimes and needs an explicit type.
        let out: Vec<_> = (v.expr() * a).eval();
        assert_eq!(vec![32u32, 64, 96], out);

        let a = 3;
        let v = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let out: ChunkedN<Vec<_>> = (v.expr() * a).eval();
        assert_eq!(ChunkedN::from_flat_with_stride(vec![3, 6, 9, 12], 2), out);
    }

    #[test]
    fn mul_with_dot() {
        let a = 32u32;
        let v = vec![1u32, 2, 3];
        let u = vec![3u32, 2, 1];
        assert_eq!(320u32, v.expr().dot::<u32, _>(u.expr()) * a);
        assert_eq!(320u32, (v.expr().dot_op(u.expr()) * a.expr()).eval());
    }

    #[test]
    fn complex_exprs() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        let c = ChunkedN::from_flat_with_stride(vec![9, 10, 11, 12], 2);
        let out: i32 = (2i32.into_tensor() * c.expr() + (b.expr() * a.expr().dot_op(c.expr()))
            - a.expr())
        .dot(b.expr());
        assert_eq!(19626, out);
    }

    #[test]
    fn sparse_add() {
        let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
        let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
        assert_eq!(
            Sparse::from_dim(vec![0, 2, 3, 4, 5], 6, vec![1, 1, 2, 2, 6]),
            (a.expr().add(b.expr())).eval()
        );
    }

    #[test]
    fn sparse_dot() {
        let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
        let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
        assert_eq!(9, a.expr().dot(b.expr()));
    }

    #[test]
    fn chunked_dot() {
        let a = Chunked::from_sizes(vec![0, 2, 1], vec![1, 2, 3]);
        let b = Chunked::from_sizes(vec![0, 2, 1], vec![4, 5, 6]);
        assert_eq!(32, a.expr().dot(b.expr()));
    }

    #[test]
    #[ignore]
    #[should_panic]
    fn non_uniform_add() {
        // Differently sized containers cannot be added as dense expressions.
        let a = Chunked::from_sizes(vec![1, 0, 2], vec![1, 2, 3]);
        let b = Chunked::from_sizes(vec![0, 2, 1], vec![4, 5, 6]);
        let v: Chunked<Vec<i32>> = Evaluate::eval(a.expr() + b.expr());
        dbg!(v);
    }

    #[test]
    fn chunked_sparse_add() {
        let a = Chunked::from_sizes(
            vec![1, 0, 2],
            Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]),
        );
        let b = Chunked::from_sizes(
            vec![0, 2, 1],
            Sparse::from_dim(vec![3, 4, 3], 6, vec![4, 5, 6]),
        );

        let mut c = Chunked::from_offsets(vec![0], Sparse::from_dim(vec![], 6, vec![]));
        c.eval_extend(b.expr());

        assert_eq!(
            Chunked::from_sizes(
                vec![1, 2, 2],
                Sparse::from_dim(vec![0, 3, 4, 3, 5], 6, vec![1, 4, 5, 8, 3])
            ),
            Evaluate::eval(a.expr() + b.expr())
        );
    }

    #[test]
    fn sparse_chunked_add() {
        let a = Sparse::from_dim(
            vec![0, 3, 5],
            6,
            Chunked::from_sizes(vec![0, 2, 1], vec![1, 2, 3]),
        );
        let b = Sparse::from_dim(
            vec![0, 2, 5],
            6,
            Chunked::from_sizes(vec![0, 3, 1], vec![1, 2, 3, 4]),
        );
        assert_eq!(
            Sparse::from_dim(
                vec![0, 2, 3, 5],
                6,
                Chunked::from_sizes(vec![0, 3, 2, 1], vec![1, 2, 3, 1, 2, 7]),
            ),
            Evaluate::eval(a.expr() + b.expr())
        );
    }

    #[test]
    fn sparse_unichunked_add() {
        let a = Sparse::from_dim(
            vec![0, 3, 5],
            6,
            Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6]),
        );
        let b = Sparse::from_dim(
            vec![3, 4, 3],
            6,
            Chunked2::from_flat(vec![7, 8, 9, 10, 11, 12]),
        );
        assert_eq!(
            Sparse::from_dim(
                vec![0, 3, 4, 3, 5],
                6,
                Chunked2::from_flat(vec![1, 2, 10, 12, 9, 10, 11, 12, 5, 6])
            ),
            (a.expr() + b.expr()).eval()
        );
    }

    #[test]
    fn nested_unichunked_add() {
        // 2x2 Block vector
        let a = Chunked2::from_flat(Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let b = Chunked2::from_flat(Chunked2::from_flat(vec![8, 7, 6, 5, 4, 3, 2, 1]));
        assert_eq!(
            Chunked2::from_flat(Chunked2::from_flat(vec![9, 9, 9, 9, 9, 9, 9, 9])),
            (a.expr() + b.expr()).eval()
        );
    }

    #[test]
    fn subset_unichunked_add() {
        let a = Subset::from_unique_ordered_indices(
            vec![0, 3],
            Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        );
        let b = Subset::from_unique_ordered_indices(
            vec![1, 3],
            Chunked2::from_flat(vec![8, 7, 6, 5, 4, 3, 2, 1]),
        );
        assert_eq!(
            Chunked2::from_flat(vec![7, 7, 9, 9]),
            (a.expr() + b.expr()).eval()
        );
    }

    #[test]
    fn subset_vec_sub() {
        let a = Subset::from_unique_ordered_indices(vec![0, 3], vec![1u32, 2, 3, 4]);
        let b = vec![8u32, 7];
        assert_eq!(vec![9u32, 11], (a.expr() + b.expr()).eval::<Vec<u32>>());
        assert_eq!(vec![9u32, 11], (b.expr() + a.expr()).eval::<Vec<u32>>());
    }

    #[test]
    fn sparse_dense_dot() {
        let a = vec![0, 1, 4, 8, 7, 3];
        let b = Sparse::from_dim(vec![3, 4, 5], a.len(), vec![7, 8, 9]);
        // 7*8 + 8*7 + 9*3
        assert_eq!(139, a.expr().dot(b.expr()));
        assert_eq!(139, b.expr().dot(a.expr()));

        // Longer sequence
        //let a = vec![0, 1, 4, 8, 7, 3];
        //let b = Sparse::from_dim(vec![3, 4, 1, 2, 5], a.len(), vec![1, 2, 3, 4, 5]);
        //// 8*1 + 7*2 + 1*3 + 4*4 + 3*5
        //assert_eq!(56, a.expr().dot(b.expr()));
        //assert_eq!(56, b.expr().dot(a.expr()));
    }

    #[test]
    fn reduce() {
        let a = Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let b = vec![2, 1, 4];
        assert_eq!(7, b.expr().reduce(Addition));
        assert_eq!(
            [34i32, 41, 48],
            a.expr().cwise_mul(b.expr()).reduce::<[i32; 3], _>(Addition)
        );
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
        let out: Vec<i32> = Evaluate::eval(Reduce::with_op(a.expr().cwise_mul(b.expr()), Addition));
        assert_eq!(vec![34i32, 41, 48], out);
    }

    #[test]
    fn reduce_into_unichunked() {
        let b = vec![2, 1];
        let a = ChunkedN::from_flat_with_stride(
            Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            3,
        );
        let out: Chunked2<_> =
            Evaluate::eval(Reduce::with_op(a.expr().cwise_mul(b.expr()), Addition));
        assert_eq!(Chunked2::from_flat(vec![9i32, 12, 15, 18, 21, 24]), out);
    }

    #[test]
    fn contraction() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let out: Vec<i32> = Evaluate::eval(CwiseSum::new(CwiseMulExpr::new(a.expr(), b.expr())));
        assert_eq!(vec![4, 10, 18], out);

        let a = Chunked2::from_flat(vec![1, 2, 3, 4]);
        let b = Chunked2::from_flat(vec![4, 5, 6, 7]);
        assert_eq!(
            vec![14, 46],
            CwiseSum::new(CwiseMulExpr::new(a.expr(), b.expr())).eval::<Vec<i32>>()
        );

        let a = Sparse::from_dim(vec![0, 2], 4, vec![1, 2]);
        let out: Sparse<Vec<i32>, _, _> = Evaluate::eval(CwiseSum::new(a.expr()));
        assert_eq!(Sparse::from_dim(vec![0, 2], 4, vec![1, 2]), out);

        let a = Sparse::from_dim(vec![0, 2], 4, vec![2, 3]);
        let b = Sparse::from_dim(vec![1, 2], 4, vec![5, 6]);
        let out: Sparse<Vec<i32>, _, _> =
            Evaluate::eval(CwiseSum::new(CwiseMulExpr::new(a.expr(), b.expr())));
        assert_eq!(Sparse::from_dim(vec![2], 4, vec![18]), out);

        let a = Sparse::from_dim(vec![0, 2], 4, Chunked2::from_flat(vec![1, 2, 3, 4]));
        let b = Sparse::from_dim(vec![1, 2], 4, Chunked2::from_flat(vec![4, 5, 6, 7]));
        let out: Sparse<Vec<i32>, _, _> =
            Evaluate::eval(CwiseSum::new(CwiseMulExpr::new(a.expr(), b.expr())));
        assert_eq!(Sparse::from_dim(vec![2], 4, vec![46]), out);
    }

    #[test]
    fn matrix_vector_mul() {
        // Right vector multiply
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
        let b = vec![2, 1, 4];
        let out: Vec<i32> = Evaluate::eval(a.expr() * b.expr());
        assert_eq!(vec![16, 37, 58], out);

        let a = Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let b = [2, 1, 4];
        let out: Vec<i32> = Evaluate::eval(a.expr() * b.expr());
        assert_eq!(vec![16, 37, 58], out);
    }

    #[test]
    fn matrix_matrix_mul() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let out: ChunkedN<Vec<_>> = Evaluate::eval(a.expr() * a.expr());
        assert_eq!(ChunkedN::from_flat_with_stride(vec![7, 10, 15, 22], 2), out);

        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
        let out: ChunkedN<Vec<_>> = Evaluate::eval(a.expr() * a.expr());
        assert_eq!(
            ChunkedN::from_flat_with_stride(vec![30, 36, 42, 66, 81, 96, 102, 126, 150], 3),
            out
        );

        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
        let b = Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let out: Chunked3<Vec<_>> = Evaluate::eval(a.expr() * b.expr());
        assert_eq!(
            Chunked3::from_flat(vec![30, 36, 42, 66, 81, 96, 102, 126, 150]),
            out
        );
    }

    //#[test]
    //fn tensor_tensor_mul() {
    //    //let flat_a = ChunkedN::from_flat_with_stride(vec![1,2,5,6, 3,4,7,8, 9,10,13,14, 11,12,15,16], 4);
    //    // 2x2 Block matrix of 2x2 blocks
    //    let a = ChunkedN::from_flat_with_stride(
    //        Chunked2::from_flat(Chunked2::from_flat(vec![
    //            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    //        ])),
    //        2,
    //    );

    //    let out: ChunkedN<_> = Evaluate::eval(a.expr() * a.expr());
    //    assert_eq!(
    //        ChunkedN::from_flat_with_stride(
    //            Chunked2::from_flat(Chunked2::from_flat(vec![
    //                118, 132, 166, 188, 174, 188, 254, 276, 310, 356, 358, 412, 494, 540, 574, 628
    //            ])),
    //            2
    //        ),
    //        out
    //    );
    //}

    #[test]
    fn sparse_matrix_vector_mul() {
        // Variable length rows with one empty:
        // [1 0]
        // [0 0]
        //let a = Chunked::from_sizes(vec![1, 0], Sparse::from_dim(vec![0], 2, vec![1]));
        //let b = vec![1, 2];
        //let mut out = a.expr() * b.expr();
        //dbg!(&out);
        //let next = out.next().unwrap();
        //dbg!(&next);
        //let mut next = out.next().unwrap();
        //dbg!(&next);
        //let out: Vec<i32> = (a.expr() * b.expr()).eval();
        //assert_eq!(vec![1, 0], out);

        //// Empty sparse matrix
        //let a = Chunked::from_offsets(vec![0], Sparse::from_dim(Vec::new(), 1, Vec::<i32>::new()));
        //let b = vec![2];
        //let out: Vec<i32> = (a.expr() * b.expr()).eval();
        //assert_eq!(vec![0], out);

        // Sparse matrix with 2 entries in each row, there are 2 rows and 4 columns.
        let a = ChunkedN::from_flat_with_stride(
            Sparse::from_dim(vec![0, 2, 1, 3], 4, vec![1, 2, 3, 4]),
            2,
        );
        let b = vec![2, 1, 3, 4];
        let out: Vec<i32> = (a.expr() * b.expr()).eval();
        assert_eq!(vec![8, 19], out);
    }

    //TODO: There are some difficulties implementing this completely.
    //      As it stands, It seems that extending the standard Iterator to a SparseIterator Trait
    //      for all Sparse expressions should resolve some of the issues, and may even replace the
    //      SparseExpr type.
    //#[test]
    //fn sparse_matrix_matrix_mul() {
    //    // Sparse matrix with 2 entries in each row, there are 2 rows and 4 columns.
    //    // [1 . 2 . ]
    //    // [. 3 . 4 ]
    //    let a = ChunkedN::from_flat_with_stride(Sparse::from_dim(vec![0, 2, 1, 3], 4, vec![1,2,3,4]), 2);
    //    // Sparse matrix with 1 entry in each row, there are 4 rows and 2 columns.
    //    // [1 .]
    //    // [. 2]
    //    // [. 3]
    //    // [4 .]
    //    let b = ChunkedN::from_flat_with_stride(Sparse::from_dim(vec![0, 1, 1, 0], 2, vec![1,2,3,4]), 1);
    //    let ab_exp = ChunkedN::from_flat_with_stride(vec![1, 6, 16, 6], 2);
    //    let mut ab = a.expr() * b.expr();
    //    dbg!(&ab);
    //    let mut next = ab.next().unwrap();
    //    let mut v: Vec<i32> = Vec::new();
    //    v.eval_extend(next);
    //    //let mut next_next = next.expr.next().unwrap();
    //    //dbg!(&next_next);
    //    //let mut v: Vec<i32> = vec![0; 4];
    //    //v.as_mut_slice().as_mut_tensor().add_assign(next_next.expr);
    //    dbg!(&v);
    //    assert!(false);
    //    //assert_eq!(ab_exp, (a.expr() * b.expr()).eval());
    //}

    //#[test]
    //fn sparse_matrix_add() {
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
    //    let mtx = SSBlockMatrix3::from_block_triplets_iter(indices.iter().cloned(), 4, 3, chunked_blocks).data;

    //    let mtx2: Sparse<Chunked<Sparse<UniChunked<UniChunked<_,_>,_>,_,_>,_>,_,_> = Evaluate::eval(mtx.expr() + mtx.expr());

    //    let blocks = vec![
    //        // Block 1
    //        [2.0, 4.0, 6.0],
    //        [8.0, 10.0, 12.0],
    //        [14.0, 16.0, 18.0],
    //        // Block 2
    //        [2.2, 4.4, 6.6],
    //        [8.8, 11.0, 13.2],
    //        [15.4, 17.6, 19.8],
    //    ];
    //    let chunked_blocks = Chunked3::from_flat(Chunked3::from_array_vec(blocks));
    //    let exp_mtx = SSBlockMatrix3::from_block_triplets_iter(indices.iter().cloned(), 4, 3, chunked_blocks);

    //    assert_eq!(exp_mtx, mtx2);
    //}
}
