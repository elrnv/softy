/**
 * Lazy tensor arithmetic
 */
use super::*;
use std::ops::Add as AddOp;
use std::ops::Mul as MulOp;
use std::ops::Sub as SubOp;

mod eval;
pub use eval::{Evaluate, EvalExtend};

/// Recursive Sum operator.
///
/// This is similar to `std::iter::Sum` in purpose but it fits the better with the expression
/// and is recursive, which means it will sum all expressions of expressions.
pub trait SumOp {
    type Output;
    fn sum_op(self) -> Self::Output;
}

/// Define a standard dot product operator akin to ones found in `std::ops`.
pub trait DotOp<R = Self> {
    type Output;
    fn dot_op(self, rhs: R) -> Self::Output;
}

/// Define a trait for component-wise multiplication.
///
/// We reserve the standard `Mul` trait for context-sensitive multiplication (e.g. matrix multiply)
/// since these are more common than component-wise multiplication.
pub trait CwiseMulOp<R = Self> {
    type Output;
    fn cwise_mul(self, rhs: R) -> Self::Output;
}

pub struct Addition;
pub struct Subtraction;
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
impl_default!(Multiplication);

/// A marker trait to describe additive binary operations.
///
/// More precisely, given an element `a` and an identity element `id`, implementing this trait
/// indicates that `apply(a, id) = a`.
pub trait Additive {}
impl Additive for Addition {}
impl Additive for Subtraction {}

/// A marker trait to describe multiplicative binary operations.
///
/// More precisely, given an element `a` and an identity element `id`, implementing this trait
/// indicates that `apply(a, id) = id`.
pub trait Multiplicative {}
impl Multiplicative for Multiplication {}

pub trait BinOp<L, R> {
    type Output;
    fn apply(&self, lhs: L, rhs: R) -> Self::Output;
}

impl<L, R, O, F> BinOp<L, R> for F
where
    F: Fn(L, R) -> O,
{
    type Output = O;
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        self(lhs, rhs)
    }
}

impl<L, R, O> BinOp<L, R> for Addition
where
    L: AddOp<R, Output = O>,
{
    type Output = O;
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs + rhs
    }
}

impl<L, R, O> BinOp<L, R> for Subtraction
where
    L: SubOp<R, Output = O>,
{
    type Output = O;
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs - rhs
    }
}

impl<L, R, O> BinOp<L, R> for Multiplication
where
    L: CwiseMulOp<R, Output = O>,
{
    type Output = O;
    fn apply(&self, lhs: L, rhs: R) -> Self::Output {
        lhs.cwise_mul(rhs)
    }
}

/// A lazy component-wise binary expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CwiseBinExpr<L, R, F> {
    left: L,
    right: R,
    op: F,
}

impl<L, R, F: Default> CwiseBinExpr<L, R, F> {
    fn new(left: L, right: R) -> Self {
        Self::with_op(left, right, Default::default())
    }
}

impl<L, R, F> CwiseBinExpr<L, R, F> {
    pub fn with_op(left: L, right: R, op: F) -> Self {
        CwiseBinExpr { left, right, op }
    }
}

/// A lazy `Add` expression to be evaluated at a later time.
type Add<L, R> = CwiseBinExpr<L, R, Addition>;

/// A lazy `Sub` expression to be evaluated at a later time.
type Sub<L, R> = CwiseBinExpr<L, R, Subtraction>;

/// A lazy component-wise multiply expression to be evaluated at a later time.
type CwiseMul<L, R> = CwiseBinExpr<L, R, Multiplication>;

/// A lazy reduce expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reduce<E, F> {
    expr: E,
    op: F,
}

//impl<E, F: Default> Reduce<E, F> {
//    fn new(expr: E) -> Self {
//        Self::with_op(expr, Default::default())
//    }
//}
//impl<E, F> Reduce<E, F> {
//    fn with_op(expr: E, op: F) -> Self {
//        Reduce { expr, op }
//    }
//}

/// A lazy dot expression to be evaluated at a later time. This is basically a recursive reduce.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dot<L, R> {
    left: L,
    right: R,
}

impl<L, R> Dot<L, R> {
    fn new(left: L, right: R) -> Self {
        Dot { left, right }
    }
}

/// A lazy Scalar multiplication expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScalarMul<T, S> {
    tensor: T,
    scalar: S,
}

impl<T, S> ScalarMul<T, S> {
    fn new(tensor: T, scalar: S) -> Self {
        ScalarMul { tensor, scalar }
    }
}

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

/// A trait to retrieve the target type of a sparse expression.
pub trait Target {
    type Target;
    fn target(&self) -> &Self::Target;
}

impl<'a, S, T> Target for SparseIterExpr<'a, S, T> {
    type Target = T;

    fn target(&self) -> &Self::Target {
        &self.target
    }
}

impl<'a, L, R, T, F> Target for CwiseBinExpr<SparseIterExpr<'a, L, T>, SparseIterExpr<'a, R, T>, F>
where
    T: PartialEq + std::fmt::Debug,
{
    type Target = T;

    fn target(&self) -> &Self::Target {
        debug_assert_eq!(self.left.target, self.right.target);
        &self.left.target
    }
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
impl<'a, T: DenseExpr, S> DenseExpr for ScalarMul<T, S> {}
impl<'a, A: DenseExpr, B: DenseExpr, F> DenseExpr for CwiseBinExpr<A, B, F> {}
impl<'a, A: DenseExpr, B, F> DenseExpr for CwiseBinExpr<A, Tensor<B>, F> {}
impl<'a, A, B: DenseExpr, F> DenseExpr for CwiseBinExpr<Tensor<A>, B, F> {}

/// A trait describing types that can be evaluated.
pub trait Expression {
    fn eval<T>(self) -> T
    where
        Self: Sized,
        T: Evaluate<Self>,
    {
        Evaluate::eval(self)
    }

    fn dot<T, R>(self, rhs: R) -> T
    where
        Self: Sized + DotOp<R>,
        T: Evaluate<Self::Output>,
    {
        Evaluate::eval(self.dot_op(rhs))
    }

    /// Total number of elements that can be generated with this iterator
    /// counting items generated by generated iterators.
    fn total_size_hint(&self) -> usize
    where
        Self: Iterator,
    {
        self.size_hint().1.unwrap_or(self.size_hint().0)
    }
}

impl<T> Expression for Tensor<T> {}

impl<'a, T: Clone + IntoExpr> Iterator for SliceIterExpr<'a, T> {
    type Item = T::Expr;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.clone().into_expr())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<'a, T> Expression for SliceIterExpr<'a, T> where Self: Iterator {}
impl<'a, T: Clone + IntoExpr> ExactSizeIterator for SliceIterExpr<'a, T> {}

impl<'a, T: IntoExpr> Iterator for VecIterExpr<T> {
    type Item = T::Expr;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.into_expr())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<T: IntoExpr> Expression for VecIterExpr<T> {}
impl<T: IntoExpr> ExactSizeIterator for VecIterExpr<T> {}

impl<S, N> Iterator for UniChunkedIterExpr<S, U<N>>
where
    S: Set + SplitPrefix<N> + Dummy,
    S::Prefix: IntoExpr,
    N: Unsigned,
{
    type Item = <S::Prefix as IntoExpr>::Expr;
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.data.len() / N::to_usize();
        (n, Some(n))
    }
}
impl<S: Set, N> Expression for UniChunkedIterExpr<S, N>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        self.data.len()
    }
}
impl<S, N> ExactSizeIterator for UniChunkedIterExpr<S, N> where Self: Iterator {}

impl<'a, S> Iterator for ChunkedNIterExpr<S>
where
    S: Set + SplitAt + Dummy + IntoExpr,
{
    type Item = S::Expr;
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        let (l, r) = data_slice.split_at(self.chunk_size.value());
        self.data = r;
        Some(l.into_expr())
    }

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
    fn next(&mut self) -> Option<Self::Item> {
        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        self.offsets.pop_offset().map(move |n| {
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            l.into_expr()
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.offsets.len() - 1;
        (n, Some(n))
    }
}

impl<'a, S: Set> Expression for ChunkedIterExpr<'a, S>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        self.data.len()
    }
}
impl<'a, S> ExactSizeIterator for ChunkedIterExpr<'a, S> where Self: Iterator {}

impl<'a, S, T> Iterator for SparseIterExpr<'a, S, T>
where
    S: SplitFirst + Dummy,
    S::First: IntoExpr,
{
    type Item = IndexedExpr<<S::First as IntoExpr>::Expr>;
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.indices.len();
        (n, Some(n))
    }
}

impl<'a, S, T> Expression for SparseIterExpr<'a, S, T> where Self: Iterator {}
impl<'a, S, T> ExactSizeIterator for SparseIterExpr<'a, S, T> where Self: Iterator {}

impl<'a, S> Iterator for SubsetIterExpr<'a, S>
where
    S: Set + SplitAt + SplitFirst + Dummy,
    S::First: IntoExpr,
{
    type Item = <S::First as IntoExpr>::Expr;
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self.indices {
            Some(indices) => indices.len(),
            None => self.data.len(),
        };
        (n, Some(n))
    }
}

impl<'a, S> Expression for SubsetIterExpr<'a, S> where Self: Iterator {}
impl<'a, S> ExactSizeIterator for SubsetIterExpr<'a, S> where Self: Iterator {}

/// An expression with an associated index into some larger set. This allows us
/// to implement operations on sparse structures.
pub struct IndexedExpr<E> {
    index: usize,
    expr: E,
}

impl<E> From<(usize, E)> for IndexedExpr<E> {
    fn from(pair: (usize, E)) -> Self {
        IndexedExpr {
            index: pair.0,
            expr: pair.1,
        }
    }
}

pub enum BinOpResult<L, R, E> {
    Left(L),
    Right(R),
    Expr(E),
}

impl<T> From<BinOpResult<Tensor<T>,Tensor<T>,Tensor<T>>> for Tensor<T> {
    fn from(res: BinOpResult<Tensor<T>,Tensor<T>,Tensor<T>>) -> Tensor<T> {
        match res {
            BinOpResult::Left(val) => val,
            BinOpResult::Right(val) => val,
            BinOpResult::Expr(val) => val,
        }
    }
}

pub trait IntoExpr {
    type Expr;
    fn into_expr(self) -> Self::Expr;
}

impl<S: DynamicCollection, N> IntoExpr for UniChunked<S, N> {
    type Expr = UniChunkedIterExpr<S, N>;
    fn into_expr(self) -> Self::Expr {
        UniChunkedIterExpr {
            data: self.data,
            chunk_size: self.chunk_size,
        }
    }
}

impl<'a, S> IntoExpr for ChunkedView<'a, S> {
    type Expr = ChunkedIterExpr<'a, S>;
    fn into_expr(self) -> Self::Expr {
        ChunkedIterExpr {
            data: self.data,
            offsets: self.chunks,
        }
    }
}

impl<'a, S, T> IntoExpr for SparseView<'a, S, T> {
    type Expr = SparseIterExpr<'a, S, T>;
    fn into_expr(self) -> Self::Expr {
        SparseIterExpr {
            indices: self.selection.indices,
            source: self.source,
            target: self.selection.target,
        }
    }
}

impl<'a, S> IntoExpr for SubsetView<'a, S> {
    type Expr = SubsetIterExpr<'a, S>;
    fn into_expr(self) -> Self::Expr {
        let Subset { indices, data } = self;
        SubsetIterExpr { indices, data }
    }
}

impl<'a, T: Clone> IntoExpr for &'a [T] {
    type Expr = SliceIterExpr<'a, T>;
    fn into_expr(self) -> Self::Expr {
        self.expr()
    }
}

impl<T: Clone> IntoExpr for Vec<T> {
    type Expr = VecIterExpr<T>;
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
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter())
    }
}

impl<'a, T: 'a + Clone> Expr<'a> for &'a [T] {
    type Output = SliceIterExpr<'a, T>;
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter())
    }
}

impl<'a, T: 'a + Clone> Expr<'a> for Vec<T> {
    type Output = SliceIterExpr<'a, T>;
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter())
    }
}

impl<'a, S: View<'a>, N: Copy> Expr<'a> for UniChunked<S, N> {
    type Output = UniChunkedIterExpr<S::Type, N>;
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
{
    type Output = SparseIterExpr<'a, S::Type, T::Type>;
    fn expr(&'a self) -> Self::Output {
        SparseIterExpr {
            indices: self.selection.indices.view(),
            source: self.source.view(),
            target: self.selection.target.view(),
        }
    }
}

impl<'a, S, I> Expr<'a> for Subset<S, I>
where
    S: View<'a>,
    I: View<'a, Type = &'a [usize]>,
{
    type Output = SubsetIterExpr<'a, S::Type>;
    fn expr(&'a self) -> Self::Output {
        SubsetIterExpr {
            indices: self.indices.as_ref().map(|i| i.view()),
            data: self.data.view(),
        }
    }
}

impl<'a, T: Expr<'a> + ?Sized> Expr<'a> for Tensor<T> {
    type Output = T::Output;
    fn expr(&'a self) -> Self::Output {
        self.data.expr()
    }
}

macro_rules! impl_bin_op {
    (impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl<$($type_vars),*, R> $op_trait<R> for $type {
            type Output = $op_type<Self, R>;
            fn $op_fn(self, rhs: R) -> Self::Output {
                $op_type::new(self, rhs)
            }
        }
    }
}

impl_bin_op!(impl<T> CwiseMulOp for VecIterExpr<T> { CwiseMul::cwise_mul });
impl_bin_op!(impl<'a, T> CwiseMulOp for SliceIterExpr<'a, T> { CwiseMul::cwise_mul });
impl_bin_op!(impl<S, N> CwiseMulOp for UniChunkedIterExpr<S, N> { CwiseMul::cwise_mul });
impl_bin_op!(impl<'a, S> CwiseMulOp for ChunkedIterExpr<'a, S> { CwiseMul::cwise_mul });
impl_bin_op!(impl<'a, S, T> CwiseMulOp for SparseIterExpr<'a, S, T> { CwiseMul::cwise_mul });
impl_bin_op!(impl<'a, S> CwiseMulOp for SubsetIterExpr<'a, S> { CwiseMul::cwise_mul });
impl_bin_op!(impl<A, B, F> CwiseMulOp for CwiseBinExpr<A, B, F> { CwiseMul::cwise_mul });
impl_bin_op!(impl<A, B> CwiseMulOp for ScalarMul<A, B> { CwiseMul::cwise_mul });
impl_bin_op!(impl<A, B> CwiseMulOp for Dot<A, B> { CwiseMul::cwise_mul });
impl_bin_op!(impl<A, B> CwiseMulOp for Reduce<A, B> { CwiseMul::cwise_mul });

impl_bin_op!(impl<T> AddOp for VecIterExpr<T> { Add::add });
impl_bin_op!(impl<'a, T> AddOp for SliceIterExpr<'a, T> { Add::add });
impl_bin_op!(impl<S, N> AddOp for UniChunkedIterExpr<S, N> { Add::add });
impl_bin_op!(impl<'a, S> AddOp for ChunkedIterExpr<'a, S> { Add::add });
impl_bin_op!(impl<'a, S, T> AddOp for SparseIterExpr<'a, S, T> { Add::add });
impl_bin_op!(impl<'a, S> AddOp for SubsetIterExpr<'a, S> { Add::add });
impl_bin_op!(impl<A, B, F> AddOp for CwiseBinExpr<A, B, F> { Add::add });
impl_bin_op!(impl<A, B> AddOp for ScalarMul<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Dot<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Reduce<A, B> { Add::add });

impl_bin_op!(impl<T> SubOp for VecIterExpr<T> { Sub::sub });
impl_bin_op!(impl<'a, T> SubOp for SliceIterExpr<'a, T> { Sub::sub });
impl_bin_op!(impl<S, N> SubOp for UniChunkedIterExpr<S, N> { Sub::sub });
impl_bin_op!(impl<'a, S> SubOp for ChunkedIterExpr<'a, S> { Sub::sub });
impl_bin_op!(impl<'a, S, T> SubOp for SparseIterExpr<'a, S, T> { Sub::sub });
impl_bin_op!(impl<'a, S> SubOp for SubsetIterExpr<'a, S> { Sub::sub });
impl_bin_op!(impl<A, B, F> SubOp for CwiseBinExpr<A, B, F> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for ScalarMul<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Dot<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Reduce<A, B> { Sub::sub });

macro_rules! impl_scalar_mul {
    (impl<$($type_vars:tt),*> for $type:ty) => {
        impl<$($type_vars),*> MulOp<Tensor<T>> for $type where T: Scalar {
            type Output = ScalarMul<Self, T>;
            fn mul(self, rhs: Tensor<T>) -> Self::Output {
                ScalarMul::new(self, rhs.into_inner())
            }
        }
        impl<$($type_vars),*> MulOp<T> for $type where T: Scalar {
            type Output = ScalarMul<Self, T>;
            fn mul(self, rhs: T) -> Self::Output {
                ScalarMul::new(self, rhs)
            }
        }
        impl<$($type_vars),*> MulOp<$type> for Tensor<T> where T: Scalar {
            type Output = ScalarMul<$type, T>;
            fn mul(self, rhs: $type) -> Self::Output {
                ScalarMul::new(rhs, self.into_inner())
            }
        }
    }
}

impl_scalar_mul!(impl<T> for VecIterExpr<T>);
impl_scalar_mul!(impl<'a, T> for SliceIterExpr<'a, T>);
impl_scalar_mul!(impl<S, N, T> for UniChunkedIterExpr<S, N>);
impl_scalar_mul!(impl<'a, S, T> for ChunkedIterExpr<'a, S>);
impl_scalar_mul!(impl<'a, S, T, U> for SparseIterExpr<'a, S, U>);
impl_scalar_mul!(impl<'a, S, T> for SubsetIterExpr<'a, S>);
impl_scalar_mul!(impl<A, B, T, F> for CwiseBinExpr<A, B, F>);
impl_bin_op!(impl<A, B> MulOp for ScalarMul<A, B> { ScalarMul::mul });
impl_bin_op!(impl<A, B> MulOp for Dot<A, B> { ScalarMul::mul });
impl_bin_op!(impl<A, B> MulOp for Reduce<A, B> { ScalarMul::mul });

impl<L: Iterator + DenseExpr, R, F, Out> Iterator for CwiseBinExpr<L, Tensor<R>, F>
where
    R: Copy,
    F: BinOp<L::Item, Tensor<R>, Output = Out>,
{
    type Item = Out;
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().map(|l| self.op.apply(l, self.right))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left.size_hint()
    }
}

impl<L: DenseExpr + Iterator + Expression, R, F> Expression for CwiseBinExpr<L, Tensor<R>, F>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        self.left.total_size_hint()
    }
}

impl<L: DenseExpr + ExactSizeIterator, R, F> ExactSizeIterator for CwiseBinExpr<L, Tensor<R>, F> where
    Self: Iterator
{
}

impl<L, R: Iterator + DenseExpr, F, Out> Iterator for CwiseBinExpr<Tensor<L>, R, F>
where
    L: Copy,
    F: BinOp<Tensor<L>, R::Item, Output = Out>,
{
    type Item = Out;
    fn next(&mut self) -> Option<Self::Item> {
        self.right.next().map(|r| self.op.apply(self.left, r))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.right.size_hint()
    }
}

impl<L, R: DenseExpr + Iterator + Expression, F> Expression for CwiseBinExpr<Tensor<L>, R, F>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        self.right.total_size_hint()
    }
}

impl<L, R: DenseExpr + ExactSizeIterator, F> ExactSizeIterator for CwiseBinExpr<Tensor<L>, R, F> where
    Self: Iterator
{
}

impl<L: Iterator + DenseExpr, R: Iterator + DenseExpr, F, Out> Iterator for CwiseBinExpr<L, R, F>
where
    F: BinOp<L::Item, R::Item, Output = Out>,
{
    type Item = Out;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| self.op.apply(l, r)))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        // Since both iterators are dense expressions, they should know their
        // lengths exactly and they should both be the same
        assert_eq!(left, right);
        left
    }
}

impl<L: DenseExpr + Iterator + Expression, R: DenseExpr + Iterator + Expression, F> Expression
    for CwiseBinExpr<L, R, F>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        let left = self.left.total_size_hint();
        let right = self.right.total_size_hint();
        left.max(right)
    }
}

impl<L: DenseExpr + ExactSizeIterator, R: DenseExpr + ExactSizeIterator, F> ExactSizeIterator
    for CwiseBinExpr<L, R, F>
where
    Self: Iterator,
{
}

impl<'l, 'r, L, R, A, B, T, F, Out> Iterator
    for CwiseBinExpr<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>, F>
where
    SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
    SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
    F: Additive + BinOp<A, B, Output = Out>,
{
    type Item = IndexedExpr<BinOpResult<A, B, Out>>;
    fn next(&mut self) -> Option<Self::Item> {
        let left_first_index = self
            .left
            .indices
            .first()
            .cloned()
            .unwrap_or(std::usize::MAX);
        let right_first_index = self
            .right
            .indices
            .first()
            .cloned()
            .unwrap_or(std::usize::MAX);
        if left_first_index < right_first_index {
            self.left
                .next()
                .map(|IndexedExpr { index, expr }| IndexedExpr {
                    index,
                    expr: BinOpResult::Left(expr),
                })
        } else if left_first_index > right_first_index {
            self.right
                .next()
                .map(|IndexedExpr { index, expr }| IndexedExpr {
                    index,
                    expr: BinOpResult::Right(expr),
                })
        } else {
            if left_first_index == std::usize::MAX {
                return None;
            }
            Some(IndexedExpr {
                index: left_first_index,
                expr: BinOpResult::Expr(self.op.apply(
                    self.left.next().unwrap().expr,
                    self.right.next().unwrap().expr,
                )),
            })
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.min(right.0),
            left.1.and_then(|l| right.1.map(|r| l.max(r))),
        )
    }
}
impl<'l, 'r, L, R, T, F> Expression
    for CwiseBinExpr<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>, F>
where
    Self: Iterator,
    F: Additive,
    SparseIterExpr<'l, L, T>: Iterator + Expression,
    SparseIterExpr<'r, R, T>: Iterator + Expression,
{
    fn total_size_hint(&self) -> usize {
        let left = self.left.total_size_hint();
        let right = self.right.total_size_hint();
        left + right
    }
}

/*
 * SumOp impls
 */

impl<T: Scalar> SumOp for Tensor<T> {
    type Output = Tensor<T>;
    fn sum_op(self) -> Self::Output {
        self
    }
}

impl<I, A, Out> SumOp for I
where
    I: Iterator<Item = A>,
    A: SumOp<Output = Out>,
    Out: Default + AddOp<Output = Out>,
{
    type Output = Out;
    fn sum_op(mut self) -> Self::Output {
        self.fold(Default::default(), |acc, x| acc + x.sum_op())
    }
}

/*
 * DotOp Base impls
 */

impl<L: Scalar + MulOp<R>, R: Scalar> DotOp<Tensor<R>> for Tensor<L> {
    type Output = Tensor<<L as MulOp<R>>::Output>;
    fn dot_op(self, rhs: Tensor<R>) -> Self::Output {
        Tensor::new(self.data * rhs.data)
    }
}

impl<L, R, C> DotOp<R> for L
where
    L: Iterator + DenseExpr,
    R: Iterator + DenseExpr,
    L: CwiseMulOp<R, Output = C>,
    C: SumOp,
{
    type Output = C::Output;
    fn dot_op(self, rhs: R) -> Self::Output {
        self.cwise_mul(rhs).sum_op()
    }
}

impl<'l, 'r, L, R, T, C> DotOp<SparseIterExpr<'r, R, T>> for SparseIterExpr<'l, L, T>
where
    SparseIterExpr<'l, L, T>: Iterator,
    SparseIterExpr<'r, R, T>: Iterator,
    SparseIterExpr<'l, L, T>: CwiseMulOp<SparseIterExpr<'r, R, T>, Output = C>,
    C: SumOp,
{
    type Output = C::Output;
    fn dot_op(self, rhs: SparseIterExpr<'r, R, T>) -> Self::Output {
        self.cwise_mul(rhs).sum_op()
    }
}

impl<'l, L, R, A, B, T, Out> DotOp<R> for SparseIterExpr<'l, L, T>
where
    SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
    R: Iterator<Item = B> + DenseExpr + Clone,
    A: DotOp<B, Output = Out>,
    B: std::fmt::Debug,
    A: std::fmt::Debug,
    Out: Default + AddOp<Output = Out>,
{
    type Output = Out;
    fn dot_op(self, rhs: R) -> Self::Output {
        self.scan(
            (0, rhs.clone()),
            |(prev_idx, cur), IndexedExpr { index, expr }| {
                if index <= *prev_idx {
                    // Reset the rhs iterator
                    *cur = rhs.clone();
                    *prev_idx = 0;
                }
                let rhs_val = cur
                    .nth(index - *prev_idx)
                    .expect("Sparse . Dense dot product index out of bounds");
                let dot = expr.dot_op(rhs_val);
                *prev_idx = index + 1;
                Some(dot)
            },
        )
        .fold(Default::default(), |acc, x| acc + x)
    }
}

impl<'r, L, R, A, B, T, Out> DotOp<SparseIterExpr<'r, R, T>> for L
where
    L: Iterator<Item = A> + DenseExpr + Clone,
    SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
    B: DotOp<A, Output = Out>,
    B: std::fmt::Debug,
    A: std::fmt::Debug,
    Out: Default + AddOp<Output = Out>,
{
    type Output = Out;
    fn dot_op(self, rhs: SparseIterExpr<'r, R, T>) -> Self::Output {
        rhs.dot_op(self)
    }
}

impl<'l, 'r, L, R, A, B, T> Iterator
    for CwiseMul<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
where
    SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
    SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
    A: CwiseMulOp<B>,
{
    type Item = <A as CwiseMulOp<B>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        if self.left.indices.is_empty() || self.right.indices.is_empty() {
            return None;
        }
        loop {
            let left_first_index = unsafe { self.left.indices.get_unchecked(0) };
            let right_first_index = unsafe { self.right.indices.get_unchecked(0) };
            if left_first_index < right_first_index {
                self.left.next();
                if self.left.indices.is_empty() {
                    return None;
                }
            } else if left_first_index > right_first_index {
                self.right.next();
                if self.right.indices.is_empty() {
                    return None;
                }
            } else {
                return Some(self.op.apply(
                    self.left.next().unwrap().expr,
                    self.right.next().unwrap().expr,
                ));
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.min(right.0),
            // Yes min because excess elements must vanish when multiplied by zero.
            left.1.and_then(|l| right.1.map(|r| l.min(r))),
        )
    }
}

impl<'l, 'r, L, R, T> Expression for CwiseMul<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
where
    Self: Iterator,
    SparseIterExpr<'l, L, T>: Iterator + Expression,
    SparseIterExpr<'r, R, T>: Iterator + Expression,
{
    fn total_size_hint(&self) -> usize {
        let left = self.left.total_size_hint();
        let right = self.right.total_size_hint();
        left.min(right)
    }
}

// Scalar multiplication
impl<T: Iterator, S: Scalar> Iterator for ScalarMul<T, S>
where
    T::Item: CwiseMulOp<Tensor<S>>,
{
    type Item = <T::Item as CwiseMulOp<Tensor<S>>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.tensor
            .next()
            .map(|t| t.cwise_mul(Tensor::new(self.scalar)))
    }
}

impl<T: Iterator + Expression, S: Scalar> Expression for ScalarMul<T, S>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        self.tensor.total_size_hint()
    }
}


// Teach `Vec` types to extend themselves with small tensors.
macro_rules! impl_array_tensor_traits {
    () => {};
    ($n:expr) => { // Allow optional trailing comma
        impl_array_tensor_traits!($n,);
    };
    ($n:expr, $($ns:tt)*) => {
        impl<T: Scalar> IntoExpr for [T; $n] {
            type Expr = Tensor<[T; $n]>;
            fn into_expr(self) -> Self::Expr {
                Tensor::new(self)
            }
        }
        impl<T: Scalar> IntoExpr for &[T; $n] {
            type Expr = Tensor<[T; $n]>;
            fn into_expr(self) -> Self::Expr {
                Tensor::new(*self)
            }
        }
        impl<'a, T, N> IntoExpr for UniChunked<&'a [T; $n], N> {
            type Expr = UniChunkedIterExpr<&'a [T], N>;
            fn into_expr(self) -> Self::Expr {
                UniChunkedIterExpr {
                    data: self.data.view(),
                    chunk_size: self.chunk_size,
                }
            }
        }
        impl_array_tensor_traits!($($ns)*);
    };
}

impl_array_tensor_traits!(1, 2, 3, 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        assert_eq!(Tensor::new(70), DotOp::dot_op(a.expr(), b.expr()));
        assert_eq!(70, a.expr().dot(b.expr()));
        assert_eq!(
            70,
            a.view().as_tensor().expr().dot(b.view().as_tensor().expr())
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
        assert_eq!(70, a.expr().dot(b.expr()));
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
        assert_eq!(320u32, (v.expr().dot_op(u.expr()) * a.expr()).eval());
    }

    #[test]
    fn complex_exprs() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        let c = ChunkedN::from_flat_with_stride(vec![9, 10, 11, 12], 2);
        assert_eq!(
            19626,
            (Tensor::new(2) * c.expr() + b.expr() * a.expr().dot_op(c.expr()) - a.expr())
                .dot(b.expr())
        );
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
    #[should_panic]
    fn non_uniform_add() {
        // Differently sized containers cannot be added as dense expressions.
        let a = Chunked::from_sizes(vec![1, 0, 2], vec![1, 2, 3]);
        let b = Chunked::from_sizes(vec![0, 2, 1], vec![4, 5, 6]);
        let _: Chunked<Vec<i32>> = Evaluate::eval(a.expr() + b.expr());
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
        assert_eq!(
            Chunked::from_sizes(
                vec![1, 2, 2],
                Sparse::from_dim(vec![0, 3, 4, 3, 5], 6, vec![1, 4, 5, 8, 3])
            ),
            (a.expr() + b.expr()).eval()
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
        let b = Sparse::from_dim(vec![3, 4, 3], a.len(), vec![7, 8, 9]);
        // 7*8 + 8*7 + 9*8
        assert_eq!(184, a.expr().dot(b.expr()));
        assert_eq!(184, b.expr().dot(a.expr()));

        // Longer sequence
        let a = vec![0, 1, 4, 8, 7, 3];
        let b = Sparse::from_dim(vec![3, 4, 1, 2, 5], a.len(), vec![1, 2, 3, 4, 5]);
        // 8*1 + 7*2 + 1*3 + 4*4 + 3*5
        assert_eq!(56, a.expr().dot(b.expr()));
        assert_eq!(56, b.expr().dot(a.expr()));
    }

    //#[test]
    //fn matrix_mul() {
    //    let a = Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    //    let b = vec![2, 1, 4];
    //    assert_eq!(vec![34, 41, 48], a.expr().dot(b.expr()));

    //    let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3);
    //    let b = vec![2, 1, 4];
    //    assert_eq!(vec![34, 41, 48], a.expr().dot(b.expr()));
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
    //    let mtx = SSBlockMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);

    //    let mtx2 = Evaluate::eval(mtx.expr() + mtx.expr());

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
    //    let exp_mtx = SSBlockMatrix3::from_triplets(indices.iter().cloned(), 4, 3, chunked_blocks);

    //    assert_eq!(exp_mtx, mtx2);
    //}
}
