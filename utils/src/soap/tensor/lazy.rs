use super::*;
use std::ops::Add as AddOp;
use std::ops::Mul as MulOp;
use std::ops::Sub as SubOp;

/**
 * Lazy tensor arithmetic
 */

/// A lazy `Add` expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Add<L, R> {
    left: L,
    right: R,
}

/// A lazy `Sub` expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sub<L, R> {
    left: L,
    right: R,
}

/// A lazy `Dot` expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dot<L, R> {
    left: L,
    right: R,
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

/// Define a standard dot product operator akin to ones found in `std::ops`.
pub trait DotOp<R = Self> {
    type Output;
    fn dot(self, rhs: R) -> Self::Output;
}

impl<T: Scalar> DotOp for Tensor<T> {
    type Output = Tensor<T>;
    fn dot(self, rhs: Tensor<T>) -> Self::Output {
        Tensor::new(self.data * rhs.data)
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

impl<'a, L, R, T> Target for Add<SparseIterExpr<'a, L, T>, SparseIterExpr<'a, R, T>>
where
    T: PartialEq + std::fmt::Debug,
{
    type Target = T;

    fn target(&self) -> &Self::Target {
        debug_assert_eq!(self.left.target, self.right.target);
        &self.left.target
    }
}

impl<'a, L, R, T> Target for Sub<SparseIterExpr<'a, L, T>, SparseIterExpr<'a, R, T>>
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
impl<'a, T: DenseExpr, S> DenseExpr for ScalarMul<T, S> {}
impl<'a, A: DenseExpr, B: DenseExpr> DenseExpr for Add<A, B> {}
impl<'a, A: DenseExpr, B: DenseExpr> DenseExpr for Sub<A, B> {}
impl<'a, A: DenseExpr, B: DenseExpr> DenseExpr for Dot<A, B> {}

/// A trait describing types that can be evaluated.
pub trait Expression: Iterator {
    fn eval<T>(self) -> T
    where
        Self: Sized,
        T: Evaluate<Self>,
    {
        Evaluate::eval(self)
    }

    /// Total number of elements that can be generated with this iterator
    /// counting items generated by generated iterators.
    fn total_size_hint(&self) -> usize {
        self.size_hint().1.unwrap_or(self.size_hint().0)
    }
}

/// A trait describing how a value can be constructed from an iterator expression.
pub trait Evaluate<I> {
    fn eval(iter: I) -> Self;
}

/// Analogous to `std::iter::Extend` this trait allows us to reuse existing
/// buffers to store evaluated results.
pub trait EvalExtend<I> {
    fn eval_extend(&mut self, iter: I);
}

impl<'a, T: Clone> Iterator for SliceIterExpr<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.clone())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<'a, T> Expression for SliceIterExpr<'a, T> where Self: Iterator {}
impl<'a, T: Clone> ExactSizeIterator for SliceIterExpr<'a, T> {}

impl<'a, T> Iterator for VecIterExpr<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<T> Expression for VecIterExpr<T> {}
impl<T> ExactSizeIterator for VecIterExpr<T> {}

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

impl<'a, T: Expr<'a> + ?Sized> Expr<'a> for Tensor<T> {
    type Output = T::Output;
    fn expr(&'a self) -> Self::Output {
        self.data.expr()
    }
}

macro_rules! impl_bin_op {
    (impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident }) => {
        impl_bin_op!(impl<$($type_vars),*> $op_trait for $type { $op_type :: $op_fn(left, right) });
    };
    (impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_type:ident::$op_fn:ident($l:ident, $r:ident) }) => {
        impl<$($type_vars),*, R> $op_trait<R> for $type {
            type Output = $op_type<Self, R>;
            fn $op_fn(self, rhs: R) -> Self::Output {
                $op_type { $l: self, $r: rhs }
            }
        }
    }
}

impl_bin_op!(impl<T> AddOp for VecIterExpr<T> { Add::add });
impl_bin_op!(impl<'a, T> AddOp for SliceIterExpr<'a, T> { Add::add });
impl_bin_op!(impl<S, N> AddOp for UniChunkedIterExpr<S, N> { Add::add });
impl_bin_op!(impl<'a, S> AddOp for ChunkedIterExpr<'a, S> { Add::add });
impl_bin_op!(impl<'a, S, T> AddOp for SparseIterExpr<'a, S, T> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Sub<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Add<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Dot<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for ScalarMul<A, B> { Add::add });

impl_bin_op!(impl<T> SubOp for VecIterExpr<T> { Sub::sub });
impl_bin_op!(impl<'a, T> SubOp for SliceIterExpr<'a, T> { Sub::sub });
impl_bin_op!(impl<S, N> SubOp for UniChunkedIterExpr<S, N> { Sub::sub });
impl_bin_op!(impl<'a, S> SubOp for ChunkedIterExpr<'a, S> { Sub::sub });
impl_bin_op!(impl<'a, S, T> SubOp for SparseIterExpr<'a, S, T> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Sub<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Add<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Dot<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for ScalarMul<A, B> { Sub::sub });

// Decided to make all DotOp impls evaluate eagerly.
//impl_bin_op!(impl<T> DotOp for VecIterExpr<T> { Dot::dot });
//impl_bin_op!(impl<'a, T> DotOp for SliceIterExpr<'a, T> { Dot::dot });
//impl_bin_op!(impl<S, N> DotOp for UniChunkedIterExpr<S, N> { Dot::dot });
//impl_bin_op!(impl<'a, S> DotOp for ChunkedIterExpr<'a, S> { Dot::dot });
//impl_bin_op!(impl<'a, S, T> DotOp for SparseIterExpr<'a, S, T> { Dot::dot });
//impl_bin_op!(impl<A, B> DotOp for Sub<A, B> { Dot::dot });
//impl_bin_op!(impl<A, B> DotOp for Add<A, B> { Dot::dot });
//impl_bin_op!(impl<A, B> DotOp for Dot<A, B> { Dot::dot });
//impl_bin_op!(impl<A, B> DotOp for ScalarMul<A, B> { Dot::dot });

macro_rules! impl_scalar_mul {
    (impl<$($type_vars:tt),*> for $type:ty) => {
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
impl_scalar_mul!(impl<A, B, T> for Add<A, B>);
impl_scalar_mul!(impl<A, B, T> for Sub<A, B>);
impl_bin_op!(impl<A, B> MulOp for Dot<A, B> { ScalarMul::mul(tensor, scalar) });
impl_bin_op!(impl<A, B> MulOp for ScalarMul<A, B> { ScalarMul::mul(tensor, scalar) });

// Addition
impl<L: Iterator + DenseExpr, R: Iterator + DenseExpr> Iterator for Add<L, R>
where
    L::Item: AddOp<R::Item>,
{
    type Item = <L::Item as AddOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| l + r))
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
impl<L: DenseExpr + Expression, R: DenseExpr + Expression> Expression for Add<L, R>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        let left = self.left.total_size_hint();
        let right = self.right.total_size_hint();
        left.max(right)
    }
}

impl<L: DenseExpr + ExactSizeIterator, R: DenseExpr + ExactSizeIterator> ExactSizeIterator
    for Add<L, R>
where
    Self: Iterator,
{
}

macro_rules! impl_iterator_for_bin_op_sparse {
    ($bin:ident; $binop:ident::$binfn:ident) => {
        impl<'l, 'r, L, R, A, B, T> Iterator
            for $bin<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
        where
            SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
            SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
            A: $binop<B>,
            A: Into<<A as $binop<B>>::Output>,
            B: Into<<A as $binop<B>>::Output>,
        {
            type Item = IndexedExpr<<A as $binop<B>>::Output>;
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
                            expr: expr.into(),
                        })
                } else if left_first_index > right_first_index {
                    self.right
                        .next()
                        .map(|IndexedExpr { index, expr }| IndexedExpr {
                            index,
                            expr: expr.into(),
                        })
                } else {
                    if left_first_index == std::usize::MAX {
                        return None;
                    }
                    Some(
                        (
                            left_first_index,
                            self.left
                                .next()
                                .unwrap()
                                .expr
                                .$binfn(self.right.next().unwrap().expr),
                        )
                            .into(),
                    )
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
        impl<'l, 'r, L, R, T> Expression
            for $bin<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
        where
            Self: Iterator,
            SparseIterExpr<'l, L, T>: Expression,
            SparseIterExpr<'r, R, T>: Expression,
        {
            fn total_size_hint(&self) -> usize {
                let left = self.left.total_size_hint();
                let right = self.right.total_size_hint();
                left + right
            }
        }
    };
}

impl_iterator_for_bin_op_sparse!(Add; AddOp::add);

// Subtraction
impl<L: Iterator + DenseExpr, R: Iterator + DenseExpr> Iterator for Sub<L, R>
where
    L::Item: SubOp<R::Item>,
{
    type Item = <L::Item as SubOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| l - r))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        // Since both iterators are dense expressions, they should know their
        // lengths exactly and they should bot he the same
        assert_eq!(left, right);
        left
    }
}
impl<L: DenseExpr + Expression, R: DenseExpr + Expression> Expression for Sub<L, R>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        let left = self.left.total_size_hint();
        let right = self.right.total_size_hint();
        left.max(right)
    }
}

impl<L: DenseExpr + ExactSizeIterator, R: DenseExpr + ExactSizeIterator> ExactSizeIterator
    for Sub<L, R>
where
    Self: Iterator,
{
}

impl_iterator_for_bin_op_sparse!(Sub; SubOp::sub);

// Tensor contraction
impl<L: Iterator + DenseExpr, R: Iterator + DenseExpr> Iterator for Dot<L, R>
where
    L::Item: DotOp<R::Item>,
{
    type Item = <L::Item as DotOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| l.dot(r)))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        // Since both iterators are dense expressions, they should know their
        // lengths exactly and they should bot he the same
        assert_eq!(left, right);
        left
    }
}
impl<L: DenseExpr + Expression, R: DenseExpr + Expression> Expression for Dot<L, R>
where
    Self: Iterator,
{
    fn total_size_hint(&self) -> usize {
        let left = self.left.total_size_hint();
        let right = self.right.total_size_hint();
        left.min(right)
    }
}

impl<L: Iterator + DenseExpr, R: Iterator + DenseExpr> DotOp<R> for L
where
    L::Item: DotOp<R::Item>,
    <L::Item as DotOp<R::Item>>::Output: std::iter::Sum,
{
    type Output = <L::Item as DotOp<R::Item>>::Output;
    fn dot(self, rhs: R) -> Self::Output {
        std::iter::Sum::sum(Dot {
            left: self,
            right: rhs,
        })
    }
}

impl<'l, 'r, L, R, A, B, T, S: Scalar> DotOp<SparseIterExpr<'r, R, T>> for SparseIterExpr<'l, L, T>
where
    SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
    SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
    A: DotOp<B, Output = Tensor<S>>,
    Tensor<S>: num_traits::Zero,
{
    type Output = S;
    fn dot(self, rhs: SparseIterExpr<'r, R, T>) -> Self::Output {
        let tensor: Tensor<S> = std::iter::Sum::sum(Dot {
            left: self,
            right: rhs,
        });
        tensor.into_inner()
    }
}

impl<'l, 'r, L, R, A, B, T> Iterator for Dot<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
where
    SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
    SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
    A: DotOp<B>,
{
    type Item = <A as DotOp<B>>::Output;
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
                return Some(
                    self.left
                        .next()
                        .unwrap()
                        .expr
                        .dot(self.right.next().unwrap().expr),
                );
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
impl<'l, 'r, L, R, T> Expression for Dot<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
where
    Self: Iterator,
    SparseIterExpr<'l, L, T>: Expression,
    SparseIterExpr<'r, R, T>: Expression,
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
    T::Item: MulOp<S>,
{
    type Item = <T::Item as MulOp<S>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.tensor.next().map(|t| t * self.scalar)
    }
}

impl<I: Iterator, S> Evaluate<I> for ChunkedN<S>
where
    S: Set + Default + EvalExtend<I::Item>,
{
    fn eval(iter: I) -> Self {
        let mut data = S::default();
        let mut chunk_size = None;
        for elem in iter {
            let orig_len = data.len();
            data.eval_extend(elem);
            if chunk_size.is_none() {
                chunk_size = Some(data.len());
            } else {
                debug_assert_eq!(Some(data.len() - orig_len), chunk_size);
            }
        }
        UniChunked::from_flat_with_stride(data, chunk_size.unwrap())
    }
}

impl<I: Expression, S, T, J, E> Evaluate<I> for Chunked<Sparse<S, T, J>>
where
    J: Default + Push<usize> + AsRef<[usize]> + Reserve,
    I::Item: Iterator<Item = IndexedExpr<E>> + Target<Target = T>,
    T: Set + Clone + PartialEq + std::fmt::Debug,
    S: Set + Default + Reserve + EvalExtend<E>,
{
    fn eval(iter: I) -> Self {
        // Chunked
        let mut offsets = vec![0];
        let n = iter.size_hint().0;
        offsets.reserve(n);

        // Sparse
        let mut indices = J::default();
        indices.reserve(n);
        let mut target = None;
        let mut source = S::default();
        source.reserve_with_storage(n, iter.total_size_hint());

        for row in iter {
            if target.is_none() {
                target = Some(row.target().clone());
            } else {
                debug_assert_eq!(Some(row.target()), target.as_ref());
            }

            for IndexedExpr { index, expr } in row {
                indices.push(index);
                source.eval_extend(expr);
            }
            offsets.push(source.len());
        }
        Chunked::from_offsets(
            offsets,
            Sparse::new(Select::new(indices, target.unwrap()), source),
        )
    }
}

impl<I: Iterator, S> Evaluate<I> for Chunked<S>
where
    S: Set + Default + EvalExtend<I::Item>,
{
    fn eval(iter: I) -> Self {
        let mut data = S::default();
        let mut offsets = vec![0];
        offsets.reserve(iter.size_hint().0);
        for elem in iter {
            data.eval_extend(elem);
            offsets.push(data.len());
        }
        Chunked::from_offsets(offsets, data)
    }
}

impl<I, S, T, J, E> Evaluate<I> for Sparse<S, T, J>
where
    I: Iterator<Item = IndexedExpr<E>> + Target<Target = T>,
    T: Set + Clone + PartialEq + std::fmt::Debug,
    J: Push<usize> + Default + AsRef<[usize]>,
    S: Set + Default + EvalExtend<E>,
{
    fn eval(iter: I) -> Self {
        let mut indices = J::default();
        let mut source = S::default();
        let target = iter.target().clone();
        for IndexedExpr { index, expr } in iter {
            indices.push(index);
            source.eval_extend(expr);
        }
        Sparse::new(Select::new(indices, target), source)
    }
}

// Teach `Vec` types to extend themselves with small tensors.
macro_rules! impl_array_tensor_traits {
    () => {};
    ($n:expr) => { // Allow optional trailing comma
        impl_array_tensor_traits!($n,);
    };
    ($n:expr, $($ns:tt)*) => {
        impl<T: Scalar> EvalExtend<Tensor<[T; $n]>> for Vec<T> {
            #[unroll_for_loops]
            fn eval_extend(&mut self, tensor: Tensor<[T; $n]>) {
                self.reserve($n);
                for i in 0..$n {
                    self.push(tensor[i]);
                }
            }
        }
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

impl<T> EvalExtend<Tensor<T>> for Vec<T> {
    fn eval_extend(&mut self, value: Tensor<T>) {
        self.push(value.into_inner());
    }
}

impl<I, T> EvalExtend<I> for Vec<T>
where
    I: Iterator<Item = T>,
{
    fn eval_extend(&mut self, iter: I) {
        self.extend(iter);
    }
}

impl<I, S, N> EvalExtend<I> for UniChunked<S, N>
where
    I: Iterator,
    S: EvalExtend<I::Item>,
{
    fn eval_extend(&mut self, iter: I) {
        for elem in iter {
            self.data.eval_extend(elem);
        }
    }
}

impl<I, S, N> Evaluate<I> for UniChunked<S, U<N>>
where
    Self: EvalExtend<I>,
    S: Default + Set,
    N: Unsigned + Default,
{
    fn eval(iter: I) -> Self {
        let mut s = Self::default();
        s.eval_extend(iter);
        s
    }
}

impl<T, L: Iterator, R: Iterator> std::iter::Sum<Dot<L, R>> for Tensor<T>
where
    Dot<L, R>: Iterator,
    Tensor<T>: Evaluate<Dot<L, R>> + std::iter::Sum,
{
    fn sum<I: Iterator<Item = Dot<L, R>>>(iter: I) -> Tensor<T> {
        iter.map(|x| Evaluate::eval(x)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
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
    fn complex_exprs() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        let c = ChunkedN::from_flat_with_stride(vec![9, 10, 11, 12], 2);
        assert_eq!(
            19626,
            (Tensor::new(2) * c.expr() + b.expr() * a.expr().dot(c.expr()) - a.expr())
                .dot(b.expr())
        );
    }

    #[test]
    fn sparse_add() {
        let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
        let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
        assert_eq!(
            Sparse::from_dim(vec![0, 2, 3, 4, 5], 6, vec![1, 1, 2, 2, 6]),
            a.expr().add(b.expr()).eval()
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
        let _: Chunked<Vec<_>> = Evaluate::eval(a.expr() + b.expr());
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
        //assert_eq!(
        //    Sparse::from_dim(
        //        vec![0, 2, 3, 5],
        //        6,
        //        Chunked::from_sizes(vec![0, 3, 2, 1], vec![1, 2, 3, 1, 2, 7]),
        //    ),
        //    Evaluate::eval(a.expr() + b.expr())
        //);
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
