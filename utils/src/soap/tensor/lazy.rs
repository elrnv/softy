use super::*;
use std::ops::Add as AddOp;

/**
 * Lazy tensor arithmetic
 */

/// A lazy `Add` expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Add<L, R> {
    left: L,
    right: R,
}

/// A lazy `Dot` expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dot<L, R> {
    left: L,
    right: R,
}

/// Define a standard dot product operator akin to ones found in `std::ops`.
pub trait DotOp<R = Self> {
    type Output;
    fn dot(self, rhs: R) -> Self::Output;
}

/// A blanket implementation of `DotOp` for all scalar types.
impl<T: Scalar> DotOp for T {
    type Output = T;
    fn dot(self, rhs: T) -> Self::Output {
        self * rhs
    }
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
pub struct SliceIterExpr<'a, T>(std::iter::Cloned<std::slice::Iter<'a, T>>);

#[derive(Clone, Debug)]
pub struct VecIterExpr<T>(std::vec::IntoIter<T>);

#[derive(Clone, Debug)]
pub struct UniChunkedIterExpr<S, N> {
    data: S,
    chunk_size: N,
}

pub type ChunkedNIterExpr<S> = UniChunkedIterExpr<S, usize>;


/// A trait describing iterator types that can be evaluated.
pub trait IterExpr: Iterator {
    fn eval<I: EvalIterator<Self>>(self) -> I where Self: Sized {
        EvalIterator::eval(self)
    }
}

/// Blanket implementation of `IterExpr` for all `Iterator` types.
impl<T: Iterator> IterExpr for T { }


/// A trait describing how a value can be constructed from an iterator expression.
pub trait EvalIterator<I> {
    fn eval(iter: I) -> Self;
}

impl<T: Clone> Iterator for SliceIterExpr<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T: Clone, N: Dimension> Iterator for UniChunkedIterExpr<&'a [T], N> {
    type Item = SliceIterExpr<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        let (l, r) = data_slice.split_at(self.chunk_size.value());
        self.data = r;
        Some(SliceIterExpr(l.iter().cloned()))
    }
}

pub trait Expr<'a> {
    type Output;
    fn expr(&'a self) -> Self::Output;
}

impl<'a, T: 'a + Clone> Expr<'a> for [T] {
    type Output = SliceIterExpr<'a, T>;
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter().cloned())
    }
}

impl<'a, T: 'a + Clone> Expr<'a> for &'a [T] {
    type Output = SliceIterExpr<'a, T>;
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter().cloned())
    }
}

impl<'a, T: 'a + Clone> Expr<'a> for Vec<T> {
    type Output = SliceIterExpr<'a, T>;
    fn expr(&'a self) -> Self::Output {
        SliceIterExpr(self.iter().cloned())
    }
}

impl<'a, S: View<'a>, N: Copy> Expr<'a> for UniChunked<S, N> {
    type Output = UniChunkedIterExpr<S::Type, N>;
    fn expr(&'a self) -> Self::Output {
        UniChunkedIterExpr { data: self.data.view(), chunk_size: self.chunk_size }
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
                $op_type { left: self, right: rhs }
            }
        }
    }
}

impl_bin_op!(impl<'a, T> AddOp for SliceIterExpr<'a, T> { Add::add });
impl_bin_op!(impl<'a, T, N> AddOp for UniChunkedIterExpr<&'a [T], N> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Add<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Dot<A, B> { Add::add });

impl_bin_op!(impl<'a, T> DotOp for SliceIterExpr<'a, T> { Dot::dot });
impl_bin_op!(impl<'a, T, N> DotOp for UniChunkedIterExpr<&'a [T], N> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Add<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Dot<A, B> { Dot::dot });

impl<L: Iterator, R: Iterator> Iterator for Add<L, R>
where L::Item: AddOp<R::Item>
{
    type Item = <L::Item as AddOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().and_then(|l| self.right.next().map(|r| l + r))
    }
}

impl<L: Iterator, R: Iterator> Iterator for Dot<L, R>
where L::Item: DotOp<R::Item>
{
    type Item = <L::Item as DotOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().and_then(|l| self.right.next().map(|r| l.dot(r)))
    }
}


impl<I: Iterator, S> EvalIterator<I> for ChunkedN<S>
where I::Item: Iterator,
      S: Set + std::iter::FromIterator<<I::Item as Iterator>::Item>,
{
    fn eval(iter: I) -> Self {
        let mut outer_len = 0;
        let data: S = iter.flat_map(|x| {outer_len += 1; x }).collect();
        let chunk_size = data.len() / outer_len;
        UniChunked::from_flat_with_stride(data, chunk_size)
    }
}

impl<I: Iterator, S, N: Unsigned + Default> EvalIterator<I> for UniChunked<S, U<N>>
where I::Item: Iterator,
      S: Set + UniChunkable<N> + std::iter::FromIterator<<I::Item as Iterator>::Item>,
{
    fn eval(iter: I) -> Self {
        let data: S = iter.flat_map(|x| x).collect();
        UniChunked::from_flat(data)
    }
}

impl<T: Scalar, L: Iterator, R: Iterator> std::iter::Sum<Dot<L,R>> for Tensor<T>
where L::Item: DotOp<R::Item>,
      Tensor<T>: std::iter::Sum<<L::Item as DotOp<R::Item>>::Output>
{
    fn sum<I: Iterator<Item = Dot<L, R>>>(iter: I) -> Tensor<T> {
        iter.fold(Tensor { data: T::zero() }, |acc, x| acc + EvalIterator::eval(x))
    }
}

impl<T, I> EvalIterator<I> for Tensor<T>
where T: Scalar,
      I: Iterator,
      Tensor<T>: std::iter::Sum<I::Item>,
{
    fn eval(iter: I) -> Self {
        std::iter::Sum::sum(iter)
    }
}
