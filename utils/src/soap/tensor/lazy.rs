use super::*;
use crate::index::Index;
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

#[derive(Clone, Debug)]
pub struct SparseIterExpr<'a, S> {
    indices: &'a [usize],
    source: S,
}

// Trait that indicates that an iterator produces elements from a dense or
// contiguous collection as opposed to a sparse one.
pub trait Dense {}
impl<'a, T> Dense for SliceIterExpr<'a, T> {}
impl<T> Dense for VecIterExpr<T> {}
impl<S, N> Dense for UniChunkedIterExpr<S, N> {}

/// A trait describing iterator types that can be evaluated.
pub trait IterExpr: Iterator {
    fn eval<I: EvalIterator<Self>>(self) -> I
    where
        Self: Sized,
    {
        EvalIterator::eval(self)
    }
}

/// Blanket implementation of `IterExpr` for all `Iterator` types.
impl<T: Iterator> IterExpr for T {}

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

/// An expression with an associated index into some larger set. This allows us
/// to implement operations on sparse structures.
pub struct IndexedExpr<E> {
    index: Index,
    expr: E,
}

impl<E> From<(usize, E)> for IndexedExpr<E> {
    fn from(pair: (usize, E)) -> Self {
        IndexedExpr {
            index: Index::new(pair.0),
            expr: pair.1,
        }
    }
}

/// Analogous to `SplitFirst`, but this trait converts the first element into an
/// iterator automatically. This lets us abstract over types that store
/// primitive types vs. types that contain other iterable collections when
/// implementing sparse expressions.
pub trait SplitFirstExpr
where Self: Sized,
{
    type First;
    fn split_first_expr(self) -> Option<(Self::First, Self)>;
}

//impl<S, O> SplitFirstExpr for Chunked<S, O>
//{
//    type FirstExpr = S;
//    fn split_first_expr(self) -> Option<(Self::FirstExpr, Self)> {
//
//    }
//}

impl<'a, T: Scalar> Iterator for SparseIterExpr<'a, &'a [T]> {
    type Item = IndexedExpr<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        source_slice.split_first().map(|(first, rest)| {
            self.source = rest;
            // We know that sparse has at least one element, no need to check again
            let first_idx = unsafe { self.indices.get_unchecked(0) };
            self.indices = &self.indices[1..];
            (*first_idx, *first).into()
        })
    }
}

//impl<'a, S> Iterator for SparseIterExpr<'a, S>
//where
//    S: SplitFirst + Dummy,
//    S::First: IntoIterator,
//{
//    type Item = IndexedExpr<S::First>;
//    fn next(&mut self) -> Option<Self::Item> {
//        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
//        source_slice.split_first().map(|(first, rest)| {
//            self.source = rest;
//            // We know that sparse has at least one element, no need to check again
//            let first_idx = unsafe { self.indices.get_unchecked(0) };
//            self.indices = &self.indices[1..];
//            (*first_idx, first).into()
//        })
//    }
//}

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
        UniChunkedIterExpr {
            data: self.data.view(),
            chunk_size: self.chunk_size,
        }
    }
}

impl<'a, S: View<'a>, T, I: View<'a, Type = &'a [usize]>> Expr<'a> for Sparse<S, T, I> {
    type Output = SparseIterExpr<'a, S::Type>;
    fn expr(&'a self) -> Self::Output {
        SparseIterExpr {
            indices: self.indices().view(),
            source: self.source.view(),
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

impl_bin_op!(impl<'a, T> AddOp for SliceIterExpr<'a, T> { Add::add });
impl_bin_op!(impl<'a, T, N> AddOp for UniChunkedIterExpr<&'a [T], N> { Add::add });
impl_bin_op!(impl<'a, S> AddOp for SparseIterExpr<'a, S> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Sub<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Add<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Dot<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for ScalarMul<A, B> { Add::add });

impl_bin_op!(impl<'a, T> SubOp for SliceIterExpr<'a, T> { Sub::sub });
impl_bin_op!(impl<'a, T, N> SubOp for UniChunkedIterExpr<&'a [T], N> { Sub::sub });
impl_bin_op!(impl<'a, S> SubOp for SparseIterExpr<'a, S> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Sub<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Add<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Dot<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for ScalarMul<A, B> { Sub::sub });

impl_bin_op!(impl<'a, T> DotOp for SliceIterExpr<'a, T> { Dot::dot });
impl_bin_op!(impl<'a, T, N> DotOp for UniChunkedIterExpr<&'a [T], N> { Dot::dot });
impl_bin_op!(impl<'a, S> DotOp for SparseIterExpr<'a, S> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Sub<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Add<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Dot<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for ScalarMul<A, B> { Dot::dot });

impl<'a, T> MulOp<T> for SliceIterExpr<'a, T> {
    type Output = ScalarMul<Self, T>;
    fn mul(self, rhs: T) -> Self::Output {
        ScalarMul::new(self, rhs)
    }
}
impl<'a, T, N> MulOp<T> for UniChunkedIterExpr<&'a [T], N> {
    type Output = ScalarMul<Self, T>;
    fn mul(self, rhs: T) -> Self::Output {
        ScalarMul::new(self, rhs)
    }
}
impl<'a, S, T> MulOp<T> for SparseIterExpr<'a, S> {
    type Output = ScalarMul<Self, T>;
    fn mul(self, rhs: T) -> Self::Output {
        ScalarMul::new(self, rhs)
    }
}
impl<'a, A, B, T: Scalar> MulOp<T> for Add<A, B> {
    type Output = ScalarMul<Self, T>;
    fn mul(self, rhs: T) -> Self::Output {
        ScalarMul::new(self, rhs)
    }
}
impl<'a, A, B, T: Scalar> MulOp<T> for Sub<A, B> {
    type Output = ScalarMul<Self, T>;
    fn mul(self, rhs: T) -> Self::Output {
        ScalarMul::new(self, rhs)
    }
}
impl_bin_op!(impl<A, B> MulOp for Dot<A, B> { ScalarMul::mul(tensor, scalar) });
impl_bin_op!(impl<A, B> MulOp for ScalarMul<A, B> { ScalarMul::mul(tensor, scalar) });

// Addition
impl<L: Iterator, R: Iterator> Iterator for Add<L, R>
where
    L::Item: AddOp<R::Item>,
{
    type Item = <L::Item as AddOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| l + r))
    }
}

// Subtraction
impl<L: Iterator, R: Iterator> Iterator for Sub<L, R>
where
    L::Item: SubOp<R::Item>,
{
    type Item = <L::Item as SubOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| l - r))
    }
}

// Tensor contraction
impl<L: Iterator + Dense, R: Iterator + Dense> Iterator for Dot<L, R>
where
    L::Item: DotOp<R::Item>,
{
    type Item = <L::Item as DotOp<R::Item>>::Output;
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| l.dot(r)))
    }
}

impl<'l, 'r, L, R, A, B> Iterator for Dot<SparseIterExpr<'l, L>, SparseIterExpr<'r, R>>
where
    SparseIterExpr<'l, L>: Iterator<Item = IndexedExpr<A>>,
    SparseIterExpr<'r, R>: Iterator<Item = IndexedExpr<B>>,
    A: DotOp<B>,
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
                // ==
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

impl<I: Iterator, S> EvalIterator<I> for ChunkedN<S>
where
    I::Item: Iterator,
    S: Set + std::iter::FromIterator<<I::Item as Iterator>::Item>,
{
    fn eval(iter: I) -> Self {
        let mut outer_len = 0;
        let data: S = iter
            .flat_map(|x| {
                outer_len += 1;
                x
            })
            .collect();
        let chunk_size = data.len() / outer_len;
        UniChunked::from_flat_with_stride(data, chunk_size)
    }
}

impl<I: Iterator, S, N: Unsigned + Default> EvalIterator<I> for UniChunked<S, U<N>>
where
    I::Item: Iterator,
    S: Set + UniChunkable<N> + std::iter::FromIterator<<I::Item as Iterator>::Item>,
{
    fn eval(iter: I) -> Self {
        let data: S = iter.flat_map(|x| x).collect();
        UniChunked::from_flat(data)
    }
}

//impl<T: Scalar, L: Dummy, R: Dummy> std::iter::Sum<Dot<IndexedExpr<L>, IndexedExpr<R>>> for Tensor<T>
//where
//    Dot<L, R>: Iterator,
//Tensor<T>: std::iter::Sum<<Dot<L, R> as Iterator>::Item>,
//{
//    fn sum<I: Iterator<Item = Dot<L, R>>>(iter: I) -> Tensor<T> {
//        iter.fold(Tensor { data: T::zero() }, |acc, x| {
//            acc + EvalIterator::eval(x)
//        })
//    }
//}

impl<T: Scalar, L: Iterator, R: Iterator> std::iter::Sum<Dot<L, R>> for Tensor<T>
where
    Dot<L, R>: Iterator + std::fmt::Debug,
    Tensor<T>: EvalIterator<Dot<L, R>> + std::fmt::Debug,
{
    fn sum<I: Iterator<Item = Dot<L, R>>>(iter: I) -> Tensor<T> {
        println!("tensor sum");
        iter.fold(Tensor { data: T::zero() }, |acc, x| {
            println!("x = {:?}", x);
            let res = EvalIterator::eval(x);
            println!("adding to acc: {:?} + {:?}", acc, res);
            acc + res
        })
    }
}

impl<T, I> EvalIterator<I> for Tensor<T>
where
    T: Scalar,
    I: Iterator,
    Tensor<T>: std::iter::Sum<I::Item>,
    I: std::fmt::Debug,
{
    fn eval(iter: I) -> Self {
        println!("tensor eval: {:?}", iter);
        std::iter::Sum::sum(iter)
    }
}

impl<T, I> EvalIterator<I> for T
where
    T: Scalar,
    I: Iterator,
    Tensor<T>: std::iter::Sum<I::Item>,
    I: std::fmt::Debug,
{
    fn eval(iter: I) -> Self {
        println!("scalar eval: {:?}", iter);
        let tensor: Tensor<T> = std::iter::Sum::sum(iter);
        tensor.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        assert_eq!(Tensor::new(70), a.expr().dot(b.expr()).eval());
        assert_eq!(
            Tensor::new(70),
            a.view()
                .as_tensor()
                .expr()
                .dot(b.view().as_tensor().expr())
                .eval()
        );
        assert_eq!(
            Tensor::new(70),
            a.as_tensor().expr().dot(b.as_tensor().expr()).eval()
        );
    }

    #[test]
    fn unichunked_dot() {
        let a = ChunkedN::from_flat_with_stride(vec![1, 2, 3, 4], 2);
        let b = ChunkedN::from_flat_with_stride(vec![5, 6, 7, 8], 2);
        assert_eq!(Tensor::new(70), a.expr().dot(b.expr()).eval());

        let a = Chunked2::from_flat(vec![1, 2, 3, 4]);
        let b = Chunked2::from_flat(vec![5, 6, 7, 8]);
        assert_eq!(Tensor::new(70), a.expr().dot(b.expr()).eval());
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

    //#[test]
    //fn sparse_add() {
    //    let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
    //    let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
    //    assert_eq!(
    //        Sparse::from_dim(vec![0, 2, 3, 4, 5], 6, vec![1, 1, 2, 2, 6]),
    //        EvalIterator::eval(a.expr().add(b.expr()))
    //    );
    //}

    #[test]
    fn sparse_dot() {
        let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
        let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
        assert_eq!(9, EvalIterator::eval(a.expr().dot(b.expr())));
    }
}
