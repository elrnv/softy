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

impl<'a, T: Clone> Iterator for SliceIterExpr<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.clone())
    }
}

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
}

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
}

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

impl<S> IntoExpr for S
where
    S: IntoIterator + LocalGeneric,
{
    type Expr = S::IntoIter;
    fn into_expr(self) -> Self::Expr {
        self.into_iter()
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

impl<'a, S: View<'a>, T: View<'a>, I: View<'a, Type = &'a [usize]>> Expr<'a> for Sparse<S, T, I> {
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
impl_bin_op!(impl<'a, T, N> AddOp for UniChunkedIterExpr<&'a [T], N> { Add::add });
impl_bin_op!(impl<'a, S, T> AddOp for SparseIterExpr<'a, S, T> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Sub<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Add<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for Dot<A, B> { Add::add });
impl_bin_op!(impl<A, B> AddOp for ScalarMul<A, B> { Add::add });

impl_bin_op!(impl<T> SubOp for VecIterExpr<T> { Sub::sub });
impl_bin_op!(impl<'a, T> SubOp for SliceIterExpr<'a, T> { Sub::sub });
impl_bin_op!(impl<'a, T, N> SubOp for UniChunkedIterExpr<&'a [T], N> { Sub::sub });
impl_bin_op!(impl<'a, S, T> SubOp for SparseIterExpr<'a, S, T> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Sub<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Add<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for Dot<A, B> { Sub::sub });
impl_bin_op!(impl<A, B> SubOp for ScalarMul<A, B> { Sub::sub });

impl_bin_op!(impl<T> DotOp for VecIterExpr<T> { Dot::dot });
impl_bin_op!(impl<'a, T> DotOp for SliceIterExpr<'a, T> { Dot::dot });
impl_bin_op!(impl<'a, T, N> DotOp for UniChunkedIterExpr<&'a [T], N> { Dot::dot });
impl_bin_op!(impl<'a, S, T> DotOp for SparseIterExpr<'a, S, T> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Sub<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Add<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for Dot<A, B> { Dot::dot });
impl_bin_op!(impl<A, B> DotOp for ScalarMul<A, B> { Dot::dot });

impl<T> MulOp<T> for VecIterExpr<T> {
    type Output = ScalarMul<Self, T>;
    fn mul(self, rhs: T) -> Self::Output {
        ScalarMul::new(self, rhs)
    }
}
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
impl<'a, S, T, U> MulOp<U> for SparseIterExpr<'a, S, T> {
    type Output = ScalarMul<Self, U>;
    fn mul(self, rhs: U) -> Self::Output {
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
impl<L: Iterator + Dense, R: Iterator + Dense> Iterator for Add<L, R>
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

impl<'l, 'r, L, R, A, B, T> Iterator for Add<SparseIterExpr<'l, L, T>, SparseIterExpr<'r, R, T>>
where
    SparseIterExpr<'l, L, T>: Iterator<Item = IndexedExpr<A>>,
    SparseIterExpr<'r, R, T>: Iterator<Item = IndexedExpr<B>>,
    A: AddOp<B>,
    A: Into<<A as AddOp<B>>::Output>,
    B: Into<<A as AddOp<B>>::Output>,
{
    type Item = IndexedExpr<<A as AddOp<B>>::Output>;
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
                        .add(self.right.next().unwrap().expr),
                )
                    .into(),
            )
        }
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

impl<I, S, T, J> EvalIterator<I> for Sparse<S, T, J>
where
    I: Iterator<Item = IndexedExpr<S::Elem>> + Target<Target = T>,
    T: Set + Clone + PartialEq + std::fmt::Debug,
    J: Push<usize> + Default + AsRef<[usize]>,
    S: Set + Default + Push<<S as Set>::Elem>,
{
    fn eval(iter: I) -> Self {
        let mut indices = J::default();
        let mut source = S::default();
        let target = iter.target().clone();
        for IndexedExpr { index, expr } in iter {
            indices.push(index);
            source.push(expr);
        }
        Sparse::new(Select::new(indices, target), source)
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

    #[test]
    fn sparse_add() {
        let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
        let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
        assert_eq!(
            Sparse::from_dim(vec![0, 2, 3, 4, 5], 6, vec![1, 1, 2, 2, 6]),
            EvalIterator::eval(a.expr().add(b.expr()))
        );
    }

    #[test]
    fn sparse_dot() {
        let a = Sparse::from_dim(vec![0, 3, 5], 6, vec![1, 2, 3]);
        let b = Sparse::from_dim(vec![2, 4, 5], 6, vec![1, 2, 3]);
        assert_eq!(9, EvalIterator::eval(a.expr().dot(b.expr())));
    }
}
