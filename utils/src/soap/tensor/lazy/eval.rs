use super::{Scalar, Tensor};
use crate::soap::*;
use unroll::unroll_for_loops;

/// A trait describing how a value can be constructed from an iterator expression.
pub trait Evaluate<I> {
    fn eval(iter: I) -> Self;
}

/// Analogous to `std::iter::Extend` this trait allows us to reuse existing
/// buffers to store evaluated results.
pub trait EvalExtend<I> {
    fn eval_extend(&mut self, iter: I);
}

/*
 * Evaluate impls
 */

impl<T> Evaluate<SparseAddResult<Tensor<T>, Tensor<T>, Tensor<T>>> for T {
    #[inline]
    fn eval(value: SparseAddResult<Tensor<T>, Tensor<T>, Tensor<T>>) -> T {
        let t: Tensor<T> = value.into();
        t.into_inner()
    }
}

impl<T> Evaluate<Tensor<T>> for T {
    #[inline]
    fn eval(expr: Tensor<T>) -> Self {
        expr.into_inner()
    }
}

impl<T: Scalar, E, F, A> Evaluate<Reduce<E, F>> for T
where
    E: Iterator<Item = A>,
    F: BinOp<T, T, Output = T>,
    T: Evaluate<A> + Default,
{
    #[inline]
    fn eval(Reduce { expr, op }: Reduce<E, F>) -> Self {
        expr.fold(T::default(), |acc, x| op.apply(acc, Evaluate::eval(x)))
    }
}

impl<T, E, F, A> Evaluate<Reduce<E, F>> for Tensor<T>
where
    E: Iterator<Item = A>,
    Tensor<T>: Evaluate<A> + Default,
    F: BinOp<Tensor<T>, Tensor<T>, Output = Tensor<T>>,
{
    fn eval(Reduce { expr, op }: Reduce<E, F>) -> Self {
        expr.fold(Default::default(), |acc, x| {
            op.apply(acc, Evaluate::eval(x))
        })
    }
}

impl<E, T, F> Evaluate<Reduce<E, F>> for Vec<T>
where
    E: Iterator + Expression,
    F: BinOpAssign<Tensor<[T]>, E::Item>,
    Vec<T>: EvalExtend<E::Item>,
{
    fn eval(mut reduce: Reduce<E, F>) -> Self {
        let mut v = Vec::with_capacity(reduce.reserve_hint());
        if let Some(row) = reduce.expr.next() {
            let start = v.len();
            v.eval_extend(row);
            let out = v[start..].as_mut_tensor();
            for row in reduce.expr {
                reduce.op.apply_assign(out, row);
            }
        }
        v
    }
}

impl<I, T> Evaluate<I> for Vec<T>
where
    I: Iterator + Expression,
    Self: EvalExtend<I>,
{
    #[inline]
    fn eval(iter: I) -> Self {
        let mut v = Vec::with_capacity(iter.reserve_hint());
        v.eval_extend(iter);
        v
    }
}

impl<I, S, N> Evaluate<I> for UniChunked<S, U<N>>
where
    Self: EvalExtend<I>,
    S: Default + Set,
    N: Unsigned + Default,
{
    #[inline]
    fn eval(iter: I) -> Self {
        let mut s = Self::default();
        s.eval_extend(iter);
        s
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

impl<I: Iterator + Expression, S, T, J, E> Evaluate<I> for Chunked<Sparse<S, T, J>>
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
        source.reserve_with_storage(n, iter.reserve_hint());

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

impl<I, S, T, J> Evaluate<I> for Sparse<S, T, J>
where
    Self: EvalExtend<I>,
    I: Target<Target = T>,
    T: Set + Clone,
    J: Default + AsRef<[usize]>,
    S: Set + Default,
{
    fn eval(iter: I) -> Self {
        let mut out = Sparse::new(
            Select::new(J::default(), iter.target().clone()),
            S::default(),
        );
        out.eval_extend(iter);
        out
    }
}

macro_rules! impl_array_tensor_eval_traits {
    () => {};
    ($n:expr) => { // Allow optional trailing comma
        impl_array_tensor_eval_traits!($n,);
    };
    ($n:expr, $($ns:tt)*) => {
        impl<T: Scalar> EvalExtend<Tensor<[T; $n]>> for Vec<T> {
            #[unroll_for_loops]
            fn eval_extend(&mut self, tensor: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.push(tensor[i]);
                }
            }
        }
        impl<T: Scalar, E, F> Evaluate<Reduce<E, F>> for [T; $n]
        where
            E: Iterator<Item = Tensor<[T; $n]>>,
            F: BinOp<Tensor<Self>, Tensor<Self>, Output = Tensor<Self>>,
        {
            fn eval(Reduce { expr, op }: Reduce<E, F>) -> Self {
                expr.fold(Tensor::new([T::zero(); $n]), |acc, x| op.apply(acc, x)).into_inner()
            }
        }

        impl_array_tensor_eval_traits!($($ns)*);
    };
}

impl_array_tensor_eval_traits!(1, 2, 3, 4);

/*
 * Eval Extend impls
 */

impl<T, E, F> EvalExtend<Reduce<E, F>> for Vec<T>
where
    E: Iterator + Expression,
    F: BinOpAssign<Tensor<[T]>, E::Item>,
    Vec<T>: EvalExtend<E::Item>,
{
    #[inline]
    fn eval_extend(&mut self, mut reduce: Reduce<E, F>) {
        if let Some(row) = reduce.expr.next() {
            let start = self.len();
            self.eval_extend(row);
            let out = self[start..].as_mut_tensor();
            for row in reduce.expr {
                reduce.op.apply_assign(out, row);
            }
        }
    }
}

impl<T, A> EvalExtend<IndexedExpr<A>> for Vec<T>
where Self: EvalExtend<A>,
{
    #[inline]
    fn eval_extend(&mut self, expr: IndexedExpr<A>) {
        self.eval_extend(expr.expr);
    }
}

impl<T> EvalExtend<Tensor<T>> for Vec<T> {
    #[inline]
    fn eval_extend(&mut self, value: Tensor<T>) {
        self.push(value.into_inner());
    }
}

impl<I, T> EvalExtend<I> for Vec<T>
where
    I: Iterator + Expression,
    Vec<T>: EvalExtend<I::Item>,
{
    #[inline]
    fn eval_extend(&mut self, iter: I) {
        for i in iter {
            self.eval_extend(i);
        }
    }
}

impl<I, S, N> EvalExtend<I> for UniChunked<S, N>
where
    I: Iterator,
    S: Set + EvalExtend<I::Item>,
    N: Dimension,
{
    #[inline]
    fn eval_extend(&mut self, iter: I) {
        for elem in iter {
            let orig_len = self.data.len();
            self.data.eval_extend(elem);
            assert_eq!(self.data.len() - orig_len, self.chunk_size())
        }
    }
}

impl<I: Iterator + Expression, S, T, J, E> EvalExtend<I> for Chunked<Sparse<S, T, J>>
where
    J: Push<usize> + Reserve,
    I::Item: Iterator<Item = IndexedExpr<E>> + Target<Target = T>,
    T: Set + PartialEq + std::fmt::Debug,
    S: Set + Reserve + EvalExtend<E>,
{
    #[inline]
    fn eval_extend(&mut self, iter: I) {
        let Chunked {
            chunks: offsets,
            data:
                Sparse {
                    selection: Select { indices, target },
                    source,
                },
        } = self;

        // Chunked
        let n = iter.size_hint().0;
        offsets.reserve(n);

        // Sparse
        indices.reserve(n);
        source.reserve_with_storage(n, iter.reserve_hint());

        for row in iter {
            debug_assert_eq!(row.target(), target);
            for IndexedExpr { index, expr } in row {
                indices.push(index);
                source.eval_extend(expr);
            }
            offsets.push(source.len());
        }
    }
}

impl<I: Iterator, S> EvalExtend<I> for Chunked<S>
where
    S: Set + Dense + EvalExtend<I::Item>,
{
    #[inline]
    fn eval_extend(&mut self, iter: I) {
        let Chunked {
            chunks: offsets,
            data,
        } = self;
        offsets.reserve(iter.size_hint().0);
        for elem in iter {
            data.eval_extend(elem);
            offsets.push(data.len());
        }
    }
}

/// An iterator adaptor that strips away Index information from iterators over `IndexExpr`, and
/// pushes indices to a given array.
pub struct SparseValIter<'a, I, J> {
    iter: I,
    indices: &'a mut J,
}

impl<'a, I: Expression, J> Expression for SparseValIter<'a, I, J> {}
impl<'a, I: ExprSize, J> ExprSize for SparseValIter<'a, I, J> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.iter.expr_size()
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.iter.total_size_hint(cwise_reduce)
    }
}

impl<'a, J: Push<usize>, E, I: Iterator<Item = IndexedExpr<E>>> Iterator
    for SparseValIter<'a, I, J>
{
    type Item = E;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let SparseValIter { iter, indices } = self;
        iter.next().map(|IndexedExpr { index, expr }| {
            indices.push(index);
            expr
        })
    }
}

impl<I, S, T, J> EvalExtend<I> for Sparse<S, T, J>
where
    I: Iterator + Target<Target = T>,
    T: PartialEq + std::fmt::Debug,
    J: Push<usize>,
    S: for<'a> EvalExtend<SparseValIter<'a, I, J>>,
{
    #[inline]
    fn eval_extend(&mut self, iter: I) {
        let Sparse {
            selection: Select { indices, target },
            source,
        } = self;
        debug_assert_eq!(iter.target(), target);
        source.eval_extend(SparseValIter {
            iter,
            indices: indices,
        });
    }
}

/*
 * Extend by SparseAddResult
 */

impl<A, B, C, I> EvalExtend<SparseAddResult<A, B, C>> for I
where
    I: EvalExtend<A>,
    I: EvalExtend<B>,
    I: EvalExtend<C>,
{
    #[inline]
    fn eval_extend(&mut self, value: SparseAddResult<A, B, C>) {
        match value {
            SparseAddResult::Left(val) => self.eval_extend(val),
            SparseAddResult::Right(val) => self.eval_extend(val),
            SparseAddResult::Expr(val) => self.eval_extend(val),
        }
    }
}

use std::ops::{AddAssign, SubAssign};

/*
 * Implement `AddAssign` for tensor slices on iterators to teach how iterators can be reduced.
 */

impl<T, I, A> AddAssign<I> for Tensor<[T]>
where
    I: Iterator<Item = A>,
    Tensor<T>: AddAssign<A>,
{
    #[inline]
    fn add_assign(&mut self, rhs: I) {
        for (rhs, out) in rhs.zip(self.data.iter_mut()) {
            *out.as_mut_tensor() += rhs;
        }
    }
}

impl<E, T, A> AddAssign<Reduce<E, Addition>> for Tensor<T>
where
    E: Iterator<Item = A>,
    Tensor<T>: AddAssign<A>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Reduce<E, Addition>) {
        for rhs in rhs.expr {
            self.add_assign(rhs);
        }
    }
}


impl<T, I, A> SubAssign<I> for Tensor<[T]>
where
    I: Iterator<Item = A>,
    Tensor<T>: SubAssign<A>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: I) {
        for (rhs, out) in rhs.zip(self.data.iter_mut()) {
            *out.as_mut_tensor() -= rhs;
        }
    }
}

//impl<I, S, N> SubAssign<I> for UniChunked<S, N>
//where
//    I: Iterator,
//    S: Set + EvalExtend<I::Item>,
//    N: Dimension,
//{
//    #[inline]
//    fn sub_assign(&mut self, iter: I) {
//        for elem in iter {
//            let orig_len = self.data.len();
//            self.data.eval_extend(elem);
//            assert_eq!(self.data.len() - orig_len, self.chunk_size())
//        }
//    }
//}

impl<E, T, A> SubAssign<Reduce<E, Subtraction>> for Tensor<T>
where
    E: Iterator<Item = A>,
    Tensor<T>: SubAssign<A>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Reduce<E, Subtraction>) {
        for rhs in rhs.expr {
            self.sub_assign(rhs);
        }
    }
}

