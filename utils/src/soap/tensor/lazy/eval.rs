use super::{Tensor, Scalar};
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

impl<T: Scalar> Evaluate<Tensor<T>> for T {
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
    fn eval(expr: Reduce<E, F>) -> Self {
        let Reduce { expr, op } = expr;
        expr.fold(T::default(), |acc, x| op.apply(acc, Evaluate::eval(x)))
    }
}

impl<T: Scalar, E, F, A> Evaluate<Reduce<E, F>> for Tensor<T>
where
    E: Iterator<Item = A>,
    F: BinOp<T, T, Output = T>,
    T: Evaluate<A> + Default,
{
    fn eval(expr: Reduce<E, F>) -> Self {
        Tensor::new(Evaluate::eval(expr))
    }
}

impl<I, T> Evaluate<I> for Vec<T>
where
    Self: EvalExtend<I>,
{
    fn eval(iter: I) -> Self {
        let mut v = Vec::new();
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
                self.reserve($n);
                for i in 0..$n {
                    self.push(tensor[i]);
                }
            }
        }

        impl_array_tensor_eval_traits!($($ns)*);
    };
}

impl_array_tensor_eval_traits!(1, 2, 3, 4);
/*
 * Eval Extend impls
 */

impl<T> EvalExtend<Tensor<T>> for Vec<T> {
    fn eval_extend(&mut self, value: Tensor<T>) {
        self.push(value.into_inner());
    }
}

impl<I, T> EvalExtend<I> for Vec<T>
where
    I: Iterator,
    I::Item: Into<Tensor<T>>,
{
    fn eval_extend(&mut self, iter: I) {
        self.extend(iter.map(|x| x.into().into_inner()));
    }
}

impl<I, S, N> EvalExtend<I> for UniChunked<S, N>
where
    I: Iterator,
    S: Set + EvalExtend<I::Item>,
    N: Dimension,
{
    fn eval_extend(&mut self, iter: I) {
        for elem in iter {
            let orig_len = self.data.len();
            self.data.eval_extend(elem);
            debug_assert_eq!(self.data.len() - orig_len, self.chunk_size())
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
        source.reserve_with_storage(n, iter.total_size_hint());

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
pub struct ValIter<'a, I, J> {
    iter: I,
    indices: &'a mut J
}

impl<'a, J: Push<usize>, E, I: Iterator<Item = IndexedExpr<E>>> Iterator for ValIter<'a, I, J> {
    type Item = E;
    fn next(&mut self) -> Option<Self::Item> {
        let ValIter { iter, indices } = self;
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
    S: for<'a> EvalExtend<ValIter<'a, I, J>>,
{
    fn eval_extend(&mut self, iter: I) {
        let Sparse {
            selection: Select { indices, target },
            source,
        } = self;
        debug_assert_eq!(iter.target(), target);
        source.eval_extend(ValIter { iter, indices: indices });
    }
}

/*
 * Extend by BinOpResult
 */

impl<A, B, C, I> EvalExtend<BinOpResult<A, B, C>> for I
where
    I: EvalExtend<A>,
    I: EvalExtend<B>,
    I: EvalExtend<C>,
{
    fn eval_extend(&mut self, value: BinOpResult<A, B, C>) {
        match value {
            BinOpResult::Left(val) => self.eval_extend(val),
            BinOpResult::Right(val) => self.eval_extend(val),
            BinOpResult::Expr(val) => self.eval_extend(val),
        }
    }
}
