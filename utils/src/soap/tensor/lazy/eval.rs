use super::{IterExpr, Scalar, Tensor};
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
        t.data
    }
}

impl<T> Evaluate<Tensor<T>> for T {
    #[inline]
    fn eval(expr: Tensor<T>) -> Self {
        expr.data
    }
}

impl<E, T: Evaluate<E>> Evaluate<IndexedExpr<E>> for T {
    #[inline]
    fn eval(indexed_expr: IndexedExpr<E>) -> Self {
        Evaluate::eval(indexed_expr.expr)
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
    Self: EvalExtend<E::Item>,
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

macro_rules! impl_array_vector_eval_traits {
    () => {};
    ($n:expr) => { // Allow optional trailing comma
        impl_array_vector_eval_traits!($n,);
    };
    ($n:expr, $($ns:tt)*) => {
        impl<T: Copy> EvalExtend<Tensor<[Tensor<T>; $n]>> for Vec<T> {
            #[unroll_for_loops]
            fn eval_extend(&mut self, tensor: Tensor<[Tensor<T>; $n]>) {
                for i in 0..$n {
                    self.push(unsafe { tensor.data.get_unchecked(i).data });
                }
            }
        }
        impl<T: Scalar, E, F> Evaluate<Reduce<E, F>> for [T; $n]
        where
            E: Iterator<Item = Tensor<[Tensor<T>; $n]>>,
            F: BinOp<<Self as IntoTensor>::Tensor, <Self as IntoTensor>::Tensor, Output = <Self as IntoTensor>::Tensor>,
        {
            fn eval(Reduce { expr, op }: Reduce<E, F>) -> Self {
                expr.fold([T::zero(); $n].into_tensor(), |acc, x| op.apply(acc, x)).into_data()
            }
        }

        impl_array_vector_eval_traits!($($ns)*);
    };
}

impl_array_vector_eval_traits!(1, 2, 3, 4);

macro_rules! impl_array_matrix_eval_traits {
    () => {};
    (($r:expr, $c:expr, $rty:ident, $cty:ident)) => { // Allow optional trailing comma
        impl_array_matrix_eval_traits!(($r, $c, $rty, $cty),);
    };
    (($r:expr, $c:expr, $rty:ident, $cty:ident), $($ns:tt)*) => {
        impl<T: Scalar, E, F> Evaluate<Reduce<E, F>> for [[T; $c]; $r]
        where
            E: Iterator<Item = Tensor<[Tensor<[Tensor<T>; $c]>; $r]>>,
            F: BinOp<<Self as IntoTensor>::Tensor, <Self as IntoTensor>::Tensor, Output = <Self as IntoTensor>::Tensor>,
        {
            fn eval(Reduce { expr, op }: Reduce<E, F>) -> Self {
                expr.fold([[T::zero(); $c]; $r].into_tensor(), |acc, x| op.apply(acc, x)).into_data()
            }
        }
        impl<T: Scalar> EvalExtend<Tensor<[Tensor<[Tensor<T>; $c]>; $r]>> for Vec<T> {
            #[inline]
            //#[unroll_for_loops]
            fn eval_extend(&mut self, value: Tensor<[Tensor<[Tensor<T>; $c]>; $r]>) {
                for r in 0..$r {
                    for c in 0..$c {
                        self.push(value[r][c]);
                    }
                }
            }
        }

        impl<T: Scalar> EvalExtend<Tensor<[Tensor<[Tensor<T>; $c]>; $r]>> for UniChunked<Vec<T>, $cty> {
            #[inline]
            fn eval_extend(&mut self, tensor: Tensor<[Tensor<[Tensor<T>; $c]>; $r]>) {
                self.data.extend_from_slice(tensor.into_data().as_slice());
            }
        }
        impl<T: Scalar> EvalExtend<Tensor<[Tensor<[Tensor<T>; $c]>; $r]>> for UniChunked<UniChunked<Vec<T>, $cty>, $rty> {
            #[inline]
            fn eval_extend(&mut self, tensor: Tensor<[Tensor<[Tensor<T>; $c]>; $r]>) {
                self.data.data.extend_from_slice(tensor.into_data().as_slice());
            }
        }

        impl_array_matrix_eval_traits!($($ns)*);
    };
}

impl_array_matrix_eval_traits!(
    (1, 1, U1, U1),
    (1, 2, U1, U2),
    (1, 3, U1, U3),
    (1, 4, U1, U4),
    (2, 1, U2, U1),
    (2, 2, U2, U2),
    (2, 3, U2, U3),
    (2, 4, U2, U4),
    (3, 1, U3, U1),
    (3, 2, U3, U2),
    (3, 3, U3, U3),
    (3, 4, U3, U4),
    (4, 1, U4, U1),
    (4, 2, U4, U2),
    (4, 3, U4, U3),
    (4, 4, U4, U4),
);

/*
 * Eval Extend impls
 */

impl<T, E, F> EvalExtend<Reduce<E, F>> for Vec<T>
where
    E: Iterator + DenseExpr,
    F: BinOpAssign<Tensor<[T]>, E::Item>,
    Self: EvalExtend<E::Item>,
{
    #[inline]
    fn eval_extend(&mut self, Reduce { mut expr, op }: Reduce<E, F>) {
        if let Some(row) = expr.next() {
            let start = self.len();
            self.eval_extend(row);
            let out = self[start..].as_mut_tensor();
            expr.unroll4(|row| op.apply_assign(out, row));
        }
    }
}

impl<T, E, A, F> EvalExtend<Reduce<SparseExpr<E>, F>> for Vec<T>
where
    E: Iterator<Item = IndexedExpr<A>>,
    F: BinOpAssign<Tensor<[T]>, A>,
    Self: EvalExtend<A>,
{
    #[inline]
    fn eval_extend(&mut self, mut reduce: Reduce<SparseExpr<E>, F>) {
        // Ignore the indices since all entries are being reduced anyways.
        if let Some(row) = reduce.expr.next() {
            let start = self.len();
            self.eval_extend(row.expr);
            let out = self[start..].as_mut_tensor();
            for row in reduce.expr {
                reduce.op.apply_assign(out, row.expr);
            }
        }
    }
}

//impl<T, A> EvalExtend<IndexedExpr<A>> for Vec<T>
//where Self: EvalExtend<A>,
//{
//    #[inline]
//    fn eval_extend(&mut self, expr: IndexedExpr<A>) {
//        self.eval_extend(expr.expr);
//    }
//}

impl<T> EvalExtend<Tensor<T>> for Vec<T> {
    #[inline]
    fn eval_extend(&mut self, value: Tensor<T>) {
        self.push(value.data);
    }
}

impl<I, T, A> EvalExtend<I> for Vec<T>
where
    I: Iterator<Item = A> + DenseExpr,
    T: Evaluate<A>,
{
    #[inline]
    fn eval_extend(&mut self, iter: I) {
        for i in iter {
            self.push(Evaluate::eval(i));
        }
    }
}

impl<I, T, A> EvalExtend<SparseExpr<I>> for Vec<T>
where
    A: Set,
    Tensor<A>: Default + Copy,
    I: Target + Iterator<Item = IndexedExpr<Tensor<A>>>,
    Vec<T>: EvalExtend<Tensor<A>>,
{
    #[inline]
    fn eval_extend(&mut self, iter: SparseExpr<I>) {
        let mut offset = 0;
        self.reserve(iter.target_size() * Tensor::default().len());
        for IndexedExpr { index, expr } in iter {
            while offset < index {
                self.eval_extend(Tensor::default());
                offset += 1;
            }
            self.eval_extend(expr);
            offset += 1;
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
            debug_assert_eq!((self.data.len() - orig_len) % self.chunk_size(), 0);
        }
    }
}

impl<S, E, N, F, A, O> EvalExtend<Reduce<E, F>> for UniChunked<S, N>
where
    E: Iterator<Item = A>,
    A: Iterator,
    O: AsMutTensor,
    F: BinOpAssign<Tensor<O::Inner>, A::Item>,
    Self: Set + EvalExtend<A> + for<'a> ViewMutIterator<'a, Item = &'a mut O>,
    N: Dimension,
{
    #[inline]
    fn eval_extend(&mut self, Reduce { mut expr, op }: Reduce<E, F>) {
        if let Some(row) = expr.next() {
            let start = self.len();
            self.eval_extend(row);
            while let Some(next) = expr.next() {
                for (output, input) in self.view_mut_iter().skip(start).zip(next) {
                    op.apply_assign(output.as_mut_tensor(), input);
                }
            }
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

impl<'a, I, J> DenseExpr for SparseValIter<'a, I, J> {}
impl<'a, I: Expression, J> Expression for SparseValIter<'a, I, J> {}
impl<'a, I: ExprSize, J> ExprSize for SparseValIter<'a, I, J> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.iter.expr_size()
    }
}
impl<'a, I: TotalExprSize, J> TotalExprSize for SparseValIter<'a, I, J> {
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
    T: Set + PartialEq + std::fmt::Debug,
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
    T: Scalar,
    I: Iterator<Item = A>,
    Tensor<T>: AddAssign<A>,
{
    #[inline]
    fn add_assign(&mut self, rhs: I) {
        for (l, r) in self.data.iter_mut().zip(rhs) {
            l.as_mut_tensor().add_assign(r);
        }
    }
}

/*
 * Optimized multiply adds:
 */

//impl<T, L, R, A, Out> AddAssign<CwiseBinExpr<L, Tensor<R>, Multiplication>> for Tensor<[T]>
//where
//    R: Copy,
//    T: std::any::Any,
//    Out: std::any::Any + AsSlice<T>,
//    L: Iterator<Item = A>,
//    Tensor<[T]>: for<'a> AddAssign<&'a Tensor<[T]>>,
//    A: std::ops::Mul<Tensor<R>, Output = Tensor<Out>>,
//{
//    #[inline]
//    fn add_assign(&mut self, mut rhs: CwiseBinExpr<L, Tensor<R>, Multiplication>) {
//        let mut out = &mut self.data; // &mut [T]
//        let n = std::mem::size_of::<Out>() / std::mem::size_of::<T>();
//        while let Some(next) = rhs.left.next() {
//            *out[0..n].as_mut_tensor() += (next * rhs.right).data.as_slice().as_tensor();
//            if let Some(next) = rhs.left.next() {
//                *out[n..2 * n].as_mut_tensor() += (next * rhs.right).data.as_slice().as_tensor();
//                out = &mut out[2 * n..];
//            } else {
//                break;
//            }
//        }
//    }
//}
//impl<T, L, R, A, Out> AddAssign<CwiseBinExpr<Tensor<L>, R, Multiplication>> for Tensor<[T]>
//where
//    L: Copy,
//    T: std::any::Any,
//    Out: std::any::Any + AsSlice<T>,
//    R: Iterator<Item = A>,
//    Tensor<[T]>: for<'a> AddAssign<&'a Tensor<[T]>>,
//    Tensor<L>: std::ops::Mul<A, Output = Tensor<Out>>,
//{
//    #[inline]
//    fn add_assign(&mut self, mut rhs: CwiseBinExpr<Tensor<L>, R, Multiplication>) {
//        let mut out = &mut self.data; // &mut [T]
//        let n = std::mem::size_of::<Out>() / std::mem::size_of::<T>();
//        while let Some(next) = rhs.right.next() {
//            *out[0..n].as_mut_tensor() += (rhs.left * next).data.as_slice().as_tensor();
//            if let Some(next) = rhs.right.next() {
//                *out[n..2 * n].as_mut_tensor() += (rhs.left * next).data.as_slice().as_tensor();
//                out = &mut out[2 * n..];
//            } else {
//                break;
//            }
//        }
//    }
//}
//
//impl<T, L, R, A, B, Out> AddAssign<CwiseBinExpr<L, R, Multiplication>> for Tensor<[T]>
//where
//    T: std::any::Any,
//    Out: std::any::Any + AsSlice<T>,
//    L: Iterator<Item = A>,
//    R: Iterator<Item = B>,
//    A: std::ops::Mul<B, Output = Tensor<Out>>,
//    Tensor<[T]>: for<'a> AddAssign<&'a Tensor<[T]>>,
//{
//    #[inline]
//    fn add_assign(&mut self, mut rhs: CwiseBinExpr<L, R, Multiplication>) {
//        let mut out = &mut self.data; // &mut [T]
//        let n = std::mem::size_of::<Out>() / std::mem::size_of::<T>();
//        loop {
//            if let Some(r) = rhs.right.next() {
//                if let Some(l) = rhs.left.next() {
//                    *out[0..n].as_mut_tensor() += (l * r).data.as_slice().as_tensor();
//                } else {
//                    break;
//                }
//            } else {
//                break;
//            }
//
//            if let Some(r) = rhs.right.next() {
//                if let Some(l) = rhs.left.next() {
//                    *out[n..2 * n].as_mut_tensor() += (l * r).data.as_slice().as_tensor();
//                    out = &mut out[2 * n..];
//                } else {
//                    break;
//                }
//            } else {
//                break;
//            }
//        }
//    }
//}

//impl<T, I> AddAssign<I> for Tensor<Vec<T>>
//where
//    I: Iterator,
//    Tensor<[T]>: AddAssign<I>,
//{
//    #[inline]
//    fn add_assign(&mut self, rhs: I) {
//        self.data.as_mut_slice().as_mut_tensor().add_assign(rhs);
//    }
//}

impl<E, T: Scalar, A> AddAssign<Reduce<E, Addition>> for Tensor<T>
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

impl<'a, T, I, Out: 'a> AddAssign<I> for SliceIterExprMut<'a, T>
where
    Self: Iterator<Item = &'a mut Out>,
    I: Iterator,
    Out: AddAssign<I::Item>,
{
    #[inline]
    fn add_assign(&mut self, rhs: I) {
        for (rhs, out) in rhs.zip(self) {
            out.add_assign(rhs);
        }
    }
}

macro_rules! impl_bin_op_assign {
    (impl<$($type_vars:tt),*> $op_trait:ident for $type:ty { $op_fn:ident }) => {
        impl<$($type_vars),*, R, ItemIn, ItemOut> $op_trait<R> for $type
        where
            $type: Iterator<Item = ItemOut>,
            R: Iterator<Item = ItemIn>,
            ItemOut: $op_trait<ItemIn>,
        {
            #[inline]
            fn $op_fn(&mut self, rhs: R) {
                for (rhs, mut out) in rhs.zip(self) {
                    out.$op_fn(rhs);
                }
            }
        }
    }
}

impl_bin_op_assign!(impl<S, N> AddAssign for UniChunkedIterExpr<S, N> { add_assign });
impl_bin_op_assign!(impl<'a, S> AddAssign for ChunkedIterExpr<'a, S> { add_assign });
impl_bin_op_assign!(impl<'a, S> AddAssign for SubsetIterExpr<'a, S> { add_assign });

impl_bin_op_assign!(impl<S, N> SubAssign for UniChunkedIterExpr<S, N> { sub_assign });
impl_bin_op_assign!(impl<'a, S> SubAssign for ChunkedIterExpr<'a, S> { sub_assign });
impl_bin_op_assign!(impl<'a, S> SubAssign for SubsetIterExpr<'a, S> { sub_assign });

//impl<T, I, A> SubAssign<I> for Tensor<[T]>
//where
//    T: Scalar,
//    I: Iterator<Item = A>,
//    Tensor<T>: SubAssign<A>,
//{
//    #[inline]
//    fn sub_assign(&mut self, rhs: I) {
//        for (l, r) in self.data.iter_mut().zip(rhs) {
//            l.as_mut_tensor().sub_assign(r);
//        }
//    }
//}

//impl<T, I> SubAssign<I> for Tensor<Vec<T>>
//where
//    I: Iterator,
//    Tensor<[T]>: SubAssign<I>,
//{
//    #[inline]
//    fn sub_assign(&mut self, rhs: I) {
//        self.data.as_mut_slice().as_mut_tensor().sub_assign(rhs);
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

impl<'a, T, I, Out: 'a> SubAssign<I> for SliceIterExprMut<'a, T>
where
    Self: Iterator<Item = &'a mut Out>,
    I: Iterator,
    Out: SubAssign<I::Item>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: I) {
        for (rhs, out) in rhs.zip(self) {
            out.sub_assign(rhs);
        }
    }
}
