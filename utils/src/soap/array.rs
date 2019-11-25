//!
//! Implementation of generic array manipulation.
//!

use super::*;

/// Wrapper around `typenum` types to prevent downstream trait implementations.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub struct U<N>(N);

impl<N: Unsigned> std::fmt::Debug for U<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("U{}", N::to_usize())).finish()
    }
}

impl<N: Default> Default for U<N> {
    fn default() -> Self {
        U(N::default())
    }
}

macro_rules! impl_array_for_typenum {
    ($nty:ident, $n:expr) => {
        pub type $nty = U<consts::$nty>;
        impl<T> Set for [T; $n] {
            type Elem = T;
            type Atom = T;
            #[inline]
            fn len(&self) -> usize {
                $n
            }
        }
        impl<T> AsSlice<T> for [T; $n] {
            #[inline]
            fn as_slice(&self) -> &[T] {
                &self[..]
            }
        }
        impl<'a, T: 'a> View<'a> for [T; $n] {
            type Type = &'a [T];
            #[inline]
            fn view(&'a self) -> Self::Type {
                self
            }
        }
        impl<'a, T: 'a> ViewMut<'a> for [T; $n] {
            type Type = &'a mut [T];
            #[inline]
            fn view_mut(&'a mut self) -> Self::Type {
                self
            }
        }
        impl<T: Dummy + Copy> Dummy for [T; $n] {
            #[inline]
            unsafe fn dummy() -> Self {
                [Dummy::dummy(); $n]
            }
        }

        impl<'a, T, N> GetIndex<'a, &'a [T; $n]> for StaticRange<N>
            where
                N: Unsigned + Array<T>,
                <N as Array<T>>::Array: 'a,
        {
            type Output = &'a N::Array;
            #[inline]
            fn get(self, set: &&'a [T; $n]) -> Option<Self::Output> {
                if self.end() <= set.len() {
                    let slice = *set;
                    Some(unsafe { &*(slice.as_ptr().add(self.start()) as *const N::Array) })
                } else {
                    None
                }
            }
        }

        impl<'a, T, N> IsolateIndex<&'a [T; $n]> for StaticRange<N>
        where
            N: Unsigned + Array<T>,
            <N as Array<T>>::Array: 'a,
        {
            type Output = &'a N::Array;
            #[inline]
            fn try_isolate(self, set: &'a [T; $n]) -> Option<Self::Output> {
                if self.end() <= set.len() {
                    Some(unsafe { &*(set.as_ptr().add(self.start()) as *const N::Array) })
                } else {
                    None
                }
            }
        }

        impl<T> Array<T> for consts::$nty {
            type Array = [T; $n];

            #[inline]
            fn iter_mut(array: &mut Self::Array) -> std::slice::IterMut<T> {
                array.iter_mut()
            }
            #[inline]
            fn iter(array: &Self::Array) -> std::slice::Iter<T> {
                array.iter()
            }
            #[inline]
            fn as_slice(array: &Self::Array) -> &[T] {
                array
            }
        }

        impl<'a, T, N> ReinterpretAsGrouped<N> for &'a [T; $n]
        where
            N: Unsigned + Array<T>,
            consts::$nty: PartialDiv<N>,
            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
            <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array: 'a,
        {
            type Output = &'a <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
            #[inline]
            fn reinterpret_as_grouped(self) -> Self::Output {
                assert_eq!(
                    $n / N::to_usize(),
                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
                );
                unsafe {
                    &*(self as *const [T; $n]
                        as *const <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array)
                }
            }
        }

        impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut [T; $n]
        where
            N: Unsigned + Array<T>,
            consts::$nty: PartialDiv<N>,
            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
            <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array: 'a,
        {
            type Output = &'a mut <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
            #[inline]
            fn reinterpret_as_grouped(self) -> Self::Output {
                assert_eq!(
                    $n / N::to_usize(),
                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
                );
                unsafe {
                    &mut *(self as *mut [T; $n]
                        as *mut <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array)
                }
            }
        }

        impl<T, N: Array<T>> UniChunkable<N> for [T; $n] {
            type Chunk = N::Array;
        }

        impl<'a, T, N: Array<T>> UniChunkable<N> for &'a [T; $n]
            where <N as Array<T>>::Array: 'a,
        {
            type Chunk = &'a N::Array;
        }

        impl<'a, T, N: Array<T>> UniChunkable<N> for &'a mut [T; $n]
            where <N as Array<T>>::Array: 'a,
        {
            type Chunk = &'a mut N::Array;
        }

        impl<T: Clone> CloneIntoOther<[T; $n]> for [T; $n] {
            #[inline]
            fn clone_into_other(&self, other: &mut [T; $n]) {
                other.clone_from(self);
            }
        }

        impl<T: Clone> CloneIntoOther<&mut [T; $n]> for [T; $n] {
            #[inline]
            fn clone_into_other(&self, other: &mut &mut [T; $n]) {
                (*other).clone_from(self);
            }
        }

        impl<T: Scalar, S> EvalExtend<Tensor<[T; $n]>> for UniChunked<S, $nty>
            where Self: Push<[T; $n]>
        {
            #[inline]
            fn eval_extend(&mut self, tensor: Tensor<[T; $n]>) {
                self.push(tensor.into_inner());
            }
        }

        impl<T, I> std::ops::Mul<Tensor<[T; $n]>> for UniChunkedIterExpr<I, $nty> {
            type Output = CwiseUnExpr<CwiseBinExpr<UniChunkedIterExpr<I, $nty>, Repeat<Tensor<[T; $n]>>, CwiseMultiplication>, Summation>;
            #[inline]
            fn mul(self, rhs: Tensor<[T; $n]>) -> Self::Output {
                CwiseUnExpr::new(CwiseBinExpr::new(self, Repeat::new(rhs)))
            }
        }
        //impl<T: Scalar> CwiseMulOp<UniChunkedIterExpr<I, N> for Tensor<[T; $n]> {
        //    type Output = Tensor<[T; $n]>;
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn cwise_mul(mut self, rhs: Self) -> Self::Output {
        //        for i in 0..$n {
        //            self[i] *= rhs[i];
        //        }
        //        self
        //    }
        //}

        impl<'a, T: 'a> AtomIterator<'a> for [T; $n] {
            type Item = &'a T;
            type Iter = std::slice::Iter<'a, T>;
            fn atom_iter(&'a self) -> Self::Iter {
                self.iter()
            }
        }

        impl<'a, T: 'a> AtomMutIterator<'a> for [T; $n] {
            type Item = &'a mut T;
            type Iter = std::slice::IterMut<'a, T>;
            fn atom_mut_iter(&'a mut self) -> Self::Iter {
                self.iter_mut()
            }
        }

        // TODO: Figure out how to compile the below code.
        //        impl<T, N> ReinterpretAsGrouped<N> for [T; $n]
        //        where
        //            N: Unsigned + Array<T>,
        //            consts::$nty: PartialDiv<N>,
        //            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
        //        {
        //            type Output = <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
        //            #[inline]
        //            fn reinterpret_as_grouped(self) -> Self::Output {
        //                assert_eq!(
        //                    $n / N::to_usize(),
        //                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
        //                );
        //                unsafe {
        //                    std::mem::transmute::<
        //                        Self,
        //                        <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array,
        //                    >(self)
        //                }
        //            }
        //        }
    };
}

impl_array_for_typenum!(U1, 1);
impl_array_for_typenum!(U2, 2);
impl_array_for_typenum!(U3, 3);
impl_array_for_typenum!(U4, 4);
impl_array_for_typenum!(U5, 5);
impl_array_for_typenum!(U6, 6);
impl_array_for_typenum!(U7, 7);
impl_array_for_typenum!(U8, 8);
impl_array_for_typenum!(U9, 9);
impl_array_for_typenum!(U10, 10);
impl_array_for_typenum!(U11, 11);
impl_array_for_typenum!(U12, 12);
impl_array_for_typenum!(U13, 13);
impl_array_for_typenum!(U14, 14);
impl_array_for_typenum!(U15, 15);
impl_array_for_typenum!(U16, 16);


