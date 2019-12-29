//!
//! This module implements arithmetic on vectors and matrices of statically sized Rust arrays.
//! The types defined in this module make arithmetic between vectors and matrices less verbose as
//! it would otherwise be if using raw Tensors.
//!
use super::*;
use num_traits::{Float, Zero};
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use unroll::unroll_for_loops;

/// The following array math library aims to be generic over the inner tensor. This means that
/// many methods should work for vectors as well as matrices or higher order tensors.
/// This approach complicates the code a bit but makes it shorter.
macro_rules! impl_array_vectors {
    ($vecn:ident, $rowvecn:ident; $n:expr) => {
        pub type $vecn<S> = Tensor<[Tensor<S>; $n]>;
        pub type $rowvecn<S> = Tensor<[Tensor<[Tensor<S>; $n]>; 1]>;

        impl<T> Tensor<[T; $n]> {
            pub fn new<D>(data: [D; $n]) -> Self
            where
                D: IntoTensor<Tensor = T>,
                T: IntoData<Data = D>,
            {
                data.into_tensor()
            }
        }

        impl<T: IntoData> Into<[T::Data; $n]> for Tensor<[T; $n]> {
            fn into(self) -> [T::Data; $n] {
                self.into_data()
            }
        }
        impl<T: IntoTensor> From<[T; $n]> for Tensor<[T::Tensor; $n]> {
            fn from(data: [T; $n]) -> Self {
                data.into_tensor()
            }
        }
        impl<'a, T: IntoTensor> From<&'a mut [T; $n]> for &'a mut Tensor<[T::Tensor; $n]> {
            fn from(data: &'a mut [T; $n]) -> Self {
                data.as_mut_tensor()
            }
        }
        impl<'a, T: IntoTensor> From<&'a [T; $n]> for &'a Tensor<[T::Tensor; $n]> {
            fn from(data: &'a [T; $n]) -> Self {
                data.as_tensor()
            }
        }

        impl<T: IntoTensor> IntoTensor for [T; $n] {
            type Tensor = Tensor<[T::Tensor; $n]>;
            #[inline]
            fn into_tensor(self) -> Self::Tensor {
                unsafe { Tensor::reinterpret(self) }
            }
        }
        impl<T: IntoData> IntoData for Tensor<[T; $n]> {
            type Data = [T::Data; $n];
            #[inline]
            fn into_data(self) -> Self::Data {
                debug_assert_eq!(
                    std::mem::size_of::<[T::Data; $n]>(),
                    std::mem::size_of::<[Tensor<T>; $n]>(),
                );
                unsafe { std::mem::transmute_copy(&self) }
            }
        }
        impl<T: IntoData> AsData for Tensor<[T; $n]> {
            type Data = [T::Data; $n];
            #[inline]
            fn as_data(&self) -> &Self::Data {
                unsafe { &*(self as *const Tensor<[T; $n]> as *const [T::Data; $n]) }
            }
        }
        impl<T: IntoTensor> AsTensor for [T; $n]
        where
            T::Tensor: Sized,
        {
            type Inner = [T::Tensor; $n];
            #[inline]
            fn as_tensor(&self) -> &Tensor<Self::Inner> {
                unsafe { Tensor::as_ref(self) }
            }
        }
        impl<T: IntoTensor> AsMutTensor for [T; $n]
        where
            T::Tensor: Sized,
        {
            #[inline]
            fn as_mut_tensor(&mut self) -> &mut Tensor<Self::Inner> {
                unsafe { Tensor::as_mut(self) }
            }
        }

        impl<S: Scalar> Tensor<[Tensor<S>; $n]> {
            #[inline]
            fn as_slice(&self) -> &[S] {
                unsafe { reinterpret::reinterpret_slice(&self.data[..]) }
            }
            #[inline]
            fn as_mut_slice(&mut self) -> &mut [S] {
                unsafe { reinterpret::reinterpret_mut_slice(&mut self.data[..]) }
            }
        }

        impl<T: Zero + Copy> Tensor<[T; $n]> {
            pub fn zeros() -> Tensor<[T; $n]> {
                Tensor {
                    data: [T::zero(); $n],
                }
            }
        }

        impl<T: Add<Output = T> + Copy + IntoData> Tensor<[T; $n]> {
            #[inline]
            pub fn sum(&self) -> T::Data {
                self.sum_op().into_data()
            }
        }

        impl<T: AddAssign + DotOp<Output = T> + Copy + IntoData> Tensor<[T; $n]> {
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn dot(self, other: Tensor<[T; $n]>) -> T::Data {
                self.dot_op(other).into_data()
            }
        }

        impl<T: DotOp + Copy> DotOp for Tensor<[T; $n]>
        where
            T::Output: AddAssign,
        {
            type Output = T::Output;
            #[inline]
            fn dot_op(self, rhs: Self) -> T::Output {
                let mut prod = self.data[0].dot_op(rhs.data[0]);
                for i in 1..$n {
                    prod += self.data[i].dot_op(rhs.data[i]);
                }
                prod
            }
        }

        impl<S, T: DotOp + Copy> Tensor<[T; $n]>
        where
            T::Output: AddAssign + IntoData<Data = S>,
        {
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn norm_squared(&self) -> S {
                let mut prod = self.data[0].dot_op(self.data[0]);
                for i in 1..$n {
                    prod += self.data[i].dot_op(self.data[i]);
                }
                prod.into_data()
            }
        }

        impl<S, T: DotOp + Copy> Tensor<[T; $n]>
        where
            S: Float,
            T::Output: AddAssign + IntoData<Data = S>,
        {
            #[inline]
            pub fn norm(&self) -> S {
                self.norm_squared().sqrt()
            }
        }

        impl<S, T: DotOp + Copy> Tensor<[T; $n]>
        where
            S: Scalar + Float,
            T: MulAssign<Tensor<S>>,
            T::Output: AddAssign + IntoData<Data = S>,
        {
            /// Normalize vector in place. Return its norm.
            #[inline]
            #[unroll_for_loops]
            pub fn normalize(&mut self) -> S {
                let norm = self.norm();
                if norm.is_zero() {
                    return norm;
                }
                let denom = S::one() / norm;
                for i in 0..$n {
                    unsafe {
                        *self.data.get_unchecked_mut(i) *= denom.into_tensor();
                    }
                }
                norm
            }

            /// Return a normalized vector.
            #[inline]
            pub fn normalized(mut self) -> Self {
                self.normalize();
                self
            }
        }

        impl<T: PartialOrd> Tensor<[T; $n]> {
            /// Computes the index of the vector component with the largest value.
            #[inline]
            #[allow(unused_mut, unused_assignments, unused_variables)]
            #[unroll_for_loops]
            pub fn imax(&self) -> usize {
                use std::ops::Index;
                let mut max_value = self.data.index(0);
                let mut max_index = 0;
                for index in 1..$n {
                    let value = unsafe { self.data.get_unchecked(index) };
                    if value > max_value {
                        max_value = value;
                        max_index = index;
                    }
                }

                max_index
            }

            /// Computes the index of the vector component with the smallest value.
            #[inline]
            #[allow(unused_mut, unused_assignments, unused_variables)]
            #[unroll_for_loops]
            pub fn imin(&self) -> usize {
                use std::ops::Index;
                let mut min_value = self.data.index(0);
                let mut min_index = 0;
                for index in 1..$n {
                    let value = unsafe { self.data.get_unchecked(index) };
                    if value < min_value {
                        min_value = value;
                        min_index = index;
                    }
                }

                min_index
            }
        }

        impl<T: PartialOrd + num_traits::Signed> Tensor<[T; $n]> {
            /// Computes the index of the vector component with the largest absolute value.
            #[inline]
            #[allow(unused_mut, unused_assignments, unused_variables)]
            #[unroll_for_loops]
            pub fn iamax(&self) -> usize {
                use std::ops::Index;
                let mut max_value = self.data.index(0).abs();
                let mut max_index = 0;
                for index in 1..$n {
                    let value = unsafe { self.data.get_unchecked(index).abs() };
                    if value > max_value {
                        max_value = value;
                        max_index = index;
                    }
                }

                max_index
            }
            /// Computes the index of the vector component with the smallest absolute value.
            #[inline]
            #[allow(unused_mut, unused_assignments, unused_variables)]
            #[unroll_for_loops]
            pub fn iamin(&self) -> usize {
                use std::ops::Index;
                let mut min_value = self.data.index(0).abs();
                let mut min_index = 0;
                for index in 1..$n {
                    let value = unsafe { self.data.get_unchecked(index).abs() };
                    if value < min_value {
                        min_value = value;
                        min_index = index;
                    }
                }

                min_index
            }
        }

        impl<T: Copy> Tensor<[Tensor<T>; $n]> {
            /// Apply a function to each data element of this tensor along the outer dimension.
            ///
            /// This is similar to `map` but the function is applied to the `data` portion of each
            /// element of this tensor.
            #[inline]
            pub fn mapd<U: Pod, F>(&self, mut f: F) -> Tensor<[Tensor<U>; $n]>
            where
                F: FnMut(T) -> U,
            {
                self.map(|x| Tensor { data: f(x.data) })
            }
        }

        impl<T: Copy> Tensor<[T; $n]> {
            /// Low level utility to zip two tensors along the outer dimension with a given function.
            #[inline]
            #[unroll_for_loops]
            pub fn zip_with<B, U, F>(&self, other: Tensor<[B; $n]>, mut f: F) -> Tensor<[U; $n]>
            where
                B: Copy,
                U: Pod,
                F: FnMut(T, B) -> U,
            {
                // We use MaybeUninit here mostly to avoid a Zero trait bound.
                let mut out: [MaybeUninit<U>; $n] = unsafe { MaybeUninit::uninit().assume_init() };
                for i in 0..$n {
                    out[i] = MaybeUninit::new(f(self.data[i], other.data[i]));
                }
                // The Pod trait bound ensures safety here in release builds.
                // Sanity check here just in debug builds only, since this code is very likely in a
                // critical section.
                debug_assert_eq!(
                    std::mem::size_of::<[MaybeUninit<U>; $n]>(),
                    std::mem::size_of::<Tensor<[U; $n]>>()
                );
                unsafe { std::mem::transmute_copy::<_, Tensor<[U; $n]>>(&out) }
            }

            /// Apply a function to each element of the tensor along the outer dimension.
            #[inline]
            #[unroll_for_loops]
            pub fn map<U: Pod, F>(&self, mut f: F) -> Tensor<[U; $n]>
            where
                F: FnMut(T) -> U,
            {
                // We use MaybeUninit here mostly to avoid a Zero trait bound.
                let mut out: [MaybeUninit<U>; $n] = unsafe { MaybeUninit::uninit().assume_init() };
                for i in 0..$n {
                    out[i] = MaybeUninit::new(f(self.data[i]));
                }
                // The Pod trait bound ensures safety here in release builds.
                // Sanity check here just in debug builds only, since this code is very likely in a
                // critical section.
                debug_assert_eq!(
                    std::mem::size_of::<[MaybeUninit<U>; $n]>(),
                    std::mem::size_of::<Tensor<[U; $n]>>()
                );
                unsafe { std::mem::transmute_copy::<_, Tensor<[U; $n]>>(&out) }
            }

            /// Fold this tensor along the outer dimension.
            #[inline]
            #[unroll_for_loops]
            pub fn fold<B, F>(&self, mut init: B, mut f: F) -> B
            where
                F: FnMut(B, T) -> B,
            {
                for i in 0..$n {
                    init = f(init, self.data[i])
                }
                init
            }
            /// Fold this tensor along the outer dimension with an initializer that consumes the
            /// first element.
            #[inline]
            #[allow(unused_mut, unused_variables)]
            #[unroll_for_loops]
            pub fn fold1<B, F, G>(&self, mut first: G, mut f: F) -> B
            where
                G: FnOnce(T) -> B,
                F: FnMut(B, T) -> B,
            {
                let mut acc = first(self.data[0]);
                for i in 1..$n {
                    acc = f(acc, self.data[i])
                }
                acc
            }
        }

        impl<S: Scalar> Matrix for Tensor<[Tensor<S>; $n]> {
            type Transpose = $rowvecn<S>;
            #[inline]
            fn transpose(self) -> Self::Transpose {
                Tensor { data: [self; 1] }
            }
            #[inline]
            fn num_rows(&self) -> usize {
                $n
            }
            #[inline]
            fn num_cols(&self) -> usize {
                1
            }
        }

        impl<T: Copy + CwiseMulAssignOp<T>> CwiseMulAssignOp<Tensor<[Tensor<T>; $n]>>
            for Tensor<[T]>
        {
            #[inline]
            //#[unroll_for_loops]
            fn cwise_mul_assign(&mut self, rhs: Tensor<[Tensor<T>; $n]>) {
                debug_assert!(self.len() >= rhs.len());
                for i in 0..$n {
                    unsafe {
                        self.data
                            .get_unchecked_mut(i)
                            .cwise_mul_assign(rhs.data.get_unchecked(i).data)
                    };
                }
            }
        }

        impl<T: Copy + AddAssign<T>> AddAssign<Tensor<[Tensor<T>; $n]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[Tensor<T>; $n]>) {
                debug_assert!(self.len() >= rhs.len());
                for i in 0..$n {
                    unsafe { *self.data.get_unchecked_mut(i) += rhs.data.get_unchecked(i).data };
                }
            }
        }

        impl<T: Copy + SubAssign<T>> SubAssign<Tensor<[Tensor<T>; $n]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[Tensor<T>; $n]>) {
                debug_assert!(self.len() >= rhs.len());
                for i in 0..$n {
                    unsafe { *self.data.get_unchecked_mut(i) -= rhs.data.get_unchecked(i).data };
                }
            }
        }

        impl<T: MulAssign<R::Tensor>, R: Copy + IntoTensor> CwiseMulOp<R> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn cwise_mul(self, rhs: R) -> Self::Output {
                self * rhs
            }
        }

        // Scalar multiply
        impl<S: Scalar, T: MulAssign<Tensor<S>>> CwiseMulOp<Tensor<[T; $n]>> for Tensor<S> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn cwise_mul(self, rhs: Tensor<[T; $n]>) -> Self::Output {
                self * rhs
            }
        }

        impl<T: MulAssign + Copy> CwiseMulOp for Tensor<[T; $n]> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            #[unroll_for_loops]
            fn cwise_mul(mut self, rhs: Self) -> Self::Output {
                for i in 0..$n {
                    self.data[i] *= rhs.data[i];
                }
                self
            }
        }

        impl<T: Copy + RecursiveSumOp<Output = O>, O> RecursiveSumOp for Tensor<[T; $n]>
        where
            O: Zero + Add<Output = O>,
        {
            type Output = O;
            #[inline]
            fn recursive_sum(self) -> Self::Output {
                self.fold(O::zero(), |acc, x| acc + x.recursive_sum())
            }
        }
        impl<T: Copy + Add<Output = T>> SumOp for Tensor<[T; $n]> {
            type Output = T;
            #[inline]
            fn sum_op(self) -> Self::Output {
                self.fold1(|x| x, |acc, x| acc + x)
            }
        }

        impl<T: MulAssign<Tensor<R>>, R: Scalar> DotOp<Tensor<R>> for Tensor<[T; $n]> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<R>) -> Self::Output {
                rhs * self
            }
        }

        impl<S: Scalar, T: MulAssign<Tensor<S>>> DotOp<Tensor<[T; $n]>> for Tensor<S> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<[T; $n]>) -> Self::Output {
                self * rhs
            }
        }

        impl<T: AddAssign + Copy> Add for Tensor<[T; $n]> {
            type Output = Self;

            /// Add two tensor arrays together.
            #[inline]
            fn add(mut self, rhs: Self) -> Self::Output {
                self += rhs;
                self
            }
        }

        impl<T: AddAssign + Copy> AddAssign<Tensor<[T; $n]>> for Tensor<&mut [T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }

        impl<T: AddAssign + Copy> AddAssign<Tensor<[T; $n]>> for &mut Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: AddAssign + Copy> AddAssign for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: AddAssign + Copy> AddAssign<&Tensor<[T; $n]>> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: &Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: SubAssign + Copy> Sub for Tensor<[T; $n]> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self::Output {
                self -= rhs;
                self
            }
        }

        impl<T: SubAssign + Copy> SubAssign<Tensor<[T; $n]>> for Tensor<&mut [T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        impl<T: SubAssign + Copy> SubAssign<Tensor<[T; $n]>> for &mut Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        impl<T: SubAssign + Copy> SubAssign for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        impl<T: SubAssign + Copy> SubAssign<&Tensor<[T; $n]>> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: &Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        /*
         * Scalar ops
         */

        // Right multiply by a scalar.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T: MulAssign<R::Tensor>, R: Copy + IntoTensor> Mul<R> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn mul(mut self, rhs: R) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T: MulAssign<R::Tensor>, R: Copy + IntoTensor> MulAssign<R> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn mul_assign(&mut self, rhs: R) {
                for i in 0..$n {
                    self.data[i] *= rhs.into_tensor();
                }
            }
        }

        impl<T: DivAssign<R::Tensor>, R: Copy + IntoTensor> Div<R> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn div(mut self, rhs: R) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl<T: DivAssign<R::Tensor>, R: Copy + IntoTensor> DivAssign<R> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn div_assign(&mut self, rhs: R) {
                for i in 0..$n {
                    self.data[i] /= rhs.into_tensor();
                }
            }
        }

        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        impl<S: Scalar, T: MulAssign<Tensor<S>>> Mul<Tensor<[T; $n]>> for Tensor<S> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn mul(self, mut rhs: Tensor<[T; $n]>) -> Self::Output {
                for i in 0..$n {
                    rhs.data[i] *= self;
                }
                rhs
            }
        }

        /*
         * Index ops
         */

        impl<S: Scalar, I: std::slice::SliceIndex<[S]>> Index<I> for Tensor<[Tensor<S>; $n]> {
            type Output = I::Output;
            #[inline]
            fn index(&self, index: I) -> &Self::Output {
                &self.as_slice()[index]
            }
        }
        impl<S: Scalar, I: std::slice::SliceIndex<[S]>> IndexMut<I> for Tensor<[Tensor<S>; $n]> {
            #[inline]
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                &mut self.as_mut_slice()[index]
            }
        }

        impl<T: Zero + Copy + AddAssign + PartialEq> Zero for Tensor<[T; $n]> {
            #[inline]
            fn zero() -> Self {
                Tensor {
                    data: [Zero::zero(); $n],
                }
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        //impl<T> Into<[T; $n]> for Tensor<[T; $n]> {
        //    #[inline]
        //    fn into(self) -> [T; $n] {
        //        self.into_inner()
        //    }
        //}
        //impl<T> Into<[T; $n]> for Tensor<[[T; $n]; 1]> {
        //    #[inline]
        //    fn into(self) -> [T; $n] {
        //        let [x] = self.into_inner();
        //        x
        //    }
        //}
        impl<T: Copy + num_traits::ToPrimitive> Tensor<[Tensor<T>; $n]> {
            /// Casts the components of the vector into another type.
            ///
            /// # Panics
            /// This function panics if the cast fails.
            #[inline]
            pub fn cast<U: Pod + num_traits::NumCast>(&self) -> Tensor<[Tensor<U>; $n]> {
                self.mapd(|x| U::from(x).unwrap())
            }
        }

        impl<T: Pod> Tensor<[Tensor<[Tensor<T>; $n]>; $n]> {
            /// Convert this square matrix into a vector of its lower triangular entries.
            /// The entries appear in row-major order as usual.
            #[inline]
            #[unroll_for_loops]
            pub fn lower_triangular_vec(&self) -> Tensor<[Tensor<T>; ($n * ($n - 1)) / 2 + $n]> {
                const LEN: usize = ($n * ($n - 1)) / 2 + $n;
                let mut v: [MaybeUninit<T>; LEN] = unsafe { MaybeUninit::uninit().assume_init() };

                let mut i = 0;
                for row in 0..$n {
                    for col in 0..=row {
                        v[i] = MaybeUninit::new(self.data[row].data[col].data);
                        i += 1;
                    }
                }

                // Sanity check required because we can't use transmute on generic types.
                debug_assert_eq!(
                    std::mem::size_of::<[MaybeUninit<T>; LEN]>(),
                    std::mem::size_of::<Tensor<[Tensor<T>; LEN]>>()
                );

                unsafe { std::mem::transmute_copy::<_, Tensor<[Tensor<T>; LEN]>>(&v) }
            }
        }

        #[cfg(feature = "approx")]
        impl<U, T: approx::AbsDiffEq<U>> approx::AbsDiffEq<Tensor<[U; $n]>> for Tensor<[T; $n]>
        where
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;
            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }
            #[inline]
            #[unroll_for_loops]
            fn abs_diff_eq(&self, other: &Tensor<[U; $n]>, epsilon: Self::Epsilon) -> bool {
                for i in 0..$n {
                    if !self.data[i].abs_diff_eq(&other.data[i], epsilon) {
                        return false;
                    }
                }
                true
            }
        }
        #[cfg(feature = "approx")]
        impl<U, T: approx::RelativeEq<U>> approx::RelativeEq<Tensor<[U; $n]>> for Tensor<[T; $n]>
        where
            T::Epsilon: Copy,
        {
            #[inline]
            fn default_max_relative() -> Self::Epsilon {
                T::default_max_relative()
            }
            #[inline]
            #[unroll_for_loops]
            fn relative_eq(
                &self,
                other: &Tensor<[U; $n]>,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                for i in 0..$n {
                    if !self.data[i].relative_eq(&other.data[i], epsilon, max_relative) {
                        return false;
                    }
                }
                true
            }
        }
        #[cfg(feature = "approx")]
        impl<U, T: approx::UlpsEq<U>> approx::UlpsEq<Tensor<[U; $n]>> for Tensor<[T; $n]>
        where
            T::Epsilon: Copy,
        {
            #[inline]
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }
            #[inline]
            #[unroll_for_loops]
            fn ulps_eq(
                &self,
                other: &Tensor<[U; $n]>,
                epsilon: Self::Epsilon,
                max_ulps: u32,
            ) -> bool {
                for i in 0..$n {
                    if !self.data[i].ulps_eq(&other.data[i], epsilon, max_ulps) {
                        return false;
                    }
                }
                true
            }
        }
    };
}

macro_rules! impl_square_reshape {
    ($n:expr, $r:expr) => {
        impl<T: Copy> Tensor<[T; $n]> {
            /// Construct a square matrix from this vector. This is a specialized version of
            /// reshape.
            #[inline]
            pub fn mtx(&self) -> Tensor<[Tensor<[T; $r]>; $r]> {
                unsafe {
                    debug_assert_eq!(
                        std::mem::size_of::<Tensor<[T; $n]>>(),
                        std::mem::size_of::<Tensor<[Tensor<[T; $r]>; $r]>>()
                    );
                    std::mem::transmute_copy(self)
                }
            }
        }
    };
}

impl_square_reshape!(9, 3);
impl_square_reshape!(4, 2);
impl_square_reshape!(16, 4);

impl_array_vectors!(Vector1, RowVector1; 1);
impl_array_vectors!(Vector2, RowVector2; 2);
impl_array_vectors!(Vector3, RowVector3; 3);
impl_array_vectors!(Vector4, RowVector4; 4);
impl_array_vectors!(Vector5, RowVector5; 5);
impl_array_vectors!(Vector6, RowVector6; 6);
impl_array_vectors!(Vector7, RowVector7; 7);
impl_array_vectors!(Vector8, RowVector8; 8);
impl_array_vectors!(Vector9, RowVector9; 9);
impl_array_vectors!(Vector10, RowVector10; 10);
impl_array_vectors!(Vector11, RowVector11; 11);
impl_array_vectors!(Vector12, RowVector12; 12);
impl_array_vectors!(Vector13, RowVector13; 13);
impl_array_vectors!(Vector14, RowVector14; 14);
impl_array_vectors!(Vector15, RowVector15; 15);
impl_array_vectors!(Vector16, RowVector16; 16);

impl<T: Scalar> Vector3<T> {
    #[inline]
    pub fn cross(self, other: Vector3<T>) -> Vector3<T> {
        [
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ]
        .into_tensor()
    }
}

macro_rules! impl_array_matrices {
    ($mtxn:ident; $r:expr, $c:expr) => {
        // Row-major square matrix.
        pub type $mtxn<T> = Tensor<[Tensor<[Tensor<T>; $c]>; $r]>;

        impl<T: Scalar> Tensor<[Tensor<[Tensor<T>; $c]>; $r]> {
            #[inline]
            pub fn from_rows(rows: [Tensor<[Tensor<T>; $c]>; $r]) -> Self {
                Tensor { data: rows }
            }
            #[inline]
            pub fn from_cols(rows: [Tensor<[Tensor<T>; $r]>; $c]) -> Self {
                Tensor { data: rows }.transpose()
            }
        }

        impl<T> AsSlice<T> for [[T; $c]; $r] {
            #[inline]
            fn as_slice(&self) -> &[T] {
                unsafe { reinterpret::reinterpret_slice(&self[..]) }
            }
        }

        // Transposes of small matrices are implemented eagerly.
        impl<T: Scalar> Matrix for Tensor<[Tensor<[Tensor<T>; $c]>; $r]> {
            type Transpose = Tensor<[Tensor<[Tensor<T>; $r]>; $c]>;
            #[inline]
            #[unroll_for_loops]
            fn transpose(self) -> Self::Transpose {
                let mut m: [[MaybeUninit<T>; $r]; $c] =
                    unsafe { MaybeUninit::uninit().assume_init() };

                for col in 0..$c {
                    for row in 0..$r {
                        m[col][row] = MaybeUninit::new(self.data[row][col]);
                    }
                }
                // Sanity check required because we can't use transmute on generic types.
                debug_assert_eq!(
                    std::mem::size_of::<[[MaybeUninit<T>; $r]; $c]>(),
                    std::mem::size_of::<[[T; $r]; $c]>()
                );
                unsafe { std::mem::transmute_copy::<_, [[T; $r]; $c]>(&m) }.into_tensor()
            }
            #[inline]
            fn num_rows(&self) -> usize {
                $r
            }
            #[inline]
            fn num_cols(&self) -> usize {
                $c
            }
        }

        impl<T: Copy> Tensor<[Tensor<[Tensor<T>; $c]>; $r]> {
            /// Similar to `mapd` but applies the given function to each inner element.
            #[inline]
            pub fn mapd_inner<U, F>(&self, mut f: F) -> Tensor<[Tensor<[Tensor<U>; $c]>; $r]>
            where
                U: Pod,
                F: FnMut(T) -> U,
            {
                self.map_inner(|x| Tensor { data: f(x.data) })
            }
        }

        impl<T: Copy> Tensor<[Tensor<[T; $c]>; $r]> {
            /// Similar to `map` but applies the given function to each inner element.
            #[inline]
            #[unroll_for_loops]
            pub fn map_inner<U, F>(&self, mut f: F) -> Tensor<[Tensor<[U; $c]>; $r]>
            where
                U: Pod,
                F: FnMut(T) -> U,
            {
                // We use MaybeUninit here mostly to avoid a Zero trait bound.
                let mut out: [[MaybeUninit<U>; $c]; $r] =
                    unsafe { MaybeUninit::uninit().assume_init() };
                for row in 0..$r {
                    for col in 0..$c {
                        out[row][col] = MaybeUninit::new(f(self.data[row].data[col]));
                    }
                }
                // The Pod trait bound ensures safety here in release builds.
                // Sanity check here just in debug builds only, since this code is very likely in a
                // critical section.
                debug_assert_eq!(
                    std::mem::size_of::<[[MaybeUninit<U>; $c]; $r]>(),
                    std::mem::size_of::<Tensor<[Tensor<[U; $c]>; $r]>>()
                );
                unsafe { std::mem::transmute_copy::<_, Tensor<[Tensor<[U; $c]>; $r]>>(&out) }
            }
            #[inline]
            pub fn vec(&self) -> Tensor<[T; $c * $r]> {
                unsafe {
                    debug_assert_eq!(
                        std::mem::size_of::<[T; $c * $r]>(),
                        std::mem::size_of::<[[T; $c]; $r]>()
                    );
                    std::mem::transmute_copy(self)
                }
            }
        }

        impl<S: Scalar> Tensor<[Tensor<[Tensor<S>; $c]>; $r]> {
            #[inline]
            pub fn identity() -> Self {
                Self::from_diag_iter(std::iter::repeat(S::one()))
            }
            #[inline]
            pub fn diag(diag: &[S]) -> Self {
                Self::from_diag_iter(diag.into_iter().cloned())
            }
            pub fn from_diag_iter<Iter: IntoIterator<Item = S>>(diag: Iter) -> Self {
                let mut out = Self::zeros();
                for (i, elem) in diag.into_iter().take($r.min($c)).enumerate() {
                    out[i][i] = elem;
                }
                out
            }

            #[inline]
            #[unroll_for_loops]
            pub fn fold_inner<B, F>(&self, mut init: B, mut f: F) -> B
            where
                F: FnMut(B, S) -> B,
            {
                for i in 0..$r {
                    init = self[i].fold(init, |acc, x| f(acc, x.data));
                }
                init
            }

            /// Compute the sum of all entries in this matrix.
            #[inline]
            pub fn sum_inner(&self) -> S {
                self.fold_inner(S::zero(), |acc, x| acc + x)
            }

            // TODO: optimize this function
            #[inline]
            pub fn trace(&self) -> S {
                let mut tr = self[0][0];
                for i in 1..$r.min($c) {
                    tr += self[i][i];
                }
                tr
            }
            #[inline]
            pub fn frob_norm_squared(&self) -> S {
                (*self).map_inner(|x| x * x).sum_inner()
            }
        }

        impl<S: Float + Scalar> Tensor<[Tensor<[Tensor<S>; $c]>; $r]> {
            #[inline]
            pub fn frob_norm(&self) -> S {
                self.frob_norm_squared().sqrt()
            }
        }

        //impl<T: Scalar> Add for Tensor<[Tensor<[T; $c]>; $r]> {
        //    type Output = Self;

        //    /// Add two tensor arrays together.
        //    #[inline]
        //    fn add(mut self, rhs: Self) -> Self::Output {
        //        self += rhs;
        //        self
        //    }
        //}

        //impl<T: Copy + AddAssign> AddAssign<Tensor<[Tensor<[T; $c]>; $r]>> for Tensor<[T]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn add_assign(&mut self, rhs: Tensor<[Tensor<[T; $c]>; $r]>) {
        //        for i in 0..$r {
        //            for j in 0..$c {
        //                self.data[$r * i + j] += rhs.data[i][j];
        //            }
        //        }
        //    }
        //}

        //impl<T: Scalar> AddAssign<Tensor<[Tensor<[T; $c]>; $r]>> for &mut Tensor<[Tensor<[T; $c]>; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn add_assign(&mut self, rhs: Tensor<[Tensor<[T; $c]>; $r]>) {
        //        for i in 0..$r {
        //            *self.data[i].as_mut_tensor() += *rhs.data[i].as_tensor();
        //        }
        //    }
        //}
        //impl<T: Scalar> AddAssign for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn add_assign(&mut self, rhs: Self) {
        //        for i in 0..$r {
        //            *self.data[i].as_mut_tensor() += *rhs.data[i].as_tensor();
        //        }
        //    }
        //}
        //impl<T: Scalar> AddAssign<&Tensor<[[T; $c]; $r]>> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn add_assign(&mut self, rhs: &Tensor<[[T; $c]; $r]>) {
        //        for i in 0..$r {
        //            *self.data[i].as_mut_tensor() += rhs.data[i].as_tensor();
        //        }
        //    }
        //}
        //impl<T: Scalar> Sub for Tensor<[[T; $c]; $r]> {
        //    type Output = Self;

        //    #[inline]
        //    fn sub(mut self, rhs: Self) -> Self::Output {
        //        self -= rhs;
        //        self
        //    }
        //}

        //impl<T: Copy + SubAssign> SubAssign<Tensor<[[T; $c]; $r]>> for Tensor<[T]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn sub_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
        //        for i in 0..$r {
        //            for j in 0..$c {
        //                self.data[$r * i + j] -= rhs.data[i][j];
        //            }
        //        }
        //    }
        //}

        //impl<T: Scalar> SubAssign<Tensor<[[T; $c]; $r]>> for &mut Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn sub_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
        //        for i in 0..$r {
        //            *self.data[i].as_mut_tensor() -= rhs.data[i].as_tensor();
        //        }
        //    }
        //}
        //impl<T: Scalar> SubAssign for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn sub_assign(&mut self, rhs: Self) {
        //        for i in 0..$r {
        //            *self.data[i].as_mut_tensor() -= rhs.data[i].as_tensor();
        //        }
        //    }
        //}

        //impl<T: Scalar> SubAssign<&Tensor<[[T; $c]; $r]>> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn sub_assign(&mut self, rhs: &Tensor<[[T; $c]; $r]>) {
        //        for i in 0..$r {
        //            *self.data[i].as_mut_tensor() -= rhs.data[i].as_tensor();
        //        }
        //    }
        //}

        /*
         * Matrix-vector multiply
         */
        impl<T: Pod> Mul<Tensor<[T; $c]>> for Tensor<[Tensor<[T; $c]>; $r]>
        where
            Tensor<[T; $c]>: DotOp<Output = T>,
        {
            type Output = Tensor<[T; $r]>;
            #[inline]
            fn mul(self, rhs: Tensor<[T; $c]>) -> Self::Output {
                self.map(|row| row.dot_op(rhs))
            }
        }

        /*
         * Vector-RowVector multiply
         * This sepecial case treats Vectors as column vectors
         */
        impl<T, U> Mul<Tensor<[Tensor<[T; $c]>; 1]>> for Tensor<[T; $r]>
        where
            Tensor<[Tensor<[T; 1]>; $r]>: Mul<Tensor<[Tensor<[T; $c]>; 1]>, Output = U>,
        {
            type Output = U;
            #[inline]
            fn mul(self, rhs: Tensor<[Tensor<[T; $c]>; 1]>) -> Self::Output {
                let lhs: Tensor<[Tensor<[T; 1]>; $r]> = unsafe { Tensor::reinterpret(self) };
                lhs * rhs
            }
        }

        // Right scalar multiply by a raw scalar.
        //impl<T: Scalar> Mul<T> for Tensor<[[T; $c]; $r]> {
        //    type Output = Self;
        //    #[inline]
        //    fn mul(mut self, rhs: T) -> Self::Output {
        //        self *= rhs;
        //        self
        //    }
        //}

        //// Scalar multiply assign by a raw scalar.
        //impl<T: Scalar> MulAssign<T> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    fn mul_assign(&mut self, rhs: T) {
        //        *self *= Tensor::new(rhs);
        //    }
        //}

        //// Scalar divide by a raw scalar.
        //impl<T: Scalar> Div<T> for Tensor<[[T; $c]; $r]> {
        //    type Output = Self;
        //    #[inline]
        //    fn div(mut self, rhs: T) -> Self::Output {
        //        self /= rhs;
        //        self
        //    }
        //}

        //// Divide assign by a raw scalar.
        //impl<T: Scalar> DivAssign<T> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    fn div_assign(&mut self, rhs: T) {
        //        *self /= Tensor::new(rhs);
        //    }
        //}

        // Right multiply by a tensor scalar.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        //impl<T: Scalar> Mul<Tensor<T>> for Tensor<[[T; $c]; $r]> {
        //    type Output = Self;
        //    #[inline]
        //    fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
        //        self *= rhs;
        //        self
        //    }
        //}

        // Divide by a tensor scalar.
        //impl<T: Scalar> Div<Tensor<T>> for Tensor<[[T; $c]; $r]> {
        //    type Output = Self;
        //    #[inline]
        //    fn div(mut self, rhs: Tensor<T>) -> Self::Output {
        //        self /= rhs;
        //        self
        //    }
        //}

        // Multiply assign by a tensor scalar
        //impl<T: Scalar> MulAssign<Tensor<T>> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn mul_assign(&mut self, rhs: Tensor<T>) {
        //        use std::ops::IndexMut;
        //        for i in 0..$r {
        //            *self.data.index_mut(i).as_mut_tensor() *= rhs.clone();
        //        }
        //    }
        //}

        //// Divide assign by a tensor scalar
        //impl<T: Scalar> DivAssign<Tensor<T>> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn div_assign(&mut self, rhs: Tensor<T>) {
        //        use std::ops::IndexMut;
        //        for i in 0..$r {
        //            *self.data.index_mut(i).as_mut_tensor() /= rhs.clone();
        //        }
        //    }
        //}

        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        //impl<T: Scalar> Mul<Tensor<[[T; $c]; $r]>> for Tensor<T> {
        //    type Output = Tensor<[[T; $c]; $r]>;
        //    #[inline]
        //    fn mul(self, mut rhs: Tensor<[[T; $c]; $r]>) -> Self::Output {
        //        rhs *= self;
        //        rhs
        //    }
        //}
        //impl<T: Scalar> Zero for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    fn zero() -> Self {
        //        Tensor::new([[Zero::zero(); $c]; $r])
        //    }
        //    #[inline]
        //    fn is_zero(&self) -> bool {
        //        *self == Self::zero()
        //    }
        //}

        impl<T, I: std::slice::SliceIndex<[Tensor<[T; $c]>]>> Index<I>
            for Tensor<[Tensor<[T; $c]>; $r]>
        {
            type Output = I::Output;
            #[inline]
            fn index(&self, index: I) -> &Self::Output {
                &self.data[index]
            }
        }
        impl<T, I: std::slice::SliceIndex<[Tensor<[T; $c]>]>> IndexMut<I>
            for Tensor<[Tensor<[T; $c]>; $r]>
        {
            #[inline]
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                &mut self.data[index]
            }
        }
        impl<T: Copy + num_traits::ToPrimitive> Tensor<[Tensor<[Tensor<T>; $c]>; $r]> {
            /// Casts the components of the matrix into another type.
            ///
            /// # Panics
            /// This function panics if the cast fails.
            #[inline]
            pub fn cast_inner<U: Pod + num_traits::NumCast>(
                &self,
            ) -> Tensor<[Tensor<[Tensor<U>; $c]>; $r]> {
                self.map_inner(|x| Tensor {
                    data: U::from(x.data).unwrap(),
                })
            }
        }

        //impl<T: Scalar> CwiseMulAssignOp<Tensor<[[T; $c]; $r]>> for Tensor<[[T; $c]; $r]> {
        //    #[inline]
        //    #[unroll_for_loops]
        //    fn cwise_mul_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
        //        for r in 0..$r {
        //            for c in 0..$c {
        //                self[r][c] *= rhs[r][c];
        //            }
        //        }
        //    }
        //}

        //impl<T: Scalar> CwiseMulOp<Tensor<[[T; $c]; $r]>> for Tensor<[[T; $c]; $r]> {
        //    type Output = Tensor<[[T; $c]; $r]>;
        //    #[inline]
        //    fn cwise_mul(mut self, rhs: Tensor<[[T; $c]; $r]>) -> Self::Output {
        //        self.cwise_mul_assign(rhs);
        //        self
        //    }
        //}

        //impl<T: Scalar> DotOp<Tensor<T>> for Tensor<[Tensor<[T; $c]>; $r]> {
        //    type Output = Tensor<[Tensor<[T; $c]>; $r]>;
        //    #[inline]
        //    fn dot_op(self, rhs: Tensor<T>) -> Self::Output {
        //        self * rhs
        //    }
        //}

        //impl<T: Scalar> DotOp<Tensor<[Tensor<[T; $c]>; $r]>> for Tensor<T> {
        //    type Output = Tensor<[Tensor<[T; $c]>; $r]>;
        //    #[inline]
        //    fn dot_op(self, rhs: Tensor<[Tensor<[T; $c]>; $r]>) -> Self::Output {
        //        rhs * self
        //    }
        //}

        //// Dot operator as compared to multiplication is always commutative. This means that no
        //// matter the order, it will always chose to contract along the outer dimension.
        //impl<S: Scalar> DotOp<Tensor<[Tensor<S>; $r]>> for Tensor<[Tensor<[Tensor<S>; $c]>; $r]> {
        //    type Output = Tensor<[Tensor<S>; $c]>;
        //    #[inline]
        //    fn dot_op(self, rhs: Tensor<[Tensor<S>; $r]>) -> Self::Output {
        //        rhs.transpose() * self
        //    }
        //}

        //impl<S: Scalar> DotOp<Tensor<[Tensor<[Tensor<S>; $c]>; $r]>> for Tensor<[Tensor<S>; $r]> {
        //    type Output = Tensor<[Tensor<S>; $c]>;
        //    #[inline]
        //    fn dot_op(self, rhs: Tensor<[Tensor<[Tensor<S>; $c]>; $r]>) -> Self::Output {
        //        self.transpose() * rhs
        //    }
        //}

        //impl<T: Scalar> DotOp<Tensor<[Tensor<[T; $c]>; $r]>> for Tensor<[Tensor<[T; $c]>; $r]> {
        //    type Output = Tensor<T>;
        //    #[inline]
        //    fn dot_op(self, rhs: Tensor<[Tensor<[T; $c]>; $r]>) -> Self::Output {
        //        Tensor { data: self.cwise_mul(rhs).sum_inner() }
        //    }
        //}

        //impl<T: Scalar> RecursiveSumOp for Tensor<[Tensor<[T; $c]>; $r]> {
        //    type Output = Tensor<T>;
        //    #[inline]
        //    fn recursive_sum(self) -> Self::Output {
        //        Tensor { data: self.sum_inner() }
        //    }
        //}
        //impl<T: Scalar> SumOp for Tensor<[Tensor<[T; $c]>; $r]> {
        //    type Output = Tensor<[T; $c]>;
        //    #[inline]
        //    #[allow(unused_mut)]
        //    #[unroll_for_loops]
        //    fn sum_op(self) -> Self::Output {
        //        let mut sum = self[0];
        //        for i in 1..$r {
        //            sum += self[i];
        //        }
        //        sum
        //    }
        //}
    };
}

impl_array_matrices!(Matrix1; 1, 1);
impl_array_matrices!(Matrix2; 2, 2);
impl_array_matrices!(Matrix3; 3, 3);
impl_array_matrices!(Matrix4; 4, 4);
impl_array_matrices!(Matrix9; 9, 9);

// Common Rectangular matrices
impl_array_matrices!(Matrix2x1; 2, 1);
impl_array_matrices!(Matrix3x1; 3, 1);
impl_array_matrices!(Matrix4x1; 4, 1);
impl_array_matrices!(Matrix1x2; 1, 2);
impl_array_matrices!(Matrix1x3; 1, 3);
impl_array_matrices!(Matrix1x4; 1, 4);
impl_array_matrices!(Matrix3x4; 3, 4);
impl_array_matrices!(Matrix4x3; 4, 3);
impl_array_matrices!(Matrix2x4; 2, 4);
impl_array_matrices!(Matrix4x2; 4, 2);
impl_array_matrices!(Matrix2x3; 2, 3);
impl_array_matrices!(Matrix3x2; 3, 2);

macro_rules! impl_matrix_matrix_mul {
    ($m:expr, $p:expr, $n:expr) => {
        // Implement A * B matrix multiplication where A is m-by-p and B is p-by-n.
        impl<T: Pod, U: Pod> Mul<Tensor<[Tensor<[T; $n]>; $p]>> for Tensor<[Tensor<[T; $p]>; $m]>
        where
            Tensor<[T; $n]>: Mul<T, Output = Tensor<[U; $n]>>,
            Tensor<[Tensor<[U; $n]>; $p]>: SumOp<Output = Tensor<[U; $n]>>,
        {
            type Output = Tensor<[Tensor<[U; $n]>; $m]>;
            #[inline]
            fn mul(self, rhs: Tensor<[Tensor<[T; $n]>; $p]>) -> Self::Output {
                self.map(|row| rhs.zip_with(row, |rhs_row, entry| rhs_row * entry).sum_op())
            }
        }
    };
}

impl_matrix_matrix_mul!(1, 1, 1);
impl_matrix_matrix_mul!(1, 1, 2);
impl_matrix_matrix_mul!(1, 1, 3);
impl_matrix_matrix_mul!(1, 1, 4);

impl_matrix_matrix_mul!(1, 2, 1);
impl_matrix_matrix_mul!(1, 2, 2);
impl_matrix_matrix_mul!(1, 2, 3);
impl_matrix_matrix_mul!(1, 2, 4);

impl_matrix_matrix_mul!(1, 3, 1);
impl_matrix_matrix_mul!(1, 3, 2);
impl_matrix_matrix_mul!(1, 3, 3);
impl_matrix_matrix_mul!(1, 3, 4);

impl_matrix_matrix_mul!(1, 4, 1);
impl_matrix_matrix_mul!(1, 4, 2);
impl_matrix_matrix_mul!(1, 4, 3);
impl_matrix_matrix_mul!(1, 4, 4);

impl_matrix_matrix_mul!(2, 1, 1);
impl_matrix_matrix_mul!(2, 1, 2);
impl_matrix_matrix_mul!(2, 1, 3);
impl_matrix_matrix_mul!(2, 1, 4);

impl_matrix_matrix_mul!(2, 2, 1);
impl_matrix_matrix_mul!(2, 2, 2);
impl_matrix_matrix_mul!(2, 2, 3);
impl_matrix_matrix_mul!(2, 2, 4);

impl_matrix_matrix_mul!(2, 3, 1);
impl_matrix_matrix_mul!(2, 3, 2);
impl_matrix_matrix_mul!(2, 3, 3);
impl_matrix_matrix_mul!(2, 3, 4);

impl_matrix_matrix_mul!(2, 4, 1);
impl_matrix_matrix_mul!(2, 4, 2);
impl_matrix_matrix_mul!(2, 4, 3);
impl_matrix_matrix_mul!(2, 4, 4);

impl_matrix_matrix_mul!(3, 1, 1);
impl_matrix_matrix_mul!(3, 1, 2);
impl_matrix_matrix_mul!(3, 1, 3);
impl_matrix_matrix_mul!(3, 1, 4);

impl_matrix_matrix_mul!(3, 2, 1);
impl_matrix_matrix_mul!(3, 2, 2);
impl_matrix_matrix_mul!(3, 2, 3);
impl_matrix_matrix_mul!(3, 2, 4);

impl_matrix_matrix_mul!(3, 3, 1);
impl_matrix_matrix_mul!(3, 3, 2);
impl_matrix_matrix_mul!(3, 3, 3);
impl_matrix_matrix_mul!(3, 3, 4);

impl_matrix_matrix_mul!(3, 4, 1);
impl_matrix_matrix_mul!(3, 4, 2);
impl_matrix_matrix_mul!(3, 4, 3);
impl_matrix_matrix_mul!(3, 4, 4);

impl_matrix_matrix_mul!(4, 1, 1);
impl_matrix_matrix_mul!(4, 1, 2);
impl_matrix_matrix_mul!(4, 1, 3);
impl_matrix_matrix_mul!(4, 1, 4);

impl_matrix_matrix_mul!(4, 2, 1);
impl_matrix_matrix_mul!(4, 2, 2);
impl_matrix_matrix_mul!(4, 2, 3);
impl_matrix_matrix_mul!(4, 2, 4);

impl_matrix_matrix_mul!(4, 3, 1);
impl_matrix_matrix_mul!(4, 3, 2);
impl_matrix_matrix_mul!(4, 3, 3);
impl_matrix_matrix_mul!(4, 3, 4);

impl_matrix_matrix_mul!(4, 4, 1);
impl_matrix_matrix_mul!(4, 4, 2);
impl_matrix_matrix_mul!(4, 4, 3);
impl_matrix_matrix_mul!(4, 4, 4);

impl_matrix_matrix_mul!(1, 9, 9);

// Outer products:
impl_matrix_matrix_mul!(5, 1, 5);
impl_matrix_matrix_mul!(6, 1, 6);
impl_matrix_matrix_mul!(7, 1, 7);
impl_matrix_matrix_mul!(8, 1, 8);
impl_matrix_matrix_mul!(9, 1, 9);
impl_matrix_matrix_mul!(10, 1, 10);
impl_matrix_matrix_mul!(11, 1, 11);
impl_matrix_matrix_mul!(12, 1, 12);
impl_matrix_matrix_mul!(13, 1, 13);
impl_matrix_matrix_mul!(14, 1, 14);
impl_matrix_matrix_mul!(15, 1, 15);
impl_matrix_matrix_mul!(16, 1, 16);

pub trait AsMatrix {
    type Matrix;
    fn as_matrix(self) -> Self::Matrix;
}

macro_rules! impl_as_matrix {
    ($outer_n:expr; $inner_n:expr, $inner_nty:ident) => {
        // Convert UniChunked arrays into matrices
        impl<'a, S: Scalar> AsMatrix for UniChunked<&'a [S; $outer_n], $inner_nty> {
            type Matrix = &'a Tensor<[Tensor<[Tensor<S>; $inner_n]>; $outer_n / $inner_n]>;
            #[inline]
            fn as_matrix(self) -> Self::Matrix {
                self.into_arrays().as_tensor()
            }
        }
        impl<'a, S: Scalar> AsMatrix for UniChunked<&'a mut [S; $outer_n], $inner_nty> {
            type Matrix = &'a mut Tensor<[Tensor<[Tensor<S>; $inner_n]>; $outer_n / $inner_n]>;
            #[inline]
            fn as_matrix(self) -> Self::Matrix {
                self.into_arrays().as_mut_tensor()
            }
        }
    };
}

impl_as_matrix!(1; 1, U1);
impl_as_matrix!(2; 1, U1);
impl_as_matrix!(2; 2, U2);
impl_as_matrix!(3; 1, U1);
impl_as_matrix!(3; 3, U3);
impl_as_matrix!(4; 1, U1);
impl_as_matrix!(4; 2, U2);
impl_as_matrix!(4; 4, U4);
impl_as_matrix!(5; 1, U1);
impl_as_matrix!(5; 5, U5);
impl_as_matrix!(6; 1, U1);
impl_as_matrix!(6; 2, U2);
impl_as_matrix!(6; 3, U3);
impl_as_matrix!(6; 6, U6);
impl_as_matrix!(7; 1, U1);
impl_as_matrix!(7; 7, U7);
impl_as_matrix!(8; 1, U1);
impl_as_matrix!(8; 2, U2);
impl_as_matrix!(8; 4, U4);
impl_as_matrix!(8; 8, U8);
impl_as_matrix!(9; 1, U1);
impl_as_matrix!(9; 3, U3);
impl_as_matrix!(9; 9, U9);
impl_as_matrix!(10; 1, U1);
impl_as_matrix!(10; 2, U2);
impl_as_matrix!(10; 5, U5);
impl_as_matrix!(10; 10, U10);
impl_as_matrix!(11; 1, U1);
impl_as_matrix!(11; 11, U11);
impl_as_matrix!(12; 1, U1);
impl_as_matrix!(12; 2, U2);
impl_as_matrix!(12; 3, U3);
impl_as_matrix!(12; 4, U4);
impl_as_matrix!(12; 6, U6);
impl_as_matrix!(12; 12, U12);
impl_as_matrix!(13; 1, U1);
impl_as_matrix!(13; 13, U13);
impl_as_matrix!(14; 1, U1);
impl_as_matrix!(14; 7, U7);
impl_as_matrix!(14; 14, U14);
impl_as_matrix!(15; 1, U1);
impl_as_matrix!(15; 3, U3);
impl_as_matrix!(15; 5, U5);
impl_as_matrix!(15; 15, U15);
impl_as_matrix!(16; 1, U1);
impl_as_matrix!(16; 2, U2);
impl_as_matrix!(16; 4, U4);
impl_as_matrix!(16; 8, U8);
impl_as_matrix!(16; 16, U16);

/*
 * The following section defines functions on specific small matrix types.
 */

/*
 * Skew symmetric matrix representing the cross product operator
 */
impl<S: Scalar + std::ops::Neg<Output = S>> Vector3<S> {
    /// Convert this vector into a skew symmetric matrix, which represents the cross
    /// product operator (when applied to another vector).
    #[inline]
    pub fn skew(&self) -> Matrix3<S> {
        Matrix3::new([
            [S::zero(), -self[2], self[1]],
            [self[2], S::zero(), -self[0]],
            [-self[1], self[0], S::zero()],
        ])
    }
}

/*
 * The determinant is computed recursively using co-factor expansions.
 */

/// Determinant of a 1x1 Matrix.
impl<S: Scalar> Matrix1<S> {
    #[inline]
    pub fn determinant(&self) -> S {
        self[0][0]
    }
}

macro_rules! impl_determinant {
    ($n:expr) => {
        /// Determinant of a 2x2 Matrix.
        impl<S: Scalar> Tensor<[Tensor<[Tensor<S>; $n]>; $n]> {
            /// Construct a matrix smaller in both dimensions by 1 that is the same as the
            /// original matrix but without the first row and a given column. Although this is
            /// a very specific function, it is useful for efficient co-factor expansions.
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn without_row_and_first_col(
                &self,
                col: usize,
            ) -> Tensor<[Tensor<[Tensor<S>; $n - 1]>; $n - 1]> {
                // Ensure that T has the same size as MaybeUninit.
                debug_assert_eq!(
                    std::mem::size_of::<[[MaybeUninit<S>; $n - 1]; $n - 1]>(),
                    std::mem::size_of::<Tensor<[Tensor<[Tensor<S>; $n - 1]>; $n - 1]>>()
                );
                let mut m: [[MaybeUninit<S>; $n - 1]; $n - 1] =
                    unsafe { MaybeUninit::uninit().assume_init() };
                for i in 1..$n {
                    // Transmute to a MaybeUninit slice.
                    let slice = unsafe {
                        std::mem::transmute(&self[if i < col + 1 { i - 1 } else { i }][1..$n])
                    };
                    m[i - 1].copy_from_slice(slice);
                }
                // Transmute back to initialized type.
                unsafe { std::mem::transmute_copy(&m) }
            }

            /// Compute the determinant of the matrix recursively.
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn determinant(&self) -> S {
                let mut det = self[0][0] * self.without_row_and_first_col(0).determinant();
                for row in 1..$n {
                    let cofactor = self[row][0] * self.without_row_and_first_col(row).determinant();
                    if row & 1 == 0 {
                        det += cofactor;
                    } else {
                        det -= cofactor;
                    }
                }
                det
            }
        }
    };
}

impl_determinant!(2);
impl_determinant!(3);
impl_determinant!(4);

/*
 * The inverse of a matrix
 */

impl<S: Scalar> Matrix1<S> {
    /// Compute the inverse of a 1x1 matrix.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let denom = self[0][0];
        if denom != S::zero() {
            Some(Self::new([[S::one() / denom]]))
        } else {
            None
        }
    }
    /// Invert the 1x1 matrix in place. Return true if inversion was successful.
    #[inline]
    pub fn invert(&mut self) -> bool {
        let denom = self[0][0];
        if denom != S::zero() {
            self[0][0] = S::one() / denom;
            true
        } else {
            false
        }
    }
}

impl<S: Scalar + Float> Matrix2<S> {
    /// Compute the inverse of a 2x2 matrix.
    #[inline]
    pub fn inverse(&self) -> Option<Matrix2<S>> {
        let det = self.determinant();
        if det != S::zero() {
            Some(
                [
                    [self[1][1] / det, -self[0][1] / det],
                    [-self[1][0] / det, self[0][0] / det],
                ]
                .into_tensor(),
            )
        } else {
            None
        }
    }
    /// Compute the transpose of a 3x3 matrix inverse.
    #[inline]
    pub fn inverse_transpose(&self) -> Option<Self> {
        let det = self.determinant();
        if det != S::zero() {
            Some(
                [
                    [self[1][1] / det, -self[1][0] / det],
                    [-self[0][1] / det, self[0][0] / det],
                ]
                .into_tensor(),
            )
        } else {
            None
        }
    }
    /// Invert the 2x2 matrix in place. Return true if inversion was successful.
    ///
    /// Warning: Microbenchmarks show this function to be slower than inverse.
    #[inline]
    pub fn invert(&mut self) -> bool {
        let det = self.determinant();
        if det != S::zero() {
            unsafe {
                let m00 =
                    self.data.get_unchecked_mut(0).data.get_unchecked_mut(0) as *mut Tensor<S>;
                let m11 =
                    self.data.get_unchecked_mut(1).data.get_unchecked_mut(1) as *mut Tensor<S>;
                std::ptr::swap(m00, m11);
                self.data
                    .get_unchecked_mut(0)
                    .data
                    .get_unchecked_mut(1)
                    .data *= -S::one();
                self.data
                    .get_unchecked_mut(1)
                    .data
                    .get_unchecked_mut(0)
                    .data *= -S::one();
            }
            *self /= det;
            true
        } else {
            false
        }
    }
}
impl<S: Scalar + Float> Matrix3<S> {
    /// Compute the inverse of a 3x3 matrix.
    #[inline]
    pub fn inverse(&self) -> Option<Matrix3<S>> {
        self.inverse_transpose().map(|x| x.transpose())
    }
    /// Compute the transpose of a 3x3 matrix inverse.
    #[inline]
    pub fn inverse_transpose(&self) -> Option<Self> {
        let det = self.determinant();
        if det != S::zero() {
            Some(Tensor {
                data: [
                    self[1].cross(self[2]) / det,
                    self[2].cross(self[0]) / det,
                    self[0].cross(self[1]) / det,
                ],
            })
        } else {
            None
        }
    }
    /// Invert the 3x3 matrix in place. Return true if inversion was successful.
    ///
    /// Warning: Microbenchmarks show this function to be slower than inverse.
    #[inline]
    pub fn invert(&mut self) -> bool {
        match self.inverse() {
            Some(inv) => {
                *self = inv;
                true
            }
            None => false,
        }
    }
}

/* Quaternions */

/// A quaternion type.
///
/// The scalar of the quaternion is stored first, followed by the vector part.
#[derive(Copy, Clone, Debug)]
pub struct Quaternion<T>(pub T, pub Vector3<T>);

impl<T: PartialEq<U>, U> PartialEq<Quaternion<U>> for Quaternion<T> {
    fn eq(&self, other: &Quaternion<U>) -> bool {
        self.0.eq(&other.0) && self.1.eq(&other.1)
    }
}

impl<T: Real> Quaternion<T> {
    /// Construct a new quaternion from the given scalar and vector parts.
    #[inline]
    pub fn new<V: Into<[T; 3]>>(s: T, v: V) -> Self {
        Quaternion(s, v.into().into_tensor())
    }

    /// Construct a unit quaternion given only the vector part.
    ///
    /// Note that the norm of `v` is clamped to be at most 1.
    #[inline]
    pub fn unit<V: Into<[T; 3]>>(v: V) -> Self {
        let v = v.into().into_tensor();
        let norm = v.norm();
        if norm > T::one() {
            Quaternion(T::zero(), v / norm)
        } else {
            Quaternion(T::one() - norm, v)
        }
    }

    /// Construct a `Quaternion` from an axis-angle vector.
    #[inline]
    pub fn from_vector<V: Into<[T; 3]>>(k: V) -> Self {
        let k = k.into().into_tensor();
        let angle = k.norm();
        let half_angle = angle * T::from(0.5).unwrap();
        let s = half_angle.cos();
        let v = k * if angle == T::zero() {
            T::zero()
        } else {
            half_angle.sin() / angle
        };
        Quaternion(s, v)
    }

    /// Convert to an axis-angle vector.
    #[inline]
    pub fn into_vector(self) -> Vector3<T> {
        let norm = self.1.norm();
        if norm > T::zero() {
            self.1 * (T::from(2.0).unwrap() * norm.atan2(self.0) / norm)
        } else {
            Vector3::zero()
        }
    }

    /// Construct the conjugate quaternion.
    #[inline]
    pub fn conj(self) -> Quaternion<T> {
        Quaternion(self.0, -self.1)
    }

    /// Construct the inverse quaternion.
    ///
    /// Note that if this quaternion is zero, then this function may generate `NaN`s.
    #[inline]
    pub fn inv(self) -> Quaternion<T> {
        debug_assert!(self.norm() > T::zero());
        Quaternion(self.0, -self.1 / self.norm())
    }

    /// Compute the squared norm of the quaternion.
    ///
    /// This is just the sum of squares of all 4 of the elements.
    #[inline]
    pub fn norm_squared(&self) -> T {
        self.0 * self.0 + self.1.norm_squared()
    }

    /// Compute the norm of the quaternion.
    ///
    /// This is just the square root of the sum of squares of all 4 of the elements.
    #[inline]
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Normalize the quaternion in place. Return its norm.
    #[inline]
    pub fn normalize(&mut self) -> T {
        let norm = self.norm();
        if norm.is_zero() {
            return norm;
        }
        let norm_inv = T::one() / norm;
        self.0 *= norm_inv;
        self.1 *= norm_inv;
        norm
    }

    /// Return a normalized version of this quaternion.
    #[inline]
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    /// Rotate a given vector by the quaternion.
    #[inline]
    pub fn rotate<V: Into<[T; 3]> + From<[T; 3]>>(&self, v: V) -> V {
        (*self * v.into().into() * self.inv()).1.into_data().into()
    }

    /// Construct a rotation matrix corresponding to this quaternion.
    #[inline]
    pub fn rotation(&self) -> Matrix3<T> {
        let (a, [b, c, d]) = (self.0, self.1.into_data());
        let (aa, bb, cc, dd) = (a * a, b * b, c * c, d * d);
        let _2 = T::from(2.0).unwrap();
        let (ab, bc, ad, bd, ac, cd) = (a * b, b * c, a * d, b * d, a * c, c * d);
        Matrix3::new([
            [aa + bb - cc - dd, _2 * bc - _2 * ad, _2 * bd + _2 * ac],
            [_2 * bc + _2 * ad, aa - bb + cc - dd, _2 * cd - _2 * ab],
            [_2 * bd - _2 * ac, _2 * cd + _2 * ab, aa - bb - cc + dd],
        ])
    }
}

impl<T: Real> From<[T; 3]> for Quaternion<T> {
    fn from(v: [T; 3]) -> Quaternion<T> {
        Quaternion::new(T::zero(), v)
    }
}

impl<T: Real> Mul for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, other: Quaternion<T>) -> Quaternion<T> {
        let s = self.0 * other.0 - self.1.dot(other.1);
        let v = other.1 * self.0 + self.1 * other.0 + self.1.cross(other.1);
        Quaternion(s, v)
    }
}

impl<T: Real> MulAssign for Quaternion<T>
where
    Self: Mul<Output = Self>,
{
    #[inline]
    fn mul_assign(&mut self, other: Quaternion<T>) {
        *self = *self * other;
    }
}

impl<T: Real> Div for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn div(self, other: Quaternion<T>) -> Quaternion<T> {
        self * other.inv()
    }
}

impl<T: Real> DivAssign for Quaternion<T> {
    #[inline]
    fn div_assign(&mut self, other: Quaternion<T>) {
        *self *= other.inv();
    }
}

impl<T: Scalar> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, other: Quaternion<T>) -> Quaternion<T> {
        Quaternion(self.0 + other.0, self.1 + other.1)
    }
}

impl<T: Scalar> AddAssign for Quaternion<T> {
    #[inline]
    fn add_assign(&mut self, other: Quaternion<T>) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<T: Scalar> Sub for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn sub(mut self, other: Quaternion<T>) -> Quaternion<T> {
        self.0 -= other.0;
        self.1 -= other.1;
        self
    }
}

impl<T: Scalar> SubAssign for Quaternion<T> {
    #[inline]
    fn sub_assign(&mut self, other: Quaternion<T>) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

#[cfg(feature = "approx")]
impl<U: Scalar, T: Scalar + approx::AbsDiffEq<U>> approx::AbsDiffEq<Quaternion<U>> for Quaternion<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }
    #[inline]
    fn abs_diff_eq(&self, other: &Quaternion<U>, epsilon: Self::Epsilon) -> bool {
        self.0.abs_diff_eq(&other.0, epsilon) && self.1.abs_diff_eq(&other.1, epsilon)
    }
}

#[cfg(feature = "approx")]
impl<U: Scalar, T: Scalar + approx::RelativeEq<U>> approx::RelativeEq<Quaternion<U>>
    for Quaternion<T>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }
    #[inline]
    fn relative_eq(
        &self,
        other: &Quaternion<U>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0.relative_eq(&other.0, epsilon, max_relative)
            && self.1.relative_eq(&other.1, epsilon, max_relative)
    }
}
#[cfg(feature = "approx")]
impl<U: Scalar, T: Scalar + approx::UlpsEq<U>> approx::UlpsEq<Quaternion<U>> for Quaternion<T>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Quaternion<U>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.0.ulps_eq(&other.0, epsilon, max_ulps) && self.1.ulps_eq(&other.1, epsilon, max_ulps)
    }
}

/// Produce a rotation matrix given a 3D vector.
///
/// Rodrigues' formula is used for this computation.
#[inline]
pub fn rotation<T: Real,V: Into<[T;3]>>(v: V) -> Matrix3<T> {
    let v = v.into().into_tensor();
    let angle = v.norm();
    let mut res = Matrix3::identity();
    if angle > T::zero() {
        let cos = angle.cos();
        let sin = angle.sin();
        let k = (v / angle).skew();
        res += k * sin + k * k * (T::one() - cos);
    }
    res
}

/// Rotate the vector `r` by the axis-angle vector `v`.
///
/// Rodrigues' formula is used for this computation.
#[inline]
pub fn rotate<T: Real, R: Into<[T; 3]>, V: Into<[T; 3]>>(r: R, v: V) -> Vector3<T> {
    let r = r.into().into_tensor();
    let v = v.into().into_tensor();
    let angle = v.norm();
    if angle > T::zero() {
        let cos = angle.cos();
        let sin = angle.sin();
        let k = v / angle;
        r * cos + k.cross(r) * sin + k * (k.dot(r) * (T::one() - cos))
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn skew() {
        let a = Vector3::new([1.0, 2.0, 3.1]);
        let b = Vector3::new([4.0, 5.1, 6.0]);

        // Test that the result of calling `skew` on a vector produces a skew symmetric matrix.
        assert_eq!(a.skew(), -a.skew().transpose());

        // Test that the skew symmetric matrix acts as a cross product.
        let exp = a.cross(b);
        assert_relative_eq!(a.skew() * b, exp);
    }

    #[test]
    fn quaternions() {
        let a = Vector3::new([1.0, 2.0, 3.1]);
        let b = Vector3::new([0.4, 0.51, 0.6]);

        // Test axis angle quaternion round trip
        assert_relative_eq!(a, Quaternion::from_vector(a).into_vector());
        assert_relative_eq!(b, Quaternion::from_vector(b).into_vector());

        // Test that the quaternion produces the same rotation as a rotation matrix.
        assert_relative_eq!(
            rotate(b, a),
            Quaternion::from_vector(a).rotate(b),
            max_relative = 1e-7
        );

        // Test the inverse of a quaternion.
        let unit = Quaternion::unit(a);
        assert_relative_eq!(
            unit * unit.inv(),
            Quaternion::unit([0.0; 3]),
            max_relative = 1e-7
        );
        assert_relative_eq!(
            unit.inv() * unit,
            Quaternion::unit([0.0; 3]),
            max_relative = 1e-7
        );

        // Test that the unit quaternion constructor produces a unit quaternion.
        assert_relative_eq!(unit.norm(), 1.0);

        // Test composition of quaternions is the same as composition of rotation matrices.
        let qab = Quaternion::from_vector(a) * Quaternion::from_vector(b);
        assert_relative_eq!(
            qab.rotation(),
            rotation(a) * rotation(b),
            max_relative = 1e-7
        );
    }

    #[test]
    fn rotations() {
        let a = Vector3::new([1.0, 2.0, 3.1]);
        let b = Vector3::new([0.4, 0.51, 0.6]);

        // Test that the transpose is equal to the inverse.
        let rot = rotation(a);
        assert_relative_eq!(
            rot * rot.transpose(),
            Matrix3::identity(),
            max_relative = 1e-7,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            rot.transpose() * rot,
            Matrix3::identity(),
            max_relative = 1e-7
        );

        let rot = rotation(b);
        assert_relative_eq!(
            rot * rot.transpose(),
            Matrix3::identity(),
            max_relative = 1e-7
        );
        assert_relative_eq!(
            rot.transpose() * rot,
            Matrix3::identity(),
            max_relative = 1e-7
        );

        // Test that a negative vector will produce the inverse rotation.
        let rot_inv = rotation(-b);
        assert_relative_eq!(rot * rot_inv, Matrix3::identity(), max_relative = 1e-7);
        assert_relative_eq!(rot_inv * rot, Matrix3::identity(), max_relative = 1e-7);

        // Test that multiplying tby the rotation matrix is the same as rotating a vector with
        // `rotate`.
        assert_relative_eq!(rot * a, rotate(a, b), max_relative = 1e-7);
    }

    #[test]
    fn vector_scalar_mul() {
        let mut a = Vector4::new([1, 2, 3, 4]);

        // Right multiply by raw scalar.
        assert_eq!(a * 3, Vector4::new([3, 6, 9, 12]));

        // Right multiply by wrapped scalar.
        assert_eq!(a * 3.into_tensor(), Vector4::new([3, 6, 9, 12]));

        // Right assign multiply by raw scalar.
        a *= 2;
        assert_eq!(a, Vector4::new([2, 4, 6, 8]));

        // Right assign multiply by wrapped scalar.
        a *= 2.into_tensor();
        assert_eq!(a, Vector4::new([4, 8, 12, 16]));
    }

    #[test]
    fn vector_scalar_div() {
        let mut a = Vector4::new([1.0, 2.0, 4.0, 8.0]);

        // Right divide by raw scalar.
        assert_eq!(a / 2.0, Vector4::new([0.5, 1.0, 2.0, 4.0]));

        // Right assign divide by raw scalar.
        a /= 2.0;
        assert_eq!(a, Vector4::new([0.5, 1.0, 2.0, 4.0]));
    }

    #[test]
    fn vector_add() {
        let a = Vector4::new([1, 2, 3, 4]);
        let b = Vector4::new([5, 6, 7, 8]);
        assert_eq!(Vector4::new([6, 8, 10, 12]), a + b);

        let mut c = Vector4::new([0, 1, 2, 3]);
        c += a;
        assert_eq!(c, Vector4::new([1, 3, 5, 7]));
    }

    #[test]
    fn vector_sub() {
        let a = Vector4::new([1, 2, 3, 4]);
        let b = Vector4::new([5, 6, 7, 8]);
        assert_eq!(Vector4::new([4, 4, 4, 4]), b - a);

        let mut c = Vector4::new([1, 3, 5, 7]);
        c -= a;
        assert_eq!(c, Vector4::new([0, 1, 2, 3]));
    }

    #[test]
    fn lower_triangular_vec() {
        let m = Matrix3::new([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        assert_eq!(m.lower_triangular_vec(), Vector6::new([0, 3, 4, 6, 7, 8]));
    }
}
