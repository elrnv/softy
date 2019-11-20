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

macro_rules! impl_array_vectors {
    ($vecn:ident, $rowvecn:ident; $n:expr) => {
        pub type $vecn<T> = Tensor<[T; $n]>;
        pub type $rowvecn<T> = Tensor<[[T; $n]; 1]>;

        impl<T: Scalar> Matrix for Tensor<[T; $n]> {
            type Transpose = $rowvecn<T>;
            #[inline]
            fn transpose(self) -> Self::Transpose {
                Tensor::new([self.data; 1])
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

        impl<T: Scalar> CwiseMulAssignOp<Tensor<[T; $n]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn cwise_mul_assign(&mut self, rhs: Tensor<[T; $n]>) {
                debug_assert!(self.len() >= rhs.len());
                for i in 0..$n {
                    unsafe { *self.data.get_unchecked_mut(i) *= *rhs.data.get_unchecked(i) };
                }
            }
        }

        impl<T: Scalar> AddAssign<Tensor<[T; $n]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[T; $n]>) {
                debug_assert!(self.len() >= rhs.len());
                for i in 0..$n {
                    unsafe { *self.data.get_unchecked_mut(i) += *rhs.data.get_unchecked(i) };
                }
            }
        }

        impl<T: Scalar> SubAssign<Tensor<[T; $n]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[T; $n]>) {
                debug_assert!(self.len() >= rhs.len());
                for i in 0..$n {
                    unsafe { *self.data.get_unchecked_mut(i) += *rhs.data.get_unchecked(i) };
                }
            }
        }


        impl<T: Scalar> AddAssign<Tensor<[T; $n]>> for Tensor<Vec<T>> {
            #[inline]
            fn add_assign(&mut self, rhs: Tensor<[T; $n]>) {
                self.data.as_mut_slice().as_mut_tensor().add_assign(rhs);
            }
        }

        impl<T: Scalar> SubAssign<Tensor<[T; $n]>> for Tensor<Vec<T>> {
            #[inline]
            fn sub_assign(&mut self, rhs: Tensor<[T; $n]>) {
                self.data.as_mut_slice().as_mut_tensor().sub_assign(rhs);
            }
        }

        impl<T: Scalar> CwiseMulOp<Tensor<T>> for Tensor<[T; $n]> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn cwise_mul(self, rhs: Tensor<T>) -> Self::Output {
                self * rhs
            }
        }

        impl<T: Scalar> CwiseMulOp<Tensor<[T; $n]>> for Tensor<T> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn cwise_mul(self, rhs: Tensor<[T; $n]>) -> Self::Output {
                rhs * self
            }
        }

        impl<T: Scalar> CwiseMulOp for Tensor<[T; $n]> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            #[unroll_for_loops]
            fn cwise_mul(mut self, rhs: Self) -> Self::Output {
                for i in 0..$n {
                    self[i] *= rhs[i];
                }
                self
            }
        }

        impl<T: Scalar> RecursiveSumOp for Tensor<[T; $n]> {
            type Output = Tensor<T>;
            #[inline]
            fn recursive_sum(self) -> Self::Output {
                Tensor::new(self.sum())
            }
        }
        impl<T: Scalar> SumOp for Tensor<[T; $n]> {
            type Output = Tensor<T>;
            #[inline]
            fn sum_op(self) -> Self::Output {
                Tensor::new(self.sum())
            }
        }

        impl<T: Scalar> DotOp<Tensor<T>> for Tensor<[T; $n]> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<T>) -> Self::Output {
                self * rhs
            }
        }

        impl<T: Scalar> DotOp<Tensor<[T; $n]>> for Tensor<T> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<[T; $n]>) -> Self::Output {
                rhs * self
            }
        }

        impl<T: Scalar> DotOp for Tensor<[T; $n]> {
            type Output = Tensor<T>;
            #[inline]
            fn dot_op(self, rhs: Self) -> Tensor<T> {
                Tensor::new(Self::dot(self, rhs))
            }
        }

        impl<T: Scalar> Tensor<[T; $n]> {
            pub fn zeros() -> Tensor<[T; $n]> {
                Tensor::new([T::zero(); $n])
            }
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn dot(self, other: Tensor<[T; $n]>) -> T {
                let mut prod = self.data[0] * other.data[0];
                for i in 1..$n {
                    prod += self.data[i] * other.data[i];
                }
                prod
            }
            #[inline]
            pub fn norm_squared(&self) -> T {
                (*self).map(|x| x * x).sum()
            }
        }

        impl<T: Copy> Tensor<[T; $n]> {
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
                    std::mem::size_of::<[U; $n]>()
                );
                Tensor::new(unsafe { std::mem::transmute_copy::<_, [U; $n]>(&out) })
            }
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
        }

        impl<T: std::ops::Add<Output = T> + Zero + Copy> Tensor<[T; $n]> {
            #[inline]
            pub fn sum(&self) -> T {
                self.fold(T::zero(), |acc, x| acc + x)
            }
        }

        impl<T: Float + Scalar> Tensor<[T; $n]> {
            #[inline]
            pub fn norm(&self) -> T {
                self.norm_squared().sqrt()
            }
        }

        impl<T: Scalar> Add for Tensor<[T; $n]> {
            type Output = Self;

            /// Add two tensor arrays together.
            #[inline]
            fn add(mut self, rhs: Self) -> Self::Output {
                self += rhs;
                self
            }
        }

        impl<T: Scalar> AddAssign<Tensor<[T; $n]>> for &mut Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: Scalar> AddAssign for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: Scalar> AddAssign<&Tensor<[T; $n]>> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: &Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: Scalar> Sub for Tensor<[T; $n]> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self::Output {
                self -= rhs;
                self
            }
        }

        impl<T: Scalar> SubAssign<Tensor<[T; $n]>> for &mut Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        impl<T: Scalar> SubAssign for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        impl<T: Scalar> SubAssign<&Tensor<[T; $n]>> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: &Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        /*
         * Vector-RowVector multiply
         * This sepecial case treats Vectors as column vectors
         */
        impl<T: Scalar> Mul<Tensor<[[T; $n]; 1]>> for Tensor<[T; $n]> {
            type Output = Tensor<[[T; $n]; $n]>;
            #[inline]
            fn mul(self, rhs: Tensor<[[T; $n]; 1]>) -> Self::Output {
                self.map(|row| (Tensor::new(rhs.data[0]) * Tensor::new(row)).data)
            }
        }

        // Right scalar multiply by a raw scalar.
        impl<T: Scalar> Mul<T> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn mul(mut self, rhs: T) -> Self::Output {
                self *= rhs;
                self
            }
        }

        // Scalar multiply assign by a raw scalar.
        impl<T: Scalar> MulAssign<T> for Tensor<[T; $n]> {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs);
            }
        }

        // Scalar divide by a raw scalar.
        impl<T: Scalar> Div<T> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn div(mut self, rhs: T) -> Self::Output {
                self /= rhs;
                self
            }
        }

        // Scalar divide assign by a raw scalar.
        impl<T: Scalar> DivAssign<T> for Tensor<[T; $n]> {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                *self /= Tensor::new(rhs);
            }
        }

        // Right multiply by a scalar tensor.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T: Scalar> Mul<Tensor<T>> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T: Scalar> MulAssign<Tensor<T>> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn mul_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    *self.data.index_mut(i).as_mut_tensor() *= rhs.clone();
                }
            }
        }

        impl<T: Scalar> Div<Tensor<T>> for Tensor<[T; $n]> {
            type Output = Self;
            #[inline]
            fn div(mut self, rhs: Tensor<T>) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl<T: Scalar> DivAssign<Tensor<T>> for Tensor<[T; $n]> {
            #[inline]
            #[unroll_for_loops]
            fn div_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    *self.data.index_mut(i).as_mut_tensor() /= rhs.clone();
                }
            }
        }

        impl<T: Scalar> Index<usize> for Tensor<[T; $n]> {
            type Output = T;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }
        impl<T: Scalar> IndexMut<usize> for Tensor<[T; $n]> {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.data[index]
            }
        }

        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        impl<T: Scalar> Mul<Tensor<[T; $n]>> for Tensor<T> {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn mul(self, mut rhs: Tensor<[T; $n]>) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
        impl<T: Scalar> Zero for Tensor<[T; $n]> {
            #[inline]
            fn zero() -> Self {
                Tensor::new([Zero::zero(); $n])
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl<T> Into<[T; $n]> for Tensor<[T; $n]> {
            #[inline]
            fn into(self) -> [T; $n] {
                self.into_inner()
            }
        }
        impl<T> Into<[T; $n]> for Tensor<[[T; $n]; 1]> {
            #[inline]
            fn into(self) -> [T; $n] {
                let [x] = self.into_inner();
                x
            }
        }
        impl<T: Copy + num_traits::ToPrimitive> Tensor<[T; $n]> {
            /// Casts the components of the vector into another type.
            ///
            /// # Panics
            /// This function panics if the cast fails.
            #[inline]
            pub fn cast<U: Pod + num_traits::NumCast>(&self) -> Tensor<[U; $n]> {
                self.map(|x| U::from(x).unwrap())
            }
        }
    };
}

impl_array_vectors!(Vector1, RowVector1; 1);
impl_array_vectors!(Vector2, RowVector2; 2);
impl_array_vectors!(Vector3, RowVector3; 3);
impl_array_vectors!(Vector4, RowVector4; 4);
impl_array_vectors!(Vector5, RowVector5; 5);
impl_array_vectors!(Vector6, RowVector6; 6);
impl_array_vectors!(Vector7, RowVector7; 7);
impl_array_vectors!(Vector8, RowVector8; 8);
impl_array_vectors!(Vector9, RowVector9; 9);

impl<T: Scalar> Tensor<[T; 3]> {
    #[inline]
    pub fn cross(self, other: Tensor<[T; 3]>) -> Tensor<[T; 3]> {
        Tensor::new([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }
}

macro_rules! impl_array_matrices {
    ($mtxn:ident; $r:expr, $c:expr) => {
        // Row-major square matrix.
        pub type $mtxn<T> = Tensor<[[T; $c]; $r]>;

        // Transposes of small matrices are implemented eagerly.
        impl<T: Scalar> Matrix for Tensor<[[T; $c]; $r]> {
            type Transpose = Tensor<[[T; $r]; $c]>;
            #[inline]
            #[unroll_for_loops]
            fn transpose(mut self) -> Self::Transpose {
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
                Tensor::new(unsafe { std::mem::transmute_copy::<_, [[T; $r]; $c]>(&m) })
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

        impl<T: Copy> Tensor<[[T; $c]; $r]> {
            /// Similar to `map` but applies the given function to each inner element.
            #[inline]
            #[unroll_for_loops]
            pub fn map_inner<U, F>(&self, mut f: F) -> Tensor<[[U; $c]; $r]>
            where
                U: Pod,
                F: FnMut(T) -> U,
            {
                // We use MaybeUninit here mostly to avoid a Zero trait bound.
                let mut out: [[MaybeUninit<U>; $c]; $r] =
                    unsafe { MaybeUninit::uninit().assume_init() };
                for row in 0..$r {
                    for col in 0..$c {
                        out[row][col] = MaybeUninit::new(f(self.data[row][col]));
                    }
                }
                // The Pod trait bound ensures safety here in release builds.
                // Sanity check here just in debug builds only, since this code is very likely in a
                // critical section.
                debug_assert_eq!(
                    std::mem::size_of::<[[MaybeUninit<U>; $c]; $r]>(),
                    std::mem::size_of::<[[U; $c]; $r]>()
                );
                Tensor::new(unsafe { std::mem::transmute_copy::<_, [[U; $c]; $r]>(&out) })
            }
        }

        impl<T: Scalar> Tensor<[[T; $c]; $r]> {
            /// Zip the rows of this matrix and another tensor with the given combinator.
            #[inline]
            #[unroll_for_loops]
            pub fn zip_with<B, U, F>(&self, other: Tensor<[B; $r]>, mut f: F) -> Tensor<[U; $r]>
            where
                B: Copy,
                U: Zero + Copy,
                F: FnMut([T; $c], B) -> U,
            {
                let mut out = [U::zero(); $r];
                for i in 0..$r {
                    out[i] = f(self.data[i], other.data[i]);
                }
                Tensor::new(out)
            }

            #[inline]
            pub fn identity() -> Tensor<[[T; $c]; $r]> {
                Self::from_diag_iter(std::iter::repeat(T::one()))
            }
            #[inline]
            pub fn diag(diag: &[T]) -> Tensor<[[T; $c]; $r]> {
                Self::from_diag_iter(diag.into_iter().cloned())
            }
            pub fn from_diag_iter<Iter: IntoIterator<Item = T>>(
                diag: Iter,
            ) -> Tensor<[[T; $c]; $r]> {
                let mut out = Self::zeros();
                for (i, elem) in diag.into_iter().take($r.min($c)).enumerate() {
                    out.data[i][i] = elem;
                }
                out
            }
            #[inline]
            pub fn zeros() -> Tensor<[[T; $c]; $r]> {
                Tensor::new([[T::zero(); $c]; $r])
            }

            #[inline]
            #[unroll_for_loops]
            pub fn fold_inner<B, F>(&self, mut init: B, mut f: F) -> B
            where
                F: FnMut(B, T) -> B,
            {
                for i in 0..$r {
                    init = Tensor::new(self.data[i]).fold(init, |acc, x| f(acc, x));
                }
                init
            }

            /// Compute the sum of all entries in this matrix.
            #[inline]
            pub fn sum_inner(&self) -> T {
                self.fold_inner(T::zero(), |acc, x| acc + x)
            }

            // TODO: optimize this function
            #[inline]
            pub fn trace(&self) -> T {
                let mut tr = self.data[0][0];
                for i in 1..$r.min($c) {
                    tr += self.data[i][i];
                }
                tr
            }
            #[inline]
            pub fn frob_norm_squared(&self) -> T {
                (*self).map_inner(|x| x * x).sum_inner()
            }
        }

        impl<T: Float + Scalar> Tensor<[[T; $c]; $r]> {
            #[inline]
            pub fn frob_norm(&self) -> T {
                self.frob_norm_squared().sqrt()
            }
        }

        impl<T: Scalar> Add for Tensor<[[T; $c]; $r]> {
            type Output = Self;

            /// Add two tensor arrays together.
            #[inline]
            fn add(mut self, rhs: Self) -> Self::Output {
                self += rhs;
                self
            }
        }

        impl<T: Copy + AddAssign> AddAssign<Tensor<[[T; $c]; $r]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
                for i in 0..$r {
                    for j in 0..$c {
                        self.data[$r * i + j] += rhs.data[i][j];
                    }
                }
            }
        }

        impl<T: Scalar> AddAssign<Tensor<[[T; $c]; $r]>> for &mut Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
                for i in 0..$r {
                    *self.data[i].as_mut_tensor() += *rhs.data[i].as_tensor();
                }
            }
        }
        impl<T: Scalar> AddAssign for Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$r {
                    *self.data[i].as_mut_tensor() += *rhs.data[i].as_tensor();
                }
            }
        }
        impl<T: Scalar> AddAssign<&Tensor<[[T; $c]; $r]>> for Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: &Tensor<[[T; $c]; $r]>) {
                for i in 0..$r {
                    *self.data[i].as_mut_tensor() += rhs.data[i].as_tensor();
                }
            }
        }
        impl<T: Scalar> Sub for Tensor<[[T; $c]; $r]> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self::Output {
                self -= rhs;
                self
            }
        }

        impl<T: Copy + SubAssign> SubAssign<Tensor<[[T; $c]; $r]>> for Tensor<[T]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
                for i in 0..$r {
                    for j in 0..$c {
                        self.data[$r * i + j] -= rhs.data[i][j];
                    }
                }
            }
        }

        impl<T: Scalar> SubAssign<Tensor<[[T; $c]; $r]>> for &mut Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Tensor<[[T; $c]; $r]>) {
                for i in 0..$r {
                    *self.data[i].as_mut_tensor() -= rhs.data[i].as_tensor();
                }
            }
        }
        impl<T: Scalar> SubAssign for Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$r {
                    *self.data[i].as_mut_tensor() -= rhs.data[i].as_tensor();
                }
            }
        }

        impl<T: Scalar> SubAssign<&Tensor<[[T; $c]; $r]>> for Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: &Tensor<[[T; $c]; $r]>) {
                for i in 0..$r {
                    *self.data[i].as_mut_tensor() -= rhs.data[i].as_tensor();
                }
            }
        }

        /*
         * Matrix-vector multiply
         */
        impl<T: Scalar> Mul<Tensor<[T; $c]>> for Tensor<[[T; $c]; $r]> {
            type Output = Tensor<[T; $r]>;
            #[inline]
            fn mul(self, rhs: Tensor<[T; $c]>) -> Self::Output {
                Tensor::new(self.map(|row| row.as_tensor().dot(rhs)).data)
            }
        }

        // Right scalar multiply by a raw scalar.
        impl<T: Scalar> Mul<T> for Tensor<[[T; $c]; $r]> {
            type Output = Self;
            #[inline]
            fn mul(mut self, rhs: T) -> Self::Output {
                self *= rhs;
                self
            }
        }

        // Scalar multiply assign by a raw scalar.
        impl<T: Scalar> MulAssign<T> for Tensor<[[T; $c]; $r]> {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs);
            }
        }

        // Scalar divide by a raw scalar.
        impl<T: Scalar> Div<T> for Tensor<[[T; $c]; $r]> {
            type Output = Self;
            #[inline]
            fn div(mut self, rhs: T) -> Self::Output {
                self /= rhs;
                self
            }
        }

        // Divide assign by a raw scalar.
        impl<T: Scalar> DivAssign<T> for Tensor<[[T; $c]; $r]> {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                *self /= Tensor::new(rhs);
            }
        }

        // Right multiply by a tensor scalar.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T: Scalar> Mul<Tensor<T>> for Tensor<[[T; $c]; $r]> {
            type Output = Self;
            #[inline]
            fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
                self *= rhs;
                self
            }
        }

        // Divide by a tensor scalar.
        impl<T: Scalar> Div<Tensor<T>> for Tensor<[[T; $c]; $r]> {
            type Output = Self;
            #[inline]
            fn div(mut self, rhs: Tensor<T>) -> Self::Output {
                self /= rhs;
                self
            }
        }

        // Multiply assign by a tensor scalar
        impl<T: Scalar> MulAssign<Tensor<T>> for Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn mul_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$r {
                    *self.data.index_mut(i).as_mut_tensor() *= rhs.clone();
                }
            }
        }

        // Divide assign by a tensor scalar
        impl<T: Scalar> DivAssign<Tensor<T>> for Tensor<[[T; $c]; $r]> {
            #[inline]
            #[unroll_for_loops]
            fn div_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$r {
                    *self.data.index_mut(i).as_mut_tensor() /= rhs.clone();
                }
            }
        }

        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        impl<T: Scalar> Mul<Tensor<[[T; $c]; $r]>> for Tensor<T> {
            type Output = Tensor<[[T; $c]; $r]>;
            #[inline]
            fn mul(self, mut rhs: Tensor<[[T; $c]; $r]>) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
        impl<T: Scalar> Zero for Tensor<[[T; $c]; $r]> {
            #[inline]
            fn zero() -> Self {
                Tensor::new([[Zero::zero(); $c]; $r])
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl<T: Scalar> Index<usize> for Tensor<[[T; $c]; $r]> {
            type Output = Tensor<[T; $c]>;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                self.data[index].as_tensor()
            }
        }
        impl<T: Scalar> IndexMut<usize> for Tensor<[[T; $c]; $r]> {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.data[index].as_mut_tensor()
            }
        }
        impl<T: Copy + num_traits::ToPrimitive> Tensor<[[T; $c]; $r]> {
            /// Casts the components of the matrix into another type.
            ///
            /// # Panics
            /// This function panics if the cast fails.
            #[inline]
            pub fn cast_inner<U: Pod + num_traits::NumCast>(&self) -> Tensor<[[U; $c]; $r]> {
                self.map_inner(|x| U::from(x).unwrap())
            }
        }

        impl<T: Scalar> DotOp<Tensor<T>> for Tensor<[[T; $c]; $r]> {
            type Output = Tensor<[[T; $c]; $r]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<T>) -> Self::Output {
                self * rhs
            }
        }

        impl<T: Scalar> DotOp<Tensor<[[T; $c]; $r]>> for Tensor<T> {
            type Output = Tensor<[[T; $c]; $r]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<[[T; $c]; $r]>) -> Self::Output {
                rhs * self
            }
        }

        // Dot operator as compared to multiplication is always commutative. This means that no
        // matter the order, it will always chose to contract along the outer dimension.
        impl<T: Scalar> DotOp<Tensor<[T; $r]>> for Tensor<[[T; $c]; $r]> {
            type Output = Tensor<[T; $c]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<[T; $r]>) -> Self::Output {
                let Tensor { data: [vec] } = rhs.transpose() * self;
                Tensor::new(vec)
            }
        }

        impl<T: Scalar> DotOp<Tensor<[[T; $c]; $r]>> for Tensor<[T; $r]> {
            type Output = Tensor<[T; $c]>;
            #[inline]
            fn dot_op(self, rhs: Tensor<[[T; $c]; $r]>) -> Self::Output {
                let Tensor { data: [vec] } = self.transpose() * rhs;
                Tensor::new(vec)
            }
        }

        impl<T: Scalar> RecursiveSumOp for Tensor<[[T; $c]; $r]> {
            type Output = Tensor<T>;
            #[inline]
            fn recursive_sum(self) -> Self::Output {
                Tensor::new(self.sum_inner())
            }
        }
        impl<T: Scalar> SumOp for Tensor<[[T; $c]; $r]> {
            type Output = Tensor<[T; $c]>;
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            fn sum_op(self) -> Self::Output {
                let mut sum = self[0];
                for i in 1..$r {
                    sum += self[i];
                }
                sum
            }
        }
    };
}

impl_array_matrices!(Matrix1; 1, 1);
impl_array_matrices!(Matrix2; 2, 2);
impl_array_matrices!(Matrix3; 3, 3);
impl_array_matrices!(Matrix4; 4, 4);

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
        impl<T: Scalar> Mul<Tensor<[[T; $n]; $p]>> for Tensor<[[T; $p]; $m]> {
            type Output = Tensor<[[T; $n]; $m]>;
            #[inline]
            fn mul(self, rhs: Tensor<[[T; $n]; $p]>) -> Self::Output {
                Tensor::new(
                    self.map(|row| {
                        rhs.zip_with(Tensor::new(row), |rhs_row, entry| {
                            Tensor::new(rhs_row) * Tensor::new(entry)
                        })
                        .sum()
                    })
                    .map(|row_tensor| row_tensor.into_inner())
                    .data,
                )
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

pub trait AsMatrix {
    type Matrix;
    fn as_matrix(self) -> Self::Matrix;
}

macro_rules! impl_as_matrix {
    ($outer_n:expr; $inner_n:expr, $inner_nty:ident) => {
        // Convert UniChunked arrays into matrices
        impl<'a, T: Scalar> AsMatrix for UniChunked<&'a [T; $outer_n], $inner_nty> {
            type Matrix = &'a Tensor<[[T; $inner_n]; $outer_n / $inner_n]>;
            #[inline]
            fn as_matrix(self) -> Self::Matrix {
                self.into_arrays().as_tensor()
            }
        }
        impl<'a, T: Scalar> AsMatrix for UniChunked<&'a mut [T; $outer_n], $inner_nty> {
            type Matrix = &'a mut Tensor<[[T; $inner_n]; $outer_n / $inner_n]>;
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
impl<T: Scalar + std::ops::Neg<Output = T>> Vector3<T> {
    /// Convert this vector into a skew symmetric matrix, which represents the cross
    /// product operator (when applied to another vector).
    #[inline]
    pub fn skew(&self) -> Matrix3<T> {
        Matrix3::new([
            [T::zero(), -self[2], self[1]],
            [self[2], T::zero(), -self[0]],
            [-self[1], self[0], T::zero()],
        ])
    }
}

/*
 * The determinant is computed recursively using co-factor expansions.
 */

/// Determinant of a 1x1 Matrix.
impl<T: Scalar> Tensor<[[T; 1]; 1]> {
    #[inline]
    pub fn determinant(&self) -> T {
        self.data[0][0]
    }
}

macro_rules! impl_determinant {
    ($n:expr) => {
        /// Determinant of a 2x2 Matrix.
        impl<T: Scalar> Tensor<[[T; $n]; $n]> {
            /// Construct a matrix smaller in both dimensions by 1 that is the same as the
            /// original matrix but without the first row and a given column. Although this is
            /// a very specific function, it is useful for efficient co-factor expansions.
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn without_row_and_first_col(&self, col: usize) -> Tensor<[[T; $n - 1]; $n - 1]> {
                // Ensure that T has the same size as MaybeUninit.
                debug_assert_eq!(
                    std::mem::size_of::<[[MaybeUninit<T>; $n - 1]; $n - 1]>(),
                    std::mem::size_of::<[[T; $n - 1]; $n - 1]>()
                );
                let mut m: [[MaybeUninit<T>; $n - 1]; $n - 1] =
                    unsafe { MaybeUninit::uninit().assume_init() };
                for i in 1..$n {
                    // Transmute to a MaybeUninit slice.
                    let slice = unsafe {
                        std::mem::transmute(&self.data[if i < col + 1 { i - 1 } else { i }][1..$n])
                    };
                    m[i - 1].copy_from_slice(slice);
                }
                // Transmute back to initialized type.
                Tensor::new(unsafe { std::mem::transmute_copy(&m) })
            }

            /// Compute the determinant of the matrix recursively.
            #[inline]
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn determinant(&self) -> T {
                let mut det = self.data[0][0] * self.without_row_and_first_col(0).determinant();
                for row in 1..$n {
                    let cofactor =
                        self.data[row][0] * self.without_row_and_first_col(row).determinant();
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

impl<T: Scalar> Tensor<[[T; 1]; 1]> {
    /// Compute the inverse of a 1x1 matrix.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let denom = self.data[0][0];
        if denom != T::zero() {
            Some(Self::new([[T::one() / denom]]))
        } else {
            None
        }
    }
    /// Invert the 1x1 matrix in place. Return true if inversion was successful.
    #[inline]
    pub fn invert(&mut self) -> bool {
        let denom = self.data[0][0];
        if denom != T::zero() {
            self.data[0][0] = T::one() / denom;
            true
        } else {
            false
        }
    }
}

impl<T: Scalar + Float> Tensor<[[T; 2]; 2]> {
    /// Compute the inverse of a 2x2 matrix.
    #[inline]
    pub fn inverse(&self) -> Option<Tensor<[[T; 2]; 2]>> {
        let det = self.determinant();
        if det != T::zero() {
            Some(Tensor::new([
                [self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det, self.data[0][0] / det],
            ]))
        } else {
            None
        }
    }
    /// Compute the transpose of a 3x3 matrix inverse.
    #[inline]
    pub fn inverse_transpose(&self) -> Option<Self> {
        let det = self.determinant();
        if det != T::zero() {
            Some(Self::new([
                [self.data[1][1] / det, -self.data[1][0] / det],
                [-self.data[0][1] / det, self.data[0][0] / det],
            ]))
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
        if det != T::zero() {
            unsafe {
                let m00 = self.data.get_unchecked_mut(0).get_unchecked_mut(0) as *mut T;
                let m11 = self.data.get_unchecked_mut(1).get_unchecked_mut(1) as *mut T;
                std::ptr::swap(m00, m11);
                *self.data.get_unchecked_mut(0).get_unchecked_mut(1) *= -T::one();
                *self.data.get_unchecked_mut(1).get_unchecked_mut(0) *= -T::one();
            }
            *self /= det;
            true
        } else {
            false
        }
    }
}
impl<T: Scalar + Float> Tensor<[[T; 3]; 3]> {
    /// Compute the inverse of a 3x3 matrix.
    #[inline]
    pub fn inverse(&self) -> Option<Tensor<[[T; 3]; 3]>> {
        self.inverse_transpose().map(|x| x.transpose())
    }
    /// Compute the transpose of a 3x3 matrix inverse.
    #[inline]
    pub fn inverse_transpose(&self) -> Option<Self> {
        let det = self.determinant();
        if det != T::zero() {
            Some(Self::new([
                (Tensor::new(self.data[1]).cross(Tensor::new(self.data[2])) / det).into_inner(),
                (Tensor::new(self.data[2]).cross(Tensor::new(self.data[0])) / det).into_inner(),
                (Tensor::new(self.data[0]).cross(Tensor::new(self.data[1])) / det).into_inner(),
            ]))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_scalar_mul() {
        let mut a = Tensor::new([1, 2, 3, 4]);

        // Right multiply by raw scalar.
        assert_eq!(Tensor::new([3, 6, 9, 12]), a * 3);

        // Right assign multiply by raw scalar.
        a *= 2;
        assert_eq!(Tensor::new([2, 4, 6, 8]), a);
    }

    #[test]
    fn vector_scalar_div() {
        let mut a = Tensor::new([1.0, 2.0, 4.0, 8.0]);

        // Right divide by raw scalar.
        assert_eq!(Tensor::new([0.5, 1.0, 2.0, 4.0]), a / 2.0);

        // Right assign divide by raw scalar.
        a /= 2.0;
        assert_eq!(Tensor::new([0.5, 1.0, 2.0, 4.0]), a);
    }
}
