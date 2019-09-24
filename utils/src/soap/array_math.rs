//!
//! This module implements arithmetic on vectors and matrices of statically sized Rust arrays.
//! The types defined in this module make arithmetic between vectors and matrices less verbose as
//! it would otherwise be if using raw Tensors.
//!
use super::*;
use num_traits::{Float, NumAssign, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign};
use unroll::unroll_for_loops;

pub trait Scalar: Copy + Clone + std::fmt::Debug + PartialOrd + NumAssign {}
impl<T> Scalar for T where T: Copy + Clone + std::fmt::Debug + PartialOrd + NumAssign {}

macro_rules! impl_array_vectors {
    ($vecn:ident, $rowvecn:ident; $n:expr) => {
        pub type $vecn<T, I = ()> = Tensor<[T; $n], I>;
        pub type $rowvecn<T, I = ()> = Tensor<[[T; $n]; 1], I>;

        impl<T: Scalar> Matrix for $vecn<T> {
            type Transpose = $rowvecn<T, ((), ())>;
            fn transpose(self) -> Self::Transpose {
                Tensor::new([self.data; 1])
            }
            fn num_rows(&self) -> usize {
                $n
            }
            fn num_cols(&self) -> usize {
                1
            }
        }

        impl<T: Scalar, I> Tensor<[T; $n], I> {
            #[unroll_for_loops]
            pub fn map<U, F>(&self, mut f: F) -> Tensor<[U; $n], I>
            where
                U: Scalar,
                F: FnMut(T) -> U,
            {
                let mut out = Tensor::<[U; $n], I>::zeros();
                for i in 0..$n {
                    out[i] = f(self.data[i]);
                }
                out
            }
        }

        impl<T: Scalar, I> Tensor<[T; $n], I> {
            pub fn zeros() -> Tensor<[T; $n], I> {
                Tensor::new([T::zero(); $n])
            }
        }

        impl<T: Scalar, I> Tensor<[T; $n], I> {
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

        impl<T: Scalar, I> Tensor<[T; $n], I> {
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn dot(&self, other: Tensor<[T; $n], I>) -> T {
                let mut prod = self.data[0] * other.data[0];
                for i in 1..$n {
                    prod += self.data[i] * other.data[i];
                }
                prod
            }
        }

        impl<T: Scalar, I> Tensor<[T; $n], I> {
            pub fn sum(&self) -> T {
                self.fold(T::zero(), |acc, x| x + acc)
            }
        }

        impl<T: Scalar, I> Tensor<[T; $n], I> {
            pub fn norm_squared(&self) -> T {
                (*self).map(|x| x * x).sum()
            }
        }

        impl<T: Float + Scalar, I> Tensor<[T; $n], I> {
            pub fn norm(&self) -> T {
                self.norm_squared().sqrt()
            }
        }

        // Right scalar multiply by a raw scalar.
        impl<T: Scalar, I> Mul<T> for Tensor<[T; $n], I>
        where
            Tensor<[T; $n], I>: MulAssign<T>,
        {
            type Output = Self;
            fn mul(mut self, rhs: T) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T: Scalar, I> MulAssign<T> for Tensor<[T; $n], I>
        where
            Tensor<[T; $n], I>: MulAssign<Tensor<T>>,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs);
            }
        }

        impl<T: Scalar, I> Div<T> for Tensor<[T; $n], I>
        where
            Tensor<[T; $n], I>: DivAssign<T>,
        {
            type Output = Self;
            fn div(mut self, rhs: T) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl<T: Scalar, I> DivAssign<T> for Tensor<[T; $n], I>
        where
            Tensor<[T; $n], I>: DivAssign<Tensor<T>>,
        {
            fn div_assign(&mut self, rhs: T) {
                *self /= Tensor::new(rhs);
            }
        }

        impl<T: Scalar, I> Index<usize> for Tensor<[T; $n], I> {
            type Output = T;
            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }
        impl<T: Scalar, I> IndexMut<usize> for Tensor<[T; $n], I> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.data[index]
            }
        }

        // TODO: Figure out why this doesn't compile
        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        impl<T: Scalar> Mul<Tensor<[T; $n]>> for Tensor<T>
            where Tensor<[T; $n]>: MulAssign<Tensor<T>>
        {
            type Output = Tensor<[T; $n]>;
            fn mul(self, mut rhs: Tensor<[T; $n]>) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
    };
}

impl_array_vectors!(Vector1, RowVector1; 1);
impl_array_vectors!(Vector2, RowVector2; 2);
impl_array_vectors!(Vector3, RowVector3; 3);
impl_array_vectors!(Vector4, RowVector4; 4);

/*
macro_rules! impl_array_matrices {
    ($mtxn:ident; $r:expr, $c:expr) => {
        // Row-major square matrix.
        pub type $mtxn<T, I = (), J = ()> = Tensor<[[T; $c]; $r], (I, J)>;

        // Transposes of small matrices are implemented eagerly.
        impl<T: Zero + Copy, I, J> Matrix for Tensor<[[T; $c]; $r], (I, J)> {
            type Transpose = Tensor<[[T; $r]; $c], (J, I)>;
            //#[unroll_for_loops]
            fn transpose(self) -> Self::Transpose {
                let mut m = $mtxn::zeros();

                for row in 0..$r {
                    for col in 0..$c {
                        m[col][row] = self[row][col];
                    }
                }
                m
            }
            fn num_rows(&self) -> usize {
                $r
            }
            fn num_cols(&self) -> usize {
                $c
            }
        }

        impl<T: Copy, I> Tensor<[[T; $c]; $r], I> {
            /// Similar to `map` but applies the given function to each inner element.
            //#[unroll_for_loops]
            pub fn map_inner<U, F>(&self, mut f: F) -> Tensor<[[U; $c]; $r], I>
                where U: Zero + Copy,
                      F: FnMut(T) -> U,
            {
                let mut out = Tensor::zeros();
                for row in 0..$r {
                    out[row] = self[row].map(|x| f(x));
                }
                out
            }
        }

        impl<T: Zero + Copy, I> Tensor<[[T; $c]; $r], I> {
            pub fn identity() -> Tensor<[[T; $c]; $r], I> {
                Self::from_diag_iter(std::iter::repeat(T::one()))
            }
            pub fn from_diag_iter<Iter: IntoIterator<Item = T>>(diag: Iter) -> Tensor<[[T; $c]; $r], I> {
                let mut out = Self::zeros();
                for (i, elem) in diag.take($r.min($c)).enumerate() {
                    out[i][i] = elem;
                }
                out
            }
            //pub fn zeros() -> Tensor<[[T; $c]; $r], I> {
            //    Tensor::new([[T::zero(); $c]; $r])
            //}
        }

        impl<T: Copy, I> Tensor<[[T; $c]; $r], I> {
            //#[unroll_for_loops]
            pub fn fold_inner<B, F>(&self, mut init: B, mut f: F) -> B
                where F: FnMut(B, T) -> B,
            {
                for i in 0..$r {
                    init = self[i].fold(init, |acc, x| f(acc, x));
                }
                init
            }
        }

        impl<T: Add<Output = T>, I> Tensor<[[T; $c]; $r], I> {
            pub fn trace(&self) -> T {
                let mut tr = self[0][0];
                for i in 1..$r.min($c) {
                    tr += self[i][i];
                }
                tr
            }
        }

        impl<T: Mul<Output = T> + Add<Output = T>, I> Tensor<[[T; $c]; $r], I> {
            pub fn frob_norm_squared(&self) -> T {
                (*self).map_inner(|x| x*x).sum()
            }
        }

        impl<T: Float, I> Tensor<[[T; $c]; $r], I> {
            pub fn frob_norm(&self) -> T {
                self.frob_norm_squared().sqrt()
            }
        }

        // Right scalar multiply by a scalar.
        impl<T, I> Mul<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: MulAssign<T>,
        {
            type Output = Self;
            fn mul(mut self, rhs: T) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T, I> MulAssign<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: MulAssign<Tensor<T>>,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs);
            }
        }

        impl<T, I> Div<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: DivAssign<T>,
        {
            type Output = Self;
            fn div(mut self, rhs: T) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl<T, I> DivAssign<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: DivAssign<Tensor<T>>,
        {
            fn div_assign(&mut self, rhs: T) {
                *self /= Tensor::new(rhs);
            }
        }
    };
}

//impl_array_matrices!(Matrix1; 1, 1);
impl_array_matrices!(Matrix2; 2, 2);
//impl_array_matrices!(Matrix3; 3, 3);
//impl_array_matrices!(Matrix4; 4, 4);
//
//// Common Rectangular matrices
//impl_array_matrices!(Matrix3x4; 3, 4);
//impl_array_matrices!(Matrix4x3; 4, 3);
//impl_array_matrices!(Matrix2x4; 2, 4);
//impl_array_matrices!(Matrix4x2; 4, 2);
//impl_array_matrices!(Matrix2x3; 2, 3);
//impl_array_matrices!(Matrix3x2; 3, 2);

*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_scalar_mul() {
        let mut a = Tensor::flat([1, 2, 3, 4]);

        // Right multiply by raw scalar.
        assert_eq!(Tensor::new([3, 6, 9, 12]), a * 3);

        // Right assign multiply by raw scalar.
        a *= 2;
        assert_eq!(Tensor::flat([2, 4, 6, 8]), a);
    }

    #[test]
    fn vector_scalar_div() {
        let mut a = Tensor::flat([1.0, 2.0, 4.0, 8.0]);

        // Right divide by raw scalar.
        assert_eq!(Tensor::new([0.5, 1.0, 2.0, 4.0]), a / 2.0);

        // Right assign divide by raw scalar.
        a /= 2.0;
        assert_eq!(Tensor::flat([0.5, 1.0, 2.0, 4.0]), a);
    }
}
