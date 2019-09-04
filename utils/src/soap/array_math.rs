//!
//! This module implements arithmetic on vectors and matrices of statically sized Rust arrays.
//! The types defined in this module make arithmetic between vectors and matrices less verbose as
//! it would otherwise be if using raw Tensors.
//!
use super::*;
use std::ops::{Mul, MulAssign};

macro_rules! impl_array_vectors {
    ($vecn:ident; $n:expr) => {
        pub type $vecn<T> = Tensor<[T; $n]>;

        // Right scalar multiply by a raw scalar.
        impl<T: MulAssign> Mul<T> for Tensor<[T; $n]>
        where
            Tensor<T>: MulAssign + Clone,
        {
            type Output = Self;
            fn mul(mut self, rhs: T) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T: MulAssign> MulAssign<T> for Tensor<[T; $n]>
        where
            Tensor<T>: MulAssign + Clone,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs)
            }
        }

        // TODO: Figure out why this doesn't compile
        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        //impl<T: Real> Mul<Tensor<[T; $n]>> for Tensor<T>
        //    where Tensor<[T; $n]>: MulAssign<Tensor<T>>
        //{
        //    type Output = Tensor<[T; $n]>;
        //    fn mul(self, mut rhs: Tensor<[T; $n]>) -> Self::Output {
        //        //rhs *= self;
        //        rhs
        //    }
        //}
    };
}

impl_array_vectors!(Vector1; 1);
impl_array_vectors!(Vector2; 2);
impl_array_vectors!(Vector3; 3);
impl_array_vectors!(Vector4; 4);

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
}
