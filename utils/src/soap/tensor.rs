use super::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A generic type that accepts algebraic expressions.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Tensor<T: ?Sized>(pub T);

impl<T: Set> Set for Tensor<T> {
    type Elem = T::Elem;
    fn len(&self) -> usize {
        self.0.len()
    }
}

/*
 * Implement 1-tensor algebra.
 */

/*
 * Tensor addition and subtraction.
 */

impl<T: Add<Output = T> + Copy> Add for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensor slices together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(Tensor(vec![6,8,10,12]), Tensor(a.view()) + Tensor(b.view()));
    /// ```
    fn add(self, other: Self) -> Self::Output {
        assert_eq!(other.len(), self.len());
        Tensor(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(&a, &b)| a + b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<T: AddAssign + Copy> AddAssign<Tensor<&[T]>> for Tensor<&mut [T]> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut view = Tensor(a.view_mut());
    /// view += Tensor(b.view());
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: Tensor<&[T]>) {
        assert_eq!(other.len(), self.len());
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += b;
        }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract a tensor slice from another into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(Tensor(vec![4,4,4,4]), Tensor(b.view()) - Tensor(a.view()));
    /// ```
    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(other.len(), self.len());
        Tensor(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>(),
        )
    }
}

impl<T: SubAssign + Copy> SubAssign<Tensor<&[T]>> for Tensor<&mut [T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// let mut view = Tensor(b.view_mut());
    /// view -= Tensor(a.view());
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    fn sub_assign(&mut self, other: Tensor<&[T]>) {
        assert_eq!(other.len(), self.len());
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a -= b;
        }
    }
}

/*
 * Scalar multiplication
 */

impl<T: Mul<Output = T> + Copy> Mul<T> for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Multiply a tensor slice by a scalar producing a new `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// assert_eq!(Tensor(vec![3,6,9,12]), Tensor(a.view()) * 3);
    /// ```
    fn mul(self, other: T) -> Self::Output {
        Tensor(self.0.iter().map(|&a| a * other).collect::<Vec<_>>())
    }
}

impl<T: MulAssign + Copy> MulAssign<T> for Tensor<&mut [T]> {
    /// Multiply this tensor slice by a scalar.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let mut view = Tensor(a.view_mut());
    /// view *= 3;
    /// assert_eq!(vec![3,6,9,12], a);
    /// ```
    fn mul_assign(&mut self, other: T) {
        for a in self.0.iter_mut() {
            *a *= other;
        }
    }
}

// TODO: Figure out how to do tensor contractions.
/*
 * Outer product.
 * Note that 1-Tensors are considered as column vectors.
 * Row vectors are `Chunked1` flat collections.
 */
