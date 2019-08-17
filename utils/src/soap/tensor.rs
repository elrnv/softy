use super::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A generic type that accepts algebraic expressions.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Tensor<T: ?Sized>(pub T);

impl<T: Set> Set for Tensor<T> {
    type Elem = T::Elem;
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// This helper function interprets a data type `T` as a tensor. Perhaps at a
/// later point this can be replaced by aptly designed traits for composable
/// collection types.
pub(crate) fn as_tensor<T>(c: &mut T) -> &mut Tensor<T> {
    unsafe { &mut *(c as *mut T as *mut Tensor<T>) }
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

/*
 * All additions and subtractions on 1-tensors represented by chunked vectors can be performed at the lowest level (flat)
 */

macro_rules! impl_chunked_tensor_arithmetic {
    ($chunked:ident) => {
        impl<S, O> Add for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            S: ToOwnedData,
            Tensor<S>: Add<Output = Tensor<S::OwnedData>>,
        {
            type Output = Tensor<$chunked<S::OwnedData, O>>;
            fn add(self, other: Self) -> Self::Output {
                assert_eq!(self.0.len(), other.0.len());
                let $chunked { chunks, data } = self.0;
                Tensor($chunked {
                    chunks,
                    data: (Tensor(data) + Tensor(other.0.data)).0,
                })
            }
        }

        impl<S, O> Sub for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            S: ToOwnedData,
            Tensor<S>: Sub<Output = Tensor<S::OwnedData>>,
        {
            type Output = Tensor<$chunked<S::OwnedData, O>>;

            /// Subtract a tensor of chunked from another.
            fn sub(self, other: Self) -> Self::Output {
                assert_eq!(self.0.len(), other.0.len());
                let $chunked { chunks, data } = self.0;
                Tensor($chunked {
                    chunks,
                    data: (Tensor(data) - Tensor(other.0.data)).0,
                })
            }
        }

        /*
         * Scalar multiplication
         */

        impl<S, O, T> Mul<T> for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            S: ToOwnedData,
            Tensor<S>: Mul<T, Output = Tensor<S::OwnedData>>,
        {
            type Output = Tensor<$chunked<S::OwnedData, O>>;

            fn mul(self, other: T) -> Self::Output {
                let $chunked { chunks, data } = self.0;
                Tensor($chunked {
                    chunks,
                    data: (Tensor(data) * other).0,
                })
            }
        }

        /*
         * Add/Sub/Mul assign variants of the above operators.
         */

        impl<S, T, O, I> AddAssign<Tensor<$chunked<T, I>>> for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            $chunked<T, I>: Set,
            Tensor<S>: AddAssign<Tensor<T>>,
        {
            fn add_assign(&mut self, other: Tensor<$chunked<T, I>>) {
                assert_eq!(self.0.len(), other.0.len());
                let tensor = as_tensor(&mut self.0.data);
                *tensor += Tensor(other.0.data);
            }
        }

        impl<S, T, O, I> SubAssign<Tensor<$chunked<T, I>>> for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            $chunked<T, I>: Set,
            Tensor<S>: SubAssign<Tensor<T>>,
        {
            fn sub_assign(&mut self, other: Tensor<$chunked<T, I>>) {
                assert_eq!(self.0.len(), other.0.len());
                let tensor = as_tensor(&mut self.0.data);
                *tensor -= Tensor(other.0.data);
            }
        }

        impl<S, O, T> MulAssign<T> for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            Tensor<S>: MulAssign<T>,
        {
            fn mul_assign(&mut self, other: T) {
                let tensor = as_tensor(&mut self.0.data);
                *tensor *= other;
            }
        }
    };
}

impl_chunked_tensor_arithmetic!(Chunked);
impl_chunked_tensor_arithmetic!(UniChunked);

// TODO: Figure out how to do tensor contractions.
/*
 * Outer product.
 * Note that 1-Tensors are considered as column vectors.
 * Row vectors are `Chunked1` flat collections.
 */

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_chunked() {
        let offsets = [0, 3, 4];
        let mut a = Chunked::from_offsets(&offsets[..], vec![1, 2, 3, 4]);
        let b = Chunked::from_offsets(&offsets[..], vec![5, 6, 7, 8]);

        // Add
        let res = Chunked::from_offsets(&offsets[..], vec![6, 8, 10, 12]);
        assert_eq!(Tensor(res.clone()), Tensor(a.view()) + Tensor(b.view()));

        // AddAssign
        let mut tensor_a = Tensor(a.view_mut());
        tensor_a += Tensor(b.view());
        assert_eq!(res.view(), a.view());

        // MulAssign
        let res = Chunked::from_offsets(&offsets[..], vec![600, 800, 1000, 1200]);
        let mut tensor_a = Tensor(a.view_mut());
        tensor_a *= 100;
        assert_eq!(res.view(), a.view());

        // SubAssign
        let res = Chunked::from_offsets(&offsets[..], vec![595, 794, 993, 1192]);
        let mut tensor_a = Tensor(a.view_mut());
        tensor_a -= Tensor(b.view());
        assert_eq!(res.view(), a.view());
    }

    #[test]
    fn tensor_uni_chunked() {
        let a = Chunked2::from_flat(vec![1, 2, 3, 4]);
        let b = Chunked2::from_flat(vec![5, 6, 7, 8]);
        assert_eq!(
            Tensor(Chunked2::from_flat(vec![6, 8, 10, 12])),
            Tensor(a.view()) + Tensor(b.view())
        );
    }
}
