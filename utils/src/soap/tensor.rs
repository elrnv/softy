use super::*;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use unroll::unroll_for_loops;

/// A generic type that accepts algebraic expressions.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Tensor<T: ?Sized, I = ()> {
    index: PhantomData<I>,
    pub data: T,
}

impl<T, I> Tensor<T, I> {
    pub fn new(data: T) -> Tensor<T, I> {
        Tensor {
            data,
            index: PhantomData,
        }
    }

    /// Create a reference to the given type as a `Tensor`.
    pub fn as_ref(c: &T) -> &Tensor<T, I> {
        unsafe { &*(c as *const T as *const Tensor<T, I>) }
    }
    /// Same as `as_ref` but creates a mutable reference to the given type as a `Tensor`.
    pub fn as_mut(c: &mut T) -> &mut Tensor<T, I> {
        unsafe { &mut *(c as *mut T as *mut Tensor<T, I>) }
    }
}

impl<T, I> std::ops::Deref for Tensor<T, I> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, I> std::ops::DerefMut for Tensor<T, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/*
 * Tensor as a Set
 */

impl<T: Set> Set for Tensor<T> {
    type Elem = T::Elem;
    fn len(&self) -> usize {
        self.data.len()
    }
}

/*
 * View Impls
 */

impl<T: Viewed> Viewed for Tensor<T> {}

impl<'a, T: View<'a>> View<'a> for Tensor<T> {
    type Type = Tensor<T::Type>;
    fn view(&'a self) -> Self::Type {
        Tensor::new(self.data.view())
    }
}

impl<'a, T: ViewMut<'a>> ViewMut<'a> for Tensor<T> {
    type Type = Tensor<T::Type>;
    fn view_mut(&'a mut self) -> Self::Type {
        Tensor::new(self.data.view_mut())
    }
}

impl<T, I, J> Tensor<T, (I, J)> {
    pub fn transpose(self) -> Tensor<T, (J, I)> {
        Tensor::<_, (J, I)>::new(self.data)
    }
}

/*
 * Implement 0-tensor algebra.
 */

impl<T: MulAssign> Mul for Tensor<T> {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
        self *= rhs;
        self
    }
}

impl<T: MulAssign> MulAssign for Tensor<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.data *= rhs.data;
    }
}

/*
 * Implement 1-tensor algebra.
 */

/*
 * Tensor addition and subtraction.
 */

macro_rules! impl_array_tensors {
    ($nty:ident; $n:expr) => {
        impl<T: AddAssign + Copy> Add for Tensor<[T; $n]> {
            type Output = Self;

            /// Add two tensor arrays together.
            fn add(mut self, rhs: Self) -> Self::Output {
                self += rhs;
                self
            }
        }

        impl<T: AddAssign + Copy> AddAssign for Tensor<[T; $n]> {
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: SubAssign + Copy> Sub for Tensor<[T; $n]> {
            type Output = Self;

            fn sub(mut self, rhs: Self) -> Self::Output {
                self -= rhs;
                self
            }
        }

        impl<T: SubAssign + Copy> SubAssign for Tensor<[T; $n]> {
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        // Right multiply by a tensor with one less degree than Self.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T> Mul<Tensor<T>> for Tensor<[T; $n]>
        where
            Tensor<T>: MulAssign + Clone,
        {
            type Output = Self;
            fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T> MulAssign<Tensor<T>> for Tensor<[T; $n]>
        where
            Tensor<T>: MulAssign + Clone,
        {
            //#[unroll_for_loops]
            fn mul_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    let lhs = Tensor::as_mut(self.data.index_mut(i));
                    *lhs *= rhs.clone();
                }
            }
        }
    };
}

impl_array_tensors!(U1; 1);
impl_array_tensors!(U2; 2);
impl_array_tensors!(U3; 3);
impl_array_tensors!(U4; 4);

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
    /// assert_eq!(Tensor::new(vec![6,8,10,12]), Tensor::new(a.view()) + Tensor::new(b.view()));
    /// ```
    fn add(self, other: Self) -> Self::Output {
        assert_eq!(other.len(), self.len());
        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
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
    /// let mut view = Tensor::new(a.view_mut());
    /// view += Tensor::new(b.view());
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: Tensor<&[T]>) {
        assert_eq!(other.len(), self.len());
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
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
    /// assert_eq!(Tensor::new(vec![4,4,4,4]), Tensor::new(b.view()) - Tensor::new(a.view()));
    /// ```
    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(other.len(), self.len());
        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
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
    /// let mut view = Tensor::new(b.view_mut());
    /// view -= Tensor::new(a.view());
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    fn sub_assign(&mut self, other: Tensor<&[T]>) {
        assert_eq!(other.len(), self.len());
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
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
    /// assert_eq!(Tensor::new(vec![3,6,9,12]), Tensor::new(a.view()) * 3);
    /// ```
    fn mul(self, other: T) -> Self::Output {
        Tensor::new(self.data.iter().map(|&a| a * other).collect::<Vec<_>>())
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
    /// let mut view = Tensor::new(a.view_mut());
    /// view *= 3;
    /// assert_eq!(vec![3,6,9,12], a);
    /// ```
    fn mul_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a *= other;
        }
    }
}

//struct AddOp<L, R> {
//    lhs: L,
//    rhs: R,
//}
//
//impl AddOp<L, R>
//    where L: Add<R>
//{
//    fn eval() ->
//}
//
//struct BinaryOp<Op, Lhs, Rhs> {
//    lhs: Lhs,
//    rhs: Rhs,
//    op: Op,
//}

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
                assert_eq!(self.data.len(), other.data.len());
                let $chunked { chunks, data } = self.data;

                Tensor::new($chunked {
                    chunks,
                    data: (Tensor::new(data) + Tensor::new(other.data.data)).data,
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
                assert_eq!(self.data.len(), other.data.len());
                let $chunked { chunks, data } = self.data;
                Tensor::new($chunked {
                    chunks,
                    data: (Tensor::new(data) - Tensor::new(other.data.data)).data,
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
                let $chunked { chunks, data } = self.data;
                Tensor::new($chunked {
                    chunks,
                    data: (Tensor::new(data) * other).data,
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
                assert_eq!(self.data.len(), other.data.len());
                let tensor = Tensor::as_mut(&mut self.data.data);
                *tensor += Tensor::new(other.data.data);
            }
        }

        impl<S, T, O, I> SubAssign<Tensor<$chunked<T, I>>> for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            $chunked<T, I>: Set,
            Tensor<S>: SubAssign<Tensor<T>>,
        {
            fn sub_assign(&mut self, other: Tensor<$chunked<T, I>>) {
                assert_eq!(self.data.len(), other.data.len());
                let tensor = Tensor::as_mut(&mut self.data.data);
                *tensor -= Tensor::new(other.data.data);
            }
        }

        impl<S, O, T> MulAssign<T> for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            Tensor<S>: MulAssign<T>,
        {
            fn mul_assign(&mut self, other: T) {
                let tensor = Tensor::as_mut(&mut self.data.data);
                *tensor *= other;
            }
        }
    };
}

impl_chunked_tensor_arithmetic!(Chunked);
impl_chunked_tensor_arithmetic!(UniChunked);

// TODO: Generalize this operation
impl<T, N, I> SubAssign<Tensor<Subset<UniChunked<&[T], U<N>>, I>>>
    for Tensor<UniChunked<&mut [T], U<N>>>
where
    I: AsRef<[usize]>,
    N: Unsigned + Copy + Array<T>,
    <N as Array<T>>::Array: Copy,
    Tensor<<N as Array<T>>::Array>: SubAssign,
{
    fn sub_assign(&mut self, other: Tensor<Subset<UniChunked<&[T], U<N>>, I>>) {
        assert_eq!(self.len(), other.len());
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            let lhs_tensor = Tensor::as_mut(lhs);
            *lhs_tensor -= Tensor::new(*rhs);
        }
    }
}

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
        assert_eq!(
            Tensor::new(res.clone()),
            Tensor::new(a.view()) + Tensor::new(b.view())
        );

        // AddAssign
        let mut tensor_a = Tensor::new(a.view_mut());
        tensor_a += Tensor::new(b.view());
        assert_eq!(res.view(), a.view());

        // MulAssign
        let res = Chunked::from_offsets(&offsets[..], vec![600, 800, 1000, 1200]);
        let mut tensor_a = Tensor::new(a.view_mut());
        tensor_a *= 100;
        assert_eq!(res.view(), a.view());

        // SubAssign
        let res = Chunked::from_offsets(&offsets[..], vec![595, 794, 993, 1192]);
        let mut tensor_a = Tensor::new(a.view_mut());
        tensor_a -= Tensor::new(b.view());
        assert_eq!(res.view(), a.view());
    }

    #[test]
    fn tensor_uni_chunked() {
        let a = Chunked2::from_flat(vec![1, 2, 3, 4]);
        let b = Chunked2::from_flat(vec![5, 6, 7, 8]);
        assert_eq!(
            Tensor::new(Chunked2::from_flat(vec![6, 8, 10, 12])),
            Tensor::new(a.view()) + Tensor::new(b.view())
        );
    }

    #[test]
    fn tensor_subset_sub_assign() {
        let a = Subset::from_unique_ordered_indices(
            vec![1, 3],
            Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        );
        let mut b = Chunked2::from_flat(vec![9, 10, 13, 14]);
        let mut b_tensor = Tensor::new(b.view_mut());
        let a_tensor = Tensor::new(a.view());
        SubAssign::sub_assign(&mut b_tensor, a_tensor);
        assert_eq!(b.view().at(0), &[6, 6]);
        assert_eq!(b.view().at(1), &[6, 6]);
    }

    #[test]
    fn small_tensor_add() {
        let a = Tensor::new([1, 2, 3, 4]);
        let b = Tensor::new([5, 6, 7, 8]);
        assert_eq!(Tensor::new([6, 8, 10, 12]), a + b);

        let mut c = Tensor::new([0, 1, 2, 3]);
        c += a;
        assert_eq!(Tensor::new([1, 3, 5, 7]), c);
    }

    #[test]
    fn small_tensor_sub() {
        let a = Tensor::new([1, 2, 3, 4]);
        let b = Tensor::new([5, 6, 7, 8]);
        assert_eq!(Tensor::new([4, 4, 4, 4]), b - a);

        let mut c = Tensor::new([1, 3, 5, 7]);
        c -= a;
        assert_eq!(Tensor::new([0, 1, 2, 3]), c);
    }

    #[test]
    fn small_tensor_scalar_mul() {
        let mut a = Tensor::new([1, 2, 3, 4]);

        // Right multiply by wrapped scalar.
        assert_eq!(Tensor::new([3, 6, 9, 12]), a * Tensor::new(3));

        // Right assign multiply by wrapped scalar.
        a *= Tensor::new(2);
        assert_eq!(Tensor::new([2, 4, 6, 8]), a);
    }
}
