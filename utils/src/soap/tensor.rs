use super::*;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use unroll::unroll_for_loops;

/// A generic type that accepts algebraic expressions.
///
/// The type parameter `I` determines the indexing structure of the tensor.
/// For instance for unique types `I0` and `I1`, the type `I == (I0, I1)` represents a matrix with
/// outer index `I0` and inner index `I1`. This means that a transpose can be implemented simply by
/// swapping positions of `I0` and `I1`, which means a matrix with `I == (I1, I0)` has structure
/// that is transpose of the matix with `I = (I0, I1)`.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Tensor<T: ?Sized, I = ()> {
    index: PhantomData<I>,
    pub data: T,
}

impl<T> Tensor<T> {
    /// Construct a tensor without any indexing information.
    /// This is convenient for one-off computations where the indexing structure is not important.
    /// In other words, this creates a shallow or `flat` tensor without information about the
    /// deeper structures.
    pub fn flat(data: T) -> Tensor<T> {
        Tensor {
            data,
            index: PhantomData,
        }
    }
}

/// Synonymous with `AsRef<Tensor<_>>`.
pub trait AsTensor {
    fn as_tensor<I>(&self) -> &Tensor<Self, I>;
}
pub trait AsMutTensor {
    fn as_mut_tensor<I>(&mut self) -> &mut Tensor<Self, I>;
}

impl<T: ?Sized> AsTensor for T {
    fn as_tensor<I>(&self) -> &Tensor<T, I> {
        Tensor::as_ref(self)
    }
}
impl<T: ?Sized> AsMutTensor for T {
    fn as_mut_tensor<I>(&mut self) -> &mut Tensor<T, I> {
        Tensor::as_mut(self)
    }
}

impl<T, I> Tensor<T, I> {
    pub fn new(data: T) -> Tensor<T, I> {
        Tensor {
            data,
            index: PhantomData,
        }
    }

    pub fn into_inner(self) -> T {
        self.data
    }
}
impl<T: ?Sized, I> Tensor<T, I> {
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

impl<T: Set, I> Set for Tensor<T, I> {
    type Elem = T::Elem;
    fn len(&self) -> usize {
        self.data.len()
    }
}

/*
 * View Impls
 */

impl<T: Viewed, I> Viewed for Tensor<T, I> {}

impl<'a, T: View<'a>, I> View<'a> for Tensor<T, I> {
    type Type = Tensor<T::Type>;
    fn view(&'a self) -> Self::Type {
        Tensor::new(self.data.view())
    }
}

impl<'a, T: ViewMut<'a>, I> ViewMut<'a> for Tensor<T, I> {
    type Type = Tensor<T::Type>;
    fn view_mut(&'a mut self) -> Self::Type {
        Tensor::new(self.data.view_mut())
    }
}

impl<T, I, J> Tensor<T, (I, J)> {
    pub fn transpose(self) -> Tensor<T, (J, I)> {
        Tensor::new(self.data)
    }
}

/*
 * Scalar trait
 */

pub trait Scalar: Copy + Clone + std::fmt::Debug + PartialOrd + num_traits::NumAssign { }
impl<T> Scalar for T where T: Copy + Clone + std::fmt::Debug + PartialOrd + num_traits::NumAssign {}

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

impl<T: DivAssign> Div for Tensor<T> {
    type Output = Self;
    fn div(mut self, rhs: Self) -> Self {
        self /= rhs;
        self
    }
}

impl<T: DivAssign> DivAssign for Tensor<T> {
    fn div_assign(&mut self, rhs: Self) {
        self.data /= rhs.data;
    }
}

/*
 * Implement 1-tensor algebra.
 */

/*
 * Tensor addition and subtraction.
 */

macro_rules! impl_array_tensors {
    ($n:expr) => {
        impl<T> LocalGeneric for &Tensor<[T; $n]> {}

        impl<T: AddAssign + Copy, I> Add for Tensor<[T; $n], I> {
            type Output = Self;

            /// Add two tensor arrays together.
            fn add(mut self, rhs: Self) -> Self::Output {
                self += rhs;
                self
            }
        }

        impl<T: AddAssign + Copy, I> AddAssign for Tensor<[T; $n], I> {
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: AddAssign + Copy, I> AddAssign<&Tensor<[T; $n]>> for Tensor<[T; $n], I> {
            #[unroll_for_loops]
            fn add_assign(&mut self, rhs: &Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        impl<T: SubAssign + Copy, I> Sub for Tensor<[T; $n], I> {
            type Output = Self;

            fn sub(mut self, rhs: Self) -> Self::Output {
                self -= rhs;
                self
            }
        }

        impl<T: SubAssign + Copy, I> SubAssign for Tensor<[T; $n], I> {
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        impl<T: SubAssign + Copy, I> SubAssign<&Tensor<[T; $n]>> for Tensor<[T; $n], I> {
            #[unroll_for_loops]
            fn sub_assign(&mut self, rhs: &Tensor<[T; $n]>) {
                for i in 0..$n {
                    self.data[i] -= rhs.data[i];
                }
            }
        }

        // Right multiply by a tensor with one less degree than Self.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T, I> Mul<Tensor<T>> for Tensor<[T; $n], I>
        where
            Tensor<T>: MulAssign + Clone,
        {
            type Output = Self;
            fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T, I> MulAssign<Tensor<T>> for Tensor<[T; $n], I>
        where
            Tensor<T>: MulAssign + Clone,
        {
            #[unroll_for_loops]
            fn mul_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    let lhs = Tensor::as_mut(self.data.index_mut(i));
                    *lhs *= rhs.clone();
                }
            }
        }

        impl<T, I> Div<Tensor<T>> for Tensor<[T; $n], I>
        where
            Tensor<T>: DivAssign + Clone,
        {
            type Output = Self;
            fn div(mut self, rhs: Tensor<T>) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl<T, I> DivAssign<Tensor<T>> for Tensor<[T; $n], I>
        where
            Tensor<T>: DivAssign + Clone,
        {
            #[unroll_for_loops]
            fn div_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    let lhs = Tensor::as_mut(self.data.index_mut(i));
                    *lhs /= rhs.clone();
                }
            }
        }

        impl<T: Neg<Output = T> + Copy, I> Neg for Tensor<[T; $n], I> {
            type Output = Self;
            #[unroll_for_loops]
            fn neg(mut self) -> Self::Output {
                for i in 0..$n {
                    self.data[i] = -self.data[i];
                }
                self
            }
        }
    };
}

impl_array_tensors!(1);
impl_array_tensors!(2);
impl_array_tensors!(3);
impl_array_tensors!(4);

macro_rules! impl_slice_add {
    ($other:ty) => {
        fn add(self, other: $other) -> Self::Output {
            assert_eq!(other.data.len(), self.data.len());
            Tensor::new(
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>(),
            )
        }
    }
}

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
    /// assert_eq!(
    ///     Tensor::new(vec![6,8,10,12]),
    ///     Tensor::new(a.view()) + Tensor::new(b.view())
    /// );
    /// ```
    impl_slice_add!(Self);
}

impl<T: Add<Output = T> + Copy> Add for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensor slices together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     [6,8,10,12].as_tensor(),
    ///     a.view().as_tensor() + b.view().as_tensor())
    /// );
    /// ```
    impl_slice_add!(Self);
}

impl<T: Add<Output = T> + Copy> Add<Tensor<&[T]>> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensor slices together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Tensor::new(vec![6,8,10,12]),
    ///     a.view().as_tensor() + Tensor::new(b.view()))
    /// );
    /// ```
    impl_slice_add!(Tensor<&[T]>);
}

impl<T: AddAssign + Copy> Add<Tensor<Vec<T>>> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensors together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Tensor::new(vec![6,8,10,12]),
    ///     a.view().as_tensor() + Tensor::new(b)
    /// );
    /// ```
    fn add(self, mut other: Tensor<Vec<T>>) -> Self::Output {
        other.add_assign(self);
        other
    }
}

impl<T: Add<Output = T> + Copy> Add<&Tensor<[T]>> for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensor slices together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     [6,8,10,12].as_tensor(),
    ///     Tensor::new(a.view()) + b.view().as_tensor())
    /// );
    /// ```
    impl_slice_add!(&Tensor<[T]>);
}

impl<T: AddAssign + Copy> Add<Tensor<Vec<T>>> for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensor slices together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Tensor::new(vec![6,8,10,12]),
    ///     Tensor::new(a.view()) + Tensor::new(b)
    /// );
    /// ```
    fn add(self, mut other: Tensor<Vec<T>>) -> Self::Output {
        other.add_assign(self);
        other
    }
}

impl<T: AddAssign + Copy> Add<&Tensor<[T]>> for Tensor<Vec<T>> {
    type Output = Self;

    /// Add a tensor slice to a tensor `Vec` into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Tensor::new(vec![6,8,10,12]),
    ///     Tensor::new(a) + b.view().as_tensor()
    /// );
    /// ```
    fn add(mut self, other: &Tensor<[T]>) -> Self::Output {
        self.add_assign(other);
        self
    }
}

impl<T: AddAssign + Copy> Add<Tensor<&[T]>> for Tensor<Vec<T>> {
    type Output = Self;

    /// Add a tensor slice to a tensor `Vec` into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Tensor::new(vec![6,8,10,12]),
    ///     Tensor::new(a) + Tensor::new(b.view())
    /// );
    /// ```
    fn add(mut self, other: Tensor<&[T]>) -> Self::Output {
        self.add_assign(other);
        self
    }
}

macro_rules! impl_slice_add_assign {
    ($self:ident, $other:ident) => {{
        assert_eq!($other.data.len(), $self.data.len());
        for (a, &b) in $self.data.iter_mut().zip($other.data.iter()) {
            *a += b;
        }
    }};
}

impl<T: AddAssign + Copy> AddAssign<&Tensor<[T]>> for Tensor<Vec<T>> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut tensor = Tensor::new(a.clone());
    /// tensor += Tensor::new(b.view());
    /// assert_eq!(vec![6,8,10,12], tensor.data);
    /// ```
    fn add_assign(&mut self, other: &Tensor<[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<Tensor<&[T]>> for Tensor<Vec<T>> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut tensor = Tensor::new(a.clone());
    /// tensor += b.view().as_tensor();
    /// assert_eq!(vec![6,8,10,12], tensor.data);
    /// ```
    fn add_assign(&mut self, other: Tensor<&[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<Tensor<Vec<T>>> for Tensor<Vec<T>> {
    /// Add a tensor `Vec` to this tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut tensor = Tensor::new(a.clone());
    /// tensor += Tensor::new(b);
    /// assert_eq!(vec![6,8,10,12], tensor.data);
    /// ```
    fn add_assign(&mut self, other: Tensor<Vec<T>>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<Tensor<Vec<T>>> for Tensor<&mut [T]> {
    /// Add a tensor `Vec` to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut view = Tensor::new(a.view_mut());
    /// view += Tensor::new(b);
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: Tensor<Vec<T>>) {
        impl_slice_add_assign!(self, other);
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
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<&Tensor<[T]>> for Tensor<&mut [T]> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut view = Tensor::new(a.view_mut());
    /// view += Tensor::as_ref(b.view());
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: &Tensor<[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<Tensor<Vec<T>>> for Tensor<[T]> {
    /// Add a tensor `Vec` to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// *a.as_mut_slice().as_mut_tensor() += Tensor::new(b);
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: Tensor<Vec<T>>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<&Tensor<[T]>> for Tensor<[T]> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// *a.as_mut_slice().as_mut_tensor() += Tensor::as_ref(b.as_slice());
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: &Tensor<[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<Tensor<&[T]>> for Tensor<[T]> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// *a.as_mut_slice().as_mut_tensor() += Tensor::new(b.as_slice());
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: Tensor<&[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

macro_rules! impl_slice_sub {
    ($other:ty) => {
        fn sub(self, other: $other) -> Self::Output {
            assert_eq!(other.data.len(), self.data.len());
            Tensor::new(
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&a, &b)| a - b)
                    .collect::<Vec<_>>(),
            )
        }
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract one slice tensor from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Tensor::new(vec![4,4,4,4]),
    ///     Tensor::new(a.view()) - Tensor::new(b.view())
    /// );
    /// ```
    impl_slice_sub!(Self);
}

impl<T: Sub<Output = T> + Copy> Sub for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract one slice tensor from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     [4,4,4,4].as_tensor(),
    ///     a.view().as_tensor() - b.view().as_tensor())
    /// );
    /// ```
    impl_slice_sub!(Self);
}

impl<T: Sub<Output = T> + Copy> Sub<Tensor<&[T]>> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract one slice tensor from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Tensor::new(vec![4,4,4,4]),
    ///     a.view().as_tensor() - Tensor::new(b.view()))
    /// );
    /// ```
    impl_slice_sub!(Tensor<&[T]>);
}

impl<T: Sub<Output = T> + Copy> Sub<Tensor<Vec<T>>> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract a `Vec` tensor from a slice tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Tensor::new(vec![4,4,4,4]),
    ///     a.view().as_tensor() - Tensor::new(b)
    /// );
    /// ```
    fn sub(self, mut other: Tensor<Vec<T>>) -> Self::Output {
        assert_eq!(other.data.len(), self.data.len());
        for (&a, b) in self.data.iter().zip(other.data.iter_mut()) {
            *b = a - *b;
        }
        other
    }
}

impl<T: Sub<Output = T> + Copy> Sub<&Tensor<[T]>> for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract one slice tensor from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     [4,4,4,4].as_tensor(),
    ///     Tensor::new(a.view()) - b.view().as_tensor())
    /// );
    /// ```
    impl_slice_sub!(&Tensor<[T]>);
}

impl<T: Sub<Output = T> + Copy> Sub<Tensor<Vec<T>>> for Tensor<&[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract a `Vec` tensor from a slice tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Tensor::new(vec![4,4,4,4]),
    ///     Tensor::new(a.view()) - Tensor::new(b)
    /// );
    /// ```
    fn sub(self, mut other: Tensor<Vec<T>>) -> Self::Output {
        assert_eq!(other.data.len(), self.data.len());
        for (&a, b) in self.data.iter().zip(other.data.iter_mut()) {
            *b = a - *b;
        }
        other
    }
}

impl<T: SubAssign + Copy> Sub<&Tensor<[T]>> for Tensor<Vec<T>> {
    type Output = Self;

    /// Subtract a tensor slice from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Tensor::new(vec![4,4,4,4]),
    ///     Tensor::new(a) - b.view().as_tensor()
    /// );
    /// ```
    fn sub(mut self, other: &Tensor<[T]>) -> Self::Output {
        self.sub_assign(other);
        self
    }
}

impl<T: SubAssign + Copy> Sub<Tensor<&[T]>> for Tensor<Vec<T>> {
    type Output = Self;

    /// Subtract a slice tensor from a `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Tensor::new(vec![4,4,4,4]),
    ///     Tensor::new(a) - Tensor::new(b.view())
    /// );
    /// ```
    fn sub(mut self, other: Tensor<&[T]>) -> Self::Output {
        self.sub_assign(other);
        self
    }
}

macro_rules! impl_sub_assign {
    ($other:ty) => {
        fn sub_assign(&mut self, other: $other) {
            assert_eq!(other.data.len(), self.data.len());
            for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
                *a -= b;
            }
        }
    }
}

impl<T: SubAssign + Copy> SubAssign<Tensor<&[T]>> for Tensor<Vec<T>> {
    /// Subtract a tensor slice from this `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// let mut tensor = Tensor::new(b);
    /// tensor -= Tensor::new(a.view());
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(Tensor<&[T]>);
}

impl<T: SubAssign + Copy> SubAssign<&Tensor<[T]>> for Tensor<Vec<T>> {
    /// Subtract a tensor slice from this `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// let mut tensor = Tensor::new(b);
    /// tensor -= a.view().as_tensor();
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(&Tensor<[T]>);
}

impl<T: SubAssign + Copy> SubAssign<Tensor<Vec<T>>> for Tensor<&mut [T]> {
    /// Subtract a `Vec` tensor from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// let mut view = Tensor::new(b.view_mut());
    /// view -= Tensor::new(a);
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(Tensor<Vec<T>>);
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
    impl_sub_assign!(Tensor<&[T]>);
}

impl<T: SubAssign + Copy> SubAssign<&Tensor<[T]>> for Tensor<&mut [T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// let mut view = Tensor::new(b.view_mut());
    /// view -= Tensor::as_ref(a.view());
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(&Tensor<[T]>);
}

impl<T: SubAssign + Copy> SubAssign<Tensor<Vec<T>>> for Tensor<[T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// *b.view_mut().as_mut_tensor() -= Tensor::new(a);
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(Tensor<Vec<T>>);
}

impl<T: SubAssign + Copy> SubAssign<&Tensor<[T]>> for Tensor<[T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// *b.view_mut().as_mut_tensor() -= a.view().as_tensor();
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(&Tensor<[T]>);
}

impl<T: SubAssign + Copy> SubAssign<Tensor<&[T]>> for Tensor<[T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// *b.view_mut().as_mut_tensor() -= Tensor::new(a.view().as_tensor();
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    impl_sub_assign!(Tensor<&[T]>);
}

/*
 * Scalar multiplication
 */

impl<T: Scalar> Mul<T> for Tensor<&[T]> {
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

impl<T: Scalar> MulAssign<T> for Tensor<&mut [T]> {
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

impl<T: Scalar> MulAssign<T> for Tensor<[T]> {
    /// Multiply this tensor slice by a scalar.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// *a.view_mut().as_mut_tensor() *= 3;
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

/// A marker trait for local types which use generic implementations of various std::ops traits.
/// Special types which use optimized implementations will not implement this marker. This is a
/// workaround for specialization.
pub trait LocalGeneric {}

impl<'a, S> LocalGeneric for SubsetView<'a, S> {}
impl<S, N> LocalGeneric for UniChunked<S, N> {}
impl<S, O> LocalGeneric for Chunked<S, O> {}
impl<T> LocalGeneric for Tensor<T> {}

impl<T: ?Sized, U, V: ?Sized> AddAssign<Tensor<U>> for Tensor<V>
where
    V: LocalGeneric + Set + for<'b> ViewMutIterator<'b, Item = &'b mut T>,
    U: LocalGeneric + Set + for<'c> ViewIterator<'c, Item = &'c T>,
    Tensor<T>: for<'a> AddAssign<&'a Tensor<T>>,
{
    fn add_assign(&mut self, other: Tensor<U>) {
        for (out, b) in self.data.view_mut_iter().zip(other.view_iter()) {
            let out_tensor = Tensor::as_mut(out);
            *out_tensor += Tensor::as_ref(b);
        }
    }
}

impl<T: ?Sized, U, V: ?Sized> SubAssign<Tensor<U>> for Tensor<V>
where
    V: LocalGeneric + Set + for<'b> ViewMutIterator<'b, Item = &'b mut T>,
    U: LocalGeneric + Set + for<'c> ViewIterator<'c, Item = &'c T>,
    Tensor<T>: for<'a> SubAssign<&'a Tensor<T>>,
{
    fn sub_assign(&mut self, other: Tensor<U>) {
        for (out, b) in self.data.view_mut_iter().zip(other.view_iter()) {
            let out_tensor = Tensor::as_mut(out);
            *out_tensor -= Tensor::as_ref(b);
        }
    }
}

macro_rules! impl_chunked_tensor_arithmetic {
    ($chunked:ident) => {
        impl<S, O> Add for Tensor<$chunked<S, O>>
        where
            $chunked<S, O>: Set,
            S: IntoOwnedData,
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
            S: IntoOwnedData,
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
            T: Scalar,
            $chunked<S, O>: Set,
            S: IntoOwnedData,
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

        impl<S, O, T> MulAssign<T> for Tensor<$chunked<S, O>>
        where
            T: Scalar,
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
        SubAssign::sub_assign(&mut tensor_a, Tensor::new(b.view()));
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
    fn tensor_subset_add_assign() {
        let a = Subset::from_unique_ordered_indices(
            vec![1, 3],
            Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]),
        );
        let mut b = Chunked2::from_flat(vec![9, 10, 13, 14]);
        *b.as_mut_tensor() += Tensor::new(a.view());
        assert_eq!(b.view().at(0), &[12, 14]);
        assert_eq!(b.view().at(1), &[20, 22]);
    }

    #[test]
    fn small_tensor_add() {
        let a = Tensor::flat([1, 2, 3, 4]);
        let b = Tensor::flat([5, 6, 7, 8]);
        assert_eq!(Tensor::flat([6, 8, 10, 12]), a + b);

        let mut c = Tensor::flat([0, 1, 2, 3]);
        c += a;
        assert_eq!(Tensor::flat([1, 3, 5, 7]), c);
    }

    #[test]
    fn small_tensor_sub() {
        let a = Tensor::flat([1, 2, 3, 4]);
        let b = Tensor::flat([5, 6, 7, 8]);
        assert_eq!(Tensor::flat([4, 4, 4, 4]), b - a);

        let mut c = Tensor::flat([1, 3, 5, 7]);
        c -= a;
        assert_eq!(Tensor::flat([0, 1, 2, 3]), c);
    }

    #[test]
    fn small_tensor_scalar_mul() {
        let mut a = Tensor::flat([1, 2, 3, 4]);

        // Right multiply by wrapped scalar.
        assert_eq!(Tensor::flat([3, 6, 9, 12]), a * Tensor::flat(3));

        // Right assign multiply by wrapped scalar.
        a *= Tensor::flat(2);
        assert_eq!(Tensor::flat([2, 4, 6, 8]), a);
    }

    // This test demonstrates the different ways to use tensors for assignment ops like AddAssign.
    #[test]
    fn tensor_assign_ops() {
        let mut v0 = vec![1, 2, 3, 4];
        let v1 = vec![2, 3, 4, 5];

        // With RHS being a tensor reference:
        let rhs = v1.as_slice().as_tensor();

        // As transient mutable tensor reference.
        *v0.view_mut().as_mut_tensor() += rhs;
        assert_eq!(v0, vec![3, 5, 7, 9]);

        // As a persistent owned tensor object.
        let mut t0 = Tensor::new(v0.as_mut_slice());
        t0 += rhs;
        assert_eq!(&*t0.data, &[5, 8, 11, 14]);

        // With RHS being a persistent owned tensor object.
        let t1 = Tensor::new(v1.as_slice());
        t0 += t1;
        assert_eq!(&*t0.data, &[7, 11, 15, 19]);

        *v0.view_mut().as_mut_tensor() += t1;
        assert_eq!(v0, vec![9, 14, 19, 24]);
    }

    #[test]
    fn tensor_add() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let res = Tensor::new(vec![6, 8, 10, 12]);
        assert_eq!(res, Tensor::new(a.view()) + Tensor::new(b.view()));
        assert_eq!(res, a.view().as_tensor() + Tensor::new(b.view()));
        assert_eq!(res, Tensor::new(a.view()) + b.view().as_tensor());
        assert_eq!(res, a.view().as_tensor() + b.view().as_tensor());
    }
}
