//! This module implements arithmetic on vectors and matrices of statically sized Rust arrays.
//! The types defined in this module make arithmetic between vectors and matrices less verbose as
//! it would otherwise be if using raw Tensors.
//!
use super::*;
use num_traits::{Float, Zero};
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign};
use unroll::unroll_for_loops;

macro_rules! impl_array_vectors {
    ($vecn:ident, $rowvecn:ident; $n:expr) => {
        pub type $vecn<T, I = ()> = Tensor<[T; $n], I>;
        pub type $rowvecn<T, I = ()> = Tensor<[[T; $n]; 1], ((), I)>;

        impl<T: Scalar, I> Matrix for Tensor<[T; $n], I> {
            type Transpose = $rowvecn<T, ((), I)>;
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
            pub fn zeros() -> Tensor<[T; $n], I> {
                Tensor::new([T::zero(); $n])
            }
            #[allow(unused_mut)]
            #[unroll_for_loops]
            pub fn dot(&self, other: Tensor<[T; $n], I>) -> T {
                let mut prod = self.data[0] * other.data[0];
                for i in 1..$n {
                    prod += self.data[i] * other.data[i];
                }
                prod
            }
            pub fn norm_squared(&self) -> T {
                (*self).map(|x| x * x).sum()
            }
        }

        impl<T: Copy, I> Tensor<[T; $n], I> {
            #[unroll_for_loops]
            pub fn map<U, F>(&self, mut f: F) -> Tensor<[U; $n], I>
            where
                F: FnMut(T) -> U,
            {
                // We use MaybeUninit here mostly to avoid a Zero trait bound.
                let mut out: [MaybeUninit<U>; $n] = unsafe { MaybeUninit::uninit().assume_init() };
                for i in 0..$n {
                    out[i] = MaybeUninit::new(f(self.data[i]));
                }
                // Sanity check required because we can't use transmute on generic types.
                assert_eq!(
                    std::mem::size_of::<[MaybeUninit<U>; $n]>(),
                    std::mem::size_of::<[U; $n]>()
                );
                Tensor::new(unsafe { std::mem::transmute_copy::<_, [U; $n]>(&out) })
            }
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

        impl<T: std::ops::Add<Output = T> + Zero + Copy, I> Tensor<[T; $n], I> {
            pub fn sum(&self) -> T {
                self.fold(T::zero(), |acc, x| acc + x)
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

        // Scalar multiply assign by a raw scalar.
        impl<T: Scalar, I> MulAssign<T> for Tensor<[T; $n], I>
        where
            Tensor<[T; $n], I>: MulAssign<Tensor<T>>,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs);
            }
        }

        // Scalar divide by a raw scalar.
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

        // Scalar divide assign by a raw scalar.
        impl<T: Scalar, I> DivAssign<T> for Tensor<[T; $n], I>
        where
            Tensor<[T; $n], I>: DivAssign<Tensor<T>>,
        {
            fn div_assign(&mut self, rhs: T) {
                *self /= Tensor::new(rhs);
            }
        }

        // Right multiply by a scalar tensor.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T: Scalar, I> Mul<Tensor<T>> for Tensor<[T; $n], I>
        where
            Self: MulAssign<Tensor<T>>,
        {
            type Output = Self;
            fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl<T: Scalar, I> MulAssign<Tensor<T>> for Tensor<[T; $n], I>
        where
            Tensor<T>: MulAssign + Clone,
        {
            #[unroll_for_loops]
            fn mul_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    *self.data.index_mut(i).as_mut_tensor() *= rhs.clone();
                }
            }
        }

        impl<T: Scalar, I> Div<Tensor<T>> for Tensor<[T; $n], I>
        where
            Self: DivAssign<Tensor<T>>,
        {
            type Output = Self;
            fn div(mut self, rhs: Tensor<T>) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl<T: Scalar, I> DivAssign<Tensor<T>> for Tensor<[T; $n], I>
        where
            Tensor<T>: DivAssign + Clone,
        {
            #[unroll_for_loops]
            fn div_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$n {
                    *self.data.index_mut(i).as_mut_tensor() /= rhs.clone();
                }
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

        // Left scalar multiply by a scalar wrapped as a tensor.
        // Note that left scalar multiply cannot work generically with raw scalars because of
        // Rust's orphan rules. However, if we wrap a scalar in a tensor struct, this will
        // work.
        impl<T: Scalar, I> Mul<Tensor<[T; $n], I>> for Tensor<T>
        where
            Tensor<[T; $n], I>: MulAssign<Tensor<T>>,
        {
            type Output = Tensor<[T; $n], I>;
            fn mul(self, mut rhs: Tensor<[T; $n], I>) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
        impl<T: Zero + AddAssign + Copy + PartialEq, I> Zero for Tensor<[T; $n], I> {
            fn zero() -> Self {
                Tensor::new([Zero::zero(); $n])
            }
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }
    };
}

impl_array_vectors!(Vector1, RowVector1; 1);
impl_array_vectors!(Vector2, RowVector2; 2);
impl_array_vectors!(Vector3, RowVector3; 3);
impl_array_vectors!(Vector4, RowVector4; 4);

impl<T: Scalar, I> Tensor<[T; 3], I> {
    pub fn cross(&self, other: Tensor<[T; 3], I>) -> Tensor<[T; 3], I> {
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
        pub type $mtxn<T, I = (), J = ()> = Tensor<[[T; $c]; $r], (I, J)>;

        // Transposes of small matrices are implemented eagerly.
        impl<T: Scalar, I, J> Matrix for Tensor<[[T; $c]; $r], (I, J)> {
            type Transpose = Tensor<[[T; $r]; $c], (J, I)>;
            #[unroll_for_loops]
            fn transpose(self) -> Self::Transpose {
                let mut m = [[T::zero(); $r]; $c];

                for row in 0..$r {
                    for col in 0..$c {
                        m[col][row] = self[row][col];
                    }
                }
                Tensor::new(m)
            }
            fn num_rows(&self) -> usize {
                $r
            }
            fn num_cols(&self) -> usize {
                $c
            }
        }

        impl<T> $mtxn<T> {
            pub const fn def(data: [[T; $c]; $r]) -> Self {
                Self::new(data)
            }
        }

        impl<T: Scalar, I> Tensor<[[T; $c]; $r], I> {
            /// Zip the rows of this matrix and another tensor with the given combinator.
            #[unroll_for_loops]
            pub fn zip_with<B, U, F>(
                &self,
                other: Tensor<[B; $r], I>,
                mut f: F,
            ) -> Tensor<[U; $r], I>
            where
                B: Copy,
                Tensor<[U; $r], I>: Zero,
                F: FnMut([T; $c], B) -> U,
            {
                let mut out = Tensor::<[U; $r], I>::zero();
                for i in 0..$r {
                    out[i] = f(self.data[i], other.data[i]);
                }
                out
            }

            /// Similar to `map` but applies the given function to each inner element.
            #[unroll_for_loops]
            pub fn map_inner<U, F>(&self, mut f: F) -> Tensor<[[U; $c]; $r], I>
            where
                U: Scalar,
                F: FnMut(T) -> U,
            {
                let mut out = Tensor::<[[U; $c]; $r], I>::zeros();
                for row in 0..$r {
                    out[row] = Tensor::flat(self[row]).map(|x| f(x)).into_inner();
                }
                out
            }
            pub fn identity() -> Tensor<[[T; $c]; $r], I> {
                Self::from_diag_iter(std::iter::repeat(T::one()))
            }
            pub fn from_diag_iter<Iter: IntoIterator<Item = T>>(
                diag: Iter,
            ) -> Tensor<[[T; $c]; $r], I> {
                let mut out = Self::zeros();
                for (i, elem) in diag.into_iter().take($r.min($c)).enumerate() {
                    out[i][i] = elem;
                }
                out
            }
            pub fn zeros() -> Tensor<[[T; $c]; $r], I> {
                Tensor::new([[T::zero(); $c]; $r])
            }
            #[unroll_for_loops]
            pub fn fold_inner<B, F>(&self, mut init: B, mut f: F) -> B
            where
                F: FnMut(B, T) -> B,
            {
                for i in 0..$r {
                    init = Tensor::flat(self[i]).fold(init, |acc, x| f(acc, x));
                }
                init
            }

            /// Compute the sum of all entries in this matrix.
            pub fn sum_inner(&self) -> T {
                self.fold_inner(T::zero(), |acc, x| acc + x)
            }

            pub fn trace(&self) -> T {
                let mut tr = self[0][0];
                for i in 1..$r.min($c) {
                    tr += self[i][i];
                }
                tr
            }
            pub fn frob_norm_squared(&self) -> T {
                (*self).map_inner(|x| x * x).sum_inner()
            }
        }

        impl<T: Float + Scalar, I> Tensor<[[T; $c]; $r], I> {
            pub fn frob_norm(&self) -> T {
                self.frob_norm_squared().sqrt()
            }
        }

        /*
         * Matrix-vector multiply
         */
        impl<T: Scalar, I, J> Mul<Tensor<[T; $c], J>> for Tensor<[[T; $c]; $r], (I, J)>
        where
            Tensor<[T; $c], J>: MulAssign<Tensor<T>> + Clone,
        {
            type Output = Tensor<[T; $r], I>;
            fn mul(self, rhs: Tensor<[T; $c], J>) -> Self::Output {
                Tensor::new(self.map(|row| row.as_tensor().dot(rhs.clone())).data)
            }
        }

        // Right scalar multiply by a raw scalar.
        impl<T: Scalar, I> Mul<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: MulAssign<T>,
        {
            type Output = Self;
            fn mul(mut self, rhs: T) -> Self::Output {
                self *= rhs;
                self
            }
        }

        // Scalar multiply assign by a raw scalar.
        impl<T: Scalar, I> MulAssign<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: MulAssign<Tensor<T>>,
        {
            fn mul_assign(&mut self, rhs: T) {
                *self *= Tensor::new(rhs);
            }
        }

        // Scalar divide by a raw scalar.
        impl<T: Scalar, I> Div<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: DivAssign<T>,
        {
            type Output = Self;
            fn div(mut self, rhs: T) -> Self::Output {
                self /= rhs;
                self
            }
        }

        // Divide assign by a raw scalar.
        impl<T: Scalar, I> DivAssign<T> for Tensor<[[T; $c]; $r], I>
        where
            Tensor<[[T; $c]; $r], I>: DivAssign<Tensor<T>>,
        {
            fn div_assign(&mut self, rhs: T) {
                *self /= Tensor::new(rhs);
            }
        }

        // Right multiply by a tensor scalar.
        // Note the clone trait is required for cloning the RHS for every row in Self.
        impl<T: Scalar, I> Mul<Tensor<T>> for Tensor<[[T; $c]; $r], I>
        where
            Self: MulAssign<Tensor<T>>,
        {
            type Output = Self;
            fn mul(mut self, rhs: Tensor<T>) -> Self::Output {
                self *= rhs;
                self
            }
        }

        // Divide by a tensor scalar.
        impl<T: Scalar, I> Div<Tensor<T>> for Tensor<[[T; $c]; $r], I>
        where
            Self: DivAssign<Tensor<T>>,
        {
            type Output = Self;
            fn div(mut self, rhs: Tensor<T>) -> Self::Output {
                self /= rhs;
                self
            }
        }

        // Multiply assign by a tensor scalar
        impl<T: Scalar, I, J> MulAssign<Tensor<T>> for Tensor<[[T; $c]; $r], (I, J)>
        where
            Tensor<[T; $c], J>: MulAssign<Tensor<T>>,
        {
            #[unroll_for_loops]
            fn mul_assign(&mut self, rhs: Tensor<T>) {
                use std::ops::IndexMut;
                for i in 0..$r {
                    *self.data.index_mut(i).as_mut_tensor() *= rhs.clone();
                }
            }
        }

        // Divide assign by a tensor scalar
        impl<T: Scalar, I, J> DivAssign<Tensor<T>> for Tensor<[[T; $c]; $r], (I, J)>
        where
            Tensor<[T; $c], J>: DivAssign<Tensor<T>>,
        {
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
        impl<T: Scalar, I> Mul<Tensor<[[T; $c]; $r], I>> for Tensor<T>
        where
            Tensor<[[T; $c]; $r], I>: MulAssign<Tensor<T>>,
        {
            type Output = Tensor<[[T; $c]; $r], I>;
            fn mul(self, mut rhs: Tensor<[[T; $c]; $r], I>) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
    };
}

pub type Matrix1<T, I = (), J = ()> = Tensor<[[T; 1]; 1], (I, J)>;
impl_array_matrices!(Matrix2; 2, 2);
impl_array_matrices!(Matrix3; 3, 3);
impl_array_matrices!(Matrix4; 4, 4);

// Common Rectangular matrices
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
        impl<T: Scalar, I, J, K> Mul<Tensor<[[T; $n]; $p], (J, K)>>
            for Tensor<[[T; $p]; $m], (I, J)>
        where
            J: Copy,
            K: Copy,
            Tensor<[T; $n], K>: Mul<Tensor<T>, Output = Tensor<[T; $n], K>>,
        {
            type Output = Tensor<[[T; $n]; $m], (I, K)>;
            fn mul(self, rhs: Tensor<[[T; $n]; $p], (J, K)>) -> Self::Output {
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

impl_matrix_matrix_mul!(1, 1, 2);
impl_matrix_matrix_mul!(1, 1, 3);
impl_matrix_matrix_mul!(1, 1, 4);

impl_matrix_matrix_mul!(1, 2, 2);
impl_matrix_matrix_mul!(1, 2, 3);
impl_matrix_matrix_mul!(1, 2, 4);

impl_matrix_matrix_mul!(1, 3, 2);
impl_matrix_matrix_mul!(1, 3, 3);
impl_matrix_matrix_mul!(1, 3, 4);

impl_matrix_matrix_mul!(1, 4, 2);
impl_matrix_matrix_mul!(1, 4, 3);
impl_matrix_matrix_mul!(1, 4, 4);

impl_matrix_matrix_mul!(2, 1, 2);
impl_matrix_matrix_mul!(2, 1, 3);
impl_matrix_matrix_mul!(2, 1, 4);

impl_matrix_matrix_mul!(2, 2, 2);
impl_matrix_matrix_mul!(2, 2, 3);
impl_matrix_matrix_mul!(2, 2, 4);

impl_matrix_matrix_mul!(2, 3, 2);
impl_matrix_matrix_mul!(2, 3, 3);
impl_matrix_matrix_mul!(2, 3, 4);

impl_matrix_matrix_mul!(2, 4, 2);
impl_matrix_matrix_mul!(2, 4, 3);
impl_matrix_matrix_mul!(2, 4, 4);

impl_matrix_matrix_mul!(3, 1, 2);
impl_matrix_matrix_mul!(3, 1, 3);
impl_matrix_matrix_mul!(3, 1, 4);

impl_matrix_matrix_mul!(3, 2, 2);
impl_matrix_matrix_mul!(3, 2, 3);
impl_matrix_matrix_mul!(3, 2, 4);

impl_matrix_matrix_mul!(3, 3, 2);
impl_matrix_matrix_mul!(3, 3, 3);
impl_matrix_matrix_mul!(3, 3, 4);

impl_matrix_matrix_mul!(3, 4, 2);
impl_matrix_matrix_mul!(3, 4, 3);
impl_matrix_matrix_mul!(3, 4, 4);

impl_matrix_matrix_mul!(4, 1, 2);
impl_matrix_matrix_mul!(4, 1, 3);
impl_matrix_matrix_mul!(4, 1, 4);

impl_matrix_matrix_mul!(4, 2, 2);
impl_matrix_matrix_mul!(4, 2, 3);
impl_matrix_matrix_mul!(4, 2, 4);

impl_matrix_matrix_mul!(4, 3, 2);
impl_matrix_matrix_mul!(4, 3, 3);
impl_matrix_matrix_mul!(4, 3, 4);

impl_matrix_matrix_mul!(4, 4, 2);
impl_matrix_matrix_mul!(4, 4, 3);
impl_matrix_matrix_mul!(4, 4, 4);

/*
 * The following section defines functions on specific small matrix types.
 */

/*
 * Skew symmetric matrix representing the cross product operator
 */
impl<T: Scalar + std::ops::Neg<Output = T>, I> Vector3<T, I> {
    /// Convert this vector into a skew symmetric matrix, which represents the cross
    /// product operator (when applied to another vector).
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
impl<T: Scalar, I> Tensor<[[T; 1]; 1], I> {
    pub fn determinant(&self) -> T {
        self.data[0][0]
    }
}

macro_rules! impl_determinant {
    ($n:expr) => {
        /// Determinant of a 2x2 Matrix.
        impl<T: Scalar, I> Tensor<[[T; $n]; $n], I> {
            /// Construct a matrix smaller in both dimensions by 1 that is the same as the
            /// original matrix but without the first row and a given column. Although this is
            /// a very specific function, it is useful for efficient co-factor expansions.
            #[inline]
            #[unroll_for_loops]
            pub fn without_row_and_first_col(
                &self,
                col: usize,
            ) -> Tensor<[[T; $n - 1]; $n - 1], I> {
                let mut m: [[T; $n - 1]; $n - 1] = unsafe { ::std::mem::uninitialized() };
                for i in 0..$n - 1 {
                    m[i].copy_from_slice(&self[if i < col { i } else { i + 1 }][1..$n]);
                }
                Tensor::new(m)
            }

            /// Compute the determinant of the matrix recursively.
            #[inline]
            #[unroll_for_loops]
            pub fn determinant(&self) -> T {
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

impl<T: Scalar, I> Tensor<[[T; 1]; 1], I> {
    /// Compute the inverse of a 1x1 matrix.
    pub fn inverse(&self) -> Option<Self> {
        let denom = self[0][0];
        if denom != T::zero() {
            Some(Self::new([[T::one() / denom]]))
        } else {
            None
        }
    }
    /// Invert the 1x1 matrix in place. Return true if inversion was successful.
    pub fn invert(&mut self) -> bool {
        let denom = self[0][0];
        if denom != T::zero() {
            self[0][0] = T::one() / denom;
            true
        } else {
            false
        }
    }
}

impl<T: Scalar + Float, I, J> Tensor<[[T; 2]; 2], (I, J)> {
    /// Compute the inverse of a 2x2 matrix.
    pub fn inverse(&self) -> Option<Tensor<[[T; 2]; 2], (J, I)>> {
        let det = self.determinant();
        if det != T::zero() {
            Some(Tensor::new([
                [self[1][1] / det, -self[0][1] / det],
                [-self[1][0] / det, self[0][0] / det],
            ]))
        } else {
            None
        }
    }
    /// Compute the transpose of a 3x3 matrix inverse.
    pub fn inverse_transpose(&self) -> Option<Self> {
        let det = self.determinant();
        if det != T::zero() {
            Some(Self::new([
                [self[1][1] / det, -self[1][0] / det],
                [-self[0][1] / det, self[0][0] / det],
            ]))
        } else {
            None
        }
    }
}
impl<T: Scalar + Float, I> Tensor<[[T; 2]; 2], (I, I)> {
    /// Invert the 2x2 matrix in place. Return true if inversion was successful.
    pub fn invert(&mut self) -> bool {
        let det = self.determinant();
        if det != T::zero() {
            {
                let (a, b) = self.data.split_at_mut(1);
                std::mem::swap(&mut a[0][0], &mut b[0][1]);
            }
            self[0][1] = -self[0][1];
            self[1][0] = -self[1][0];
            *self /= det;
            true
        } else {
            false
        }
    }
}
impl<T: Scalar + Float, I, J> Tensor<[[T; 3]; 3], (I, J)> {
    /// Compute the inverse of a 3x3 matrix.
    pub fn inverse(&self) -> Option<Tensor<[[T; 3]; 3], (J, I)>> {
        self.inverse_transpose().map(|x| x.transpose())
    }
    /// Compute the transpose of a 3x3 matrix inverse.
    pub fn inverse_transpose(&self) -> Option<Self> {
        let det = self.determinant();
        if det != T::zero() {
            Some(Self::new([
                (Tensor::flat(self[1]).cross(Tensor::new(self[2])) / det).into_inner(),
                (Tensor::flat(self[2]).cross(Tensor::new(self[0])) / det).into_inner(),
                (Tensor::flat(self[0]).cross(Tensor::new(self[1])) / det).into_inner(),
            ]))
        } else {
            None
        }
    }
}
impl<T: Scalar + Float, I> Tensor<[[T; 3]; 3], (I, I)> {
    /// Invert the 3x3 matrix in place. Return true if inversion was successful.
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
