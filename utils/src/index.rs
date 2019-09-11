use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

/// A possibly invalid unsigned index.
/// The maximum `usize` integer represents an invalid index.
/// This index type is ideal for storage.
/// Overflow is not handled by this type. Instead we rely on Rust's internal overflow panics during
/// debug builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Index(usize);

impl Index {
    /// Invalid index instance.
    pub const INVALID: Index = Index(std::usize::MAX);

    /// An alternative way to create an invalid index using an inline.
    #[inline]
    pub fn invalid() -> Index {
        Self::INVALID
    }

    /// Create a valid index from a usize type. This constructor does the necessary check
    /// for debug builds only.
    #[inline]
    pub fn new(i: usize) -> Index {
        debug_assert!(Index::fits(i));
        Index(i)
    }

    /// Convert this `Index` into `Option<usize>`, which is a larger struct.
    #[inline]
    pub fn into_option(self) -> Option<usize> {
        self.into()
    }

    /// Return true if stored index is valid and false otherwise.
    #[inline]
    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }

    /// Convert the index into an unsigned integer. Panic with given `msg` if index is invalid
    #[inline]
    pub fn expect(self, msg: &str) -> usize {
        if self.is_valid() {
            self.0
        } else {
            panic!("{}", msg)
        }
    }

    /// Convert the index into an unsigned integer. Panic if index is invalid.
    #[inline]
    pub fn unwrap(self) -> usize {
        self.expect("Unhandled Invalid Index.")
    }

    /// Manipulate the inner representation of the index. This method avoids the additional check
    /// used in `map`. Use this to opt out of automatic index checking.
    #[inline]
    pub fn map_inner<F: FnOnce(usize) -> usize>(self, f: F) -> Index {
        Index(f(self.0))
    }

    /// Checked map over inner index. This allows operations on valid indices only.
    #[inline]
    pub fn map<F: FnOnce(usize) -> usize>(self, f: F) -> Index {
        if self.is_valid() {
            Index::new(f(self.0 as usize))
        } else {
            self
        }
    }

    /// Checked `and_then` over inner index. This allows operations on valid indices only.
    #[inline]
    pub fn and_then<F: FnOnce(usize) -> Index>(self, f: F) -> Index {
        if self.is_valid() {
            f(self.0 as usize)
        } else {
            self
        }
    }

    /// Apply a function to the inner `usize` index. The index remains unchanged if invalid.
    #[inline]
    pub fn apply<F: FnOnce(&mut usize)>(&mut self, f: F) {
        if self.is_valid() {
            f(&mut self.0);
        }
    }

    /// Check that the given index fits inside the internal Index representation. This is used
    /// internally to check various conversions and arithmetics for debug builds.
    #[inline]
    fn fits(i: usize) -> bool {
        i != std::usize::MAX
    }
}

macro_rules! impl_from_unsigned {
    ($u:ty) => {
        /// Create a valid index from a `usize` type. This converter does the necessary bounds
        /// check for debug builds only.
        impl From<$u> for Index {
            #[inline]
            fn from(i: $u) -> Self {
                Index::new(i as usize)
            }
        }
    };
}

impl_from_unsigned!(usize);
impl_from_unsigned!(u64);
impl_from_unsigned!(u32);
impl_from_unsigned!(u16);
impl_from_unsigned!(u8);

macro_rules! impl_from_signed {
    ($i:ty) => {
        /// Create an index from a signed integer type. If the given argument is negative, the
        /// created index will be invalid.
        impl From<$i> for Index {
            #[inline]
            fn from(i: $i) -> Self {
                if i < 0 {
                    Index::invalid()
                } else {
                    Index(i as usize)
                }
            }
        }
    };
}

impl_from_signed!(isize);
impl_from_signed!(i64);
impl_from_signed!(i32);
impl_from_signed!(i16);
impl_from_signed!(i8);

impl Into<Option<usize>> for Index {
    #[inline]
    fn into(self) -> Option<usize> {
        if self.is_valid() {
            Some(self.0 as usize)
        } else {
            None
        }
    }
}

impl Add<usize> for Index {
    type Output = Index;

    #[inline]
    fn add(self, rhs: usize) -> Index {
        self.map(|x| x + rhs)
    }
}

impl AddAssign<usize> for Index {
    #[inline]
    fn add_assign(&mut self, rhs: usize) {
        self.apply(|x| *x += rhs)
    }
}

impl Add<Index> for usize {
    type Output = Index;

    #[inline]
    fn add(self, rhs: Index) -> Index {
        rhs + self
    }
}

impl Add for Index {
    type Output = Index;

    #[inline]
    fn add(self, rhs: Index) -> Index {
        // Note: add with overflow is checked by Rust for debug builds.
        self.and_then(|x| rhs.map(|y| x + y))
    }
}

impl Sub<usize> for Index {
    type Output = Index;

    #[inline]
    fn sub(self, rhs: usize) -> Index {
        self.map(|x| x - rhs)
    }
}

impl Sub<Index> for usize {
    type Output = Index;

    #[inline]
    fn sub(self, rhs: Index) -> Index {
        rhs.map(|x| self - x)
    }
}

impl Sub for Index {
    type Output = Index;

    #[inline]
    fn sub(self, rhs: Index) -> Index {
        // Note: subtract with overflow is checked by Rust for debug builds.
        self.and_then(|x| rhs.map(|y| x - y))
    }
}

impl Mul<usize> for Index {
    type Output = Index;

    #[inline]
    fn mul(self, rhs: usize) -> Index {
        self.map(|x| x * rhs)
    }
}

impl Mul<Index> for usize {
    type Output = Index;

    #[inline]
    fn mul(self, rhs: Index) -> Index {
        rhs * self
    }
}

// It often makes sense to divide an index by a non-zero integer
impl Div<usize> for Index {
    type Output = Index;

    #[inline]
    fn div(self, rhs: usize) -> Index {
        if rhs != 0 {
            self.map(|x| x / rhs)
        } else {
            Index::invalid()
        }
    }
}

impl Rem<usize> for Index {
    type Output = Index;

    #[inline]
    fn rem(self, rhs: usize) -> Index {
        if rhs != 0 {
            self.map(|x| x % rhs)
        } else {
            Index::invalid()
        }
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::invalid()
    }
}

impl fmt::Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// TODO: Implement these for SliceIndex if/when it gets stabilized
//impl<T: std::ops::Index<usize>> std::ops::Index<Index> for T {
//    type Output = T::Output;
//    fn index(&self, idx: Index) -> &Self::Output {
//        &self[idx.unwrap()]
//    }
//}
//
//impl<T: std::ops::IndexMut<usize>> std::ops::IndexMut<Index> for T {
//    fn index_mut(&mut self, idx: Index) -> &Self::Output {
//        &mut self[idx.unwrap()]
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_test() {
        let i = Index::new(2);
        let j = Index::new(4);
        let k = Index::invalid();
        assert_eq!(i + j, Index::new(6));
        assert_eq!(j - i, Index::new(2));
        assert_eq!(i - k, Index::invalid());
        assert_eq!(i + k, Index::invalid());
        assert_eq!(k * 2, Index::invalid());
    }
}
