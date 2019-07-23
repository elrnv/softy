mod subset;
mod chunked;
mod uniform;
mod view;

pub use subset::*;
pub use chunked::*;
pub use uniform::*;
pub use view::*;

// Helper module defines a few useful unsigned type level integers.
// This is to avoid having to depend on yet another crate.
pub mod num {
    pub trait Unsigned {
        fn new() -> Self;
        fn value() -> usize;
    }

    macro_rules! def_num {
        ($(($nty:ident, $n:expr)),*) => {
            $(
                #[derive(Debug, Copy, Clone, PartialEq)]
                pub struct $nty;
                impl Unsigned for $nty {
                    fn new() -> Self { $nty }
                    fn value() -> usize {
                        $n
                    }
                }
             )*
        }
    }

    def_num!((U1, 1), (U2, 2), (U3, 3));
}

/// A trait defining a raw buffer of data. This data is typed but not annotated so it can represent
/// anything. For example a buffer of floats can represent a set of vertex colours or vertex
/// positions.
pub trait Set {
    /// Owned element of the set.
    type Elem;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An analog to the `ToOwned` trait from `std` that works for chunked views.
pub trait ToOwned where Self: Sized {
    type Owned;
    fn to_owned(self) -> Self::Owned;
    fn clone_into(self, target: &mut Self::Owned) {
        *target = self.to_owned();
    }
}

/// Blanked implementation of `ToOwned` for references of types that are already
/// `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> ToOwned for &S {
    type Owned = S::Owned;
    fn to_owned(self) -> Self::Owned {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// Blanked implementation of `ToOwned` for mutable references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> ToOwned for &mut S {
    type Owned = S::Owned;
    fn to_owned(self) -> Self::Owned {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// A helper trait analogous to `SliceIndex` from the standard library.
pub trait GetIndex<'i, 'o, S> {
    type Output;
    fn get(self, set: &'i S) -> Option<Self::Output>;
}

/// A helper trait like `GetIndex` but for mutable references.
pub trait GetMutIndex<'i, 'o, S> {
    type Output;
    fn get_mut(self, set: &'i mut S) -> Option<Self::Output>;
}

/// An index trait for `Set` types.
pub trait Get<'i, 'o, I> {
    type Output;
    fn get(&'i self, idx: I) -> Self::Output;
}

impl<'o, 'i: 'o, T, I> Get<'i, 'o, I> for [T]
where
    I: std::slice::SliceIndex<[T]>,
    <I as std::slice::SliceIndex<[T]>>::Output: 'o,
{
    type Output = &'o <[T] as std::ops::Index<I>>::Output;
    /// Index into a standard slice `[T]` using the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!(*utils::soap::Get::get(&[1,2,3,4,5][..], 2), 3);
    /// ```
    fn get(&'i self, idx: I) -> Self::Output {
        std::ops::Index::index(self, idx)
    }
}

impl<'o, 'i: 'o, T, I> Get<'i, 'o, I> for Vec<T>
where I: std::slice::SliceIndex<[T]>,
      <I as std::slice::SliceIndex<[T]>>::Output: 'o,
      T: Clone,
{
    type Output = &'o I::Output;
    /// Index into a `Vec` using the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v = vec![1,2,3,4,5];
    /// assert_eq!(*utils::soap::Get::get(&v, 2), 3);
    /// ```
    fn get(&'i self, idx: I) -> Self::Output {
        std::ops::Index::index(self, idx)
    }
}

impl<'o, 'i: 'o, S, I> Get<'i, 'o, I> for &'i S
where
    S: Get<'i, 'o, I> + ?Sized,
{
    type Output = S::Output;
    /// Index into any borrowed collection `S`, which itself implements
    /// `Get<'i, 'o, I>`.
    fn get(&'i self, idx: I) -> Self::Output {
        (*self).get(idx)
    }
}

impl<T> Set for Vec<T> {
    type Elem = T;
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T> Set for [T] {
    type Elem = T;
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<'a, S: Set + ?Sized> Set for &'a S {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}
impl<'a, S: Set + ?Sized> Set for &'a mut S {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

/// Abstraction for pushing elements of type `T` onto a collection.
pub trait Push<T> {
    fn push(&mut self, element: T);
}

impl<T> Push<T> for Vec<T> {
    fn push(&mut self, element: T) {
        Vec::push(self, element);
    }
}

/// A helper trait to split a set into two sets at a given index.
/// This trait is used to implement iteration over `ChunkedView`s.
pub trait SplitAt where Self: Sized {
    /// Split self into two sets at the given midpoint.
    /// This function is analogous to `<[T]>::split_at`.
    fn split_at(self, mid: usize) -> (Self, Self);
}

/// A helper trait to split owned sets into two sets at a given index.
/// This trait is used to implement iteration over `Chunked`s.
pub trait SplitOff {
    /// Split self into two sets at the given midpoint.
    /// This function is analogous to `Vec::split_off`.
    /// `self` contains elements `[0, mid)`, and The returned `Self` contains
    /// elements `[mid, len)`.
    fn split_off(&mut self, mid: usize) -> Self;
}


/// Convert a collection into its underlying representation, effectively
/// stripping any organizational info.
pub trait IntoFlat {
    type FlatType;
    fn into_flat(self) -> Self::FlatType;
}

impl<T> IntoFlat for Vec<T> {
    type FlatType = Vec<T>;
    /// Since a `Vec` has no information about the structure of its underlying
    /// data, this is effectively a no-op.
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<'a, T> IntoFlat for &'a [T] {
    type FlatType = &'a [T];
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<'a, T> IntoFlat for &'a mut [T] {
    type FlatType = &'a mut [T];
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<T> SplitAt for Vec<T> {
    fn split_at(mut self, mid: usize) -> (Self, Self) {
        let r = self.split_off(mid);
        (self, r)
    }
}

/// A helper trait for constructing placeholder sets for use in `std::mem::replace`.
/// These don't necessarily have to correspond to bona-fide sets.
pub trait Dummy {
    fn dummy() -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test iteration of a `Chunked` inside a `Chunked`.
    #[test]
    fn var_of_uni_iter_test() {
        let u0 = UniChunked::<_, num::U2>::from_flat((1..=12).collect::<Vec<_>>());
        let v1 = Chunked::from_offsets(vec![0,2,3,6], u0);

        let mut iter1 = v1.iter();
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[1,2]), iter0.next());
        assert_eq!(Some(&[3,4]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[5,6]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[7,8]), iter0.next());
        assert_eq!(Some(&[9,10]), iter0.next());
        assert_eq!(Some(&[11,12]), iter0.next());
        assert_eq!(None, iter0.next());
    }
}
