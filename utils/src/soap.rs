mod chunked;
mod subset;
mod uniform;
mod view;

pub use chunked::*;
pub use subset::*;
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

/// A marker trait to indicate an owned collection type. This is to distinguish
/// them from borrowed slices, which essential to resolve implementation collisions.
pub trait Owned {}
impl<S, I> Owned for Subset<S, I> {}
impl<S, I> Owned for Chunked<S, I> {}
impl<S, N> Owned for UniChunked<S, N> {}
impl<T> Owned for Vec<T> {}

/// A marker trait to indicate a collection type that can be chunked. More precisely this is a type that can be composed with types in this crate.
pub trait Chunkable<'a>: Set + Get<'a, 'a, std::ops::Range<usize>> + RemovePrefix {}
impl<'a, T: Clone> Chunkable<'a> for &'a [T] {}
impl<'a, T: Clone> Chunkable<'a> for &'a mut [T] {}
impl<'a, T: Clone + 'a> Chunkable<'a> for Vec<T> {}

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
pub trait ToOwned
where
    Self: Sized,
{
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

/// A type of range whose size is determined at compile time.
/// This represents a range `[start..start + N::value()]`.
/// This aids `UniChunked` types when indexing.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct StaticRange<N> {
    pub start: usize,
    pub phantom: std::marker::PhantomData<N>,
}

impl<N> StaticRange<N> {
    fn new(start: usize) -> Self {
        StaticRange {
            start,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<N: num::Unsigned> StaticRange<N> {
    fn start(&self) -> usize {
        self.start
    }
    fn end(&self) -> usize {
        self.start + N::value()
    }
}

impl<'o, 'i: 'o, T, N> GetIndex<'i, 'o, [T]> for StaticRange<N>
where
    N: num::Unsigned,
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'o,
{
    type Output = &'o T::Type;
    fn get(self, set: &'i [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { &*(set.as_ptr().add(self.start()) as *const T::Type) })
        } else {
            None
        }
    }
}
impl<'o, 'i: 'o, T, N> GetMutIndex<'i, 'o, [T]> for StaticRange<N>
where
    N: num::Unsigned,
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'o,
{
    type Output = &'o mut T::Type;
    fn get_mut(self, set: &'i mut [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { &mut *(set.as_mut_ptr().add(self.start()) as *mut T::Type) })
        } else {
            None
        }
    }
}

/// A helper trait analogous to `SliceIndex` from the standard library.
pub trait GetIndex<'i, 'o, S>
where
    S: ?Sized,
{
    type Output;
    fn get(self, set: &'i S) -> Option<Self::Output>;
    //unsafe fn get_unchecked(self, set: &'i S) -> Self::Output;
}

/// A helper trait like `GetIndex` but for mutable references.
pub trait GetMutIndex<'i, 'o, S>
where
    S: ?Sized,
{
    type Output;
    fn get_mut(self, set: &'i mut S) -> Option<Self::Output>;
    //unsafe fn get_unchecked_mut(self, set: &'i mut S) -> Self::Output;
}

/// An index trait for collection types.
/// Here `'i` indicates the lifetime of the input while `'o` indicates that of
/// the output.
pub trait Get<'i, 'o, I> {
    type Output;
    //unsafe fn get_unchecked(&'i self, idx: I) -> Self::Output;
    fn get(&'i self, idx: I) -> Option<Self::Output>;
    /// Return a value at the given index. This is provided as the checked
    /// version of `get` that will panic if the equivalent `get` call is `None`,
    /// which typically means that the given index is out of bounds.
    ///
    /// # Panics
    ///
    /// This function will panic if `self.get(idx)` returns `None`.
    fn at(&'i self, idx: I) -> Self::Output {
        self.get(idx).expect("Index out of bounds")
    }
}

/// An index trait for mutable collection types.
/// Here `'i` indicates the lifetime of the input while `'o` indicates that of
/// the output.
pub trait GetMut<'i, 'o, I> {
    type Output;
    //unsafe fn get_unchecked_mut(&'i self, idx: I) -> Self::Output;
    fn get_mut(&'i mut self, idx: I) -> Option<Self::Output>;
    /// Return a value at the given index. This is provided as the checked
    /// version of `get` that will panic if the equivalent `get` call is `None`,
    /// which typically means that the given index is out of bounds.
    ///
    /// # Panics
    ///
    /// This function will panic if `self.get(idx)` returns `None`.
    fn at_mut(&'i mut self, idx: I) -> Self::Output {
        self.get_mut(idx).expect("Index out of bounds")
    }
}

impl<'o, 'i: 'o, T, I> GetIndex<'i, 'o, [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <[T] as std::ops::Index<I>>::Output: 'o,
{
    type Output = &'o <[T] as std::ops::Index<I>>::Output;
    fn get(self, set: &'i [T]) -> Option<Self::Output> {
        Some(std::ops::Index::<I>::index(set, self))
    }
}

impl<'o, 'i: 'o, T, I> GetMutIndex<'i, 'o, [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <[T] as std::ops::Index<I>>::Output: 'o,
{
    type Output = &'o mut <[T] as std::ops::Index<I>>::Output;
    fn get_mut(self, set: &'i mut [T]) -> Option<Self::Output> {
        Some(std::ops::IndexMut::<I>::index_mut(set, self))
    }
}

impl<'o, 'i: 'o, S> GetIndex<'i, 'o, S> for std::ops::RangeFrom<usize>
where
    S: Set,
    std::ops::Range<usize>: GetIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'i, 'o, S>>::Output;

    fn get(self, set: &'i S) -> Option<Self::Output> {
        (self.start..set.len()).get(set)
    }
}

impl<'o, 'i: 'o, S> GetIndex<'i, 'o, S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: GetIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'i, 'o, S>>::Output;

    fn get(self, set: &'i S) -> Option<Self::Output> {
        (0..self.end).get(set)
    }
}

impl<'o, 'i: 'o, S> GetIndex<'i, 'o, S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: GetIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'i, 'o, S>>::Output;

    fn get(self, set: &'i S) -> Option<Self::Output> {
        (0..set.len()).get(set)
    }
}

impl<'o, 'i: 'o, S> GetIndex<'i, 'o, S> for std::ops::RangeInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'i, 'o, S>>::Output;

    fn get(self, set: &'i S) -> Option<Self::Output> {
        if *self.end() == usize::max_value() {
            None
        } else {
            (*self.start()..*self.end() + 1).get(set)
        }
    }
}

impl<'o, 'i: 'o, S> GetIndex<'i, 'o, S> for std::ops::RangeToInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'i, 'o, S>>::Output;

    fn get(self, set: &'i S) -> Option<Self::Output> {
        (0..=self.end).get(set)
    }
}

impl<'o, 'i: 'o, S> GetMutIndex<'i, 'o, S> for std::ops::RangeFrom<usize>
where
    S: Set,
    std::ops::Range<usize>: GetMutIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetMutIndex<'i, 'o, S>>::Output;

    fn get_mut(self, set: &'i mut S) -> Option<Self::Output> {
        (self.start..set.len()).get_mut(set)
    }
}

impl<'o, 'i: 'o, S> GetMutIndex<'i, 'o, S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: GetMutIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetMutIndex<'i, 'o, S>>::Output;

    fn get_mut(self, set: &'i mut S) -> Option<Self::Output> {
        (0..self.end).get_mut(set)
    }
}

impl<'o, 'i: 'o, S> GetMutIndex<'i, 'o, S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: GetMutIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetMutIndex<'i, 'o, S>>::Output;

    fn get_mut(self, set: &'i mut S) -> Option<Self::Output> {
        (0..set.len()).get_mut(set)
    }
}

impl<'o, 'i: 'o, S> GetMutIndex<'i, 'o, S> for std::ops::RangeInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: GetMutIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetMutIndex<'i, 'o, S>>::Output;

    fn get_mut(self, set: &'i mut S) -> Option<Self::Output> {
        if *self.end() == usize::max_value() {
            None
        } else {
            (*self.start()..*self.end() + 1).get_mut(set)
        }
    }
}

impl<'o, 'i: 'o, S> GetMutIndex<'i, 'o, S> for std::ops::RangeToInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: GetMutIndex<'i, 'o, S>,
{
    type Output = <std::ops::Range<usize> as GetMutIndex<'i, 'o, S>>::Output;

    fn get_mut(self, set: &'i mut S) -> Option<Self::Output> {
        (0..=self.end).get_mut(set)
    }
}

impl<'o, 'i: 'o, T: 'o, I> Get<'i, 'o, I> for [T]
where
    I: GetIndex<'i, 'o, [T]>,
{
    type Output = I::Output;
    /// Index into a standard slice `[T]` using the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!(utils::soap::Get::get(&[1,2,3,4,5][..], 2), Some(&3));
    /// ```
    fn get(&'i self, idx: I) -> Option<Self::Output> {
        GetIndex::get(idx, self)
    }
}

impl<'o, 'i: 'o, T: 'o, I> Get<'i, 'o, I> for Vec<T>
where
    I: GetIndex<'i, 'o, [T]>,
    T: Clone,
{
    type Output = I::Output;
    /// Index into a `Vec` using the `Get` trait.
    /// This function is not intended to be used directly, but it allows getters
    /// for chunked collections to work.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v = vec![1,2,3,4,5];
    /// assert_eq!(utils::soap::Get::get(&v, 2), Some(&3));
    /// ```
    fn get(&'i self, idx: I) -> Option<Self::Output> {
        GetIndex::get(idx, self.as_slice())
    }
}

impl<'o, 'i: 'o, T: 'o, I> GetMut<'i, 'o, I> for [T]
where
    I: GetMutIndex<'i, 'o, [T]>,
{
    type Output = I::Output;
    /// Mutably index into a standard slice `[T]` using the `GetMut` trait.
    /// This function is not intended to be used directly, but it allows getters
    /// for chunked collections to work.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec![1,2,3,4,5];
    /// *utils::soap::GetMut::get_mut(v.as_mut_slice(), 2).unwrap() = 100;
    /// assert_eq!(v, vec![1,2,100,4,5]);
    /// ```
    fn get_mut(&'i mut self, idx: I) -> Option<Self::Output> {
        GetMutIndex::get_mut(idx, self)
    }
}

impl<'o, 'i: 'o, T: 'o, I> GetMut<'i, 'o, I> for Vec<T>
where
    I: GetMutIndex<'i, 'o, [T]>,
    T: Clone,
{
    type Output = I::Output;
    /// Index into a `Vec` using the `Get` trait.
    /// This function is not intended to be used directly, but it allows getters
    /// for chunked collections to work.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec![1,2,3,4,5];
    /// *utils::soap::GetMut::get_mut(&mut v, 2).unwrap() = 100;
    /// assert_eq!(v, vec![1,2,100,4,5]);
    /// ```
    fn get_mut(&'i mut self, idx: I) -> Option<Self::Output> {
        GetMutIndex::get_mut(idx, self)
    }
}

impl<'o, 'i: 'o, S, I> GetMut<'i, 'o, I> for &mut S
where
    S: GetMut<'i, 'o, I> + ?Sized,
{
    type Output = S::Output;
    /// Index into any borrowed collection `S`, which itself implements
    /// `Get<'i, 'o, I>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec![1,2,3,4,5];
    /// let mut s = v.as_mut_slice();
    /// *utils::soap::GetMut::get_mut(&mut s, 2).unwrap() = 100;
    /// assert_eq!(v, vec![1,2,100,4,5]);
    /// ```
    fn get_mut(&'i mut self, idx: I) -> Option<Self::Output> {
        (*self).get_mut(idx)
    }
}

impl<'o, 'i: 'o, S, I> Get<'i, 'o, I> for &S
where
    S: Get<'i, 'o, I> + ?Sized,
{
    type Output = S::Output;
    /// Index into any borrowed collection `S`, which itself implements
    /// `Get<'i, 'o, I>`.
    fn get(&'i self, idx: I) -> Option<Self::Output> {
        (*self).get(idx)
    }
}

impl<'o, 'i: 'o, S, I> Get<'i, 'o, I> for &mut S
where
    S: Get<'i, 'o, I> + ?Sized,
{
    type Output = S::Output;
    /// Index into any borrowed collection `S`, which itself implements
    /// `Get<'i, 'o, I>`.
    fn get(&'i self, idx: I) -> Option<Self::Output> {
        (**self).get(idx) // Borrows S by &'i reference.
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

impl<S: Set + ?Sized> Set for &S {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}
impl<S: Set + ?Sized> Set for &mut S {
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
pub trait SplitAt
where
    Self: Sized,
{
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

pub trait SplitPrefix<N>
where
    Self: Sized,
{
    type Prefix;
    fn split_prefix(self) -> Option<(Self::Prefix, Self)>;
}

/// `SplitFirst` is an alias for `SplitPrefix<num::U1>`.
pub trait SplitFirst
where
    Self: Sized,
{
    type First;
    fn split_first(self) -> Option<(Self::First, Self)>;
}

impl<T> SplitFirst for T
where
    T: SplitPrefix<num::U1>,
{
    type First = T::Prefix;
    fn split_first(self) -> Option<(Self::First, Self)> {
        self.split_prefix()
    }
}

impl<T, N> SplitPrefix<N> for Vec<T>
where
    T: Grouped<N>,
    <T as Grouped<N>>::Type: Default,
    N: num::Unsigned,
{
    type Prefix = T::Type;

    fn split_prefix(mut self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }
        // Note: This is inefficient ( as is the implementation for `remove_prefix` ).
        // As such it shouldn't be used when iterating over `Subset`s of
        // `Vec<T>` or `Subset`s of any other chunked collection that uses
        // `Vec<T>` for storage. We should be able to specialize the
        // implementation of subsets of `Vec<T>` types for better performance.
        self.rotate_left(N::value());
        let at = self.len() - N::value();
        let mut out: T::Type = Default::default();
        unsafe {
            self.set_len(at);
            std::ptr::copy_nonoverlapping(
                self.as_ptr().add(at),
                &mut out as *mut T::Type as *mut T,
                N::value(),
            );
        }
        Some((out, self))
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a [T]
where
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
    N: num::Unsigned,
{
    type Prefix = &'a T::Type;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }

        let (prefix, rest) = unsafe {
            let prefix = self.as_ptr() as *const T::Type;
            (&*prefix, &self[N::value()..])
        };
        Some((prefix, rest))
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a mut [T]
where
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
    N: num::Unsigned,
{
    type Prefix = &'a mut T::Type;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }

        let (prefix, rest) = unsafe {
            let prefix = self.as_mut_ptr() as *mut T::Type;
            (&mut *prefix, &mut self[N::value()..])
        };
        Some((prefix, rest))
    }
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

impl<'a, T> SplitAt for &mut [T] {
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at_mut(mid)
    }
}

impl<'a, T> SplitAt for &[T] {
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at(mid)
    }
}

/// A helper trait for constructing placeholder sets for use in `std::mem::replace`.
/// These don't necessarily have to correspond to bona-fide sets.
pub trait Dummy {
    fn dummy() -> Self;
}

impl<T> Dummy for &[T] {
    fn dummy() -> Self {
        &[]
    }
}

impl<T> Dummy for &mut [T] {
    fn dummy() -> Self {
        &mut []
    }
}

/// A helper trait used to help implement the Subset. This trait allows
/// abstract collections to remove a number of elements from the
/// beginning, which is what we need for subsets.
pub trait RemovePrefix {
    /// Remove `n` elements from the beginning.
    fn remove_prefix(&mut self, n: usize);
}

impl<T> RemovePrefix for Vec<T> {
    fn remove_prefix(&mut self, n: usize) {
        self.rotate_left(n);
        self.truncate(self.len() - n);
    }
}

impl<T> RemovePrefix for &[T] {
    fn remove_prefix(&mut self, n: usize) {
        let (_, r) = self.split_at(n);
        *self = r;
    }
}

impl<T> RemovePrefix for &mut [T] {
    /// Remove a prefix of size `n` from this mutable slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// use::utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut s = v.as_mut_slice();
    /// s.remove_prefix(2);
    /// assert_eq!(&[3,4,5], s);
    /// ```
    fn remove_prefix(&mut self, n: usize) {
        let data = std::mem::replace(self, &mut []);

        let (_, r) = data.split_at_mut(n);
        *self = r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test iteration of a `Chunked` inside a `Chunked`.
    #[test]
    fn var_of_uni_iter_test() {
        let u0 = UniChunked::<_, num::U2>::from_flat((1..=12).collect::<Vec<_>>());
        let v1 = Chunked::from_offsets(vec![0, 2, 3, 6], u0);

        let mut iter1 = v1.iter();
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[1, 2]), iter0.next());
        assert_eq!(Some(&[3, 4]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[5, 6]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[7, 8]), iter0.next());
        assert_eq!(Some(&[9, 10]), iter0.next());
        assert_eq!(Some(&[11, 12]), iter0.next());
        assert_eq!(None, iter0.next());
    }
}
