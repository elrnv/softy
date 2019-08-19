mod chunked;
mod range;
mod select;
mod slice;
mod sparse;
mod subset;
mod tensor;
mod uniform;
mod vec;
mod view;

pub use chunked::*;
pub use range::*;
pub use select::*;
pub use slice::*;
pub use sparse::*;
pub use subset::*;
pub use tensor::*;
pub use uniform::*;
pub use vec::*;
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
//TODO: Rename this, since Chunked Views are also considered "Owned", this is a misnomer.
// Maybe "ValueType" makes sense
pub trait Owned {}
impl<S, I> Owned for Subset<S, I> {}
impl<S, I> Owned for Chunked<S, I> {}
impl<S, N> Owned for UniChunked<S, N> {}

/// A marker trait to indicate a collection type that can be chunked. More precisely this is a type that can be composed with types in this crate.
//pub trait Chunkable<'a>:
//    Set + Get<'a, 'a, std::ops::Range<usize>> + RemovePrefix + View<'a> + PartialEq
//{
//}
//impl<'a, T: Clone + PartialEq> Chunkable<'a> for &'a [T] {}
//impl<'a, T: Clone + PartialEq> Chunkable<'a> for &'a mut [T] {}
//impl<'a, T: Clone + PartialEq + 'a> Chunkable<'a> for Vec<T> {}

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

/// In contrast to `ToOwned`, this trait produces a clone with owned data, but
/// potentially borrowed structure of the collection.
pub trait ToOwnedData
where
    Self: Sized,
{
    type OwnedData;
    fn to_owned_data(self) -> Self::OwnedData;
    fn clone_into(self, target: &mut Self::OwnedData) {
        *target = self.to_owned_data();
    }
}

/// Blanked implementation of `ToOwnedData` for references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> ToOwnedData for &S {
    type OwnedData = S::Owned;
    fn to_owned_data(self) -> Self::OwnedData {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// Blanked implementation of `ToOwnedData` for mutable references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> ToOwnedData for &mut S {
    type OwnedData = S::Owned;
    fn to_owned_data(self) -> Self::OwnedData {
        std::borrow::ToOwned::to_owned(self)
    }
}

pub trait Clear {
    /// Remove all elements from the current set without necessarily
    /// deallocating the space previously used.
    fn clear(&mut self);
}

// A Note on indexing:
// ===================
// Following the standard library we support indexing by usize only.
// However, Ranges as collections can be supported for other types as well.

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

/// A helper trait analogous to `SliceIndex` from the standard library.
pub trait GetIndex<'a, S>
where
    S: ?Sized,
{
    type Output;
    fn get(self, set: &S) -> Option<Self::Output>;
    //unsafe fn get_unchecked(self, set: &'i S) -> Self::Output;
}

/// A helper trait like `GetIndex` but for `Isolate` types.
pub trait IsolateIndex<S> {
    type Output;
    fn try_isolate(self, set: S) -> Option<Self::Output>;
    //unsafe fn get_unchecked_mut(self, set: &'i mut S) -> Self::Output;
}

/// An index trait for collection types.
/// Here `'i` indicates the lifetime of the input while `'o` indicates that of
/// the output.
pub trait Get<'a, I> {
    type Output;
    //unsafe fn get_unchecked(&'i self, idx: I) -> Self::Output;
    fn get(&self, idx: I) -> Option<Self::Output>;
    /// Return a value at the given index. This is provided as the checked
    /// version of `get` that will panic if the equivalent `get` call is `None`,
    /// which typically means that the given index is out of bounds.
    ///
    /// # Panics
    ///
    /// This function will panic if `self.get(idx)` returns `None`.
    fn at(&self, idx: I) -> Self::Output {
        self.get(idx).expect("Index out of bounds")
    }
}

/// A blanket implementation of `Get` for any collection which has an implementation for `GetIndex`.
impl<'a, S, I> Get<'a, I> for S
where
    I: GetIndex<'a, Self>,
{
    type Output = I::Output;
    fn get(&self, idx: I) -> Option<I::Output> {
        idx.get(self)
    }
}

/// Since we cannot alias mutable references, in order to index a mutable view
/// of elements, we must consume the original mutable reference. Since we can't
/// use slices for general composable collections, its impossible to match
/// against a `&mut self` in the getter function to be able to use it with owned
/// collections, so we opt to have an interface that is designed specifically
/// for mutably borrowed collections. For composable collections, this is better
/// described by a subview operator, which is precisely what this trait
/// represents. Incidentally this can also work for owned collections, which is
/// why it's called `Isolate` instead of `SubView`.
pub trait Isolate<I> {
    type Output;
    //unsafe fn isolate_unchecked(&'i self, idx: I) -> Self::Output;
    fn try_isolate(self, idx: I) -> Option<Self::Output>;
    /// Return a value at the given index. This is provided as the checked
    /// version of `try_isolate` that will panic if the equivalent `try_isolate`
    /// call is `None`, which typically means that the given index is out of
    /// bounds.
    ///
    /// # Panics
    ///
    /// This function will panic if `self.get(idx)` returns `None`.
    fn isolate(self, idx: I) -> Self::Output
    where
        Self: Sized,
    {
        self.try_isolate(idx).expect("Index out of bounds")
    }
}

impl<'a, S> GetIndex<'a, S> for std::ops::RangeFrom<usize>
where
    S: Set + Owned,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (self.start..set.len()).get(set)
    }
}

impl<'a, S: Owned> GetIndex<'a, S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (0..self.end).get(set)
    }
}

impl<'a, S: Owned> GetIndex<'a, S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (0..set.len()).get(set)
    }
}

impl<'a, S: Owned> GetIndex<'a, S> for std::ops::RangeInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[allow(clippy::range_plus_one)]
    fn get(self, set: &S) -> Option<Self::Output> {
        if *self.end() == usize::max_value() {
            None
        } else {
            (*self.start()..*self.end() + 1).get(set)
        }
    }
}

impl<'a, S: Owned> GetIndex<'a, S> for std::ops::RangeToInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (0..=self.end).get(set)
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeFrom<usize>
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        (self.start..set.len()).try_isolate(set)
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        (0..self.end).try_isolate(set)
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        (0..set.len()).try_isolate(set)
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[allow(clippy::range_plus_one)]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        if *self.end() == usize::max_value() {
            None
        } else {
            (*self.start()..*self.end() + 1).try_isolate(set)
        }
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeToInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        (0..=self.end).try_isolate(set)
    }
}

//impl<'a, S, I> GetMut<'a, I> for &mut S
//where
//    S: Owned + GetMut<'a, I>,
//{
//    type Output = S::Output;
//    /// Index into any borrowed collection `S`, which itself implements
//    /// `Get<'a, I>`.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// let mut v = vec![1,2,3,4,5];
//    /// let mut s = v.as_mut_slice();
//    /// *utils::soap::GetMut::get_mut(&mut s, 2).unwrap() = 100;
//    /// assert_eq!(v, vec![1,2,100,4,5]);
//    /// ```
//    fn get_mut(&mut self, idx: I) -> Option<Self::Output> {
//        (*self).get_mut(idx)
//    }
//}
//
//impl<'a, S, I> Get<'a, I> for &S
//where
//    S: Owned + Get<'a, I>,
//{
//    type Output = S::Output;
//    /// Index into any borrowed collection `S`, which itself implements
//    /// `Get<'a, I>`.
//    fn get(&self, idx: I) -> Option<Self::Output> {
//        (**self).get(idx)
//    }
//}
//
//impl<'a, S, I> Get<'a, I> for &mut S
//where
//    S: Owned + Get<'a, I>,
//{
//    type Output = S::Output;
//    /// Index into any mutably borrowed collection `S`, which itself implements
//    /// `Get<'a, I>`.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// let mut v = vec![1,2,3,4,5];
//    /// let mut s = v.as_mut_slice();
//    /// assert_eq!(utils::soap::Get::get(&mut s, 2).unwrap(), &3);
//    /// ```
//    fn get(&self, idx: I) -> Option<Self::Output> {
//        Get::<'a, I>::get(&**self, idx) // Borrows S by &'i reference.
//    }
//}

impl<N: num::Unsigned> Set for StaticRange<N> {
    type Elem = usize;
    fn len(&self) -> usize {
        N::value()
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

impl<S: Set + ?Sized> Set for std::cell::Ref<'_, S> {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

impl<S: Set + ?Sized> Set for std::cell::RefMut<'_, S> {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

/// Abstraction for pushing elements of type `T` onto a collection.
pub trait Push<T> {
    fn push(&mut self, element: T);
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

/// Convert a collection into its underlying representation, effectively
/// stripping any organizational info.
pub trait IntoFlat {
    type FlatType;
    fn into_flat(self) -> Self::FlatType;
}

pub trait CloneWithFlat<FlatType> {
    type CloneType;
    fn clone_with_flat(&self, flat: FlatType) -> Self::CloneType;
}

/// A helper trait for constructing placeholder sets for use in `std::mem::replace`.
/// These don't necessarily have to correspond to bona-fide sets.
pub trait Dummy {
    fn dummy() -> Self;
}

/// A helper trait used to help implement the Subset. This trait allows
/// abstract collections to remove a number of elements from the
/// beginning, which is what we need for subsets.
pub trait RemovePrefix {
    /// Remove `n` elements from the beginning.
    fn remove_prefix(&mut self, n: usize);
}

/// This trait generalizes the method `chunks` available on slices in the
/// standard library. Collections that can be chunked by a runtime stride should
/// implement this behaviour such that they can be composed with `ChunkedN`
/// types.
pub trait IntoChunkIterator {
    type Item;
    type IterType: Iterator<Item = Self::Item>;
    fn into_chunk_iter(&self) -> Self::IterType;
}

/*
 * Tests
 */

///```compile_fail
/// use utils::soap::*;
/// // This shouldn't compile
/// let v: Vec<usize> = (1..=10).collect();
/// let chunked = Chunked::from_offsets(vec![0, 3, 5, 8, 10], v);
/// let mut chunked = Chunked::from_offsets(vec![0, 1, 4], chunked);
/// let mut mut_view = chunked.view_mut();
///
/// let mut1 = mut_view.at_mut(1).at_mut(1);
/// // We should fail to compile when trying to get a second mut ref.
/// let mut2 = mut_view.at_mut(1).at_mut(1);
///```
pub fn multiple_mut_refs_compile_test() {}

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
