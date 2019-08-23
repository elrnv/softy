mod chunked;
mod matrix;
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
pub use matrix::*;
pub use range::*;
pub use select::*;
pub use slice::*;
pub use sparse::*;
pub use subset::*;
pub use tensor::*;
pub use uniform::*;
pub use vec::*;
pub use view::*;

pub use typenum::consts;
use typenum::type_operators::PartialDiv;
use typenum::Unsigned;

/// Wrapper around `typenum` types to prevent downstream trait implementations.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct U<N>(N);

impl<N: Default> Default for U<N> {
    fn default() -> Self {
        U(N::default())
    }
}

pub trait Array<T> {
    type Array;
    fn iter_mut(array: &mut Self::Array) -> std::slice::IterMut<T>;
    fn iter(array: &Self::Array) -> std::slice::Iter<T>;
    fn as_slice(array: &Self::Array) -> &[T];
}

macro_rules! impl_array_for_typenum {
    ($nty:ident, $n:expr) => {
        pub type $nty = U<consts::$nty>;
        impl<T> Set for [T; $n] {
            type Elem = T;
            fn len(&self) -> usize {
                $n
            }
        }
        impl<'a, T: 'a> View<'a> for [T; $n] {
            type Type = &'a [T];
            fn view(&'a self) -> Self::Type {
                self
            }
        }
        impl<'a, T: 'a> ViewMut<'a> for [T; $n] {
            type Type = &'a mut [T];
            fn view_mut(&'a mut self) -> Self::Type {
                self
            }
        }
        impl<T: Dummy + Copy> Dummy for [T; $n] {
            unsafe fn dummy() -> Self {
                [Dummy::dummy(); $n]
            }
        }
        impl<T> Array<T> for consts::$nty {
            type Array = [T; $n];

            fn iter_mut(array: &mut Self::Array) -> std::slice::IterMut<T> {
                array.iter_mut()
            }
            fn iter(array: &Self::Array) -> std::slice::Iter<T> {
                array.iter()
            }
            fn as_slice(array: &Self::Array) -> &[T] {
                array
            }
        }

        impl<'a, T, N> ReinterpretAsGrouped<N> for &'a [T; $n]
        where
            N: Unsigned + Array<T>,
            consts::$nty: PartialDiv<N>,
            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
            <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array: 'a,
        {
            type Output = &'a <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
            #[inline]
            fn reinterpret_as_grouped(self) -> Self::Output {
                assert_eq!(
                    $n / N::to_usize(),
                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
                );
                unsafe {
                    &*(self as *const [T; $n]
                        as *const <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array)
                }
            }
        }

        impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut [T; $n]
        where
            N: Unsigned + Array<T>,
            consts::$nty: PartialDiv<N>,
            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
            <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array: 'a,
        {
            type Output = &'a mut <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
            #[inline]
            fn reinterpret_as_grouped(self) -> Self::Output {
                assert_eq!(
                    $n / N::to_usize(),
                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
                );
                unsafe {
                    &mut *(self as *mut [T; $n]
                        as *mut <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array)
                }
            }
        }

        // TODO: Figure out how to compile the below code.
        //        impl<T, N> ReinterpretAsGrouped<N> for [T; $n]
        //        where
        //            N: Unsigned + Array<T>,
        //            consts::$nty: PartialDiv<N>,
        //            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
        //        {
        //            type Output = <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
        //            #[inline]
        //            fn reinterpret_as_grouped(self) -> Self::Output {
        //                assert_eq!(
        //                    $n / N::to_usize(),
        //                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
        //                );
        //                unsafe {
        //                    std::mem::transmute::<
        //                        Self,
        //                        <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array,
        //                    >(self)
        //                }
        //            }
        //        }
    };
}

impl_array_for_typenum!(U1, 1);
impl_array_for_typenum!(U2, 2);
impl_array_for_typenum!(U3, 3);
impl_array_for_typenum!(U4, 4);
impl_array_for_typenum!(U5, 5);
impl_array_for_typenum!(U6, 6);
impl_array_for_typenum!(U7, 7);
impl_array_for_typenum!(U9, 9);
impl_array_for_typenum!(U10, 10);
impl_array_for_typenum!(U11, 11);
impl_array_for_typenum!(U12, 12);
impl_array_for_typenum!(U13, 13);
impl_array_for_typenum!(U14, 14);
impl_array_for_typenum!(U15, 15);
impl_array_for_typenum!(U16, 16);

impl<S: Set> Set for Box<S> {
    type Elem = S::Elem;
    fn len(&self) -> usize {
        S::len(self)
    }
}
impl<'a, S: View<'a>> View<'a> for Box<S> {
    type Type = <S as View<'a>>::Type;
    fn view(&'a self) -> Self::Type {
        S::view(self)
    }
}
impl<'a, S: ViewMut<'a>> ViewMut<'a> for Box<S> {
    type Type = <S as ViewMut<'a>>::Type;
    fn view_mut(&'a mut self) -> Self::Type {
        S::view_mut(self)
    }
}
impl<S: Dummy> Dummy for Box<S> {
    unsafe fn dummy() -> Self {
        Box::new(Dummy::dummy())
    }
}

/// A marker trait to indicate an owned collection type. This is to distinguish
/// them from borrowed slices, which essential to resolve implementation collisions.
//TODO: Rename this, since Chunked Views are also considered "Owned", this is a misnomer.
// Maybe "ValueType" makes sense
pub trait Owned {}
impl<S, T, I> Owned for Sparse<S, T, I> {}
impl<S, I> Owned for Select<S, I> {}
impl<S, I> Owned for Subset<S, I> {}
impl<S, I> Owned for Chunked<S, I> {}
impl<S, N> Owned for UniChunked<S, N> {}

impl<S: Viewed, T: Viewed, I: Viewed> Viewed for Sparse<S, T, I> {}
impl<S: Viewed, I: Viewed> Viewed for Select<S, I> {}
impl<S: Viewed, I: Viewed> Viewed for Subset<S, I> {}
impl<S: Viewed, I: Viewed> Viewed for Chunked<S, I> {}
impl<S: Viewed, N> Viewed for UniChunked<S, N> {}

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

pub trait StaticallySplittable:
    IntoStaticChunkIterator<consts::U2>
    + IntoStaticChunkIterator<consts::U3>
    + IntoStaticChunkIterator<consts::U4>
    + IntoStaticChunkIterator<consts::U5>
    + IntoStaticChunkIterator<consts::U6>
    + IntoStaticChunkIterator<consts::U7>
    + IntoStaticChunkIterator<consts::U8>
    + IntoStaticChunkIterator<consts::U9>
    + IntoStaticChunkIterator<consts::U10>
    + IntoStaticChunkIterator<consts::U11>
    + IntoStaticChunkIterator<consts::U12>
    + IntoStaticChunkIterator<consts::U13>
    + IntoStaticChunkIterator<consts::U14>
    + IntoStaticChunkIterator<consts::U15>
    + IntoStaticChunkIterator<consts::U16>
{
}

impl<T> StaticallySplittable for T where
    T: IntoStaticChunkIterator<consts::U2>
        + IntoStaticChunkIterator<consts::U3>
        + IntoStaticChunkIterator<consts::U4>
        + IntoStaticChunkIterator<consts::U5>
        + IntoStaticChunkIterator<consts::U6>
        + IntoStaticChunkIterator<consts::U7>
        + IntoStaticChunkIterator<consts::U8>
        + IntoStaticChunkIterator<consts::U9>
        + IntoStaticChunkIterator<consts::U10>
        + IntoStaticChunkIterator<consts::U11>
        + IntoStaticChunkIterator<consts::U12>
        + IntoStaticChunkIterator<consts::U13>
        + IntoStaticChunkIterator<consts::U14>
        + IntoStaticChunkIterator<consts::U15>
        + IntoStaticChunkIterator<consts::U16>
{
}

pub trait ReadSet<'a>:
    Set
    + View<'a>
    + Get<'a, usize>
    + Get<'a, std::ops::Range<usize>>
    + Isolate<usize>
    + Isolate<std::ops::Range<usize>>
    + ToOwned
    + ToOwnedData
    + SplitAt
    + SplitOff
    + SplitFirst
    + IntoFlat
    + Dummy
    + RemovePrefix
    + IntoChunkIterator
    + StaticallySplittable
    + Viewed
    + IntoIterator
{
}

impl<'a, T> ReadSet<'a> for T where
    T: Set
        + View<'a>
        + Get<'a, usize>
        + Get<'a, std::ops::Range<usize>>
        + Isolate<usize>
        + Isolate<std::ops::Range<usize>>
        + ToOwned
        + ToOwnedData
        + SplitAt
        + SplitOff
        + SplitFirst
        + IntoFlat
        + Dummy
        + RemovePrefix
        + IntoChunkIterator
        + StaticallySplittable
        + Viewed
        + IntoIterator
{
}

pub trait WriteSet<'a>: ReadSet<'a> + ViewMut<'a> {}
impl<'a, T> WriteSet<'a> for T where T: ReadSet<'a> + ViewMut<'a> {}

pub trait OwnedSet<'a>:
    Set
    + View<'a>
    + ViewMut<'a>
    + Get<'a, usize>
    + Get<'a, std::ops::Range<usize>>
    + Isolate<usize>
    + Isolate<std::ops::Range<usize>>
    + ToOwned
    + ToOwnedData
    + SplitOff
    + IntoFlat
    + Dummy
    + RemovePrefix
    + IntoChunkIterator
    + StaticallySplittable
    + Owned
{
}
impl<'a, T> OwnedSet<'a> for T where
    T: Set
        + View<'a>
        + ViewMut<'a>
        + Get<'a, usize>
        + Get<'a, std::ops::Range<usize>>
        + Isolate<usize>
        + Isolate<std::ops::Range<usize>>
        + ToOwned
        + ToOwnedData
        + SplitOff
        + IntoFlat
        + Dummy
        + RemovePrefix
        + IntoChunkIterator
        + StaticallySplittable
        + Owned
{
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

/// A helper trait to identify valid types for Range bounds for use as Sets.
pub trait IntBound:
    std::ops::Sub<Self, Output = Self>
    + std::ops::Add<usize, Output = Self>
    + Into<usize>
    + From<usize>
    + Clone
{
}

impl<T> IntBound for T where
    T: std::ops::Sub<Self, Output = Self>
        + std::ops::Add<usize, Output = Self>
        + Into<usize>
        + From<usize>
        + Clone
{
}

/// A definition of a bounded range.
pub trait BoundedRange {
    type Index: IntBound;
    fn start(&self) -> Self::Index;
    fn end(&self) -> Self::Index;
    fn distance(&self) -> Self::Index {
        self.end() - self.start()
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

impl<N: Unsigned> BoundedRange for StaticRange<N> {
    type Index = usize;
    fn start(&self) -> usize {
        self.start
    }
    fn end(&self) -> usize {
        self.start + N::to_usize()
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

/// A blanket implementation of `Isolate` for any collection which has an implementation for `IsolateIndex`.
impl<S, I> Isolate<I> for S
where
    I: IsolateIndex<Self>,
{
    type Output = I::Output;
    fn try_isolate(self, idx: I) -> Option<Self::Output> {
        idx.try_isolate(self)
    }
}

impl<'a, S, N> GetIndex<'a, S> for StaticRange<N>
where
    S: Set + Owned,
    N: Unsigned,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (self.start..self.start + N::to_usize()).get(set)
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

impl<S, N: Unsigned> IsolateIndex<S> for StaticRange<N>
where
    S: Set + Owned,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(self.start..self.start + N::to_usize(), set)
    }
}

impl<S> IsolateIndex<S> for std::ops::RangeFrom<usize>
where
    S: Set + Owned,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(self.start..set.len(), set)
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..self.end, set)
    }
}

impl<S: Owned> IsolateIndex<S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..set.len(), set)
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
            IsolateIndex::try_isolate(*self.start()..*self.end() + 1, set)
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
        IsolateIndex::try_isolate(0..=self.end, set)
    }
}

impl<N: Unsigned> Set for StaticRange<N> {
    type Elem = usize;
    fn len(&self) -> usize {
        N::to_usize()
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

/// Truncate the collection to be a specified length.
pub trait Truncate {
    fn truncate(&mut self, len: usize);
}

/// Split off a number of elements from the beginning of the collection where the number is determined at compile time.
pub trait SplitPrefix<N>
where
    Self: Sized,
{
    type Prefix;
    fn split_prefix(self) -> Option<(Self::Prefix, Self)>;
}

/// Split out the first element of a collection.
pub trait SplitFirst
where
    Self: Sized,
{
    type First;
    fn split_first(self) -> Option<(Self::First, Self)>;
}

//TODO: Rename to IntoStorage
/// Convert a collection into its underlying representation, effectively
/// stripping any organizational info.
pub trait IntoFlat {
    type FlatType;
    fn into_flat(self) -> Self::FlatType;
}

/// Get an immutable reference to the underlying storage type.
pub trait Storage {
    type Storage;
    fn storage(&self) -> &Self::Storage;
}

/// Get a mutable reference to the underlying storage type.
pub trait StorageMut: Storage {
    fn storage_mut(&mut self) -> &mut Self::Storage;
}

/// Transform the access pattern of the underlying storage type. This is useful
/// when the storage is not just a simple `Vec` or slice but a combination of independent
/// collections.
pub trait StorageInto<Target> {
    type Output;
    fn storage_into(self) -> Self::Output;
}

pub trait CloneWithFlat<FlatType> {
    type CloneType;
    fn clone_with_flat(&self, flat: FlatType) -> Self::CloneType;
}

/// A helper trait for constructing placeholder sets for use in `std::mem::replace`.
/// These don't necessarily have to correspond to bona-fide sets and can
/// potentially produce invalid sets. For this reason this function can be
/// unsafe since it can generate collections that don't uphold their invariants
/// for the sake of avoiding allocations.
pub trait Dummy {
    unsafe fn dummy() -> Self;
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

    /// Produce a chunk iterator with the given stride `chunk_size`.
    /// One notable difference between this trait and `chunks*` methods on slices is that
    /// `chunks_iter` should panic when the underlying data cannot split into `chunk_size` sized
    /// chunks exactly.
    fn into_chunk_iter(self, chunk_size: usize) -> Self::IterType;
}

// Implement IntoChunkIterator for all types that implement Set, SplitAt and Dummy.
// This should be reimplemented like IntoStaticChunkIterator to avoid expensive iteration of allocating types.
impl<S> IntoChunkIterator for S
where
    S: Set + SplitAt + Dummy,
{
    type Item = S;
    type IterType = ChunkedNIter<S>;

    fn into_chunk_iter(self, chunk_size: usize) -> Self::IterType {
        assert_eq!(self.len() % chunk_size, 0);
        ChunkedNIter {
            chunk_size,
            data: self,
        }
    }
}

pub struct ChunkedNIter<S> {
    chunk_size: usize,
    data: S,
}

impl<S> Iterator for ChunkedNIter<S>
where
    S: Set + SplitAt + Dummy,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        let (l, r) = data_slice.split_at(self.chunk_size);
        self.data = r;
        Some(l)
    }
}

/// Iterate over chunks whose size is determined at compile time.
/// Note that each chunk may not be a simple array, although a statically sized
/// chunk of a slice is an array.
pub trait IntoStaticChunkIterator<N>
where
    Self: Sized + Set + Dummy,
    N: Unsigned,
{
    type Item;
    type IterType: Iterator<Item = Self::Item>;

    /// This function panics if this collection length is not a multiple of `N`.
    fn into_static_chunk_iter(self) -> Self::IterType;

    /// Simply call this method for all types that implement `SplitPrefix<N>`.
    #[inline]
    fn into_generic_static_chunk_iter(self) -> UniChunkedIter<Self, N> {
        assert_eq!(self.len() % N::to_usize(), 0);
        UniChunkedIter {
            chunk_size: std::marker::PhantomData,
            data: self,
        }
    }
}

/// Generic static sized chunk iterater appropriate for any lightweight view type collection.
pub struct UniChunkedIter<S, N> {
    chunk_size: std::marker::PhantomData<N>,
    data: S,
}

impl<S, N> Iterator for UniChunkedIter<S, N>
where
    S: Set + SplitPrefix<N> + Dummy,
    N: Unsigned,
{
    type Item = S::Prefix;

    fn next(&mut self) -> Option<Self::Item> {
        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        data_slice.split_prefix().map(|(prefix, rest)| {
            self.data = rest;
            prefix
        })
    }
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
        let u0 = Chunked2::from_flat((1..=12).collect::<Vec<_>>());
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
