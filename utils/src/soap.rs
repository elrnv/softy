/*
 * Define macros to be used for implementing various traits in submodules
 */

macro_rules! impl_atom_iterators_recursive {
    (impl<S, $($type_vars_decl:tt),*> for $type:ident<S, $($type_vars:tt),*> { $data_field:ident }) => {
        impl<'a, S, $($type_vars_decl,)*> AtomIterator<'a> for $type<S, $($type_vars,)*>
        where S: AtomIterator<'a>,
        {
            type Item = S::Item;
            type Iter = S::Iter;
            fn atom_iter(&'a self) -> Self::Iter {
                self.$data_field.atom_iter()
            }
        }

        impl<'a, S, $($type_vars_decl,)*> AtomMutIterator<'a> for $type<S, $($type_vars,)*>
        where S: AtomMutIterator<'a>
        {
            type Item = S::Item;
            type Iter = S::Iter;
            fn atom_mut_iter(&'a mut self) -> Self::Iter {
                self.$data_field.atom_mut_iter()
            }
        }
    }
}

macro_rules! impl_isolate_index_for_static_range {
    (impl<$($type_vars:ident),*> for $type:ty) => {
        impl_isolate_index_for_static_range! { impl<$($type_vars),*> for $type where }
    };
    (impl<$($type_vars:ident),*> for $type:ty where $($constraints:tt)*) => {
        impl<$($type_vars,)* N: Unsigned> IsolateIndex<$type> for StaticRange<N>
        where
            std::ops::Range<usize>: IsolateIndex<$type>,
            $($constraints)*
        {
            type Output = <std::ops::Range<usize> as IsolateIndex<$type>>::Output;

            fn try_isolate(self, set: $type) -> Option<Self::Output> {
                IsolateIndex::try_isolate(self.start..self.start + N::to_usize(), set)
            }
        }
    }
}

mod array;
mod array_math;
mod boxed;
pub mod chunked;
mod matrix;
mod range;
mod select;
mod slice;
mod sparse;
mod subset;
mod tensor;
mod tuple;
mod uniform;
mod vec;
mod view;

pub use array::*;
pub use array_math::*;
pub use boxed::*;
pub use chunked::*;
pub use matrix::*;
pub use range::*;
pub use select::*;
pub use slice::*;
pub use sparse::*;
pub use subset::*;
pub use tensor::*;
pub use tuple::*;
pub use uniform::*;
pub use vec::*;
pub use view::*;

pub use typenum::consts;
use typenum::type_operators::PartialDiv;
use typenum::Unsigned;

/*
 * Set is the most basic trait that annotates finite collections that contain data.
 */
/// A trait defining a raw buffer of data. This data is typed but not annotated so it can represent
/// anything. For example a buffer of floats can represent a set of vertex colours or vertex
/// positions.
pub trait Set {
    /// Owned element of the set.
    type Elem;
    /// The most basic element contained by this collection.
    /// If this collection contains other collections, this type should be
    /// different than `Elem`.
    type Atom;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<N: Unsigned> Set for StaticRange<N> {
    type Elem = usize;
    type Atom = usize;
    fn len(&self) -> usize {
        N::to_usize()
    }
}

impl<S: Set + ?Sized> Set for &S {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}
impl<S: Set + ?Sized> Set for &mut S {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

impl<S: Set + ?Sized> Set for std::cell::Ref<'_, S> {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

impl<S: Set + ?Sized> Set for std::cell::RefMut<'_, S> {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

/*
 * Array manipulation
 */

pub trait Array<T> {
    type Array: Set<Elem = T>;
    fn iter_mut(array: &mut Self::Array) -> std::slice::IterMut<T>;
    fn iter(array: &Self::Array) -> std::slice::Iter<T>;
    fn as_slice(array: &Self::Array) -> &[T];
}

/*
 * Marker and utility traits to help with Coherence rules of Rust.
 */

/// A marker trait for local types which use generic implementations of various std::ops traits.
/// Special types which use optimized implementations will not implement this marker. This is a
/// workaround for specialization.
pub trait LocalGeneric {}
impl<S, I> LocalGeneric for Subset<S, I> {}
impl<S, N> LocalGeneric for UniChunked<S, N> {}
impl<S, O> LocalGeneric for Chunked<S, O> {}
impl<S, T, I> LocalGeneric for Sparse<S, T, I> {}

/// A marker trait to indicate an owned collection type. This is to distinguish
/// them from borrowed slices, which essential to resolve implementation collisions.
pub trait ValueType {}
impl<S, T, I> ValueType for Sparse<S, T, I> {}
impl<S, I> ValueType for Select<S, I> {}
impl<S, I> ValueType for Subset<S, I> {}
impl<S, I> ValueType for Chunked<S, I> {}
impl<S, N> ValueType for UniChunked<S, N> {}

impl<S: Viewed, T: Viewed, I: Viewed> Viewed for Sparse<S, T, I> {}
impl<S: Viewed, I: Viewed> Viewed for Select<S, I> {}
impl<S: Viewed, I: Viewed> Viewed for Subset<S, I> {}
impl<S: Viewed, I: Viewed> Viewed for Chunked<S, I> {}
impl<S: Viewed, N> Viewed for UniChunked<S, N> {}

/// A marker trait identifying types that have dynamically determined sizes. These are basically all non-array types.
pub trait DynamicCollection {}
impl<S, I> DynamicCollection for Select<S, I> {}
impl<S, I> DynamicCollection for Subset<S, I> {}
impl<S, N> DynamicCollection for UniChunked<S, N> {}
impl<S, O> DynamicCollection for Chunked<S, O> {}
impl<S, T, I> DynamicCollection for Sparse<S, T, I> {}
impl<T> DynamicCollection for std::ops::Range<T> {}
impl<T> DynamicCollection for Vec<T> {}
impl<T> DynamicCollection for [T] {}
impl<'a, T> DynamicCollection for &'a [T] {}
impl<'a, T> DynamicCollection for &'a mut [T] {}

/// Many implementations do something special for sparse sets. All other sets are marked as dense
/// to avoid collisions.
pub trait Dense {}
impl<S, I> Dense for Select<S, I> {}
impl<S, I> Dense for Subset<S, I> {}
impl<S, N> Dense for UniChunked<S, N> {}
impl<S, O> Dense for Chunked<S, O> {}
impl<T> Dense for std::ops::Range<T> {}
impl<T> Dense for Vec<T> {}
impl<T> Dense for [T] {}
impl<'a, T> Dense for &'a [T] {}
impl<'a, T> Dense for &'a mut [T] {}

/// A marker trait to indicate a collection type that can be chunked. More precisely this is a type that can be composed with types in this crate.
//pub trait Chunkable<'a>:
//    Set + Get<'a, 'a, std::ops::Range<usize>> + RemovePrefix + View<'a> + PartialEq
//{
//}
//impl<'a, T: Clone + PartialEq> Chunkable<'a> for &'a [T] {}
//impl<'a, T: Clone + PartialEq> Chunkable<'a> for &'a mut [T] {}
//impl<'a, T: Clone + PartialEq + 'a> Chunkable<'a> for Vec<T> {}

/*
 * Aggregate traits
 */

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
    + IntoOwned
    + IntoOwnedData
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
        + IntoOwned
        + IntoOwnedData
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
    + IntoOwned
    + IntoOwnedData
    + SplitOff
    + IntoFlat
    + Dummy
    + RemovePrefix
    + IntoChunkIterator
    + StaticallySplittable
    + ValueType
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
        + IntoOwned
        + IntoOwnedData
        + SplitOff
        + IntoFlat
        + Dummy
        + RemovePrefix
        + IntoChunkIterator
        + StaticallySplittable
        + ValueType
{
}

/*
 * Allocation
 */

/// Abstraction for pushing elements of type `T` onto a collection.
pub trait Push<T> {
    fn push(&mut self, element: T);
}

pub trait ExtendFromSlice {
    type Item;
    fn extend_from_slice(&mut self, other: &[Self::Item]);
}

/*
 * Deallocation
 */

/// Truncate the collection to be a specified length.
pub trait Truncate {
    fn truncate(&mut self, len: usize);
}

pub trait Clear {
    /// Remove all elements from the current set without necessarily
    /// deallocating the space previously used.
    fn clear(&mut self);
}

/*
 * Conversion
 */

//TODO: Rename to IntoStorage
/// Convert a collection into its underlying representation, effectively
/// stripping any organizational info.
pub trait IntoFlat {
    type FlatType;
    fn into_flat(self) -> Self::FlatType;
}

/// Transform the access pattern of the underlying storage type. This is useful
/// when the storage is not just a simple `Vec` or slice but a combination of independent
/// collections.
pub trait StorageInto<Target> {
    type Output;
    fn storage_into(self) -> Self::Output;
}

pub trait CloneWithStorage<S> {
    type CloneType;
    fn clone_with_storage(&self, storage: S) -> Self::CloneType;
}

/// An analog to the `ToOwned` trait from `std` that works for chunked views.
/// As the name suggests, this version of `ToOwned` takes `self` by value.
pub trait IntoOwned
where
    Self: Sized,
{
    type Owned;
    fn into_owned(self) -> Self::Owned;
    fn clone_into(self, target: &mut Self::Owned) {
        *target = self.into_owned();
    }
}

/// Blanket implementation of `IntoOwned` for references of types that are already
/// `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwned for &S {
    type Owned = S::Owned;
    fn into_owned(self) -> Self::Owned {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// Blanket implementation of `IntoOwned` for mutable references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwned for &mut S {
    type Owned = S::Owned;
    fn into_owned(self) -> Self::Owned {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// In contrast to `IntoOwned`, this trait produces a clone with owned data, but
/// potentially borrowed structure of the collection.
pub trait IntoOwnedData
where
    Self: Sized,
{
    type OwnedData;
    fn into_owned_data(self) -> Self::OwnedData;
    fn clone_into(self, target: &mut Self::OwnedData) {
        *target = self.into_owned_data();
    }
}

/// Blanked implementation of `IntoOwnedData` for references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwnedData for &S {
    type OwnedData = S::Owned;
    fn into_owned_data(self) -> Self::OwnedData {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// Blanked implementation of `IntoOwnedData` for mutable references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwnedData for &mut S {
    type OwnedData = S::Owned;
    fn into_owned_data(self) -> Self::OwnedData {
        std::borrow::ToOwned::to_owned(self)
    }
}

/*
 * Indexing
 */

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
    S: Set + ValueType,
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
    S: Set + ValueType,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (self.start..set.len()).get(set)
    }
}

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (0..self.end).get(set)
    }
}

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (0..set.len()).get(set)
    }
}

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeInclusive<usize>
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

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeToInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    fn get(self, set: &S) -> Option<Self::Output> {
        (0..=self.end).get(set)
    }
}

//impl<S, N: Unsigned> IsolateIndex<S> for StaticRange<N>
//where
//    S: Set + ValueType,
//    std::ops::Range<usize>: IsolateIndex<S>,
//{
//    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;
//
//    fn try_isolate(self, set: S) -> Option<Self::Output> {
//        IsolateIndex::try_isolate(self.start..self.start + N::to_usize(), set)
//    }
//}

impl<S> IsolateIndex<S> for std::ops::RangeFrom<usize>
where
    S: Set + ValueType,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(self.start..set.len(), set)
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..self.end, set)
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..set.len(), set)
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeInclusive<usize>
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

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeToInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..=self.end, set)
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

/// Get an immutable reference to the underlying storage type.
pub trait Storage {
    type Storage: ?Sized;
    fn storage(&self) -> &Self::Storage;
}

pub trait StorageView<'a> {
    type StorageView;
    fn storage_view(&'a self) -> Self::StorageView;
}

/// Get a mutable reference to the underlying storage type.
pub trait StorageMut: Storage {
    fn storage_mut(&mut self) -> &mut Self::Storage;
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
// Technically this is a deallocation trait, but it's only used to enable
// iteration on subsets so it's here.
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

/// A trait intended to be implemented on collection types to define the type of
/// a statically sized chunk in this collection.
/// This trait is required for composing with `UniChunked`.
pub trait UniChunkable<N> {
    type Chunk;
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

    /// This function should panic if this collection length is not a multiple
    /// of `N`.
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

/// A trait that allows the container to allocate additional space without
/// changing any of the data. The container should allocate space for at least
/// `n` additional elements.
///
/// Composite collections such a `Chunked` or `Select` may choose to only
/// reserve primary level storage if the amount of total storage required cannot
/// be specified by a single number in `reserve`. This is the default behaviour
/// of the `reserve` function below. The `reserve_with_storage` method allows
/// the caller to also specify the amount of storage needed for the container at
/// the lowest level.
pub trait Reserve {
    fn reserve(&mut self, n: usize) {
        self.reserve_with_storage(n, 0); // By default we ignore storage.
    }
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize);
}

/*
 * New experimental traits below
 */

pub trait SwapChunks {
    /// Swap equal sized contiguous chunks in this collection.
    fn swap_chunks(&mut self, begin_a: usize, begin_b: usize, chunk_size: usize);
}

pub trait Sort {
    /// Sort the given indices into this collection with respect to values provided by this collection.
    fn sort_indices(&self, indices: &mut [usize]);
}

pub trait PermuteInPlace {
    fn permute_in_place(&mut self, indices: &[usize], seen: &mut [bool]);
}

/// This trait is used to produce the chunk size of a collection if it contains uniformly chunked
/// elements.
pub trait ChunkSize {
    fn chunk_size(&self) -> usize;
}

/// Clone self into a potentially different collection.
pub trait CloneIntoOther<T = Self>
where
    T: ?Sized,
{
    fn clone_into_other(&self, other: &mut T);
}

impl<T: Clone> CloneIntoOther<&mut T> for &T {
    fn clone_into_other(&self, other: &mut &mut T) {
        other.clone_from(self);
    }
}

pub trait AtomIterator<'a> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn atom_iter(&'a self) -> Self::Iter;
}

pub trait AtomMutIterator<'a> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn atom_mut_iter(&'a mut self) -> Self::Iter;
}

// Blanket implementations of AtomIterator/AtomMutIterator for references
impl<'a, S: ?Sized> AtomIterator<'a> for &S
where
    S: AtomIterator<'a>,
{
    type Item = S::Item;
    type Iter = S::Iter;
    fn atom_iter(&'a self) -> Self::Iter {
        S::atom_iter(self)
    }
}

impl<'a, S: ?Sized> AtomMutIterator<'a> for &mut S
where
    S: AtomMutIterator<'a>,
{
    type Item = S::Item;
    type Iter = S::Iter;
    fn atom_mut_iter(&'a mut self) -> Self::Iter {
        S::atom_mut_iter(self)
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
