use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

/// A `Sparse` data set `S` where the sparsity pattern is given by `I` as select
/// indices into a larger range.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sparse<S, T, I = Vec<usize>> {
    pub(crate) selection: Select<T, I>,
    pub(crate) source: S,
}

/// A borrowed view of a sparse collection.
pub type SparseView<'a, S, T> = Sparse<S, T, &'a [usize]>;

impl<S, I> Sparse<S, Range<usize>, I>
where
    S: Set,
    I: AsRef<[usize]>,
{
    /// Create a sparse collection from the given set of `indices`, a
    /// `dim`ension and a set of `values`.
    /// The corresponding sparse collection will represent a collection
    /// of size `dim` which stores only the given `values` at the specified
    /// `indices`. Note that `dim` may be smaller than `values.len()`, in
    /// which case a position in the sparse data structure may contain multiple
    /// values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let sparse = Sparse::from_dim(vec![0,2,0,2,0,3], 4, v.as_slice());
    ///
    /// // The iterator traverses only non-vacant elements.
    /// let mut iter = sparse.iter(); // Returns (position, source, target) pairs
    /// assert_eq!(Some((0, &1, 0)), iter.next());
    /// assert_eq!(Some((2, &2, 2)), iter.next());
    /// assert_eq!(Some((0, &3, 0)), iter.next());
    /// assert_eq!(Some((2, &4, 2)), iter.next());
    /// assert_eq!(Some((0, &5, 0)), iter.next());
    /// assert_eq!(Some((3, &6, 3)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_dim(indices: I, dim: usize, values: S) -> Self {
        Sparse::new(Select::new(indices, 0..dim), values)
    }
}

impl<S, T, I> Sparse<S, T, I>
where
    S: Set,
    T: Set,
    I: AsRef<[usize]>,
{
    /// The most general constructor for a sparse collection taking a selection
    /// of values and their corresponding data.
    ///
    /// # Panics
    /// This function will panic if `selection` and `source` have different sizes.
    pub fn new(selection: Select<T, I>, source: S) -> Self {
        Self::validate(Sparse { selection, source })
    }

    /// Panics if the number of indices doesn't equal to the number of elements in the source data set.
    #[inline]
    fn validate(self) -> Self {
        assert_eq!(self.source.len(), self.selection.len());
        self
    }
}

/*
impl<S, T, I> Sparse<S, T, I>
where
    S: Set + Default + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Default,
{
    /// Convert this sparse collection into its dense representation.
    pub fn dense(&self) -> S {
        // TODO: This can be optimized with a trait that allows pre-allocating memory.
        let mut dense = S::default();
        for (i, v) in self.iter() {
            while dense.len() < i {
                dense.push(Default::default());
            }
            dense.push(v);
        }

        dense
    }
}
*/

impl<'a, S, T, I> Sparse<S, T, I> {
    /// Get a reference to the underlying source data.
    pub fn source(&self) -> &S {
        &self.source
    }
    /// Get a mutable reference to the underlying source data.
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
    /// Get a reference to the underlying selection.
    pub fn selection(&self) -> &Select<T, I> {
        &self.selection
    }

    pub fn selection_mut(&mut self) -> &mut Select<T, I> {
        &mut self.selection
    }

    /// Get a reference to the underlying indices.
    pub fn indices(&self) -> &I {
        &self.selection.indices
    }

    pub fn indices_mut(&mut self) -> &mut I {
        &mut self.selection.indices
    }
}

// Note to self:
// To enable a collection to be chunked, we need to implement:
// Set, View, SplitAt
// For mutability we also need ViewMut,
// For UniChunked we need:
// Set, Vew, ReinterpretSet (this needs to be refined)

// Required for `Chunked` and `UniChunked` subsets.
impl<S: Set, T, I> Set for Sparse<S, T, I> {
    type Elem = S::Elem;
    /// Get the length of this sparse collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let sparse = Sparse::from_dim(vec![0,2,2,1,1], 3, v.as_slice());
    /// assert_eq!(5, sparse.len());
    /// ```
    fn len(&self) -> usize {
        self.source.len()
    }
}

// Required for `Chunked` and `UniChunked` subsets.
impl<'a, S, T, I> View<'a> for Sparse<S, T, I>
where
    S: View<'a>,
    T: View<'a>,
    I: AsRef<[usize]>,
    //Select<<T as View<'a>>::Type, &'a [usize]>: IntoIterator,
    //Sparse<<S as View<'a>>::Type, <T as View<'a>>::Type, &'a [usize]>: IntoIterator,
{
    type Type = Sparse<S::Type, T::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        Sparse {
            selection: self.selection.view(),
            source: self.source.view(),
        }
    }
}

impl<'a, S, T, I> ViewMut<'a> for Sparse<S, T, I>
where
    S: Set + ViewMut<'a>,
    T: Set + View<'a>,
    I: AsMut<[usize]>,
    //Sparse<<S as ViewMut<'a>>::Type, <T as View<'a>>::Type, &'a [usize]>: IntoIterator,
{
    type Type = Sparse<S::Type, T::Type, &'a mut [usize]>;
    fn view_mut(&'a mut self) -> Self::Type {
        let Sparse {
            selection: Select {
                indices,
                ref target,
            },
            source,
        } = self;
        Sparse {
            selection: Select {
                indices: indices.as_mut(),
                target: target.view(),
            },
            source: source.view_mut(),
        }
    }
}

// This impl enables `Chunked` `Subset`s
impl<S, T, I> SplitAt for Sparse<S, T, I>
where
    S: Set + SplitAt,
    T: Set + Clone,
    I: SplitAt,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Sparse { selection, source } = self;
        let (selection_l, selection_r) = selection.split_at(mid);
        let (source_l, source_r) = source.split_at(mid);
        (
            Sparse {
                selection: selection_l,
                source: source_l,
            },
            Sparse {
                selection: selection_r,
                source: source_r,
            },
        )
    }
}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

impl<'a, S, T, I> GetIndex<'a, Sparse<S, T, I>> for usize
where
    I: AsRef<[usize]>,
    S: Get<'a, usize>,
{
    type Output = (usize, <S as Get<'a, usize>>::Output);

    fn get(self, sparse: &Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, source } = sparse;
        let selected = selection.indices.as_ref();
        source.get(self).map(|item| (selected[self], item))
    }
}

impl<S, T, I> IsolateIndex<Sparse<S, T, I>> for usize
where
    I: Isolate<usize>,
    <I as Isolate<usize>>::Output: std::borrow::Borrow<usize>,
    S: Isolate<usize>,
    T: Isolate<usize>,
{
    type Output = (
        <I as Isolate<usize>>::Output,
        <S as Isolate<usize>>::Output,
        <T as Isolate<usize>>::Output,
    );

    fn try_isolate(self, sparse: Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, source } = sparse;
        source
            .try_isolate(self)
            // TODO: selection.isolate can be unchecked.
            .map(|item| {
                let (idx, target) = selection.isolate(self);
                (idx, item, target)
            })
    }
}

impl<S, T, I> IsolateIndex<Sparse<S, T, I>> for std::ops::Range<usize>
where
    S: Isolate<std::ops::Range<usize>>,
    I: Isolate<std::ops::Range<usize>>,
{
    type Output = Sparse<S::Output, T, I::Output>;

    fn try_isolate(self, sparse: Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, source } = sparse;
        source.try_isolate(self.clone()).and_then(|source| {
            // TODO: selection.try_isolate can be unchecked.
            selection
                .try_isolate(self)
                .map(|selection| Sparse { selection, source })
        })
    }
}

//impl<S, T, I, Idx> Isolate<Idx> for Sparse<S, T, I>
//where
//    Idx: IsolateIndex<Self>,
//{
//    type Output = Idx::Output;
//
//    fn try_isolate(self, range: Idx) -> Option<Self::Output> {
//        range.try_isolate(self)
//    }
//}

/*
 * Iteration
 */

impl<'a, S, T> IntoIterator for SparseView<'a, S, T>
where
    S: SplitFirst + Dummy,
{
    type Item = (usize, S::First);
    type IntoIter = SparseIter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        SparseIter {
            indices: self.selection.indices,
            source: self.source,
        }
    }
}

pub struct SparseIter<'a, S> {
    indices: &'a [usize],
    source: S,
}

impl<'a, S> Iterator for SparseIter<'a, S>
where
    S: SplitFirst + Dummy,
{
    type Item = (usize, S::First);

    fn next(&mut self) -> Option<Self::Item> {
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        source_slice.split_first().and_then(|(first, rest)| {
            self.source = rest;
            self.indices.split_first().map(|(first_idx, rest_indices)| {
                self.indices = rest_indices;
                (*first_idx, first)
            })
        })
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set,
    T: Set + Get<'a, usize> + View<'a>,
    I: AsRef<[usize]>,
    <S as View<'a>>::Type: IntoIterator,
    <T as View<'a>>::Type: IntoIterator<Item = T::Output>,
{
    pub fn iter(
        &'a self,
    ) -> impl Iterator<
        Item = (
            usize,
            <<S as View<'a>>::Type as IntoIterator>::Item,
            <T as Get<'a, usize>>::Output,
        ),
    > {
        self.selection
            .iter()
            .zip(self.source.view().into_iter())
            .map(|((i, t), s)| (i, s, t))
    }
}

/// A mutable iterator can only iterate over the source elements in `S` and not
/// the target elements in `T` since we would need scheduling to modify
/// potentially overlapping mutable references.
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoIterator,
    I: AsRef<[usize]>,
{
    pub fn source_iter_mut(
        &'a mut self,
    ) -> impl Iterator<Item = (usize, <<S as ViewMut<'a>>::Type as IntoIterator>::Item)> {
        self.selection
            .index_iter()
            .cloned()
            .zip(self.source.view_mut().into_iter())
    }
}

/// A mutable iterator can only iterate over the source elements in `S` and not
/// the target elements in `T` since we would need scheduling to modify
/// potentially overlapping mutable references.
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoIterator,
    I: AsMut<[usize]>,
{
    pub fn iter_mut(
        &'a mut self,
    ) -> impl Iterator<
        Item = (
            &'a mut usize,
            <<S as ViewMut<'a>>::Type as IntoIterator>::Item,
        ),
    > {
        self.selection
            .index_iter_mut()
            .zip(self.source.view_mut().into_iter())
    }
}

/// Mutably iterate over the selected indices.
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoIterator,
    I: AsMut<[usize]>,
{
    pub fn index_iter_mut(
        &'a mut self,
    ) -> impl Iterator<Item = (&'a mut usize, <<S as View<'a>>::Type as IntoIterator>::Item)> {
        self.selection
            .index_iter_mut()
            .zip(self.source.view().into_iter())
    }
}

impl<S: Dummy, T: Dummy, I: Dummy> Dummy for Sparse<S, T, I> {
    unsafe fn dummy() -> Self {
        Sparse {
            selection: Dummy::dummy(),
            source: Dummy::dummy(),
        }
    }
}

impl<S: Truncate, T, I: Truncate> Truncate for Sparse<S, T, I> {
    fn truncate(&mut self, new_len: usize) {
        self.selection.truncate(new_len);
        self.source.truncate(new_len);
    }
}

/*
 * Conversions
 */

/// Pass through the conversion for structure type `Subset`.
impl<S: StorageInto<U>, T, I, U> StorageInto<U> for Sparse<S, T, I> {
    type Output = Sparse<S::Output, T, I>;
    fn storage_into(self) -> Self::Output {
        Sparse {
            source: self.source.storage_into(),
            selection: self.selection,
        }
    }
}

impl<S: IntoFlat, T, I> IntoFlat for Sparse<S, T, I> {
    type FlatType = S::FlatType;
    /// Convert the sparse set into its raw storage representation.
    fn into_flat(self) -> Self::FlatType {
        self.source.into_flat()
    }
}

impl<T: Clone, S: CloneWithStorage<U>, I: Clone, U> CloneWithStorage<U> for Sparse<S, T, I> {
    type CloneType = Sparse<S::CloneType, T, I>;
    fn clone_with_storage(&self, storage: U) -> Self::CloneType {
        Sparse {
            selection: self.selection.clone(),
            source: self.source.clone_with_storage(storage),
        }
    }
}

/*
 * Storage Access
 */

impl<S: Storage, T, I> Storage for Sparse<S, T, I> {
    type Storage = S::Storage;
    /// Return an immutable reference to the underlying storage type of source data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let s0 = Chunked3::from_flat(v.clone());
    /// let s1 = Sparse::from_dim(vec![0, 2, 2, 0], 4, s0.clone());
    /// assert_eq!(s1.storage(), &v);
    /// ```
    fn storage(&self) -> &Self::Storage {
        self.source.storage()
    }
}

impl<S: StorageMut, T, I> StorageMut for Sparse<S, T, I> {
    /// Return a mutable reference to the underlying storage type of source data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let mut s0 = Chunked3::from_flat(v.clone());
    /// let mut s1 = Sparse::from_dim(vec![0, 2, 2, 0], 4, s0.clone());
    /// assert_eq!(s1.storage_mut(), &mut v);
    /// ```
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self.source.storage_mut()
    }
}

impl<S: PermuteInPlace, T, I: PermuteInPlace> PermuteInPlace for Sparse<S, T, I> {
    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
        let Sparse {
            selection: Select { indices, .. },
            source,
        } = self;

        indices.permute_in_place(permutation, seen);
        seen.iter_mut().for_each(|x| *x = false);
        source.permute_in_place(permutation, seen);
    }
}
