use super::*;
use std::ops::Range;

/// A `Sparse` data set `S` where the sparsity pattern is given by `I` as select
/// indices into a larger range.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sparse<S, T, I = Vec<usize>> {
    pub(crate) selection: Select<T, I>,
    pub(crate) data: S,
}

/// A borrowed view of a sparse collection.
pub type SparseView<'a, S, T> = Sparse<S, T, &'a [usize]>;

impl<S, I> Sparse<S, Range<usize>, I>
where
    S: Set,
    I: std::borrow::Borrow<[usize]>,
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
    I: std::borrow::Borrow<[usize]>,
{
    /// The most general constructor for a sparse collection taking a selection
    /// of values and their corresponding data.
    ///
    /// # Panics
    /// This function will panic if `selection` and `data` have different sizes.
    pub fn new(selection: Select<T, I>, data: S) -> Self {
        Self::validate(Sparse { selection, data })
    }

    /// Panics if the number of indices doesn't equal to the number of elements in the data set.
    #[inline]
    fn validate(self) -> Self {
        assert_eq!(self.data.len(), self.selection.len());
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
    /// Get a reference to the underlying selection.
    pub fn selection(&self) -> &Select<T, I> {
        &self.selection
    }

    /// Get a reference to the underlying indices.
    pub fn indices(&self) -> &I {
        self.selection.indices()
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
        self.data.len()
    }
}

// Required for `Chunked` and `UniChunked` subsets.
impl<'a, S, T, I> View<'a> for Sparse<S, T, I>
where
    S: Set + View<'a>,
    T: Set + View<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: Set,
    <T as View<'a>>::Type: Set,
{
    type Type = Sparse<S::Type, T::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        Sparse {
            selection: self.selection.view(),
            data: self.data.view(),
        }
    }
}

impl<'a, S, T, I> ViewMut<'a> for Sparse<S, T, I>
where
    S: Set + ViewMut<'a>,
    T: Set + ViewMut<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as ViewMut<'a>>::Type: Set,
    <T as ViewMut<'a>>::Type: Set,
{
    type Type = Sparse<S::Type, T::Type, &'a [usize]>;
    fn view_mut(&'a mut self) -> Self::Type {
        let Sparse { selection, data } = self;
        Sparse {
            selection: selection.view_mut(),
            data: data.view_mut(),
        }
    }
}

// This impl enables `Chunked` `Subset`s
impl<S, T> SplitAt for Sparse<S, T, &[usize]>
where
    S: Set + SplitAt,
    T: Set + Clone,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Sparse { selection, data } = self;
        let (selection_l, selection_r) = selection.split_at(mid);
        let (data_l, data_r) = data.split_at(mid);
        (
            Sparse {
                selection: selection_l,
                data: data_l,
            },
            Sparse {
                selection: selection_r,
                data: data_r,
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
    I: std::borrow::Borrow<[usize]>,
    S: Get<'a, usize>,
{
    type Output = (usize, <S as Get<'a, usize>>::Output);

    fn get(self, sparse: &Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, data } = sparse;
        let selected = selection.indices.borrow();
        data.get(self).map(|item| (selected[self], item))
    }
}

impl<S, T, I> IsolateIndex<Sparse<S, T, I>> for usize
where
    I: std::borrow::Borrow<[usize]>,
    S: Isolate<usize>,
{
    type Output = (usize, <S as Isolate<usize>>::Output);

    fn try_isolate(self, sparse: Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, data } = sparse;
        data.try_isolate(self)
            .map(|item| (selection.indices().borrow()[self], item))
    }
}

impl<'a, S, T, I, Idx> Get<'a, Idx> for Sparse<S, T, I>
where
    Idx: GetIndex<'a, Self>,
{
    type Output = Idx::Output;

    fn get(&self, range: Idx) -> Option<Self::Output> {
        range.get(self)
    }
}

impl<S, T, I, Idx> Isolate<Idx> for Sparse<S, T, I>
where
    Idx: IsolateIndex<Self>,
{
    type Output = Idx::Output;

    fn try_isolate(self, range: Idx) -> Option<Self::Output> {
        range.try_isolate(self)
    }
}

/*
 * Iteration
 */

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: Set + Get<'a, usize> + View<'a>,
    T: Set + Get<'a, usize> + View<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: IntoIterator<Item = S::Output>,
    <T as View<'a>>::Type: IntoIterator<Item = T::Output>,
{
    pub fn iter(
        &'a self,
    ) -> impl Iterator<
        Item = (
            usize,
            <S as Get<'a, usize>>::Output,
            <T as Get<'a, usize>>::Output,
        ),
    > {
        self.selection
            .iter()
            .zip(self.data.view().into_iter())
            .map(|((i, t), s)| (i, s, t))
    }
}
/*

pub struct SubsetIterMut<'a, V> {
    indices: Option<&'a [usize]>,
    data: V,
}

impl<'a, V: 'a> Iterator for SubsetIterMut<'a, V>
where
    V: SplitAt + SplitFirst + Set + Dummy,
{
    type Item = V::First;

    fn next(&mut self) -> Option<Self::Item> {
        let SubsetIterMut { indices, data } = self;
        let data_slice = std::mem::replace(data, Dummy::dummy());
        match indices {
            Some(ref mut indices) => indices.split_first().map(|(first, rest)| {
                let (item, right) = data_slice.split_first().expect("Corrupt subset");
                if let Some((second, _)) = rest.split_first() {
                    let (_, r) = right.split_at(*second - *first - 1);
                    *data = r;
                } else {
                    let n = data.len();
                    let (_, r) = right.split_at(n);
                    *data = r;
                }
                *indices = rest;
                item
            }),
            None => data_slice.split_first().map(|(item, rest)| {
                *data = rest;
                item
            }),
        }
    }
}

impl<'a, S, I> Subset<S, I>
where
    S: Set + ViewMut<'a>,
    I: std::borrow::Borrow<[usize]>,
{
    /// Mutably iterate over a borrowed subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// for i in subset.iter_mut() {
    ///     *i += 1;
    /// }
    /// assert_eq!(v, vec![2,2,4,4,6]);
    /// ```
    pub fn iter_mut(&'a mut self) -> SubsetIterMut<'a, <S as ViewMut<'a>>::Type> {
        SubsetIterMut {
            indices: self.indices.as_ref().map(|indices| indices.borrow()),
            data: self.data.view_mut(),
        }
    }
}

impl<S: Dummy, I> Dummy for Subset<S, I> {
    fn dummy() -> Self {
        Subset {
            data: Dummy::dummy(),
            indices: None,
        }
    }
}

*/
