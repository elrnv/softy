use super::*;

/// A Set that is a non-contiguous subset of some larger collection.
/// `B` can be any borrowed collection type that implements the [`Set`] and [`RemovePrefix`] traits.
///
/// # Example
///
/// The following example shows how to create a `Subset` from a standard `Vec`.
///
/// ```rust
/// use utils::soap::*;
/// let v = vec![1,2,3,4,5];
/// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
/// let mut subset_iter = subset.iter();
/// assert_eq!(Some(&1), subset_iter.next());
/// assert_eq!(Some(&3), subset_iter.next());
/// assert_eq!(Some(&5), subset_iter.next());
/// assert_eq!(None, subset_iter.next());
/// ```
///
/// The next example shows how to create a `Subset` from a [`UniChunked`] collection.
///
/// ```rust
/// use utils::soap::*;
/// let mut v = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
/// let mut subset = Subset::from_indices(vec![0,2,4], v.view_mut());
/// {
///     let mut subset_iter = subset.iter();
///     assert_eq!(Some(&[1,2,3]), subset_iter.next());
///     assert_eq!(Some(&[7,8,9]), subset_iter.next());
///     assert_eq!(Some(&[13,14,15]), subset_iter.next());
///     assert_eq!(None, subset_iter.next());
/// }
/// *subset.get_mut(1) = [0; 3];
/// assert_eq!(&[0,0,0], subset.get(1));
/// ```
// A note about translation independence:
// ======================================
// This struct is very similar to `Chunked`, with the main difference being that
// each index corresponds to a single element instead of a chunk starting point.
// To be able to split subsets, we need to make indices translation independent
// so that we don't have to modify their values when we split the collection.
// When the indices are owned, we simply modify the indices when we split the
// subset, but when the indices are a borrowed slice, we always chop the part of
// data below the first index to ensure that the first index serves as an offset
// to the rest of the indices, making the entire index array translation
// independent.
#[derive(Copy, Clone, Debug)]
pub struct Subset<S, I = Vec<usize>> {
    /// An optional set of indices. When this is `None`, the subset is
    /// considered to be entire. Empty subsets are represented by a zero length
    /// array of indices: either `Some(&[])` or `Some(Vec::new())`.
    pub(crate) indices: Option<I>,
    pub(crate) data: S,
}

/// A borrowed subset.
pub type SubsetView<'a, S> = Subset<S, &'a [usize]>;

impl<S: Set + RemovePrefix> Subset<S> {
    /// Create a subset of elements from the original set given at the specified indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3];
    /// let subset = Subset::from_indices(vec![0,2], v.as_slice());
    /// assert_eq!(1, subset[0]);
    /// assert_eq!(3, subset[1]);
    /// ```
    pub fn from_indices(mut indices: Vec<usize>, mut data: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.
        // Failure to enforce this invariant can cause race conditions.

        indices.sort_unstable();
        indices.dedup();

        if let Some(first) = indices.first() {
            data.remove_prefix(*first);
        }

        Self::validate(Subset {
            indices: Some(indices),
            data,
        })
    }
}

impl<S: Set + RemovePrefix, I: std::borrow::Borrow<[usize]>> Subset<S, I> {
    /// Create a subset of elements from the original scollection corresponding to the given indices.
    /// In contrast to `Subset::from_indices`, this function expects the indices
    /// to be unique and in sorted order, instead of manully making it so.
    ///
    /// # Panics
    ///
    /// This function panics when given a collection of unsorted indices.
    /// It also panics when indices are repeated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![0,1,2,3];
    /// let indices = vec![1,3];
    ///
    /// let subset_view = Subset::from_unique_ordered_indices(indices.as_slice(), v.as_slice());
    /// assert_eq!(1, subset_view[0]);
    /// assert_eq!(3, subset_view[1]);
    ///
    /// let subset = Subset::from_unique_ordered_indices(indices, v.as_slice());
    /// assert_eq!(1, subset[0]);
    /// assert_eq!(3, subset[1]);
    /// ```
    pub fn from_unique_ordered_indices(indices: I, mut data: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.

        assert!(Self::is_sorted(&indices));
        assert!(!Self::has_duplicates(&indices));

        if let Some(first) = indices.borrow().first() {
            data.remove_prefix(*first);
        }

        Self::validate(Subset {
            indices: Some(indices),
            data,
        })
    }
}

impl<S: Set + RemovePrefix> Subset<S, &[usize]> {
    /// Create a subset with all elements from the original set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let subset = Subset::all(vec![1,2,3]);
    /// let mut subset_iter = subset.iter();
    /// assert_eq!(Some(&1), subset_iter.next());
    /// assert_eq!(Some(&2), subset_iter.next());
    /// assert_eq!(Some(&3), subset_iter.next());
    /// assert_eq!(None, subset_iter.next());
    /// ```
    pub fn all(data: S) -> Self {
        Subset {
            indices: None,
            data,
        }
    }
}

impl<'a, S, I: std::borrow::Borrow<[usize]>> Subset<S, I> {
    /// A helper function that checks if a given collection of indices has duplicates.
    /// It is assumed that the given indices are already in sorted order.
    fn has_duplicates(indices: &I) -> bool {
        let mut index_iter = indices.borrow().iter().cloned();
        if let Some(mut prev) = index_iter.next() {
            for cur in index_iter {
                if cur == prev {
                    return true;
                } else {
                    prev = cur;
                }
            }
        }
        false
    }

    /// Checks that the given set of indices are sorted.
    // TODO: replace this with std version when RFC 2351 lands
    // (https://github.com/rust-lang/rust/issues/53485)
    fn is_sorted(indices: &I) -> bool {
        Self::is_sorted_by(indices, |a, b| a.partial_cmp(b))
    }

    /// Checks that the given set of indices are sorted by the given compare function.
    fn is_sorted_by<F>(indices: &I, mut compare: F) -> bool
    where
        F: FnMut(&usize, &usize) -> Option<std::cmp::Ordering>,
    {
        let mut iter = indices.borrow().iter();
        let mut last = match iter.next() {
            Some(e) => e,
            None => return true,
        };

        while let Some(curr) = iter.next() {
            if compare(&last, &curr)
                .map(|o| o == std::cmp::Ordering::Greater)
                .unwrap_or(true)
            {
                return false;
            }
            last = curr;
        }

        true
    }
}

impl<'a, S: Set, I> Subset<S, I> {
    /// Get a references to the underlying indices. If `None` is returned, then
    /// this subset spans the entire domain `data`.
    pub fn indices(&self) -> Option<&I> {
        self.indices.as_ref()
    }

    /// Return the superset of this `Subset`. This is just the set it was created with.
    pub fn into_super(self) -> S {
        self.data
    }
}

impl<'a, S: Set, I: std::borrow::Borrow<[usize]>> Subset<S, I> {
    /// Panics if this subset is invald.
    #[inline]
    fn validate(self) -> Self {
        if let Some(ref indices) = self.indices {
            let indices = indices.borrow();
            if let Some(first) = indices.first() {
                for &i in indices.iter() {
                    assert!(i - *first < self.data.len(), "Subset index out of bounds.");
                }
            }
        }
        self
    }
}

// Note to self:
// To enable a collection to be chunked, we need to implement:
// Set, View, SplitAt
// For mutability we also need ViewMut,
// For UniChunked we need:
// Set, Vew, ReinterpretSet (this needs to be refined)

/// Required for `Chunked` and `UniChunked` subsets.
impl<S: Set, I: std::borrow::Borrow<[usize]>> Set for Subset<S, I> {
    type Elem = S::Elem;
    /// Get the length of this subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
    /// assert_eq!(3, subset.len());
    /// ```
    fn len(&self) -> usize {
        self.indices
            .as_ref()
            .map_or(self.data.len(), |indices| indices.borrow().len())
    }
}

/// Required for `Chunked` and `UniChunked` subsets.
impl<'a, S, I> View<'a> for Subset<S, I>
where
    S: Set + View<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: Set,
{
    type Type = Subset<S::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        // Note: it is assumed that the first index corresponds to the first
        // element in data, regardless of what the value of the index is.
        Subset {
            indices: self.indices.as_ref().map(|indices| indices.borrow()),
            data: self.data.view(),
        }
    }
}

impl<'a, S, I> ViewMut<'a> for Subset<S, I>
where
    S: Set + ViewMut<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = Subset<S::Type, &'a [usize]>;
    /// Create a mutable view into this subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// let mut view = subset.view_mut();
    /// for i in view.iter_mut() {
    ///     *i += 1;
    /// }
    /// assert_eq!(v, vec![2,2,4,4,6]);
    /// ```
    fn view_mut(&'a mut self) -> Self::Type {
        // Note: it is assumed that the first index corresponds to the first
        // element in data, regardless of what the value of the index is.
        Subset {
            indices: self.indices.as_ref().map(|indices| indices.borrow()),
            data: self.data.view_mut(),
        }
    }
}

/// This impl enables `Chunked` `Subset`s
impl<V> SplitAt for Subset<V, &[usize]>
where
    V: Set + SplitAt,
{
    /// Split this subset into two at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![0,2,4];
    /// let subset = Subset::from_unique_ordered_indices(indices.as_slice(), v.as_slice());
    /// let (l, r) = subset.split_at(1);
    /// let mut iter_l = l.iter();
    /// assert_eq!(Some(&1), iter_l.next());
    /// assert_eq!(None, iter_l.next());
    /// let mut iter_r = r.iter();
    /// assert_eq!(Some(&3), iter_r.next());
    /// assert_eq!(Some(&5), iter_r.next());
    /// assert_eq!(None, iter_r.next());
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        if let Some(ref indices) = self.indices {
            let (indices_l, indices_r) = indices.split_at(mid);
            let n = self.data.len();
            let offset = indices_r
                .first()
                .map(|first| *first - *indices_l.first().unwrap_or(first))
                .unwrap_or(n);
            let (data_l, data_r) = self.data.split_at(offset);
            (
                Subset {
                    indices: Some(indices_l),
                    data: data_l,
                },
                Subset {
                    indices: Some(indices_r),
                    data: data_r,
                },
            )
        } else {
            let (data_l, data_r) = self.data.split_at(mid);
            (
                Subset {
                    indices: None,
                    data: data_l,
                },
                Subset {
                    indices: None,
                    data: data_r,
                },
            )
        }
    }
}

impl<'o, 'i: 'o, S, I> Subset<S, I>
where
    S: Set + Get<'i, 'o, usize, Output = &'o <S as Set>::Elem> + View<'i>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'i>>::Type: IntoIterator<Item = S::Output>,
    <S as Set>::Elem: 'o,
{
    /// The typical way to use this function is to clone from a `SubsetView`
    /// into an owned `S` type.
    ///
    /// # Panics
    ///
    /// This function panics if `other` has a length unequal to `self.len()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![0,2,4];
    /// let subset = Subset::from_unique_ordered_indices(indices.as_slice(), v.as_slice());
    /// let mut owned = vec![0; 4];
    /// subset.clone_into_other(&mut owned[..3]); // Need 3 elements to avoid panics.
    /// let mut iter_owned = owned.iter();
    /// assert_eq!(owned, vec![1,3,5,0]);
    /// ```
    pub fn clone_into_other<'a, V>(&'i self, other: &'a mut V)
    where
        V: ViewMut<'a> + ?Sized,
        <V as ViewMut<'a>>::Type: Set + IntoIterator<Item = &'o mut <S as Set>::Elem>,
        <S as Set>::Elem: Clone,
    {
        let other_view = other.view_mut();
        assert_eq!(other_view.len(), self.len());
        for (theirs, mine) in other_view.into_iter().zip(self.iter()) {
            theirs.clone_from(&mine);
        }
    }
}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Subset<S, O>> for usize
where
    O: Get<'i, 'o, usize, Output = &'o usize>,
    usize: GetIndex<'i, 'o, S>,
{
    type Output = <usize as GetIndex<'i, 'o, S>>::Output;

    fn get(self, subset: &'i Subset<S, O>) -> Option<Self::Output> {
        // TODO: too much bounds checking here, add a get_unchecked call to GetIndex.
        if let Some(ref indices) = subset.indices {
            GetIndex::get(*indices.get(self), &subset.data)
        } else {
            GetIndex::get(self, &subset.data)
        }
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Subset<S, O>> for usize
where
    O: Get<'i, 'o, usize, Output = &'o usize>,
    usize: GetMutIndex<'i, 'o, S>,
{
    type Output = <usize as GetMutIndex<'i, 'o, S>>::Output;

    fn get_mut(self, subset: &'i mut Subset<S, O>) -> Option<Self::Output> {
        // TODO: too much bounds checking here, add a get_unchecked call to GetIndex.
        if let Some(ref indices) = subset.indices {
            GetMutIndex::get_mut(*indices.get(self), &mut subset.data)
        } else {
            GetMutIndex::get_mut(self, &mut subset.data)
        }
    }
}

impl<'o, 'i: 'o, S, I, O> Get<'i, 'o, I> for Subset<S, O>
where
    I: GetIndex<'i, 'o, Self>,
{
    type Output = I::Output;

    fn get(&'i self, range: I) -> Self::Output {
        range.get(self).expect("Index out of bounds")
    }
}

impl<'o, 'i: 'o, S, I, O> GetMut<'i, 'o, I> for Subset<S, O>
where
    I: GetMutIndex<'i, 'o, Self>,
{
    type Output = I::Output;

    fn get_mut(&'i mut self, range: I) -> Self::Output {
        range.get_mut(self).expect("Index out of bounds")
    }
}

impl<'a, S, I> std::ops::Index<usize> for Subset<&'a S, I>
where
    S: std::ops::Index<usize> + ?Sized,
    I: std::borrow::Borrow<[usize]>,
{
    type Output = S::Output;
    /// Immutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.data
            .index(self.indices.as_ref().map_or(idx, |indices| {
                let indices = indices.borrow();
                indices[idx] - *indices.first().unwrap()
            }))
    }
}

impl<'a, S, I> std::ops::Index<usize> for Subset<&'a mut S, I>
where
    S: std::ops::Index<usize> + ?Sized,
    I: std::borrow::Borrow<[usize]>,
{
    type Output = S::Output;
    /// Immutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.data
            .index(self.indices.as_ref().map_or(idx, |indices| {
                let indices = indices.borrow();
                indices[idx] - *indices.first().unwrap()
            }))
    }
}

impl<'a, S, I> std::ops::IndexMut<usize> for Subset<&'a mut S, I>
where
    S: std::ops::IndexMut<usize> + ?Sized,
    I: std::borrow::Borrow<[usize]>,
{
    /// Mutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(subset[1], 3);
    /// subset[1] = 100;
    /// assert_eq!(subset[0], 1);
    /// assert_eq!(subset[1], 100);
    /// assert_eq!(subset[2], 5);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.data
            .index_mut(self.indices.as_ref().map_or(idx, |indices| {
                let indices = indices.borrow();
                indices[idx] - *indices.first().unwrap()
            }))
    }
}

/*
 * Iteration
 */

impl<'o, 'i: 'o, S, I> Subset<S, I>
where
    S: Set + Get<'i, 'o, usize> + View<'i>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'i>>::Type: IntoIterator<Item = S::Output>,
{
    pub fn iter(&'i self) -> impl Iterator<Item = <S as Get<'i, 'o, usize>>::Output> {
        let iters = match self.indices.as_ref() {
            Some(indices) => {
                let indices = indices.borrow();
                let first = *indices.first().unwrap_or(&0);
                (
                    None,
                    Some(indices.iter().map(move |&i| self.data.get(i - first))),
                )
            }
            None => (Some(self.data.view().into_iter()), None),
        };
        iters
            .0
            .into_iter()
            .flatten()
            .chain(iters.1.into_iter().flatten())
    }
}

pub struct SubsetIter<'a, V> {
    indices: Option<&'a [usize]>,
    data: V,
}

impl<'a, V: 'a> Iterator for SubsetIter<'a, V>
where
    V: SplitAt + SplitFirst + Set + Dummy,
{
    type Item = V::First;

    fn next(&mut self) -> Option<Self::Item> {
        let SubsetIter { indices, data } = self;
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

impl<'a, T: 'a> Iterator for SubsetIter<'a, &'a mut [T]> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let SubsetIter { indices, data } = self;
        let data_slice = std::mem::replace(data, &mut []);
        match indices {
            Some(ref mut indices) => indices.split_first().map(move |(first, rest)| {
                let (item, right) = data_slice.split_first_mut().expect("Corrupt subset");
                if let Some((second, _)) = rest.split_first() {
                    let (_, r) = right.split_at_mut(*second - *first - 1);
                    *data = r;
                } else {
                    let n = data.len();
                    let (_, r) = right.split_at_mut(n);
                    *data = r;
                }
                *indices = rest;
                item
            }),
            None => data_slice.split_first_mut().map(|(item, rest)| {
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
    pub fn iter_mut(&'a mut self) -> SubsetIter<'a, <S as ViewMut<'a>>::Type> {
        SubsetIter {
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
