use super::*;

/// A Set that is a non-contiguous subset of some larger set (which could have any type).
/// `B` can be any borrowed collection type.
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
/////
///// The next example shows how to create a `Subset` from a [`UniSet`].
/////
///// ```rust
///// use utils::soap::*;
///// let mut v = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
///// let mut subset = Subset::from_indices(vec![0,2,4], v.view_mut());
///// {
/////     let mut subset_iter = subset.iter();
/////     assert_eq!(Some(&[1,2,3]), subset_iter.next());
/////     assert_eq!(Some(&[7,8,9]), subset_iter.next());
/////     assert_eq!(Some(&[13,14,15]), subset_iter.next());
/////     assert_eq!(None, subset_iter.next());
///// }
///// subset[1] = [0; 3];
///// assert_eq!([0,0,0], subset[1]);
///// ```
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

impl<'a, S: Set> Subset<S> {
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
    pub fn from_indices(mut indices: Vec<usize>, data: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.
        // Failure to enforce this invariant can cause race conditions.

        indices.sort_unstable();
        indices.dedup();

        Self::validate(Subset {
            indices: Some(indices),
            data,
        })
    }

    /// collection of indices to be in sorted order and have no duplicates.
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
    /// let v = vec![1,2,3];
    /// let subset = Subset::from_indices(vec![0,2], v.as_slice());
    /// assert_eq!(1, subset[0]);
    /// assert_eq!(3, subset[1]);
    /// ```
    pub fn from_uniqe_ordered_indices(indices: Vec<usize>, data: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.

        assert!(Self::is_sorted(&indices));
        assert!(!Self::has_duplicates(&indices));

        Self::validate(Subset {
            indices: Some(indices),
            data,
        })
    }

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
            for &i in indices.iter() {
                assert!(i < self.data.len(), "Subset index out of bounds.");
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
impl<S: Set, I: Set> Set for Subset<S, I> {
    type Elem = S::Elem;
    fn len(&self) -> usize {
        self.indices
            .as_ref()
            .map_or(self.data.len(), |indices| indices.len())
    }
}

/// Required for `Chunked` and `UniChunked` subsets.
impl<'a, S> View<'a> for Subset<S>
where
    S: Set + View<'a>,
    <S as View<'a>>::Type: Set + SplitAt,
{
    type Type = Subset<S::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        // Converting to index slices requires us to chop the beginning of the
        // data set before the first index.
        match self.indices {
            Some(ref indices) => {
                if let Some(first) = indices.first() {
                    let (_, data_view) = self.data.view().split_at(*first);
                    Subset {
                        indices: Some(indices.as_slice()),
                        data: data_view,
                    }
                } else {
                    Subset {
                        indices: Some(indices.as_slice()),
                        data: self.data.view(),
                    }
                }
            }
            None => Subset {
                indices: None,
                data: self.data.view(),
            },
        }
    }
}

impl<'a, S> View<'a> for Subset<S, &'a [usize]>
where
    S: Set + View<'a>,
    <S as View<'a>>::Type: Set,
{
    type Type = Subset<S::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        // For subset with &'a [usize] indices, it is assumed that the first
        // index corresponds to the first element in data, regardless of what
        // the value of the index is.
        Subset {
            indices: self.indices,
            data: self.data.view(),
        }
    }
}

macro_rules! impl_view_mut {
    ($self:ident, $split_at_fn:ident) => {{
        // Converting to index slices requires us to chop the beginning of the
        // data set before the first index.
        match $self.indices {
            Some(ref indices) => {
                if let Some(first) = indices.first() {
                    let (_, data_view) = $self.data.view_mut().$split_at_fn(*first);
                    Subset {
                        indices: Some(indices.as_slice()),
                        data: data_view,
                    }
                } else {
                    Subset {
                        indices: Some(indices.as_slice()),
                        data: $self.data.view_mut(),
                    }
                }
            }
            None => Subset {
                indices: None,
                data: $self.data.view_mut(),
            },
        }
    }};
}

/// Required for mutable `Chunked` and `UniChunked` subsets.
impl<'a, S> ViewMut<'a> for Subset<S>
where
    S: Set + ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + SplitAt,
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
        impl_view_mut!(self, split_at)
    }
}

impl<'a, S> ViewMut<'a> for Subset<S, &'a [usize]>
where
    S: Set + ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = Subset<S::Type, &'a [usize]>;
    fn view_mut(&'a mut self) -> Self::Type {
        // For subset with &'a [usize] indices, it is assumed that the first
        // index corresponds to the first element in data, regardless of what
        // the value of the index is.
        Subset {
            indices: self.indices,
            data: self.data.view_mut(),
        }
    }
}

macro_rules! impl_split_at_fn {
    ($self:ident, $split_fn:ident, $mid:expr) => {{
        if let Some(ref indices) = $self.indices {
            let (indices_l, indices_r) = indices.split_at($mid);
            let n = $self.data.len();
            let offset = indices_r
                .first()
                .map(|first| *first - *indices_l.first().unwrap_or(first))
                .unwrap_or(n);
            let (data_l, data_r) = $self.data.$split_fn(offset);
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
            let (data_l, data_r) = $self.data.$split_fn($mid);
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
    }};
}

/// This impl enables `Chunked` `Subset`s
impl<V> SplitAt for Subset<V, &[usize]>
where
    V: Set + SplitAt + std::fmt::Debug,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at, mid)
    }
}

//impl<T> SplitAt for Subset<&[T], &[usize]> {
//    fn split_at(self, mid: usize) -> (Self, Self) {
//        impl_split_at_fn!(self, split_at, mid)
//    }
//}

//impl<T> SplitAt for Subset<&mut [T], &[usize]> {
//    fn split_at(self, mid: usize) -> (Self, Self) {
//        impl_split_at_fn!(self, split_at_mut, mid)
//    }
//}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

impl<'a, S: std::ops::Index<usize> + ?Sized> std::ops::Index<usize> for Subset<&'a S> {
    type Output = S::Output;
    /// Immutably index the subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], &v);
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.data
            .index(self.indices.as_ref().map_or(idx, |indices| indices[idx]))
    }
}

impl<'a, S: std::ops::Index<usize> + ?Sized> std::ops::Index<usize> for Subset<&'a mut S> {
    type Output = S::Output;
    /// Immutably index the subset.
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
            .index(self.indices.as_ref().map_or(idx, |indices| indices[idx]))
    }
}

impl<'a, S: std::ops::IndexMut<usize> + ?Sized> std::ops::IndexMut<usize> for Subset<&'a mut S> {
    /// Mutably index the subset.
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
            .index_mut(self.indices.as_ref().map_or(idx, |indices| indices[idx]))
    }
}

//impl<S, N> std::ops::Index<usize> for Subset<UniSet<S, N>> {
//    type Output = <UniSet<S, N> as std::ops::Index<usize>>::Output;
//    /// Immutably index the subset.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let mut v = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
//    /// let subset = Subset::from_indices(vec![0,2,4], v.view());
//    /// assert_eq!([7,8,9], subset[1]);
//    /// ```
//    fn index(&self, idx: usize) -> &Self::Output {
//        self.data.index(self.indices[idx])
//    }
//}

/*
 * Iteration
 */

impl<S: Set> Subset<S> {
    pub fn iter<'o, 'i: 'o>(&'i self) -> impl Iterator<Item = <S as Get<'i, 'o, usize>>::Output>
    where
        S: Get<'i, 'o, usize> + View<'i>,
        <S as View<'i>>::Type: IntoIterator<Item = S::Output>,
    {
        let iters = match self.indices {
            Some(ref indices) => (None, Some(indices.iter().map(move |&i| self.data.get(i)))),
            None => (Some(self.data.view().into_iter()), None),
        };
        iters
            .0
            .into_iter()
            .flatten()
            .chain(iters.1.into_iter().flatten())
    }
}

impl<S: Set> Subset<S, &[usize]> {
    pub fn iter<'o, 'i: 'o>(&'i self) -> impl Iterator<Item = <S as Get<'i, 'o, usize>>::Output>
    where
        S: Get<'i, 'o, usize> + View<'i> + std::fmt::Debug,
        <S as View<'i>>::Type: IntoIterator<Item = S::Output>,
    {
        let iters = match self.indices {
            Some(indices) => {
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

impl<'a, T: 'a + std::fmt::Debug> Iterator for SubsetIter<'a, &'a mut [T]> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let SubsetIter { indices, data } = self;
        let data_slice = std::mem::replace(data, &mut []);
        match indices {
            Some(ref mut indices) => indices.split_first().map(move |(first, rest)| {
                let (item, right) = data_slice.split_first_mut().expect("Corrupt subset");
                if let Some((second, _)) = rest.split_first() {
                    dbg!(&right);
                    dbg!(first);
                    dbg!(second);
                    dbg!(*second - *first - 1);
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
