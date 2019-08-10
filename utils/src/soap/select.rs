use super::*;

/// A Set that is a non-contiguous, unordered and possibly duplicated selection
/// of some larger collection. `S` can be any borrowed collection type that
/// implements [`Set`]. Note that it doesn't make much sense to have a `Select`
/// type own the data that it selects from, although it's possible to create
/// one.
///
/// # Simple Usage Examples
///
/// The following example shows how to `Select` from a range.
///
/// ```rust
/// use utils::soap::*;
/// let selection = Select::new(vec![0,2,4,0,1], 5..10);
/// let mut iter = selection.iter();
/// assert_eq!(Some(5), iter.next());
/// assert_eq!(Some(7), iter.next());
/// assert_eq!(Some(9), iter.next());
/// assert_eq!(Some(5), iter.next());
/// assert_eq!(Some(6), iter.next());
/// assert_eq!(None, iter.next());
/// ```
///
/// The next example shows how to `Select` from a [`UniChunked`] view.
///
/// ```rust
/// use utils::soap::*;
/// let mut v = Chunked3::from_flat((1..=15).collect::<Vec<_>>());
/// let mut selection = Select::new(vec![1,0,4,4,1], v.view_mut());
/// *selection.at_mut(0) = [0; 3];
/// {
///     let mut iter = selection.iter();
///     assert_eq!(Some(&[0,0,0]), iter.next());
///     assert_eq!(Some(&[1,2,3]), iter.next());
///     assert_eq!(Some(&[13,14,15]), iter.next());
///     assert_eq!(Some(&[13,14,15]), iter.next());
///     assert_eq!(Some(&[0,0,0]), iter.next());
///     assert_eq!(None, iter.next());
/// }
/// ```
///
/// # Mutable `Select`ions
///
/// A `Select`ion of a mutable borrow cannot be [`SplitAt`], which means it
/// cannot be [`Chunked`]. This is because a split selection must have a copy of
/// the mutable borrow since an index from any half of the split can access any
/// part of the data. This of course breaks Rust's aliasing rules. It is
/// possible, however to bypass this restriction by using interior mutability.
///
///
/// # Common Uses
///
/// Selections are a useful way to annotate arrays of indices into some other
/// array or even a range. It is not uncommon to use a `Vec<usize>` to represent
/// indices into another collection. Using `Select` instead lets the user be
/// explicit about where these indices are pointing without having to annotate
/// the indices themselves.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Select<S, I = Vec<usize>> {
    pub(crate) indices: I,
    pub(crate) data: S,
}

/// A borrowed selection.
pub type SelectView<'a, S> = Select<S, &'a [usize]>;

impl<S, I> Select<S, I> {
    /// Create a selection of elements from the original set from the given
    /// indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3];
    /// let selection = Select::new(vec![1,2,1], v.as_slice());
    /// assert_eq!(2, selection[0]);
    /// assert_eq!(3, selection[1]);
    /// assert_eq!(2, selection[2]);
    /// ```
    pub fn new(indices: I, data: S) -> Self {
        Select { indices, data }
    }
}

impl<'a, S, I> Select<S, I> {
    /// Get a references to the underlying indices.
    pub fn indices(&self) -> &I {
        &self.indices
    }

    /// Get a reference to the underlying data.
    pub fn data(&self) -> &S {
        &self.data
    }

    /// Get a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }
}

// Note to self:
// To enable a collection to be chunked, we need to implement:
// Set, View, SplitAt
// For mutability we also need ViewMut,
// For UniChunked we need:
// Set, Vew, ReinterpretSet (this needs to be refined)

// Required for `Chunked` and `UniChunked` selections.
impl<S: Set, I: std::borrow::Borrow<[usize]>> Set for Select<S, I> {
    type Elem = S::Elem;
    /// Get the number of selected elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let selection = Select::new(vec![4,0,1,4], v.as_slice());
    /// assert_eq!(4, selection.len());
    /// ```
    fn len(&self) -> usize {
        self.indices.borrow().len()
    }
}

// Required for `Chunked` and `UniChunked` selections.
impl<'a, S, I> View<'a> for Select<S, I>
where
    S: Set + View<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: Set,
{
    type Type = Select<S::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        Select {
            indices: self.indices.borrow(),
            data: self.data.view(),
        }
    }
}

impl<'a, S, I> ViewMut<'a> for Select<S, I>
where
    S: Set + ViewMut<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = Select<S::Type, &'a [usize]>;
    /// Create a mutable view into this selection.
    ///
    // TODO: implement iter_mut
    ///// # Example
    /////
    ///// ```rust
    ///// use utils::soap::*;
    ///// let mut v = vec![1,2,3,4,5];
    ///// let mut selection = Select::new(vec![1,2,4,1], v.as_mut_slice());
    ///// let mut view = selection.view_mut();
    ///// for i in view.iter_mut() {
    /////     *i += 1;
    ///// }
    ///// assert_eq!(v, vec![1,4,4,4,6]);
    ///// ```
    fn view_mut(&'a mut self) -> Self::Type {
        Select {
            indices: self.indices.borrow(),
            data: self.data.view_mut(),
        }
    }
}

// This impl enables `Chunked` `Select`ions
impl<V> SplitAt for Select<V, &[usize]>
where
    V: Set + Clone,
{
    /// Split this selection into two at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![3,2,0,4,2];
    /// let selection = Select::new(indices.as_slice(), v.as_slice());
    /// let (l, r) = selection.split_at(2);
    /// let mut iter_l = l.iter();
    /// assert_eq!(Some(&4), iter_l.next());
    /// assert_eq!(Some(&3), iter_l.next());
    /// assert_eq!(None, iter_l.next());
    /// let mut iter_r = r.iter();
    /// assert_eq!(Some(&1), iter_r.next());
    /// assert_eq!(Some(&5), iter_r.next());
    /// assert_eq!(Some(&3), iter_r.next()); // Note that 3 is shared between l and r
    /// assert_eq!(None, iter_r.next());
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Select { data, indices } = self;
        let (indices_l, indices_r) = indices.split_at(mid);
        (
            Select {
                indices: indices_l,
                data: data.clone(),
            },
            Select {
                indices: indices_r,
                data: data,
            },
        )
    }
}

impl<'a, S, I> Select<S, I>
where
    S: Set + Get<'a, usize, Output = &'a <S as Set>::Elem> + View<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: IntoIterator<Item = S::Output>,
    <S as Set>::Elem: 'a,
{
    /// The typical way to use this function is to clone from a `SelectView`
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
    /// let indices = vec![3,3,4,0];
    /// let selection = Select::new(indices.as_slice(), v.as_slice());
    /// let mut owned = vec![0; 5];
    /// selection.clone_into_other(&mut owned[..4]); // Need 4 elements to avoid panics.
    /// let mut iter_owned = owned.iter();
    /// assert_eq!(owned, vec![4,4,5,1,0]);
    /// ```
    pub fn clone_into_other<V>(&'a self, other: &'a mut V)
    where
        V: ViewMut<'a> + ?Sized,
        <V as ViewMut<'a>>::Type: Set + IntoIterator<Item = &'a mut <S as Set>::Elem>,
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

impl<'a, S, I> GetIndex<'a, Select<S, I>> for usize
where
    I: std::borrow::Borrow<[usize]>,
    S: Get<'a, usize>,
{
    type Output = <S as Get<'a, usize>>::Output;

    fn get(self, selection: &Select<S, I>) -> Option<Self::Output> {
        selection
            .indices
            .borrow()
            .get(self)
            .and_then(|&cur| Get::get(&selection.data, cur))
    }
}

impl<'a, S, I> GetMutIndex<'a, Select<S, I>> for usize
where
    I: std::borrow::Borrow<[usize]>,
    S: GetMut<'a, usize>,
{
    type Output = <S as GetMut<'a, usize>>::Output;

    fn get_mut(self, selection: &mut Select<S, I>) -> Option<Self::Output> {
        let Select { indices, data } = selection;
        indices
            .borrow()
            .get(self)
            .and_then(move |&cur| GetMut::get_mut(data, cur))
    }
}

impl<'a, S, I, Idx> Get<'a, Idx> for Select<S, I>
where
    Idx: GetIndex<'a, Self>,
{
    type Output = Idx::Output;

    /// Get a reference to an element in this selection.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let selection = Select::new(vec![0,0,4], v.as_slice());
    /// assert_eq!(&1, selection.get(1).unwrap());
    /// ```
    fn get(&self, range: Idx) -> Option<Self::Output> {
        range.get(self)
    }
}

impl<'a, S, I, Idx> GetMut<'a, Idx> for Select<S, I>
where
    Idx: GetMutIndex<'a, Self>,
{
    type Output = Idx::Output;

    /// Get a mutable reference to an element in this selection.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let selection = Select::new(vec![0,0,4], v.as_slice());
    /// assert_eq!(&1, selection.get(1).unwrap());
    /// ```
    fn get_mut(&mut self, range: Idx) -> Option<Self::Output> {
        range.get_mut(self)
    }
}

impl<'a, S, I> std::ops::Index<usize> for Select<S, I>
where
    S: std::ops::Index<usize> + Set + Owned,
    I: std::borrow::Borrow<[usize]>,
{
    type Output = S::Output;

    /// Immutably index the selection.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let selection = Select::new(vec![0,2,0,4], Chunked2::from_flat(1..=12));
    /// assert_eq!(1..3, selection.at(0));
    /// assert_eq!(5..7, selection.at(1));
    /// assert_eq!(1..3, selection.at(2));
    /// assert_eq!(9..11, selection.at(3));
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.data.index(self.indices.borrow()[idx])
    }
}

impl<'a, S, I> std::ops::IndexMut<usize> for Select<S, I>
where
    S: std::ops::IndexMut<usize> + Set + Owned,
    I: std::borrow::Borrow<[usize]>,
{
    /// Mutably index the selection.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut selection = Select::new(vec![0,2,0,4], v.as_mut_slice());
    /// assert_eq!(selection[0], 1);
    /// assert_eq!(selection[1], 3);
    /// assert_eq!(selection[2], 1);
    /// assert_eq!(selection[3], 5);
    /// selection[2] = 100;
    /// assert_eq!(selection[0], 100);
    /// assert_eq!(selection[1], 3);
    /// assert_eq!(selection[2], 100);
    /// assert_eq!(selection[3], 5);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.data.index_mut(self.indices.borrow()[idx])
    }
}

impl<'a, T, I> std::ops::Index<usize> for Select<&'a [T], I>
where
    I: std::borrow::Borrow<[usize]>,
{
    type Output = T;
    /// Immutably index the selection of elements from a borrowed slice.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let selection = Select::new(vec![0,2,0,4], v.as_slice());
    /// assert_eq!(3, selection[1]);
    /// assert_eq!(1, selection[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.data.index(self.indices.borrow()[idx])
    }
}

impl<'a, T, I> std::ops::Index<usize> for Select<&'a mut [T], I>
where
    I: std::borrow::Borrow<[usize]>,
{
    type Output = T;
    /// Immutably index a selection of elements from a mutably borrowed slice.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![3,2,0,4], v.as_mut_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.data.index(self.indices.borrow()[idx])
    }
}

impl<'a, T, I> std::ops::IndexMut<usize> for Select<&'a mut [T], I>
where
    I: std::borrow::Borrow<[usize]>,
{
    /// Mutably index a selection of elements from a mutably borrowed slice.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut selection = Select::new(vec![4,0,2,4], v.as_mut_slice());
    /// assert_eq!(selection[0], 5);
    /// selection[0] = 100;
    /// assert_eq!(selection[0], 100);
    /// assert_eq!(selection[1], 1);
    /// assert_eq!(selection[2], 3);
    /// assert_eq!(selection[3], 100);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.data.index_mut(self.indices.borrow()[idx])
    }
}

/*
 * Iteration
 */

impl<'a, S, I> Select<S, I>
where
    S: Set + Get<'a, usize> + View<'a>,
    I: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: IntoIterator<Item = S::Output>,
{
    pub fn iter(&'a self) -> impl Iterator<Item = <S as Get<'a, usize>>::Output> {
        self.indices
            .borrow()
            .iter()
            .filter_map(move |&i| self.data.get(i))
    }
}

// TODO: need a GetMut::get_mut_ptr implementation to get a raw pointer here.
/*
pub struct SelectIterMut<'a, V> {
    indices: &'a [usize],
    data: V,
}

impl<'a, V: 'a> Iterator for SelectIterMut<'a, V>
where
    V: Set + GetMut<'a, usize>,
{
    type Item = V::Output;

    fn next(&mut self) -> Option<Self::Item> {
        let SelectIterMut { indices, data } = self;
        let data_slice = std::mem::replace(data, Dummy::dummy());
        let data[
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
