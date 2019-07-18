use super::*;

/*
 * VarSet
 */

/// A set of variable length elements. Each offset represents one element and gives the offset into
/// the data buffer for the first of subelement in the Set.
/// Offsets always ends with the length of the buffer minus the value of the first offset.
#[derive(Copy, Clone, Debug)]
pub struct VarSet<S, O = Vec<usize>> {
    pub(crate) data: S,
    pub(crate) offsets: O,
}

impl<S: Set, O: Buffer<usize>> VarSet<S, O> {
    /// Construct a `VarSet` from a `Vec` of offsets into another set. This is
    /// the most efficient constructor, although it is also the most error
    /// prone.
    ///
    /// # Panics
    ///
    /// The absolute value of `offsets` is not significant, however their
    /// relative quantities are. More specifically, if `x` is the first offset,
    /// then the last element of offsets must always be `data.len() + x`.
    /// This also implies that `offsets` cannot be empty. This function panics
    /// if any one of these invariants isn't satisfied.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut varset_iter = s.iter();
    /// assert_eq!(vec![1,2,3], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    pub fn from_offsets(offsets: O, data: S) -> Self {
        assert!(!offsets.is_empty());
        assert_eq!(*offsets.last().unwrap(), data.len() + *offsets.first().unwrap());
        VarSet { offsets, data }
    }
}

impl<S> VarSet<S>
where
    S: Set + AppendVec<Item = <S as Set>::Elem> + Default,
    <S as Set>::Elem: Sized,
{
    /// Construct a `VarSet` from a nested set of `Vec`s.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::<Vec<_>>::from_nested_vec(vec![vec![1,2,3],vec![4],vec![5,6]]);
    /// let mut varset_iter = s.iter();
    /// assert_eq!(vec![1,2,3], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    pub fn from_nested_vec(nested_data: Vec<Vec<<S as Set>::Elem>>) -> Self {
        nested_data.into_iter().collect()
    }
}

// NOTE: There is currently no way to split ownership of a Vec without
// allocating. For this reason we opt to use a slice and defer allocation to
// a later step when the results may be collected into another Vec. This saves
// an extra allocation. We could make this more righteous with a custom
// allocator.
impl<'a, S> std::iter::FromIterator<&'a mut [<S as Set>::Elem]> for VarSet<S>
where
    S: Set
        + ExtendFromSlice<Item = <S as Set>::Elem>
        + Default
        + std::iter::FromIterator<&'a mut [<S as Set>::Elem]>,
    <S as Set>::Elem: Sized + 'a,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a mut [<S as Set>::Elem]>,
    {
        let mut s = VarSet::default();
        for i in iter {
            s.push_slice(i);
        }
        s
    }
}

// For convenience we also implement a `FromIterator` trait for building from
// nested `Vec`s, however as mentioned in the note above, this is typically
// inefficient because it relies on intermediate allocations. This is acceptable
// during initialization, for instance.
impl<S> std::iter::FromIterator<Vec<<S as Set>::Elem>> for VarSet<S>
where
    S: Set + AppendVec<Item = <S as Set>::Elem> + Default,
    <S as Set>::Elem: Sized,
{
    /// Construct a `VarSet` from an iterator over `Vec` types.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// use std::iter::FromIterator;
    /// let s = VarSet::<Vec<_>>::from_iter(vec![vec![1,2,3],vec![4],vec![5,6]].into_iter());
    /// let mut varset_iter = s.iter();
    /// assert_eq!(vec![1,2,3], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<<S as Set>::Elem>>,
    {
        let mut s = VarSet::default();
        for i in iter {
            s.push_vec(i);
        }
        s
    }
}

impl<S, O> Set for VarSet<S, O>
where S: Set, O: Set
{
    type Elem = Vec<S::Elem>;
    /// Get the number of elements in a `VarSet`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// assert_eq!(3, s.len());
    /// ```
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

//impl<'a, S: std::ops::Index<> + Clone> GetElem<'a> for VarSet<S> {
//    fn get(&'a self, idx: usize) -> Self::Elem {
//        self.data[self.offsets[idx]..self.offsets[idx+1]].collect()
//    }
//}

impl<S> VarSet<S>
where S: Set + AppendVec<Item = <S as Set>::Elem>,
    <S as Set>::Elem: Sized
{
    /// Push a `Vec` of elements onto this `VarSet`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = VarSet::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.push_vec(vec![1,2]);
    /// assert_eq!(3, s.len());
    /// ```
    pub fn push_vec(&mut self, mut element: Vec<<S as Set>::Elem>) {
        self.data.append(&mut element);
        self.offsets.push(self.data.len());
    }
}

//impl<S: Set> IntoFlatVec for VarSet<S> {
//    type SubItem = <S as IntoFlatVec>::SubItem;
//    fn into_flat_vec(self) -> Vec<Self::SubItem> {
//        self.data.into_flat_vec()
//    }
//}

impl<S> VarSet<S>
where
    S: Set + ExtendFromSlice<Item = <S as Set>::Elem>,
    <S as Set>::Elem: Sized
{
    /// Push a slice of elements onto this `VarSet`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = VarSet::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.push_slice(&[1,2]);
    /// assert_eq!(3, s.len());
    /// ```
    pub fn push_slice(&mut self, element: &[<S as Set>::Elem]) {
        self.data.extend_from_slice(element);
        self.offsets.push(self.data.len());
    }
}

impl<S: Set + Default> Default for VarSet<S> {
    /// Construct an empty `VarSet`.
    fn default() -> Self {
        Self::from_offsets(vec![0], S::default())
    }
}

impl<'a, S> VarSet<S>
where
    S: View<'a>,
{
    /// Produce an iterator over elements (borrowed slices) of a `VarSet`.
    ///
    /// # Examples
    ///
    /// The following simple example demonstrates how to iterate over a `VarSet`
    /// of integers stored in a flat `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut varset_iter = s.view().iter();
    /// let mut e0_iter = varset_iter.next().unwrap().iter();
    /// assert_eq!(Some(&1), e0_iter.next());
    /// assert_eq!(Some(&2), e0_iter.next());
    /// assert_eq!(Some(&3), e0_iter.next());
    /// assert_eq!(None, e0_iter.next());
    /// assert_eq!(Some(&[4][..]), varset_iter.next());
    /// assert_eq!(Some(&[5,6][..]), varset_iter.next());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    ///
    /// Nested `VarSet`s can also be used to create more complex data organization:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s0 = VarSet::from_offsets(vec![0,3,4,6,9,11], vec![1,2,3,4,5,6,7,8,9,10,11]);
    /// let s1 = VarSet::from_offsets(vec![0,1,4,5], s0);
    /// let mut iter1 = s1.iter();
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter(&'a self) -> VarIter<'a, <S as View<'a>>::Type> {
        VarIter {
            offsets: &self.offsets,
            data: self.data.view(),
        }
    }
}

impl<'a, S> VarSetView<'a, S>
where
    S: View<'a>,
{
    /// Produce an iterator over elements (borrowed slices) of a `VarSetView`.
    ///
    /// # Examples
    ///
    /// The following simple example demonstrates how to iterate over a `VarSet`
    /// of integers stored in a flat `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut varset_iter = s.view().iter();
    /// let mut e0_iter = varset_iter.next().unwrap().iter();
    /// assert_eq!(Some(&1), e0_iter.next());
    /// assert_eq!(Some(&2), e0_iter.next());
    /// assert_eq!(Some(&3), e0_iter.next());
    /// assert_eq!(None, e0_iter.next());
    /// assert_eq!(Some(&[4][..]), varset_iter.next());
    /// assert_eq!(Some(&[5,6][..]), varset_iter.next());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    ///
    /// Nested `VarSet`s can also be used to create more complex data organization:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s0 = VarSet::from_offsets(vec![0,3,4,6,9,11], vec![1,2,3,4,5,6,7,8,9,10,11]);
    /// let s1 = VarSet::from_offsets(vec![0,1,4,5], s0);
    /// let mut iter1 = s1.view().iter();
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter(self) -> VarIter<'a, <S as View<'a>>::Type> {
        VarIter {
            offsets: self.view.offsets,
            data: self.view.data,
        }
    }
}

impl<'a, S> VarSet<S>
where
    S: ViewMut<'a>,
{
    /// Produce a mutable iterator over elements (borrowed slices) of a
    /// `VarSet`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = VarSet::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// for i in s.iter_mut() {
    ///     for j in i.iter_mut() {
    ///         *j += 1;
    ///     }
    /// }
    /// let mut v = s.view_mut();
    /// let mut varset_iter = v.iter();
    /// assert_eq!(vec![2,3,4], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![6,7], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    ///
    /// Nested `VarSet`s can also be used to create more complex data organization:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s0 = VarSet::from_offsets(vec![0,3,4,6,9,11], vec![0,1,2,3,4,5,6,7,8,9,10]);
    /// let mut s1 = VarSet::from_offsets(vec![0,1,4,5], s0);
    /// for mut v0 in s1.iter_mut() {
    ///     for i in v0.iter_mut() {
    ///         for j in i.iter_mut() {
    ///             *j += 1;
    ///         }
    ///     }
    /// }
    /// let mut iter1 = s1.view().iter();
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter_mut(&'a mut self) -> VarIterMut<'a, <S as ViewMut<'a>>::Type> {
        VarIterMut {
            offsets: &self.offsets,
            data: self.data.view_mut(),
        }
    }
}

impl<'a, S> VarSetViewMut<'a, S>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Default,
{
    /// Produce a mutable iterator over elements (borrowed slices) of a
    /// `VarSetViewMut`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = VarSet::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// for i in s.view_mut().iter_mut() {
    ///     for j in i.iter_mut() {
    ///         *j += 1;
    ///     }
    /// }
    /// let mut varset_iter = s.iter();
    /// assert_eq!(vec![2,3,4], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![6,7], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    ///
    /// Nested `VarSet`s can also be used to create more complex data organization:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s0 = VarSet::from_offsets(vec![0,3,4,6,9,11], vec![0,1,2,3,4,5,6,7,8,9,10]);
    /// let mut s1 = VarSet::from_offsets(vec![0,1,4,5], s0);
    /// for v0 in s1.view_mut().iter_mut() {
    ///     for i in v0.iter_mut() {
    ///         for j in i.iter_mut() {
    ///             *j += 1;
    ///         }
    ///     }
    /// }
    /// let mut iter1 = s1.view().iter();
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter_mut(&'a mut self) -> VarIterMut<'a, <S as ViewMut<'a>>::Type> {
        let VarSetViewMut {
            view: VarSet {
                offsets,
                data,
            },
            ..
        } = self;

        let data = std::mem::replace(data, Default::default());
        VarIterMut {
            offsets: offsets,
            data: data,
        }
    }
}

/*
 * Utility traits intended to expose the necessary behaviour to implement `VarSet`s
 */
//pub trait IntoFlatVec {
//    type SubItem;
//    fn into_flat_vec(self) -> Vec<Self::SubItem>;
//}

pub trait ExtendFromSlice {
    type Item;
    fn extend_from_slice(&mut self, other: &[Self::Item]);
}
pub trait AppendVec {
    type Item;
    fn append(&mut self, other: &mut Vec<Self::Item>);
}

/// A helper trait to convert a set view into a slice.
pub trait IntoSlice<'a> {
    type Item;
    fn into_slice(self) -> &'a [Self::Item];
}

/// A mutable version of the `IntoSlice` trait.
pub trait IntoMutSlice<'a>: IntoSlice<'a> {
    fn into_mut_slice(self) -> &'a mut [Self::Item];
}

/*
 * Implement helper traits for supported `Set` types
 */

//impl<T> IntoFlatVec for Vec<T> {
//    type SubItem = T;
//    fn into_flat_vec(self) -> Vec<Self::SubItem> {
//        self
//    }
//}

impl<T: Clone> ExtendFromSlice for Vec<T> {
    type Item = T;
    fn extend_from_slice(&mut self, other: &[Self::Item]) {
        Vec::extend_from_slice(self, other);
    }
}
impl<T> AppendVec for Vec<T> {
    type Item = T;
    fn append(&mut self, other: &mut Vec<Self::Item>) {
        Vec::append(self, other);
    }
}

impl<'a, T> IntoSlice<'a> for &'a [T] {
    type Item = T;
    fn into_slice(self) -> &'a [Self::Item] {
        self
    }
}
impl<'a, T> IntoSlice<'a> for &'a mut [T] {
    type Item = T;
    fn into_slice(self) -> &'a [Self::Item] {
        self
    }
}
impl<'a, T> IntoMutSlice<'a> for &'a mut [T] {
    fn into_mut_slice(self) -> &'a mut [Self::Item] {
        self
    }
}

impl<'a, S> IntoSlice<'a> for VarSet<S, &'a [usize]>
    where S: IntoSlice<'a>,
{
    type Item = <S as IntoSlice<'a>>::Item;
    fn into_slice(self) -> &'a [Self::Item] {
        self.data.into_slice()
    }
}
impl<'a, S> IntoMutSlice<'a> for VarSet<S, &'a [usize]>
    where S: IntoMutSlice<'a>,
{
    fn into_mut_slice(self) -> &'a mut [Self::Item] {
        self.data.into_mut_slice()
    }
}

impl<'a, T: 'a> SplitAt<'a> for &'a [T] {
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[T]>::split_at(self, mid)
    }
}
impl<'a, T: 'a> SplitAt<'a> for &'a mut [T] {
    fn split_at(self, mid: usize) -> (Self, Self) {
        <[T]>::split_at_mut(self, mid)
    }
}

impl<'a, S> SplitAt<'a> for VarSetView<'a, S>
where S: View<'a>,
      <S as View<'a>>::Type: SplitAt<'a> + Set,
{
    fn split_at(mut self, mid: usize) -> (Self, Self) {
        let (offsets_l, offsets_r) = split_offsets_at(self.view.offsets, mid);
        let (data_l, data_r) = self.view.data.split_at(unsafe { *offsets_r.get_unchecked(0) - *offsets_l.get_unchecked(0) });
        self.view.offsets = offsets_r;
        self.view.data = data_r;
        (VarSetView::from_offsets(offsets_l, data_l), self)
    }
}

impl<'a, S> SplitAt<'a> for VarSetViewMut<'a, S>
where S: ViewMut<'a>,
      <S as ViewMut<'a>>::Type: SplitAt<'a> + Set,
{
    fn split_at(mut self, mid: usize) -> (Self, Self) {
        let (offsets_l, offsets_r) = split_offsets_at(self.view.offsets, mid);
        let (data_l, data_r) = self.view.data.split_at(unsafe { *offsets_r.get_unchecked(0) - *offsets_l.get_unchecked(0) });
        self.view.offsets = offsets_r;
        self.view.data = data_r;
        (VarSetViewMut::from_offsets(offsets_l, data_l), self)
    }
}

/// A special iterator capable of iterating over a `VarSet`.
pub struct VarIter<'a, S> {
    offsets: &'a [usize],
    data: S,
}

/// Splits a slice of offsets at the given index into two slices such that each
/// slice is a valid slice of offsets. This means that the element at index
/// `mid` is shared between the two output slices.
fn split_offsets_at(offsets: &[usize], mid: usize) -> (&[usize], &[usize]) {
    debug_assert!(!offsets.is_empty());
    debug_assert!(mid < offsets.len());
    let (l, _) = offsets.split_at(mid+1);
    let (_, r) = offsets.split_at(mid);
    (l, r)
}

/// Test for the `split_offset_at` helper function.
#[test]
fn split_offset_at_test() {
    let offsets = vec![0,1,2,3,4,5];
    let (l, r) = split_offsets_at(offsets.as_slice(), 3);
    assert_eq!(l, &[0,1,2,3]);
    assert_eq!(r, &[3,4,5]);
}


/// Pops an offset from the given slice of offsets and produces an increment for
/// advancing the data pointer. This is a helper function for implementing
/// iterators over `VarSet` types.
/// This function panics if offsets is empty.
fn pop_offset(offsets: &mut &[usize]) -> Option<usize> {
    debug_assert!(!offsets.is_empty(), "VarSet is corrupted and cannot be iterated.");
    offsets.split_first().and_then(|(head, tail)| {
        if tail.is_empty() {
            return None;
        }
        *offsets = tail;
        Some(unsafe { *tail.get_unchecked(0) } - *head)
    })
}

impl<'a, V> Iterator for VarIter<'a, VarSetView<'a, V>>
where V: View<'a>,
      <V as View<'a>>::Type: SplitAt<'a> + Set,
{
    type Item = VarSetView<'a, V>;

    fn next(&mut self) -> Option<Self::Item> {
        pop_offset(&mut self.offsets).map(|n| {
            let (l, r) = self.data.split_at(n);
            self.data = r;
            l
        })
    }
}

impl<'a, T: 'a> Iterator for VarIter<'a, &'a [T]>
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        pop_offset(&mut self.offsets).map(|n| {
            let (l, r) = self.data.split_at(n);
            self.data = r;
            l
        })
    }
}

/// Mutable variant of `VarIter`.
pub struct VarIterMut<'a, S> {
    offsets: &'a [usize],
    data: S,
}

impl<'a, V> Iterator for VarIterMut<'a, VarSetViewMut<'a, V>>
where V: ViewMut<'a>,
      <V as ViewMut<'a>>::Type: SplitAt<'a> + Set + Default,
{
    type Item = VarSetViewMut<'a, V>;

    fn next(&mut self) -> Option<Self::Item> {
        // Get a unique mutable reference for the data.
        let data_slice = std::mem::replace(&mut self.data, VarSetViewMut { view: VarSet { offsets: &[], data: Default::default() }, phantom: std::marker::PhantomData });

        pop_offset(&mut self.offsets).map(move |n| {
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            l
        })
    }
}

impl<'a, T: 'a> Iterator for VarIterMut<'a, &'a mut [T]> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        // Get a unique mutable reference for the data.
        let data_slice = std::mem::replace(&mut self.data, &mut []);

        pop_offset(&mut self.offsets).map(move |n| {
            let (l, r) = data_slice.split_at_mut(n);
            self.data = r;
            l
        })
    }
}

/*
 * `IntoIterator` implementation for `VarSet`. Note that this type of
 * iterator allocates a new `Vec` at each iteration. This is an expensive
 * operation and is here for compatibility with the rest of Rust's ecosystem.
 * However, this iterator should be used sparingly.
 */

/// IntoIter for `VarSet`.
pub struct VarIntoIter<T> {
    offsets: std::iter::Peekable<std::vec::IntoIter<usize>>,
    data: Vec<T>,
}

impl<T> Iterator for VarIntoIter<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let begin = self
            .offsets
            .next()
            .expect("Var Set is corrupted and cannot be iterated.");
        if self.offsets.len() <= 1 {
            return None; // Ignore the last offset
        }
        let end = *self.offsets.peek().unwrap();
        let n = end - begin;
        let mut rest = self.data.split_off(n);
        std::mem::swap(&mut rest, &mut self.data);
        Some(rest) // These are the elements [0..n).
    }
}

//impl<S> IntoIterator for VarSet<S>
//where
//    S: Set,
//{
//    type Item = Vec<<Self as IntoFlatVec>::SubItem>;
//    type IntoIter = VarIntoIter<<Self as IntoFlatVec>::SubItem>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        let VarSet {
//            offsets,
//            data,
//        } = self;
//        VarIntoIter {
//            offsets: offsets.into_iter().peekable(),
//            data: data.into_flat_vec(),
//        }
//    }
//}

//impl<'a, S: Set + Push<&'a <S as Set>::Elem>> Push<&'a <Self as Set>::Elem> for VarSet<S> {
//    fn push(&mut self, element: &<Self as Set>::Elem) {
//        for i in element.iter() {
//            self.data.push(i);
//        }
//        self.offsets.push(self.data.len());
//    }
//}
//
//impl<'a, S: Set + View<'a>> Push<<S as View<'a>>::Type> for VarSet<S> {
//    fn push(&mut self, element: <S as View<'a>>::Type) {
//        self.
//    }
//}

impl<'a, S: 'a> View<'a> for VarSet<S>
where
    S: Set + View<'a>,
    <S as View<'a>>::Type: Set,
{
    type Type = VarSetView<'a, S>;

    /// Create a contiguous immutable (shareable) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let v1 = s.view();
    /// let v2 = v1.clone();
    /// let mut view1_iter = v1.iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// for (a,b) in v1.iter().zip(v2.iter()) {
    ///     assert_eq!(a,b);
    /// }
    /// ```
    fn view(&'a self) -> Self::Type {
        VarSetView::from_offsets(self.offsets.as_slice(), self.data.view())
    }
}

impl<'a, S: 'a> ViewMut<'a> for VarSet<S>
where
    S: Set + ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = VarSetViewMut<'a, S>;

    /// Create a contiguous mutable (unique) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = VarSet::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let mut v1 = s.view_mut();
    /// v1.iter_mut().next().unwrap()[0] = 100;
    /// let mut view1_iter = v1.iter();
    /// assert_eq!(Some(&[100][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// ```
    fn view_mut(&'a mut self) -> Self::Type {
        VarSetViewMut::from_offsets(self.offsets.as_slice(), self.data.view_mut())
    }
}

/// A helper trait used to abstract over owned `Vec`s and slices.
pub trait Buffer<T>: Set {
    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
}

impl<T> Buffer<T> for Vec<T> {
    fn first(&self) -> Option<&T> {
        <[T]>::first(self)
    }
    fn last(&self) -> Option<&T> {
        <[T]>::last(self)
    }
}

impl<T> Buffer<T> for [T] {
    fn first(&self) -> Option<&T> {
        <[T]>::first(self)
    }
    fn last(&self) -> Option<&T> {
        <[T]>::last(self)
    }
}

impl<'a, T, S: Buffer<T> + Set + ?Sized> Buffer<T> for &'a S {
    fn first(&self) -> Option<&T> {
        <S as Buffer<T>>::first(*self)
    }
    fn last(&self) -> Option<&T> {
        <S as Buffer<T>>::last(*self)
    }
}

impl<'a, T, S: Buffer<T> + Set + ?Sized> Buffer<T> for &'a mut S {
    fn first(&self) -> Option<&T> {
        <S as Buffer<T>>::first(*self)
    }
    fn last(&self) -> Option<&T> {
        <S as Buffer<T>>::last(*self)
    }
}
