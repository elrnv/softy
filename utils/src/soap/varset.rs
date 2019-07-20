use super::*;

/*
 * VarSet
 */

/// A set of variable length elements. Each offset represents one element and gives the offset into
/// the data buffer for the first of subelement in the Set.
/// Offsets always ends with the length of the buffer minus the value of the first offset.
#[derive(Copy, Clone, Debug, PartialEq)]
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
    S: Set + Push<<S as Set>::Elem> + Default,
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
    S: Set + Default + Push<<S as Set>::Elem>,
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
    /// dbg!(&s);
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
            s.push(i);
        }
        s
    }
}

//impl<'a, S: std::ops::Index<> + Clone> GetElem<'a> for VarSet<S> {
//    fn get(&'a self, idx: usize) -> Self::Elem {
//        self.data[self.offsets[idx]..self.offsets[idx+1]].collect()
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

impl<'a, T> IntoIterator for VarView<'a, &'a [T]> {
    type Item = &'a [T];
    type IntoIter = VarIter<'a, &'a [T]>;

    fn into_iter(self) -> Self::IntoIter {
        VarIter {
            offsets: self.offsets,
            data: self.data,
        }
    }
}

impl<'a, S, O> VarSet<S, O>
where S: View<'a>,
      O: std::borrow::Borrow<[usize]>,
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
    /// let mut varset_iter = s.iter();
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
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter(&'a self) -> VarIter<'a, <S as View<'a>>::Type> {
        VarIter {
            offsets: self.offsets.borrow(),
            data: self.data.view(),
        }
    }
}


impl<'a, S, O> VarSet<S, O>
where
    S: ViewMut<'a>,
    O: std::borrow::Borrow<[usize]>,
{
    /// Produce a mutable iterator over elements (borrowed slices) of a
    /// `VarSet`.
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
    /// for mut v0 in s1.view_mut().iter_mut() {
    ///     for i in v0.iter_mut() {
    ///         for j in i.iter_mut() {
    ///             *j += 1;
    ///         }
    ///     }
    /// }
    /// let v1 = s1.view();
    /// let mut iter1 = v1.iter();
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter_mut(&'a mut self) -> VarIterMut<'a, <S as ViewMut<'a>>::Type> {
        VarIterMut {
            offsets: self.offsets.borrow(),
            data: self.data.view_mut(),
        }
    }
}

macro_rules! impl_split_at_fn {
    ($self:ident, $split_fn:ident, $mid:expr) => {
        {
            let (offsets_l, offsets_r, off) = split_offsets_at($self.offsets, $mid);
            let (data_l, data_r) = $self.data.$split_fn(off);
            ( VarSet { offsets: offsets_l, data: data_l },
              VarSet { offsets: offsets_r, data: data_r } )
        }
    }
}

impl<V: SplitAt + Set> SplitAt for VarView<'_, V> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at, mid)
    }
}

impl<T> SplitAt for VarView<'_, &[T]> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at, mid)
    }
}

impl<T> SplitAt for VarView<'_, &mut [T]> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at_mut, mid)
    }
}

/// A special iterator capable of iterating over a `VarSet`.
pub struct VarIter<'a, S> {
    offsets: &'a [usize],
    data: S,
}

/// Splits a slice of offsets at the given index into two slices such that each
/// slice is a valid slice of offsets. This means that the element at index
/// `mid` is shared between the two output slices. In addition, return the
/// offset of the middle element: this is the value `offsets[mid] - offsets[0]`.
///
/// # WARNING
/// Calling this function with an empty `offsets` slice or with `mid >=
/// offsets.len()` will cause Undefined Behaviour.
fn split_offsets_at(offsets: &[usize], mid: usize) -> (&[usize], &[usize], usize) {
    debug_assert!(!offsets.is_empty());
    debug_assert!(mid < offsets.len());
    let l = &offsets[..mid+1];
    let r = &offsets[mid..];
    // Skip bounds checking here since this function is not exposed to the user.
    let off = unsafe { *r.get_unchecked(0) - *l.get_unchecked(0) };
    (l, r, off)
}

/// Test for the `split_offset_at` helper function.
#[test]
fn split_offset_at_test() {
    let offsets = vec![0,1,2,3,4,5];
    let (l, r, off) = split_offsets_at(offsets.as_slice(), 3);
    assert_eq!(l, &[0,1,2,3]);
    assert_eq!(r, &[3,4,5]);
    assert_eq!(off, 3);
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

impl<'a, V> Iterator for VarIter<'a, V>
where V: SplitAt + Set + Dummy,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        let data_slice = std::mem::replace(&mut self.data, Dummy::dummy());
        pop_offset(&mut self.offsets).map(|n| {
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            l
        })
    }
}

impl<'a, T> Iterator for VarIter<'a, &'a [T]>
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

impl<'a, V: 'a> Iterator for VarIterMut<'a, V>
where V: SplitAt + Set + Dummy,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        // Get a unique mutable reference for the data.
        let data_slice = std::mem::replace(&mut self.data, Dummy::dummy());

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
pub struct VarIntoIter<S> {
    offsets: std::iter::Peekable<std::vec::IntoIter<usize>>,
    data: S,
}

impl<S> Iterator for VarIntoIter<S>
    where S: SplitOff + Set
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        let begin = self
            .offsets
            .next()
            .expect("VarSet is corrupted and cannot be iterated.");
        if self.offsets.len() <= 1 {
            return None; // Ignore the last offset
        }
        let end = *self.offsets.peek().unwrap();
        let n = end - begin;
        let mut l = self.data.split_off(n);
        std::mem::swap(&mut l, &mut self.data);
        Some(l) // These are the elements [0..n).
    }
}

impl<S: SplitOff + Set> SplitOff for VarSet<S> {
    fn split_off(&mut self, mid: usize) -> Self {
        // Note: Allocations in this function heavily outweigh any cost in bounds checking.
        assert!(!self.offsets.is_empty());
        assert!(mid < self.offsets.len());
        let off = self.offsets[mid] - self.offsets[0];
        let offsets_l = self.offsets[..mid+1].to_vec();
        let offsets_r = self.offsets[mid..].to_vec();
        self.offsets = offsets_l;
        let data_r = self.data.split_off(off);
        VarSet::from_offsets(offsets_r, data_r)
    }
}

impl<T> SplitOff for Vec<T> {
    fn split_off(&mut self, mid: usize) -> Self {
        Vec::split_off(self, mid)
    }
}


impl<S> IntoIterator for VarSet<S>
where
    S: SplitOff + Set,
{
    type Item = S;
    type IntoIter = VarIntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        let VarSet {
            offsets,
            data,
        } = self;
        VarIntoIter {
            offsets: offsets.into_iter().peekable(),
            data: data,
        }
    }
}

impl<'a, S: Set + Push<<S as Set>::Elem>> Push<<Self as Set>::Elem> for VarSet<S> {
    /// Push an element onto this `VarSet`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = VarSet::<Vec<usize>>::from_offsets(vec![0,1,4], vec![0,1,2,3]);
    /// s.push(vec![4,5]);
    /// let v1 = s.view();
    /// let mut view1_iter = v1.into_iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// ```
    fn push(&mut self, element: <Self as Set>::Elem) {
        for elem in element.into_iter() {
            self.data.push(elem);
        }
        self.offsets.push(self.data.len());
    }
}

impl<'a, S, O> View<'a> for VarSet<S, O>
where
    S: Set + View<'a>,
    <S as View<'a>>::Type: Set,
    O: std::borrow::Borrow<[usize]>,
{
    type Type = VarView<'a, <S as View<'a>>::Type>;

    /// Create a contiguous immutable (shareable) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = VarSet::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let v1 = s.view();
    /// let v2 = v1.clone();
    /// let mut view1_iter = v1.into_iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// for (a,b) in v1.into_iter().zip(v2.into_iter()) {
    ///     assert_eq!(a,b);
    /// }
    /// ```
    fn view(&'a self) -> Self::Type {
        VarSet::from_offsets(self.offsets.borrow(), self.data.view())
    }
}

impl<'a, S, O> ViewMut<'a> for VarSet<S, O>
where
    S: Set + ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set,
    O: std::borrow::Borrow<[usize]>,
{
    type Type = VarView<'a, <S as ViewMut<'a>>::Type>;

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
        VarView::from_offsets(self.offsets.borrow(), self.data.view_mut())
    }
}

/*
 * Utility traits intended to expose the necessary behaviour to implement `VarSet`s
 */

pub trait ExtendFromSlice {
    type Item;
    fn extend_from_slice(&mut self, other: &[Self::Item]);
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

impl<T> IntoFlat for Vec<T> {
    type FlatType = Vec<T>;
    /// Since a `Vec` has no information about the structure of its underlying
    /// data, this is effectively a no-op.
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<S> IntoFlat for VarSet<S> {
    type FlatType = S;
    /// Strip the organizational information from this set, returning the
    /// underlying collection type.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s0 = VarSet::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// let s1 = VarSet::from_offsets(vec![0,1,4,5], s0.clone());
    /// assert_eq!(s1.into_flat(), s0.clone());
    /// assert_eq!(s0.into_flat(), v);
    /// ```
    fn into_flat(self) -> Self::FlatType {
        self.data
    }
}

impl<T: Clone> ExtendFromSlice for Vec<T> {
    type Item = T;
    fn extend_from_slice(&mut self, other: &[Self::Item]) {
        Vec::extend_from_slice(self, other);
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

impl<S: Default> Default for VarSet<S> {
    /// Construct an empty `VarSet`.
    fn default() -> Self {
        VarSet {
            data: Default::default(),
            offsets: vec![0],
        }
    }
}

impl<S: Dummy, O: Dummy> Dummy for VarSet<S, O> {
    fn dummy() -> Self {
        VarSet { data: Dummy::dummy(), offsets: Dummy::dummy() }
    }
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
