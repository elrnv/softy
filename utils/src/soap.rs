use std::marker::PhantomData;
//use std::rc::Rc;
//use cell::{Ref, RefMut, RefCell};
//use std::sync::{Arc, RwLock};

// Helper module defines a few useful unsigned type level integers.
// This is to avoid having to depend on yet another crate.
pub mod num {
    pub trait Unsigned {
        fn value() -> usize;
    }

    macro_rules! def_num {
        ($(($nty:ident, $n:expr)),*) => {
            $(
                #[derive(Debug, Clone, PartialEq)]
                pub struct $nty;
                impl Unsigned for $nty {
                    fn value() -> usize {
                        $n
                    }
                }
             )*
        }
    }

    def_num!((U1, 1), (U2, 2), (U3, 3));
}

/// A trait defining a raw buffer of data. This data is typed but not annotated so it can represent
/// anything. For example a buffer of floats can represent a set of vertex colours or vertex
/// positions.
pub trait Set {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    //fn push(&mut self, element: Self::Item);
}

impl<T> Set for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
    //fn push(&mut self, element: T) {
    //    self.push(element);
    //}
}

impl<T> Set for &Vec<T> {
    fn len(&self) -> usize {
        Vec::<T>::len(self)
    }
}

impl<T> Set for &mut Vec<T> {
    fn len(&self) -> usize {
        Vec::<T>::len(self)
    }
}

impl<T> Set for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T> Set for &mut [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

// Reference into a set.
//pub struct RefIter<'a, I> {
//    s: Ref<'a, I>,
//}
/*

impl<S: Set> Set for Rc<RefCell<S>> {
    fn len(&self) -> usize { self.borrow().len() }
}

impl<S: Set> VarIter for Rc<RefCell<S>> {
    type Iter = std::cell::Ref<S::Iter>;
    type IterMut = std::cell::RefMut<S::IterMut>;
    fn iter(&self) -> Self::Iter {
        std::cell::Ref::map(self.borrow(), |x| x.iter())
    }
    fn iter_mut(&self) -> Self::IterMut {
        std::cell::RefMut::map(self.borrow_mut(), |x| x.iter_mut())
    }
}

impl<S: Set> Set for Arc<RwLock<S>> {
    fn len(&self) -> usize { self.read().unwrap().len() }
}

impl<S: Set> VarIter for Arc<RwLock<S>> {
    type Iter = S::Iter;
    type IterMut = S::IterMut;
    fn iter(&self) -> Self::Iter {
        self.read().unwrap().iter()
    }
    fn iter_mut(&self) -> Self::IterMut {
        self.write().unwrap().iter_mut()
    }
}
*/

/*
 * VarSet
 */

/// A set of variable length elements. Each offset represents one element and gives the offset into
/// the data buffer for the first of subelement in the Set.
/// Offsets always begins with a 0 and ends with the length of the buffer.
#[derive(Clone, Debug)]
pub struct VarSet<S> {
    pub offsets: Vec<usize>,
    pub data: S,
}

impl<S: Set> VarSet<S> {
    pub fn from_offsets(offsets: Vec<usize>, data: S) -> Self {
        assert!(offsets.len() > 0);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap(), data.len());
        VarSet { offsets, data }
    }

    pub fn data(&self) -> &S {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl<S> VarSet<S>
where
    S: Set + IntoIterator
        + AppendVec<Item = <S as IntoIterator>::Item>
        + Default
        + std::iter::FromIterator<std::vec::Vec<<S as std::iter::IntoIterator>::Item>>,
{
    pub fn from_nested_vec(nested_data: Vec<Vec<<S as IntoIterator>::Item>>) -> Self {
        nested_data.into_iter().collect()
    }
}

// NOTE: There is currently no way to split ownership of a Vec without
// allocating. For this reason we opt to use a slice and defer allocation to
// a later step when the results may be collected into another Vec. This saves
// an extra allocation. We could make this more righteous with a custom
// allocator.
impl<'a, S> std::iter::FromIterator<&'a mut [<S as IntoIterator>::Item]> for VarSet<S>
where
    S: Set
        + ExtendFromSlice<Item = <S as IntoIterator>::Item>
        + Default
        + IntoIterator
        + std::iter::FromIterator<&'a mut [<S as IntoIterator>::Item]>,
    <S as IntoIterator>::Item: 'a,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a mut [<S as IntoIterator>::Item]>,
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
impl<S> std::iter::FromIterator<Vec<<S as IntoIterator>::Item>> for VarSet<S>
where
    S: Set
        + AppendVec<Item = <S as IntoIterator>::Item>
        + Default
        + IntoIterator
        + std::iter::FromIterator<Vec<<S as IntoIterator>::Item>>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<<S as IntoIterator>::Item>>,
    {
        let mut s = VarSet::default();
        for i in iter {
            s.push(i);
        }
        s
    }
}

impl<S: Set + Clone> Set for VarSet<S> {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl<S: Set + IntoIterator + AppendVec<Item = <S as IntoIterator>::Item>> VarSet<S> {
    fn push(&mut self, mut element: Vec<<S as IntoIterator>::Item>) {
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

impl<S: Set + IntoIterator + ExtendFromSlice<Item = <S as IntoIterator>::Item>> VarSet<S> {
    pub fn push_slice(&mut self, element: &[<S as IntoIterator>::Item]) {
        self.data.extend_from_slice(element);
        self.offsets.push(self.data.len());
    }
}

impl<S: Set + Default> Default for VarSet<S> {
    fn default() -> Self {
        Self::from_offsets(vec![0], S::default())
    }
}

impl<S> VarSet<S>
where
    S: Set + AsMutSlice,
{
    pub fn iter(&self) -> VarIter<<S as AsSlice>::Item> {
        VarIter {
            offsets: &self.offsets,
            data: self.data.as_slice(),
        }
    }
    pub fn iter_mut(&mut self) -> VarIterMut<<S as AsSlice>::Item> {
        VarIterMut {
            offsets: &self.offsets,
            data: self.data.as_mut_slice(),
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

pub trait AsSlice {
    type Item;
    fn as_slice(&self) -> &[Self::Item];
}
pub trait AsMutSlice: AsSlice {
    fn as_mut_slice(&mut self) -> &mut [Self::Item];
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

impl<T> AsSlice for Vec<T> {
    type Item = T;
    fn as_slice(&self) -> &[Self::Item] {
        Vec::as_slice(self)
    }
}
impl<T> AsMutSlice for Vec<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        Vec::as_mut_slice(self)
    }
}

/// A special iterator capable of iterating over a `VarSet`.
pub struct VarIter<'a, T> {
    offsets: &'a [usize],
    data: &'a [T],
}

/// Mutable variant of `VarIter`.
pub struct VarIterMut<'a, T> {
    offsets: &'a [usize],
    data: &'a mut [T],
}

impl<'a, T: 'a> Iterator for VarIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        match self.offsets.split_first() {
            Some((head, tail)) => {
                if tail.is_empty() {
                    return None;
                }
                self.offsets = tail;
                let n = unsafe { *tail.get_unchecked(0) } - *head;
                let (l, r) = self.data.split_at(n);
                self.data = r;
                Some(l)
            }
            None => {
                panic!("Var Set is corrupted and cannot be iterated.");
            }
        }
    }
}

impl<'a, T: 'a> Iterator for VarIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        // Get a unique mutable reference for the data.
        let data_slice = std::mem::replace(&mut self.data, &mut []);

        match self.offsets.split_first() {
            Some((head, tail)) => {
                if tail.is_empty() {
                    return None;
                }
                self.offsets = tail;
                let n = unsafe { *tail.get_unchecked(0) } - *head;
                let (l, r) = data_slice.split_at_mut(n);
                self.data = r;
                Some(l)
            }
            None => {
                panic!("Var Set is corrupted and cannot be iterated.");
            }
        }
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

/*
 * Uniformly spaced Set
 */

/// Assigns a uniform stride to the specified buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct UniSet<S, N> {
    pub data: S,
    phantom: PhantomData<N>,
}

impl<S: Set, N: num::Unsigned> UniSet<S, N> {
    pub fn from_flat(data: S) -> Self {
        assert_eq!(data.len() % N::value(), 0);
        UniSet {
            data,
            phantom: PhantomData,
        }
    }
    pub fn data(&self) -> &S {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }
}

//impl<S: Set, N> IntoFlatVec for UniSet<S, N> {
//    type SubItem = <S as IntoFlatVec>::SubItem;
//    fn into_flat_vec(self) -> Vec<Self::SubItem> {
//        self.data.into_flat_vec()
//    }
//}

pub trait Push<T> {
    fn push(&mut self, element: T);
}

impl<T> Push<T> for Vec<T> {
    fn push(&mut self, element: T) {
        Vec::push(self, element);
    }
}
impl<S: IntoIterator + Push<<S as IntoIterator>::Item>> Push<[<S as IntoIterator>::Item; 3]> for UniSet<S, num::U3> {
    fn push(&mut self, element: [<S as IntoIterator>::Item; 3]) {
        let [a, b, c] = element;
        self.data.push(a);
        self.data.push(b);
        self.data.push(c);
    }
}

impl<S, N> IntoIterator for UniSet<S, N>
where
    S: Set + IntoIterator + ReinterpretSet<N>,
    N: num::Unsigned,
{
    type Item = <<S as ReinterpretSet<N>>::Output as IntoIterator>::Item;
    type IntoIter = <<S as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.reinterpret_set().into_iter()
    }
}

impl<S> std::iter::FromIterator<[<S as IntoIterator>::Item; 3]> for UniSet<S, num::U3>
where
    S: Set + IntoIterator + Push<<S as IntoIterator>::Item> + Default + std::iter::FromIterator<<S as IntoIterator>::Item>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = [<S as IntoIterator>::Item; 3]>,
    {
        let mut s = UniSet::default();
        for i in iter {
            s.push(i);
        }
        s
    }
}

impl<S: Set + Default, N: num::Unsigned> Default for UniSet<S, N> {
    fn default() -> Self {
        Self::from_flat(S::default())
    }
}

impl<'a, S, N> UniSet<S, N>
where
    S: Set + 'a,
    &'a S: IntoIterator + ReinterpretSet<N>,
    &'a mut S: IntoIterator + ReinterpretSet<N>,
    N: num::Unsigned,
{
    pub fn iter(&'a self) -> <<&'a S as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter {
        (&self.data).reinterpret_set().into_iter()
    }
    pub fn iter_mut(
        &'a mut self,
    ) -> <<&'a mut S as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter {
        (&mut self.data).reinterpret_set().into_iter()
    }
}

pub trait ReinterpretSet<N> {
    type Output: IntoIterator;
    fn reinterpret_set(self) -> Self::Output;
}

impl<'a, T: 'a> ReinterpretSet<num::U3> for Vec<T> {
    type Output = Vec<[T; 3]>;
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_vec(self)
    }
}

impl<'a, T: 'a> ReinterpretSet<num::U3> for &'a Vec<T> {
    type Output = &'a [[T; 3]];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_slice(self.as_slice())
    }
}

impl<'a, T: 'a> ReinterpretSet<num::U3> for &'a mut Vec<T> {
    type Output = &'a mut [[T; 3]];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_mut_slice(self.as_mut_slice())
    }
}

/*
 * When N is U1, reinterpret_set is a noop.
 * We could implement it as returning an array of size 1, but this is not useful.
 */
impl<'a, T: 'a> ReinterpretSet<num::U1> for Vec<T> {
    type Output = Vec<T>;
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        self
    }
}

impl<'a, T: 'a> ReinterpretSet<num::U1> for &'a Vec<T> {
    type Output = &'a [T];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        self
    }
}

impl<'a, T: 'a> ReinterpretSet<num::U1> for &'a mut Vec<T> {
    type Output = &'a mut [T];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        self
    }
}

impl<S: Set + ReinterpretSet<N>, N: num::Unsigned> Set for UniSet<S, N> {
    fn len(&self) -> usize {
        self.data.len() / N::value()
    }
}

/*
 * Strict subset types corresponding to each of the set types.
 */

/// A VarSet that is a contiguous subset of some larger set (which could have any type).
/// `S` is any borrowed collection type.
#[derive(Clone, Debug)]
pub struct VarSubview<'a, S: ?Sized + 'a> {
    offset: &'a [usize],
    data: &'a S,
}

/// A UniSet that is a contiguous subset of some larger set (which could have any type).
/// `S` is any borrowed collection type.
#[derive(Clone, Debug)]
pub struct UniSubview<'a, S: ?Sized + 'a, N> {
    data: &'a S,
    phantom_size: PhantomData<N>,
}

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
#[derive(Clone, Debug)]
pub struct Subset<S> {
    indices: Vec<usize>,
    set: S,
}

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
    pub fn from_indices(mut indices: Vec<usize>, set: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.
        // Failure to enforce this invariant can cause race conditions.

        indices.sort_unstable();
        indices.dedup();

        Self::validate(Subset {
            indices,
            set,
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
    pub fn all(set: S) -> Self {
        Self::validate(Subset {
            indices: (0..set.len()).collect(),
            set,
        })
    }

    /// Panics if this subset is invald.
    #[inline]
    fn validate(self) -> Self {
        for &i in self.indices.iter() {
            assert!(i < self.set.len(), "Subset index out of bounds.");
        }
        self
    }
}

impl<'a, T> std::ops::Index<usize> for Subset<&'a [T]> {
    type Output = T;
    /// Immutably index the subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
    /// assert_eq!(subset[1], 3);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.set.index(self.indices[idx])
    }
}

impl<'a, T> std::ops::Index<usize> for Subset<&'a mut [T]> {
    type Output = T;
    /// Immutably index the subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(subset[1], 3);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.set.index(self.indices[idx])
    }
}

impl<'a, T> std::ops::IndexMut<usize> for Subset<&'a mut [T]> {
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
        self.set.index_mut(self.indices[idx])
    }
}

impl<S> Subset<S> {
    pub fn iter<'a, T: 'a>(&'a self) -> impl Iterator<Item = &'a T>
        where S: std::borrow::Borrow<[T]>
    {
        let ptr = self.set.borrow().as_ptr();
        self.indices.iter().map(move |&i| unsafe { &*ptr.add(i) })
    }
}

impl<S> Subset<S> {
    /// Mutably iterate over a subset.
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
    pub fn iter_mut<'a, T: 'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
        where S: std::borrow::BorrowMut<[T]>
    {
        let ptr = self.set.borrow_mut().as_mut_ptr();
        self.indices.iter().map(move |&i| unsafe { &mut *ptr.add(i) })
    }
}
