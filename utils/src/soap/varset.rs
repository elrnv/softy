use super::*;

/*
 * VarSet
 */

/// A set of variable length elements. Each offset represents one element and gives the offset into
/// the data buffer for the first of subelement in the Set.
/// Offsets always begins with a 0 and ends with the length of the buffer.
#[derive(Clone, Debug)]
pub struct VarSet<S, O = Vec<usize>> {
    data: S,
    offsets: O,
}

impl<S: Set> VarSet<S> {
    /// Construct a `VarSet` from a `Vec` of offsets into another set. This is
    /// the most efficient constructor, although it is also the most error
    /// prone.
    ///
    /// # Panics
    ///
    /// `offsets` must always begin with `0` and end with `data.len()`.
    /// This also implies that it cannot be empty. This function panics if these
    /// invariants aren't satisfied.
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
    pub fn from_offsets(offsets: Vec<usize>, data: S) -> Self {
        assert!(offsets.len() > 0);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap(), data.len());
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

impl<S> Set for VarSet<S>
where S: Set + Clone,
      <S as Set>::Elem: Sized,
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
    <S as View<'a>>::Type: IntoSlice<'a>,
{
    /// Produce an iterator over elements (borrowed slices) of a `VarSet`.
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
    pub fn iter(&'a self) -> VarIter<<<S as View<'a>>::Type as IntoSlice<'a>>::Item> {
        VarIter {
            offsets: &self.offsets,
            data: self.data.view().into_slice(),
        }
    }
}

impl<'a, S> VarSet<S>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: IntoMutSlice<'a>,
{
    /// Produce an iterator over elements (borrowed slices) of a `VarSet`.
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
    /// let mut varset_iter = s.iter();
    /// assert_eq!(vec![2,3,4], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(vec![6,7], varset_iter.next().unwrap().to_vec());
    /// assert_eq!(None, varset_iter.next());
    /// ```
    pub fn iter_mut(&'a mut self) -> VarIterMut<<<S as ViewMut<'a>>::Type as IntoSlice<'a>>::Item> {
        VarIterMut {
            offsets: &self.offsets,
            data: self.data.view_mut().into_mut_slice(),
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

pub trait IntoSlice<'a> {
    type Item;
    fn into_slice(self) -> &'a [Self::Item];
}
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
