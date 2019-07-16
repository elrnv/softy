use super::*;

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
}

impl<S> VarSet<S>
where
    S: Set
        + IntoIterator
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
    type Elem = Vec<S::Elem>;
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

//impl<'a, S: std::ops::Index<> + Clone> GetElem<'a> for VarSet<S> {
//    fn get(&'a self, idx: usize) -> Self::Elem {
//        self.data[self.offsets[idx]..self.offsets[idx+1]].collect()
//    }
//}

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
