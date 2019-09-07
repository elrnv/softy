use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

/// A collection of offsets into another collection.
/// This newtype is intended to verify basic invariants about offsets into
/// another collection, namely that the collection is monotonically increasing
/// and non-empty.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Offsets<O = Vec<usize>>(pub(crate) O);

impl<O: Set> Set for Offsets<O> {
    type Elem = O::Elem;
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<O: Viewed> Viewed for Offsets<O> {}

impl<'a> Offsets<&'a [usize]> {
    /// Pops an offset from the given slice of offsets and produces an increment for
    /// advancing the data pointer. This is a helper function for implementing
    /// iterators over `Chunked` types.
    /// This function panics if offsets is empty.
    pub(crate) fn pop_offset(&mut self) -> Option<usize> {
        debug_assert!(
            !self.is_empty(),
            "Chunked is corrupted and cannot be iterated."
        );
        self.0.split_first().and_then(|(head, tail)| {
            if tail.is_empty() {
                return None;
            }
            self.0 = tail;
            Some(unsafe { *tail.get_unchecked(0) } - *head)
        })
    }
}

impl<'a, O: AsRef<[usize]>> View<'a> for Offsets<O> {
    type Type = Offsets<&'a [usize]>;
    fn view(&'a self) -> Self::Type {
        Offsets(self.0.as_ref())
    }
}

impl<'a, O: AsMut<[usize]>> ViewMut<'a> for Offsets<O> {
    type Type = Offsets<&'a mut [usize]>;
    fn view_mut(&'a mut self) -> Self::Type {
        Offsets(self.0.as_mut())
    }
}

impl<O: AsRef<[usize]>> From<O> for Offsets<O> {
    fn from(offsets: O) -> Self {
        Offsets::new(offsets)
    }
}

impl<O: AsRef<[usize]>> AsRef<[usize]> for Offsets<O> {
    fn as_ref(&self) -> &[usize] {
        self.0.as_ref()
    }
}

impl<O: AsMut<[usize]>> AsMut<[usize]> for Offsets<O> {
    fn as_mut(&mut self) -> &mut [usize] {
        self.0.as_mut()
    }
}

/// A default set of offsets must allocate.
impl Default for Offsets<Vec<usize>> {
    fn default() -> Self {
        Offsets(vec![0])
    }
}

impl<O: Dummy> Dummy for Offsets<O> {
    unsafe fn dummy() -> Self {
        Offsets(Dummy::dummy())
    }
}

impl<O: AsRef<[usize]>> Offsets<O> {
    pub fn new(offsets: O) -> Self {
        let offsets_borrow = offsets.as_ref();
        assert!(!offsets_borrow.is_empty());
        Offsets(offsets)
    }
}

impl<O: AsMut<[usize]>> Offsets<O> {
    /// Moves an offset back by a specified amount, effectively transferring
    /// elements from the previous chunk to the specified chunk.
    ///
    /// # Panics
    /// This function panics if `at` is out of bounds and may cause Undefined Behavior in
    /// `Chunked` collections if zero.
    pub(crate) fn move_back(&mut self, at: usize, by: usize) {
        let offsets = self.as_mut();
        debug_assert!(at > 0 && at < offsets.len());
        offsets[at] -= by;
    }
    /// Moves an offset forward by a specified amount, effectively transferring
    /// elements from the previous chunk to the specified chunk.
    ///
    /// # Panics
    /// This function panics if `at` is out of bounds.
    pub(crate) fn move_forward(&mut self, at: usize, by: usize) {
        let offsets = self.as_mut();
        debug_assert!(at < offsets.len());
        offsets[at] += by;
    }

    /// Extend the last offset, which effectively increases the last chunk size.
    /// This function is similar to `self.move_forward(self.len() - 1)` but it does not perform
    /// bounds checking.
    pub(crate) fn extend_last(&mut self, by: usize) {
        let offsets = self.as_mut();
        let last = offsets.len() - 1;
        unsafe { *offsets.get_unchecked_mut(last) += by };
    }
}

impl<O: Push<usize>> Push<usize> for Offsets<O> {
    fn push(&mut self, item: usize) {
        self.0.push(item);
    }
}

impl<I: std::slice::SliceIndex<[usize]>, O: AsRef<[usize]>> std::ops::Index<I> for Offsets<O> {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.0.as_ref().index(index)
    }
}

impl<I: std::slice::SliceIndex<[usize]>, O: AsRef<[usize]> + AsMut<[usize]>> std::ops::IndexMut<I>
    for Offsets<O>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.as_mut().index_mut(index)
    }
}

impl<O: IntoIterator> IntoIterator for Offsets<O> {
    type Item = O::Item;
    type IntoIter = O::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Clear for Offsets {
    fn clear(&mut self) {
        self.0.clear();
    }
}

impl<O: std::iter::FromIterator<usize> + AsRef<[usize]>> std::iter::FromIterator<usize>
    for Offsets<O>
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        Offsets::new(O::from_iter(iter))
    }
}

impl<'a> SplitOffsetsAt for Offsets<&'a [usize]> {
    /// Splits a slice of offsets at the given index into two slices such that each
    /// slice is a valid slice of offsets. This means that the element at index
    /// `mid` is shared between the two output slices. In addition, return the
    /// offset of the middle element: this is the value `offsets[mid] - offsets[0]`.
    ///
    /// # WARNING
    /// Calling this function with an empty `offsets` slice or with `mid >=
    /// offsets.len()` will cause Undefined Behaviour.
    fn split_offsets_at(self, mid: usize) -> (Offsets<&'a [usize]>, Offsets<&'a [usize]>, usize) {
        debug_assert!(!self.is_empty());
        debug_assert!(mid < self.len());
        let l = &self.0[..=mid];
        let r = &self.0[mid..];
        // Skip bounds checking here since this function is not exposed to the user.
        let off = unsafe { *r.get_unchecked(0) - *l.get_unchecked(0) };
        (Offsets(l), Offsets(r), off)
    }
}

impl<O: AsRef<[usize]>> IndexRange for Offsets<O> {
    /// Return the `[begin..end)` bound of the chunk at the given index.
    fn index_range(&self, range: Range<usize>) -> Option<Range<usize>> {
        let offsets = self.0.as_ref();
        if range.end < offsets.len() {
            let first = unsafe { offsets.get_unchecked(0) };
            let cur = unsafe { offsets.get_unchecked(range.start) };
            let next = unsafe { offsets.get_unchecked(range.end) };
            let begin = cur - first;
            let end = next - first;
            Some(begin..end)
        } else {
            None
        }
    }
}

impl<'a, O: Get<'a, Range<usize>>> GetIndex<'a, Offsets<O>> for Range<usize> {
    type Output = Offsets<O::Output>;
    fn get(mut self, offsets: &Offsets<O>) -> Option<Self::Output> {
        self.end += 1;
        offsets.0.get(self).map(|offsets| Offsets(offsets))
    }
}

impl<O: Isolate<Range<usize>>> IsolateIndex<Offsets<O>> for Range<usize> {
    type Output = Offsets<O::Output>;
    fn try_isolate(mut self, offsets: Offsets<O>) -> Option<Self::Output> {
        self.end += 1;
        offsets.0.try_isolate(self).map(|offsets| Offsets(offsets))
    }
}

impl<O: Truncate> Truncate for Offsets<O> {
    fn truncate(&mut self, new_len: usize) {
        self.0.truncate(new_len);
    }
}

impl<O: RemovePrefix> RemovePrefix for Offsets<O> {
    fn remove_prefix(&mut self, n: usize) {
        self.0.remove_prefix(n);
    }
}

impl<O: ToOwned> ToOwned for Offsets<O> {
    type Owned = Offsets<O::Owned>;
    fn to_owned(self) -> Self::Owned {
        Offsets(self.0.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Test for the `split_offset_at` helper function.
    #[test]
    fn split_offset_at_test() {
        let offsets = Offsets(vec![0, 1, 2, 3, 4, 5]);
        let (l, r, off) = offsets.view().split_offsets_at(3);
        assert_eq!(l.0, &[0, 1, 2, 3]);
        assert_eq!(r.0, &[3, 4, 5]);
        assert_eq!(off, 3);
    }
}
