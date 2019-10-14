use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

/// An annotated offset collection to be used as indices for `Chunked` collections that indicates if the
/// indexed chunks are indeed sorted.
/// Note that `offsets` themeselves are *always* sorted. This type annotates whether the *data* is
/// sorted when this type is used for the offsets in `Chunked` types.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SortedChunks<O = Vec<usize>> {
    pub(crate) sorted: bool,
    pub(crate) offsets: Offsets<O>,
}

impl<O: AsRef<[usize]>> AsRef<[usize]> for SortedChunks<O> {
    fn as_ref(&self) -> &[usize] {
        self.offsets.as_ref()
    }
}

impl<O: AsMut<[usize]>> AsMut<[usize]> for SortedChunks<O> {
    fn as_mut(&mut self) -> &mut [usize] {
        self.offsets.as_mut()
    }
}

impl<O: Set> Set for SortedChunks<O> {
    type Elem = O::Elem;
    type Atom = O::Atom;
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<O: Viewed> Viewed for SortedChunks<O> {}

impl<'a, O: AsRef<[usize]>> View<'a> for SortedChunks<O> {
    type Type = SortedChunks<&'a [usize]>;
    fn view(&'a self) -> Self::Type {
        SortedChunks {
            sorted: self.sorted,
            offsets: self.offsets.view(),
        }
    }
}

impl<'a, O: AsMut<[usize]>> ViewMut<'a> for SortedChunks<O> {
    type Type = SortedChunks<&'a mut [usize]>;
    fn view_mut(&'a mut self) -> Self::Type {
        SortedChunks {
            sorted: self.sorted,
            offsets: self.offsets.view_mut(),
        }
    }
}

impl<O: AsRef<[usize]>> From<O> for SortedChunks<O> {
    fn from(offsets: O) -> Self {
        SortedChunks::new(offsets)
    }
}

/// A default set of offsets must allocate.
impl Default for SortedChunks<Vec<usize>> {
    fn default() -> Self {
        SortedChunks {
            sorted: true,
            offsets: Default::default(),
        }
    }
}

impl<O: Dummy> Dummy for SortedChunks<O> {
    unsafe fn dummy() -> Self {
        SortedChunks {
            sorted: true,
            offsets: Dummy::dummy(),
        }
    }
}

impl<O: AsRef<[usize]>> SortedChunks<O> {
    pub fn new(offsets: O) -> Self {
        SortedChunks {
            sorted: false,
            offsets: Offsets::new(offsets),
        }
    }
}

impl<O: Push<usize>> Push<usize> for SortedChunks<O> {
    fn push(&mut self, item: usize) {
        self.offsets.push(item);
    }
}

impl<I: std::slice::SliceIndex<[usize]>, O: AsRef<[usize]>> std::ops::Index<I> for SortedChunks<O> {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        self.offsets.index(index)
    }
}

impl<I: std::slice::SliceIndex<[usize]>, O: AsRef<[usize]> + AsMut<[usize]>> std::ops::IndexMut<I>
    for SortedChunks<O>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.offsets.index_mut(index)
    }
}

impl<O: IntoIterator> IntoIterator for SortedChunks<O> {
    type Item = O::Item;
    type IntoIter = O::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.offsets.into_iter()
    }
}

impl Clear for SortedChunks {
    fn clear(&mut self) {
        self.offsets.clear();
    }
}

impl<O: std::iter::FromIterator<usize> + AsRef<[usize]>> std::iter::FromIterator<usize>
    for SortedChunks<O>
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        SortedChunks::new(O::from_iter(iter))
    }
}

impl<'a> SplitOffsetsAt for SortedChunks<&'a [usize]> {
    fn split_offsets_at(
        self,
        mid: usize,
    ) -> (SortedChunks<&'a [usize]>, SortedChunks<&'a [usize]>, usize) {
        let (offsets_l, offsets_r, off) = self.offsets.split_offsets_at(mid);
        (
            SortedChunks {
                sorted: self.sorted,
                offsets: offsets_l,
            },
            SortedChunks {
                sorted: self.sorted,
                offsets: offsets_r,
            },
            off,
        )
    }
}

impl<O: AsRef<[usize]>> IndexRange for SortedChunks<O> {
    /// Return the `[begin..end)` bound of the chunk at the given index.
    fn index_range(&self, range: Range<usize>) -> Option<Range<usize>> {
        self.offsets.index_range(range)
    }
}

impl<'a, O: Get<'a, Range<usize>>> GetIndex<'a, SortedChunks<O>> for Range<usize> {
    type Output = SortedChunks<O::Output>;
    fn get(self, sorted_chunks: &SortedChunks<O>) -> Option<Self::Output> {
        let SortedChunks { offsets, sorted } = sorted_chunks;
        offsets.get(self).map(move |offsets| SortedChunks {
            sorted: *sorted,
            offsets,
        })
    }
}

impl<O: Isolate<Range<usize>>> IsolateIndex<SortedChunks<O>> for Range<usize> {
    type Output = SortedChunks<O::Output>;
    fn try_isolate(self, sorted_chunks: SortedChunks<O>) -> Option<Self::Output> {
        let SortedChunks { offsets, sorted } = sorted_chunks;
        offsets
            .try_isolate(self)
            .map(move |offsets| SortedChunks { sorted, offsets })
    }
}

impl<O: Truncate> Truncate for SortedChunks<O> {
    fn truncate(&mut self, new_len: usize) {
        self.offsets.truncate(new_len);
    }
}

impl<O: RemovePrefix> RemovePrefix for SortedChunks<O> {
    fn remove_prefix(&mut self, n: usize) {
        self.offsets.remove_prefix(n);
    }
}

impl<O: IntoOwned> IntoOwned for SortedChunks<O> {
    type Owned = SortedChunks<O::Owned>;
    fn into_owned(self) -> Self::Owned {
        SortedChunks {
            sorted: self.sorted,
            offsets: self.offsets.into_owned(),
        }
    }
}

impl<O: Reserve> Reserve for SortedChunks<O> {
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.offsets.reserve_with_storage(n, storage_n);
    }
}
