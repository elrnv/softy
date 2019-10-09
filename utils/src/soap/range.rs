/**
 * Ranges can be used as collections of read-only indices that can be truncated from either end.
 * It can be useful to specify chunked ranges or a selection from a range.
 * Since our collections must know about the length, only finite ranges are supported.
 */
use super::*;
use std::ops::{Range, RangeInclusive, RangeTo, RangeToInclusive};

impl<T: IntBound> BoundedRange for Range<T> {
    type Index = T;
    fn start(&self) -> Self::Index {
        self.start.clone()
    }
    fn end(&self) -> Self::Index {
        self.end.clone()
    }
}

impl<T: IntBound> BoundedRange for RangeInclusive<T> {
    type Index = T;
    fn start(&self) -> Self::Index {
        RangeInclusive::start(self).clone()
    }
    fn end(&self) -> Self::Index {
        RangeInclusive::end(self).clone() + 1
    }
}

impl<T: IntBound> BoundedRange for RangeTo<T> {
    type Index = T;
    fn start(&self) -> Self::Index {
        0usize.into()
    }
    fn end(&self) -> Self::Index {
        self.end.clone()
    }
}

impl<T: IntBound> BoundedRange for RangeToInclusive<T> {
    type Index = T;
    fn start(&self) -> Self::Index {
        0usize.into()
    }
    fn end(&self) -> Self::Index {
        self.end.clone() + 1
    }
}

macro_rules! impls_for_range {
    ($range:ident) => {
        impl<T> ValueType for $range<T> {}
        impl<I: IntBound> Set for $range<I> {
            type Elem = <Self as BoundedRange>::Index;
            type Atom = <Self as BoundedRange>::Index;
            fn len(&self) -> usize {
                (BoundedRange::end(self) - BoundedRange::start(self)).into()
            }
        }

        impl<N, I> UniChunkable<N> for $range<I> {
            type Chunk = StaticRange<N>;
        }
        impl<'a, I: IntBound> View<'a> for $range<I>
        where
            Self: IntoIterator,
        {
            type Type = Self;

            fn view(&'a self) -> Self::Type {
                self.clone()
            }
        }
    };
}

impls_for_range!(Range);
impls_for_range!(RangeInclusive);
impls_for_range!(RangeTo);
impls_for_range!(RangeToInclusive);

impl<'a, R> GetIndex<'a, R> for usize
where
    R: BoundedRange + Set,
{
    type Output = R::Index;
    fn get(self, rng: &R) -> Option<Self::Output> {
        if self >= rng.len() {
            return None;
        }
        Some(rng.start() + self)
    }
}

impl<'a, R> GetIndex<'a, R> for Range<usize>
where
    R: BoundedRange + Set,
{
    type Output = Range<R::Index>;
    fn get(self, rng: &R) -> Option<Self::Output> {
        if self.end > rng.len() {
            return None;
        }
        Some(Range {
            start: rng.start() + self.start,
            end: rng.start() + self.end,
        })
    }
}

impl<I, N> SplitPrefix<N> for Range<I>
where
    I: IntBound + Default + Copy + From<usize>,
    std::ops::RangeFrom<I>: Iterator<Item = I>,
    N: Unsigned + Array<I>,
    <N as Array<I>>::Array: Default,
{
    type Prefix = N::Array;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            return None;
        }

        let std::ops::Range { start, end } = self;

        let mut prefix: N::Array = Default::default();
        for (i, item) in (start.clone()..).zip(N::iter_mut(&mut prefix)) {
            *item = i;
        }

        let start = start.clone().into();

        let rest = Range {
            start: (start + N::to_usize()).into(),
            end,
        };

        Some((prefix, rest))
    }
}

impl<I, N> IntoStaticChunkIterator<N> for Range<I>
where
    Self: Set + SplitPrefix<N> + Dummy,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

impl<T> IntoFlat for Range<T> {
    type FlatType = Range<T>;
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<T> SplitAt for Range<T>
where
    T: From<usize>,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Range { start, end } = self;
        (
            Range {
                start,
                end: mid.into(),
            },
            Range {
                start: mid.into(),
                end,
            },
        )
    }
}

impl<T> Dummy for Range<T>
where
    T: Default,
{
    unsafe fn dummy() -> Self {
        Range {
            start: T::default(),
            end: T::default(),
        }
    }
}

impl<T> RemovePrefix for Range<T>
where
    T: From<usize>,
{
    fn remove_prefix(&mut self, n: usize) {
        self.start = n.into();
    }
}

impl<'a, T: Clone> StorageView<'a> for Range<T> {
    type StorageView = Self;
    fn storage_view(&'a self) -> Self::StorageView {
        self.clone()
    }
}

impl<T> Storage for Range<T> {
    type Storage = Range<T>;
    /// A range is a type of storage, simply return an immutable reference to self.
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<T> StorageMut for Range<T> {
    /// A range is a type of storage, simply return a mutable reference to self.
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<T: IntBound> Truncate for Range<T> {
    /// Truncate the range to a specified length.
    fn truncate(&mut self, new_len: usize) {
        self.end = self.start.clone() + new_len;
    }
}

impl<T: IntBound> IsolateIndex<Range<T>> for usize {
    type Output = T;
    fn try_isolate(self, rng: Range<T>) -> Option<Self::Output> {
        if self < rng.distance().into() {
            Some(rng.start + self)
        } else {
            None
        }
    }
}

impl<T: IntBound> IsolateIndex<Range<T>> for std::ops::Range<usize> {
    type Output = Range<T>;

    fn try_isolate(self, rng: Range<T>) -> Option<Self::Output> {
        if self.start >= rng.distance().into() || self.end > rng.distance().into() {
            return None;
        }

        Some(Range {
            start: rng.start.clone() + self.start,
            end: rng.start + self.end,
        })
    }
}

impl<T> IntoOwned for std::ops::Range<T> {
    type Owned = Self;
    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<T> IntoOwnedData for std::ops::Range<T> {
    type OwnedData = Self;
    fn into_owned_data(self) -> Self::OwnedData {
        self
    }
}

// Ranges are lightweight and are considered to be viewed types since they are
// cheap to operate on.
impl<T> Viewed for Range<T> {}
impl<T> Viewed for RangeInclusive<T> {}
impl<T> Viewed for RangeTo<T> {}
impl<T> Viewed for RangeToInclusive<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_set() {
        assert_eq!(Set::len(&(0..100)), 100);
        assert_eq!(Set::len(&(1..=100)), 100);
        assert_eq!(Set::len(&(..100)), 100);
        assert_eq!(Set::len(&(..=99)), 100);
    }
}
