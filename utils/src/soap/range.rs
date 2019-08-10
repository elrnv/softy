/**
 * Ranges can be used as collections of read-only indices that can be truncated from either end.
 * It can be useful to specify chunked ranges or a selection from a range.
 * Since our collections must know about the length, only finite ranges are supported.
 */
use super::*;
use std::ops::{Range, RangeInclusive, RangeTo, RangeToInclusive};

impl<T> Owned for Range<T> {}

// To clarify this implements indexing into a Range using a StaticRange.
//impl<'a, T, N> GetIndex<'a, Range<T>> for StaticRange<N>
//where
//    N: num::Unsigned,
//    T: Grouped<N>,
//    <T as Grouped<N>>::Type: 'a,
//{
//    type Output = T::Type;
//    fn get(self, rng: &Range<T>) -> Option<Self::Output> {
//        //TODO: implement this
//        let size = rng.len();
//    }
//}

impl<I: Into<usize> + Clone> Set for Range<I> {
    type Elem = I;
    fn len(&self) -> usize {
        self.end.clone().into() - self.start.clone().into()
    }
}

impl<I: Into<usize> + Clone> Set for RangeInclusive<I> {
    type Elem = I;
    fn len(&self) -> usize {
        self.end().clone().into() - self.start().clone().into() + 1
    }
}

impl<I: Into<usize> + Clone> Set for RangeTo<I> {
    type Elem = I;
    fn len(&self) -> usize {
        self.end.clone().into()
    }
}

impl<I: Into<usize> + Clone> Set for RangeToInclusive<I> {
    type Elem = I;
    fn len(&self) -> usize {
        self.end.clone().into() + 1
    }
}

impl<I, N> SplitPrefix<N> for Range<I>
where
    I: Into<usize> + Default + Copy + From<usize> + Grouped<N>,
    std::ops::RangeFrom<I>: Iterator<Item = I>,
    N: num::Unsigned,
{
    type Prefix = I::Type;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }

        let std::ops::Range { start, end } = self;

        let prefix = I::from_range(std::ops::RangeFrom {
            start: start.clone(),
        });
        let start = start.clone().into();

        let rest = Range {
            start: (start + N::value()).into(),
            end,
        };

        Some((prefix, rest))
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
    fn dummy() -> Self {
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
