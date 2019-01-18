//! This modules defines specialized iterators useful for mapping between iterators filtered by a
//! mask and iterators expanded by a mask of boolean values.
//!
//! Note: This may be specialized to bitsets later.

/// Filter out elements whose corresponding value in the mask iterator is false.
#[derive(Clone, Debug)]
pub struct FilterMaskedIterator<I, M> {
    iter: I,
    mask: M,
}

impl<I, M> Iterator for FilterMaskedIterator<I, M>
where
    M: Iterator<Item = bool>,
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        let FilterMaskedIterator {
            ref mut iter,
            ref mut mask,
        } = *self;

        for (x, m) in iter.zip(mask) {
            if m {
                return Some(x);
            }
        }
        None
    }
}

pub trait FilterMaskedTrait {
    fn filter_masked<M>(self, mask_iter: M) -> FilterMaskedIterator<Self, M>
    where
        Self: Sized + Iterator,
        M: Iterator<Item = bool>,
    {
        FilterMaskedIterator {
            iter: self,
            mask: mask_iter,
        }
    }
}

impl<I: Iterator> FilterMaskedTrait for I {}

/// Expand a given iterator into an iterator which spits out values wherever the given mask
/// evaluates to `true` and some default value otherwise.
/// This can be seen as the reverse operation of the `FilterMaskedIterator`.
#[derive(Clone, Debug)]
pub struct DistributeIterator<I, M, T> {
    iter: I,
    mask: M,
    default: T,
}

impl<I, M, T> Iterator for DistributeIterator<I, M, T>
where
    T: Clone,
    M: Iterator<Item = bool>,
    I: Iterator<Item = T>,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        let DistributeIterator {
            ref mut iter,
            ref mut mask,
            ref mut default,
        } = *self;

        mask.next().and_then(|m| {
            if m {
                iter.next()
            } else {
                Some(default.clone())
            }
        })
    }
}

pub trait DistributeTrait {
    fn distribute<M, T>(self, mask_iter: M, def: T) -> DistributeIterator<Self, M, T>
    where
        Self: Sized + Iterator<Item = T>,
        T: Clone,
        M: Iterator<Item = bool>,
    {
        DistributeIterator {
            iter: self,
            mask: mask_iter,
            default: def,
        }
    }
}

impl<T: Clone, I: Iterator<Item = T>> DistributeTrait for I {}
