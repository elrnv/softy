use super::*;

/// `UniChunked` Assigns a stride `N` to the specified collection.
///
/// # Example
///
/// ```rust
/// use utils::soap::*;
/// let s = UniChunked::<_, num::U2>::from_flat(vec![1,2,3,4,5,6]);
/// let mut iter = s.iter();
/// assert_eq!(Some(&[1,2]), iter.next());
/// assert_eq!(Some(&[3,4]), iter.next());
/// assert_eq!(Some(&[5,6]), iter.next());
/// assert_eq!(None, iter.next());
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct UniChunked<S, N> {
    pub(crate) data: S,
    pub(crate) chunks: N,
}

macro_rules! impl_from_grouped {
    ($nty:ty, $n:expr) => {
        impl<T> UniChunked<Vec<T>, $nty> {
            /// Create a `UniChunked` collection from a `Vec` of arrays.
            pub fn from_grouped_vec(data: Vec<[T; $n]>) -> UniChunked<Vec<T>, $nty> {
                use num::Unsigned;
                UniChunked {
                    chunks: <$nty>::new(),
                    data: reinterpret::reinterpret_vec(data),
                }
            }
            /// Create a `UniChunked` collection from a slice of arrays.
            pub fn from_grouped_slice(data: &[[T; $n]]) -> UniChunked<&[T], $nty> {
                use num::Unsigned;
                UniChunked {
                    chunks: <$nty>::new(),
                    data: reinterpret::reinterpret_slice(data),
                }
            }
            /// Create a `UniChunked` collection from a mutable slice of arrays.
            pub fn from_grouped_mut_slice(data: &mut [[T; $n]]) -> UniChunked<&mut [T], $nty> {
                use num::Unsigned;
                UniChunked {
                    chunks: <$nty>::new(),
                    data: reinterpret::reinterpret_mut_slice(data),
                }
            }
        }

        impl<T: Clone> Into<Vec<[T; $n]>> for UniChunked<Vec<T>, $nty> {
            fn into(self) -> Vec<[T; $n]> {
                ReinterpretAsGrouped::<$nty>::reinterpret_as_grouped(self.into_inner())
            }
        }
        impl<'a, T: Clone> Into<&'a [[T; $n]]> for UniChunked<&'a [T], $nty> {
            fn into(self) -> &'a [[T; $n]] {
                ReinterpretAsGrouped::<$nty>::reinterpret_as_grouped(self.into_inner())
            }
        }
        impl<'a, T: Clone> Into<&'a mut [[T; $n]]> for UniChunked<&'a mut [T], $nty> {
            fn into(self) -> &'a mut [[T; $n]] {
                ReinterpretAsGrouped::<$nty>::reinterpret_as_grouped(self.into_inner())
            }
        }
    };
}

impl_from_grouped!(num::U2, 2);
impl_from_grouped!(num::U3, 3);

/// Define aliases for common uniform chunked types.
pub type Chunked3<S> = UniChunked<S, num::U3>;
pub type Chunked2<S> = UniChunked<S, num::U2>;

impl<S, N> UniChunked<S, N> {
    /// Get a immutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let s = Chunked3::from_flat(v.clone());
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data(&self) -> &S {
        &self.data
    }
    /// Get a mutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6];
    /// let mut s = Chunked3::from_flat(v.clone());
    /// v[2] = 100;
    /// s.data_mut()[2] = 100;
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }
}

impl<S: Set, N: num::Unsigned> UniChunked<S, N> {
    /// Create a `UniChunked` collection that groups the elements of the
    /// original set into uniformly sized groups at compile time.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniChunked::<_, num::U3>::from_flat(vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_flat(data: S) -> Self {
        assert_eq!(data.len() % N::value(), 0);
        UniChunked {
            chunks: N::new(), // Zero sized type.
            data,
        }
    }

    /// Convert this `UniChunked` collection into its inner representation.
    pub fn into_inner(self) -> S {
        self.data
    }
}

impl<T, N> UniChunked<Vec<T>, N> {
    /// This function panics if `src` has doesn't have a length equal to `self.len()*N::value()`.
    pub fn copy_from_flat(&mut self, src: &[T])
    where
        T: Copy,
    {
        assert_eq!(src.len(), self.data.len());
        self.data.copy_from_slice(src);
    }
    /// This function panics if `src` has doesn't have a length equal to `self.len()*N::value()`.
    pub fn clone_from_flat(&mut self, src: &[T])
    where
        T: Clone,
    {
        assert_eq!(src.len(), self.data.len());
        self.data.clone_from_slice(src);
    }
}

/// An implementation of `Set` for a `UniChunked` collection of any type that
/// can be grouped as `N` sub-elements.
impl<S: Set, N: num::Unsigned> Set for UniChunked<S, N>
where
    S::Elem: Grouped<N>,
{
    type Elem = <S::Elem as Grouped<N>>::Type;

    /// Compute the length of this `UniChunked` collection as the number of
    /// grouped elements in the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniChunked::<_, num::U2>::from_flat(vec![0,1,2,3,4,5]);
    /// assert_eq!(s.len(), 3);
    /// let s = UniChunked::<_, num::U3>::from_flat(vec![0,1,2,3,4,5]);
    /// assert_eq!(s.len(), 2);
    /// ```
    fn len(&self) -> usize {
        self.data.len() / N::value()
    }
}

impl<S, N> Push<<<S as Set>::Elem as Grouped<N>>::Type> for UniChunked<S, N>
where
    S: Set + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Grouped<N>,
{
    /// Push a grouped element onto the `UniChunked` type. The pushed element must
    /// have exactly `N` sub-elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, num::U3>::from_flat(vec![1,2,3]);
    /// s.push([4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn push(&mut self, element: <<S as Set>::Elem as Grouped<N>>::Type) {
        Grouped::<N>::push_to(element, &mut self.data);
    }
}

impl<S, N> IntoIterator for UniChunked<S, N>
where
    S: Set + ReinterpretAsGrouped<N>,
    N: num::Unsigned,
{
    type Item = <<S as ReinterpretAsGrouped<N>>::Output as IntoIterator>::Item;
    type IntoIter = <<S as ReinterpretAsGrouped<N>>::Output as IntoIterator>::IntoIter;

    /// Convert a `UniChunked` collection into an iterator over grouped elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, num::U3>::from_flat(vec![1,2,3,4,5,6]);
    /// let mut iter = s.into_iter();
    /// assert_eq!(Some([1,2,3]), iter.next());
    /// assert_eq!(Some([4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.data.reinterpret_as_grouped().into_iter()
    }
}

impl<S, N> std::iter::FromIterator<<<S as Set>::Elem as Grouped<N>>::Type> for UniChunked<S, N>
where
    N: num::Unsigned,
    S: Set + Default + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Grouped<N>,
{
    /// Construct a `UniChunked` collection from an iterator that produces
    /// chunked elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![[1,2,3],[4,5,6]];
    /// let s: UniChunked::<Vec<usize>, num::U3> = v.into_iter().collect();
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = <<S as Set>::Elem as Grouped<N>>::Type>,
    {
        let mut s = UniChunked::default();
        for i in iter {
            s.push(i);
        }
        s
    }
}

impl<S: Set + Default, N: num::Unsigned> Default for UniChunked<S, N> {
    fn default() -> Self {
        Self::from_flat(S::default())
    }
}

pub trait ReinterpretAsGrouped<N> {
    type Output: IntoIterator;
    fn reinterpret_as_grouped(self) -> Self::Output;
}

impl<'a, S, N: num::Unsigned, M: num::Unsigned> ReinterpretAsGrouped<N> for UniChunked<S, M>
where
    S: ReinterpretAsGrouped<M>,
    <S as ReinterpretAsGrouped<M>>::Output: ReinterpretAsGrouped<N>,
{
    type Output = <<S as ReinterpretAsGrouped<M>>::Output as ReinterpretAsGrouped<N>>::Output;
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        self.data.reinterpret_as_grouped().reinterpret_as_grouped()
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretAsGrouped<N> for Vec<T>
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = Vec<<T as Grouped<N>>::Type>;
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        reinterpret::reinterpret_vec(self)
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretAsGrouped<N> for &'a [T]
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        reinterpret::reinterpret_slice(self)
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretAsGrouped<N> for &'a mut [T]
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a mut [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        reinterpret::reinterpret_mut_slice(self)
    }
}

//impl<'a, S, N> ReinterpretAsGrouped<N> for &'a S
//where
//    S: Set,
//    T: Grouped<N>,
//    <T as Grouped<N>>::Type: 'a,
//    N: num::Unsigned
//{
//    type Output = &'a [<T as Grouped<N>>::Type];
//    #[inline]
//    fn reinterpret_as_grouped(self) -> Self::Output {
//        reinterpret::reinterpret_slice(self.as_slice())
//    }
//}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretAsGrouped<N> for &'a Vec<T>
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        reinterpret::reinterpret_slice(self.as_slice())
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretAsGrouped<N> for &'a mut Vec<T>
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a mut [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        reinterpret::reinterpret_mut_slice(self.as_mut_slice())
    }
}

pub trait Grouped<N>
where
    Self: Sized,
{
    type Type;
    /// This method tells this type how it can be pushed to a set as a grouped
    /// type.
    fn push_to<S: Push<Self>>(element: Self::Type, set: &mut S);

    /// Construct a grouped type from a `RangeFrom<T>`.
    fn from_range(rng: std::ops::RangeFrom<Self>) -> Self::Type
    where
        Self: Default + Copy,
        std::ops::RangeFrom<Self>: Iterator<Item = Self>;
}

impl<T> Grouped<num::U1> for T {
    type Type = Self;
    fn push_to<S: Push<Self>>(element: Self::Type, set: &mut S) {
        set.push(element);
    }

    fn from_range(rng: std::ops::RangeFrom<T>) -> Self::Type
    where
        T: Default + Copy,
        std::ops::RangeFrom<T>: Iterator<Item = T>,
    {
        rng.start
    }
}

macro_rules! impl_grouped {
    ($nty:ty, $n:expr) => {
        impl<T: Clone> Grouped<$nty> for T {
            type Type = [Self; $n];
            fn push_to<S: Push<Self>>(element: Self::Type, set: &mut S) {
                for i in &element {
                    set.push(i.clone());
                }
            }
            fn from_range(rng: std::ops::RangeFrom<T>) -> Self::Type
            where
                T: Default + Copy,
                std::ops::RangeFrom<T>: Iterator<Item = T>,
            {
                let mut grp = [Default::default(); $n];
                for (i, item) in rng.zip(grp.iter_mut()) {
                    *item = i;
                }
                grp
            }
        }
    };
}

impl_grouped!(num::U2, 2);
impl_grouped!(num::U3, 3);

impl<T, N> std::borrow::Borrow<[T]> for UniChunked<&[T], N> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

impl<T, N> std::borrow::Borrow<[T]> for UniChunked<&mut [T], N> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

impl<T, N> std::borrow::BorrowMut<[T]> for UniChunked<&mut [T], N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.data
    }
}

macro_rules! impl_borrow_chunked {
    ($nty:ty, $n:expr) => {
        impl<T> std::borrow::Borrow<[[T; $n]]> for UniChunked<&[T], $nty> {
            fn borrow(&self) -> &[[T; $n]] {
                reinterpret::reinterpret_slice(self.data)
            }
        }
        impl<T> std::borrow::Borrow<[[T; $n]]> for UniChunked<&mut [T], $nty> {
            fn borrow(&self) -> &[[T; $n]] {
                reinterpret::reinterpret_slice(self.data)
            }
        }
        impl<T> std::borrow::BorrowMut<[[T; $n]]> for UniChunked<&mut [T], $nty> {
            fn borrow_mut(&mut self) -> &mut [[T; $n]] {
                reinterpret::reinterpret_mut_slice(self.data)
            }
        }
    };
}

impl_borrow_chunked!(num::U2, 2);
impl_borrow_chunked!(num::U3, 3);

impl<S, N> ToOwned for UniChunked<S, N>
where
    S: ToOwned,
    N: num::Unsigned,
{
    type Owned = UniChunked<<S as ToOwned>::Owned, N>;

    /// Convert this `UniChunked` collection to an owned one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let s_view = UniChunked::<_, num::U3>::from_flat(v.as_slice());
    /// let s_owned = UniChunked::<_, num::U3>::from_flat(v.clone());
    /// assert_eq!(utils::soap::ToOwned::to_owned(s_view), s_owned);
    /// ```
    fn to_owned(self) -> Self::Owned {
        UniChunked {
            data: self.data.to_owned(),
            chunks: N::new(),
        }
    }
}

/*
 * Indexing
 */

impl<'a, S, N> GetIndex<'a, UniChunked<S, N>> for usize
where
    S: Set + Get<'a, StaticRange<N>>,
    <S as Set>::Elem: Grouped<N>,
    N: num::Unsigned,
{
    type Output = S::Output;

    /// Get an element of the given `UniChunked` collection.
    fn get(self, chunked: &UniChunked<S, N>) -> Option<Self::Output> {
        if self <= chunked.len() {
            chunked.data.get(StaticRange::new(self * N::value()))
        } else {
            None
        }
    }
}

impl<'a, S, N> GetIndex<'a, UniChunked<S, N>> for std::ops::Range<usize>
where
    S: Set + Get<'a, std::ops::Range<usize>>,
    <S as Set>::Elem: Grouped<N>,
    N: num::Unsigned,
{
    type Output = UniChunked<S::Output, N>;

    /// Get a `[begin..end)` subview of the given `UniChunked` collection.
    fn get(self, chunked: &UniChunked<S, N>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            chunked
                .data
                .get(N::value() * self.start..N::value() * self.end)
                .map(|data| UniChunked {
                    data,
                    chunks: N::new(),
                })
        } else {
            None
        }
    }
}

impl<'a, S, N, I> Get<'a, I> for UniChunked<S, N>
where
    I: GetIndex<'a, Self>,
{
    type Output = I::Output;
    /// Get a subview from this `UniChunked` collection according to the given
    /// range. If the range is a single index, then a single chunk is returned
    /// instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3, 4,5,6, 7,8,9, 10,11,12];
    /// let s = UniChunked::<_, num::U3>::from_flat(v.view());
    ///
    /// assert_eq!(s.get(2), Some(&[7,8,9])); // Single index
    /// assert_eq!(s.get(2), Some(&s[2]));
    ///
    /// let r = s.get(1..3).unwrap();         // Range
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(Some(&[7,8,9]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// let r = s.get(2..).unwrap();         // RangeFrom
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[7,8,9]), iter.next());
    /// assert_eq!(Some(&[10,11,12]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// let r = s.get(..2).unwrap();         // RangeTo
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// assert_eq!(s.view(), s.get(..).unwrap()); // RangeFull
    /// assert_eq!(s.view(), s.view().get(..).unwrap());
    ///
    /// let r = s.get(1..=2).unwrap();       // RangeInclusive
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(Some(&[7,8,9]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// let r = s.get(..=1).unwrap();        // RangeToInclusive
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn get(&self, range: I) -> Option<I::Output> {
        range.get(self)
    }
}

//impl<'o, 'i: 'o, S, N> GetMutIndex<'i, 'o, UniChunked<S, N>> for usize
//where
//    S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
//    <S as Set>::Elem: Grouped<N>,
//    N: num::Unsigned,
//{
//    type Output = S::Output;
//
//    /// Get a mutable chunk reference of the given `UniChunked` collection.
//    fn get_mut(self, chunked: &'i mut UniChunked<S, N>) -> Option<Self::Output> {
//        if self <= chunked.len() {
//            Some(
//                chunked
//                    .data
//                    .get_mut(N::value() * self..N::value() * (self + 1)),
//            )
//        } else {
//            None
//        }
//    }
//}

impl<'a, S, N> GetMutIndex<'a, UniChunked<S, N>> for usize
where
    S: Set + GetMut<'a, StaticRange<N>>,
    <S as Set>::Elem: Grouped<N>,
    N: num::Unsigned,
{
    type Output = S::Output;

    /// Get a mutable chunk reference of the given `UniChunked` collection.
    fn get_mut(self, chunked: &mut UniChunked<S, N>) -> Option<Self::Output> {
        if self <= chunked.len() {
            chunked.data.get_mut(StaticRange::new(self * N::value()))
        } else {
            None
        }
    }
}

impl<'a, S, N> GetMutIndex<'a, UniChunked<S, N>> for std::ops::Range<usize>
where
    S: Set + GetMut<'a, std::ops::Range<usize>>,
    <S as Set>::Elem: Grouped<N>,
    N: num::Unsigned,
{
    type Output = UniChunked<S::Output, N>;

    /// Get a mutable `[begin..end)` subview of the given `UniChunked` collection.
    fn get_mut(self, chunked: &mut UniChunked<S, N>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            chunked
                .data
                .get_mut(N::value() * self.start..N::value() * self.end)
                .map(|data| UniChunked {
                    data,
                    chunks: N::new(),
                })
        } else {
            None
        }
    }
}

impl<'a, S, N, I> GetMut<'a, I> for UniChunked<S, N>
where
    I: GetMutIndex<'a, Self>,
{
    type Output = I::Output;
    /// Get a mutable subview from this `UniChunked` collection according to the
    /// given range. If the range is a single index, then a single chunk is
    /// returned instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3, 4,5,6, 0,0,0, 10,11,12];
    /// let mut s = Chunked3::from_flat(v.as_mut_slice());
    ///
    /// s.get_mut(2).unwrap().copy_from_slice(&[7,8,9]);
    /// assert_eq!(s.get(2), Some(&[7,8,9])); // Single index
    /// assert_eq!(v, vec![1,2,3, 4,5,6, 7,8,9, 10,11,12]);
    /// ```
    fn get_mut(&mut self, range: I) -> Option<I::Output> {
        range.get_mut(self)
    }
}

impl<T, N> std::ops::Index<usize> for UniChunked<Vec<T>, N>
where
    N: num::Unsigned,
    T: Grouped<N>,
{
    type Output = T::Type;

    /// Index the `UniChunked` `Vec` by `usize`. Note that this
    /// works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked,
    /// which cannot be represented by a single borrow. For more complex
    /// indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
    /// assert_eq!([7,8,9], s[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&self.data).index(idx)
    }
}

impl<T, N> std::ops::Index<usize> for UniChunked<&[T], N>
where
    N: num::Unsigned,
    T: Grouped<N>,
{
    type Output = T::Type;

    /// Immutably index the `UniChunked` borrowed slice by `usize`. Note
    /// that this works for chunked collections that are themselves not chunked,
    /// since the item at the index of a doubly chunked collection is itself
    /// chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
    /// assert_eq!([7,8,9], s.view()[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(self.data).index(idx)
    }
}

impl<T, N> std::ops::Index<usize> for UniChunked<&mut [T], N>
where
    N: num::Unsigned,
    T: Grouped<N>,
{
    type Output = T::Type;

    /// Immutably index the `UniChunked` mutably borrowed slice by `usize`. Note
    /// that this works for chunked collections that are themselves not chunked,
    /// since the item at the index of a doubly chunked collection is itself
    /// chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
    /// assert_eq!([7,8,9], s.view_mut()[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&*self.data).index(idx)
    }
}

impl<T, N> std::ops::IndexMut<usize> for UniChunked<Vec<T>, N>
where
    N: num::Unsigned,
    T: Grouped<N>,
{
    /// Mutably index the `UniChunked` `Vec` by `usize`. Note that this
    /// works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked,
    /// which cannot be represented by a single borrow. For more complex
    /// indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,0,0,0,10,11,12];
    /// let mut s = Chunked3::from_flat(v);
    /// s[2] = [7,8,9];
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11,12], s.into_flat().to_vec());
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&mut self.data).index_mut(idx)
    }
}

impl<T, N> std::ops::IndexMut<usize> for UniChunked<&mut [T], N>
where
    N: num::Unsigned,
    T: Grouped<N>,
{
    /// Mutably index the `UniChunked` mutably borrowed slice by `usize`.
    /// Note that this works for chunked collections that are themselves not
    /// chunked, since the item at the index of a doubly chunked collection is
    /// itself chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,0,0,0,10,11,12];
    /// let mut s = Chunked3::from_flat(v.as_mut_slice());
    /// s[2] = [7,8,9];
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11,12], v);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&mut *self.data).index_mut(idx)
    }
}

/*
 * Iteration
 */

impl<'a, S, N> UniChunked<S, N>
where
    S: View<'a>,
    <S as View<'a>>::Type: ReinterpretAsGrouped<N>,
    N: num::Unsigned,
{
    /// Produce an iterator over borrowed grouped elements of the `UniChunked`.
    ///
    /// # Examples
    ///
    /// The following is a simple test for iterating over a uniformly organized `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, num::U2>::from_flat(vec![1,2,3,4]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2]), iter.next());
    /// assert_eq!(Some(&[3,4]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// A more complex example consists of data organized as a nested `UniChunked`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s0 = UniChunked::<_, num::U2>::from_flat(vec![1,2, 3,4, 5,6, 7,8, 9,10, 11,12]);
    /// let s1 = UniChunked::<_, num::U3>::from_flat(s0);
    /// let mut iter1 = s1.iter();
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[1,2]), iter0.next());
    /// assert_eq!(Some(&[3,4]), iter0.next());
    /// assert_eq!(Some(&[5,6]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[7,8]), iter0.next());
    /// assert_eq!(Some(&[9,10]), iter0.next());
    /// assert_eq!(Some(&[11,12]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter(
        &'a self,
    ) -> <<<S as View<'a>>::Type as ReinterpretAsGrouped<N>>::Output as IntoIterator>::IntoIter
    {
        self.data.view().reinterpret_as_grouped().into_iter()
    }
}

impl<'a, S, N> UniChunked<S, N>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: ReinterpretAsGrouped<N>,
    N: num::Unsigned,
{
    /// Produce an iterator over mutably borrowed grouped elements of `UniChunked`.
    ///
    /// # Examples
    ///
    /// The following example shows a simple modification of a uniformly organized `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, num::U2>::from_flat(vec![0,1,2,3]);
    /// for i in s.iter_mut() {
    ///     i[0] += 1;
    ///     i[1] += 1;
    /// }
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2]), iter.next());
    /// assert_eq!(Some(&[3,4]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Nested `UniChunked`s can also be modified as follows:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s0 = UniChunked::<_, num::U2>::from_flat(vec![1,2, 3,4, 5,6, 7,8, 9,10, 11,12]);
    /// let mut s1 = UniChunked::<_, num::U3>::from_flat(s0);
    /// for i in s1.iter_mut() {
    ///     for j in i.iter_mut() {
    ///         j[0] += 1;
    ///         j[1] += 2;
    ///     }
    /// }
    /// let mut iter1 = s1.iter();
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[2,4]), iter0.next());
    /// assert_eq!(Some(&[4,6]), iter0.next());
    /// assert_eq!(Some(&[6,8]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let mut iter0 = iter1.next().unwrap().iter();
    /// assert_eq!(Some(&[8,10]), iter0.next());
    /// assert_eq!(Some(&[10,12]), iter0.next());
    /// assert_eq!(Some(&[12,14]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter_mut(
        &'a mut self,
    ) -> <<<S as ViewMut<'a>>::Type as ReinterpretAsGrouped<N>>::Output as IntoIterator>::IntoIter
    {
        self.data.view_mut().reinterpret_as_grouped().into_iter()
    }
}

impl<'a, S, N> View<'a> for UniChunked<S, N>
where
    S: Set + View<'a>,
    N: num::Unsigned + Copy,
    <S as View<'a>>::Type: Set,
{
    type Type = UniChunked<<S as View<'a>>::Type, N>;

    /// Create a `UniChunked` contiguous immutable (shareable) view into the
    /// underlying collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniChunked::<_, num::U2>::from_flat(vec![0,1,2,3]);
    /// let v1 = s.view(); // s is now inaccessible.
    /// let v2 = v1.clone();
    /// let mut view1_iter = v1.iter();
    /// assert_eq!(Some(&[0,1]), view1_iter.next());
    /// assert_eq!(Some(&[2,3]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// for ((a, b), c) in v1.iter().zip(v2.iter()).zip(s.iter()) {
    ///     assert_eq!(a,b);
    ///     assert_eq!(b,c);
    /// }
    /// ```
    fn view(&'a self) -> Self::Type {
        UniChunked::from_flat(self.data.view())
    }
}

impl<'a, S, N> ViewMut<'a> for UniChunked<S, N>
where
    S: Set + ViewMut<'a>,
    N: num::Unsigned,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = UniChunked<<S as ViewMut<'a>>::Type, N>;

    /// Create a `UniChunked` contiguous mutable (unique) view into the
    /// underlying collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, num::U2>::from_flat(vec![0,1,2,3]);
    /// let mut v = s.view_mut();
    /// {
    ///    v.iter_mut().next().unwrap()[0] = 100;
    /// }
    /// let mut view_iter = v.iter();
    /// assert_eq!(Some(&[100,1]), view_iter.next());
    /// assert_eq!(Some(&[2,3]), view_iter.next());
    /// assert_eq!(None, view_iter.next());
    /// ```
    fn view_mut(&'a mut self) -> Self::Type {
        UniChunked::from_flat(self.data.view_mut())
    }
}

impl<S: SplitAt + Set, N: num::Unsigned> SplitAt for UniChunked<S, N> {
    /// Split the current set into two distinct sets at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniChunked::<_, num::U2>::from_flat(vec![0,1,2,3]);
    /// let (l, r) = s.split_at(1);
    /// assert_eq!(l, UniChunked::<_, num::U2>::from_flat(vec![0,1]));
    /// assert_eq!(r, UniChunked::<_, num::U2>::from_flat(vec![2,3]));
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l, r) = self.data.split_at(mid * N::value());
        (UniChunked::from_flat(l), UniChunked::from_flat(r))
    }
}

impl<S: SplitPrefix<N> + Set, N: num::Unsigned> SplitPrefix<num::U1> for UniChunked<S, N> {
    type Prefix = S::Prefix;
    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        self.data
            .split_prefix()
            .map(|(prefix, rest)| (prefix, UniChunked::from_flat(rest)))
    }
}

impl<S: Dummy, N: num::Unsigned> Dummy for UniChunked<S, N> {
    fn dummy() -> Self {
        UniChunked {
            data: Dummy::dummy(),
            chunks: N::new(),
        }
    }
}

impl<S: IntoFlat, N> IntoFlat for UniChunked<S, N> {
    type FlatType = <S as IntoFlat>::FlatType;
    /// Strip away the uniform organization of the underlying data, and return the underlying data.
    fn into_flat(self) -> Self::FlatType {
        self.data.into_flat()
    }
}

/// Required for building `Subset`s of `UniChunked` types.
impl<S: RemovePrefix, N: num::Unsigned> RemovePrefix for UniChunked<S, N> {
    fn remove_prefix(&mut self, n: usize) {
        self.data.remove_prefix(n * N::value());
    }
}

impl<S: Clear, N> Clear for UniChunked<S, N> {
    fn clear(&mut self) {
        self.data.clear();
    }
}
