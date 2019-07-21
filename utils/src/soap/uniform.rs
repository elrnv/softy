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
    S: Set + ReinterpretSet<N>,
    N: num::Unsigned,
{
    type Item = <<S as ReinterpretSet<N>>::Output as IntoIterator>::Item;
    type IntoIter = <<S as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter;

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
        self.data.reinterpret_set().into_iter()
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

pub trait ReinterpretSet<N> {
    type Output: IntoIterator;
    fn reinterpret_set(self) -> Self::Output;
}

impl<'a, S, N: num::Unsigned, M: num::Unsigned> ReinterpretSet<N> for UniChunked<S, M>
where
    S: ReinterpretSet<M>,
    <S as ReinterpretSet<M>>::Output: ReinterpretSet<N>,
{
    type Output = <<S as ReinterpretSet<M>>::Output as ReinterpretSet<N>>::Output;
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        self.data.reinterpret_set().reinterpret_set()
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretSet<N> for Vec<T>
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = Vec<<T as Grouped<N>>::Type>;
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_vec(self)
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretSet<N> for &'a Vec<T>
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_slice(self.as_slice())
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretSet<N> for &'a mut Vec<T>
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a mut [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_mut_slice(self.as_mut_slice())
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretSet<N> for &'a [T]
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_slice(self)
    }
}

impl<'a, T: Grouped<N>, N: num::Unsigned> ReinterpretSet<N> for &'a mut [T]
where
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a mut [<T as Grouped<N>>::Type];
    #[inline]
    fn reinterpret_set(self) -> Self::Output {
        reinterpret::reinterpret_mut_slice(self)
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
}

impl<T> Grouped<num::U1> for T {
    type Type = Self;
    fn push_to<S: Push<Self>>(element: Self::Type, set: &mut S) {
        set.push(element);
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
        }
    };
}

impl_grouped!(num::U2, 2);
impl_grouped!(num::U3, 3);


//impl<'a, S, N> Get<'a, usize> for UniChunked<S, N>
//where
//    S: Set + Get<'a, usize>,
//    N: num::Unsigned,
//    <S as Set>::Elem: Grouped<N>,
//{
//    type Output = <S as Set>::Elem;
//    fn get(&self, idx: usize) -> Self::Output {
//        self.data.[idx]
//    }
//}

impl<T> std::borrow::Borrow<[T]> for UniChunked<&[T], num::U1> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

impl<T> std::borrow::Borrow<[T]> for UniChunked<&mut [T], num::U1> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

impl<T> std::borrow::BorrowMut<[T]> for UniChunked<&mut [T], num::U1> {
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

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

//impl<S: std::ops::Index<usize>, N> std::ops::Index<usize> for UniChunked<S, N> {
//    type Output = <UniChunked<S, N> as Set>::Elem;
//    /// Immutably index the subset.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let s = UniChunked::<_, num::U3>::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
//    /// assert_eq!([7,8,9], s[2]);
//    /// ```
//    fn index(&self, idx: usize) -> &Self::Output {
//        self.data.index(idx)
//    }
//}

impl<'a, S, N> UniChunked<S, N>
where
    S: View<'a>,
    <S as View<'a>>::Type: ReinterpretSet<N>,
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
    ) -> <<<S as View<'a>>::Type as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter {
        self.data.view().reinterpret_set().into_iter()
    }
}

impl<'a, S, N> UniChunked<S, N>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: ReinterpretSet<N>,
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
    ) -> <<<S as ViewMut<'a>>::Type as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter {
        self.data.view_mut().reinterpret_set().into_iter()
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

macro_rules! impl_split_at_fn {
    ($self:ident, $split_fn:ident, $n:ty, $mid:expr) => {
        {
            let (l, r) = $self.data.$split_fn($mid * <$n>::value());
            (UniChunked::from_flat(l), UniChunked::from_flat(r))
        }
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
        impl_split_at_fn!(self, split_at, N, mid)
    }
}

impl<T, N: num::Unsigned> SplitAt for UniChunked<&[T], N> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at, N, mid)
    }
}

impl<T, N: num::Unsigned> SplitAt for UniChunked<&mut [T], N> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at_mut, N, mid)
    }
}


impl<S: Dummy, N: num::Unsigned> Dummy for UniChunked<S, N> {
    fn dummy() -> Self {
        UniChunked { data: Dummy::dummy(), chunks: N::new() }
    }
}

impl<S: IntoFlat, N> IntoFlat for UniChunked<S, N> {
    type FlatType = <S as IntoFlat>::FlatType;
    /// Strip away the uniform organization of the underlying data, and return the underlying data.
    fn into_flat(self) -> Self::FlatType {
        self.data.into_flat()
    }
}
