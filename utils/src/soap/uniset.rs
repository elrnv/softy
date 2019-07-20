use super::*;
use std::marker::PhantomData;

/*
 * Uniformly spaced Set
 */

/// `UniSet` Assigns a uniform stride to the specified buffer.
///
/// # Example
///
/// ```rust
/// use utils::soap::*;
/// let s = UniSet::<_, num::U2>::from_flat(vec![1,2,3,4,5,6]);
/// let mut uniset_iter = s.iter();
/// assert_eq!(Some(&[1,2]), uniset_iter.next());
/// assert_eq!(Some(&[3,4]), uniset_iter.next());
/// assert_eq!(Some(&[5,6]), uniset_iter.next());
/// assert_eq!(None, uniset_iter.next());
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct UniSet<S, N> {
    pub data: S,
    phantom: PhantomData<N>,
}

impl<S: Set, N: num::Unsigned> UniSet<S, N> {
    /// Create a `UniSet` from another set, which effectively groups the
    /// elements of the original set into uniformly sized groups.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6]);
    /// let mut uniset_iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), uniset_iter.next());
    /// assert_eq!(Some(&[4,5,6]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    pub fn from_flat(data: S) -> Self {
        assert_eq!(data.len() % N::value(), 0);
        UniSet {
            data,
            phantom: PhantomData,
        }
    }
}

/// An implementation of `Set` for `UniSet` of any type that can be grouped as `N` sub-elements.
impl<S: Set, N: num::Unsigned> Set for UniSet<S, N>
where
    S::Elem: Grouped<N>,
{
    type Elem = <S::Elem as Grouped<N>>::Type;
    /// Compute the length of this set as the number of grouped elements in the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniSet::<_, num::U2>::from_flat(vec![0,1,2,3,4,5]);
    /// assert_eq!(s.len(), 3);
    /// let s = UniSet::<_, num::U3>::from_flat(vec![0,1,2,3,4,5]);
    /// assert_eq!(s.len(), 2);
    /// ```
    fn len(&self) -> usize {
        self.data.len() / N::value()
    }
}

impl<S, N> Push<<<S as Set>::Elem as Grouped<N>>::Type> for UniSet<S, N>
where
    S: Set + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Grouped<N>,
{
    /// Push a grouped element onto the `UniSet`. Each element must have exactly
    /// `N` sub-elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniSet::<_, num::U3>::from_flat(vec![1,2,3]);
    /// s.push([4,5,6]);
    /// let mut uniset_iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), uniset_iter.next());
    /// assert_eq!(Some(&[4,5,6]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    fn push(&mut self, element: <<S as Set>::Elem as Grouped<N>>::Type) {
        Grouped::<N>::push_to(element, &mut self.data);
    }
}

impl<S, N> IntoIterator for UniSet<S, N>
where
    S: Set + ReinterpretSet<N>,
    N: num::Unsigned,
{
    type Item = <<S as ReinterpretSet<N>>::Output as IntoIterator>::Item;
    type IntoIter = <<S as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter;

    /// Convert a `UniSet` into an iterator over grouped elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6]);
    /// let mut uniset_iter = s.into_iter();
    /// assert_eq!(Some([1,2,3]), uniset_iter.next());
    /// assert_eq!(Some([4,5,6]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.data.reinterpret_set().into_iter()
    }
}

impl<S, N> std::iter::FromIterator<<<S as Set>::Elem as Grouped<N>>::Type> for UniSet<S, N>
where
    N: num::Unsigned,
    S: Set + Default + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Grouped<N>,
{
    /// Construct a `UniSet` from an iterator that produces grouped elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![[1,2,3],[4,5,6]];
    /// let s: UniSet::<Vec<usize>, num::U3> = v.into_iter().collect();
    /// let mut uniset_iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), uniset_iter.next());
    /// assert_eq!(Some(&[4,5,6]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = <<S as Set>::Elem as Grouped<N>>::Type>,
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

pub trait ReinterpretSet<N> {
    type Output: IntoIterator;
    fn reinterpret_set(self) -> Self::Output;
}

impl<'a, S, N: num::Unsigned, M: num::Unsigned> ReinterpretSet<N> for UniSet<S, M>
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


//impl<'a, S, N> Get<'a, usize> for UniSet<S, N>
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

impl<T> std::borrow::Borrow<[T]> for UniSet<&[T], num::U1> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

impl<T> std::borrow::Borrow<[T]> for UniSet<&mut [T], num::U1> {
    fn borrow(&self) -> &[T] {
        self.data
    }
}

impl<T> std::borrow::BorrowMut<[T]> for UniSet<&mut [T], num::U1> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.data
    }
}

macro_rules! impl_borrow_uniset {
    ($nty:ty, $n:expr) => {
        impl<T> std::borrow::Borrow<[[T; $n]]> for UniSet<&[T], $nty> {
            fn borrow(&self) -> &[[T; $n]] {
                reinterpret::reinterpret_slice(self.data)
            }
        }
        impl<T> std::borrow::Borrow<[[T; $n]]> for UniSet<&mut [T], $nty> {
            fn borrow(&self) -> &[[T; $n]] {
                reinterpret::reinterpret_slice(self.data)
            }
        }
        impl<T> std::borrow::BorrowMut<[[T; $n]]> for UniSet<&mut [T], $nty> {
            fn borrow_mut(&mut self) -> &mut [[T; $n]] {
                reinterpret::reinterpret_mut_slice(self.data)
            }
        }
    };
}

impl_borrow_uniset!(num::U2, 2);
impl_borrow_uniset!(num::U3, 3);

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

//impl<S: std::ops::Index<usize>, N> std::ops::Index<usize> for UniSet<S, N> {
//    type Output = <UniSet<S, N> as Set>::Elem;
//    /// Immutably index the subset.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let s = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
//    /// assert_eq!([7,8,9], s[2]);
//    /// ```
//    fn index(&self, idx: usize) -> &Self::Output {
//        self.data.index(idx)
//    }
//}

impl<'a, S, N> UniSet<S, N>
where
    S: View<'a>,
    <S as View<'a>>::Type: ReinterpretSet<N>,
    N: num::Unsigned,
{
    /// Produce an iterator over borrowed grouped elements of the `UniSet`.
    ///
    /// # Examples
    ///
    /// The following is a simple test for iterating over a uniformly organized `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniSet::<_, num::U2>::from_flat(vec![1,2,3,4]);
    /// let mut uniset_iter = s.iter();
    /// assert_eq!(Some(&[1,2]), uniset_iter.next());
    /// assert_eq!(Some(&[3,4]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    ///
    /// A more complex example consists of data organized as a nested `UniSet`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s0 = UniSet::<_, num::U2>::from_flat(vec![1,2, 3,4, 5,6, 7,8, 9,10, 11,12]);
    /// let s1 = UniSet::<_, num::U3>::from_flat(s0);
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

impl<'a, S, N> UniSet<S, N>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: ReinterpretSet<N>,
    N: num::Unsigned,
{
    /// Produce an iterator over mutably borrowed grouped elements of the `UniSet`.
    ///
    /// # Examples
    ///
    /// The following example shows a simple modification of a uniformly organized `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniSet::<_, num::U2>::from_flat(vec![0,1,2,3]);
    /// for i in s.iter_mut() {
    ///     i[0] += 1;
    ///     i[1] += 1;
    /// }
    /// let mut uniset_iter = s.iter();
    /// assert_eq!(Some(&[1,2]), uniset_iter.next());
    /// assert_eq!(Some(&[3,4]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    ///
    /// Nested `UniSet`s can also be modified as follows:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s0 = UniSet::<_, num::U2>::from_flat(vec![1,2, 3,4, 5,6, 7,8, 9,10, 11,12]);
    /// let mut s1 = UniSet::<_, num::U3>::from_flat(s0);
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


impl<'a, S, N> View<'a> for UniSet<S, N>
where
    S: Set + View<'a>,
    N: num::Unsigned + Copy,
    <S as View<'a>>::Type: Set,
{
    type Type = UniView<<S as View<'a>>::Type, N>;

    /// Create a contiguous immutable (shareable) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniSet::<_, num::U2>::from_flat(vec![0,1,2,3]);
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
        UniSet::from_flat(self.data.view())
    }
}

impl<'a, S, N> ViewMut<'a> for UniSet<S, N>
where
    S: Set + ViewMut<'a>,
    N: num::Unsigned,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = UniView<<S as ViewMut<'a>>::Type, N>;

    /// Create a contiguous mutable (unique) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniSet::<_, num::U2>::from_flat(vec![0,1,2,3]);
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
        UniSet::from_flat(self.data.view_mut())
    }
}

macro_rules! impl_split_at_fn {
    ($self:ident, $split_fn:ident, $n:ty, $mid:expr) => {
        {
            let (l, r) = $self.data.$split_fn($mid * <$n>::value());
            (UniSet::from_flat(l), UniSet::from_flat(r))
        }
    }
}


impl<S: SplitAt + Set, N: num::Unsigned> SplitAt for UniSet<S, N> {
    /// Split the current set into two distinct sets at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniSet::<_, num::U2>::from_flat(vec![0,1,2,3]);
    /// let (l, r) = s.split_at(1);
    /// assert_eq!(l, UniSet::<_, num::U2>::from_flat(vec![0,1]));
    /// assert_eq!(r, UniSet::<_, num::U2>::from_flat(vec![2,3]));
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at, N, mid)
    }
}

impl<T, N: num::Unsigned> SplitAt for UniSet<&[T], N> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at, N, mid)
    }
}

impl<T, N: num::Unsigned> SplitAt for UniSet<&mut [T], N> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        impl_split_at_fn!(self, split_at_mut, N, mid)
    }
}


impl<S: Dummy, N> Dummy for UniSet<S, N> {
    fn dummy() -> Self {
        UniSet { data: Dummy::dummy(), phantom: PhantomData }
    }
}
