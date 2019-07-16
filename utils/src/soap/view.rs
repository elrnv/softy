use super::*;

/*
 * Strict subview types corresponding to each of the set types.
 */

/// A VarSet that is a contiguous sub-view of some larger set (which could have
/// any type).
#[derive(Clone)]
pub struct VarSetView<'a, V> where V: View<'a> {
    offset: &'a [usize],
    data: <V as View<'a>>::Type,
}

/// A VarSet that is a contiguous mutable sub-view of some larger set (which
/// could have any type).
pub struct VarSetViewMut<'a, V> where V: ViewMut<'a> {
    offset: &'a [usize],
    data: <V as ViewMut<'a>>::Type,
}

/// A contiguous immutable UniSet view of some larger set (which could have any type).
pub type UniSetView<'a, R, N> = UniSet<<R as View<'a>>::Type, N>;

/// A contiguous mutable UniSet view of some larger set (which could have any type).
pub type UniSetViewMut<'a, R, N> = UniSet<<R as ViewMut<'a>>::Type, N>;

/// A trait defining a set that can be accessed via a contiguous immutable
/// (shared) view.  This type of view can be cloned.
pub trait View<'a> {
    type Type: Clone;

    fn view(&'a self) -> Self::Type;
}

/// A trait defining a set that can be accessed via a contiguous mutable
/// (unique) view.
pub trait ViewMut<'a> {
    type Type;

    fn view_mut(&'a mut self) -> Self::Type;
}

impl<'a, T: 'a> View<'a> for Vec<T> {
    type Type = &'a [T];

    fn view(&'a self) -> Self::Type {
        self.as_slice()
    }
}
impl<'a, T: 'a> ViewMut<'a> for Vec<T> {
    type Type = &'a mut [T];

    fn view_mut(&'a mut self) -> Self::Type {
        self.as_mut_slice()
    }
}

/// Blanket implementation for all immutable borrows.
impl<'a, S: ?Sized + 'a> View<'a> for &'a S {
    type Type = &'a S;

    fn view(&'a self) -> Self::Type {
        *self
    }
}

/// Blanket implementation for all mutable borrows.
impl<'a, S: ?Sized + 'a> ViewMut<'a> for &'a mut S {
    type Type = &'a mut S;

    fn view_mut(&'a mut self) -> Self::Type {
        *self
    }
}

impl<'a, S, N> View<'a> for UniSet<S, N>
    where S: Set + View<'a>,
          N: num::Unsigned,
          <S as View<'a>>::Type: Set,
{
    type Type = UniSetView<'a, S, N>;

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
    /// for (a,b) in v1.iter().zip(v2.iter()) {
    ///     assert_eq!(a,b);
    /// }
    /// ```
    fn view(&'a self) -> Self::Type {
        UniSet::from_flat(self.data.view())
    }
}
impl<'a, S, N> ViewMut<'a> for UniSet<S, N>
    where S: Set + ViewMut<'a>,
          N: num::Unsigned,
          <S as ViewMut<'a>>::Type: Set,
{
    type Type = UniSetViewMut<'a, S, N>;

    fn view_mut(&'a mut self) -> Self::Type {
        UniSet::from_flat(self.data.view_mut())
    }
}

impl<'a, S, N> UniSet<S, N>
where
    S: View<'a>,
    <S as View<'a>>::Type: ReinterpretSet<N>,
    N: num::Unsigned,
{
    /// Produce an iterator over borrowed grouped elements of the `UniSet`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniSet::<_, num::U2>::from_flat(vec![1,2,3,4]);
    /// let mut uniset_iter = s.iter();
    /// assert_eq!(Some(&[1,2]), uniset_iter.next());
    /// assert_eq!(Some(&[3,4]), uniset_iter.next());
    /// assert_eq!(None, uniset_iter.next());
    /// ```
    pub fn iter(&'a self) -> <<<S as View<'a>>::Type as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter {
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
    /// # Example
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
    pub fn iter_mut(
        &'a mut self,
    ) -> <<<S as ViewMut<'a>>::Type as ReinterpretSet<N>>::Output as IntoIterator>::IntoIter {
        self.data.view_mut().reinterpret_set().into_iter()
    }
}
