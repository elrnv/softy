use super::*;

/*
 * Strict subview types corresponding to each of the set types.
 */

/// A VarSet that is a contiguous sub-view of some larger set (which could have
/// any type).
pub type VarSetView<'a, V> = VarSet<<V as View<'a>>::Type, &'a [usize]>;

/// A VarSet that is a contiguous mutable sub-view of some larger set (which
/// could have any type).
pub type VarSetViewMut<'a, V> = VarSet<<V as ViewMut<'a>>::Type, &'a [usize]>;

/// A contiguous immutable UniSet view of some larger set (which could have any type).
pub type UniSetView<'a, V, N> = UniSet<<V as View<'a>>::Type, N>;

/// A contiguous mutable UniSet view of some larger set (which could have any type).
pub type UniSetViewMut<'a, V, N> = UniSet<<V as ViewMut<'a>>::Type, N>;

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

impl<'a, T: 'a> View<'a> for [T] {
    type Type = &'a [T];

    fn view(&'a self) -> Self::Type {
        self
    }
}
impl<'a, T: 'a> ViewMut<'a> for [T] {
    type Type = &'a mut [T];

    fn view_mut(&'a mut self) -> Self::Type {
        self
    }
}

/// Blanket implementation of `View` for all immutable borrows.
impl<'a, 'b: 'a, S: ?Sized + 'b> View<'a> for &'b S {
    type Type = &'a S;

    fn view(&'a self) -> Self::Type {
        *self
    }
}

/// Blanket implementation of `View` for all mutable borrows.
impl<'a, 'b: 'a, S: ?Sized + 'b> View<'a> for &'b mut S {
    type Type = &'a S;

    fn view(&'a self) -> Self::Type {
        *self
    }
}

/// Blanket implementation of `ViewMut` for all mutable borrows.
impl<'a, 'b: 'a, S: ?Sized + 'b> ViewMut<'a> for &'b mut S {
    type Type = &'a mut S;

    fn view_mut(&'a mut self) -> Self::Type {
        *self
    }
}

impl<'a, S, N> View<'a> for UniSet<S, N>
where
    S: Set + View<'a>,
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
where
    S: Set + ViewMut<'a>,
    N: num::Unsigned,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = UniSetViewMut<'a, S, N>;

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

//impl<'a, S, N> View<'a> for VarSet<S, N>
//where
//    S: Set + View<'a>,
//    N: num::Unsigned,
//    <S as View<'a>>::Type: Set,
//{
//    type Type = UniSetView<'a, S, N>;
//
//    /// Create a contiguous immutable (shareable) view into this set.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let s = UniSet::<_, num::U2>::from_flat(vec![0,1,2,3]);
//    /// let v1 = s.view(); // s is now inaccessible.
//    /// let v2 = v1.clone();
//    /// let mut view1_iter = v1.iter();
//    /// assert_eq!(Some(&[0,1]), view1_iter.next());
//    /// assert_eq!(Some(&[2,3]), view1_iter.next());
//    /// assert_eq!(None, view1_iter.next());
//    /// for (a,b) in v1.iter().zip(v2.iter()) {
//    ///     assert_eq!(a,b);
//    /// }
//    /// ```
//    fn view(&'a self) -> Self::Type {
//        UniSet::from_flat(self.data.view())
//    }
//}
