use std::marker::PhantomData;
use super::*;

/*
 * Strict subview types corresponding to each of the set types.
 */

/// A VarSet that is a contiguous sub-view of some larger set (which could have
/// any type).
pub struct VarSetView<'a, V> where V: View<'a> {
    pub(crate) view: VarSet<<V as View<'a>>::Type, &'a [usize]>,
    pub(crate) phantom: PhantomData<&'a V>,
}

impl<'a, V> std::fmt::Debug for VarSetView<'a, V>
where
    V: View<'a>,
    <V as View<'a>>::Type: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "VarSetView {{ view: {:?}, .. }}", self.view)
    }
}

impl<'a, V> Clone for VarSetView<'a, V> where V: View<'a> {
    fn clone(&self) -> Self {
        VarSetView {
            view: self.view.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'a, V> Copy for VarSetView<'a, V> where V: View<'a> {}

impl<'a, V> Set for VarSetView<'a, V>
where V: View<'a>,
      <V as View<'a>>::Type: Set
{
    type Elem = <<V as View<'a>>::Type as Set>::Elem;
    fn len(&self) -> usize {
        self.view.len()
    }

}

impl<'a, V> VarSetView<'a, V>
where V: View<'a>,
      <V as View<'a>>::Type: Set,
{
    pub fn from_offsets(offsets: &'a [usize], data: <V as View<'a>>::Type) -> Self {
        VarSetView {
            view: VarSet::from_offsets(offsets, data),
            phantom: PhantomData,
        }
    }
}



/// A VarSet that is a contiguous mutable sub-view of some larger set (which
/// could have any type).
pub struct VarSetViewMut<'a, V> where V: ViewMut<'a> {
    pub(crate) view: VarSet<<V as ViewMut<'a>>::Type, &'a [usize]>,
    pub(crate) phantom: PhantomData<&'a V>,
}

//impl<'a, V: ViewMut<'a>> std::ops::DerefMut for VarSetViewMut<'a, V> {
//    fn deref_mut(&mut self) -> &mut Self::Target {
//        &mut self.view
//    }
//}
//impl<'a, V: View<'a>> std::ops::Deref for VarSetView<'a, V> {
//    type Target = VarSet<<V as View<'a>>::Type, &'a [usize]>;
//
//    fn deref(&self) -> &Self::Target {
//        &self.view
//    }
//}

impl<'a, V: View<'a> + ViewMut<'a>> std::ops::Deref for VarSetViewMut<'a, V> {
    type Target = VarSetView<'a, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl<'a, V: View<'a> + ViewMut<'a>> std::ops::DerefMut for VarSetViewMut<'a, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl<'a, V: ViewMut<'a> + 'a> std::ops::Deref for VarSet<V, &'a [usize]> {
    type Target = VarSetViewMut<'a, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl<'a, V: ViewMut<'a> + 'a> std::ops::DerefMut for VarSet<V, &'a [usize]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self as *mut Self::Target) }
    }
}

impl<'a, V> std::fmt::Debug for VarSetViewMut<'a, V>
where
    V: ViewMut<'a>,
    <V as ViewMut<'a>>::Type: std::fmt::Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "VarSetViewMut {{ view: {:?}, .. }}", self.view)
    }
}

impl<'a, V> Set for VarSetViewMut<'a, V>
where V: ViewMut<'a>,
      <V as ViewMut<'a>>::Type: Set
{
    type Elem = <<V as ViewMut<'a>>::Type as Set>::Elem;
    fn len(&self) -> usize {
        self.view.len()
    }
}

impl<'a, V> VarSetViewMut<'a, V>
where V: ViewMut<'a>,
      <V as ViewMut<'a>>::Type: Set,
{
    pub fn from_offsets(offsets: &'a [usize], data: <V as ViewMut<'a>>::Type) -> Self {
        VarSetViewMut {
            view: VarSet::from_offsets(offsets, data),
            phantom: PhantomData,
        }
    }
}

/// A contiguous immutable UniSet view of some larger set (which could have any type).
pub type UniSetView<'a, V, N> = UniSet<<V as View<'a>>::Type, N>;

/// A contiguous mutable UniSet view of some larger set (which could have any type).
pub type UniSetViewMut<'a, V, N> = UniSet<<V as ViewMut<'a>>::Type, N>;

/// A trait defining a set that can be accessed via a contiguous immutable
/// (shared) view.  This type of view can be cloned.
pub trait View<'a> {
    type Type: Clone + Copy;

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
impl<'a, S: ?Sized + 'a> View<'a> for &S {
    type Type = &'a S;

    fn view(&'a self) -> Self::Type {
        *self
    }
}

/// Blanket implementation of `View` for all mutable borrows.
impl<'a, S: ?Sized + 'a> View<'a> for &mut S {
    type Type = &'a S;

    fn view(&'a self) -> Self::Type {
        *self
    }
}

/// Blanket implementation of `ViewMut` for all mutable borrows.
impl<'a, S: ?Sized + 'a> ViewMut<'a> for &mut S {
    type Type = &'a mut S;

    fn view_mut(&'a mut self) -> Self::Type {
        *self
    }
}

impl<'a, S, N> View<'a> for UniSet<S, N>
where
    S: Set + View<'a>,
    N: num::Unsigned + Copy,
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

