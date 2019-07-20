use super::*;

/*
 * Strict subview types corresponding to each of the set types.
 */

/// A `VarSet` that is a contiguous sub-view of some larger set (which could have
/// any type).
pub type VarView<'a, V> = VarSet<V, &'a [usize]>;

/// A contiguous `UniSet` view of some larger set (which could have any type).
pub type UniView<V, N> = UniSet<V, N>;

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
