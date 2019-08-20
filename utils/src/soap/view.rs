/// A marker trait to indicate a viewed collection type. Note that collections
/// can be partially viewed, but only completely viewed collections are marked
/// by `Viewed`.
pub trait Viewed {}

/// A trait defining a collection that can be accessed via an
/// immutable (shared) view. This type of view can be cloned and copied.
pub trait View<'a> {
    type Type: Viewed;

    fn view(&'a self) -> Self::Type;
}

/// A trait defining a collection that can be accessed via a mutable (unique)
/// view.
pub trait ViewMut<'a> {
    type Type: Viewed;

    fn view_mut(&'a mut self) -> Self::Type;
}

/// Blanket implementation of `View` for all immutable borrows.
impl<'a, S: ?Sized + 'a + View<'a>> View<'a> for &S {
    type Type = S::Type;

    fn view(&'a self) -> Self::Type {
        <S as View<'a>>::view(*self)
    }
}

/// Blanket implementation of `View` for all mutable borrows.
impl<'a, S: ?Sized + 'a + View<'a>> View<'a> for &mut S {
    type Type = S::Type;

    fn view(&'a self) -> Self::Type {
        <S as View<'a>>::view(*self)
    }
}

/// Blanket implementation of `ViewMut` for all mutable borrows.
impl<'a, S: ?Sized + 'a + ViewMut<'a>> ViewMut<'a> for &mut S {
    type Type = S::Type;

    fn view_mut(&'a mut self) -> Self::Type {
        <S as ViewMut<'a>>::view_mut(*self)
    }
}
