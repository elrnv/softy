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

pub trait IntoView {
    type View;
    fn into_view(self) -> Self::View;
}

impl<'a, V: View<'a>> IntoView for &'a V {
    type View = V::Type;
    fn into_view(self) -> Self::View {
        self.view()
    }
}

impl<'a, V: ViewMut<'a>> IntoView for &'a mut V {
    type View = V::Type;
    fn into_view(self) -> Self::View {
        self.view_mut()
    }
}

/// A convenience trait to allow generic implementations to call an iterator over the view. This
/// is necessary because the `View` trait has an explicit lifetime parameter, which makes it
/// difficult or impossible to use in generic functions.
pub trait ViewIterator<'a> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;

    fn view_iter(&'a self) -> Self::Iter;
}

pub trait ViewMutIterator<'a> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;

    fn view_mut_iter(&'a mut self) -> Self::Iter;
}
