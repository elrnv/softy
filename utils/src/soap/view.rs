use super::Chunked;

pub type ChunkedView<'a, S> = Chunked<S, &'a [usize]>;

/// A trait defining a collection that can be accessed via an
/// immutable (shared) view. This type of view can be cloned and copied.
pub trait View<'a> {
    type Type: Clone + Copy;

    fn view(&'a self) -> Self::Type;
}

/// A trait defining a collection that can be accessed via a mutable (unique)
/// view.
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
