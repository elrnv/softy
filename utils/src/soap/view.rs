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

pub trait View<'a> {
    type Type: Clone;

    fn view(&'a self) -> Self::Type;
}

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

impl<'a, S, N> View<'a> for UniSet<S, N>
    where S: Set + View<'a>,
          N: num::Unsigned,
          <S as View<'a>>::Type: Set,
{
    type Type = UniSetView<'a, S, N>;

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
