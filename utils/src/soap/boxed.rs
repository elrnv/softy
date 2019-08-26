//!
//! Implementations for boxed collections.
//!

use super::*;

impl<S: Set> Set for Box<S> {
    type Elem = S::Elem;
    fn len(&self) -> usize {
        S::len(self)
    }
}
impl<'a, S: View<'a>> View<'a> for Box<S> {
    type Type = <S as View<'a>>::Type;
    fn view(&'a self) -> Self::Type {
        S::view(self)
    }
}
impl<'a, S: ViewMut<'a>> ViewMut<'a> for Box<S> {
    type Type = <S as ViewMut<'a>>::Type;
    fn view_mut(&'a mut self) -> Self::Type {
        S::view_mut(self)
    }
}
impl<S: Dummy> Dummy for Box<S> {
    unsafe fn dummy() -> Self {
        Box::new(Dummy::dummy())
    }
}
