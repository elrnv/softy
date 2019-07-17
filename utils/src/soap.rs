mod subset;
mod uniset;
mod varset;
mod view;

pub use subset::*;
pub use uniset::*;
pub use varset::*;
pub use view::*;

// Helper module defines a few useful unsigned type level integers.
// This is to avoid having to depend on yet another crate.
pub mod num {
    pub trait Unsigned {
        fn value() -> usize;
    }

    macro_rules! def_num {
        ($(($nty:ident, $n:expr)),*) => {
            $(
                #[derive(Debug, Clone, PartialEq)]
                pub struct $nty;
                impl Unsigned for $nty {
                    fn value() -> usize {
                        $n
                    }
                }
             )*
        }
    }

    def_num!((U1, 1), (U2, 2), (U3, 3));
}

/// A trait defining a raw buffer of data. This data is typed but not annotated so it can represent
/// anything. For example a buffer of floats can represent a set of vertex colours or vertex
/// positions.
pub trait Set {
    /// Owned element of the set.
    type Elem;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait SetView<'a>: Set {
    type ElemRef;
    type ElemMut;
}

/// A helper trait analogous to `SliceIndex` from the standard library.
pub trait GetIndex<T> {
    type Output: ?Sized;
    fn get(self, set: &T) -> Option<&Self::Output>;
    fn get_mut(self, set: &mut T) -> Option<&mut Self::Output>;
}

/// Blanket implementation of `GetIndex` for all std index types over slices.
impl<I, S> GetIndex<S> for I
where
    I: std::slice::SliceIndex<S>,
    S: std::ops::Index<I> + std::ops::IndexMut<I>,
{
    type Output = <S as std::ops::Index<I>>::Output;
    fn get(self, set: &S) -> Option<&Self::Output> {
        Some(set.index(self))
    }
    fn get_mut(self, set: &mut S) -> Option<&mut Self::Output> {
        Some(set.index_mut(self))
    }
}
//
//impl<S, N> GetIndex<UniSet<S, N>> for usize
//where
//    S: Set + ReinterpretSet<N>,
//{
//    type Output = <<S as Set>::Elem as Grouped<N>>::Type;
//    fn get(self, set: &S) -> Option<&Self::Output> {
//        Some()
//    }
//    fn get_mut(self, set: &mut S) -> Option<&mut Self::Output> {
//        Some()
//    }
//}

/// An index trait for `Set` types.
pub trait Get<'a, I> {
    type Output;
    fn get(&'a self, idx: I) -> Self::Output;
}

impl<'a, S, I> Get<'a, I> for &'a S
where
    S: std::ops::Index<I> + ?Sized,
    <S as std::ops::Index<I>>::Output: 'a,
    I: std::slice::SliceIndex<S>,
{
    type Output = &'a <S as std::ops::Index<I>>::Output;
    fn get(&'a self, idx: I) -> Self::Output {
        self.index(idx)
    }
}

impl<'a, S, I> Get<'a, I> for &'a mut S
where
    S: std::ops::Index<I> + ?Sized,
    <S as std::ops::Index<I>>::Output: 'a,
    I: std::slice::SliceIndex<S>,
{
    type Output = &'a <S as std::ops::Index<I>>::Output;
    fn get(&'a self, idx: I) -> Self::Output {
        self.index(idx)
    }
}

impl<T: Clone> Get<'_, usize> for Vec<T> {
    type Output = T;
    fn get(&self, idx: usize) -> Self::Output {
        self[idx].clone()
    }
}

impl<T> Set for Vec<T> {
    type Elem = T;
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> Set for [T] {
    type Elem = T;
    fn len(&self) -> usize {
        self.len()
    }
}

impl<'a, S: Set + ?Sized> Set for &'a S {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}
impl<'a, S: Set + ?Sized> Set for &'a mut S {
    type Elem = <S as Set>::Elem;
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

/// The element of a set is a view into the set of size one.
impl<'a, S: Set + View<'a> + ViewMut<'a>> SetView<'a> for S {
    type ElemRef = <S as View<'a>>::Type;
    type ElemMut = <S as ViewMut<'a>>::Type;
}

/// Abstraction for pushing elements of type `T` onto a collection.
pub trait Push<T> {
    fn push(&mut self, element: T);
}

impl<T> Push<T> for Vec<T> {
    fn push(&mut self, element: T) {
        Vec::push(self, element);
    }
}
