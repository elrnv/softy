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
                #[derive(Debug, Copy, Clone, PartialEq)]
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
        Vec::len(self)
    }
}

impl<T> Set for [T] {
    type Elem = T;
    fn len(&self) -> usize {
        <[T]>::len(self)
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

/// A helper trait to split a set into two sets at a given index.
/// This trait is used to implement iteration over `VarView`s.
pub trait SplitAt where Self: Sized {
    /// Split self into two sets at the given midpoint.
    /// This function is analogous to `<[T]>::split_at`.
    fn split_at(self, mid: usize) -> (Self, Self);
}

/// A helper trait to split owned sets into two sets at a given index.
/// This trait is used to implement iteration over `VarSet`s.
pub trait SplitOff {
    /// Split self into two sets at the given midpoint.
    /// This function is analogous to `Vec::split_off`.
    /// `self` contains elements `[0, mid)`, and The returned `Self` contains
    /// elements `[mid, len)`.
    fn split_off(&mut self, mid: usize) -> Self;
}


/// Convert a collection into its underlying representation, effectively
/// stripping any organizational info.
pub trait IntoFlat {
    type FlatType;
    fn into_flat(self) -> Self::FlatType;
}

impl<T> IntoFlat for Vec<T> {
    type FlatType = Vec<T>;
    /// Since a `Vec` has no information about the structure of its underlying
    /// data, this is effectively a no-op.
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<'a, T> IntoFlat for &'a [T] {
    type FlatType = &'a [T];
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<'a, T> IntoFlat for &'a mut [T] {
    type FlatType = &'a mut [T];
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<T> SplitAt for Vec<T> {
    fn split_at(mut self, mid: usize) -> (Self, Self) {
        let r = self.split_off(mid);
        (self, r)
    }
}

/// A helper trait for constructing placeholder sets for use in `std::mem::replace`.
/// These don't necessarily have to correspond to bona-fide sets.
pub trait Dummy {
    fn dummy() -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test iteration of a `UniSet` inside a `VarSet`.
    #[test]
    fn var_of_uni_iter_test() {
        let u0 = UniSet::<_, num::U2>::from_flat((1..=12).collect::<Vec<_>>());
        let v1 = VarSet::from_offsets(vec![0,2,3,6], u0);

        let mut iter1 = v1.iter();
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[1,2]), iter0.next());
        assert_eq!(Some(&[3,4]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[5,6]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[7,8]), iter0.next());
        assert_eq!(Some(&[9,10]), iter0.next());
        assert_eq!(Some(&[11,12]), iter0.next());
        assert_eq!(None, iter0.next());
    }
}
