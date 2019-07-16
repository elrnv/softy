mod uniset;
mod varset;
mod view;
mod subset;

pub use uniset::*;
pub use varset::*;
pub use view::*;
pub use subset::*;

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
    type Elem;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait GetIndex<T> {
    type Output: ?Sized;
    fn get(self, set: &T) -> Option<&Self::Output>;
    fn get_mut(self, set: &mut T) -> Option<&mut Self::Output>;
}

pub trait Get<'a, I> {
    type Output;
    fn get(&'a self, idx: I) -> Self::Output;
}

impl<'a, S, I> Get<'a, I> for &'a S
    where S: std::ops::Index<I> + ?Sized,
          <S as std::ops::Index<I>>::Output: 'a,
          I: std::slice::SliceIndex<S>,
{
    type Output = &'a <S as std::ops::Index<I>>::Output;
    fn get(&'a self, idx: I) -> Self::Output {
        self.index(idx)
    }
}

impl<'a, S, I> Get<'a, I> for &'a mut S
    where S: std::ops::Index<I> + ?Sized,
          <S as std::ops::Index<I>>::Output: 'a,
          I: std::slice::SliceIndex<S>,
{
    type Output = &'a <S as std::ops::Index<I>>::Output;
    fn get(&'a self, idx: I) -> Self::Output {
        self.index(idx)
    }
}

impl<T> Set for Vec<T> {
    type Elem = T;
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Clone> Get<'_, usize> for Vec<T> {
    type Output = T;
    fn get(&self, idx: usize) -> Self::Output {
        self[idx].clone()
    }
}

impl<'a, T> Set for &'a Vec<T> {
    type Elem = &'a T;
    fn len(&self) -> usize {
        Vec::<T>::len(self)
    }
}

impl<'a, T> Set for &'a mut Vec<T> {
    type Elem = &'a mut T;
    fn len(&self) -> usize {
        Vec::<T>::len(self)
    }
}

impl<'a, T> Set for &'a [T] {
    type Elem = &'a T;
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<'a, T> Set for &'a mut [T] {
    type Elem = &'a mut T;
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

// reference into a set.
//pub struct RefIter<'a, I> {
//    s: Ref<'a, I>,
//}
/*

impl<S: Set> Set for Rc<RefCell<S>> {
    fn len(&self) -> usize { self.borrow().len() }
}

impl<S: Set> VarIter for Rc<RefCell<S>> {
    type Iter = std::cell::Ref<S::Iter>;
    type IterMut = std::cell::RefMut<S::IterMut>;
    fn iter(&self) -> Self::Iter {
        std::cell::Ref::map(self.borrow(), |x| x.iter())
    }
    fn iter_mut(&self) -> Self::IterMut {
        std::cell::RefMut::map(self.borrow_mut(), |x| x.iter_mut())
    }
}

impl<S: Set> Set for Arc<RwLock<S>> {
    fn len(&self) -> usize { self.read().unwrap().len() }
}

impl<S: Set> VarIter for Arc<RwLock<S>> {
    type Iter = S::Iter;
    type IterMut = S::IterMut;
    fn iter(&self) -> Self::Iter {
        self.read().unwrap().iter()
    }
    fn iter_mut(&self) -> Self::IterMut {
        self.write().unwrap().iter_mut()
    }
}
*/
