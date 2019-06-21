use std::marker::PhantomData;
use std::rc::Rc;
use cell::{Ref, RefMut, RefCell};
use std::sync::{Arc, RwLock};

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

    def_num!(
        (U1, 1),
        (U2, 2),
        (U3, 3)
    );
}

/// A trait defining a raw buffer of data. This data is typed but not annotated so it can represent
/// anything. For example a buffer of floats can represent a set of vertex colours or vertex
/// positions.
pub trait Set: Clone + IntoIterator {
    //type Iter;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

// Reference into a set.
//pub struct RefIter<'a, I> {
//    s: Ref<'a, I>,
//}
//
//impl<'a, 'b: 'a, I: StaticIter + 'a> IntoIterator for &'b RefSet<'a, I> {
//    type Item = &'a <Self as Set>::Element;
//    type IntoIter = I::Iter;
//
//    fn into_iter(self) -> Self::IntoIter {
//        self.0.iter()
//    }
//}
/*
/// Statically checked borrowing iterator.
pub trait StaticIter {
    type Iter;
    type IterMut;
    fn iter(&self) -> Self::Iter;
    fn iter_mut(&mut self) -> Self::IterMut;
}

/// Dynamically checked borrowing iterator.
pub trait DynamicIter {
    type Iter;
    type IterMut;
    fn iter(&self) -> Self::Iter;
    fn iter_mut(&self) -> Self::IterMut;
}

impl<T: Clone> Set for Vec<T> {
    fn len(&self) -> usize { self.len() }
}

impl<T> StaticIter<'a> for Vec<T> {
    type Iter = std::slice::Iter<T>;
    type IterMut = std::slice::IterMut<T>;
    fn iter(&self) -> Self::Iter {
        self.iter()
    }
    fn iter_mut(&mut self) -> Self::IterMut {
        self.iter_mut()
    }
}

impl<S: Set> Set for Rc<RefCell<S>> {
    fn len(&self) -> usize { self.borrow().len() }
}

impl<S: Set> DynamicIter for Rc<RefCell<S>> {
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

impl<S: Set> DynamicIter for Arc<RwLock<S>> {
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

/// A set of variable length elements. Each offset represents one element and gives the offset into
/// the data buffer for the first of subelement in the Set.
/// Offsets always begins with a 0 and ends with the length of the buffer.
#[derive(Clone, Debug)]
pub struct DynamicSet<S> {
    pub offsets: Vec<usize>,
    pub data: S,
}

impl<S: Set> DynamicSet<S> {
    pub fn new(offsets: Vec<usize>, data: S) -> Self {
        assert!(offsets.len() > 0);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap(), data.len());
        DynamicSet { offsets, data }
    }

    pub fn data(&self) -> &S {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}


/// Assigns a uniform stride to the specified buffer.
#[derive(Clone, Debug)]
pub struct UniformSet<S: Set, N> {
    pub data: S,
    phantom: PhantomData<N>,
}

impl<S: Set, N: num::Unsigned> UniformSet<S, N> {
    pub fn new(data: S) -> Self {
        assert_eq!(data.len() % N::value(), 0);
        UniformSet { data, phantom: PhantomData }
    }

    pub fn data(&self) -> &S {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len() % N::value()
    }

    //pub fn iter(&self) -> S::Iter {
    //    self.data.len() % N::value()
    //}
}

//impl<S: Set> UniformSet<S, num::U1> {
//    pub fn iter(&self) -> S::Iter<> {
//        self.data.len() % N::value()
//    }
//}

/*
 * Strict subset types corresponding to each of the set types.
 */

#[derive(Clone, Debug)]
pub struct DynamicSubset<'a, T: 'static> {
    pub offset: usize,
    pub data: &'a [T],
}

#[derive(Clone, Debug)]
pub struct UniformSubset<'a, T: 'static> {
    pub data: &'a [T],
}

mod tests {
    use super::*;

    /// A displacement node.
    #[derive(Debug)]
    struct Displacement([f64; 3]);

    /// A vertex node.
    struct Vertex {
        //pos: [f64; 3],
        vel: [f64; 3],
    }

    struct VertexRef<'a> {
        //pos: [f64; 3],
        vel: &'a [f64; 3],
    }

    struct VertexMut<'a> {
        //pos: [f64; 3],
        vel: &'a mut [f64; 3],
    }

    #[derive(Debug)]
    struct VertexSet {
        //pos: UniformSet<Vec<f64>, num::U3>,
        vel: Vec<[f64;3]>,
        //data_rc: Rc<RefCell<Vec<[f64;3]>>>,
        //data_arc: Arc<RwLock<Vec<[f64;3]>>>,
    }

    impl VertexSet {
        fn iter(&self) -> VertexIter {
            VertexIter {
                //pos: self.pos.iter(),
                vel: self.vel.iter(),
            }
        }
        fn iter_mut(&mut self) -> VertexIterMut {
            VertexIterMut {
                //pos: self.pos.iter_mut(),
                vel: self.vel.iter_mut(),
            }
        }
    }

    // Should be automatically generated
    struct VertexIter<'a> {
        //pos: std::slice::Iter<[f64; 3]>,
        vel: std::slice::Iter<'a, [f64; 3]>,
    }
    struct VertexIterMut<'a> {
        //pos: std::slice::IterMut<[f64; 3]>,
        vel: std::slice::IterMut<'a, [f64; 3]>,
    }
    struct VertexIntoIter {
        //pos: std::vec::IntoIter<[f64; 3]>,
        vel: std::vec::IntoIter<[f64; 3]>,
    }

    impl<'a> Iterator for VertexIterMut<'a> {
        type Item = VertexMut<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            self.vel.next().map(|vel| VertexMut { vel })
        }
    }

    impl<'a> Iterator for VertexIter<'a> {
        type Item = VertexRef<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            self.vel.next().map(|vel| VertexRef { vel })
        }
    }

    impl Iterator for VertexIntoIter {
        type Item = Vertex;
        fn next(&mut self) -> Option<Self::Item> {
            self.vel.next().map(|vel| Vertex { vel })
        }
    }

    // Implement iterator on `VertexSet`
    impl IntoIterator for VertexSet {
        type Item = Vertex;
        type IntoIter = VertexIntoIter;

        fn into_iter(self) -> Self::IntoIter {
            VertexIntoIter {
                //pos: self.pos.into_iter(),
                vel: self.vel.into_iter(),
            }
        }
    }

    impl VertexMut<'_> {
        /// A sample function that simply modifies `Self`.
        fn integrate(self, dt: &f64) {
            for i in 0..3 {
                //self.pos[i] += dt*self.vel[i];
                self.vel[i] *= dt;
            }
        }
    }

    // Implement kernels
    impl VertexRef<'_> {
        /// A function that generates new data.
        fn displacement(self, dt: &f64) -> Displacement {
            let mut disp = Displacement([0.0; 3]);
            for i in 0..3 {
                //disp[i] = self.pos[i] + dt*self.vel[i];
                disp.0[i] = self.vel[i];
            }
            disp
        }
    }

    #[test]
    fn basic_set() {
        let mut vs = VertexSet {
            //pos: UniformSet::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vel: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        };

        let dt = 0.1;

        // The operators available on VertexSet determine how the data will be accessed. This means
        // the data will be available through a borrow, mutable borrow or by value. This is
        // consistent with `.iter`, `.iter_mut` and `.into_iter` functions on a `Vec`. We should
        // possibly preserve this interface for sequential access.
        vs.iter_mut().for_each(|x| x.integrate(&dt));
        let disp: Vec<_> = vs.iter().map(|x| x.displacement(&dt)).collect();
    }
}
