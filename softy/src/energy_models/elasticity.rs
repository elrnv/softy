//mod tet_inv_nh;
mod tet_nh;
//mod tri_nh;

pub use tet_nh::*;
//pub use tet_inv_nh::*;
//pub use tri_nh::*;

/// This trait defines an accessor for an elastic energy model. Elastic objects can implement this
/// trait to have a unified method for getting an elastic energy model.
pub trait Elasticity<'a, E> {
    fn elasticity(&'a self) -> E;
}
