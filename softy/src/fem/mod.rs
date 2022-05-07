pub mod nl;
#[cfg(feature = "optsolver")]
pub mod opt;
#[cfg(feature = "optsolver")]
pub mod state;

use geo::prim::{Tetrahedron, Triangle};
use tensr::Vector3;

use crate::Real;

// pub use self::nl::problem::*;
// pub use self::nl::solver::*;

/// Get reference tetrahedron.
///
/// This routine assumes that there is a vertex attribute called `ref` of type `[f32; 3]`.
pub fn ref_tet<T: Real>(ref_pos: &[[T; 3]]) -> Tetrahedron<f64> {
    Tetrahedron::new([
        Vector3::new(ref_pos[0]).cast::<f64>().into(),
        Vector3::new(ref_pos[1]).cast::<f64>().into(),
        Vector3::new(ref_pos[2]).cast::<f64>().into(),
        Vector3::new(ref_pos[3]).cast::<f64>().into(),
    ])
}

/// Get reference triangle.
pub fn ref_tri<T: Real>(ref_tri: &[[T; 3]]) -> Triangle<f64> {
    Triangle::new([
        Vector3::new(ref_tri[0]).cast::<f64>().into(),
        Vector3::new(ref_tri[1]).cast::<f64>().into(),
        Vector3::new(ref_tri[2]).cast::<f64>().into(),
    ])
}
