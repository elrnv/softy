pub mod nl;
pub mod opt;
pub mod state;

use geo::prim::Tetrahedron;
use tensr::Vector3;

use crate::RefPosType;

// pub use self::nl::problem::*;
// pub use self::nl::solver::*;

/// Get reference tetrahedron.
///
/// This routine assumes that there is a vertex attribute called `ref` of type `[f32; 3]`.
pub fn ref_tet(ref_pos: &[RefPosType]) -> Tetrahedron<f64> {
    Tetrahedron::new([
        Vector3::new(ref_pos[0]).cast::<f64>().into(),
        Vector3::new(ref_pos[1]).cast::<f64>().into(),
        Vector3::new(ref_pos[2]).cast::<f64>().into(),
        Vector3::new(ref_pos[3]).cast::<f64>().into(),
    ])
}
