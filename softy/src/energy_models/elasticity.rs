pub mod tet_nh;
pub mod tet_inv_nh;
pub mod tri_nh;

use crate::energy::Energy;

/// This trait defines an accessor for an elastic energy model. Elastic objects can implement this
/// trait to have a unified method for getting an elastic energy model.
pub trait Elasticity<E, T> where E: Energy<T> + EnergyGradient<T> + EnergyHessian {
    fn elasticity(&self) -> E;
}
