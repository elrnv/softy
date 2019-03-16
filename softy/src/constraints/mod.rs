pub mod point_contact;
pub mod implicit_contact;
pub mod volume;

use crate::constraint::*;
use std::{cell::RefCell, rc::Rc};
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactType {
    Implicit,
    Point
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SmoothContactParams {
    pub contact_type: ContactType,
    pub radius: f64,
    pub tolerance: f64,
}

/// Construct a new contact constraint based on the given parameters. There are more than
/// one type of contact constraint, which is resolved using dynamic dispatch.
/// This approach reduces a lot of boiler plate code compared to using enums.
pub fn build_contact_constraint(tetmesh_rc: &Rc<RefCell<TetMesh>>,
                                trimesh_rc: &Rc<RefCell<TriMesh>>,
                                params: SmoothContactParams) -> Result<Box<dyn ContactConstraint>, crate::Error> {
    Ok(match params.contact_type {
        ContactType::Implicit =>
            Box::new(ImplicitContactConstraint::new(tetmesh_rc, trimesh_rc, params.radius, params.tolerance)?),
        ContactType::Point =>
            Box::new(PointContactConstraint::new(tetmesh_rc, trimesh_rc, params.radius, params.tolerance)?),
    })
}

pub trait ContactConstraint: Constraint<f64> + ConstraintJacobian<f64> + ConstraintHessian<f64> {
    /// Get the radius of influence.
    fn contact_radius(&self) -> f64;
    /// Update the radius of influence.
    fn update_radius(&mut self, radius: f64);
    /// Update the cache of query point neighbourhoods and return `true` if cache has changed.
    fn update_cache(&mut self) -> bool;
    fn cached_neighbourhood_indices(&self) -> Vec<Index>;
    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative will be denser
    /// than needed, which typically results in slower performance. If it is set too low, there
    /// will be errors in the derivative. It is the callers responsibility to set this step
    /// accurately.
    fn update_max_step(&mut self, max_step: f64);
    fn build_constraint_mapping(&self, old_constrained_points: &[Index], new_to_old_constraint_mapping: &mut [Index]);
}

pub use self::point_contact::*;
pub use self::implicit_contact::*;
pub use self::volume::*;
