pub mod implicit_contact;
pub mod point_contact;
pub mod volume;

use crate::constraint::*;
use crate::contact::*;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use std::{cell::RefCell, rc::Rc};

pub use self::implicit_contact::*;
pub use self::point_contact::*;
pub use self::volume::*;

/// Construct a new contact constraint based on the given parameters. There are more than
/// one type of contact constraint, which is resolved using dynamic dispatch.
/// This approach reduces a lot of boiler plate code compared to using enums.
pub fn build_contact_constraint(
    tetmesh_rc: &Rc<RefCell<TetMesh>>,
    trimesh_rc: &Rc<RefCell<TriMesh>>,
    params: SmoothContactParams,
) -> Result<Box<dyn ContactConstraint>, crate::Error> {
    Ok(match params.contact_type {
        ContactType::Implicit => Box::new(ImplicitContactConstraint::new(
            tetmesh_rc,
            trimesh_rc,
            params.kernel,
            params.friction_params,
        )?),
        ContactType::Point => Box::new(PointContactConstraint::new(
            tetmesh_rc,
            trimesh_rc,
            params.kernel,
            params.friction_params,
        )?),
    })
}

/// A common pattern occuring with contact constraints becoming active and inactive is remapping
/// values computed in a simulation step to the values available in the next step with a different
/// set of active constraints. This is necessary for pure contact warm starts as well as friction
/// impulses being carried over to the next step.
///
/// `values` is a set of values that need to be remapped to the new active set.
/// `old_indices` is a set of indices corresponding to the old active set.
/// `new_indices` is a set of indices corresponding to the new active set.
/// It is assumed that `old_indices` and `new_indices` are given in a sorted order.
/// It is assumed that old_indices return the same number of elements as `values`.
pub fn remap_values<T: Copy>(
    values: impl Iterator<Item = T>,
    default: T,
    old_indices: impl Iterator<Item = usize> + Clone,
    new_indices: impl Iterator<Item = usize> + Clone,
) -> Vec<T> {
    // Check that both input slices are sorted.
    debug_assert!(is_sorted::IsSorted::is_sorted(&mut old_indices.clone()));
    debug_assert!(is_sorted::IsSorted::is_sorted(&mut new_indices.clone()));
    let mut old_iter = values.zip(old_indices);

    new_indices
        .map(move |new_idx| {
            let mut new_val = default;
            for (val, old_idx) in &mut old_iter {
                if old_idx < new_idx {
                    continue;
                }

                if old_idx == new_idx {
                    new_val = val;
                }

                break;
            }
            new_val
        })
        .collect()
}

pub trait ContactConstraint:
    Constraint<f64> + ConstraintJacobian<f64> + ConstraintHessian<f64>
{
    fn clear_friction_force(&mut self);
    /// Update the underlying friction impulse based on the given predictive step.
    fn update_friction_force(
        &mut self,
        contact_force: &[f64],
        x: &[[f64; 3]],
        dx: &[[f64; 3]],
    ) -> bool;
    /// Subtract the frictional impulse from the given gradient vector.
    fn subtract_friction_force(&self, grad: &mut [f64]);
    /// Compute the frictional energy dissipation.
    fn frictional_dissipation(&self, dx: &[f64]) -> f64;
    /// Remap existing friction forces to an updated neighbourhood set. This function will be
    /// called when neighbourhood information changes to ensure correct correspondence of friction
    /// forces to vertices. It may be not necessary to implement this function if friction forces are
    /// stored on the entire mesh.
    fn remap_friction(&mut self, old_set: &[usize], new_set: &[usize]) {}
    fn compute_contact_impulse(
        &self,
        x: &[f64],
        contact_force: &[f64],
        dt: f64,
        impulse: &mut [[f64; 3]],
    );
    /// Retrieve a vector of contact normals. These are unit vectors pointing
    /// away from the surface. These normals are returned for each query point
    /// even if it is not touching the surface. This function returns an error if
    /// there are no cached query points.
    fn contact_normals(&self, x: &[f64], dx: &[f64]) -> Result<Vec<[f64; 3]>, crate::Error>;
    /// Get the radius of influence.
    fn contact_radius(&self) -> f64;
    /// Update the multiplier for the radius of influence.
    fn update_radius_multiplier(&mut self, radius_multiplier: f64);
    /// A `Vec` of active constraint indices. This will return an error if there were no
    /// query points cached.
    fn active_constraint_indices(&self) -> Result<Vec<usize>, crate::Error>;
    /// Update the cache of query point neighbourhoods and return `true` if cache has changed.
    /// Note that this function doesn't remap any data corresponding to the old neighbourhood
    /// information. Instead, use `update_cache_with_mapping`, which also returns the mapping to
    /// old data needed to perform the remapping of any user data.
    fn update_cache(&mut self, pos: Option<&[f64]>, disp: Option<&[f64]>) -> bool;
    fn cached_neighbourhood_indices(&self) -> Vec<Index>;
    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative will be denser
    /// than needed, which typically results in slower performance. If it is set too low, there
    /// will be errors in the derivative. It is the callers responsibility to set this step
    /// accurately.
    fn update_max_step(&mut self, max_step: f64);

    ///// Update cache, return true if changed along with the mapping to original constraints.
    ///// For constraints with no mapping to an old constraint, the corresponding index will be
    ///// invalid.
    //fn update_cache_with_mapping(&mut self, pos: &[f64], disp: &[f64]) -> Result<(bool, Vec<Index>), crate::Error> {
    //    let old_indices = self.active_constraint_indices()?;
    //    let seq = (0..).map(|i| Index::new(i));
    //    if self.update_cache(Some(pos), Some(disp)) {
    //        let new_indices = self.active_constraint_indices()?;
    //        let mapping = remap_values(seq, Index::invalid(), &old_indices, &new_indices);
    //        Ok((true, mapping))
    //    } else {
    //        Ok((false, seq.take(old_indices.len()).collect::<Vec<_>>()))
    //    }
    //}
}
