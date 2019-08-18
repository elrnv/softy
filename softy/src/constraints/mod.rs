pub mod implicit_contact;
pub mod point_contact;
//pub mod sp_implicit_contact;
pub mod volume;

use crate::attrib_defines::*;
use crate::constraint::*;
use crate::contact::*;
use crate::fem::problem::Var;
use crate::friction::FrictionalContact;
use crate::Index;
use crate::TriMesh;
use geo::math::Vector3;

pub use self::implicit_contact::*;
pub use self::point_contact::*;
//pub use self::sp_implicit_contact::*;
pub use self::volume::*;
use utils::aref::*;
use utils::soap::*;

/// Construct a new contact constraint based on the given parameters. There are more than
/// one type of contact constraint, which is resolved using dynamic dispatch.
/// This approach reduces a lot of boiler plate code compared to using enums.
pub fn build_contact_constraint(
    object: Var<&TriMesh>,
    collider: Var<&TriMesh>,
    params: FrictionalContactParams,
) -> Result<Box<dyn ContactConstraint>, crate::Error> {
    Ok(match params.contact_type {
        ContactType::SPImplicit =>
        //    Box::new(SPImplicitContactConstraint::new(
        //    object,
        //    collider,
        //    params.kernel,
        //    params.friction_params,
        //    )?)
        {
            unimplemented!()
        }
        ContactType::Implicit => Box::new(ImplicitContactConstraint::new(
            object.untag(),
            collider.untag(),
            params.kernel,
            params.friction_params,
        )?),
        ContactType::Point => Box::new(PointContactConstraint::new(
            object,
            collider,
            params.kernel,
            params.friction_params,
        )?),
    })
}

/// A common pattern occurring with contact constraints becoming active and inactive is remapping
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
    let mut old_iter = values.zip(old_indices).peekable();

    new_indices
        .map(move |new_idx| {
            while let Some(&(_, old_idx)) = old_iter.peek() {
                if old_idx < new_idx {
                    // Trash old value, no corresponding new value here.
                    old_iter.next();
                    continue;
                }

                if old_idx == new_idx {
                    // A match! Consume old value.
                    return old_iter.next().unwrap().0;
                }

                // Otherwise, we don't consume old and increment new_iter only.

                break;
            }
            default
        })
        .collect()
}

#[test]
fn remap_values_identity_test() {
    let old_indices = vec![284, 288, 572, 573, 574, 575];
    let new_indices = vec![284, 288, 572, 573, 574, 575];
    let values: Vec<f64> = old_indices.iter().map(|&x| (x as f64)).collect();
    let new_values = remap_values(
        values.into_iter(),
        0.0,
        old_indices.into_iter(),
        new_indices.into_iter(),
    );
    assert_eq!(new_values, vec![284.0, 288.0, 572.0, 573.0, 574.0, 575.0]);
}

#[test]
fn remap_values_more_new_test() {
    let old_indices = vec![284, 288, 572, 573, 574, 575];
    let new_indices = vec![
        284, 288, 295, 297, 304, 306, 316, 317, 318, 572, 573, 574, 575,
    ];
    let values: Vec<f64> = old_indices.iter().map(|&x| (x as f64)).collect();
    let new_values = remap_values(
        values.into_iter(),
        0.0,
        old_indices.into_iter(),
        new_indices.into_iter(),
    );
    assert_eq!(
        new_values,
        vec![284.0, 288.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 572.0, 573.0, 574.0, 575.0]
    );
}
#[test]
fn remap_values_more_old_test() {
    let old_indices = vec![
        284, 288, 295, 297, 304, 306, 316, 317, 318, 572, 573, 574, 575,
    ];
    let new_indices = vec![284, 288, 572, 573, 574, 575];
    let values: Vec<f64> = old_indices.iter().map(|&x| (x as f64)).collect();
    let new_values = remap_values(
        values.into_iter(),
        0.0,
        old_indices.into_iter(),
        new_indices.into_iter(),
    );
    assert_eq!(new_values, vec![284.0, 288.0, 572.0, 573.0, 574.0, 575.0]);
}
#[test]
fn remap_values_complex_test() {
    let old_indices = vec![1, 2, 6, 7, 10, 11];
    let new_indices = vec![0, 1, 3, 4, 5, 6, 7, 8, 11, 12];
    let values: Vec<f64> = old_indices.iter().map(|&x| (x as f64)).collect();
    let new_values = remap_values(
        values.into_iter(),
        0.0,
        old_indices.into_iter(),
        new_indices.into_iter(),
    );
    assert_eq!(
        new_values,
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 11.0, 0.0]
    );
}

pub trait ContactConstraint:
    for<'a> Constraint<'a, f64, Input = [SubsetView<'a, Chunked3<&'a [f64]>>; 2]>
    + for<'a> ConstraintJacobian<'a, f64>
    + for<'a> ConstraintHessian<'a, f64, InputDual = &'a [f64]>
    + std::fmt::Debug
{
    /// Provide the frictional contact data.
    fn frictional_contact(&self) -> Option<&FrictionalContact>;
    /// Provide the frictional contact mutable data.
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact>;
    /// Return a set of surface vertex indices that could be in contact.
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]>;

    /// Clear the saved frictional contact impulse.
    fn clear_frictional_contact_impulse(&mut self) {
        if let Some(ref mut frictional_contact) = self.frictional_contact_mut() {
            frictional_contact.object_impulse.clear();
            frictional_contact.collider_impulse.clear();
        }
    }

    /// Compute the contact Jacobian as an ArrayFire matrix.
    fn contact_jacobian_af(&self) -> af::Array<f64>;

    /// Update the underlying friction impulse based on the given predictive step.
    fn update_frictional_contact_impulse(
        &mut self,
        contact_impulse: &[f64],
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        dx: [SubsetView<Chunked3<&[f64]>>; 2],
        constraint_values: &[f64],
        friction_steps: u32,
    ) -> u32;

    fn add_mass_weighted_frictional_contact_impulse(
        &self,
        x: [SubsetView<Chunked3<&mut [f64]>>; 2],
    );

    /// Add the frictional impulse to the given gradient vector.
    fn add_friction_impulse(
        &self,
        mut grad: [SubsetView<Chunked3<&mut [f64]>>; 2],
        multiplier: f64,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.object_impulse.is_empty() {
                return;
            }

            let indices = self.active_surface_vertex_indices();
            if indices.is_empty() {
                return;
            }

            assert_eq!(indices.len(), frictional_contact.object_impulse.len());
            for (contact_idx, (&i, &r)) in indices
                .iter()
                .zip(frictional_contact.object_impulse.iter())
                .enumerate()
            {
                let r_t = if !frictional_contact.contact_basis.is_empty() {
                    let f = frictional_contact
                        .contact_basis
                        .to_contact_coordinates(r, contact_idx);
                    Vector3(
                        frictional_contact
                            .contact_basis
                            .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                            .into(),
                    )
                } else {
                    Vector3::zeros()
                };

                grad[0][i] = (Vector3(grad[0][i]) + r_t * multiplier).into();
            }
        }
    }

    /// Compute the frictional energy dissipation.
    fn frictional_dissipation(&self, vel: [SubsetView<Chunked3<&[f64]>>; 2]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.object_impulse.is_empty() {
                return dissipation;
            }

            let indices = self.active_surface_vertex_indices();
            if indices.is_empty() {
                return dissipation;
            }

            assert_eq!(indices.len(), frictional_contact.object_impulse.len());

            for (contact_idx, (&i, &r)) in indices
                .iter()
                .zip(frictional_contact.object_impulse.iter())
                .enumerate()
            {
                let r_t = if !frictional_contact.contact_basis.is_empty() {
                    let f = frictional_contact
                        .contact_basis
                        .to_contact_coordinates(r, contact_idx);
                    Vector3(
                        frictional_contact
                            .contact_basis
                            .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                            .into(),
                    )
                } else {
                    Vector3::zeros()
                };

                dissipation += Vector3(vel[0][i]).dot(r_t);
            }
        }
        dissipation
    }

    /// Remap existing friction impulses to an updated neighbourhood set. This function will be
    /// called when neighbourhood information changes to ensure correct correspondence of friction
    /// impulses to vertices. It may be not necessary to implement this function if friction
    /// impulses are stored on the entire mesh.
    fn remap_frictional_contact(&mut self, _old_set: &[usize], _new_set: &[usize]) {}
    fn add_contact_impulse(
        &self,
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        contact_impulse: &[f64],
        impulse: [Chunked3<&mut [f64]>; 2],
    );
    /// Retrieve a vector of contact normals. These are unit vectors pointing
    /// away from the surface. These normals are returned for each query point
    /// even if it is not touching the surface. This function returns an error if
    /// there are no cached query points.
    fn contact_normals(
        &self,
        x: [SubsetView<Chunked3<&[f64]>>; 2],
    ) -> Result<Chunked3<Vec<f64>>, crate::Error>;
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
    fn update_cache(
        &mut self,
        object_pos: SubsetView<Chunked3<&[f64]>>,
        collider_pos: SubsetView<Chunked3<&[f64]>>,
    ) -> bool;
    fn cached_neighbourhood_indices(&self) -> Vec<Index>;
    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative will be denser
    /// than needed, which typically results in slower performance. If it is set too low, there
    /// will be errors in the derivative. It is the callers responsibility to set this step
    /// accurately.
    fn update_max_step(&mut self, max_step: f64);
}
