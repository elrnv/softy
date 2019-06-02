pub mod implicit_contact;
pub mod point_contact;
pub mod sp_implicit_contact;
pub mod volume;

use crate::attrib_defines::*;
use crate::constraint::*;
use crate::contact::*;
use crate::energy_models::volumetric_neohookean::ElasticTetMeshEnergy;
use crate::friction::FrictionalContact;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use geo::{
    math::Vector3,
    mesh::{topology::*, Attrib},
};
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};

pub use self::implicit_contact::*;
pub use self::point_contact::*;
pub use self::sp_implicit_contact::*;
pub use self::volume::*;

/// Construct a new contact constraint based on the given parameters. There are more than
/// one type of contact constraint, which is resolved using dynamic dispatch.
/// This approach reduces a lot of boiler plate code compared to using enums.
pub fn build_contact_constraint(
    tetmesh_rc: &Rc<RefCell<TetMesh>>,
    trimesh_rc: &Rc<RefCell<TriMesh>>,
    params: SmoothContactParams,
    density: f64,
    time_step: f64,
    energy_model: ElasticTetMeshEnergy,
) -> Result<Box<dyn ContactConstraint>, crate::Error> {
    Ok(match params.contact_type {
        ContactType::SPImplicit => Box::new(SPImplicitContactConstraint::new(
            tetmesh_rc,
            trimesh_rc,
            params.kernel,
            params.friction_params,
            density,
        )?),
        ContactType::Implicit => Box::new(ImplicitContactConstraint::new(
            tetmesh_rc,
            trimesh_rc,
            params.kernel,
            params.friction_params,
            density,
            time_step,
            energy_model,
        )?),
        ContactType::Point => Box::new(PointContactConstraint::new(
            tetmesh_rc,
            trimesh_rc,
            params.kernel,
            params.friction_params,
            density,
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

/// Return a vector of masses per simulation mesh vertex.
pub fn compute_vertex_masses(tetmesh: &TetMesh, density: f64) -> Vec<f64> {
    let mut all_masses = vec![0.0; tetmesh.num_vertices()];

    for (&vol, cell) in tetmesh
        .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
        .unwrap()
        .zip(tetmesh.cell_iter())
    {
        for i in 0..4 {
            all_masses[cell[i]] += 0.25 * vol * density;
        }
    }

    all_masses
}

/// A reference abstracting over borrowed cell refs and plain old & style refs.
pub enum ARef<'a, T: ?Sized> {
    Plain(&'a T),
    Cell(std::cell::Ref<'a, T>),
}

impl<'a, T: ?Sized> std::ops::Deref for ARef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            ARef::Plain(r) => r,
            ARef::Cell(r) => r,
        }
    }
}

pub trait ContactConstraint:
    Constraint<f64> + ConstraintJacobian<f64> + ConstraintHessian<f64>
{
    /// Provide the frictional contact data.
    fn frictional_contact(&self) -> Option<&FrictionalContact>;
    /// Provide the frictional contact mutable data.
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact>;

    /// Return a slice that maps from a given surface vertex index to a corresponding simulation
    /// mesh vertex.
    fn vertex_index_mapping(&self) -> Option<&[usize]>;

    /// Return a set of surface vertex indices that could be in contact.
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]>;

    /// Clear the saved frictional contact impulse.
    fn clear_frictional_contact_impulse(&mut self) {
        if let Some(ref mut frictional_contact) = self.frictional_contact_mut() {
            frictional_contact.impulse.clear();
        }
    }

    /// Compute the contact Jacobian as an ArrayFire matrix.
    fn contact_jacobian_af(&self) -> af::Array<f64>;

    /// Compute the contact Jacobian as a sparse matrix.
    fn contact_jacobian_sprs(&self) -> sprs::CsMat<f64>;

    /// Update the underlying friction impulse based on the given predictive step.
    fn update_frictional_contact_impulse(
        &mut self,
        contact_force: &[f64],
        x: &[[f64; 3]],
        dx: &[[f64; 3]],
        constraint_values: &[f64],
        friction_steps: u32,
    ) -> u32;

    fn add_mass_weighted_frictional_contact_impulse(&self, x: &mut [f64]);
    /// Add the frictional impulse to the given gradient vector.
    fn add_friction_impulse(&self, grad: &mut [f64], multiplier: f64) {
        let grad: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.impulse.is_empty() {
                return;
            }

            let indices = self.active_surface_vertex_indices();
            if indices.is_empty() {
                return;
            }

            assert_eq!(indices.len(), frictional_contact.impulse.len());
            for (contact_idx, (&i, &r)) in indices
                .iter()
                .zip(frictional_contact.impulse.iter())
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

                let vertex_idx = self.vertex_index_mapping().map_or(i, |m| m[i]);
                grad[vertex_idx] += Vector3(r_t.into()) * multiplier;
            }
        }
    }
    /// Compute the frictional energy dissipation.
    fn frictional_dissipation(&self, v: &[f64]) -> f64 {
        let vel: &[Vector3<f64>] = reinterpret_slice(v);
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.impulse.is_empty() {
                return dissipation;
            }

            let indices = self.active_surface_vertex_indices();
            if indices.is_empty() {
                return dissipation;
            }

            assert_eq!(indices.len(), frictional_contact.impulse.len());

            for (contact_idx, (&i, &r)) in indices
                .iter()
                .zip(frictional_contact.impulse.iter())
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

                let vertex_idx = self.vertex_index_mapping().map_or(i, |m| m[i]);
                dissipation += vel[vertex_idx].dot(r_t);
            }
        }
        dissipation
    }

    /// Remap existing friction impulses to an updated neighbourhood set. This function will be
    /// called when neighbourhood information changes to ensure correct correspondence of friction
    /// impulses to vertices. It may be not necessary to implement this function if friction
    /// impulses are stored on the entire mesh.
    fn remap_frictional_contact(&mut self, _old_set: &[usize], _new_set: &[usize]) {}
    fn compute_contact_impulse(&self, x: &[f64], contact_force: &[f64], impulse: &mut [[f64; 3]]);
    /// Retrieve a vector of contact normals. These are unit vectors pointing
    /// away from the surface. These normals are returned for each query point
    /// even if it is not touching the surface. This function returns an error if
    /// there are no cached query points.
    fn contact_normals(&self, x: &[f64]) -> Result<Vec<[f64; 3]>, crate::Error>;
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
    fn update_cache(&mut self, x: Option<&[f64]>) -> bool;
    fn cached_neighbourhood_indices(&self) -> Vec<Index>;
    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative will be denser
    /// than needed, which typically results in slower performance. If it is set too low, there
    /// will be errors in the derivative. It is the callers responsibility to set this step
    /// accurately.
    fn update_max_step(&mut self, max_step: f64);
}
