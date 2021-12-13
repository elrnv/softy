#[cfg(feature = "optsolver")]
pub mod point_contact;
pub mod volume;

#[cfg(feature = "optsolver")]
use crate::attrib_defines::*;
#[cfg(feature = "optsolver")]
use crate::contact::*;
#[cfg(feature = "optsolver")]
use num_traits::Zero;

use crate::constraint::*;
use crate::matrix::MatrixElementIndex;
use crate::Error;
use crate::Real;

//pub use self::linearized_point_contact::*;
#[cfg(feature = "optsolver")]
pub use self::point_contact::*;
pub use self::volume::*;
use tensr::*;

/// An struct describing a fixed, rigid or deformable contact surface.
#[cfg(feature = "optsolver")]
#[derive(Copy, Clone, Debug)]
pub struct ContactSurface<M, T> {
    pub mesh: M,
    pub kind: SurfaceKind<T>,
}

/// The kind of contact surface: fixed, rigid or deformable.
#[cfg(feature = "optsolver")]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SurfaceKind<T> {
    Fixed,
    Rigid { mass: T, inertia: Matrix3<T> },
    Deformable,
}

#[cfg(feature = "optsolver")]
impl<M, T> ContactSurface<M, T> {
    pub fn deformable(mesh: M) -> Self {
        ContactSurface {
            mesh,
            kind: SurfaceKind::Deformable,
        }
    }
    pub fn rigid(mesh: M, mass: T, inertia: Matrix3<T>) -> Self {
        ContactSurface {
            mesh,
            kind: SurfaceKind::Rigid { mass, inertia },
        }
    }
    pub fn fixed(mesh: M) -> Self {
        ContactSurface {
            mesh,
            kind: SurfaceKind::Fixed,
        }
    }
    #[inline]
    pub fn map<U, F: FnOnce(M) -> U>(self, f: F) -> ContactSurface<U, T> {
        ContactSurface {
            mesh: f(self.mesh),
            kind: self.kind,
        }
    }
}

/// Construct a new contact constraint based on the given parameters.
#[cfg(feature = "optsolver")]
pub fn build_contact_constraint<T: Real>(
    object: ContactSurface<&crate::TriMesh, f64>,
    collider: ContactSurface<&crate::TriMesh, f64>,
    params: FrictionalContactParams,
) -> Result<std::cell::RefCell<PointContactConstraint<T>>, crate::Error> {
    Ok(std::cell::RefCell::new(PointContactConstraint::new(
        object,
        collider,
        params.kernel,
        params.friction_params,
        params.contact_offset,
        params.contact_type == ContactType::LinearizedPoint,
    )?))
}

// TODO: move to a designated contact constraint.
pub fn compute_contact_penalty<S: Real>(d: S, delta: f32) -> S {
    if d.to_f32().unwrap() >= delta {
        S::zero()
    } else {
        let delta = S::from(delta).unwrap();
        let dd = delta - d;
        (dd * dd) / delta
    }
}
//pub fn compute_contact_penalty<S: Real>(
//    // Input distance & Output force magnitude
//    lambda: &mut [S],
//    delta: f32,
//) {
//    lambda.iter_mut().for_each(|lambda| {
//        let d = *lambda;
//        *lambda = if d.to_f32().unwrap() >= delta {
//            S::zero()
//        } else {
//            let delta = S::from(delta).unwrap();
//            let dd = delta - d;
//            (dd * dd * dd) / delta
//        }
//    });
//}

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
    new_indices: impl ExactSizeIterator<Item = usize> + Clone,
) -> Vec<T> {
    remap_values_iter(values, default, old_indices, new_indices).collect()
}

/// Same as above but this returns an iterator.
pub fn remap_values_iter<T: Copy>(
    values: impl Iterator<Item = T>,
    default: T,
    old_indices: impl Iterator<Item = usize> + Clone,
    new_indices: impl ExactSizeIterator<Item = usize> + Clone,
) -> impl ExactSizeIterator<Item = T> {
    // Check that both input slices are sorted.
    debug_assert!(is_sorted::IsSorted::is_sorted(&mut old_indices.clone()));
    debug_assert!(is_sorted::IsSorted::is_sorted(&mut new_indices.clone()));
    let mut old_iter = values.zip(old_indices).peekable();

    new_indices.map(move |new_idx| {
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

#[cfg(feature = "optsolver")]
pub trait ContactConstraint<T: Real>:
    for<'a> Constraint<'a, T, Input = [SubsetView<'a, Chunked3<&'a [T]>>; 2]>
    + for<'a> ContactConstraintJacobian<'a, T>
    + for<'a> ContactConstraintHessian<'a, T, InputDual = &'a [T]>
    + std::fmt::Debug
{
    /// Total number of contacts that could occur.
    fn num_potential_contacts(&self) -> usize;
    /// Provide the frictional contact data.
    fn frictional_contact(&self) -> Option<&crate::friction::FrictionImpulses<T>>;
    /// Provide the frictional contact mutable data.
    fn frictional_contact_mut(&mut self) -> Option<&mut crate::friction::FrictionImpulses<T>>;
    /// Return a set of surface vertex indices that could be in contact.
    fn active_surface_vertex_indices(&self) -> utils::aref::ARef<'_, [usize]>;

    /// Clear the saved frictional contact impulse.
    fn clear_frictional_contact_impulse(&mut self) {
        if let Some(ref mut frictional_contact) = self.frictional_contact_mut() {
            frictional_contact.object_impulse.clear();
            frictional_contact.collider_impulse.clear();
        }
    }

    /// Compute the contact Jacobian as an ArrayFire matrix.
    #[cfg(feature = "af")]
    fn contact_jacobian_af(&self) -> af::Array<f64>;

    /// Project friction impulses to the tangential plane according to the provided positional
    /// configuration.
    fn project_friction_impulses(&mut self, x: [SubsetView<Chunked3<&[T]>>; 2]);

    /// Update the position configuration of contacting objects using the given position data.
    fn update_contact_pos(&mut self, x: [SubsetView<Chunked3<&[T]>>; 2]);

    /// Update the underlying friction impulse based on the given predictive step.
    fn update_frictional_contact_impulse(
        &mut self,
        contact_impulse: &[T],
        x: [SubsetView<Chunked3<&[T]>>; 2],
        dx: [SubsetView<Chunked3<&[T]>>; 2],
        rigid_motion: [Option<[[T; 3]; 2]>; 2],
        constraint_values: &[T],
        friction_steps: u32,
    ) -> u32;

    fn add_mass_weighted_frictional_contact_impulse_to_object(
        &self,
        x: SubsetView<Chunked3<&mut [T]>>,
    );

    fn add_mass_weighted_frictional_contact_impulse_to_collider(
        &self,
        x: SubsetView<Chunked3<&mut [T]>>,
    );

    fn smooth_collider_values(&self, _: SubsetView<&mut [T]>) {}

    /// Add the friction corrector impulse to the given vector.
    fn add_friction_corrector_impulse(&self, _: [SubsetView<Chunked3<&mut [T]>>; 2], _: T) {}

    fn collider_contact_normals(&mut self, _: Chunked3<&mut [T]>) {}

    /// Add the frictional impulse to the given gradient vector representing the object.
    fn add_friction_impulse_to_object(&self, grad: SubsetView<Chunked3<&mut [T]>>, multiplier: T);

    /// Add the frictional impulse to the given gradient vector representing the collider.
    fn add_friction_impulse_to_collider(&self, grad: SubsetView<Chunked3<&mut [T]>>, multiplier: T);

    /// Compute the frictional energy dissipation.
    fn frictional_dissipation(&self, vel: [SubsetView<Chunked3<&[T]>>; 2]) -> T {
        let mut dissipation = T::zero();
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.object_impulse.is_empty() {
                return dissipation;
            }

            let indices = self.active_surface_vertex_indices();
            if indices.is_empty() {
                return dissipation;
            }

            assert_eq!(indices.len(), frictional_contact.object_impulse.len());

            for (contact_idx, (&i, (_, &r))) in indices
                .iter()
                .zip(frictional_contact.object_impulse.iter())
                .enumerate()
            {
                let r_t = if !frictional_contact.contact_basis.is_empty() {
                    let f = frictional_contact
                        .contact_basis
                        .to_contact_coordinates(r, contact_idx);
                    tensr::Vector3::new(
                        frictional_contact
                            .contact_basis
                            .from_contact_coordinates([T::zero(), f[1], f[2]], contact_idx)
                            .into(),
                    )
                } else {
                    tensr::Vector3::zero()
                };

                dissipation += tensr::Vector3::new(vel[0][i]).dot(r_t);
            }
        }
        dissipation
    }

    fn add_contact_impulse(
        &mut self,
        x: [SubsetView<Chunked3<&[T]>>; 2],
        contact_impulse: &[T],
        impulse: [Chunked3<&mut [T]>; 2],
    );
    /// Retrieve a vector of contact normals. These are unit vectors pointing
    /// away from the surface. These normals are returned for each query point
    /// even if it is not touching the surface. This function returns an error if
    /// there are no cached query points.
    fn contact_normals(&self) -> Vec<[T; 3]>;
    /// Get the radius of influence.
    fn contact_radius(&self) -> f64;
    /// Update the multiplier for the radius of influence.
    fn update_radius_multiplier(&mut self, radius_multiplier: f64);
    /// A `Vec` of active constraint indices. This will return an error if there were no
    /// query points cached.
    fn active_constraint_indices(&self) -> Vec<usize>;
    /// Update the query point neighborhoods and return `true` if changed.
    /// Note that this function doesn't remap any data corresponding to the old neighborhood
    /// information.
    fn update_neighbors(
        &mut self,
        object_pos: SubsetView<Chunked3<&[T]>>,
        collider_pos: SubsetView<Chunked3<&[T]>>,
    ) -> bool;
    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative will be denser
    /// than needed, which typically results in slower performance. If it is set too low, there
    /// will be errors in the derivative. It is the callers responsibility to set this step
    /// accurately.
    fn update_max_step(&mut self, max_step: f64);
    /// Precompute constraint functions based on the given position to be used in the optimization.
    fn linearize_constraint(
        &mut self,
        _object_pos: SubsetView<Chunked3<&[T]>>,
        _collider_pos: SubsetView<Chunked3<&[T]>>,
    ) {
    }
    fn is_linear(&self) -> bool {
        false
    }
}

pub trait ContactConstraintJacobian<'a, T: Scalar>: Constraint<'a, T> {
    /// The number of non-zeros in the Jacobian matrix of the constraint provided by the
    /// `constraint_jacobian_indices` and `constraint_jacobian_values` functions.
    fn constraint_jacobian_size(&self) -> usize;

    fn constraint_jacobian_indices_iter<'b>(
        &'b self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'b>;

    /// Compute the values of the constraint Jacobian.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_jacobian_values(
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        values: &mut [T],
    ) -> Result<(), Error>;
}

pub trait ContactConstraintHessian<'a, T: Scalar>: Constraint<'a, T> {
    type InputDual;
    /// The number of non-zeros in the Hessian matrix of the constraint.
    fn constraint_hessian_size(&self) -> usize;
    /// Compute the Hessian matrix values (multiplied by `lambda`) corresponding to their positions
    /// in the matrix returned by `constraint_hessian_indices`. This means that the vector returned
    /// from this function must have the same length as the vector returned by
    /// `constraint_hessian_indices`.
    ///
    ///   - `x` is the variable expected by the specific constraint for the previous configuration.
    ///   - `dx` is the independent variable being optimized over, it is not necessarily the
    ///     differential of `x` but it often is.
    fn constraint_hessian_values(
        &mut self,
        x: Self::Input,
        dx: Self::Input,
        lambda: Self::InputDual,
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error>;

    /// Compute the Hessian row and column indices of the matrix resulting from the constraint
    /// Hessian multiplied by the Lagrange multiplier vector.
    fn constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'b>, Error>;
}
