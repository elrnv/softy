use geo::math::{Matrix3, Vector2};
use reinterpret::*;
use crate::contact::*;

use unroll::unroll_for_loops;
use utils::zip;

use crate::Error;

/// Contact solver.
pub struct ContactSolver<'a, CJI> {
    /// A set of tangential velocities in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    velocity: &'a [f64],
    /// A set of friction impulses for each contact point.
    friction_impulse: &'a [Vector2<f64>],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Contact Jacobian is a sparse matrix that maps vectors from vertices to contact points.
    /// If the `None` is specified, it is assumed that the contact Jacobian is the identity matrix,
    /// meaning that contacts occur at vertex positions.
    contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
    /// Vertex masses.
    masses: &'a [f64],
}

impl<'a> ContactSolver<'a, std::iter::Empty<(usize, usize)>> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_impulse` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn without_contact_jacobian(
        velocity: &'a [f64],
        friction_impulse: &'a [[f64; 2]],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
    ) -> ContactSolver<'a, std::iter::Empty<(usize, usize)>> {
        Self::new_impl(velocity, friction_impulse, contact_basis, masses, None)
    }
}

impl<'a, CJI: Iterator<Item=(usize, usize)>> ContactSolver<'a, CJI> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `friction_impulse` is the
    /// tangential component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [f64],
        friction_impulse: &'a [[f64;2]],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        contact_jacobian: (&'a [Matrix3<f64>], CJI),
    ) -> ContactSolver<'a, CJI> {
        Self::new_impl(velocity, friction_impulse, contact_basis, masses, Some(contact_jacobian))
    }

    fn new_impl(
        velocity: &'a [f64],
        friction_impulse: &'a [[f64;2]],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
    ) -> ContactSolver<'a, CJI> {
        ContactSolver {
            velocity,
            friction_impulse: reinterpret_slice(friction_impulse),
            contact_basis,
            contact_jacobian,
            masses,
        }
    }

    /// Solve one step.
    pub fn step(&mut self) -> Vec<f64> {
        // Solve quadratic optimization problem
        let mut contact_impulse = vec![0.0; self.velocity.len()];
        for (rn, &v, &m) in zip!(contact_impulse.iter_mut(), self.velocity.iter(), self.masses.iter()) {
            *rn = (-v * m).max(0.0);
        }
        
        reinterpret_vec(contact_impulse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn contact_test() -> Result<(), Error> {
        let velocity = vec![-0.1]; // downward velocity
        let friction_impulse = vec![[-1.0, 0.0]];
        let masses = vec![1.0];

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = ContactSolver::without_contact_jacobian(&velocity, &friction_impulse, &contact_basis, &masses);
        let solution = solver.step();

        let impulse = solution[0];
        let final_velocity = velocity[0] + impulse / masses[0];
        dbg!(impulse);
        dbg!(final_velocity);

        Ok(())
    }
}
