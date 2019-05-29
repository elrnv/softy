use super::FrictionParams;
use crate::contact::*;
use crate::energy_models::volumetric_neohookean::ElasticTetMeshEnergy;
use geo::math::{Matrix3, Vector2};
use reinterpret::*;

use utils::zip;

pub struct ElasticEnergyParams {
    pub energy_model: ElasticTetMeshEnergy,
    pub time_step: f64,
}

/// Elastic Friction solver. This solver uses the elasticity model when computing friction to
/// propagate the frictional contact forces through the solid. This is critical in several
/// scenarios like pinching.
pub struct ElasticFrictionSolver<'a, CJI> {
    /// A set of generalized velocities.
    velocity: &'a [Vector3<f64>],
    /// A set of contact forces for each contact point.
    contact_impulse: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Friction coefficient.
    mu: f64,
    /// Contact Jacobian is a sparse matrix that maps vectors from generalized coordinates to
    /// physical coordinates (e.g. from vertices to contact points).  If the `None` is specified,
    /// it is assumed that the contact Jacobian is the identity matrix, meaning that contacts occur
    /// at vertex positions.
    contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
    /// Vertex masses.
    masses: &'a [f64],

    elastic_energy: Option<ElasticEnergyParams>,
}

impl<'a> ElasticFrictionSolver<'a, std::iter::Empty<(usize, usize)>> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_impulse` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn without_contact_jacobian(
        velocity: &'a [[f64; 3]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        elastic_energy: Option<ElasticEnergyParams>,
    ) -> ElasticFrictionSolver<'a, std::iter::Empty<(usize, usize)>> {
        Self::new_impl(
            velocity,
            contact_impulse,
            contact_basis,
            masses,
            params,
            None,
            elastic_energy,
        )
    }
}

impl<'a, CJI: Iterator<Item = (usize, usize)>> ElasticFrictionSolver<'a, CJI> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_impulse` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [[f64; 3]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: (&'a [Matrix3<f64>], CJI),
    ) -> ElasticFrictionSolver<'a, CJI> {
        Self::new_impl(
            velocity,
            contact_impulse,
            contact_basis,
            masses,
            params,
            Some(contact_jacobian),
            None,
        )
    }

    fn new_impl(
        velocity: &'a [[f64; 3]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
        elastic_energy: Option<ElasticEnergyParams>,
    ) -> ElasticFrictionSolver<'a, CJI> {
        ElasticFrictionSolver {
            velocity: reinterpret_slice(velocity),
            contact_impulse,
            contact_basis,
            mu: params.dynamic_friction,
            contact_jacobian,
            masses,
            elastic_energy,
        }
    }

    /// Solve one step.
    pub fn step(&mut self) -> Vec<[f64; 3]> {
        // Solve quadratic optimization problem
        let mut friction_impulse = vec![Vector2::zeros(); self.velocity.len()];
        for (r, &v, &m, &cr) in zip!(
            friction_impulse.iter_mut(),
            self.velocity.iter(),
            self.masses.iter(),
            self.contact_impulse.iter()
        ) {
            let rc = -v * m; // Impulse candidate

            // Project onto the unit circle.
            let radius = self.mu * cr.abs();
            if rc.dot(rc) > radius * radius {
                *r = rc * (radius / rc.norm());
            } else {
                *r = rc;
            }
        }

        reinterpret_vec(friction_impulse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use approx::*;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(0.000001, mass)?;

        // Check that the point still has velocity in the positive x direction
        dbg!(&velocity);
        dbg!(&impulse);
        assert!(velocity[0] > 0.8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn sticking_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(1.5, mass)?;
        // Check that the point gets stuck
        dbg!(&impulse);
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6, epsilon = 1e-8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);
        Ok(())
    }

    fn sliding_point_tester(mu: f64, mass: f64) -> Result<(Vector2<f64>, Vector2<f64>), Error> {
        let params = FrictionParams {
            dynamic_friction: mu,
            inner_iterations: 30,
            tolerance: 1e-10,
            print_level: 5,
        };

        let velocity = vec![[1.0, 0.0]]; // one point sliding right.
        let contact_impulse = vec![10.0 * mass];
        let masses = vec![mass; 1];

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = ElasticFrictionSolver::without_contact_jacobian(
            &velocity,
            &contact_impulse,
            &contact_basis,
            &masses,
            params,
            None,
        );
        let solution = solver.step();

        let impulse = Vector2(solution[0]);
        let final_velocity = Vector2(velocity[0]) + impulse / mass;

        Ok((final_velocity, impulse))
    }

    /// A tetrahedron sliding on a slanted surface.
    #[test]
    fn sliding_tet() -> Result<(), Error> {
        let params = FrictionParams {
            dynamic_friction: 0.001,
            inner_iterations: 40,
            tolerance: 1e-10,
            print_level: 5,
        };

        let velocity = vec![
            [0.07225747944670913, 0.0000001280108566301736],
            [0.06185827187696774, -0.0060040275393186595],
        ]; // tet vertex velocities
        let contact_impulse = vec![-0.0000000018048827573828247, -0.00003259055555607145];
        let masses = vec![0.0003720701030949866, 0.0003720701030949866];

        let mut contact_basis = ContactBasis::new();
        let normals = vec![
            [-0.0, -0.7071067811865476, -0.7071067811865476],
            [-0.0, -0.7071067811865476, -0.7071067811865476],
        ];
        contact_basis.update_from_normals(normals);

        let mut solver = ElasticFrictionSolver::without_contact_jacobian(
            &velocity,
            &contact_impulse,
            &contact_basis,
            &masses,
            params,
            None,
        );
        let solution = solver.step();

        let final_velocity: Vec<_> = zip!(velocity.iter(), solution.iter(), masses.iter())
            .map(|(&v, &r, &m)| Vector2(v) + Vector2(r) / m)
            .collect();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    final_velocity[i][j],
                    velocity[i][j],
                    max_relative = 1e-2,
                    epsilon = 1e-5
                );
            }
        }

        Ok(())
    }
}
