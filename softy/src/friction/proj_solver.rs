#![allow(dead_code)]
use super::FrictionParams;
use crate::contact::*;
use geo::math::Vector2;
use reinterpret::*;

use utils::zip;

/// Friction solver.
pub struct FrictionSolver<'a> {
    /// A set of tangential impulses in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    predictor_impulse: &'a [Vector2<f64>],
    /// A set of contact forces for each contact point.
    contact_impulse: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Friction coefficient.
    mu: f64,
    /// Contact Jacobian is a sparse matrix that maps vectors from vertices to contact points.
    /// If the `None` is specified, it is assumed that the contact Jacobian is the identity matrix,
    /// meaning that contacts occur at vertex positions.
    contact_jacobian: Option<ContactJacobianView<'a>>,
    mass_inv_mtx: EffectiveMassInvView<'a>,
}

impl<'a> FrictionSolver<'a> {
    /// Build a new solver for the friction problem. The given `predictor_impulse` is a stacked
    /// vector of tangential impulses for each contact point in contact space. `contact_impulse` is
    /// the normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn without_contact_jacobian(
        predictor_impulse: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
    ) -> FrictionSolver<'a> {
        Self::new_impl(
            predictor_impulse,
            contact_impulse,
            contact_basis,
            mass_inv_mtx,
            params,
            None,
        )
    }
}

impl<'a> FrictionSolver<'a> {
    /// Build a new solver for the friction problem. The given `predictor_impulse` is a stacked
    /// vector of tangential impulses for each contact point in contact space. `contact_impulse` is
    /// the normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub(crate) fn new(
        predictor_impulse: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
        contact_jacobian: ContactJacobianView<'a>,
    ) -> FrictionSolver<'a> {
        Self::new_impl(
            predictor_impulse,
            contact_impulse,
            contact_basis,
            mass_inv_mtx,
            params,
            Some(contact_jacobian),
        )
    }

    fn new_impl(
        predictor_impulse: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
        contact_jacobian: Option<ContactJacobianView<'a>>,
    ) -> FrictionSolver<'a> {
        FrictionSolver {
            predictor_impulse: reinterpret_slice(predictor_impulse),
            contact_impulse,
            contact_basis,
            mu: params.dynamic_friction,
            contact_jacobian,
            mass_inv_mtx,
        }
    }

    /// Solve one step.
    pub fn step(&mut self) -> Vec<[f64; 2]> {
        // Solve quadratic optimization problem
        let mut friction_impulse = vec![Vector2::zeros(); self.predictor_impulse.len()];
        for (r, &v, &cr) in zip!(
            friction_impulse.iter_mut(),
            self.predictor_impulse.iter(),
            //self.mass_inv_mtx.iter(),
            self.contact_impulse.iter()
        ) {
            // This needs to incorporate effective mass.
            eprintln!("WARNING: Calling incomplete friction implementation!");
            let rc = -v; // Impulse candidate

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
    use utils::soap::*;
    use crate::Error;
    use approx::*;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let mass = 10.0;
        let (predictor_impulse, impulse) = sliding_point_tester(0.000001, mass)?;

        // Check that the point still has predictor_impulse in the positive x direction
        dbg!(&predictor_impulse);
        dbg!(&impulse);
        assert!(predictor_impulse[0] > 0.8);

        // Sanity check that no impulses were produced in the process
        assert_relative_eq!(predictor_impulse[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);
        Ok(())
    }

    //#[test]
    fn sticking_point() -> Result<(), Error> {
        let mass = 10.0;
        let (predictor_impulse, impulse) = sliding_point_tester(1.5, mass)?;
        // Check that the point gets stuck
        dbg!(&impulse);
        assert_relative_eq!(predictor_impulse[0], 0.0, max_relative = 1e-6, epsilon = 1e-8);

        // Sanity check that no perpendicular impulses were produced in the process
        assert_relative_eq!(predictor_impulse[1], 0.0, max_relative = 1e-6);
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

        let predictor_impulse  = vec![[1.0 * mass, 0.0]]; // one point sliding right.
        let contact_impulse = vec![10.0 * mass];
        let mass_inv_mtx: DSMatrix3 = Tensor::new(Chunked3::from_flat(vec![1.0 / mass; 3])).into();

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = FrictionSolver::without_contact_jacobian(
            &predictor_impulse,
            &contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        );
        let solution = solver.step();

        let impulse = Vector2(solution[0]);
        let final_predictor = Vector2(predictor_impulse[0]) + impulse / mass;

        Ok((final_predictor, impulse))
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

        let contact_impulse = vec![-0.0000000018048827573828247, -0.00003259055555607145];
        let masses = vec![0.0003720701030949866, 0.0003720701030949866];
        let predictor_impulse = vec![
            [0.07225747944670913 * masses[0], 0.0000001280108566301736 * masses[0]],
            [0.06185827187696774 * masses[1], -0.0060040275393186595 * masses[1]],
        ]; // tet vertex impulses

        let mass_inv_mtx: DSMatrix3 =
            Tensor::new(Chunked3::from_array_vec(vec![[1.0 / masses[0]; 3], [1.0 / masses[1]; 3]])).into();

        let mut contact_basis = ContactBasis::new();
        let normals = vec![
            [-0.0, -0.7071067811865476, -0.7071067811865476],
            [-0.0, -0.7071067811865476, -0.7071067811865476],
        ];
        contact_basis.update_from_normals(normals);

        let mut solver = FrictionSolver::without_contact_jacobian(
            &predictor_impulse,
            &contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
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
