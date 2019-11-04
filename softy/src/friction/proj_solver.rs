#![allow(dead_code)]
use super::{FrictionParams, FrictionSolveResult};
use crate::contact::*;
use utils::soap::*;

use utils::zip;

/// Friction solver.
pub struct FrictionSolver<'a> {
    /// A set of tangential momenta in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    predictor_impulse: Chunked3<&'a [f64]>,
    prev_friction_impulse_t: Chunked2<&'a [f64]>,
    /// A set of contact forces for each contact point.
    contact_impulse: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Friction coefficient.
    mu: f64,
    mass_inv_mtx: EffectiveMassInvView<'a>,
}

impl<'a> FrictionSolver<'a> {
    pub fn new(
        predictor_impulse: &'a [[f64; 3]],
        prev_friction_impulse_t: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
    ) -> Result<FrictionSolver<'a>, ()> {
        Ok(FrictionSolver {
            predictor_impulse: Chunked3::from_array_slice(predictor_impulse),
            prev_friction_impulse_t: Chunked2::from_array_slice(prev_friction_impulse_t),
            contact_impulse,
            contact_basis,
            mu: params.dynamic_friction,
            mass_inv_mtx,
        })
    }

    /// Solve one step.
    pub fn step(&mut self) -> Result<FrictionSolveResult, ()> {
        // Solve quadratic optimization problem
        let mut friction_impulse =
            Chunked2::from_array_vec(vec![[0.0; 2]; self.predictor_impulse.len()]);
        //let predictor_impulse = (self.mass_inv_mtx.view() * Tensor::new(self.predictor_impulse)).into_inner();
        let predictor_impulse_t: Vec<_> = self
            .contact_basis
            .to_tangent_space(self.predictor_impulse.view().into())
            .collect();
        for (r, &pred_p, &prev_r, &cr) in zip!(
            friction_impulse.iter_mut(),
            predictor_impulse_t.iter(),
            self.prev_friction_impulse_t.iter(),
            self.contact_impulse.iter()
        ) {
            // This needs to incorporate effective mass.
            let rc = Tensor::new(pred_p); // Impulse candidate

            // Project onto the unit circle.
            let radius = self.mu * cr.abs();
            if rc.dot(rc) > radius * radius {
                *r = ((rc - Tensor::new(prev_r)) * (radius / rc.norm())).into_inner();
            } else {
                *r = (rc - Tensor::new(prev_r)).into_inner();
            }
        }

        Ok(FrictionSolveResult {
            objective_value: 0.0, // Skipping this.
            solution: friction_impulse.into(),
        })
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
        assert!(velocity[1] > 0.8);

        // Sanity check that no impulses were produced in the process
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[0], 0.0, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn sticking_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(1.5, mass)?;
        // Check that the point gets stuck
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6, epsilon = 1e-8);

        // Sanity check that no perpendicular impulses were produced in the process
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[0], 0.0, max_relative = 1e-6);
        Ok(())
    }

    fn sliding_point_tester(mu: f64, mass: f64) -> Result<(Vector2<f64>, Vector2<f64>), Error> {
        let params = FrictionParams {
            dynamic_friction: mu,
            inner_iterations: 30,
            tolerance: 1e-10,
            print_level: 0,
        };

        let prev_friction_impulse_t = vec![[0.0, 0.0]];
        let predictor_impulse = vec![[-1.0 * mass, 0.0, 0.0]]; // one point sliding right.
        let contact_impulse = vec![10.0 * mass];
        let mass_inv_mtx: DSBlockMatrix3 =
            DiagonalBlockMatrix::new(Chunked3::from_flat(vec![1.0 / mass; 3])).into();

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = FrictionSolver::new(
            &predictor_impulse,
            &prev_friction_impulse_t,
            &contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        )
        .unwrap();
        let result = solver.step().unwrap();

        let impulse = Vector2::new(result.solution[0]);
        let p_imp_t: Vec<_> = contact_basis.to_tangent_space(&predictor_impulse).collect();
        let final_velocity = (impulse - Vector2::new(p_imp_t[0])) / mass;

        dbg!(&p_imp_t);
        dbg!(&result.solution);
        dbg!(&final_velocity);

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

        let contact_impulse = vec![-0.0000000018048827573828247, -0.00003259055555607145];
        let masses = vec![0.0003720701030949866, 0.0003720701030949866];
        let prev_friction_impulse_t = vec![[0.0, 0.0], [0.0, 0.0]];
        let predictor_impulse = vec![
            [
                0.0,
                -0.07225747944670913 * masses[0],
                -0.0000001280108566301736 * masses[0],
            ],
            [
                0.0,
                -0.06185827187696774 * masses[1],
                0.0060040275393186595 * masses[1],
            ],
        ]; // tet vertex momenta

        let mass_inv_mtx: DSBlockMatrix3 =
            DiagonalBlockMatrix::new(Chunked3::from_array_vec(vec![
                [1.0 / masses[0]; 3],
                [1.0 / masses[1]; 3],
            ]))
            .into();

        let mut contact_basis = ContactBasis::new();
        let normals = vec![
            [-0.0, -0.7071067811865476, -0.7071067811865476],
            [-0.0, -0.7071067811865476, -0.7071067811865476],
        ];
        contact_basis.update_from_normals(normals);

        let mut solver = FrictionSolver::new(
            &predictor_impulse,
            &prev_friction_impulse_t,
            &contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        )
        .unwrap();
        let result = solver.step().unwrap();

        let p_imp_t: Vec<_> = contact_basis.to_tangent_space(&predictor_impulse).collect();
        let final_momentum: Vec<_> = zip!(p_imp_t.iter(), result.solution.iter())
            .map(|(&pred_p, &r)| (Vector2::new(r) - Vector2::new(pred_p)))
            .collect();

        dbg!(&p_imp_t);
        dbg!(&result.solution);
        dbg!(&final_momentum);

        // Check that the tet continues with sligtly less momentum
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    final_momentum[i][j],
                    -p_imp_t[i][j],
                    max_relative = 1e-2,
                    epsilon = 1e-5
                );
                if j == 0 {
                    assert!(final_momentum[i][j].abs() < p_imp_t[i][j].abs());
                }
            }
        }
        Ok(())
    }
}
