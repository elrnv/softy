use super::solver::FrictionSolveResult; // Reuse the result from the xy solver.
use super::FrictionParams;
use geo::Real;
use ipopt::{self, Index, Ipopt, Number};
use reinterpret::*;

use unroll::unroll_for_loops;
use utils::zip;

use crate::Error;

/// A two dimensional vector in polar coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Polar2<T: Real> {
    pub radius: T,
    pub angle: T,
}

/// Friction solver.
pub struct FrictionPolarSolver<'a> {
    /// Non-linear solver.
    solver: Ipopt<FrictionPolarProblem<'a>>,
}

impl<'a> FrictionPolarSolver<'a> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_force` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [[f64; 2]],
        contact_force: &'a [f64],
        params: FrictionParams,
    ) -> Result<FrictionPolarSolver<'a>, Error> {
        let problem = FrictionPolarProblem {
            velocity: reinterpret_slice(velocity),
            contact_force,
            mu: params.dynamic_friction,
        };

        let mut ipopt = Ipopt::new_newton(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        ipopt.set_option("nlp_scaling_max_gradient", 1e-3);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("max_iter", params.inner_iterations as i32);

        Ok(FrictionPolarSolver { solver: ipopt })
    }

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn step(&mut self) -> Result<FrictionSolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            objective_value,
            status,
            ..
        } = self.solver.solve();

        let result = FrictionSolveResult {
            objective_value,
            friction_force: reinterpret_vec(solver_data.solution.primal_variables.to_vec()),
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::FrictionSolveError(e)),
        }
    }
}

pub(crate) struct FrictionPolarProblem<'a> {
    /// A set of tangential velocities in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    velocity: &'a [Polar2<f64>],
    /// A set of contact forces for each contact point.
    contact_force: &'a [f64],
    /// Friction coefficient.
    mu: f64,
}

impl FrictionPolarProblem<'_> {
    fn num_contacts(&self) -> usize {
        self.velocity.len()
    }
}

/// Return an angle in [-PI, PI] at 90 degrees to the given angle. The input angle is given in
/// radians.
fn negate_angle(angle: f64) -> f64 {
    normalize_angle(angle + std::f64::consts::PI)
}

/// Given an angle in radians, translate it to the canonical [-PI, PI] range. We choose this range
/// because this what atan2 maps to. This function doesn't distinguish between -PI and PI, so the
/// normalized angles overlap at PI and -PI.
fn normalize_angle(angle: f64) -> f64 {
    use std::f64::consts::PI;
    (angle + PI) % (2.0 * PI) + if angle < 0.0 { PI } else { -PI }
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for FrictionPolarProblem<'_> {
    fn num_variables(&self) -> usize {
        2 * self.num_contacts()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        let x_l: &mut [Polar2<f64>] = reinterpret_mut_slice(x_l);
        let x_u: &mut [Polar2<f64>] = reinterpret_mut_slice(x_u);
        for (l, u, &cf) in zip!(x_l.iter_mut(), x_u.iter_mut(), self.contact_force.iter()) {
            // First coordinate is the radius and second is the angle.
            l.radius = 0.0; // radius is never negative
            u.radius = self.mu * cf.abs();
            // Angle is unconstrained.
            l.angle = -2e19;
            u.angle = 2e19;
        }
        true
    }

    fn initial_point(&self, f: &mut [Number]) -> bool {
        let forces: &mut [Polar2<f64>] = reinterpret_mut_slice(f);
        for (f, &cf, &v) in zip!(
            forces.iter_mut(),
            self.contact_force.iter(),
            self.velocity.iter()
        ) {
            if v.radius > 0.0 {
                f.radius = self.mu * cf.abs();
                f.angle = negate_angle(v.angle);
            } else {
                *f = Polar2 {
                    radius: 0.0,
                    angle: 0.0,
                };
            }
        }
        true
    }

    fn objective(&self, f: &[Number], obj: &mut Number) -> bool {
        let forces: &[Polar2<f64>] = reinterpret_slice(f);
        assert_eq!(self.velocity.len(), forces.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &f) in self.velocity.iter().zip(forces.iter()) {
            *obj += f.radius * f64::cos(f.angle - v.angle);
        }

        true
    }

    fn objective_grad(&self, f: &[Number], grad_f: &mut [Number]) -> bool {
        let forces: &[Polar2<Number>] = reinterpret_slice(f);
        let grad_f: &mut [Polar2<Number>] = reinterpret_mut_slice(grad_f);
        assert_eq!(self.velocity.len(), grad_f.len());

        for g in grad_f.iter_mut() {
            *g = Polar2 {
                radius: 0.0,
                angle: 0.0,
            };
        }

        for (g, &v, &f) in zip!(grad_f.iter_mut(), self.velocity.iter(), forces.iter()) {
            g.radius += f64::cos(f.angle - v.angle);
            g.angle -= f.radius * f64::sin(f.angle - v.angle);
        }

        true
    }
}

impl ipopt::NewtonProblem for FrictionPolarProblem<'_> {
    fn num_hessian_non_zeros(&self) -> usize {
        // Objective hessian is block diagonal. (lower triangular part only)
        3 * self.num_contacts()
    }

    #[unroll_for_loops]
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut counter = 0;
        for i in 0..self.num_contacts() {
            // Constraint Hessian (only interested in lower triangular part
            for c in 0..2 {
                for r in c..2 {
                    rows[counter] = (2 * i + r) as Index;
                    cols[counter] = (2 * i + c) as Index;
                    counter += 1;
                }
            }
        }
        true
    }
    fn hessian_values(&self, f: &[Number], vals: &mut [Number]) -> bool {
        let hess_vals: &mut [[Number; 3]] = reinterpret_mut_slice(vals);
        let forces: &[Polar2<Number>] = reinterpret_slice(f);
        for ((h, &f), &v) in hess_vals
            .iter_mut()
            .zip(forces.iter())
            .zip(self.velocity.iter())
        {
            *h = [0.0; 3];
            if f.radius.abs() > 0.0 {
                *h = [
                    0.0,
                    -f64::sin(f.angle - v.angle),
                    -f.radius * f64::cos(f.angle - v.angle),
                ];
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    //use crate::test_utils::*;
    //use crate::*;
    //use std::path::PathBuf;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        use std::f64::consts::PI;
        let params = FrictionParams {
            dynamic_friction: 1.5,
            inner_iterations: 30,
            tolerance: 1e-5,
            print_level: 5,
        };
        let mass = 1.0;

        let velocity = vec![[1.0, PI]]; // one point sliding up.
        let contact_force = vec![10.0 * mass];

        let mut solver = FrictionPolarSolver::new(&velocity, &contact_force, params)?;
        let result = solver.step()?;
        let FrictionSolveResult {
            friction_force,
            objective_value,
        } = result;

        assert_relative_eq!(friction_force[0][0], 15.0, max_relative = 1e-5);
        assert_relative_eq!(
            friction_force[0][1],
            0.0,
            max_relative = 1e-5,
            epsilon = 1e-8
        );
        assert_relative_eq!(objective_value, -15.0, max_relative = 1e-5);

        Ok(())
    }

    #[test]
    fn normalize_angle_test() {
        use std::f64::consts::PI;
        let inputs = vec![
            6.0 * PI,
            5.0 * PI,
            -5.0 * PI,
            3.0 * PI / 4.0,
            9.0 * PI / 4.0,
            3.0 * PI / 2.0,
            9.0 * PI / 2.0,
            10.0 * PI / 3.0,
            -9.0 * PI / 4.0,
            -9.0 * PI / 2.0,
            -10.0 * PI / 3.0,
        ];
        let expected = vec![
            0.0,
            -PI,
            PI,
            3.0 * PI / 4.0,
            PI / 4.0,
            -PI / 2.0,
            PI / 2.0,
            -2.0 * PI / 3.0,
            -PI / 4.0,
            -PI / 2.0,
            2.0 * PI / 3.0,
        ];

        for (i, o) in inputs.into_iter().zip(expected.into_iter()) {
            assert_relative_eq!(normalize_angle(i), o, max_relative = 1e-5, epsilon = 1e-8);
        }
    }

    #[test]
    fn negate_angle_test() {
        use std::f64::consts::PI;
        let inputs = vec![PI, 0.0, PI / 3.0, -PI / 4.0, 2.0 * PI / 3.0];
        let expected = vec![0.0, -PI, -2.0 * PI / 3.0, 3.0 * PI / 4.0, -PI / 3.0];

        for (i, o) in inputs.into_iter().zip(expected.into_iter()) {
            assert_relative_eq!(negate_angle(i), o, max_relative = 1e-5, epsilon = 1e-8);
        }
    }

    ///// Pinch a box between two probes.
    ///// Given sufficient friction, the box should not fall.
    //fn pinch_tester(
    //    sc_params: SmoothContactParams,
    //) -> Result<(), Error> {
    //    use geo::mesh::attrib::Attrib;
    //    use geo::mesh::VertexPositions;
    //    use geo::mesh::topology::*;

    //    let params = SimParams {
    //        max_iterations: 200,
    //        max_outer_iterations: 20,
    //        gravity: [0.0f32, -9.81, 0.0],
    //        time_step: Some(0.01),
    //        print_level: 5,
    //        friction_iterations: 1,
    //        ..DYNAMIC_PARAMS
    //    };

    //    let material = Material {
    //        elasticity: ElasticityParameters::from_young_poisson(1e6, 0.45),
    //        ..SOLID_MATERIAL
    //    };

    //    let clamps = geo::io::load_polymesh(&PathBuf::from("assets/clamps.vtk"))?;
    //    let mut box_mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk"))?;
    //    box_mesh.remove_attrib::<VertexIndex>("fixed")?;

    //    let mut solver = fem::SolverBuilder::new(params.clone())
    //        .solid_material(material)
    //        .add_solid(box_mesh)
    //        .add_shell(clamps)
    //        .smooth_contact_params(sc_params)
    //        .build()?;

    //    for _ in 0..50 {
    //        let res = solver.step()?;
    //
    //        //println!("res = {:?}", res);
    //        assert!(
    //            res.iterations <= params.max_outer_iterations,
    //            "Exceeded max outer iterations."
    //        );

    //        // Check that the mesh hasn't fallen.
    //        let tetmesh = solver.borrow_mesh();
    //
    //        for v in tetmesh.vertex_position_iter() {
    //            assert!(v[1] > -0.6);
    //        }
    //    }

    //    Ok(())
    //}

    ///// Pinch a box against a couple of implicit surfaces.
    //#[test]
    //fn pinch_against_implicit() -> Result<(), Error> {
    //    let sc_params = SmoothContactParams {
    //        contact_type: ContactType::Point,
    //        kernel: KernelType::Cubic {
    //            radius_multiplier: 1.5,
    //        },
    //        friction_params: Some(FrictionParams {
    //            dynamic_friction: 0.4,
    //            inner_iterations: 40,
    //            tolerance: 1e-5,
    //            print_level: 5,
    //        }),
    //    };

    //    pinch_tester(sc_params)
    //}
}
