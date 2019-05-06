use super::FrictionParams;
use geo::math::{Matrix2, Vector2};
use ipopt::{self, Index, Ipopt, Number};
use reinterpret::*;

use unroll::unroll_for_loops;
use utils::zip;

use crate::Error;

/// Result from one inner friction step.
#[derive(Clone, Debug, PartialEq)]
pub struct FrictionSolveResult {
    /// The value of the dissipation objective at the end of the step.
    pub objective_value: f64,
    /// Resultant friction force in contact space.
    pub friction_force: Vec<[f64; 2]>,
}

/// Friction solver.
pub struct FrictionSolver<'a> {
    /// Non-linear solver.
    solver: Ipopt<FrictionProblem<'a>>,
}

impl<'a> FrictionSolver<'a> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_force` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [[f64; 2]],
        contact_force: &'a [f64],
        params: FrictionParams,
    ) -> Result<FrictionSolver<'a>, Error> {
        let problem = FrictionProblem {
            velocity: reinterpret_slice(velocity),
            contact_force,
            mu: params.dynamic_friction,
        };

        let mut ipopt = Ipopt::new(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        ipopt.set_option("nlp_scaling_max_gradient", 1e-3);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("max_iter", params.inner_iterations as i32);

        Ok(FrictionSolver { solver: ipopt })
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

pub(crate) struct FrictionProblem<'a> {
    /// A set of tangential velocities in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    velocity: &'a [Vector2<f64>],
    /// A set of contact forces for each contact point.
    contact_force: &'a [f64],
    /// Friction coefficient.
    mu: f64,
}

impl FrictionProblem<'_> {
    fn num_contacts(&self) -> usize {
        self.velocity.len()
    }
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for FrictionProblem<'_> {
    fn num_variables(&self) -> usize {
        2 * self.num_contacts()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        let bound = 2e19;
        x_l.iter_mut().for_each(|x| *x = -bound);
        x_u.iter_mut().for_each(|x| *x = bound);
        true
    }

    fn initial_point(&self, f: &mut [Number]) -> bool {
        let forces: &mut [Vector2<f64>] = reinterpret_mut_slice(f);
        for (f, &cf, &v) in zip!(
            forces.iter_mut(),
            self.contact_force.iter(),
            self.velocity.iter()
        ) {
            let v_norm = v.norm();
            if v_norm > 0.0 {
                *f = v * ((-self.mu * cf.abs()) / v_norm)
            } else {
                *f = Vector2::zeros();
            }
        }
        true
    }

    fn objective(&self, f: &[Number], obj: &mut Number) -> bool {
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(self.velocity.len(), forces.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &f) in self.velocity.iter().zip(forces.iter()) {
            *obj += v.dot(f)
        }

        true
    }

    fn objective_grad(&self, _f: &[Number], grad_f: &mut [Number]) -> bool {
        let velocity_values: &[f64] = reinterpret_slice(self.velocity);
        assert_eq!(velocity_values.len(), grad_f.len());

        for g in grad_f.iter_mut() {
            *g = 0.0;
        }

        for (g, v) in grad_f.iter_mut().zip(velocity_values) {
            *g += v;
        }

        true
    }
}

impl ipopt::ConstrainedProblem for FrictionProblem<'_> {
    fn num_constraints(&self) -> usize {
        self.contact_force.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2 * self.num_constraints()
    }

    fn constraint(&self, f: &[Number], g: &mut [Number]) -> bool {
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(forces.len(), g.len());
        for (c, f) in g.iter_mut().zip(forces.iter()) {
            *c = f.norm()
        }
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        for ((l, u), &cf) in g_l
            .iter_mut()
            .zip(g_u.iter_mut())
            .zip(self.contact_force.iter())
        {
            *l = -2e19; // norm can never be negative, so leave this unconstrained.
            *u = self.mu * cf.abs();
        }
        true
    }

    #[unroll_for_loops]
    fn constraint_jacobian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        for constraint_idx in 0..self.num_constraints() {
            for j in 0..2 {
                rows[2 * constraint_idx + j] = constraint_idx as Index;
                cols[2 * constraint_idx + j] = (2 * constraint_idx + j) as Index;
            }
        }
        true
    }

    fn constraint_jacobian_values(&self, f: &[Number], vals: &mut [Number]) -> bool {
        let jacobian: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        for (jac, &f) in jacobian.iter_mut().zip(forces.iter()) {
            let f_norm = f.norm();
            *jac = Vector2::zeros();
            if f_norm > 0.0 {
                *jac = f / f_norm;
            }
        }
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        // Objective Hessian is zero.
        // Constraint hessian is block diagonal. (lower triangular part only)
        3 * self.num_constraints()
    }

    #[unroll_for_loops]
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut counter = 0;
        for i in 0..self.num_constraints() {
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
    fn hessian_values(
        &self,
        f: &[Number],
        _obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let hess_vals: &mut [[f64; 3]] = reinterpret_mut_slice(vals);
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(forces.len(), lambda.len());
        assert_eq!(hess_vals.len(), lambda.len());
        for ((h, &f), &l) in hess_vals.iter_mut().zip(forces.iter()).zip(lambda.iter()) {
            let f_norm = f.norm();
            *h = [0.0; 3];
            if f_norm > 0.0 {
                let f_norm_inv = 1.0 / f_norm;
                let f_unit = f * f_norm_inv;
                let hess = (Matrix2::identity() - f_unit * f_unit.transpose()) * (f_norm_inv * l);
                *h = [hess[0][0], hess[0][1], hess[1][1]];
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let params = FrictionParams {
            dynamic_friction: 1.5,
            inner_iterations: 30,
            tolerance: 1e-5,
            print_level: 0,
        };
        let mass = 1.0;

        let velocity = vec![[1.0, 0.0]]; // one point sliding right.
        let contact_force = vec![10.0 * mass];

        let mut solver = FrictionSolver::new(&velocity, &contact_force, params)?;
        let result = solver.step()?;
        let FrictionSolveResult {
            friction_force,
            objective_value,
        } = result;

        assert_relative_eq!(friction_force[0][0], -15.0, max_relative = 1e-6);
        assert_relative_eq!(friction_force[0][1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(objective_value, -15.0, max_relative = 1e-6);

        Ok(())
    }
}
