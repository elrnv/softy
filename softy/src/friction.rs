use crate::contact::*;
use geo::math::{Matrix3, Vector3};
use ipopt::{self, Index, Ipopt, Number, SolverData, SolverDataMut};
use reinterpret::*;

use approx::*;

use crate::inf_norm;
use crate::mask_iter::*;

use crate::{Error, PointCloud, PolyMesh, TetMesh, TriMesh};

/// Result from one inner friction step.
#[derive(Clone, Debug, PartialEq)]
pub struct FrictionSolveResult {
    /// The value of the dissipation objective at the end of the step.
    pub objective_value: f64,
}

/// Friction solver.
pub struct FrictionSolver<'v> {
    /// Non-linear solver.
    solver: Ipopt<FrictionProblem<'v>>,
}

impl<'v> FrictionSolver<'v> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point.
    pub fn new(velocity: &'v [[f64; 2]]) -> Result<FrictionSolver<'v>, Error> {
        let problem = FrictionProblem {
            velocity,
        };

        let mut ipopt = Ipopt::new(problem)?;
        ipopt.set_option("print_level", 5);

        Ok(FrictionSolver { solver: ipopt })
    }

    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &'v FrictionProblem {
        self.solver.solver_data().problem
    }

    /// Get a mutable reference to the underlying problem.
    fn problem_mut(&mut self) -> &'v mut FrictionProblem {
        self.solver.solver_data_mut().problem
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

        let result = FrictionSolveResult { objective_value };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::FrictionSolveError(e, result)),
        }
    }
}

pub(crate) struct FrictionProblem<'v> {
    /// A set of tangential velocities in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    velocity: &'v [[f64; 2]],
    /// A set of contact forces for each contact point.
    contact_force: &'v [f64],
    /// Friction coefficient.
    mu: f64,
}

impl FrictionProblem<'_> {
    fn num_contacts(&self) -> usize {
        velocity.len()
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

    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.iter_mut().for_each(|x| *x = 0.0);
        true
    }

    fn objective(&self, f: &[Number], obj: &mut Number) -> bool {
        let forces: &[[f64; 2]] = reinterpret_slice(f);
        assert_eq!(self.velocity.len(), forces.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &f) in self.velocity.iter().zip(forces.iter()) {
            *obj += Vector3(v).dot(Vector3(f))
        }

        true
    }

    fn objective_grad(&self, _f: &[Number], grad_f: &mut [Number]) -> bool {
        let velocity_values: &[f64] = reinterpret_slice(self.velocity);
        assert_eq!(velocity_values.len(), grad_f.len());
        
        for (g, v) in grad_f.iter_mut().zip(velocity_values) {
            *g += v;
        }

        true
    }
}

impl ipopt::ConstrainedProblem for FrictionProblem<'_> {
    fn num_constraints(&self) -> usize {
        1
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {

    }

    fn constraint(&self, dx: &[Number], g: &mut [Number]) -> bool {

    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {

    }

    fn constraint_jacobian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {

    }

    fn constraint_jacobian_values(&self, dx: &[Number], vals: &mut [Number]) -> bool {

    }

    // Hessian is zero.
    fn num_hessian_non_zeros(&self) -> usize {
        0
    }
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        true
    }
    fn hessian_values(&self, x: &[Number], obj_vactor: Number, lambda: &[Number], vals: &mut [Number]) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
