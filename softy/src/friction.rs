use crate::contact::*;
use geo::math::{Matrix3, Vector3};
use ipopt::{self, Ipopt, SolverData, SolverDataMut, Number, Index};
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
pub struct FrictionSolver {
    /// Non-linear solver.
    solver: Ipopt<FrictionProblem>,
}

impl FrictionSolver {
    /// Build a new solver for the friction problem.
    pub fn new() -> Result<FrictionSolver, Error> {
        let problem = FrictionProblem {
        };

        let mut ipopt = Ipopt::new_newton(problem)?;
        ipopt.set_option("print_level", 5);

        Ok(FrictionSolver {
            solver: ipopt,
        })
    }

    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &FrictionProblem {
        self.solver.solver_data().problem
    }

    /// Get a mutable reference to the underlying problem.
    fn problem_mut(&mut self) -> &mut FrictionProblem {
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

        let result = FrictionSolveResult {
            objective_value,
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::FrictionSolveError(e, result)),
        }
    }
}

pub(crate) struct FrictionProblem {
}

impl FrictionProblem {
    // TODO: implement this
    fn num_contacts(&self) -> usize {
        0
    }
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for FrictionProblem {
    fn num_variables(&self) -> usize {
        3 * self.num_contacts()
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

    fn objective(&self, dx: &[Number], obj: &mut Number) -> bool {
        // TODO: Implement this
        *obj = 0.0;
        true
    }

    fn objective_grad(&self, dx: &[Number], grad_f: &mut [Number]) -> bool {
        // TODO: implement this
        true
    }
}

impl ipopt::NewtonProblem for FrictionProblem {
    fn num_hessian_non_zeros(&self) -> usize {
        0
    }
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // TODO: Implement this
        false
    }
    fn hessian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        // TODO: Implement this
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
