//! A module for FEM solvers for non-linear equations.

pub mod linsolve;
pub mod mcp;
pub mod newton;
pub mod problem;
pub mod solver;
pub mod state;
pub mod trust_region;

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt::{Display, Formatter};
use std::time::Duration;
use thiserror::Error;

use crate::constraints::{FrictionJacobianTimings, FrictionTimings};
pub use mcp::*;
pub use newton::*;
pub use problem::*;
pub use solver::*;
pub use trust_region::*;

/// Time integration method.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TimeIntegration {
    /// Backward Euler integration.
    BE,
    /// Trapezoid Rule integration.
    TR,
    ///// Second order Backward Differentiation Formula.
    BDF2,
    ///// TR and BDF2 mixed method.
    //TRBDF2,
}

impl Default for TimeIntegration {
    fn default() -> Self {
        TimeIntegration::BE
    }
}

/// Simulation parameters.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimParams {
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    /// Clear the velocity between steps with frequency given in Hz.
    ///
    /// This can emulate quasi-static simulation (or critical damping)
    /// where the time step and density effectively determines the
    /// regularization.
    pub velocity_clear_frequency: f32,
    pub residual_tolerance: Option<f32>,
    pub acceleration_tolerance: Option<f32>,
    pub velocity_tolerance: Option<f32>,
    pub max_iterations: u32,
    pub linsolve: LinearSolver,
    pub line_search: LineSearch,
    /// Test that the problem Jacobian is correct.
    pub derivative_test: u8,
    /// The velocity error tolerance for sticking between objects.
    pub friction_tolerance: f32,
    /// The distance tolerance between objects in contact.
    pub contact_tolerance: f32,
    /// Number of contact iterations.
    pub contact_iterations: u32,
    pub time_integration: TimeIntegration,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Missing tolerance parameter 'tol'")]
    MissingTolerance,
    #[error("Missing maximum number of iterations parameter 'max_iter'")]
    MissingMaximumIterations,
}

pub struct CallbackArgs<'a, T>
where
    T: 'static,
{
    pub iteration: u32,
    pub residual: &'a [T],
    pub x: &'a [T],
    pub problem: &'a mut dyn NonLinearProblem<T>,
}

pub type Callback<T> = Box<dyn FnMut(CallbackArgs<T>) -> bool + Send + 'static>;

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// Solve was successful.
    Success,
    /// The number of iterations exceeded maximum before stopping criterion was satisfied.
    MaximumIterationsExceeded,
    /// The number of contact iterations exceeded the maximum before stopping criterion was satisfied.
    MaximumContactIterationsExceeded,
    /// A problem with the linear solve occurred.
    ///
    /// This is typically a conditioning or invertibility issue.
    LinearSolveError(SparseSolveError),
    Diverged,
    StepTooLarge,
    NothingToSolve,
    Interrupted,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ResidualTimings {
    pub total: Duration,
    pub energy_gradient: Duration,
    pub prepare_contact: Duration,
    pub contact_force: Duration,
    pub contact_jacobian: Duration,
    pub friction_force: FrictionTimings,
}

impl ResidualTimings {
    pub fn clear(&mut self) {
        self.total = Duration::new(0, 0);
        self.energy_gradient = Duration::new(0, 0);
        self.prepare_contact = Duration::new(0, 0);
        self.contact_force = Duration::new(0, 0);
        self.contact_jacobian = Duration::new(0, 0);
        self.friction_force.clear();
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Timings {
    line_search_assist: Duration,
    residual: ResidualTimings,
    friction_jacobian: FrictionJacobianTimings,
    linear_solve: Duration,
    direct_solve: Duration,
    jacobian_product: Duration,
    jacobian_values: Duration,
    linsolve_debug_info: Duration,
    line_search: Duration,
    total: Duration,
}

impl Display for Timings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Timings (ms):")?;
        writeln!(
            f,
            "  Line search assist time:    {}",
            self.line_search_assist.as_millis()
        )?;
        writeln!(
            f,
            "  Balance equation time:      {}",
            self.residual.total.as_millis()
        )?;
        writeln!(
            f,
            "    Energy gradient time:     {}",
            self.residual.energy_gradient.as_millis()
        )?;
        writeln!(
            f,
            "    Contact prep time:        {}",
            self.residual.prepare_contact.as_millis()
        )?;
        writeln!(
            f,
            "    Contact force time:       {}",
            self.residual.contact_force.as_millis()
        )?;
        writeln!(
            f,
            "    Friction force time:      {}",
            self.residual.friction_force.total.as_millis()
        )?;
        writeln!(
            f,
            "      Jac + basis mul time:   {}",
            self.residual.friction_force.jac_basis_mul.as_millis()
        )?;
        writeln!(
            f,
            "      Contact vel prep time:  {}",
            self.residual
                .friction_force
                .contact_velocity
                .prep
                .as_millis()
        )?;
        writeln!(
            f,
            "      Contact Jacobian time:  {}",
            self.residual
                .friction_force
                .contact_velocity
                .contact_jac
                .as_millis()
        )?;
        writeln!(
            f,
            "      Contact velocity time:  {}",
            self.residual
                .friction_force
                .contact_velocity
                .velocity
                .as_millis()
        )?;
        writeln!(
            f,
            "  Linear solve time:          {}",
            self.linear_solve.as_millis()
        )?;
        writeln!(
            f,
            "    Jacobian product time:    {}",
            self.jacobian_product.as_millis()
        )?;
        writeln!(
            f,
            "    Jacobian values time:     {}",
            self.jacobian_values.as_millis()
        )?;
        writeln!(
            f,
            "      Friction constraint F:  {}",
            self.friction_jacobian.constraint_friction_force.as_millis()
        )?;
        writeln!(
            f,
            "      Friction contact J:     {}",
            self.friction_jacobian.contact_jacobian.as_millis()
        )?;
        writeln!(
            f,
            "      Friction contact G:     {}",
            self.friction_jacobian.contact_gradient.as_millis()
        )?;
        writeln!(
            f,
            "      Friction constraint J:  {}",
            self.friction_jacobian.constraint_jacobian.as_millis()
        )?;
        writeln!(
            f,
            "      Friction constraint G:  {}",
            self.friction_jacobian.constraint_gradient.as_millis()
        )?;
        writeln!(
            f,
            "      Friction f lambda jac:  {}",
            self.friction_jacobian.f_lambda_jac.as_millis()
        )?;
        writeln!(
            f,
            "      Friction A:             {}",
            self.friction_jacobian.a.as_millis()
        )?;
        writeln!(
            f,
            "      Friction B:             {}",
            self.friction_jacobian.b.as_millis()
        )?;
        writeln!(
            f,
            "      Friction C:             {}",
            self.friction_jacobian.c.as_millis()
        )?;
        writeln!(
            f,
            "      Friction D Half:        {}",
            self.friction_jacobian.d_half.as_millis()
        )?;
        writeln!(
            f,
            "      Friction D:             {}",
            self.friction_jacobian.d.as_millis()
        )?;
        writeln!(
            f,
            "      Friction E:             {}",
            self.friction_jacobian.e.as_millis()
        )?;
        writeln!(
            f,
            "    Direct solve time:        {}",
            self.direct_solve.as_millis()
        )?;
        writeln!(
            f,
            "    Debug info time:          {}",
            self.linsolve_debug_info.as_millis()
        )?;
        writeln!(
            f,
            "  Line search time:           {}",
            self.line_search.as_millis()
        )?;
        writeln!(
            f,
            "  Total solve time            {}",
            self.total.as_millis()
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult {
    /// Number of successful iterations.
    pub iterations: u32,
    /// Solve status.
    pub status: Status,
    /// Timing results.
    pub timings: Timings,
    /// Iteration info.
    pub stats: Vec<IterationInfo>,
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = IterationInfo::header();
        writeln!(f, "{}", header[0])?;
        writeln!(f, "{}", header[1])?;
        for info in self.stats.iter() {
            writeln!(f, "{}", info)?;
        }
        writeln!(f, "Status:           {:?}", self.status)?;
        writeln!(f, "Total iterations: {}", self.iterations)?;
        writeln!(f, "Timings:\n{}", self.timings)
    }
}

pub trait NLSolver<P, T> {
    /// Gets a reference to the outer callback function.
    fn outer_callback(&self) -> &RefCell<Callback<T>>;
    /// Gets a reference to the inner callback function.
    fn inner_callback(&self) -> &RefCell<Callback<T>>;
    /// Gets a reference to the underlying problem instance.
    fn problem(&self) -> &P;
    /// Gets a mutable reference the underlying problem instance.
    fn problem_mut(&mut self) -> &mut P;
    /// Updates the constraint set which may change the jacobian sparsity.
    fn update_jacobian_indices(&mut self);
    /// Solves the problem and returns the solution along with the solve result
    /// info.
    fn solve(&mut self) -> (Vec<T>, SolveResult);
    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&mut self, x: &mut [T], update_jacobian_indices: bool) -> SolveResult;
}
