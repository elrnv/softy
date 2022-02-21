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
    //BDF2,
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
    pub jacobian_test: bool,
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
    pub friction_force: Duration,
}

impl ResidualTimings {
    pub fn clear(&mut self) {
        self.total = Duration::new(0, 0);
        self.energy_gradient = Duration::new(0, 0);
        self.prepare_contact = Duration::new(0, 0);
        self.contact_force = Duration::new(0, 0);
        self.friction_force = Duration::new(0, 0);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Timings {
    line_search_assist: Duration,
    residual: ResidualTimings,
    linear_solve: Duration,
    jacobian_product: Duration,
    line_search: Duration,
    total: Duration,
}

impl Display for Timings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Line search assist time: {}ms",
            self.line_search_assist.as_millis()
        )?;
        writeln!(
            f,
            "Balance equation computation time: {}ms",
            self.residual.total.as_millis()
        )?;
        writeln!(
            f,
            "   Energy gradient time: {}ms",
            self.residual.energy_gradient.as_millis()
        )?;
        writeln!(
            f,
            "   Contact prep time: {}ms",
            self.residual.prepare_contact.as_millis()
        )?;
        writeln!(
            f,
            "   Contact force time: {}ms",
            self.residual.contact_force.as_millis()
        )?;
        writeln!(
            f,
            "   Friction force time: {}ms",
            self.residual.friction_force.as_millis()
        )?;
        writeln!(f, "Linear solve time: {}ms", self.linear_solve.as_millis())?;
        writeln!(
            f,
            "   Jacobian product time: {}ms",
            self.jacobian_product.as_millis()
        )?;
        writeln!(f, "Line search time: {}ms", self.line_search.as_millis())?;
        writeln!(f, "Total solve time {}ms", self.total.as_millis())
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
