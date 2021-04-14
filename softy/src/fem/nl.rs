//! A module for FEM solvers for non-linear equations.

pub mod mcp;
pub mod newton;
pub mod problem;
pub mod solver;
pub mod trust_region;

use std::cell::RefCell;
use thiserror::Error;

pub use mcp::*;
pub use newton::*;
pub use problem::*;
pub use solver::*;
pub use trust_region::*;

/// Simulation parameters.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimParams {
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    /// Clear the velocity between consecutive steps to emulate quasi-static simulation (or
    /// critical damping) where the time step and density effectively determines the
    /// regularization.
    pub clear_velocity: bool,
    pub tolerance: f32,
    pub max_iterations: u32,
    /// Test that the problem Jacobian is correct.
    pub jacobian_test: bool,
    pub line_search: LineSearch,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Missing tolerance parameter 'tol'")]
    MissingTolerance,
    #[error("Missing maximum number of iterations parameter 'max_iter'")]
    MissingMaximumIterations,
}

pub struct CallbackArgs<'a, T> {
    pub residual: &'a [T],
}

pub type Callback<T> = Box<dyn FnMut(CallbackArgs<T>) -> bool + Send + 'static>;

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Success,
    MaximumIterationsExceeded,
    LinearSolveError,
    Interrupted,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult {
    /// Number of successful iterations.
    pub iterations: u32,
    /// Solve status.
    pub status: Status,
}

pub trait NLSolver<P, T> {
    /// Gets a reference to the intermediate callback function.
    fn intermediate_callback(&self) -> &RefCell<Callback<T>>;
    /// Gets a reference to the underlying problem instance.
    fn problem(&self) -> &P;
    /// Gets a mutable reference the underlying problem instance.
    fn problem_mut(&mut self) -> &mut P;
    /// Solves the problem and returns the solution along with the solve result
    /// info.
    fn solve(&self) -> (Vec<T>, SolveResult);
    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&self, x: &mut [T]) -> SolveResult;
}
