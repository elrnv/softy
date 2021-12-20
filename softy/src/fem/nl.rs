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
use thiserror::Error;

pub use mcp::*;
pub use newton::*;
pub use problem::*;
pub use solver::*;
pub use trust_region::*;

/// Simulation parameters.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimParams {
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    /// Clear the velocity between consecutive steps to emulate quasi-static simulation (or
    /// critical damping) where the time step and density effectively determines the
    /// regularization.
    pub clear_velocity: bool,
    pub residual_tolerance: Option<f32>,
    pub acceleration_tolerance: Option<f32>,
    pub velocity_tolerance: Option<f32>,
    pub max_iterations: u32,
    pub linsolve_tolerance: f32,
    pub max_linsolve_iterations: u32,
    pub line_search: LineSearch,
    /// Test that the problem Jacobian is correct.
    pub jacobian_test: bool,
    /// The velocity error tolerance for sticking between objects.
    pub friction_tolerance: f32,
    /// The distance tolerance between objects in contact.
    pub contact_tolerance: f32,
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
    LinearSolveError(SparseDirectSolveError),
    Diverged,
    StepTooLarge,
    NothingToSolve,
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
    /// Gets a reference to the outer callback function.
    fn outer_callback(&self) -> &RefCell<Callback<T>>;
    /// Gets a reference to the inner callback function.
    fn inner_callback(&self) -> &RefCell<Callback<T>>;
    /// Gets a reference to the underlying problem instance.
    fn problem(&self) -> &P;
    /// Gets a mutable reference the underlying problem instance.
    fn problem_mut(&mut self) -> &mut P;
    /// Solves the problem and returns the solution along with the solve result
    /// info.
    fn solve(&mut self) -> (Vec<T>, SolveResult);
    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&mut self, x: &mut [T]) -> SolveResult;
}
