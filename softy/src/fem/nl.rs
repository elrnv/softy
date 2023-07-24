//! A module for FEM solvers for non-linear equations.

pub mod linsolve;
pub mod mcp;
pub mod newton;
pub mod problem;
pub mod solver;
pub mod state;
pub mod time_integration;
pub mod timing;

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::path::PathBuf;
use thiserror::Error;

pub use mcp::*;
pub use newton::*;
pub use problem::*;
pub use solver::*;
pub use time_integration::*;
pub use timing::*;

/// Diagonal preconditioner used to scale the problem as well as postcondition the iterative
/// solve if used.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Preconditioner {
    /// No preconditioner.
    None,
    /// Incomplete Jacobi preconditioner addresses only the elasticity, damping and inertia part of
    /// the Jacobian.
    IncompleteJacobi,
    /// Approximate Jacobi preconditioner combines the `IncompleteJacobi` preconditioner
    /// with an approximation to the constraint jacobian diagonal.
    ApproximateJacobi,
}

impl Default for Preconditioner {
    fn default() -> Self {
        Preconditioner::ApproximateJacobi
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ZoneParams {
    pub zone_pressurizations: Vec<f32>,
    pub compression_coefficients: Vec<f32>,
    pub hessian_approximation: Vec<bool>,
}

/// Simulation parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    pub solver_type: SolverType,
    /// Test that the problem Jacobian is correct.
    pub derivative_test: u8,
    /// Number of contact iterations.
    pub contact_iterations: u32,
    pub time_integration: TimeIntegration,
    pub preconditioner: Preconditioner,
    /// Path to a file where to store logs.
    pub log_file: Option<PathBuf>,
    /// If true the per element FEM Hessians will be projected to be positive semi-definite.
    pub project_element_hessians: bool,
}

impl SimParams {
    pub fn should_compute_jacobian_matrix(&self) -> bool {
        self.derivative_test > 0 || matches!(self.linsolve, LinearSolver::Direct)
    }
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
    FailedToInitializeJacobian,
    Interrupted,
    FailedJacobianCheck,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct StepResult {
    /// Collection of all integration stages in a step.
    pub stage_solves: Vec<StageResult>,
}

impl StepResult {
    /// Get the first solve result.
    ///
    /// # Panics
    ///
    /// There should always be at least one, but if there is an internal bug and no
    /// results are available this function will panic.
    pub fn first_solve_result(&self) -> &SolveResult {
        &self.stage_solves[0].solve_results[0].1
    }
}

impl std::fmt::Display for StepResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Begin step solve")?;
        for res in self.stage_solves.iter() {
            writeln!(f, "{}", res)?;
        }
        writeln!(f, "End of step solve")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct StageResult {
    /// Time integration type for this single step.
    pub time_integration: SingleStepTimeIntegration,
    /// Collection of all solves involved in resolving this stage.
    ///
    /// Each additional solve corresponds to some constraint violation.
    /// Additional information associated with each solve is stored in `ProblemInfo`.
    pub solve_results: Vec<(ProblemInfo, SolveResult)>,
    /// Number of contact violations.
    ///
    /// This indicates the number of times contact was violated at the end of the step.
    pub contact_violations: u32,
    /// Number of violations for the maximum step allowed.
    ///
    /// This indicates the number of times contact point velocity has exceeded the implicit
    /// surface radius, where the potential is computed.
    pub max_step_violations: u32,
}

impl StageResult {
    pub fn new(time_integration: SingleStepTimeIntegration) -> Self {
        StageResult {
            time_integration,
            ..Default::default()
        }
    }
}

impl std::fmt::Display for StageResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Begin integration stage: {:?}", self.time_integration)?;
        for (pi, res) in self.solve_results.iter() {
            writeln!(f, "{}", res)?;
            writeln!(f, "{}", pi)?;
        }
        writeln!(f, "Total contact violations:  {}", self.contact_violations)?;
        writeln!(f, "Total max step violations: {}", self.max_step_violations)?;
        writeln!(f, "End of stage: {:?}", self.time_integration)?;
        Ok(())
    }
}

/// Additional information associated with each `SolveResult`.
///
/// This information is not part of `SolveResult` since it is specific to the problem, and not
/// the overall non-linear solve.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ProblemInfo {
    /// Total number of contacts.
    ///
    /// These are vertices within `delta` of the surface.
    pub total_contacts: u64,
    /// Total vertices in close proximity.
    pub total_in_proximity: u64,
}

impl std::fmt::Display for ProblemInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Total in proximity: {}", self.total_in_proximity)?;
        writeln!(f, "Total contacts:     {}", self.total_contacts)
    }
}

/// Result of a single non-linear solve.
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

impl Default for SolveResult {
    fn default() -> Self {
        SolveResult {
            iterations: 0,
            status: Status::NothingToSolve,
            timings: Timings::default(),
            stats: Vec::new(),
        }
    }
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = IterationInfo::header();
        writeln!(f, "{}", header[0])?;
        writeln!(f, "{}", header[1])?;
        for info in self.stats.iter() {
            writeln!(f, "{}", info)?;
        }
        writeln!(f, "Status:             {:?}", self.status)?;
        let lin_steps = self
            .stats
            .iter()
            .map(|s| s.linsolve_result.iterations)
            .sum::<u32>();
        writeln!(f, "Total linear steps: {}", lin_steps)?;
        let ls_steps = self.stats.iter().map(|s| s.ls_steps).sum::<u32>();
        writeln!(f, "Total ls steps:     {}", ls_steps)?;
        writeln!(f, "Total iterations:   {}", self.iterations)?;
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
    fn update_jacobian_indices(&mut self) -> bool;
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
