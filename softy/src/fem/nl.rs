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
use std::path::PathBuf;
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
    /// Second order Backward Differentiation Formula.
    BDF2,
    /// TR followed by BDF2.
    ///
    /// The given float indicates the fraction of the TR step.
    TRBDF2(f32),
    /// BE followed by a linear interpolation between next and current forces.
    SDIRK2,
}

impl Default for TimeIntegration {
    fn default() -> Self {
        TimeIntegration::BE
    }
}

impl TimeIntegration {
    /// Given the stage index determines which single step integrator to use.
    pub fn step_integrator(&self, stage: u8) -> (SingleStepTimeIntegration, f64) {
        match *self {
            TimeIntegration::BE => (SingleStepTimeIntegration::BE, 1.0),
            TimeIntegration::TR => (SingleStepTimeIntegration::TR, 1.0),
            TimeIntegration::BDF2 => (SingleStepTimeIntegration::BDF2, 1.0),
            TimeIntegration::TRBDF2(t) => {
                if stage % 2 == 0 {
                    (SingleStepTimeIntegration::TR, t as f64)
                } else {
                    // The factor is builtin to the MixedBDF2 single step because it is unique to TRBDF2.
                    (SingleStepTimeIntegration::MixedBDF2(1.0 - t), 1.0)
                }
            }
            TimeIntegration::SDIRK2 => {
                if stage % 2 == 0 {
                    let factor = 1.0 - 0.5 * 2.0_f64.sqrt();
                    (SingleStepTimeIntegration::BE, factor)
                } else {
                    // The factor is builtin to the SDRIK2 single step because it is unique.
                    (SingleStepTimeIntegration::SDIRK2, 1.0)
                }
            }
        }
    }

    pub fn num_stages(&self) -> u8 {
        match self {
            TimeIntegration::BE | TimeIntegration::TR | TimeIntegration::BDF2 => 1,
            TimeIntegration::TRBDF2(_) | TimeIntegration::SDIRK2 => 2,
        }
    }
}

/// Time integration method.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SingleStepTimeIntegration {
    /// Backward Euler integration.
    BE,
    /// Trapezoid Rule integration.
    TR,
    /// Second order Backward Differentiation Formula.
    BDF2,
    /// Second order Backward Differentiation Formula mixed with another integrator within a single step.
    ///
    /// The specified float indicates how much of the BDF2 step should be taken.
    MixedBDF2(f32),
    /// Second stage of the SDIRK2 scheme.
    ///
    /// This interpolates between explicit and implicit steps, but uses the `prev` velocity and
    /// position during the advance step.
    SDIRK2,
}

impl SingleStepTimeIntegration {
    /// Returns the fraction of the implicit step represented by this single step integrator.
    pub fn implicit_factor(&self) -> f32 {
        match self {
            SingleStepTimeIntegration::BE => 1.0,
            SingleStepTimeIntegration::TR => 0.5,
            SingleStepTimeIntegration::BDF2 => 2.0 / 3.0,
            SingleStepTimeIntegration::MixedBDF2(t) => t / (1.0 + t),
            SingleStepTimeIntegration::SDIRK2 => 1.0 - 0.5 * 2.0_f32.sqrt(),
        }
    }
    /// Returns the fraction of the explicit step represented by this single step integrator.
    ///
    /// This is not always `1 - implicit_factor` (e.g. check BDF2).
    pub fn explicit_factor(&self) -> f32 {
        match self {
            SingleStepTimeIntegration::BE => 0.0,
            SingleStepTimeIntegration::TR => 0.5,
            // BDF2 objective is computed same as BE, but note that vtx.cur is different.
            // vtx.cur is set in update_cur_vertices at the beginning of the step.
            SingleStepTimeIntegration::BDF2 => 0.0,
            SingleStepTimeIntegration::MixedBDF2(_) => 0.0,
            SingleStepTimeIntegration::SDIRK2 => 0.5 * 2.0_f32.sqrt(),
        }
    }
}

impl Default for SingleStepTimeIntegration {
    fn default() -> Self {
        SingleStepTimeIntegration::BE
    }
}

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
    /// Test that the problem Jacobian is correct.
    pub derivative_test: u8,
    /// The velocity error tolerance for sticking between objects.
    pub friction_tolerance: f32,
    /// The distance tolerance between objects in contact.
    pub contact_tolerance: f32,
    /// Number of contact iterations.
    pub contact_iterations: u32,
    pub time_integration: TimeIntegration,
    pub preconditioner: Preconditioner,
    /// Path to a file where to store logs.
    pub log_file: Option<PathBuf>,
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
    Interrupted,
    FailedJacobianCheck,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct JacobianTimings {
    pub total: Duration,
    pub fem: Duration,
    pub diag: Duration,
    pub volume: Duration,
    pub contact: Duration,
    pub friction: Duration,
}

impl JacobianTimings {
    pub fn clear(&mut self) {
        self.fem = Duration::new(0, 0);
        self.diag = Duration::new(0, 0);
        self.volume = Duration::new(0, 0);
        self.contact = Duration::new(0, 0);
        self.friction = Duration::new(0, 0);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ResidualTimings {
    pub total: Duration,
    pub energy_gradient: Duration,
    pub update_state: Duration,
    pub update_distance_potential: Duration,
    pub update_constraint_gradient: Duration,
    pub update_multipliers: Duration,
    pub update_sliding_basis: Duration,
    pub contact_force: Duration,
    pub volume_force: Duration,
    pub contact_jacobian: Duration,
    pub jacobian: JacobianTimings,
    pub friction_force: FrictionTimings,
    pub preconditioner: Duration,
}

impl ResidualTimings {
    pub fn clear(&mut self) {
        self.total = Duration::new(0, 0);
        self.energy_gradient = Duration::new(0, 0);
        self.update_state = Duration::new(0, 0);
        self.update_distance_potential = Duration::new(0, 0);
        self.update_constraint_gradient = Duration::new(0, 0);
        self.update_multipliers = Duration::new(0, 0);
        self.update_sliding_basis = Duration::new(0, 0);
        self.contact_force = Duration::new(0, 0);
        self.contact_jacobian = Duration::new(0, 0);
        self.volume_force = Duration::new(0, 0);
        self.friction_force.clear();
        self.preconditioner = Duration::new(0, 0);
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
            "    Volume force time:        {}",
            self.residual.volume_force.as_millis()
        )?;
        writeln!(
            f,
            "    Contact prep time:        {}",
            self.residual.update_state.as_millis()
                + self.residual.update_distance_potential.as_millis()
                + self.residual.update_multipliers.as_millis()
        )?;
        writeln!(
            f,
            "      Update state time:      {}",
            self.residual.update_state.as_millis()
        )?;
        writeln!(
            f,
            "      Update distance:        {}",
            self.residual.update_distance_potential.as_millis()
        )?;
        writeln!(
            f,
            "      Update multipliers:     {}",
            self.residual.update_multipliers.as_millis()
        )?;
        writeln!(
            f,
            "      Update sliding basis:   {}",
            self.residual.update_sliding_basis.as_millis()
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
            "    Jacobian values time:     {}",
            self.residual.jacobian.total.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian FEM:           {}",
            self.residual.jacobian.fem.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Diagonal:      {}",
            self.residual.jacobian.diag.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Volume:        {}",
            self.residual.jacobian.volume.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Contact:       {}",
            self.residual.jacobian.contact.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Friction:      {}",
            self.residual.jacobian.friction.as_millis()
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
            "  Preconditioner time:        {}",
            self.residual.preconditioner.as_millis()
        )?;
        writeln!(
            f,
            "  Total solve time            {}",
            self.total.as_millis()
        )
    }
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
