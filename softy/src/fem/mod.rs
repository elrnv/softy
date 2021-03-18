// pub mod nl;
pub mod opt;

use geo::prim::Tetrahedron;
use tensr::Vector3;

use crate::RefPosType;

pub(crate) use self::opt::problem::*;
pub use self::opt::solver::*;

// pub use self::nl::problem::*;
// pub use self::nl::solver::*;

/// Barrier parameter reduction strategy for interior point methods.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MuStrategy {
    Monotone,
    Adaptive,
}

/// Simulation parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct SimParams {
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    /// Clear the velocity between consecutive steps to emulate quasi-static simulation (or
    /// critical damping) where the time step and density effectively determines the
    /// regularization.
    pub clear_velocity: bool,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub max_outer_iterations: u32,
    pub friction_iterations: u32,
    pub outer_tolerance: f32,
    pub print_level: u32,
    pub derivative_test: u32,
    pub mu_strategy: MuStrategy,
    pub max_gradient_scaling: f32,
    pub log_file: Option<std::path::PathBuf>,
}

/// Result from one inner simulation step.
#[derive(Clone, Debug, PartialEq)]
pub struct InnerSolveResult {
    /// Number of inner iterations in one step.
    pub iterations: u32,
    /// The value of the objective at the end of the step.
    pub objective_value: f64,
    /// Constraint values at the solution of the inner step.
    pub constraint_values: Vec<f64>,
}

/// Result from one simulation step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolveResult {
    /// Maximum number of inner iterations during one outer step.
    pub max_inner_iterations: u32,
    /// Number of total accumulated inner iterations.
    pub inner_iterations: u32,
    /// Number of outer iterations of the step.
    pub iterations: u32,
    /// The value of the objective at the end of the time step.
    pub objective_value: f64,
}

impl SolveResult {
    fn combine_inner_step_data(self, iterations: u32, objective_value: f64) -> SolveResult {
        SolveResult {
            // Aggregate max number of iterations.
            max_inner_iterations: iterations.max(self.max_inner_iterations),

            inner_iterations: iterations + self.inner_iterations,

            // Adding a new inner solve result means we have completed another inner solve: +1
            // outer iterations.
            iterations: self.iterations + 1,

            // The objective value of the solution is the objective value of the last inner solve.
            objective_value,
        }
    }
    /// Add an inner solve result to this solve result.
    fn combine_inner_result(self, inner: &InnerSolveResult) -> SolveResult {
        self.combine_inner_step_data(inner.iterations, inner.objective_value)
    }
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Iterations: {}\nObjective: {}\nMax Inner Iterations: {}",
            self.iterations, self.objective_value, self.max_inner_iterations
        )
    }
}

/// Get reference tetrahedron.
///
/// This routine assumes that there is a vertex attribute called `ref` of type `[f32; 3]`.
pub fn ref_tet(ref_pos: &[RefPosType]) -> Tetrahedron<f64> {
    Tetrahedron::new([
        Vector3::new(ref_pos[0]).cast::<f64>().into(),
        Vector3::new(ref_pos[1]).cast::<f64>().into(),
        Vector3::new(ref_pos[2]).cast::<f64>().into(),
        Vector3::new(ref_pos[3]).cast::<f64>().into(),
    ])
}
