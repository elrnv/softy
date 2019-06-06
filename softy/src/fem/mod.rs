pub mod problem;
pub mod solver;

// Re-export FEM solver and problem types and traits.
pub(crate) use self::problem::*;
pub use self::solver::*;

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
