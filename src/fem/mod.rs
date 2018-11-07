pub mod problem;
pub mod solver;

// Re-export FEM solver and problem types and traits.
pub(crate) use self::problem::*;
pub use self::solver::*;

/// Simulation parameters.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimParams {
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    pub tolerance: f32,
    pub max_iterations: u32,
}
