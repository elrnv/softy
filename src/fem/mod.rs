pub mod solver;
pub mod problem;

// Re-export FEM solver and problem types and traits.
pub use self::solver::*;
pub(crate) use self::problem::*;

/// Simulation parameters.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimParams {
    pub material: MaterialProperties,
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub volume_constraint: bool,
}

