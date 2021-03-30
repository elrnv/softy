//! An FEM solver using non-linear equations of motion.

pub mod newton;
pub mod problem;
pub mod solver;

pub use newton::SolveResult;
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
}
