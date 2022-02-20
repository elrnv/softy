mod bicgstab;
mod cr;

pub use bicgstab::*;
pub use cr::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Status {
    Success,
    MaximumIterationsExceeded,
    Interrupted,
    InterruptedPreconditionerSolve,
}

impl Default for Status {
    fn default() -> Self {
        Status::Success
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct SolveResult {
    /// Number of iterations of an iterative solver.
    pub iterations: u32,
    /// Absoulte residual 2-norm.
    pub residual: f64,
    /// Relative residual 2-norm.
    ///
    /// Residual divided by the norm of the right-hand-side.
    pub error: f64,
    /// Final status of the linear solve.
    pub status: Status,
}
