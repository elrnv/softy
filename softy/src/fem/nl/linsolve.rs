mod bicgstab;
mod cr;

pub use bicgstab::*;
pub use cr::*;

use thiserror::Error;

#[derive(Copy, Clone, Debug, PartialEq, Error)]
pub enum Status {
    #[error("Success")]
    Success,
    #[error("Maximum number of linear solver iterations exceeded")]
    MaximumIterationsExceeded,
    #[error("Linear solve interrupted")]
    Interrupted,
    #[error("Linear solve interrupted during preconditioner solve")]
    InterruptedPreconditionerSolve,
    #[error("NaN detected")]
    NanDetected,
    #[error("Singular matrix detected")]
    SingularMatrix,
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
