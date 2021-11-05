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
    pub iterations: u32,
    pub residual: f64,
    pub error: f64,
    pub status: Status,
}
