use std::cell::RefCell;

use super::problem::NonLinearProblem;
use tensr::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Missing tolerance parameter 'tol'")]
    MissingTolerance,
    #[error("Missing maximum number of iterations parameter 'max_iter'")]
    MissingMaximumIterations,
}

pub struct CallbackArgs<'a, T> {
    residual: &'a [T],
}

pub type Callback<T> = Box<dyn FnMut(CallbackArgs<T>) -> bool + 'static>;

// Parameters for the Newton solver.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NewtonParams {
    pub tol: f32,
    pub max_iter: u32,
}

impl Default for NewtonParams {
    fn default() -> NewtonParams {
        NewtonParams {
            tol: 1e-3,
            max_iter: 1000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Success,
    MaximumIterationsExceeded,
    LLTError(sprs::errors::LinalgError),
    Interrupted,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult {
    /// Number of successful iterations.
    pub iterations: u32,
    /// Solve status.
    pub status: Status,
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Newton<P, T> {
    pub problem: P,
    pub params: NewtonParams,
    pub intermediate_callback: RefCell<Callback<T>>,
}

impl<T: Real, P: NonLinearProblem<T>> Newton<P, T>
where
    for<'a> &'a T: std::ops::Mul<&'a T, Output = T>, // Needed for sprs matrix multiply.
{
    /// Solves the problem and returns the solution along with the solve result
    /// info.
    pub fn solve(&self) -> (Vec<T>, SolveResult) {
        let mut x = self.problem.initial_point();
        let res = self.solve_with(x.as_mut_slice());
        (x, res)
    }

    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    pub fn solve_with(&self, x: &mut [T]) -> SolveResult {
        let mut iterations = 0;

        // Number of variables.
        let n = self.problem.num_variables();

        // Initialize the residual.
        let mut r = vec![T::zero(); n];

        // Construct the sparse Jacobian.
        let (j_rows, j_cols) = self.problem.jacobian_indices();
        assert_eq!(j_rows.len(), j_cols.len());
        let j_nnz = j_rows.len();
        let mut j_vals = vec![T::zero(); j_nnz];

        // Construct a CSR matrix
        let mut j = sprs::TriMat::from_triplets((n, n), j_rows, j_cols, j_vals).to_csr::<usize>();

        let tol = T::from(self.params.tol).unwrap();

        // Initialize the residual.
        self.problem.residual(x, r.as_mut_slice());
        let initial_residual_norm_squared = r.as_slice().as_tensor().norm_squared();

        loop {
            if !(self.intermediate_callback.borrow_mut())(CallbackArgs {
                residual: r.as_slice(),
            }) {
                return SolveResult {
                    iterations,
                    status: Status::Interrupted,
                };
            }

            // Check the convergence condition.
            if r.as_slice().as_tensor().norm_squared() < tol * tol * initial_residual_norm_squared {
                return SolveResult {
                    iterations,
                    status: Status::Success,
                };
            }

            // Check that we are running no more than the maximum allowed iterations.
            if iterations >= self.params.max_iter {
                return SolveResult {
                    iterations,
                    status: Status::MaximumIterationsExceeded,
                };
            }

            // Update Jacobian values.
            self.problem.jacobian_values(x, j.data_mut());

            if let Err(err) =
                sprs::linalg::trisolve::lsolve_csr_dense_rhs(j.view(), r.as_mut_slice())
            {
                return SolveResult {
                    iterations,
                    status: Status::LLTError(err),
                };
            }

            // The solve converts the rhs r into the unknown negative search direction -p.

            // Take the entire step.
            *x.as_mut_tensor() -= r.as_slice().as_tensor();

            iterations += 1;

            // Update the residual.
            self.problem.residual(x, r.as_mut_slice());
        }
    }
}
