use std::cell::RefCell;

use super::problem::NonLinearProblem;
use crate::inf_norm;
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
    pub residual: &'a [T],
}

pub type Callback<T> = Box<dyn FnMut(CallbackArgs<T>) -> bool + Send + 'static>;

// Parameters for the Newton solver.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NewtonParams {
    /// Residual tolerance.
    pub r_tol: f32,
    /// Variable tolerance.
    pub x_tol: f32,
    /// Maximum number of iterations permitted.
    pub max_iter: u32,
    /// Line search method.
    pub line_search: LineSearch,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Success,
    MaximumIterationsExceeded,
    LinearSolveError,
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
    T: na::RealField,
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

        // Create bounds.
        let (l, u) = self.problem.initial_bounds();
        assert_eq!(l.len(), n);
        assert_eq!(u.len(), n);

        // Construct the sparse Jacobian.
        let (j_rows, j_cols) = self.problem.jacobian_indices();
        assert_eq!(j_rows.len(), j_cols.len());
        let j_nnz = j_rows.len();
        let mut j_vals = vec![T::zero(); j_nnz];

        // Construct a mapping from original triplets to final compressed matrix.
        let mut mapping = (0..j_nnz).collect::<Vec<_>>();

        mapping.sort_by(|&a, &b| {
            j_rows[a]
                .cmp(&j_rows[b])
                .then_with(|| j_cols[a].cmp(&j_cols[b]))
        });

        // We use tensr to build the CSR matrix since it allows us to track
        // where each element goes after compression.
        let triplet_iter = mapping.iter().map(|&i| (j_rows[i], j_cols[i], j_vals[i]));

        let j_uncompressed = DSMatrix::from_sorted_triplets_iter_uncompressed(triplet_iter, n, n);

        let order = mapping.clone();

        // Compress the CSR Jacobian matrix.
        let mut j = j_uncompressed.pruned(
            |_, _, _| true,
            |src, dst| {
                mapping[order[src]] = dst;
            },
        );

        // Convert to sprs format for debugging. The CSR structure is preserved.
        let mut j_sprs: sprs::CsMat<T> = j.clone().into();

        // Initialize previous iterate.
        let mut x_prev = x.to_vec();

        let r_tol = T::from(self.params.r_tol).unwrap();
        let x_tol = T::from(self.params.x_tol).unwrap();

        // Initialize the residual.
        self.problem.residual(x, r.as_mut_slice());

        let mut r_next = r.clone();
        let mut r_cur = r.clone();

        log_debug_stats_header();
        log_debug_stats(0, 0, &r, x, &x_prev);
        loop {
            if !(self.intermediate_callback.borrow_mut())(CallbackArgs {
                residual: r.as_slice(),
            }) {
                return SolveResult {
                    iterations,
                    status: Status::Interrupted,
                };
            }

            // Update Jacobian values.
            self.problem.jacobian_values(x, &mut j_vals);

            // Update the Jacobian matrix.
            for (src, &dst) in mapping.iter().enumerate() {
                j.storage_mut()[dst] = j_vals[src];
                j_sprs.data_mut()[dst] = j_vals[src];
            }

            r_cur.clone_from_slice(&r);

            //log::trace!("r = {:?}", &r);

            if !sid_solve_mut(j.view(), r.as_mut_slice()) {
                return SolveResult {
                    iterations,
                    status: Status::LinearSolveError,
                };
            }

            // r is now the search direction, rename to avoid confusion.
            let p = r.as_slice();

            //log::trace!("x = {:?}", &r);

            //print_sprs(&j_sprs.view());

            //// Check solution:
            //// Compute out = J * p
            //let mut out = vec![0.0; n];
            //for (row_idx, row) in j.as_data().iter().enumerate() {
            //    for (col_idx, val, _) in row.iter() {
            //        out[row_idx] += (*val * p[col_idx]).to_f64().unwrap();
            //        if col_idx != row_idx {
            //            out[col_idx] += (*val * p[row_idx]).to_f64().unwrap();
            //        }
            //    }
            //}

            //for (&a, &x) in out.iter().zip(r_cur.iter()) {
            //    log::trace!(
            //        "newton solve residual: {:?}",
            //        (a - x.to_f64().unwrap()).abs()
            //    );
            //}

            // The solve converts the rhs r into the unknown negative search direction -p.

            // probe
            //let mut probe_x = x.to_vec();
            //let mut probe_r = r.clone();
            //for alpha in (0..1000).map(|i| T::from(i).unwrap()/T::from(1000.0).unwrap()) {
            //    zip!(probe_x.iter_mut(), r.iter(), x.iter()).for_each(|(px, &r, &x)| {
            //        *px = x - alpha*r
            //    });
            //    self.problem.residual(&probe_x, probe_r.as_mut_slice());
            //    log::trace!("alpha = {:?}: res-2 = {:10.3e}", alpha, probe_r.as_slice().as_tensor().norm().to_f64().unwrap());
            //}

            // Update previous step.
            x_prev.clone_from_slice(x);

            let rho = self.params.line_search.step_factor();
            let ls_count = if rho >= 1.0 {
                // Take the full step
                *x.as_mut_tensor() -= p.as_tensor();
                self.problem.residual(x, r_next.as_mut_slice());
                1
            } else {
                // Line search.
                let mut alpha = 1.0;
                let mut ls_count = 1;

                loop {
                    // Step and project.
                    zip!(x.iter_mut(), x_prev.iter(), p.iter(), l.iter(), u.iter()).for_each(
                        |(x, &x0, &p, &l, &u)| {
                            *x = num_traits::clamp(
                                num_traits::Float::mul_add(p, T::from(-alpha).unwrap(), x0),
                                T::from(l).unwrap(),
                                T::from(u).unwrap(),
                            );
                        },
                    );

                    // Compute the candidate residual.
                    self.problem.residual(x, r_next.as_mut_slice());

                    if (&r_next).as_tensor().norm().to_f64().unwrap()
                        <= (1.0 - self.params.line_search.armijo_coeff() * alpha)
                            * (&r_cur).as_tensor().norm().to_f64().unwrap()
                    {
                        break;
                    }

                    alpha *= rho;

                    ls_count += 1;
                }
                ls_count
            };

            iterations += 1;

            log_debug_stats(iterations, ls_count, &r_next, x, &x_prev);

            let x_norm = inf_norm(x.iter().cloned());

            // Check the convergence condition.
            if inf_norm(r_next.iter().cloned()) < r_tol
                || inf_norm(x_prev.iter().zip(x.iter()).map(|(&a, &b)| a - b)) < x_tol * x_norm
            {
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

            // Reset r to be a valid residual for the next iteration.
            r.copy_from_slice(&r_next);
        }
    }
}

fn log_debug_stats_header() {
    log::debug!("    i |   res-2    |  res-inf   |   d-inf    |   x-inf    | ls # ");
    log::debug!("------+------------+------------+------------+------------+------");
}
fn log_debug_stats<T: Real>(iterations: u32, ls_steps: u32, r: &[T], x: &[T], x_prev: &[T]) {
    log::debug!(
        "{i:>5} |  {res2:10.3e} |{resi:10.3e} | {di:10.3e} | {xi:10.3e} | {ls:>4} ",
        i = iterations,
        res2 = r.as_tensor().norm().to_f64().unwrap(),
        resi = inf_norm(r.iter().cloned()).to_f64().unwrap(),
        di = inf_norm(x_prev.iter().zip(x.iter()).map(|(&a, &b)| a - b))
            .to_f64()
            .unwrap(),
        xi = inf_norm(x.iter().cloned()).to_f64().unwrap(),
        ls = ls_steps
    );
}

/// Solves a sparse symmetric indefinite system `Ax = b`.
///
/// Here `A` is the lower triangular part of a symmetric sparse matrix in CSR format,
/// `b` is a dense right hand side vector and the output `x` is computed in `b`.
#[allow(non_snake_case)]
fn sid_solve_mut<T: Real + na::ComplexField>(A: DSMatrixView<T>, b: &mut [T]) -> bool {
    // nalgebra dense prototype using lu.
    let mut dense = na::DMatrix::zeros(A.num_rows(), A.num_cols());

    for (row_idx, row) in A.as_data().iter().enumerate() {
        for (col_idx, val, _) in row.iter() {
            *dense.index_mut((row_idx, col_idx)) = *val;
            if row_idx != col_idx {
                *dense.index_mut((col_idx, row_idx)) = *val;
            }
        }
    }

    let mut b_vec: na::DVectorSliceMut<T> = b.into();
    dense.lu().solve_mut(&mut b_vec)
}

fn print_sprs<T: Real>(mat: &sprs::CsMatView<T>) {
    eprintln!("mat = [");
    for r in 0..mat.rows() {
        for c in 0..mat.cols() {
            eprint!(
                "{} ",
                mat.get(r, c).map(|&x| x.to_f64().unwrap()).unwrap_or(0.0)
            );
        }
        eprintln!(";");
    }
    eprintln!("]");
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum LineSearch {
    /// Backtracking line search method decreases the step `α` by `rho` to
    /// Satisfy the sufficient decrease condition:
    /// f(xₖ + αpₖ) ≤ f(xₖ) + cα∇fₖᵀpₖ
    BackTracking {
        c: f64,
        rho: f64,
    },
    None,
}

impl Default for LineSearch {
    fn default() -> LineSearch {
        LineSearch::default_backtracking()
    }
}

impl LineSearch {
    pub const fn default_backtracking() -> Self {
        LineSearch::BackTracking { c: 1e-4, rho: 0.9 }
    }
    /// Gets the factor by which the step size should be decreased.
    pub fn step_factor(&self) -> f64 {
        match self {
            LineSearch::BackTracking { rho, .. } => *rho,
            LineSearch::None => 1.0,
        }
    }

    // Gets the coefficient for the Armijo condition.
    pub fn armijo_coeff(&self) -> f64 {
        match self {
            LineSearch::BackTracking { c, .. } => *c,
            LineSearch::None => 1.0,
        }
    }
}
