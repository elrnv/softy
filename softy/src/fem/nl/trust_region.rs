use std::cell::RefCell;

use super::problem::NonLinearProblem;
use super::{Callback, CallbackArgs};
use crate::inf_norm;
use crate::Real;
use tensr::*;

// Parameters for the Newton solver.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct TrustRegionParams {
    /// Residual tolerance.
    pub r_tol: f32,
    /// Variable tolerance.
    pub x_tol: f32,
    /// Maximum number of iterations permitted.
    pub max_iter: u32,
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

pub struct TrustRegionWorkspace<T> {
    x_prev: Vec<T>,
    r: Vec<T>,
    //jtr: Vec<T>,
    r_cur: Vec<T>,
    r_next: Vec<T>,
    j_rows: Vec<usize>,
    j_cols: Vec<usize>,
    j_vals: Vec<T>,
    /// Mapping from original triplets given by the `j_*` members to the final
    /// compressed sparse matrix.
    j_mapping: Vec<usize>,
    j: DSMatrix<T>,
}

pub struct TrustRegion<P, T>
where
    T: 'static,
{
    pub problem: P,
    pub params: TrustRegionParams,
    pub outer_callback: RefCell<Callback<T>>,
    pub inner_callback: RefCell<Callback<T>>,
    pub workspace: RefCell<TrustRegionWorkspace<T>>,
}

impl<T, P> TrustRegion<P, T>
where
    T: Real + na::RealField,
    P: NonLinearProblem<T>,
{
    pub fn new(
        problem: P,
        params: TrustRegionParams,
        outer_callback: Callback<T>,
        inner_callback: Callback<T>,
    ) -> Self {
        let n = problem.num_variables();
        // Initialize previous iterate.
        let x_prev = vec![T::zero(); n];

        let r = vec![T::zero(); n];
        //let jtr = r.clone();
        let r_next = r.clone();
        let r_cur = r.clone();

        // Construct the sparse Jacobian.

        let (j_rows, j_cols) = problem.jacobian_indices(true);
        assert_eq!(j_rows.len(), j_cols.len());
        let j_nnz = j_rows.len();
        let j_vals = vec![T::zero(); j_nnz];

        // Construct a mapping from original triplets to final compressed matrix.
        let mut j_mapping = (0..j_nnz).collect::<Vec<_>>();

        j_mapping.sort_by(|&a, &b| {
            j_rows[a]
                .cmp(&j_rows[b])
                .then_with(|| j_cols[a].cmp(&j_cols[b]))
        });

        // We use tensr to build the CSR matrix since it allows us to track
        // where each element goes after compression.
        let triplet_iter = j_mapping.iter().map(|&i| (j_rows[i], j_cols[i], j_vals[i]));

        let j_uncompressed = DSMatrix::from_sorted_triplets_iter_uncompressed(triplet_iter, n, n);

        let order = j_mapping.clone();

        // Compress the CSR Jacobian matrix.
        let j = j_uncompressed.pruned(
            |_, _, _| true,
            |src, dst| {
                j_mapping[order[src]] = dst;
            },
        );

        TrustRegion {
            problem,
            params,
            outer_callback: RefCell::new(outer_callback),
            inner_callback: RefCell::new(inner_callback),
            workspace: RefCell::new(TrustRegionWorkspace {
                x_prev,
                r,
                //jtr,
                r_cur,
                r_next,
                j_rows,
                j_cols,
                j_vals,
                j_mapping,
                j,
            }),
        }
    }

    /// Solves the problem and returns the solution along with the solve result
    /// info.
    pub fn solve(&mut self) -> (Vec<T>, SolveResult) {
        let mut x = self.problem.initial_point();
        let res = self.solve_with(x.as_mut_slice());
        (x, res)
    }

    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    pub fn solve_with(&mut self, x: &mut [T]) -> SolveResult {
        let Self {
            problem,
            params,
            outer_callback,
            //inner_callback,
            workspace,
            ..
        } = self;
        let TrustRegionWorkspace {
            r,
            //jtr,
            r_cur,
            r_next,
            j_rows,
            j_cols,
            j_vals,
            j_mapping,
            j,
            x_prev,
            ..
        } = &mut *workspace.borrow_mut();

        let mut iterations = 0;

        // Initialize the residual.
        problem.residual(x, r.as_mut_slice(), false);

        //let cr = ConjugateResidual::new(
        //    |x, out| {

        //    },
        //    x,
        //    r,
        //);

        // Convert to sprs format for debugging. The CSR structure is preserved.
        let mut j_sprs: sprs::CsMat<T> = j.clone().into();

        let r_tol = T::from(params.r_tol).unwrap();
        let x_tol = T::from(params.x_tol).unwrap();

        log_debug_stats_header();
        log_debug_stats(0, 0, r, x, x_prev);
        loop {
            if !(outer_callback.borrow_mut())(CallbackArgs {
                iteration: iterations,
                residual: r.as_slice(),
                x,
                problem,
            }) {
                return SolveResult {
                    iterations,
                    status: Status::Interrupted,
                };
            }

            // Update Jacobian values.
            problem.jacobian_values(x, r, j_rows, j_cols, j_vals.as_mut_slice());

            //log::trace!("j_vals = {:?}", &j_vals);

            // Update the Jacobian matrix.
            for (src, &dst) in j_mapping.iter().enumerate() {
                j.storage_mut()[dst] = j_vals[src];
                j_sprs.data_mut()[dst] = j_vals[src];
            }

            r_cur.copy_from_slice(r);

            //log::trace!("r = {:?}", &r);

            //print_sprs(&j_sprs.view());

            if !sid_solve_mut(j.view(), r.as_mut_slice()) {
                return SolveResult {
                    iterations,
                    status: Status::LinearSolveError,
                };
            }

            // r is now the search direction, rename to avoid confusion.
            let p = r.as_slice();

            // log::trace!("p = {:?}", &r);

            // Check solution:
            // Compute out = J * p
            //let mut out = vec![0.0; problem.num_variables()];
            //for (row_idx, row) in j.as_data().iter().enumerate() {
            //    for (col_idx, val, _) in row.iter() {
            //        out[row_idx] += (*val * p[col_idx]).to_f64().unwrap();
            //    }
            //}
            //for (&a, &x) in out.iter().zip(r_cur.iter()) {
            //    log::trace!(
            //        "newton solve residual: {:?}",
            //        (a - x.to_f64().unwrap()).abs()
            //    );
            //}

            // The solve converts the rhs r into the unknown negative search direction p.

            // probe
            let mut probe_x = x.to_vec();
            let mut probe_r = r.clone();
            {
                use std::io::Write;
                let mut f =
                    std::fs::File::create(format!("./out/alpha_res_{}.jl", iterations)).unwrap();
                writeln!(&mut f, "alpha_res2 = [").ok();
                for alpha in (0..100).map(|i| T::from(i).unwrap() / T::from(100.0).unwrap()) {
                    zip!(probe_x.iter_mut(), r.iter(), x.iter())
                        .for_each(|(px, &r, &x)| *px = x - alpha * r);
                    problem.residual(&probe_x, probe_r.as_mut_slice(), false);
                    writeln!(
                        &mut f,
                        "{:?} {:10.3e};",
                        alpha,
                        probe_r.as_slice().as_tensor().norm().to_f64().unwrap()
                    )
                    .ok();
                }
                writeln!(&mut f, "]").ok();
            }

            // Update previous step.
            x_prev.copy_from_slice(x);

            // Take the full step
            *x.as_mut_tensor() -= p.as_tensor();
            problem.residual(x, r_next.as_mut_slice(), false);

            iterations += 1;

            log_debug_stats(iterations, 1, r_next, x, x_prev);

            let denom = inf_norm(x.iter().cloned()) + T::one();

            // Check the convergence condition.
            if inf_norm(r_next.iter().cloned()) < r_tol
                || inf_norm(x_prev.iter().zip(x.iter()).map(|(&a, &b)| a - b)) < x_tol * denom
            {
                return SolveResult {
                    iterations,
                    status: Status::Success,
                };
            }

            // Check that we are running no more than the maximum allowed iterations.
            if iterations >= params.max_iter {
                return SolveResult {
                    iterations,
                    status: Status::MaximumIterationsExceeded,
                };
            }

            // Reset r to be a valid residual for the next iteration.
            r.copy_from_slice(r_next);
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
        }
    }

    let mut b_vec: na::DVectorSliceMut<T> = b.into();
    dense.lu().solve_mut(&mut b_vec)
}

#[allow(dead_code)]
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
