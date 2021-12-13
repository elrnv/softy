use std::cell::RefCell;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tensr::*;

use super::linsolve::*;
use super::problem::NonLinearProblem;
use super::{Callback, CallbackArgs, NLSolver, SolveResult, Status};
use crate::inf_norm;
use crate::Real;

// Parameters for the Newton solver.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NewtonParams {
    /// Residual tolerance.
    pub r_tol: f32,
    /// Variable tolerance.
    pub x_tol: f32,
    /// Maximum number of Newton iterations permitted.
    pub max_iter: u32,
    /// Residual tolerance for the linear solve.
    pub linsolve_tol: f32,
    /// Maximum number of iterations permitted for the linear solve.
    pub linsolve_max_iter: u32,
    /// Line search method.
    pub line_search: LineSearch,
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct NewtonWorkspace<T: Real> {
    linsolve: BiCGSTAB<T>,
    x_prev: Vec<T>,
    r: Vec<T>,
    p: Vec<T>,
    jp: Vec<T>,
    r_cur: Vec<T>,
    r_next: Vec<T>,
    j_rows: Vec<usize>,
    j_cols: Vec<usize>,
    //j_vals: Vec<T>,
    ////// Mapping from original triplets given by the `j_*` members to the final
    ////// compressed sparse matrix.
    //j_mapping: Vec<usize>,
    //j: DSMatrix<T>,
}

pub struct Newton<P, T: Real> {
    pub problem: P,
    pub params: NewtonParams,
    /// A function called for every iteration of the iterative linear solver.
    ///
    /// If this function returns false, the solve is interrupted.
    pub inner_callback: RefCell<Callback<T>>,
    /// A function called for every Newton iteration.
    ///
    /// If this function returns false, the solve is interrupted.
    pub outer_callback: RefCell<Callback<T>>,
    pub workspace: RefCell<NewtonWorkspace<T>>,
}

impl<T, P> Newton<P, T>
where
    T: Real,
    P: NonLinearProblem<T>,
{
    pub fn new(
        problem: P,
        params: NewtonParams,
        outer_callback: Callback<T>,
        inner_callback: Callback<T>,
    ) -> Self {
        let n = problem.num_variables();
        // Initialize previous iterate.
        let x_prev = vec![T::zero(); n];

        let r = x_prev.clone();
        let jp = r.clone();
        let r_next = r.clone();
        let r_cur = r.clone();
        let p = r.clone();

        // Construct the sparse Jacobian.

        let (j_rows, j_cols) = problem.jacobian_indices();
        assert_eq!(j_rows.len(), j_cols.len());
        //let j_nnz = j_rows.len();
        //let j_vals = vec![T::zero(); j_nnz];

        //// Construct a mapping from original triplets to final compressed matrix.
        //let mut j_mapping = (0..j_nnz).collect::<Vec<_>>();

        //j_mapping.sort_by(|&a, &b| {
        //    j_rows[a]
        //        .cmp(&j_rows[b])
        //        .then_with(|| j_cols[a].cmp(&j_cols[b]))
        //});

        //// We use tensr to build the CSR matrix since it allows us to track
        //// where each element goes after compression.
        //let triplet_iter = j_mapping.iter().map(|&i| (j_rows[i], j_cols[i], j_vals[i]));

        //let j_uncompressed = DSMatrix::from_sorted_triplets_iter_uncompressed(triplet_iter, n, n);

        //let order = j_mapping.clone();

        //// Compress the CSR Jacobian matrix.
        //let j = j_uncompressed.pruned(
        //    |_, _, _| true,
        //    |src, dst| {
        //        j_mapping[order[src]] = dst;
        //    },
        //);

        // Allocate space for the linear solver.
        let linsolve = BiCGSTAB::new(n, params.linsolve_max_iter, params.linsolve_tol);

        Newton {
            problem,
            params,
            outer_callback: RefCell::new(outer_callback),
            inner_callback: RefCell::new(inner_callback),
            workspace: RefCell::new(NewtonWorkspace {
                linsolve,
                x_prev,
                r,
                p,
                jp,
                r_cur,
                r_next,
                j_rows,
                j_cols,
                //j_vals,
                //j_mapping,
                //j,
            }),
        }
    }
}

impl<T, P> NLSolver<P, T> for Newton<P, T>
where
    T: Real,
    P: NonLinearProblem<T>,
{
    /// Gets a reference to the outer callback function.
    ///
    /// This callback gets called at the beginning of every Newton iteration.
    fn outer_callback(&self) -> &RefCell<Callback<T>> {
        &self.outer_callback
    }
    /// Gets a reference to the inner callback function.
    ///
    /// This is the callback that gets called for every inner linear solve.
    fn inner_callback(&self) -> &RefCell<Callback<T>> {
        &self.inner_callback
    }
    /// Gets a reference to the underlying problem instance.
    fn problem(&self) -> &P {
        &self.problem
    }
    /// Gets a mutable reference the underlying problem instance.
    fn problem_mut(&mut self) -> &mut P {
        &mut self.problem
    }

    /// Solves the problem and returns the solution along with the solve result
    /// info.
    fn solve(&mut self) -> (Vec<T>, SolveResult) {
        let mut x = self.problem.initial_point();
        let res = self.solve_with(x.as_mut_slice());
        (x, res)
    }

    /// Solves the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&mut self, x: &mut [T]) -> SolveResult {
        let Self {
            problem,
            params,
            inner_callback,
            outer_callback,
            workspace,
        } = self;

        let NewtonWorkspace {
            linsolve,
            r,
            p,
            jp,
            r_cur,
            r_next,
            j_rows,
            j_cols,
            //j_vals,
            //j_mapping,
            //j,
            x_prev,
        } = &mut *workspace.borrow_mut();

        let mut iterations = 0;

        // Initialize the residual.
        problem.residual(x, r.as_mut_slice());

        // Convert to sprs format for debugging. The CSR structure is preserved.
        //let mut j_sprs: sprs::CsMat<T> = j.clone().into();

        let r_tol = T::from(params.r_tol).unwrap();
        let x_tol = T::from(params.x_tol).unwrap();

        let mut linsolve_result = super::linsolve::SolveResult::default();

        log_debug_stats_header();
        log_debug_stats(0, 0, linsolve_result, linsolve.tol, &r, x, &x_prev);

        // Remember original tolerance so we can reset it later.
        let orig_linsolve_tol = linsolve.tol;

        // Timing stats
        let mut linsolve_time = Duration::new(0, 0);
        let mut ls_time = Duration::new(0, 0);
        let mut jprod_linsolve_time = Duration::new(0, 0);
        let mut jprod_ls_time = Duration::new(0, 0);
        let mut residual_time = Duration::new(0, 0);

        // Keep track of norms to avoid having to recompute them
        let mut r_prev_norm = r.as_tensor().norm().to_f64().unwrap();
        let mut r_cur_norm = r_prev_norm;
        let mut r_next_norm = r_cur_norm;

        let mut j_dense =
            ChunkedN::from_flat_with_stride(x.len(), vec![T::zero(); x.len() * r.len()]);
        let mut identity =
            ChunkedN::from_flat_with_stride(x.len(), vec![T::zero(); x.len() * r.len()]);
        for (i, id) in identity.iter_mut().enumerate() {
            id[i] = T::one();
        }

        let result = loop {
            //log::trace!("Previous r norm: {}", r_prev_norm);
            //log::trace!("Current  r norm: {}", r_cur_norm);
            if !(outer_callback.borrow_mut())(CallbackArgs {
                residual: r.as_slice(),
                x,
                problem,
            }) {
                break SolveResult {
                    iterations,
                    status: Status::Interrupted,
                };
            }

            // Update Jacobian values.
            //let before_j = Instant::now();
            //problem.jacobian_values(x, &r, &j_rows, &j_cols, j_vals.as_mut_slice());
            // TODO: For debugging only
            for (jp, p) in j_dense.iter_mut().zip(identity.iter()) {
                problem.jacobian_product(x, &p, &r, &j_rows, &j_cols, jp)
            }
            //eprintln!("J = [");
            //for jp in j_dense.iter() {
            //    for j in jp.iter() {
            //        eprint!("{:?} ", j);
            //    }
            //    eprintln!(";");
            //}
            //eprintln!("]");
            svd_values(j_dense.view());
            //write_jacobian_img(j_dense.view(), iterations);
            //jprod_time += Instant::now() - before_j;

            ////log::trace!("j_vals = {:?}", &j_vals);

            //// Zero out Jacobian.
            //j.storage_mut().iter_mut().for_each(|x| *x = T::zero());
            //j_sprs.data_mut().iter_mut().for_each(|x| *x = T::zero());

            // Update the Jacobian matrix.
            //for (&pos, &j_val) in j_mapping.iter().zip(j_vals.iter()) {
            //j.storage_mut()[pos] += j_val;
            //    j_sprs.data_mut()[pos] += j_val;
            //}

            //log::trace!("r = {:?}", &r);

            if !r_cur_norm.is_finite() {
                break SolveResult {
                    iterations,
                    status: Status::Diverged,
                };
            }

            //print_sprs(&j_sprs.view());

            let t_begin_linsolve = Instant::now();

            // Update tolerance (forcing term)
            linsolve.tol = orig_linsolve_tol
                .min(((r_cur_norm - linsolve_result.residual).abs() / r_prev_norm) as f32);

            r_cur.copy_from_slice(&r);
            linsolve_result = linsolve.solve(
                |p, out| {
                    let t_begin_jprod = Instant::now();
                    problem.jacobian_product(x, p, r_cur, j_rows, j_cols, out);
                    //out.iter_mut().for_each(|x| *x = T::zero());
                    //j.view()
                    //    .add_mul_in_place_par(p.as_tensor(), out.as_mut_tensor());
                    jprod_linsolve_time += Instant::now() - t_begin_jprod;
                    inner_callback.borrow_mut()(CallbackArgs {
                        residual: r_cur.as_slice(),
                        x,
                        problem,
                    })
                },
                p.as_mut_slice(),
                r.as_mut_slice(),
            );

            linsolve_time += Instant::now() - t_begin_linsolve;
            //log::trace!("linsolve result: {:?}", linsolve_result);

            //if !sid_solve_mut(j.view(), r.as_mut_slice()) {
            //    break SolveResult {
            //        iterations,
            //        status: Status::LinearSolveError,
            //    };
            //}

            // r is now the search direction, rename to avoid confusion.
            //let p = r.as_slice();

            // log::trace!("p = {:?}", &r);

            // Check solution:
            // Compute out = J * p
            //jp.iter_mut().for_each(|x| *x = T::zero());
            //j.view()
            //    .add_right_mul_in_place(p.as_tensor(), jp.as_mut_tensor());
            //for (&a, &x) in out.iter().zip(r_cur.iter()) {
            //    log::trace!(
            //        "newton solve residual: {:?}",
            //        (a - x.to_f64().unwrap()).abs()
            //    );
            //}

            // The solve converts the rhs r into the unknown negative search direction p.

            // probe
            //let mut probe_x = x.to_vec();
            //let mut probe_r = r.clone();
            //{
            //    use std::io::Write;
            //    let mut f =
            //        std::fs::File::create(format!("./out/alpha_res_{}.jl", iterations)).unwrap();
            //    writeln!(&mut f, "alpha_res2 = [").ok();
            //    for alpha in (0..100).map(|i| T::from(i).unwrap() / T::from(100.0).unwrap()) {
            //        zip!(probe_x.iter_mut(), r.iter(), x.iter())
            //            .for_each(|(px, &r, &x)| *px = x - alpha * r);
            //        problem.residual(&probe_x, probe_r.as_mut_slice());
            //        writeln!(
            //            &mut f,
            //            "{:?} {:10.3e};",
            //            alpha,
            //            probe_r.as_slice().as_tensor().norm().to_f64().unwrap()
            //        )
            //        .ok();
            //    }
            //    writeln!(&mut f, "]").ok();
            //}

            if !p.as_tensor().norm_squared().is_finite() {
                break SolveResult {
                    iterations,
                    status: Status::StepTooLarge,
                };
            }

            // Update previous step.
            x_prev.copy_from_slice(x);

            let t_begin_ls = Instant::now();
            let rho = params.line_search.step_factor();

            // Take the full step
            *x.as_mut_tensor() -= p.as_tensor();

            // Compute the residual for the full step.
            let t_begin_residual = Instant::now();

            problem.residual(&x, r_next.as_mut_slice());

            residual_time += Instant::now() - t_begin_residual;

            let ls_count = if rho >= 1.0 {
                1
            } else {
                // Line search.
                let mut alpha = 1.0;
                let mut ls_count = 1;
                let mut sigma = linsolve.tol as f64;

                loop {
                    // Compute gradient of the merit function 0.5 r'r  multiplied by p, which is r' dr/dx p.
                    // Gradient dot search direction.
                    // Check Armijo condition:
                    // Given f(x) = 0.5 || r(x) ||^2 = 0.5 r(x)'r(x)
                    // Test f(x + p) < f(x) - cα(J(x)'r(x))'p(x)

                    let t_begin_jprod = Instant::now();
                    //jp.iter_mut().for_each(|x| *x = T::zero());
                    //j.view()
                    //    .add_mul_in_place_par(p.as_tensor(), jp.as_mut_tensor());
                    problem.jacobian_product(x, p, r_cur, j_rows, j_cols, jp.as_mut_slice());
                    jprod_ls_time += Instant::now() - t_begin_jprod;

                    // TRADITIONAL BACKTRACKING:
                    //let rjp = jp
                    //    .iter()
                    //    .zip(r_cur.iter())
                    //    .fold(0.0, |acc, (&jp, &r)| acc + (jp * r).to_f64().unwrap());
                    //if (&r_next).as_tensor().norm_squared().to_f64().unwrap()
                    //    <= (&r_cur).as_tensor().norm_squared().to_f64().unwrap()
                    //        - 2.0 * params.line_search.armijo_coeff() * alpha * rjp
                    //{
                    //    break;
                    //}

                    r_next_norm = r_next.as_tensor().norm().to_f64().unwrap();
                    //log::trace!("Next     r norm: {}", r_next_norm);
                    if r_next_norm <= r_cur_norm * (1.0 - rho * (1.0 - sigma)) {
                        break;
                    }

                    alpha *= rho;
                    sigma = 1.0 - alpha * (1.0 - sigma);

                    if alpha < 1e-3 {
                        break;
                    }

                    // Take a fractional step.
                    zip!(x.iter_mut(), x_prev.iter(), p.iter()).for_each(|(x, &x0, &p)| {
                        *x = num_traits::Float::mul_add(p, T::from(-alpha).unwrap(), x0);
                    });

                    // Compute the candidate residual.
                    let before_residual = Instant::now();
                    problem.residual(x, r_next.as_mut_slice());
                    residual_time += Instant::now() - before_residual;

                    ls_count += 1;
                }
                ls_count
            };

            iterations += 1;

            ls_time += Instant::now() - t_begin_ls;

            log_debug_stats(
                iterations,
                ls_count,
                linsolve_result,
                linsolve.tol,
                &r_next,
                x,
                &x_prev,
            );

            let denom = x.as_tensor().norm() + T::one();

            // Check the convergence condition.
            if r_next_norm < r_tol.to_f64().unwrap()
                && num_traits::Float::sqrt(
                    x_prev
                        .iter()
                        .zip(x.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<T>(),
                ) < x_tol * denom
            {
                break SolveResult {
                    iterations,
                    status: Status::Success,
                };
            }

            // Check that we are running no more than the maximum allowed iterations.
            if iterations >= params.max_iter {
                break SolveResult {
                    iterations,
                    status: Status::MaximumIterationsExceeded,
                };
            }

            // Reset r to be a valid residual for the next iteration.
            r.copy_from_slice(&r_next);

            // Update norms
            r_prev_norm = r_cur_norm;
            r_cur_norm = r_next_norm;
        };

        // Reset tolerance for future solves.
        linsolve.tol = orig_linsolve_tol;

        log::debug!(
            "Balance equation computation time: {}ms",
            residual_time.as_millis()
        );
        log::debug!("Linear solve time: {}ms", linsolve_time.as_millis());
        log::debug!(
            "   Jacobian product time: {}ms",
            jprod_linsolve_time.as_millis()
        );
        log::debug!("Line search time: {}ms", ls_time.as_millis());
        log::debug!("   Jacobian product time: {}ms", jprod_ls_time.as_millis());
        result
    }
}

/*
 * Status print routines.
 * i       - iteration number
 * res-2   - 2-norm of the residual
 * res-inf - inf-norm of the residual
 * d-inf   - inf-norm of the step vector
 * x-inf   - inf-norm of the variable vector
 * lin #   - number of linear solver iterations
 * ls #    - number of line search steps
 */
fn log_debug_stats_header() {
    log::debug!(
        "    i |   res-2    |   d-inf    |   x-inf    | lin # |  lin err   |   sigma    | ls # "
    );
    log::debug!(
        "------+------------+------------+------------+-------+------------+------------+------"
    );
}
fn log_debug_stats<T: Real>(
    iterations: u32,
    ls_steps: u32,
    linsolve_result: super::linsolve::SolveResult,
    sigma: f32,
    r: &[T],
    x: &[T],
    x_prev: &[T],
) {
    log::debug!(
        "{i:>5} |  {res2:10.3e} | {di:10.3e} | {xi:10.3e} | {lin:>5} | {linerr:10.3e} | {sigma:10.3e} | {ls:>4} ",
        i = iterations,
        res2 = r.as_tensor().norm().to_f64().unwrap(),
        di = inf_norm(x_prev.iter().zip(x.iter()).map(|(&a, &b)| a - b))
            .to_f64()
            .unwrap(),
        xi = inf_norm(x.iter().cloned()).to_f64().unwrap(),
        lin = linsolve_result.iterations,
        linerr = linsolve_result.error,
        sigma = sigma,
        ls = ls_steps
    );
}

/// Solves a sparse symmetric indefinite system `Ax = b`.
///
/// Here `A` is the lower triangular part of a symmetric sparse matrix in CSR format,
/// `b` is a dense right hand side vector and the output `x` is computed in `b`.
#[allow(non_snake_case)]
#[allow(dead_code)]
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
fn eig<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>) {
    // nalgebra dense prototype using lu.
    let mut dense = na::DMatrix::zeros(mtx.len(), mtx.len());

    for (row_idx, row) in mtx.iter().enumerate() {
        for (col_idx, val) in row.iter().enumerate() {
            *dense.index_mut((row_idx, col_idx)) = *val;
        }
    }

    log::debug!("J eigenvalues: {:?}", dense.complex_eigenvalues());
}

fn svd_values<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>) {
    // nalgebra dense prototype using lu.
    let mut dense = na::DMatrix::zeros(mtx.len(), mtx.len());

    for (row_idx, row) in mtx.iter().enumerate() {
        for (col_idx, val) in row.iter().enumerate() {
            *dense.index_mut((row_idx, col_idx)) = *val;
        }
    }

    log::debug!("J singular values: {:?}", dense.singular_values());
}
#[allow(dead_code)]
fn write_jacobian_img<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>, iter: u32) {
    // nalgebra dense prototype using lu.
    let mut dense = na::DMatrix::zeros(mtx.len(), mtx.len());

    for (row_idx, row) in mtx.iter().enumerate() {
        for (col_idx, val) in row.iter().enumerate() {
            *dense.index_mut((row_idx, col_idx)) = val.to_f64().unwrap();
        }
    }
    super::problem::write_jacobian_img(&dense, iter);
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

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
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
