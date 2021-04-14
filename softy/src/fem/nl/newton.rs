use std::cell::RefCell;

use super::problem::NonLinearProblem;
use super::{Callback, CallbackArgs, NLSolver, SolveResult, Status};
use crate::inf_norm;
use tensr::*;

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

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct NewtonWorkspace<T> {
    x_prev: Vec<T>,
    r: Vec<T>,
    jtr: Vec<T>,
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

pub struct Newton<P, T> {
    pub problem: P,
    pub params: NewtonParams,
    pub intermediate_callback: RefCell<Callback<T>>,
    pub workspace: RefCell<NewtonWorkspace<T>>,
}

impl<T, P> Newton<P, T>
where
    T: Real + na::RealField,
    P: NonLinearProblem<T>,
{
    pub fn new(problem: P, params: NewtonParams, intermediate_callback: Callback<T>) -> Self {
        let n = problem.num_variables();
        // Initialize previous iterate.
        let x_prev = vec![T::zero(); n];

        let r = vec![T::zero(); n];
        let jtr = r.clone();
        let r_next = r.clone();
        let r_cur = r.clone();

        // Construct the sparse Jacobian.

        let (j_rows, j_cols) = problem.jacobian_indices();
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

        Newton {
            problem,
            params,
            intermediate_callback: RefCell::new(intermediate_callback),
            workspace: RefCell::new(NewtonWorkspace {
                x_prev,
                r,
                jtr,
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
}

impl<T, P> NLSolver<P, T> for Newton<P, T>
where
    T: Real + na::RealField,
    P: NonLinearProblem<T>,
{
    /// Gets a reference to the intermediate callback function.
    fn intermediate_callback(&self) -> &RefCell<Callback<T>> {
        &self.intermediate_callback
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
    fn solve(&self) -> (Vec<T>, SolveResult) {
        let mut x = self.problem.initial_point();
        let res = self.solve_with(x.as_mut_slice());
        (x, res)
    }

    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&self, x: &mut [T]) -> SolveResult {
        let NewtonWorkspace {
            r,
            jtr,
            r_cur,
            r_next,
            j_rows,
            j_cols,
            j_vals,
            j_mapping,
            j,
            x_prev,
        } = &mut *self.workspace.borrow_mut();

        let mut iterations = 0;

        // Initialize the residual.
        self.problem.residual(x, r.as_mut_slice());

        //let cr = ConjugateResidual::new(
        //    |x, out| {

        //    },
        //    x,
        //    r,
        //);

        // Convert to sprs format for debugging. The CSR structure is preserved.
        let mut j_sprs: sprs::CsMat<T> = j.clone().into();

        let r_tol = T::from(self.params.r_tol).unwrap();
        let x_tol = T::from(self.params.x_tol).unwrap();

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
            self.problem
                .advanced_jacobian_values(x, &r, &j_rows, &j_cols, j_vals.as_mut_slice());

            //log::trace!("j_vals = {:?}", &j_vals);

            // Zero out Jacobian.
            j.storage_mut().iter_mut().for_each(|x| *x = T::zero());
            j_sprs.data_mut().iter_mut().for_each(|x| *x = T::zero());

            // Update the Jacobian matrix.
            for (&pos, &j_val) in j_mapping.iter().zip(j_vals.iter()) {
                j.storage_mut()[pos] += j_val;
                j_sprs.data_mut()[pos] += j_val;
            }

            r_cur.copy_from_slice(&r);

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
            //let mut out = vec![0.0; self.problem.num_variables()];
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
            //        self.problem.residual(&probe_x, probe_r.as_mut_slice());
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

            // Update previous step.
            x_prev.copy_from_slice(x);

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
                    zip!(x.iter_mut(), x_prev.iter(), p.iter()).for_each(|(x, &x0, &p)| {
                        *x = num_traits::Float::mul_add(p, T::from(-alpha).unwrap(), x0);
                    });

                    // Compute the candidate residual.
                    self.problem.residual(&x, r_next.as_mut_slice());

                    // Compute gradient of the merit function 0.5 r'r, which is r' dr/dx.
                    jtr.iter_mut().for_each(|x| *x = T::zero());
                    j.view()
                        .add_left_mul_in_place(r_cur.as_tensor(), jtr.as_mut_tensor());

                    // Residual dot search direction.
                    // Check Armijo condition:
                    // Given f(x) = 0.5 || r(x) ||^2 = 0.5 r(x)'r(x)
                    // Test f(x + p) < f(x) - cα(J(x)'r(x))'p(x)
                    let jtr_dot_p = jtr
                        .iter()
                        .zip(p.iter())
                        .fold(0.0, |acc, (&jtr, &p)| acc + (jtr * p).to_f64().unwrap());
                    if (&r_next).as_tensor().norm_squared().to_f64().unwrap()
                        <= (&r_cur).as_tensor().norm_squared().to_f64().unwrap()
                            - 2.0 * self.params.line_search.armijo_coeff() * alpha * jtr_dot_p
                    {
                        break;
                    }

                    alpha *= rho;

                    //if alpha < 1e-3 {
                    //    break;
                    //}

                    ls_count += 1;
                }
                ls_count
            };

            iterations += 1;

            log_debug_stats(iterations, ls_count, &r_next, x, &x_prev);

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

/// Implementation of the conjugate residual method.
///
/// The Conjugate Residual method solves the system `Ax = b` where `A` is Hermitian, and `b` is non-zero.
/// https://en.wikipedia.org/wiki/Conjugate_residual_method
#[allow(non_snake_case)]
pub struct ConjugateResidual<'a, T, F> {
    /// Function to compute the `Ax` product.
    Ax: F,
    x: &'a mut [T],
    r: &'a mut [T],
    p: Vec<T>,
    Ar: Vec<T>,
    Ap: Vec<T>,
    /// Preconditioner.
    M: Option<DSMatrix<T>>,
}

impl<'a, T, F> ConjugateResidual<'a, T, F>
where
    T: Real,
    F: FnMut(&[T], &mut [T]),
{
    #[allow(non_snake_case)]
    pub fn new(mut Ax: F, x: &'a mut [T], b: &'a mut [T]) -> Self {
        let mut p = vec![T::zero(); x.len()];

        // r0 = b - Ax0
        Ax(x, &mut p);
        b.iter_mut().zip(p.iter()).for_each(|(b, &p)| *b -= p);

        // p0 = r0
        p.copy_from_slice(b);

        // Initialize Ap0 & Ar0
        let mut Ar = vec![T::zero(); x.len()];
        Ax(b, &mut Ar);

        let Ap = Ar.clone();

        ConjugateResidual {
            Ax,
            x,
            r: b,
            p,
            Ar,
            Ap,
            M: None,
        }
    }

    //pub fn solve()
}
