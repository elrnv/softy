use std::convert::TryFrom;
use std::cell::RefCell;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use accelerate::*;
use lazycell::LazyCell;
#[cfg(not(target_os = "macos"))]
use mkl_corrode as mkl;
use serde::{Deserialize, Serialize};
use tensr::*;
use thiserror::Error;

use super::linsolve::*;
use super::problem::NonLinearProblem;
use super::{Callback, CallbackArgs, NLSolver, SolveResult, Status};
use crate::Index;
use crate::Real;

// Parameters for the Newton solver.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct NewtonParams {
    /// Residual tolerance.
    pub r_tol: f32,
    /// Variable tolerance.
    pub x_tol: f32,
    /// Acceleration tolerance.
    pub a_tol: f32,
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

#[derive(Debug, Copy, Clone, PartialEq, Error)]
pub enum SparseSolveError {
    #[error("Iterative solve error")]
    Iteartive(#[from] SparseIterativeSolveError),
    #[error("Direct solve error")]
    Direct(#[from] SparseDirectSolveError),
}

#[derive(Debug, Copy, Clone, PartialEq, Error)]
pub enum SparseIterativeSolveError {
    #[error("Reached the maximum number of iterations")]
    MaxIterations,
    #[error("Parameter error")]
    ParameterError,
    #[error("Matrix is ill-conditioned")]
    IllConditioned,
    #[error("Internal error")]
    InternalError,
}

impl SparseIterativeSolveError {
    #[cfg(target_os = "macos")]
    pub fn result_from_status<T>(
        val: T,
        status: SparseIterativeStatus,
    ) -> Result<T, SparseIterativeSolveError> {
        match status {
            SparseIterativeStatus::Converged => Ok(val),
            SparseIterativeStatus::MaxIterations => Err(SparseIterativeSolveError::MaxIterations),
            SparseIterativeStatus::ParameterError => Err(SparseIterativeSolveError::ParameterError),
            SparseIterativeStatus::IllConditioned => Err(SparseIterativeSolveError::IllConditioned),
            SparseIterativeStatus::InternalError => Err(SparseIterativeSolveError::InternalError),
        }
    }
    #[cfg(target_os = "macos")]
    pub fn status(self) -> SparseIterativeStatus {
        match self {
            SparseIterativeSolveError::MaxIterations => SparseIterativeStatus::MaxIterations,
            SparseIterativeSolveError::ParameterError => SparseIterativeStatus::ParameterError,
            SparseIterativeSolveError::IllConditioned => SparseIterativeStatus::IllConditioned,
            SparseIterativeSolveError::InternalError => SparseIterativeStatus::InternalError,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Error)]
pub enum SparseDirectSolveError {
    #[error("Factorization failed")]
    FactorizationFailed,
    #[error("Matrix is singular")]
    MatrixIsSingular,
    #[error("Internal error")]
    InternalError,
    #[error("Parameter error")]
    ParameterError,
    #[error("Matrix released")]
    Released,
    #[cfg(not(target_os = "macos"))]
    #[error("MKL error")]
    MKLError(#[from] mkl::dss::Error),
}

impl SparseDirectSolveError {
    #[cfg(target_os = "macos")]
    pub fn result_from_status<T>(
        val: T,
        status: SparseStatus,
    ) -> Result<T, SparseDirectSolveError> {
        match status {
            SparseStatus::Ok => Ok(val),
            SparseStatus::FactorizationFailed => Err(SparseDirectSolveError::FactorizationFailed),
            SparseStatus::MatrixIsSingular => Err(SparseDirectSolveError::MatrixIsSingular),
            SparseStatus::InternalError => Err(SparseDirectSolveError::InternalError),
            SparseStatus::ParameterError => Err(SparseDirectSolveError::ParameterError),
            SparseStatus::Released => Err(SparseDirectSolveError::Released),
        }
    }
    #[cfg(target_os = "macos")]
    pub fn status(self) -> SparseStatus {
        match self {
            SparseDirectSolveError::FactorizationFailed => SparseStatus::FactorizationFailed,
            SparseDirectSolveError::MatrixIsSingular => SparseStatus::MatrixIsSingular,
            SparseDirectSolveError::InternalError => SparseStatus::InternalError,
            SparseDirectSolveError::ParameterError => SparseStatus::ParameterError,
            SparseDirectSolveError::Released => SparseStatus::Released,
        }
    }
}

#[cfg(target_os = "macos")]
pub enum Factorization {
    Symbolic(SparseSymbolicFactorization),
    Numeric(SparseFactorization<f64>),
}

#[cfg(target_os = "macos")]
impl From<SparseSymbolicFactorization> for Factorization {
    fn from(s: SparseSymbolicFactorization) -> Factorization {
        Factorization::Symbolic(s)
    }
}

#[cfg(target_os = "macos")]
impl Factorization {
    fn factor(&mut self, mtx: &SparseMatrixF64) -> Result<(), SparseDirectSolveError> {
        //let nfopts = SparseNumericFactorOptions {
        //    pivot_tolerance: 0.001,
        //    zero_tolerance: 0.001,
        //    ..Default::default()
        //};
        match self {
            Factorization::Symbolic(sym) => {
                //let f = mtx.factor_with_options(sym, nfopts);
                let f = mtx.factor_with(sym);
                let status = f.status();
                *self = Factorization::Numeric(f);
                SparseDirectSolveError::result_from_status((), status)
            }
            Factorization::Numeric(num) => {
                //let status = num.refactor_with_options(mtx, nfopts);
                let status = num.refactor(mtx);
                SparseDirectSolveError::result_from_status((), status)
            }
        }
    }

    /// Solve system using the current numerical factorization.
    ///
    /// # Panics
    ///
    /// This function will panic if the matrix has not yet been numerically factorized.
    /// To avoid panics, call `factor` before calling this function.
    fn solve_in_place<'a>(&self, rhs: &'a mut [f64]) -> Result<&'a [f64], SparseDirectSolveError> {
        match self {
            Factorization::Symbolic(_) => unreachable!(),
            Factorization::Numeric(num) => {
                let status = num.solve_in_place(&mut *rhs);
                SparseDirectSolveError::result_from_status(&*rhs, status)
            }
        }
    }
}

/// Solver solely responsible for the sparse iterative linear solve.
///
/// This is done via third party libraries like MKL or Accelerate.
/// This struct also helps isolate conditionally compiled code from the rest of the solver.
#[cfg(target_os = "macos")]
pub struct SparseIterativeSolver {
    b64: Vec<f64>,
    r64: Vec<f64>,
    mtx: SparseMatrixF64<'static>,
}

#[cfg(target_os = "macos")]
impl SparseIterativeSolver {
    pub fn new<T: Real>(m_t: DSMatrixView<T>) -> SparseIterativeSolver {
        let num_variables = m_t.num_cols();
        let mtx = new_sparse(m_t, true);
        SparseIterativeSolver {
            b64: vec![0.0; num_variables],
            r64: vec![0.0; num_variables],
            mtx,
        }
    }

    pub fn update_values<T: Real>(&mut self, data: &[T]) {
        assert_eq!(data.len(), self.mtx.data().len());
        self.mtx
            .data_mut()
            .iter_mut()
            .zip(data.iter())
            .for_each(|(output, input)| *output = input.to_f64().unwrap());
    }

    pub fn update_rhs<T: Real>(&mut self, b: &[T]) {
        self.b64
            .iter_mut()
            .zip(b.iter())
            .for_each(|(out_f64, in_t)| *out_f64 = in_t.to_f64().unwrap());
    }

    pub fn solve_with_values<T: Real>(
        &mut self,
        b: &[T],
        values: &[T],
    ) -> Result<&[f64], SparseIterativeSolveError> {
        self.update_rhs(&b);
        self.update_values(values);
        self.solve()
    }

    pub fn solve(&mut self) -> Result<&[f64], SparseIterativeSolveError> {
        let method = SparseIterativeMethod::lsmr();
        let status = method.solve_precond(
            &self.mtx,
            &mut self.b64,
            &mut self.r64,
            SparsePreconditioner::Diagonal,
        );
        SparseIterativeSolveError::result_from_status(&self.r64, status)
    }
}

/// Solver solely responsible for the sparse direct linear solve.
///
/// This is done via third party libraries like MKL or Accelerate.
/// This struct also helps isolate conditionally compiled code from the rest of the solver.
pub struct SparseDirectSolver {
    r64: Vec<f64>,
    #[cfg(target_os = "macos")]
    mtx: SparseMatrixF64<'static>,
    #[cfg(target_os = "macos")]
    factorization: Factorization,
    #[cfg(not(target_os = "macos"))]
    solver: mkl::dss::Solver<f64>,
    #[cfg(not(target_os = "macos"))]
    buf: Vec<f64>,
    #[cfg(not(target_os = "macos"))]
    sol: Vec<f64>,
}

#[cfg(target_os = "macos")]
// Given a transpose matrix in CSR, returns a new CSC sparse matrix with the same sparsity pattern.
fn new_sparse<T: Real>(m: DSMatrixView<T>, transposed: bool) -> SparseMatrixF64<'static> {
    let (num_rows, num_cols) = if transposed {
        (m.num_cols(), m.num_rows())
    } else {
        (m.num_rows(), m.num_cols())
    };
    let m = m.into_data();
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    values.reserve(m.storage().len());
    row_indices.reserve(m.storage().len());
    col_indices.reserve(m.storage().len());
    for (row_idx, row) in m.into_data().iter().enumerate() {
        for (col_idx, val) in row.into_iter() {
            values.push(val.to_f64().unwrap());
            let (row_idx, col_idx) = if transposed {
                (col_idx, row_idx)
            } else {
                (row_idx, col_idx)
            };
            row_indices.push(i32::try_from(row_idx).unwrap());
            col_indices.push(i32::try_from(col_idx).unwrap());
        }
    }

    SparseMatrixF64::from_coordinate(
        i32::try_from(num_rows).unwrap(),
        i32::try_from(num_cols).unwrap(),
        1,
        // Since the output is CSC whereas the input is CSR, the transpose roles are reversed:
        if transposed {
            SparseAttributes::new()
        } else {
            SparseAttributes::new().transposed()
        },
        col_indices.as_slice(),
        row_indices.as_slice(),
        values.as_slice(),
    )
}

// Assumes values are in csc order.
fn update_sparse_values<T: Real>(m: &mut SparseMatrixF64, values: &[T]) {
    m
        .data_mut()
        .iter_mut()
        .zip(values.iter())
        .for_each(|(output, input)| *output = input.to_f64().unwrap());
}

impl SparseDirectSolver {
    #[cfg(target_os = "macos")]
    pub fn new<T: Real>(m_t: DSMatrixView<T>) -> Option<Self> {
        let num_variables = m_t.num_cols();
        let mtx = new_sparse(m_t, true);
        let t_begin_symbolic = Instant::now();
        let symbolic_factorization = mtx.structure().symbolic_factor(SparseFactorizationType::QR);
        let t_end_symbolic = Instant::now();
        log::debug!(
            "Symbolic time: {}ms",
            (t_end_symbolic - t_begin_symbolic).as_millis()
        );
        Some(SparseDirectSolver {
            r64: vec![0.0; num_variables],
            mtx,
            factorization: Factorization::from(symbolic_factorization),
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new<T: Real>(m: DSMatrixView<T>) -> Option<Self> {
        let num_variables = m.num_rows();
        let m = m.into_data();
        let values: Vec<_> = m.storage().iter().map(|x| x.to_f64().unwrap()).collect();
        let row_offsets = m.chunks.as_ref();
        let col_indices = m.data.indices().storage();

        let mtx = mkl::dss::SparseMatrix::try_convert_from_csr(
            row_offsets,
            col_indices,
            values.as_slice(),
            mkl::dss::MatrixStructure::NonSymmetric,
        )
        .ok()?;
        let t_begin = Instant::now();
        let solver = mkl::dss::Solver::try_factor(&mtx, mkl::dss::Definiteness::Indefinite).ok()?;
        let t_end = Instant::now();
        log::debug!("First factor time: {}ms", (t_end - t_begin).as_millis());
        Some(SparseDirectSolver {
            r64: vec![0.0; num_variables],
            solver,
            buf: vec![0.0; num_variables],
            sol: vec![0.0; num_variables],
        })
    }

    #[cfg(target_os = "macos")]
    pub fn update_values<T: Real>(&mut self, data: &[T]) {
        assert_eq!(data.len(), self.mtx.data().len());
        update_sparse_values(&mut self.mtx, data);
    }

    #[cfg(target_os = "macos")]
    pub fn refactor(&mut self) -> Result<(), SparseDirectSolveError> {
        self.factorization.factor(&self.mtx)
    }

    #[cfg(target_os = "macos")]
    pub fn solve(&mut self) -> Result<&[f64], SparseDirectSolveError> {
        self.factorization.solve_in_place(&mut self.r64)
    }

    pub fn update_rhs<T: Real>(&mut self, r: &[T]) {
        self.r64
            .iter_mut()
            .zip(r.iter())
            .for_each(|(out_f64, in_t)| *out_f64 = in_t.to_f64().unwrap());
    }

    #[cfg(not(target_os = "macos"))]
    pub fn solve_with_values<T: Real>(
        &mut self,
        r: &[T],
        values: &[T],
    ) -> Result<&[f64], SparseDirectSolveError> {
        let t_begin = Instant::now();
        // Update values and refactor
        self.update_rhs(&r);
        let t_update = Instant::now();
        let values: Vec<_> = values.iter().map(|&x| x.to_f64().unwrap()).collect();
        self.solver
            .refactor(values.as_slice(), mkl::dss::Definiteness::Indefinite);
        let t_factor = Instant::now();
        self.solver
            .solve_into(&mut self.sol, &mut self.buf, &mut self.r64)?;
        let t_end = Instant::now();
        Ok(&self.sol)
    }

    #[cfg(target_os = "macos")]
    pub fn solve_with_values<T: Real>(
        &mut self,
        r: &[T],
        values: &[T],
    ) -> Result<&[f64], SparseDirectSolveError> {
        self.update_values(values);
        self.update_rhs(&r);
        self.refactor()?;
        self.solve()
    }
}

pub struct SparseJacobian<T: Real> {
    j_rows: Vec<usize>,
    j_cols: Vec<usize>,
    j_vals: Vec<T>,
    /// Mapping from original triplets given by the `j_*` members to the final
    /// compressed sparse matrix.
    j_mapping: Vec<Index>,
    j_t_mapping: Vec<Index>,
    j: DSMatrix<T>,
    j_t: DSMatrix<T>,
    #[cfg(target_os = "macos")]
    j_sparse: LazyCell<SparseMatrixF64<'static>>,
    sparse_solver: LazyCell<SparseDirectSolver>,
    #[cfg(target_os = "macos")]
    sparse_iterative_solver: LazyCell<SparseIterativeSolver>,
    p64: Vec<f64>,
    out64: Vec<f64>,
}

pub struct NewtonWorkspace<T: Real> {
    linsolve: BiCGSTAB<T>,
    x_prev: Vec<T>,
    r: Vec<T>,
    p: Vec<T>,
    jp: Vec<T>,
    r_cur: Vec<T>,
    r_next: Vec<T>,
    sparse_jacobian: SparseJacobian<T>,
}

unsafe impl<T: Real> Send for NewtonWorkspace<T> {}

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

fn sparse_matrix_and_mapping<T: Real>(
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
    mtx_size: usize,
) -> (DSMatrix<T>, Vec<Index>) {
    let nnz = rows.len();
    assert_eq!(nnz, cols.len());
    assert_eq!(nnz, vals.len());
    // Construct a mapping from original triplets to final compressed matrix.
    let mut entries = (0..nnz).collect::<Vec<_>>();

    entries.sort_by(|&a, &b| rows[a].cmp(&rows[b]).then_with(|| cols[a].cmp(&cols[b])));

    let mut mapping = vec![Index::INVALID; entries.len()];
    let entries = entries
        .into_iter()
        .filter(|&i| rows[i] < mtx_size && cols[i] < mtx_size)
        .collect::<Vec<_>>();

    // We use tensr to build the CSR matrix since it allows us to track
    // where each element goes after compression.
    let triplet_iter = entries.iter().map(|&i| (rows[i], cols[i], vals[i]));

    let uncompressed =
        DSMatrix::from_sorted_triplets_iter_uncompressed(triplet_iter, mtx_size, mtx_size);

    // Compress the CSR matrix.
    let mtx = uncompressed.pruned(
        |_, _, _| true,
        |src, dst| {
            mapping[entries[src]] = Index::new(dst);
        },
    );
    (mtx, mapping)
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
                sparse_jacobian: SparseJacobian {
                    j_vals: Vec::new(),
                    j_cols: Vec::new(),
                    j_rows: Vec::new(),
                    j: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
                    j_t: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
                    j_sparse: LazyCell::new(),
                    j_mapping: Vec::new(),
                    j_t_mapping: Vec::new(),
                    sparse_solver: LazyCell::new(),
                    #[cfg(target_os = "macos")]
                    sparse_iterative_solver: LazyCell::new(),
                    p64: Vec::new(),
                    out64: Vec::new(),
                },
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
    fn update_jacobian_indices(&mut self) {
        // Construct the sparse Jacobian.
        let n = self.problem().num_variables();
        let (j_rows, j_cols) = self.problem.jacobian_indices();
        assert_eq!(j_rows.len(), j_cols.len());
        let j_nnz = j_rows.len();
        let mut ws = self.workspace.borrow_mut();
        let sj = &mut ws.sparse_jacobian;
        sj.j_vals.resize(j_nnz, T::zero());
        let (j, j_mapping) = sparse_matrix_and_mapping(&j_rows, &j_cols, &sj.j_vals, n);
        let (j_t, j_t_mapping) = sparse_matrix_and_mapping(&j_cols, &j_rows, &sj.j_vals, n);
        sj.j_cols = j_cols;
        sj.j_rows = j_rows;
        sj.j = j;
        #[cfg(target_os = "macos")]
        sj.j_sparse.replace(new_sparse(j_t.view(), true));
        sj.j_t = j_t;
        sj.j_mapping = j_mapping;
        sj.j_t_mapping = j_t_mapping;
        #[cfg(target_os = "macos")]
        sj.sparse_solver
            .replace(SparseDirectSolver::new(sj.j_t.view()).unwrap());
        #[cfg(not(target_os = "macos"))]
            sj.sparse_solver
            .replace(SparseDirectSolver::new(sj.j.view()).unwrap());
        #[cfg(target_os = "macos")]
        sj.sparse_iterative_solver
            .replace(SparseIterativeSolver::new(sj.j_t.view()));
        sj.p64.resize(n, 0.0);
        sj.out64.resize(n, 0.0);
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
        if x.is_empty() {
            return SolveResult {
                iterations: 0,
                status: Status::NothingToSolve,
            };
        }
        let Self {
            problem,
            params,
            inner_callback,
            outer_callback,
            workspace,
            ..
        } = self;

        let NewtonWorkspace {
            linsolve,
            x_prev,
            r,
            p,
            jp,
            r_cur,
            r_next,
            sparse_jacobian,
        } = &mut *workspace.borrow_mut();

        let SparseJacobian {
            j_rows,
            j_cols,
            j_vals,
            j_mapping,
            j_t_mapping,
            j,
            j_t,
            #[cfg(target_os = "macos")]
            j_sparse,
            sparse_solver,
            #[cfg(target_os = "macos")]
            sparse_iterative_solver,
            p64,
            out64: _,
        } = sparse_jacobian;

        // Unwrap the sparse solver. In case of panic check update_jacobian_indices function.
        let sparse_solver = sparse_solver
            .borrow_mut()
            .expect("Uninitialized direct sparse solver.");

        #[cfg(target_os = "macos")]
        let sparse_iterative_solver = sparse_iterative_solver
            .borrow_mut()
            .expect("Uninitialized iterative sparse solver.");

        let mut iterations = 0;

        // Initialize the residual.
        problem.residual(x, r.as_mut_slice());

        let a_tol = T::from(params.a_tol).unwrap();
        let r_tol = T::from(params.r_tol).unwrap();
        let x_tol = T::from(params.x_tol).unwrap();

        let mut linsolve_result = super::linsolve::SolveResult::default();

        log_debug_stats_header();
        log_debug_stats(0, 0, linsolve_result, linsolve.tol, std::f64::INFINITY, &r, x, &x_prev);

        // Remember original tolerance so we can reset it later.
        let orig_linsolve_tol = linsolve.tol;

        // Timing stats
        let mut linsolve_time = Duration::new(0, 0);
        let mut ls_time = Duration::new(0, 0);
        let mut jprod_linsolve_time = Duration::new(0, 0);
        let mut jprod_ls_time = Duration::new(0, 0);
        let mut residual_time = Duration::new(0, 0);

        //let merit = |r: &[T]| 0.5 * r.as_tensor().norm_squared().to_f64().unwrap();
        let merit = |_: &[T]| problem.objective(x).to_f64().unwrap();

        // Keep track of merit function to avoid having to recompute it
        let mut merit_cur = merit(r);
        //let mut merit_prev = merit_cur;
        let mut merit_next;

        let mut j_dense =
            ChunkedN::from_flat_with_stride(x.len(), vec![T::zero(); x.len() * r.len()]);
        //let mut identity =
        //    ChunkedN::from_flat_with_stride(x.len(), vec![T::zero(); x.len() * r.len()]);
        //for (i, id) in identity.iter_mut().enumerate() {
        //    id[i] = T::one();
        //}

        let result = loop {
            //log::trace!("Previous r norm: {}", r_prev_norm);
            //log::trace!("Current  r norm: {}", r_cur_norm);
            if !(outer_callback.borrow_mut())(CallbackArgs {
                iteration: iterations,
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
            problem.jacobian_values(x, &r, &j_rows, &j_cols, j_vals.as_mut_slice());
            // TODO: For debugging only
            //for (jp, p) in j_dense.iter_mut().zip(identity.iter()) {
            //    problem.jacobian_product(x, &p, &r, jp)
            //}
            build_dense(j_dense.view_mut(), &j_rows, &j_cols, &j_vals, x.len());
            //print_dense(j_dense.view());
            //log::debug!("J singular values: {:?}", svd_values(j_dense.view()));
            log::debug!("Condition number: {:?}", condition_number(j_dense.view()));
            //write_jacobian_img(j_dense.view(), iterations);
            //jprod_time += Instant::now() - before_j;

            ////log::trace!("j_vals = {:?}", &j_vals);

            //// Zero out Jacobian.
            j.storage_mut().iter_mut().for_each(|x| *x = T::zero());
            j_t.storage_mut().iter_mut().for_each(|x| *x = T::zero());

            // Update the Jacobian matrix.
            for (&pos, &j_val) in j_mapping.iter().zip(j_vals.iter()) {
                if let Some(pos) = pos.into_option() {
                    j.storage_mut()[pos] += j_val;
                }
            }
            for (&pos, &j_val) in j_t_mapping.iter().zip(j_vals.iter()) {
                if let Some(pos) = pos.into_option() {
                    j_t.storage_mut()[pos] += j_val;
                }
            }

            //log::trace!("r = {:?}", &r);

            if !merit_cur.is_finite() {
                break SolveResult {
                    iterations,
                    status: Status::Diverged,
                };
            }

            let t_begin_linsolve = Instant::now();

            // Update tolerance (forcing term)
            //linsolve.tol = orig_linsolve_tol
            //    .min(((merit_cur - linsolve_result.residual).abs() / merit_prev) as f32);

            r_cur.copy_from_slice(&r);

            #[cfg(target_os = "macos")]
            update_sparse_values(j_sparse.borrow_mut().unwrap(), j_t.storage());

            //linsolve_result = linsolve.solve(
            //    |p, out| {
            //        let t_begin_jprod = Instant::now();
            //        //problem.jacobian_product(x, p, r_cur, out);
            //        p.iter().zip(p64.iter_mut()).for_each(|(&x,p64)| *p64 = x.to_f64().unwrap());
            //        out64.iter_mut().for_each(|x| *x = 0.0);
            //        j_sparse.borrow().unwrap().add_mul_vec(p64, out64);
            //        out.iter_mut().zip(out64.iter()).for_each(|(out, &out64)| *out = T::from(out64).unwrap());
            //        jprod_linsolve_time += Instant::now() - t_begin_jprod;
            //        inner_callback.borrow_mut()(CallbackArgs {
            //            residual: r_cur.as_slice(),
            //            x,
            //            problem,
            //            iteration: iterations,
            //        })
            //    },
            //    p.as_mut_slice(),
            //    r.as_mut_slice(),
            //);

            //log::trace!("linsolve result: {:?}", linsolve_result);

            //let result = sparse_iterative_solver.solve_with_values(&r, j_t.storage());
            let result = sparse_solver.solve_with_values(&r, j_t.storage());
            let r64 = match result {
                Err(err) => {
                    break SolveResult {
                        iterations,
                        status: Status::LinearSolveError(err.into()),
                    }
                }
                Ok(r) => r,
            };

            // r is now the search direction, rename to avoid confusion.
            p.iter_mut()
                .zip(r64.iter())
                .for_each(|(p, &r64)| *p = T::from(r64).unwrap());

            //log::trace!("p = {:?}", &r);

            linsolve_time += Instant::now() - t_begin_linsolve;

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
                merit_next = problem.objective(x).to_f64().unwrap();//merit(r_next);
                1
            } else {
                // Line search.
                let mut alpha = 1.0;
                let mut ls_count = 1;
                //let mut sigma = linsolve.tol as f64;

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
                    problem.jacobian_product(x, p, r_cur, jp.as_mut_slice());
                    jprod_ls_time += Instant::now() - t_begin_jprod;

                    // Compute the merit function
                    merit_next = problem.objective(x).to_f64().unwrap();//merit(r_next);

                    // TRADITIONAL BACKTRACKING:
                    let rjp = jp
                        .iter()
                        .zip(r_cur.iter())
                        .fold(0.0, |acc, (&jp, &r)| acc + (jp * r).to_f64().unwrap());
                    if merit_next <= merit_cur - params.line_search.armijo_coeff() * alpha * rjp {
                        break;
                    }

                    // INEXACT NEWTON:
                    // if merit_next <= merit_cur * (1.0 - rho * (1.0 - sigma)) {
                    //     break;
                    // }
                    //
                    alpha *= rho;
                    // sigma = 1.0 - alpha * (1.0 - sigma);

                    // Break if alpha becomes too small. This is usually a bad sign.
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
                merit_next,
                &r_next,
                x,
                &x_prev,
            );

            let denom = x.as_tensor().norm() + T::one();

            let dx_norm = num_traits::Float::sqrt(
                x_prev
                    .iter()
                    .zip(x.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<T>(),
            );

            // Check the convergence condition.
            if (r_tol > T::zero() && merit_next < r_tol.to_f64().unwrap())
                || (x_tol > T::zero() && dx_norm < x_tol * denom)
                //|| (a_tol > T::zero() && merit_next < a_tol.to_f64().unwrap())
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

            // Update merit function
            //merit_prev = merit_cur;
            merit_cur = merit_next;
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
        "    i |   merit    |    d-2     |    x-2     | lin # |  lin err   |   sigma    | ls # "
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
    merit: f64,
    r: &[T],
    x: &[T],
    x_prev: &[T],
) {
    log::debug!(
        "{i:>5} |  {merit:10.3e} | {di:10.3e} | {xi:10.3e} | {lin:>5} | {linerr:10.3e} | {sigma:10.3e} | {ls:>4} ",
        i = iterations,
        merit = merit,
        di = x_prev.iter().zip(x.iter()).map(|(&a, &b)| (a - b)*(a-b)).sum::<T>()
            .to_f64()
            .unwrap().sqrt(),
        xi = x.as_tensor().norm().to_f64().unwrap(),
        lin = linsolve_result.iterations,
        linerr = linsolve_result.error,
        sigma = sigma,
        ls = ls_steps
    );
}

/// Solves a dense potentially indefinite system `Ax = b`.
#[allow(non_snake_case)]
#[allow(dead_code)]
fn dense_solve_mut<T: Real + na::ComplexField>(A: DSMatrixView<T>, b: &mut [T]) -> bool {
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

// Helper function for debugging the jacobian.
#[allow(dead_code)]
fn build_dense<T: Real>(
    mut j_dense: ChunkedN<&mut [T]>,
    j_rows: &[usize],
    j_cols: &[usize],
    j_vals: &[T],
    num_variables: usize,
) {
    // Clear j_dense
    for jd in j_dense.storage_mut().iter_mut() {
        *jd = T::zero();
    }
    // Copy j_vals to j_dense
    for ((&r, &c), &v) in j_rows.iter().zip(j_cols.iter()).zip(j_vals.iter()) {
        if r < num_variables && c < num_variables {
            j_dense[r][c] += v;
        }
    }
}

// Helper function for debugging the jacobian.
#[allow(dead_code)]
fn print_dense<T: Real>(j_dense: ChunkedN<&[T]>) {
    eprintln!("J = [");
    for jp in j_dense.iter() {
        for j in jp.iter() {
            eprint!("{:?} ", j);
        }
        eprintln!(";");
    }
    eprintln!("]");
}

#[allow(dead_code)]
fn condition_number<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>) -> T {
    let svd = svd_values(mtx);
    let max_sigma = svd
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min_sigma = svd
        .iter()
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    max_sigma / min_sigma
}

#[allow(dead_code)]
fn svd_values<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>) -> Vec<T> {
    // nalgebra dense prototype using lu.
    let mut dense = na::DMatrix::zeros(mtx.len(), mtx.len());

    for (row_idx, row) in mtx.iter().enumerate() {
        for (col_idx, val) in row.iter().enumerate() {
            *dense.index_mut((row_idx, col_idx)) = *val;
        }
    }
    let v = dense.singular_values();
    v.into_iter().cloned().collect()
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
