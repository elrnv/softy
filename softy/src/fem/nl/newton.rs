use std::cell::RefCell;
use std::fmt::{Display, Formatter};
use std::time::Instant;

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
use crate::nl_fem::Timings;
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
    /// Linear solver configuration,
    pub linsolve: LinearSolver,
    /// Line search method.
    pub line_search: LineSearch,
    /// Inter step Jacobian check.
    ///
    /// If true this causes a fine grained derivative check at each Newton iteration.
    pub derivative_check: bool,
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
        self.update_rhs(b);
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
    use std::convert::TryFrom;
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
#[cfg(target_os = "macos")]
fn update_sparse_values<T: Real>(m: &mut SparseMatrixF64, values: &[T]) {
    m.data_mut()
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
            "Symbolic factor time: {}ms",
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
        // Update values and refactor
        let t_begin = Instant::now();
        self.update_rhs(&r);
        let t_update_rhs = Instant::now();
        let values: Vec<_> = values.iter().map(|&x| x.to_f64().unwrap()).collect();
        let t_values = Instant::now();
        self.solver
            .refactor(values.as_slice(), mkl::dss::Definiteness::Indefinite)?;
        let t_refactor = Instant::now();
        self.solver
            .solve_into(&mut self.sol, &mut self.buf, &mut self.r64)?;
        let t_solve = Instant::now();
        log::trace!("Update time:   {}ms", (t_update_rhs - t_begin).as_millis());
        log::trace!("Values time:   {}ms", (t_values - t_update_rhs).as_millis());
        log::trace!("Refactor time: {}ms", (t_refactor - t_values).as_millis());
        log::trace!("Solve time:    {}ms", (t_solve - t_refactor).as_millis());
        Ok(&self.sol)
    }

    #[cfg(target_os = "macos")]
    pub fn solve_with_values<T: Real>(
        &mut self,
        r: &[T],
        values: &[T],
    ) -> Result<&[f64], SparseDirectSolveError> {
        let t_begin = Instant::now();
        self.update_values(values);
        let t_update_values = Instant::now();
        self.refactor()?;
        let t_refactor = Instant::now();
        self.update_rhs(r);
        let t_update_rhs = Instant::now();
        let result = self.solve();
        let t_solve = Instant::now();
        log::trace!(
            "Update time:   {}ms",
            (t_update_rhs - t_refactor).as_millis()
        );
        log::trace!(
            "Values time:   {}ms",
            (t_update_values - t_begin).as_millis()
        );
        log::trace!(
            "Refactor time: {}ms",
            (t_refactor - t_update_values).as_millis()
        );
        log::trace!("Solve time:    {}ms", (t_solve - t_refactor).as_millis());
        result
    }
}

pub struct DirectSolver<T: Real> {
    /// Mapping from original triplets given by the `j_*` members to the final
    /// compressed sparse matrix.
    j_mapping: Vec<Index>,
    j: DSMatrix<T>,
    //#[cfg(target_os = "macos")]
    //j_sparse: LazyCell<SparseMatrixF64<'static>>,
    sparse_solver: LazyCell<SparseDirectSolver>,
    //#[cfg(target_os = "macos")]
    //sparse_iterative_solver: LazyCell<SparseIterativeSolver>,
    p64: Vec<f64>,
    out64: Vec<f64>,
}

pub struct SparseJacobian<T> {
    j_rows: Vec<usize>,
    j_cols: Vec<usize>,
    j_vals: Vec<T>,
}

impl<T> Default for SparseJacobian<T> {
    fn default() -> Self {
        SparseJacobian {
            j_rows: Vec::new(),
            j_cols: Vec::new(),
            j_vals: Vec::new(),
        }
    }
}

impl<T: Real> SparseJacobian<T> {
    // Used for debugging
    #[allow(dead_code)]
    fn compute_product(&self, p: &[T], jp: &mut [T]) {
        let num_rows = jp.len();
        let num_cols = p.len();
        // Clear output
        jp.iter_mut().for_each(|jp| *jp = T::zero());
        self.j_rows
            .iter()
            .zip(self.j_cols.iter())
            .zip(self.j_vals.iter())
            .filter(|((&row, &col), _)| row < num_rows && col < num_cols)
            .for_each(|((&row, &col), &val)| {
                jp[row] += p[col] * val;
            });
    }
}

pub enum LinearSolverWorkspace<T: Real> {
    Iterative(BiCGSTAB<na::DVector<T>>),
    Direct(DirectSolver<T>),
}

impl<T: Real> LinearSolverWorkspace<T> {
    #[allow(dead_code)]
    fn iterative_tolerance(&self) -> f32 {
        if let LinearSolverWorkspace::Iterative(bicgstab) = self {
            bicgstab.tol
        } else {
            0.0 // no tolerance in direct solves.
        }
    }
}
pub struct NewtonWorkspace<T: Real> {
    linsolve: LinearSolverWorkspace<T>,
    sparse_jacobian: SparseJacobian<T>,
    init_sparse_jacobian_vals: Vec<T>,
    init_sparse_solver: LazyCell<SparseDirectSolver>,
    init_j_mapping: Vec<Index>,
    x_prev: Vec<T>,
    r: Vec<T>,
    p: Vec<T>,
    jp: Vec<T>,
    r_cur: Vec<T>,
    r_next: Vec<T>,
    r_next_unscaled: Vec<T>,
    r_lagged: Vec<T>,
    precond: Vec<T>,
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

fn sparse_matrix_and_mapping<'a, T: Real>(
    mut rows: &'a [usize],
    mut cols: &'a [usize],
    vals: &[T],
    mtx_size: usize,
    transpose: bool,
) -> (DSMatrix<T>, Vec<Index>) {
    if transpose {
        std::mem::swap(&mut rows, &mut cols);
    }
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
        let r_next_unscaled = r.clone();
        let r_lagged = r.clone();
        let r_cur = r.clone();
        let p = r.clone();

        let precond = vec![T::one(); n];

        // Allocate space for the linear solver.
        let linsolve = match params.linsolve {
            LinearSolver::Iterative {
                tolerance,
                max_iterations,
            } => {
                log::debug!("ITERATIVE SOLVE");
                LinearSolverWorkspace::Iterative(BiCGSTAB::new(n, max_iterations, tolerance))
            }
            LinearSolver::Direct => {
                log::debug!("DIRECT SOLVE");
                LinearSolverWorkspace::Direct(DirectSolver {
                    j: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
                    //#[cfg(target_os = "macos")]
                    //j_sparse: LazyCell::new(),
                    j_mapping: Vec::new(),
                    sparse_solver: LazyCell::new(),
                    //#[cfg(target_os = "macos")]
                    //sparse_iterative_solver: LazyCell::new(),
                    p64: Vec::new(),
                    out64: Vec::new(),
                })
            }
        };

        Newton {
            problem,
            params,
            outer_callback: RefCell::new(outer_callback),
            inner_callback: RefCell::new(inner_callback),
            workspace: RefCell::new(NewtonWorkspace {
                linsolve,
                sparse_jacobian: SparseJacobian::default(),
                init_sparse_jacobian_vals: Vec::new(),
                init_sparse_solver: LazyCell::new(),
                init_j_mapping: Vec::new(),
                x_prev,
                r,
                p,
                jp,
                r_cur,
                r_next,
                r_next_unscaled,
                r_lagged,
                precond,
            }),
        }
    }
    fn update_jacobian_indices(
        problem: &P,
        sparse_jacobian: &mut SparseJacobian<T>,
        _init_sparse_jacobian_vals: &mut Vec<T>,
        _init_sparse_solver: &mut LazyCell<SparseDirectSolver>,
        _init_j_mapping: &mut Vec<Index>,
        linsolve: &mut LinearSolverWorkspace<T>,
    ) {
        // Construct the sparse Jacobian.
        let n = problem.num_variables();
        let with_constraints = matches!(linsolve, LinearSolverWorkspace::Direct(_));
        let (j_rows, j_cols) = problem.jacobian_indices(with_constraints);
        assert_eq!(j_rows.len(), j_cols.len());
        let j_nnz = j_rows.len();
        if j_nnz == 0 {
            return;
        }
        log::debug!("Number of Jacobian non-zeros: {}", j_nnz);
        sparse_jacobian.j_rows = j_rows;
        sparse_jacobian.j_cols = j_cols;
        sparse_jacobian.j_vals.resize(j_nnz, T::zero());
        // let (j, j_mapping) = sparse_matrix_and_mapping(
        //     &sparse_jacobian.j_cols,
        //     &sparse_jacobian.j_rows,
        //     &sparse_jacobian.j_vals,
        //     n,
        //     #[cfg(target_os = "macos")]
        //     true,
        //     #[cfg(not(target_os = "macos"))]
        //     false,
        // );
        // init_sparse_solver.replace(SparseDirectSolver::new(j.view()).unwrap());
        // *init_j_mapping = j_mapping;
        // init_sparse_jacobian_vals.clone_from(j.storage());
        if let LinearSolverWorkspace::Direct(ds) = linsolve {
            let (j, j_mapping) = sparse_matrix_and_mapping(
                &sparse_jacobian.j_rows,
                &sparse_jacobian.j_cols,
                &sparse_jacobian.j_vals,
                n,
                #[cfg(target_os = "macos")]
                true,
                #[cfg(not(target_os = "macos"))]
                false,
            );

            //#[cfg(target_os = "macos")]
            //ds.j_sparse.replace(new_sparse(j_t.view(), true));
            ds.j = j;
            ds.j_mapping = j_mapping;
            ds.sparse_solver
                .replace(SparseDirectSolver::new(ds.j.view()).unwrap());
            //#[cfg(target_os = "macos")]
            //ds.sparse_iterative_solver
            //    .replace(SparseIterativeSolver::new(ds.j_t.view()));
            ds.p64.resize(n, 0.0);
            ds.out64.resize(n, 0.0);
        }
    }
}

fn unscale_vector<T: Real>(precond: &[T], r: &mut [T]) {
    precond.iter().zip(r.iter_mut()).for_each(|(&p, r)| {
        *r /= p;
    });
}

fn rescale_vector<T: Real>(precond: &[T], r: &mut [T]) {
    precond.iter().zip(r.iter_mut()).for_each(|(&p, r)| {
        *r *= p;
    });
}

fn rescale_jacobian_values<T: Real>(precond: &[T], rows: &[usize], vals: &mut [T]) {
    rows.iter().zip(vals.iter_mut()).for_each(|(&r, v)| {
        if r < precond.len() {
            *v *= precond[r];
        }
    });
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
        let NewtonWorkspace {
            sparse_jacobian: sj,
            init_sparse_jacobian_vals,
            init_sparse_solver,
            init_j_mapping,
            linsolve,
            ..
        } = &mut *self.workspace.borrow_mut();
        Self::update_jacobian_indices(
            &self.problem,
            sj,
            init_sparse_jacobian_vals,
            init_sparse_solver,
            init_j_mapping,
            linsolve,
        );
    }

    /// Solves the problem and returns the solution along with the solve result
    /// info.
    fn solve(&mut self) -> (Vec<T>, SolveResult) {
        let mut x = self.problem.initial_point();
        let res = self.solve_with(x.as_mut_slice(), true);
        (x, res)
    }

    /// Solves the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&mut self, x: &mut [T], update_jacobian_indices: bool) -> SolveResult {
        let mut stats = Vec::new();
        let mut timings = Timings::default();
        let is_iterative = matches!(
            self.workspace.borrow().linsolve,
            LinearSolverWorkspace::Iterative(_)
        );

        if x.is_empty() {
            return SolveResult {
                iterations: 0,
                status: Status::NothingToSolve,
                timings,
                stats,
            };
        }

        self.problem.residual_timings().clear();
        let t_begin_solve = Instant::now();

        if update_jacobian_indices {
            self.update_jacobian_indices();
        }

        // let mut merit = |problem: &P, _: &[T], residual: &[T], init_sparse_solver: &mut SparseDirectSolver| {
        //     init_sparse_solver.update_rhs(residual);
        //     let result = init_sparse_solver.solve();
        //     let r64 = result.expect("Initial Jacobian is singular.");
        //     0.5 * r64.as_tensor().norm_squared()
        // };
        let merit = |_: &P, _: &[T], r: &[T]| 0.5 * r.as_tensor().norm_squared().to_f64().unwrap();
        let merit_obj = |problem: &P, x: &[T]| problem.objective(x).to_f64().unwrap();

        // Computes the product of merit function with the search direction.
        // In this version, we leverage that Jp is already computed.
        // let mut merit_jac_prod =
        //     |problem: &P, jp: &[T], r: &[T], init_sparse_solver: &mut SparseDirectSolver| {
        //         init_sparse_solver.update_rhs(jp);
        //         let result = init_sparse_solver.solve();
        //         let jinv_jp = result.expect("Initial Jacobian is singular.").to_vec();
        //         // TODO: don't need to compute this again in general.
        //         init_sparse_solver.update_rhs(r);
        //         let result = init_sparse_solver.solve();
        //         let jinv_r = result.expect("Initial Jacobian is singular.");
        //         jinv_jp
        //             .iter()
        //             .zip(jinv_r.iter())
        //             .fold(0.0, |acc, (&jinv_jp, &jinv_r)| acc + jinv_jp * jinv_r)
        //     };
        let merit_jac_prod = |_: &P, jp: &[T], r: &[T]| {
            //r.as_tensor().norm_squared().to_f64().unwrap()
            jp.iter()
                .zip(r.iter())
                .fold(0.0, |acc, (&jp, &r)| acc + (jp * r).to_f64().unwrap())
        };
        let merit_obj_jac_prod = |_: &P, _: &[T], p: &[T], r_lagged: &mut [T]| {
            // problem.residual_symmetric(x, r_lagged);
            r_lagged
                .iter()
                .zip(p.iter())
                .map(|(&r, &p)| (r * p).to_f64().unwrap())
                .sum::<f64>()
        };

        {
            let Self {
                problem, workspace, ..
            } = self;

            let NewtonWorkspace { r, precond, .. } = &mut *workspace.borrow_mut();

            // Update all state to correspond to x being the next velocity.
            problem.update_state(x, true, !is_iterative);

            // Compute preconditioner.
            // TODO: determine if it's any better doing this for every step or just once at the beginning is enough.
            problem.diagonal_preconditioner(x, precond);

            // Initialize the residual.
            problem.residual(x, r.as_mut_slice(), false);
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
            sparse_jacobian,
            //init_sparse_jacobian_vals,
            //init_sparse_solver,
            // init_j_mapping,
            x_prev,
            r,
            p,
            jp,
            r_cur,
            r_next,
            r_next_unscaled,
            r_lagged,
            precond,
            ..
        } = &mut *workspace.borrow_mut();

        // let SparseJacobian {
        //     j_rows,
        //     j_cols,
        //     j_vals,
        //     //j_mapping,
        //     j_t_mapping,
        //     //j,
        //     j_t,
        //     #[cfg(target_os = "macos")]
        //     j_sparse,
        //     sparse_solver,
        //     //#[cfg(target_os = "macos")]
        //     //sparse_iterative_solver,
        //     p64,
        //     out64: _,
        // } = sparse_jacobian;
        //
        // // Unwrap the sparse solver. In case of panic check update_jacobian_indices function.
        //  let sparse_solver = sparse_solver
        //      .borrow_mut()
        //      .expect("Uninitialized direct sparse solver.");

        // #[cfg(target_os = "macos")]
        // let sparse_iterative_solver = sparse_iterative_solver
        //     .borrow_mut()
        //     .expect("Uninitialized iterative sparse solver.");

        let mut iterations = 0;

        let mut linsolve_result = super::linsolve::SolveResult::default();

        // Forcing term only used for inexact newton.
        let orig_linsolve_tol = linsolve.iterative_tolerance();

        let header = IterationInfo::header();
        log::debug!("{}", header[0]);
        log::debug!("{}", header[1]);
        let info = IterationInfo::new(
            0,
            0,
            linsolve_result,
            orig_linsolve_tol,
            1.0,
            f64::INFINITY,
            r,
            x_prev,
            x,
            &*problem.lumped_mass_inv(),
        );
        log::debug!("{}", info);
        stats.push(info);

        if sparse_jacobian.j_rows.is_empty() {
            return SolveResult {
                iterations: 0,
                status: Status::NothingToSolve,
                timings,
                stats,
            };
        }

        // We do this late, so that the first IterationInfo gets an unscaled value.
        rescale_vector(precond, r.as_mut_slice());

        // Keep track of merit function to avoid having to recompute it. This must be after rescale.
        let mut merit_cur = merit(problem, x, r);
        // let mut merit_cur = merit_obj(&self.problem, x);
        let mut merit_prev = merit_cur;
        let mut merit_next;

        log::trace!("ls: initial merit = {merit_cur:?}");

        let mut j_dense_ad = LazyCell::new();
        let mut j_dense = LazyCell::new();

        let (iterations, status) = loop {
            if !(outer_callback.borrow_mut())(CallbackArgs {
                iteration: iterations,
                residual: r.as_slice(),
                x,
                problem,
            }) {
                break (iterations, Status::Interrupted);
            }

            if !merit_cur.is_finite() {
                break (iterations, Status::Diverged);
            }

            let t_begin_linsolve = Instant::now();

            r_cur.copy_from_slice(r);

            let mut used_linsolve_tol = orig_linsolve_tol;

            match linsolve {
                LinearSolverWorkspace::Iterative(linsolve) => {
                    // Update tolerance (forcing term) (Eisenstat and Walker paper)
                    // CHOICE 1
                    // sigma = orig_sigma.min(((merit_cur - linsolve_result.residual).abs() / merit_prev) as f32);
                    // CHOICE 2
                    let eta_prev2 = linsolve.tol * linsolve.tol;
                    let power = 0.5 * (1.0 + 0.5_f32.sqrt()); // Golden ratio
                    let mut eta =
                        f32::EPSILON.max((merit_cur / merit_prev).powf(0.5 * power as f64) as f32);
                    log::trace!("proposed eta = {:?}", eta);
                    // Safeguard to prevent oversolving (see paper)
                    if eta_prev2 > 0.1 {
                        eta = eta.max(eta_prev2);
                    }
                    linsolve.tol = orig_linsolve_tol.min(eta as f32);
                    used_linsolve_tol = linsolve.tol;

                    // let j_dense = j_dense.borrow_mut_with(|| {
                    //     ChunkedN::from_flat_with_stride(
                    //         x.len(),
                    //         vec![T::zero(); x.len() * r.len()],
                    //     )
                    // });
                    // build_dense_from_product(j_dense.view_mut(), |i, col| {
                    //     problem.jacobian_product(x, &identity[i], r_cur, col);
                    // }, x.len());
                    // print_dense(j_dense.view());

                    // problem.jacobian_values(
                    //     x,
                    //     r_cur,
                    //     &sparse_jacobian.j_rows,
                    //     &sparse_jacobian.j_cols,
                    //     sparse_jacobian.j_vals.as_mut_slice(),
                    // );

                    linsolve_result = linsolve.solve_precond(
                        |p, out| {
                            let t_begin_jprod = Instant::now();
                            problem.jacobian_product(x, p, r_cur, out);
                            // Explicit preconditioning
                            rescale_vector(precond, out);
                            // sparse_jacobian.compute_product(p, out);
                            timings.jacobian_product += Instant::now() - t_begin_jprod;
                            inner_callback.borrow_mut()(CallbackArgs {
                                residual: r_cur.as_slice(),
                                x,
                                problem,
                                iteration: iterations,
                            })
                        },
                        p.as_mut_slice(),
                        r.as_mut_slice(),
                        |s, buf| -> &[T] {
                            s.iter().zip(precond.iter()).zip(buf.iter_mut()).for_each(
                                |((&s, &m), buf)| {
                                    *buf = s * m;
                                },
                            );
                            buf
                        },
                    );
                }
                LinearSolverWorkspace::Direct(DirectSolver {
                    j_mapping,
                    j,
                    sparse_solver,
                    ..
                }) => {
                    let t_begin_solve = Instant::now();
                    problem.jacobian_values(
                        x,
                        r_cur,
                        &sparse_jacobian.j_rows,
                        &sparse_jacobian.j_cols,
                        sparse_jacobian.j_vals.as_mut_slice(),
                    );
                    rescale_jacobian_values(
                        precond,
                        &sparse_jacobian.j_rows,
                        sparse_jacobian.j_vals.as_mut_slice(),
                    );

                    // let mut diag = vec![0.0; x.len()];
                    // for ((&r, &c), &v) in sparse_jacobian
                    //     .j_rows
                    //     .iter()
                    //     .zip(sparse_jacobian.j_cols.iter())
                    //     .zip(sparse_jacobian.j_vals.iter())
                    // {
                    //     if r == c && r < x.len() {
                    //         diag[r] += v.to_f32().unwrap();
                    //     }
                    // }
                    // eprintln!("diag = {:?}", &diag);

                    // Build dense jacobian for testing gradient
                    // let j_dense_ad = j_dense_ad.borrow_mut_with(|| {
                    //     ChunkedN::from_flat_with_stride(x.len(), vec![T::zero(); x.len() * r.len()])
                    // });
                    // let mut zero_col = vec![T::zero(); x.len()];
                    // build_dense_from_product(
                    //     j_dense_ad.view_mut(),
                    //     |i, col| {
                    //         // eprintln!("PRODUCT WRT {}", i);
                    //         zero_col[i] = T::one();
                    //         problem.jacobian_product(x, &zero_col, r, col);
                    //         rescale_vector(precond, col);
                    //         zero_col[i] = T::zero();
                    //     },
                    //     x.len(),
                    // );

                    // Compute gradient for reporting and debugging
                    // let mut jtr = vec![T::zero(); r.len()];
                    // for ((&row, &col), &val) in sparse_jacobian
                    //     .j_rows
                    //     .iter()
                    //     .zip(sparse_jacobian.j_cols.iter())
                    //     .zip(sparse_jacobian.j_vals.iter())
                    // {
                    //     if row < r.len() && col < x.len() {
                    //         jtr[col] += r_cur[row] * val;
                    //     }
                    // }

                    // let jtr_norm_inf = jtr.as_tensor().lp_norm(LpNorm::Inf).to_f64().unwrap();
                    // let jtr_norm_2 = jtr.as_tensor().lp_norm(LpNorm::P(2)).to_f64().unwrap();
                    // let r_norm_inf = r_cur.as_tensor().lp_norm(LpNorm::Inf).to_f64().unwrap();
                    // let r_norm_2 = r_cur.as_tensor().lp_norm(LpNorm::P(2)).to_f64().unwrap();
                    // eprintln!("  jtr_inf   |   jtr_2    |   r_inf    |    r_2     |     rTp    |");
                    // eprint!("{jtr_norm_inf:10.3e} | {jtr_norm_2:10.3e} | {r_norm_inf:10.3e} | {r_norm_2:10.3e} |");

                    let t_jacobian_values = Instant::now();
                    if params.derivative_check {
                        let j_dense_ad = j_dense_ad.borrow_mut_with(|| {
                            ChunkedN::from_flat_with_stride(
                                x.len(),
                                vec![T::zero(); x.len() * r.len()],
                            )
                        });
                        // A utility vector used to simulate multiplication by identity
                        let mut zero_col = vec![T::zero(); x.len()];
                        build_dense_from_product(
                            j_dense_ad.view_mut(),
                            |i, col| {
                                // eprintln!("PRODUCT WRT {}", i);
                                zero_col[i] = T::one();
                                problem.jacobian_product(x, &zero_col, r, col);
                                rescale_vector(precond, col);
                                zero_col[i] = T::zero();
                            },
                            x.len(),
                        );
                        // print_dense(j_dense_ad.view());

                        let j_dense = j_dense.borrow_mut_with(|| {
                            ChunkedN::from_flat_with_stride(
                                x.len(),
                                vec![T::zero(); x.len() * r.len()],
                            )
                        });
                        build_dense(
                            j_dense.view_mut(),
                            &sparse_jacobian.j_rows,
                            &sparse_jacobian.j_cols,
                            &sparse_jacobian.j_vals,
                            x.len(),
                        );
                        // dbg!(x.len());
                        // eprintln!("{:?}", &r);
                        // print_dense(j_dense.view());
                        //log::debug!("J singular values: {:?}", svd_values(j_dense.view()));
                        //write_jacobian_img(j_dense.view(), iterations);

                        let mut success = true;
                        for i in 0..j_dense.len() {
                            for j in 0..j_dense.len() {
                                let a = j_dense.view().at(i).at(j).to_f64().unwrap();
                                let b = j_dense_ad.view().at(j).at(i).to_f64().unwrap();
                                if !approx::relative_eq!(a, b, max_relative = 1e-6, epsilon = 1e-3)
                                {
                                    eprintln!("({},{}): {} vs {}", i, j, a, b);
                                    success = false;
                                }
                            }
                        }
                        if !success {
                            return SolveResult {
                                iterations,
                                status: Status::FailedJacobianCheck,
                                timings,
                                stats,
                            };
                        }
                    }
                    log::trace!("Bound estimate and condition: {:?}", {
                        let j_dense = j_dense.borrow_mut_with(|| {
                            ChunkedN::from_flat_with_stride(
                                x.len(),
                                vec![T::zero(); x.len() * r.len()],
                            )
                        });
                        build_dense(
                            j_dense.view_mut(),
                            &sparse_jacobian.j_rows,
                            &sparse_jacobian.j_cols,
                            &sparse_jacobian.j_vals,
                            x.len(),
                        );
                        // log::debug!("J singular values: {:?}", svd_values(j_dense.view()));
                        //write_jacobian_img(j_dense.view(), iterations);
                        // print_dense(j_dense.view());
                        //condition_number(j_dense.view()).to_f64().unwrap()
                        let (b, c) = bound_estimate_and_condition(
                            &*problem.lumped_mass_inv(),
                            j_dense.view(),
                            T::from(problem.time_step()).unwrap(),
                        );
                        (b.to_f64().unwrap(), c.to_f64().unwrap())
                        // max_sigma(j_dense.view())
                    });
                    let t_linsolve_debug_info = Instant::now();

                    ////log::trace!("j_vals = {:?}", &j_vals);

                    // //// Zero out Jacobian.
                    j.storage_mut().iter_mut().for_each(|x| *x = T::zero());
                    // // Update the Jacobian matrix.
                    // for (row_idx, mut sparse_row) in j.view_mut().into_data().iter_mut().enumerate()
                    // {
                    //     for (col_idx, entry) in sparse_row.iter_mut() {
                    //         *entry = j_dense_ad[*col_idx][row_idx];
                    //     }
                    // }
                    for (&pos, &j_val) in j_mapping.iter().zip(sparse_jacobian.j_vals.iter()) {
                        if let Some(pos) = pos.into_option() {
                            j.storage_mut()[pos] += j_val;
                        }
                    }

                    let sparse_solver = sparse_solver
                        .borrow_mut()
                        .expect("Uninitialized iterative sparse solver.");
                    let result = sparse_solver.solve_with_values(r, j.storage());
                    let r64 = match result {
                        Err(err) => break (iterations, Status::LinearSolveError(err.into())),
                        Ok(r) => r,
                    };

                    // r is now the search direction, rename to avoid confusion.
                    p.iter_mut()
                        .zip(r64.iter())
                        .for_each(|(p, &r64)| *p = T::from(r64).unwrap());

                    // Check
                    // let mut jp = vec![T::zero(); r.len()];
                    // problem.jacobian_product(x, p, r_cur, jp.as_mut_slice());
                    // eprintln!("p = {:?}", &p);
                    // eprintln!("r = {:?}", &r_cur);
                    // eprintln!("jp_check = {:?}", &jp);

                    //log::trace!("linsolve result: {:?}", linsolve_result);
                    timings.direct_solve += Instant::now() - t_linsolve_debug_info;
                    timings.jacobian_values += t_jacobian_values - t_begin_solve;
                    timings.linsolve_debug_info += t_linsolve_debug_info - t_jacobian_values;
                }
            }

            // Negate
            p.iter_mut().for_each(|p| *p = -*p);

            timings.linear_solve += Instant::now() - t_begin_linsolve;

            // The solve converts the rhs r into the unknown negative search direction p.

            if !p.as_tensor().norm_squared().is_finite() {
                break (iterations, Status::StepTooLarge);
            }

            // Update previous step.
            x_prev.copy_from_slice(x);

            let rho = params.line_search.step_factor();

            // Compute Jacobian product to be used to check armijo condition.
            // We do this before incrementing x to avoid updating state back and forth.
            let merit_jac_p = if rho <= 1.0 {
                problem.jacobian_product(x, p, r_cur, jp.as_mut_slice());
                rescale_vector(precond, jp.as_mut_slice());
                merit_jac_prod(problem, jp.as_slice(), r_cur)
                // merit_obj_jac_prod(problem, x_prev, p, r_lagged)
            } else {
                0.0
            };

            // Take the full step
            *x.as_mut_tensor() += p.as_tensor();

            // This ensures that next time jacobian_product is requested, the full jacobian values
            // are recomputed for a fresh x.
            problem.invalidate_cached_jacobian_product_values();

            // Update all state corresponding to the full new velocity x + p
            problem.update_state(x, true, !is_iterative);

            // Compute the residual for the full step.
            problem.residual(x, r_next.as_mut_slice(), false);
            rescale_vector(precond, r_next.as_mut_slice());
            // dbg!(r_next.as_slice());

            let (ls_count, alpha) = add_time! {
                timings.line_search;
                if rho >= 1.0 {
                    merit_next = merit(problem, x, r_next);//, init_sparse_solver);
                    // merit_next = merit_obj(problem, x);
                    (1, 1.0)
                } else {
                    // Line search.
                    let mut alpha = 1.0;
                    let mut ls_count = 1;

                    //dbg!(merit_jac_p);
                    //dbg!(r_cur.as_tensor().norm_squared().to_f64().unwrap());

                    if matches!(linsolve, LinearSolverWorkspace::Iterative(_)) && merit_jac_p > 0.0 {
                        log::trace!("positive search direction reversed: {merit_jac_p:10.3e}");
                        // Reverse direction if it's not a descent direction.
                        *p.as_mut_tensor() *= -T::one();
                    }

                    if params.line_search.is_assisted() {
                        let new_alpha;
                        add_time!(
                            timings.line_search_assist;
                            new_alpha = problem
                                .assist_line_search(T::from(alpha).unwrap(), p, x_prev, r_cur, r_next)
                                .to_f64()
                                .unwrap()
                        );

                        // Avoid recomputing residual and updating state needlessly.
                        // IMPORTANT: This works because if alpha is unchanged, then the friction
                        // assist step would have have used alpha = 1.0 when updating contact state,
                        // which doesn't break the next call to jacobian since we avoid
                        // rebuilding the rtree there for efficiency.
                        if new_alpha < alpha {
                            // Take a fractional step.
                            zip!(x.iter_mut(), x_prev.iter(), p.iter()).for_each(|(x, &x0, &p)| {
                                *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                            });

                            // Compute the candidate residual.
                            problem.update_state(x, true, !is_iterative);
                            problem.residual(x, r_next.as_mut_slice(), false);
                            rescale_vector(precond, r_next.as_mut_slice());

                            alpha = new_alpha;
                        }
                    }

                    log::trace!("ls: starting alpha = {alpha:?}");

                    loop {
                        // Compute gradient of the merit function 0.5 r'r  multiplied by p, which is r' dr/dx p.
                        // Gradient dot search direction.
                        // Check Armijo condition:
                        // Given f(x) = 0.5 || r(x) ||^2 = 0.5 r(x)'r(x)
                        // Test f(x + αp) < f(x) - cα(J(x)'r(x))'p(x)
                        // Test f(x + αp) < f(x) - cα r(x)'J(x)p(x)

                        // Compute the merit function
                        merit_next = merit(problem, x, r_next);//, init_sparse_solver);
                        // log::trace!("ls: merit = {:?}", merit_next);
                        // merit_next = merit_obj(problem, x);//, init_sparse_solver);

                        match linsolve {
                            LinearSolverWorkspace::Iterative(linsolve) => {
                                // INEXACT NEWTON:
                                let t = params.line_search.armijo_coeff();
                                let factor = 1.0 - t * alpha * (1.0 - linsolve.tol as f64);
                                if merit_next <= merit_cur * factor * factor {
                                    break;
                                }
                                // TODO: Use another factor during backtracking.
                                alpha *= rho;
                            }
                            LinearSolverWorkspace::Direct(_) => {
                                // TRADITIONAL BACKTRACKING:
                                let increment = params.line_search.armijo_coeff() * alpha * merit_jac_p;
                                if merit_next <= merit_cur + increment
                                {
                                    log::trace!("ls: backtracking success: {merit_next:?} <= {merit_cur:?} + {increment:?}");
                                    break;
                                }
                                alpha *= rho;
                            }
                        }
                        // log::trace!("ls: alpha = {alpha:?}");

                        // Break if alpha becomes too small. This is usually a bad sign.
                        if alpha < 1e-5 && ls_count > 2 {
                            log::trace!("ls: backtracking fail: alpha too small: {alpha:?}");
                            break;
                        }

                        // Take a fractional step.
                        zip!(x.iter_mut(), x_prev.iter(), p.iter()).for_each(|(x, &x0, &p)| {
                            *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                        });

                        // Update state and compute the candidate residual for the next line search iteration.
                        problem.update_state(x, true, !is_iterative);
                        problem.residual(x, r_next.as_mut_slice(), false);
                        rescale_vector(precond, r_next.as_mut_slice());

                        ls_count += 1;
                    }

                    // This ensures that next time jacobian_product is requested, the full Jacobian values
                    // are recomputed for a fresh x.
                    problem.invalidate_cached_jacobian_product_values();

                    // Increment linear solve tolerance.
                    if let LinearSolverWorkspace::Iterative(linsolve) = linsolve {
                        linsolve.tol = 1.0 - alpha as f32 * (1.0 - linsolve.tol);
                        log::trace!("incremented eta: {:?}", linsolve.tol);
                    }

                    if alpha < 1e-5 && ls_count > 80 {
                        problem.invalidate_cached_jacobian_product_values();
                        dbg!(alpha);
                        let max_alpha = alpha;
                        let mut merit_data = vec![];
                        let mut r0 = vec![];
                        let mut f = vec![];
                        let mut xs = vec![];
                        let mut probe_r = vec![T::zero(); r_next.len()];
                        let mut probe_x = vec![T::zero(); x.len()];
                        use std::io::Write;
                        let mut file = std::fs::File::create(&format!("./out/debug_data_{iterations}.jl")).unwrap();
                        let last_index = 2000;
                        for i in 0..=last_index {
                            let alpha: f64 = (1.0e3*max_alpha).min(1.0)*(1.2 * 0.0005 * i as f64 - 0.2);
                            zip!(probe_x.iter_mut(), x_prev.iter(), p.iter()).for_each(
                                |(x, &x0, &p)| {
                                    *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                                },
                            );
                            problem.update_state(&probe_x, true, false);
                            problem.residual(&probe_x, probe_r.as_mut_slice(), false);
                            rescale_vector(precond, probe_r.as_mut_slice());
                            if i == 0  {
                                writeln!(file, "r_begin = {:?}", &probe_r).unwrap();
                            } else if i == last_index  {
                                writeln!(file, "r_end = {:?}", &probe_r).unwrap();
                            }
                            //geo::io::save_mesh(&problem.mesh(), &format!("./out/dbg_mesh_{}.vtk", i)).unwrap();
                            let probe_f = problem.debug_friction();
                            let probe = merit(problem, &probe_x, &probe_r);
                            xs.push(alpha);
                            f.push(probe_f.view().into_tensor().norm_squared());
                            merit_data.push(probe);
                            problem.update_state(&x_prev, true, false);
                            problem.jacobian_product(&x_prev, p, &probe_r, jp.as_mut_slice());
                            rescale_vector(precond, jp.as_mut_slice());
                            r0.push(merit_jac_prod(problem, &jp, &probe_r));
                        }
                        writeln!(file, "merit_cur = {:?}", merit_cur).unwrap();
                        writeln!(file, "merit_next = {:?}", merit_next).unwrap();
                        writeln!(file, "xs = {:?}", &xs).unwrap();
                        writeln!(file, "merit_data = {:?}", &merit_data).unwrap();
                        writeln!(file, "r0 = {:?}", &r0).unwrap();
                        writeln!(file, "f = {:?}", &f).unwrap();
                        writeln!(file, "xs_length = {:?}", xs.len()).unwrap();

                        merit_data.clear();
                        for i in 0..=last_index {
                            // let alpha: f64 = /*(4.0*max_alpha).min*/(1.2)*0.0005 * i as f64 - 0.2;
                            let alpha: f64 = (1.0e3*max_alpha).min(1.0)*(1.2 * 0.0005 * i as f64 - 0.2);
                            zip!(probe_x.iter_mut(), x_prev.iter(), p.iter()).for_each(
                                |(x, &x0, &p)| {
                                    *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                                },
                            );
                            problem.update_state(&probe_x, true, false);
                            problem.residual(&probe_x, probe_r.as_mut_slice(), false);
                            let probe = merit(problem, &probe_x, &probe_r);
                            merit_data.push(probe);
                        }
                        writeln!(file, "merit_data_u = {:?}", &merit_data).unwrap();

                        merit_data.clear();
                        let mut merit_data_alt = merit_data.clone();
                        for i in 0..=last_index {
                            // let alpha: f64 = /*(4.0*max_alpha).min*/(1.2)*0.0005 * i as f64 - 0.2;
                            let alpha: f64 = (1.0e3*max_alpha).min(1.0)*(1.2 * 0.0005 * i as f64 - 0.2);
                            zip!(probe_x.iter_mut(), x_prev.iter(), p.iter()).for_each(
                                |(x, &x0, &p)| {
                                    *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                                },
                            );
                            let probe = merit_obj(problem, &probe_x);
                            merit_data.push(probe);
                            problem.update_state(&probe_x, true, false);
                            problem.residual(&probe_x, probe_r.as_mut_slice(), true);
                            let probe = merit(problem, &probe_x, &probe_r);
                            merit_data_alt.push(probe);
                        }
                        writeln!(file, "merit_data_obj = {:?}", &merit_data).unwrap();
                        writeln!(file, "merit_data_alt = {:?}", &merit_data_alt).unwrap();

                        // Also compute the gradient and output that.
                        let mut zero_col = vec![T::zero(); x.len()];
                        let mut jcol = vec![T::zero(); x.len()];
                        let mut grad = vec![T::zero(); x.len()];
                        problem.update_state(&x_prev, true, false);
                        for i in 0..x.len() {
                            zero_col[i] = T::one();
                            problem.jacobian_product(&x_prev, &zero_col, &r_cur, jcol.as_mut_slice());
                            rescale_vector(precond, jcol.as_mut_slice());
                            for j in 0..jcol.len() {
                                grad[i] -= r_cur[j] * jcol[j];
                            }
                            zero_col[i] = T::zero();
                        }

                        merit_data.clear();
                        for i in 0..=last_index {
                            // let alpha: f64 = /*(4.0*max_alpha).min*/(1.5)*0.0005 * i as f64 - 0.5;
                            let alpha: f64 = (1.0e3*max_alpha).min(1.0)*(1.2 * 0.0005 * i as f64 - 0.2);
                            zip!(probe_x.iter_mut(), x_prev.iter(), grad.iter()).for_each(
                                |(x, &x0, &p)| {
                                    *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                                },
                            );
                            problem.update_state(&probe_x, true, false);
                            problem.residual(&probe_x, probe_r.as_mut_slice(), false);
                            rescale_vector(precond, probe_r.as_mut_slice());
                            if i == 0  {
                                writeln!(file, "r_begin_g = {:?}", &probe_r).unwrap();
                            } else if i == last_index  {
                                writeln!(file, "r_end_g = {:?}", &probe_r).unwrap();
                            }
                            let probe = merit(problem, &probe_x, &probe_r);
                            merit_data.push(probe);
                        }
                        writeln!(file, "merit_data_g = {:?}", &merit_data).unwrap();
                        writeln!(file, "pgrad = {:?}", &grad).unwrap();

                        if sparse_jacobian.j_rows.len() < 100_000 {
                            // Print the jacobian if it's small enough
                            let j_dense = j_dense.borrow_mut_with(|| {
                                ChunkedN::from_flat_with_stride(
                                    x.len(),
                                    vec![T::zero(); x.len() * r.len()],
                                )
                            });
                            build_dense(
                                j_dense.view_mut(),
                                &sparse_jacobian.j_rows,
                                &sparse_jacobian.j_cols,
                                &sparse_jacobian.j_vals,
                                x.len(),
                            );
                            writeln!(file, "Jrows = [").unwrap();
                            for (&row,&col) in sparse_jacobian.j_rows.iter().zip(sparse_jacobian.j_cols.iter()) {
                                if row < r_cur.len() && col < x.len() {
                                    write!(file, "{:?}, ", row+1).unwrap();
                                }
                            }
                            writeln!(file, "]").unwrap();
                            writeln!(file, "Jcols = [").unwrap();
                            for (&row,&col) in sparse_jacobian.j_rows.iter().zip(sparse_jacobian.j_cols.iter()) {
                                if row < r_cur.len() && col < x.len() {
                                    write!(file, "{:?}, ", col+1).unwrap();
                                }
                            }
                            writeln!(file, "]").unwrap();
                            writeln!(file, "Jvals = [").unwrap();
                            for ((&row,&col), val) in sparse_jacobian.j_rows.iter().zip(sparse_jacobian.j_cols.iter()).zip(sparse_jacobian.j_vals.iter()) {
                                if row < r_cur.len() && col < x.len() {
                                    write!(file, "{:?}, ", val).unwrap();
                                }
                            }
                            writeln!(file, "]").unwrap();
                            writeln!(file, "J = sparse(Jrows, Jcols, Jvals, {:?}, {:?})", r_cur.len(), x.len()).unwrap();
                        }
                        writeln!(file, "lumped_mass_inv = {:?}", &problem.lumped_mass_inv()).unwrap();
                        writeln!(file, "lumped_stiffness = {:?}", &problem.lumped_stiffness()).unwrap();

                        let mut jp = vec![T::zero(); r.len()];
                        problem.invalidate_cached_jacobian_product_values();
                        problem.update_state(x, true, false);
                        problem.jacobian_product(x, p, r_cur, jp.as_mut_slice());
                        rescale_vector(precond, jp.as_mut_slice());
                        writeln!(file, "p = {:?}", &p).unwrap();
                        writeln!(file, "r = {:?}", &r_cur).unwrap();
                        writeln!(file, "jp_check = {:?}", &jp).unwrap();
                        panic!("STOP");
                    }
                    (ls_count, alpha)
                }
            };

            iterations += 1;

            // Reset r to be a valid residual for the next iteration.
            r.copy_from_slice(r_next);
            r_next_unscaled.copy_from_slice(r_next);
            unscale_vector(precond, r_next_unscaled);

            let info = IterationInfo::new(
                iterations,
                ls_count,
                linsolve_result,
                used_linsolve_tol,
                alpha as f32,
                merit_next,
                r_next_unscaled,
                x_prev,
                x,
                &*problem.lumped_mass_inv(),
            );
            log::debug!("{}", &info);
            stats.push(info);

            // Check the convergence condition.
            if problem.converged(
                x_prev,
                x,
                r_next_unscaled,
                merit_next,
                params.x_tol,
                params.r_tol,
                params.a_tol,
            ) {
                break (iterations, Status::Success);
            }

            // Check that we are running no more than the maximum allowed iterations.
            if iterations >= params.max_iter {
                break (iterations, Status::MaximumIterationsExceeded);
            }

            // Update merit function
            merit_prev = merit_cur;
            merit_cur = merit_next;
        };

        // Restore linsolve tolernace
        if let LinearSolverWorkspace::Iterative(linsolve) = linsolve {
            linsolve.tol = orig_linsolve_tol;
        }

        timings.total = Instant::now() - t_begin_solve;
        timings.residual = *self.problem.residual_timings();
        timings.friction_jacobian = *self.problem.jacobian_timings();

        log::debug!("Status:           {:?}", status);
        let lin_steps = stats
            .iter()
            .map(|s| s.linsolve_result.iterations)
            .sum::<u32>();
        log::debug!("Total linear steps: {}", lin_steps);
        let ls_steps = stats.iter().map(|s| s.ls_steps).sum::<u32>();
        log::debug!("Total ls steps:     {}", ls_steps);
        log::debug!("Total Iterations:   {}", iterations);

        for line in format!("{}", timings).split('\n') {
            log::debug!("{}", line);
        }

        SolveResult {
            iterations,
            status,
            timings,
            stats,
        }
    }
}

/// Information about each iteration.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IterationInfo {
    /// Iteration counter.
    pub iteration: u32,
    /// Number of line search steps taken.
    pub ls_steps: u32,
    /// Linear solve result for this iteration.
    pub linsolve_result: super::linsolve::SolveResult,
    /// Linear solve tolerance used for this iteration.
    pub eta: f32,
    /// Fraction of the Newton step taken.
    ///
    /// Damping parameter of damped Newton.
    pub alpha: f32,
    /// Merit function value.
    pub merit: f64,
    /// Infinity norm of the residual.
    pub r_inf: f64,
    /// Infinity norm of the nodal acceleration residual.
    pub a_inf: f64,
    /// 2-norm of the current velocity.
    pub x_2: f64,
    /// 2-norm of the step vector
    pub d_2: f64,
}

impl IterationInfo {
    /// Constructs a new iteration info struct.
    pub fn new<T: Real>(
        iteration: u32,
        ls_steps: u32,
        linsolve_result: super::linsolve::SolveResult,
        eta: f32,
        alpha: f32,
        merit: f64,
        r: &[T],
        x_prev: &[T],
        x: &[T],
        mass_inv: &[T],
    ) -> Self {
        // let a_abs = T::from(9.81 * a_tol).unwrap() * dt;
        // let a_tol = T::from(a_tol).unwrap();
        // Chunked3::from_flat(&*r)
        //     .iter()
        //     .zip(mass_inv.iter())
        //     .zip(Chunked3::from_flat(&*x).iter())
        //     .all(|((&r, &m_inv), &v)| {
        //         let v = Vector3::from(v);
        //         let r = Vector3::from(r);
        //         (r * m_inv).norm() / dt
        //     })
        IterationInfo {
            iteration,
            ls_steps,
            linsolve_result,
            eta,
            alpha,
            merit,
            r_inf: r.as_tensor().lp_norm(LpNorm::Inf).to_f64().unwrap(),
            a_inf: r
                .chunks_exact(3)
                .zip(mass_inv.iter())
                .map(|(r, &m_inv)| Vector3::new([r[0], r[1], r[2]]).norm() * m_inv)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                .unwrap_or(T::zero())
                .to_f64()
                .unwrap(),
            x_2: x.as_tensor().norm().to_f64().unwrap(),
            d_2: x_prev
                .iter()
                .zip(x.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<T>()
                .to_f64()
                .unwrap()
                .sqrt(),
        }
    }

    /// Header for info printed by this struct.
    ///
    /// ```verbatim
    /// i       - iteration number
    /// merit   - Merit function
    /// res-inf - inf-norm of the residual
    /// a-inf   - inf-norm of the nodal acceleration residual
    /// d-2     - 2-norm of the step vector
    /// x-2     - 2-norm of the variable vector
    /// lin #   - number of linear solver iterations
    /// lin err - linear solver error (residual 2-norm divided by rhs 2-norm)
    /// lin tol - linear solver tolerance
    /// alpha   - Fraction of the step taken
    /// ls #    - number of line search steps
    /// ```
    pub fn header() -> [&'static str; 2] {
        [
            "    i |  res-inf   |   a-inf    |   merit    |    d-2     |    x-2     | lin # |  lin err   |  lin tol   |   alpha   | ls # ",
            "------+------------+------------+------------+------------+------------+-------+------------+------------+-----------+------"
        ]
    }
}

impl Display for IterationInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "{i:>5} | {resinf:10.3e} | {ainf:10.3e} | {merit:10.3e} | {di:10.3e} | {xi:10.3e} | {lin:>5} | {linerr:10.3e} | {lintol:10.3e} | {alpha:10.3e} | {ls:>4} ",
            i = self.iteration,
            resinf = self.r_inf,
            ainf = self.a_inf,
            merit = self.merit,
            di = self.d_2,
            xi = self.x_2,
            lin = self.linsolve_result.iterations,
            linerr = self.linsolve_result.error,
            lintol = self.eta,
            alpha = self.alpha,
            ls = self.ls_steps
        )
    }
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

// This makes sense only when j is col major ( which is only on macos )
#[allow(dead_code)]
fn build_sparse_from_product<T: Real>(
    j: &mut DSMatrix<T>,
    mut jprod: impl FnMut(usize, &mut [T]),
    num_variables: usize,
) {
    let mut j = j.view_mut().into_data();
    for (col_idx, mut sparse_col) in j.iter_mut().enumerate() {
        let mut col = vec![T::zero(); num_variables];
        col.iter_mut().for_each(|x| *x = T::zero());
        jprod(col_idx, &mut col);
        for (entry_idx, entry) in sparse_col.iter_mut() {
            *entry = col[*entry_idx];
        }
    }
}

// Helper function for debugging the jacobian.
#[allow(dead_code)]
fn build_dense_from_product<T: Real>(
    mut j_dense: ChunkedN<&mut [T]>,
    mut jprod: impl FnMut(usize, &mut [T]),
    num_variables: usize,
) {
    // Clear j_dense
    for jd in j_dense.storage_mut().iter_mut() {
        *jd = T::zero();
    }
    // Copy j_vals to j_dense
    for c in 0..num_variables {
        jprod(c, &mut j_dense[c]);
    }
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
fn max_sigma<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>) -> T {
    let svd = svd_values(mtx);
    let max_sigma = svd
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    max_sigma
}

#[allow(dead_code)]
fn max_min_sigma<T: Real + na::ComplexField>(mtx: ChunkedN<&[T]>) -> (T, T) {
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
    (max_sigma, min_sigma)
}

#[allow(dead_code)]
fn bound_estimate_and_condition<T: Real + na::ComplexField>(
    mass_inv: &[T],
    mtx: ChunkedN<&[T]>,
    dt: T,
) -> (T, T) {
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
    let mut max_mass_inv = T::zero();
    let mut min_mass_inv = T::infinity();
    for &mass_inv in mass_inv.iter() {
        if mass_inv > max_mass_inv {
            max_mass_inv = mass_inv;
        }
        if mass_inv > T::zero() && mass_inv < min_mass_inv {
            min_mass_inv = mass_inv;
        }
    }

    // eprintln!("dt: {dt:?}; max: {max_sigma}; massmax: {max_mass_inv:?}; min: {min_sigma:?}, massmin: {min_mass_inv:?}");
    (
        num_traits::Float::max(
            T::one() - dt * dt * max_mass_inv * max_sigma,
            dt * dt * min_sigma * min_mass_inv - T::one(),
        ),
        max_sigma / min_sigma,
    )
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
pub enum LinearSolver {
    Iterative {
        /// Residual tolerance for the linear solve.
        tolerance: f32,
        /// Maximum number of iterations permitted for the linear solve.
        max_iterations: u32,
    },
    Direct,
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
    /// Contact aware line search that truncates `α` according to expected friction and contact
    /// potentials before using backtracking to ensure sufficient decrease.
    AssistedBackTracking {
        c: f64,
        rho: f64,
    },
    None,
}

impl Default for LineSearch {
    fn default() -> LineSearch {
        LineSearch::default_assisted_backtracking()
    }
}

impl LineSearch {
    pub const fn default_assisted_backtracking() -> Self {
        LineSearch::AssistedBackTracking { c: 1e-4, rho: 0.9 }
    }
    pub const fn default_backtracking() -> Self {
        LineSearch::BackTracking { c: 1e-4, rho: 0.9 }
    }
    pub fn is_assisted(&self) -> bool {
        matches!(self, LineSearch::AssistedBackTracking { .. })
    }
    /// Gets the factor by which the step size should be decreased.
    pub fn step_factor(&self) -> f64 {
        match self {
            LineSearch::BackTracking { rho, .. } | LineSearch::AssistedBackTracking { rho, .. } => {
                *rho
            }
            LineSearch::None => 1.0,
        }
    }

    // Gets the coefficient for the Armijo condition.
    pub fn armijo_coeff(&self) -> f64 {
        match self {
            LineSearch::BackTracking { c, .. } | LineSearch::AssistedBackTracking { c, .. } => *c,
            LineSearch::None => 1.0,
        }
    }
}
