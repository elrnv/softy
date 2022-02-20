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
    /// Linear solver configuration,
    pub linsolve: LinearSolver,
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
        // Update values and refactor
        self.update_rhs(&r);
        let values: Vec<_> = values.iter().map(|&x| x.to_f64().unwrap()).collect();
        self.solver
            .refactor(values.as_slice(), mkl::dss::Definiteness::Indefinite)?;
        self.solver
            .solve_into(&mut self.sol, &mut self.buf, &mut self.r64)?;
        Ok(&self.sol)
    }

    #[cfg(target_os = "macos")]
    pub fn solve_with_values<T: Real>(
        &mut self,
        r: &[T],
        values: &[T],
    ) -> Result<&[f64], SparseDirectSolveError> {
        self.update_values(values);
        self.refactor()?;
        self.update_rhs(&r);
        self.solve()
    }
}

pub struct DirectSolver<T: Real> {
    /// Mapping from original triplets given by the `j_*` members to the final
    /// compressed sparse matrix.
    #[cfg(not(target_os = "macos"))]
    j_mapping: Vec<Index>,
    #[cfg(target_os = "macos")]
    j_t_mapping: Vec<Index>,
    #[cfg(not(target_os = "macos"))]
    j: DSMatrix<T>,
    #[cfg(target_os = "macos")]
    j_t: DSMatrix<T>,
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
    Iterative(BiCGSTAB<T>),
    Direct(DirectSolver<T>),
}

impl<T: Real> LinearSolverWorkspace<T> {
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
    init_j_t_mapping: Vec<Index>,
    x_prev: Vec<T>,
    r: Vec<T>,
    p: Vec<T>,
    jp: Vec<T>,
    r_cur: Vec<T>,
    r_next: Vec<T>,
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
                    #[cfg(not(target_os = "macos"))]
                    j: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
                    #[cfg(target_os = "macos")]
                    j_t: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
                    //#[cfg(target_os = "macos")]
                    //j_sparse: LazyCell::new(),
                    #[cfg(not(target_os = "macos"))]
                    j_mapping: Vec::new(),
                    #[cfg(target_os = "macos")]
                    j_t_mapping: Vec::new(),
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
                init_j_t_mapping: Vec::new(),
                x_prev,
                r,
                p,
                jp,
                r_cur,
                r_next,
            }),
        }
    }
    fn update_jacobian_indices(
        problem: &P,
        sparse_jacobian: &mut SparseJacobian<T>,
        init_sparse_jacobian_vals: &mut Vec<T>,
        init_sparse_solver: &mut LazyCell<SparseDirectSolver>,
        init_j_t_mapping: &mut Vec<Index>,
        linsolve: &mut LinearSolverWorkspace<T>,
    ) {
        // Construct the sparse Jacobian.
        let n = problem.num_variables();
        let (j_rows, j_cols) = problem.jacobian_indices();
        assert_eq!(j_rows.len(), j_cols.len());
        let j_nnz = j_rows.len();
        log::debug!("Number of Jacobian non-zeros: {}", j_nnz);
        sparse_jacobian.j_rows = j_rows;
        sparse_jacobian.j_cols = j_cols;
        sparse_jacobian.j_vals.resize(j_nnz, T::zero());
        let (j_t, j_t_mapping) = sparse_matrix_and_mapping(
            &sparse_jacobian.j_cols,
            &sparse_jacobian.j_rows,
            &sparse_jacobian.j_vals,
            n,
        );
        init_sparse_solver.replace(SparseDirectSolver::new(j_t.view()).unwrap());
        *init_j_t_mapping = j_t_mapping;
        init_sparse_jacobian_vals.clone_from(j_t.storage());
        if let LinearSolverWorkspace::Direct(ds) = linsolve {
            //let (j, j_mapping) = sparse_matrix_and_mapping(&j_rows, &j_cols, &sj.j_vals, n);
            #[cfg(not(target_os = "macos"))]
            let (j, j_mapping) = sparse_matrix_and_mapping(
                &sparse_jacobian.j_rows,
                &sparse_jacobian.j_cols,
                &sparse_jacobian.j_vals,
                n,
            );
            #[cfg(target_os = "macos")]
            let (j_t, j_t_mapping) = sparse_matrix_and_mapping(
                &sparse_jacobian.j_cols,
                &sparse_jacobian.j_rows,
                &sparse_jacobian.j_vals,
                n,
            );

            //#[cfg(target_os = "macos")]
            //ds.j_sparse.replace(new_sparse(j_t.view(), true));
            #[cfg(target_os = "macos")]
            {
                ds.j_t = j_t;
                ds.j_t_mapping = j_t_mapping;
            }
            #[cfg(not(target_os = "macos"))]
            {
                ds.j = j;
                ds.j_mapping = j_mapping;
            }
            #[cfg(target_os = "macos")]
            ds.sparse_solver
                .replace(SparseDirectSolver::new(ds.j_t.view()).unwrap());
            #[cfg(not(target_os = "macos"))]
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
            init_j_t_mapping,
            linsolve,
            ..
        } = &mut *self.workspace.borrow_mut();
        Self::update_jacobian_indices(
            &self.problem,
            sj,
            init_sparse_jacobian_vals,
            init_sparse_solver,
            init_j_t_mapping,
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
        if x.is_empty() {
            return SolveResult {
                iterations: 0,
                status: Status::NothingToSolve,
            };
        }

        // Timing stats
        let mut linsolve_time = Duration::new(0, 0);
        let mut ls_time = Duration::new(0, 0);
        let mut jprod_linsolve_time = Duration::new(0, 0);
        // let mut jprod_ls_time = Duration::new(0, 0);
        let mut residual_time = Duration::new(0, 0);
        let mut assist_time = Duration::new(0, 0);
        let mut total_solve_time = Duration::new(0, 0);
        self.problem.timings().clear();
        let t_begin_solve = Instant::now();

        {
            let Self {
                problem, workspace, ..
            } = self;

            let NewtonWorkspace { r, .. } = &mut *workspace.borrow_mut();

            // Initialize the residual.
            add_time!(residual_time; problem.residual(x, r.as_mut_slice()));
        }

        if update_jacobian_indices {
            self.update_jacobian_indices();
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
            init_sparse_jacobian_vals,
            init_sparse_solver,
            init_j_t_mapping,
            x_prev,
            r,
            p,
            jp,
            r_cur,
            r_next,
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

        let sigma = linsolve.iterative_tolerance();
        //let orig_sigma = sigma;

        log_debug_stats_header();
        log_debug_stats(0, 0, linsolve_result, 1.0, f64::INFINITY, &r, x, &x_prev);

        // Prepare initial Jacobian used in the merit function.
        problem.jacobian_values(
            x,
            &r,
            &sparse_jacobian.j_rows,
            &sparse_jacobian.j_cols,
            sparse_jacobian.j_vals.as_mut_slice(),
        );

        init_sparse_jacobian_vals
            .iter_mut()
            .for_each(|x| *x = T::zero());
        for (&pos, &j_val) in init_j_t_mapping.iter().zip(sparse_jacobian.j_vals.iter()) {
            if let Some(pos) = pos.into_option() {
                init_sparse_jacobian_vals[pos] += j_val;
            }
        }

        let init_sparse_solver = init_sparse_solver
            .borrow_mut()
            .expect("Uninitialized iterative sparse solver.");
        #[cfg(target_os = "macos")]
        {
            init_sparse_solver.update_values(&init_sparse_jacobian_vals);
            match init_sparse_solver.refactor() {
                Err(err) => {
                    return SolveResult {
                        iterations,
                        status: Status::LinearSolveError(err.into()),
                    }
                }
                Ok(_) => {}
            }
        }

        // let mut merit = |problem: &P, _: &[T], residual: &[T], init_sparse_solver: &mut SparseDirectSolver| {
        //     init_sparse_solver.update_rhs(residual);
        //     let result = init_sparse_solver.solve();
        //     let r64 = result.expect("Initial Jacobian is singular.");
        //     0.5 * r64.as_tensor().norm_squared()
        // };
        let merit = |_: &P, _: &[T], r: &[T], _: &SparseDirectSolver| {
            0.5 * r.as_tensor().norm_squared().to_f64().unwrap()
        };
        // let merit = |problem: &P, x: &[T], _: &[T], _: &SparseDirectSolver| {
        //     problem.objective(x).to_f64().unwrap()
        // };

        // Computes the product of merit function with the search direction.
        // In this version, we leverage that Jp is already computed.
        // let mut merit_jac_prod = |problem: &P, jp: &[T], r: &[T], init_sparse_solver: &mut SparseDirectSolver| {
        //     init_sparse_solver.update_rhs(jp);
        //     let result = init_sparse_solver.solve();
        //     let jinv_jp = result.expect("Initial Jacobian is singular.").to_vec();
        //     // TODO: don't need to compute this again in general.
        //     init_sparse_solver.update_rhs(r);
        //     let result = init_sparse_solver.solve();
        //     let jinv_r = result.expect("Initial Jacobian is singular.");
        //     jinv_jp
        //         .iter()
        //         .zip(jinv_r.iter())
        //         .fold(0.0, |acc, (&jinv_jp, &jinv_r)| acc + jinv_jp * jinv_r)
        // };
        let merit_jac_prod = |_: &P, jp: &[T], r: &[T], _: &SparseDirectSolver| {
            r.as_tensor().norm_squared().to_f64().unwrap()
            // jp.iter()
            //     .zip(r.iter())
            //     .fold(0.0, |acc, (&jp, &r)| acc + (jp * r).to_f64().unwrap())
        };

        // Keep track of merit function to avoid having to recompute it
        let mut merit_cur = merit(problem, x, r, init_sparse_solver);
        //let mut merit_prev = merit_cur;
        let mut merit_next;

        let mut j_dense_ad = LazyCell::new();
        let mut j_dense = LazyCell::new();
        let mut identity =
            ChunkedN::from_flat_with_stride(x.len(), vec![T::zero(); x.len() * r.len()]);
        for (i, id) in identity.iter_mut().enumerate() {
            id[i] = T::one();
        }

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

            //log::trace!("r = {:?}", &r);

            if !merit_cur.is_finite() {
                break SolveResult {
                    iterations,
                    status: Status::Diverged,
                };
            }

            let t_begin_linsolve = Instant::now();

            // Update tolerance (forcing term)
            //sigma = orig_sigma.min(((merit_cur - linsolve_result.residual).abs() / merit_prev) as f32);

            // Update Jacobian values.
            //let before_j = Instant::now();

            r_cur.copy_from_slice(&r);

            match linsolve {
                LinearSolverWorkspace::Iterative(linsolve) => {
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

                    linsolve_result = linsolve.solve_precond(
                        |p, out| {
                            let t_begin_jprod = Instant::now();
                            problem.jacobian_product(x, p, r_cur, out);
                            //sparse_jacobian.compute_product(p, out);
                            jprod_linsolve_time += Instant::now() - t_begin_jprod;
                            inner_callback.borrow_mut()(CallbackArgs {
                                residual: r_cur.as_slice(),
                                x,
                                problem,
                                iteration: iterations,
                            })
                        },
                        p.as_mut_slice(),
                        r.as_mut_slice(),
                        |_| true,
                    );
                }
                LinearSolverWorkspace::Direct(DirectSolver {
                    #[cfg(not(target_os = "macos"))]
                    j_mapping,
                    #[cfg(not(target_os = "macos"))]
                    j,
                    #[cfg(target_os = "macos")]
                        j_t_mapping: j_mapping,
                    #[cfg(target_os = "macos")]
                        j_t: j,
                    sparse_solver,
                    ..
                }) => {
                    problem.jacobian_values(
                        x,
                        &r_cur,
                        &sparse_jacobian.j_rows,
                        &sparse_jacobian.j_cols,
                        sparse_jacobian.j_vals.as_mut_slice(),
                    );
                    log::trace!("Condition number: {:?}", {
                        let j_dense_ad = j_dense_ad.borrow_mut_with(|| {
                            ChunkedN::from_flat_with_stride(
                                x.len(),
                                vec![T::zero(); x.len() * r.len()],
                            )
                        });
                        build_dense_from_product(
                            j_dense_ad.view_mut(),
                            |i, col| {
                                problem.jacobian_product(x, &identity[i], r, col);
                            },
                            x.len(),
                        );
                        print_dense(j_dense_ad.view());

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
                        print_dense(j_dense.view());
                        //log::debug!("J singular values: {:?}", svd_values(j_dense.view()));
                        //write_jacobian_img(j_dense.view(), iterations);

                        // let mut success = true;
                        // for i in 0..j_dense.len() {
                        //     for j in 0..j_dense.len() {
                        //         let a = *j_dense.view().at(i).at(j);
                        //         let b = *j_dense_ad.view().at(j).at(i);
                        //         if num_traits::Float::abs(a - b) > T::from(1e-4).unwrap() {
                        //             eprintln!("({},{}): {} vs {}", i, j, a, b);
                        //             success = false;
                        //         }
                        //     }
                        // }
                        // if !success {
                        //     panic!("Jacobian Error");
                        // }
                        condition_number(j_dense.view())
                    });
                    //jprod_time += Instant::now() - before_j;

                    ////log::trace!("j_vals = {:?}", &j_vals);

                    //// Zero out Jacobian.
                    //j.storage_mut().iter_mut().for_each(|x| *x = T::zero());
                    j.storage_mut().iter_mut().for_each(|x| *x = T::zero());

                    // Update the Jacobian matrix.
                    // for (&pos, &j_val) in j_mapping.iter().zip(j_vals.iter()) {
                    //     if let Some(pos) = pos.into_option() {
                    //         j.storage_mut()[pos] += j_val;
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
                    let result = sparse_solver.solve_with_values(&r, j.storage());
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

                    // Check
                    // problem.jacobian_product(x, p, r_cur, jp.as_mut_slice());
                    // eprintln!("r = {:?}", &r_cur);
                    // eprintln!("jp_check = {:?}", &jp);

                    //log::trace!("linsolve result: {:?}", linsolve_result);
                }
            }

            // Negate
            p.iter_mut().for_each(|p| *p = -*p);

            //log::trace!("p = {:?}", &r);

            linsolve_time += Instant::now() - t_begin_linsolve;

            // The solve converts the rhs r into the unknown negative search direction p.

            if !p.as_tensor().norm_squared().is_finite() {
                break SolveResult {
                    iterations,
                    status: Status::StepTooLarge,
                };
            }

            // Update previous step.
            x_prev.copy_from_slice(x);

            let rho = params.line_search.step_factor();

            // Take the full step
            *x.as_mut_tensor() += p.as_tensor();

            // Compute the residual for the full step.
            let t_begin_residual = Instant::now();
            problem.residual(&x, r_next.as_mut_slice());
            residual_time += Instant::now() - t_begin_residual;

            let (ls_count, alpha) = add_time! {
                ls_time;
                if rho >= 1.0 {
                    merit_next = merit(problem, x, r_next, init_sparse_solver);
                    (1, 1.0)
                } else {
                    // Line search.
                    let mut alpha = 1.0;
                    let mut ls_count = 1;
                    //let mut sigma = linsolve.tol as f64;

                    // let t_begin_jprod = Instant::now();
                    //jp.iter_mut().for_each(|x| *x = T::zero());
                    //j.view()
                    //    .add_mul_in_place_par(p.as_tensor(), jp.as_mut_tensor());
                    // problem.jacobian_product(x_prev, p, r_cur, jp.as_mut_slice());
                    //sparse_jacobian.compute_product(p, jp.as_mut_slice());
                    // jprod_ls_time += Instant::now() - t_begin_jprod;
                    // eprintln!("jp = {:?}", &jp);
                    let merit_jac_p = merit_jac_prod(problem, r_cur, r_cur, init_sparse_solver);
                    // let merit_jac_p = merit_jac_prod(problem, p, r_cur, init_sparse_solver);
                    // dbg!(&x_prev);
                    // dbg!(&x);
                    // dbg!(&p);
                    // dbg!(&r_cur);
                    // dbg!(&r_next);
                    // dbg!(merit_jac_p);
                    // dbg!(merit_cur);

                    if params.line_search.is_assisted() {
                        add_time!(
                            assist_time;
                            alpha = problem
                                .assist_line_search(T::from(alpha).unwrap(), p, x_prev, r_cur, r_next)
                                .to_f64()
                                .unwrap()
                        );
                    }

                    loop {
                        // Compute gradient of the merit function 0.5 r'r  multiplied by p, which is r' dr/dx p.
                        // Gradient dot search direction.
                        // Check Armijo condition:
                        // Given f(x) = 0.5 || r(x) ||^2 = 0.5 r(x)'r(x)
                        // Test f(x + αp) < f(x) - cα(J(x)'r(x))'p(x)
                        // Test f(x + αp) < f(x) - cα r(x)'J(x)p(x)

                        // Compute the merit function
                        merit_next = merit(problem, x, r_next, init_sparse_solver);

                        // TRADITIONAL BACKTRACKING:
                        if merit_next
                            <= merit_cur - params.line_search.armijo_coeff() * alpha * merit_jac_p
                        {
                            // eprintln!("success: {merit_next} <= {merit_cur} - {}; alpha <- {}", params.line_search.armijo_coeff() * alpha * merit_jac_p, alpha*rho);
                            break;
                        }

                        // eprintln!("backtracking: {merit_next} > {merit_cur} - {}; alpha <- {}", params.line_search.armijo_coeff() * alpha * merit_jac_p, alpha*rho);

                        // INEXACT NEWTON:
                        // if merit_next <= merit_cur * (1.0 - rho * (1.0 - sigma)) {
                        //     break;
                        // }
                        //
                        alpha *= rho;

                        // sigma = 1.0 - alpha * (1.0 - sigma);

                        // Break if alpha becomes too small. This is usually a bad sign.
                        if alpha < 1e-5 {
                            break;
                        }

                        // Take a fractional step.
                        zip!(x.iter_mut(), x_prev.iter(), p.iter()).for_each(|(x, &x0, &p)| {
                            *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                        });

                        // Compute the candidate residual.
                        add_time!(residual_time; problem.residual(x, r_next.as_mut_slice()));

                        ls_count += 1;
                    }

                    // dbg!(alpha);
                    // if ls_count > 10 {
                    //     let mut merit_data = vec![];
                    //     let mut r0 = vec![];
                    //     let mut r1 = vec![];
                    //     let mut r2 = vec![];
                    //     let mut r3 = vec![];
                    //     let mut r4 = vec![];
                    //     let mut r5 = vec![];
                    //     let mut r6 = vec![];
                    //     let mut r7 = vec![];
                    //     let mut r8 = vec![];
                    //     let mut r9 = vec![];
                    //     let mut r10 = vec![];
                    //     let mut r11 = vec![];
                    //     let mut f0 = vec![];
                    //     let mut f1 = vec![];
                    //     let mut f2 = vec![];
                    //     let mut f3 = vec![];
                    //     let mut f4 = vec![];
                    //     let mut f5 = vec![];
                    //     let mut f6 = vec![];
                    //     let mut f7 = vec![];
                    //     let mut f8 = vec![];
                    //     let mut f9 = vec![];
                    //     let mut f10 = vec![];
                    //     let mut f11 = vec![];
                    //
                    //     let mut xs = vec![];
                    //     let mut probe_r = vec![T::zero(); r_next.len()];
                    //     let mut probe_x = vec![T::zero(); x.len()];
                    //     for i in 0..=1000 {
                    //         let alpha: f64 = 0.00001 * i as f64;
                    //         zip!(probe_x.iter_mut(), x_prev.iter(), p.iter()).for_each(
                    //             |(x, &x0, &p)| {
                    //                 *x = num_traits::Float::mul_add(p, T::from(alpha).unwrap(), x0);
                    //             },
                    //         );
                    //         problem.residual(&probe_x, probe_r.as_mut_slice());
                    //         if i == 0 || i == 1000 {
                    //             dbg!(&probe_r);
                    //         }
                    //         let probe_f = problem.debug_friction();
                    //         let probe = merit(problem, &probe_x, &probe_r, init_sparse_solver);
                    //         xs.push(alpha);
                    //         merit_data.push(probe);
                    //         r0.push(probe_r[0]);
                    //         r1.push(probe_r[1]);
                    //         r2.push(probe_r[2]);
                    //         r3.push(probe_r[3]);
                    //         r4.push(probe_r[4]);
                    //         r5.push(probe_r[5]);
                    //         r6.push(probe_r[30]);
                    //         r7.push(probe_r[31]);
                    //         r8.push(probe_r[32]);
                    //         r9.push(probe_r[33]);
                    //         r10.push(probe_r[34]);
                    //         r11.push(probe_r[35]);
                    //         f0.push(probe_f[0]);
                    //         f1.push(probe_f[1]);
                    //         f2.push(probe_f[2]);
                    //         f3.push(probe_f[3]);
                    //         f4.push(probe_f[4]);
                    //         f5.push(probe_f[5]);
                    //         f6.push(probe_f[30]);
                    //         f7.push(probe_f[31]);
                    //         f8.push(probe_f[32]);
                    //         f9.push(probe_f[33]);
                    //         f10.push(probe_f[34]);
                    //         f11.push(probe_f[35]);
                    //     }
                    //     use std::io::Write;
                    //     let mut file = std::fs::File::create("./out/debug_data.jl").unwrap();
                    //     writeln!(file, "xs = {:?}", xs);
                    //     writeln!(file, "merit_data = {:?}", merit_data);
                    //     writeln!(file, "r0 = {:?}", r0);
                    //     writeln!(file, "r1 = {:?}", r1);
                    //     writeln!(file, "r2 = {:?}", r2);
                    //     writeln!(file, "r3 = {:?}", r3);
                    //     writeln!(file, "r4 = {:?}", r4);
                    //     writeln!(file, "r5 = {:?}", r5);
                    //     writeln!(file, "r6 = {:?}", r6);
                    //     writeln!(file, "r7 = {:?}", r7);
                    //     writeln!(file, "r8 = {:?}", r8);
                    //     writeln!(file, "r9 = {:?}", r9);
                    //     writeln!(file, "r10 = {:?}", r10);
                    //     writeln!(file, "r11 = {:?}", r11);
                    //     writeln!(file, "f0 = {:?}", f0);
                    //     writeln!(file, "f1 = {:?}", f1);
                    //     writeln!(file, "f2 = {:?}", f2);
                    //     writeln!(file, "f3 = {:?}", f3);
                    //     writeln!(file, "f4 = {:?}", f4);
                    //     writeln!(file, "f5 = {:?}", f5);
                    //     writeln!(file, "f6 = {:?}", f6);
                    //     writeln!(file, "f7 = {:?}", f7);
                    //     writeln!(file, "f8 = {:?}", f8);
                    //     writeln!(file, "f9 = {:?}", f9);
                    //     writeln!(file, "f10 = {:?}", f10);
                    //     writeln!(file, "f11 = {:?}", f11);
                    //     writeln!(file, "xs_length = {:?}", xs.len());
                    //     panic!("STOP");
                    // }
                    (ls_count, alpha)
                }
            };

            iterations += 1;

            log_debug_stats(
                iterations,
                ls_count,
                linsolve_result,
                alpha as f32,
                merit_next,
                &r_next,
                x,
                &x_prev,
            );

            // Check the convergence condition.
            if problem.converged(
                x_prev,
                x,
                r_next,
                merit_next,
                params.x_tol,
                params.r_tol,
                params.a_tol,
            ) {
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

        total_solve_time = Instant::now() - t_begin_solve;

        log::debug!("Line search assist time: {}ms", assist_time.as_millis());
        log::debug!(
            "Balance equation computation time: {}ms",
            residual_time.as_millis()
        );
        log::debug!(
            "   Energy gradient time: {}ms",
            self.problem.timings().energy_gradient.as_millis()
        );
        log::debug!(
            "   Contact prep time: {}ms",
            self.problem.timings().prepare_contact.as_millis()
        );
        log::debug!(
            "   Contact force time: {}ms",
            self.problem.timings().contact_force.as_millis()
        );
        log::debug!(
            "   Friction force time: {}ms",
            self.problem.timings().friction_force.as_millis()
        );
        log::debug!("Linear solve time: {}ms", linsolve_time.as_millis());
        log::debug!(
            "   Jacobian product time: {}ms",
            jprod_linsolve_time.as_millis()
        );
        log::debug!("Line search time: {}ms", ls_time.as_millis());
        // log::debug!("   Jacobian product time: {}ms", jprod_ls_time.as_millis());
        log::debug!("Total solve time {}ms", total_solve_time.as_millis());
        result
    }
}

/*
 * Status print routines.
 * i       - iteration number
 * res-2   - 2-norm of the residual
 * merit   - Merit function
 * res-inf - inf-norm of the residual
 * d-inf   - inf-norm of the step vector
 * x-inf   - inf-norm of the variable vector
 * lin #   - number of linear solver iterations
 * ls #    - number of line search steps
 */
fn log_debug_stats_header() {
    log::debug!(
        "    i |  res-inf   |   merit    |    d-2     |    x-2     | lin # |  lin err   |   alpha    | ls # "
    );
    log::debug!(
        "------+------------+------------+------------+------------+-------+------------+------------+------"
    );
}
fn log_debug_stats<T: Real>(
    iterations: u32,
    ls_steps: u32,
    linsolve_result: super::linsolve::SolveResult,
    alpha: f32,
    merit: f64,
    r: &[T],
    x: &[T],
    x_prev: &[T],
) {
    log::debug!(
        "{i:>5} | {resinf:10.3e} | {merit:10.3e} | {di:10.3e} | {xi:10.3e} | {lin:>5} | {linerr:10.3e} | {alpha:10.3e} | {ls:>4} ",
        i = iterations,
        resinf = r.as_tensor().lp_norm(LpNorm::Inf).to_f64().unwrap(),
        merit = merit,
        di = x_prev.iter().zip(x.iter()).map(|(&a, &b)| (a - b)*(a-b)).sum::<T>()
            .to_f64()
            .unwrap().sqrt(),
        xi = x.as_tensor().norm().to_f64().unwrap(),
        lin = linsolve_result.iterations,
        linerr = linsolve_result.error,
        alpha = alpha,
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
fn build_dense_from_product<T: Real>(
    mut j_dense: ChunkedN<&mut [T]>,
    jprod: impl Fn(usize, &mut [T]),
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
        match self {
            LineSearch::AssistedBackTracking { .. } => true,
            _ => false,
        }
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
