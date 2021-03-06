use std::cell::RefCell;

use rayon::prelude::*;
use tensr::*;

use super::newton::{Newton, NewtonParams};
use super::problem::{MixedComplementarityProblem, NonLinearProblem};
use super::trust_region::{TrustRegion, TrustRegionParams};
use super::NLSolver;
use super::{Callback, SolveResult};

/// A mixed complementarity problem solver.
pub struct MCPSolver<S> {
    solver: S,
}

/// A wrapper around an MCP that makes it look like a regular NonLinearProblem to the Newton solver.
pub struct MCP<P> {
    problem: P,
    bounds: RefCell<(Vec<f64>, Vec<f64>)>,
    workspace_seen: RefCell<Vec<bool>>,
}

impl<T, P> NonLinearProblem<T> for MCP<P>
where
    T: Real,
    P: MixedComplementarityProblem<T>,
{
    fn num_variables(&self) -> usize {
        self.problem.num_variables()
    }
    fn initial_point(&self) -> Vec<T> {
        self.problem.initial_point()
    }
    fn residual(&self, x: &[T], r: &mut [T]) {
        self.problem.residual(x, r);
        let (l, u) = &*self.bounds.borrow();
        zip!(r.iter_mut(), x.iter(), l.iter(), u.iter()).for_each(|(r, &x, &l, &u)| {
            *r = r.max(x - T::from(u).unwrap()).min(x - T::from(l).unwrap());
        });
    }
    fn jacobian_indices(&self) -> (Vec<usize>, Vec<usize>) {
        self.problem.jacobian_indices()
    }
    fn jacobian_values(&self, x: &[T], r: &[T], rows: &[usize], cols: &[usize], values: &mut [T]) {
        self.problem.jacobian_values(x, r, rows, cols, values);
        let mut seen = self.workspace_seen.borrow_mut();
        seen.iter_mut().for_each(|x| *x = false);

        let (l, u) = &*self.bounds.borrow();
        for (&i, &j, v) in zip!(rows.iter(), cols.iter(), values.iter_mut()) {
            let u = T::from(u[i]).unwrap();
            let l = T::from(l[i]).unwrap();
            if r[i] <= x[i] - u || r[i] >= x[i] - l {
                *v = if i == j && !seen[i] {
                    // We are modifying values which can have repeated elements on the diagonal.
                    // Using the seen vector we ensure to set only one diagonal to 1.0.
                    seen[i] = true;
                    T::one()
                } else {
                    T::zero()
                };
            }
        }
    }
    fn jacobian_product(
        &self,
        x: &[T],
        p: &[T],
        r: &[T],
        rows: &[usize],
        cols: &[usize],
        jp: &mut [T],
    ) {
        self.problem.jacobian_product(x, p, r, rows, cols, jp);
        let (l, u) = &*self.bounds.borrow();
        zip!(
            x.par_iter(),
            l.par_iter(),
            u.par_iter(),
            r.par_iter(),
            jp.par_iter_mut()
        )
        .for_each(|(&x, &l, &u, &r, jp)| {
            let u = T::from(u).unwrap();
            let l = T::from(l).unwrap();
            if r <= x - u {
                *jp = x - u;
            } else if r >= x - l {
                *jp = x - l;
            }
        });
    }
}

impl<S, T, P> NLSolver<P, T> for MCPSolver<S>
where
    S: NLSolver<MCP<P>, T>,
    T: Real + na::RealField,
    P: MixedComplementarityProblem<T>,
{
    /// Gets a reference to the outer callback function.
    ///
    /// This callback gets called at the beginning of every Newton iteration.
    fn outer_callback(&self) -> &RefCell<Callback<T>> {
        &self.solver.outer_callback()
    }
    /// Gets a reference to the inner callback function.
    ///
    /// This is the callback that gets called for every inner linear solve.
    fn inner_callback(&self) -> &RefCell<Callback<T>> {
        &self.solver.inner_callback()
    }
    /// Gets a reference to the underlying problem instance.
    fn problem(&self) -> &P {
        &self.solver.problem().problem
    }
    /// Gets a mutable reference the underlying problem instance.
    fn problem_mut(&mut self) -> &mut P {
        &mut self.solver.problem_mut().problem
    }
    /// Solves the problem and returns the solution along with the solve result
    /// info.
    fn solve(&mut self) -> (Vec<T>, SolveResult) {
        {
            let (l, u) = &mut *self.solver.problem().bounds.borrow_mut();
            self.problem()
                .update_bounds(l.as_mut_slice(), u.as_mut_slice());
            // This scope drops the l and u mutable borrows.
        }
        self.solver.solve()
    }

    /// Solve the problem given an initial guess `x` and returns the solve result
    /// info.
    ///
    /// This version of [`solve`] does not rely on the `initial_point` method of
    /// the problem definition. Instead the given `x` is used as the initial point.
    fn solve_with(&mut self, x: &mut [T]) -> SolveResult {
        {
            let (l, u) = &mut *self.solver.problem().bounds.borrow_mut();
            self.problem()
                .update_bounds(l.as_mut_slice(), u.as_mut_slice());
            // This scope drops the l and u mutable borrows.
        }
        self.solver.solve_with(x)
    }
}

impl<T, P> MCPSolver<Newton<MCP<P>, T>>
where
    T: Real + na::RealField,
    P: MixedComplementarityProblem<T>,
{
    pub fn newton(
        problem: P,
        params: NewtonParams,
        outer_callback: Callback<T>,
        inner_callback: Callback<T>,
    ) -> Self {
        let n = problem.num_variables();
        // Create bounds.
        let (l, u) = problem.initial_bounds();
        assert_eq!(l.len(), n);
        assert_eq!(u.len(), n);

        let mcp = MCP {
            problem,
            bounds: RefCell::new((l, u)),
            workspace_seen: RefCell::new(vec![false; n]),
        };

        let newton = Newton::new(mcp, params, outer_callback, inner_callback);

        MCPSolver { solver: newton }
    }
}
impl<T, P> MCPSolver<TrustRegion<MCP<P>, T>>
where
    T: Real + na::RealField,
    P: MixedComplementarityProblem<T>,
{
    pub fn trust_region(
        problem: P,
        params: TrustRegionParams,
        outer_callback: Callback<T>,
        inner_callback: Callback<T>,
    ) -> Self {
        let n = problem.num_variables();
        // Create bounds.
        let (l, u) = problem.initial_bounds();
        assert_eq!(l.len(), n);
        assert_eq!(u.len(), n);

        let mcp = MCP {
            problem,
            bounds: RefCell::new((l, u)),
            workspace_seen: RefCell::new(vec![false; n]),
        };

        let trust_region = TrustRegion::new(mcp, params, outer_callback, inner_callback);

        MCPSolver {
            solver: trust_region,
        }
    }
}
