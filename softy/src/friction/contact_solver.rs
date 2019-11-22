use super::FrictionParams;
use crate::contact::*;
use ipopt::{self, Index, Ipopt, Number};
use utils::soap::*;

use crate::Error;

/// Contact solver.
pub struct ContactSolver<'a> {
    /// Non-linear solver.
    solver: Ipopt<ContactProblem<'a>>,
}

impl<'a> ContactSolver<'a> {
    pub fn new(
        predictor_impulse: &'a [[f64; 3]],
        contact_impulse_n: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
    ) -> Result<ContactSolver<'a>, Error> {
        let basis_mtx = contact_basis.normal_basis_matrix();
        let hessian = mass_inv_mtx
            .clone()
            .diagonal_congruence_transform3x1(basis_mtx.view());

        let predictor_impulse = Chunked3::from_array_slice(predictor_impulse);
        let pred = mass_inv_mtx.view() * Tensor::new(predictor_impulse.view());
        let predictor_impulse_n: Vec<f64> = contact_basis
            .to_normal_space(pred.view().into_inner().into())
            .collect();

        let problem = ContactProblem {
            predictor_impulse_n,
            init_contact_impulse_n: contact_impulse_n,
            hessian,
        };

        let mut ipopt = Ipopt::new_newton(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        //ipopt.set_option("nlp_scaling_max_gradient", 1.0);
        //ipopt.set_option("nlp_scaling_method", "user-scaling");
        //ipopt.set_option("derivative_test", "second-order");
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("hessian_constant", "yes");
        ipopt.set_option("max_iter", params.inner_iterations as i32);

        Ok(ContactSolver { solver: ipopt })
    }

    /// Solve one step.
    pub fn step(&mut self) -> Result<Vec<f64>, Error> {
        let ipopt::SolveResult {
            solver_data,
            status,
            ..
        } = self.solver.solve();

        let solution = solver_data.solution.primal_variables;

        match status {
            ipopt::SolveStatus::SolveSucceeded
            | ipopt::SolveStatus::SolvedToAcceptableLevel
            | ipopt::SolveStatus::MaximumIterationsExceeded => Ok(solution.to_vec()),
            e => Err(Error::ContactSolveError { status: e }),
        }
    }
}

pub(crate) struct ContactProblem<'a> {
    predictor_impulse_n: Vec<f64>,
    init_contact_impulse_n: &'a [f64],
    /// Workspace Hessian matrix.
    hessian: DSMatrix,
}

impl<'a> ContactProblem<'a> {
    pub fn num_contacts(&self) -> usize {
        self.predictor_impulse_n.len()
    }
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for ContactProblem<'_> {
    fn num_variables(&self) -> usize {
        self.num_contacts()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.iter_mut().for_each(|x| *x = -2e19);
        x_u.iter_mut().for_each(|x| *x = 0.0);
        true
    }

    fn initial_point(&self, r: &mut [Number]) -> bool {
        r.copy_from_slice(self.init_contact_impulse_n);
        true
    }

    fn objective(&self, r_n: &[Number], obj: &mut Number) -> bool {
        let ContactProblem {
            predictor_impulse_n,
            hessian,
            ..
        } = self;

        assert_eq!(predictor_impulse_n.len(), r_n.len());

        let mut rhs = hessian.view() * (Tensor::new(r_n) * 0.5).view();
        rhs -= predictor_impulse_n.view().as_tensor();

        *obj = r_n.expr().dot(rhs.expr());

        true
    }

    fn objective_grad(&self, r_n: &[Number], grad_f_n: &mut [Number]) -> bool {
        let ContactProblem {
            predictor_impulse_n,
            hessian,
            ..
        } = self;

        let mut rhs = hessian.view() * Tensor::new(r_n);
        rhs -= predictor_impulse_n.view().as_tensor();

        assert_eq!(grad_f_n.len(), rhs.len());
        grad_f_n.copy_from_slice(rhs.view().into_inner());

        true
    }
}

impl ipopt::NewtonProblem for ContactProblem<'_> {
    fn num_hessian_non_zeros(&self) -> usize {
        let mut idx = 0;
        for (row_idx, row) in self.hessian.data.iter().enumerate() {
            for col_idx in row.index_iter() {
                if row_idx >= col_idx {
                    idx += 1;
                }
            }
        }
        idx
    }

    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // Diagonal objective Hessian.
        let mut idx = 0;
        for (row_idx, row) in self.hessian.data.iter().enumerate() {
            for col_idx in row.index_iter() {
                if row_idx >= col_idx {
                    rows[idx] = row_idx as Index;
                    cols[idx] = col_idx as Index;
                    idx += 1;
                }
            }
        }
        true
    }

    fn hessian_values(&self, _r: &[Number], vals: &mut [Number]) -> bool {
        let mut idx = 0;
        for (row_idx, row) in self.hessian.data.iter().enumerate() {
            for (col_idx, &entry) in row.indexed_source_iter() {
                if row_idx >= col_idx {
                    vals[idx] = entry;
                    idx += 1;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn penetrating_point() -> Result<(), Error> {
        let mass = 1.0;

        let params = FrictionParams {
            smoothing_weight: 0.0,
            dynamic_friction: 0.0, // ignored
            inner_iterations: 30,
            tolerance: 1e-15,
            print_level: 0,
        };

        let predictor_impulse = vec![[0.0, 0.1 * mass, 0.0]];
        let init_contact_impulse = vec![0.0];
        let mass_inv_mtx: DSBlockMatrix3 =
            DiagonalBlockMatrix::new(Chunked3::from_flat(vec![1.0 / mass; 3])).into();

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = ContactSolver::new(
            &predictor_impulse,
            &init_contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        )?;

        let soln = solver.step()?;

        dbg!(&soln);
        assert!(soln[0] <= 0.0);
        Ok(())
    }
}
