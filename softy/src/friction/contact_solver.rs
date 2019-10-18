use super::FrictionParams;
use ipopt::{self, Index, Ipopt, Number};
use crate::contact::*;
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

        // TODO: Remove this test

        {
            use approx::*;
            let r: Chunked3<Vec<f64>> = contact_basis.from_normal_space(contact_impulse_n).collect();
            let test_rhs = mass_inv_mtx.view() * *r.view().as_tensor();
            let test_rhs_n: Vec<f64> = contact_basis.to_normal_space(test_rhs.view().into_inner().into()).collect();

            let rhs_n = hessian.view() * Tensor::new(contact_impulse_n);

            assert!(rhs_n.into_inner().iter().zip(test_rhs_n.iter()).all(|(&a, &b)| relative_eq!(a, b)));
        }

        // End of test

        let problem = ContactProblem {
            predictor_impulse,
            init_contact_impulse_n: contact_impulse_n,
            contact_basis,
            mass_inv_mtx,
            hessian,
        };


        let mut ipopt = Ipopt::new_newton(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        //ipopt.set_option("nlp_scaling_max_gradient", 1.0);
        //ipopt.set_option("nlp_scaling_method", "user-scaling");
        ipopt.set_option("derivative_test", "second-order");
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
    predictor_impulse: Chunked3<&'a [f64]>,
    init_contact_impulse_n: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Vertex masses.
    mass_inv_mtx: EffectiveMassInvView<'a>,
    /// Workspace Hessian matrix.
    hessian: DSMatrix,
}

impl<'a> ContactProblem<'a> {
    pub fn num_contacts(&self) -> usize {
        self.predictor_impulse.len()
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
            predictor_impulse,
            contact_basis,
            mass_inv_mtx,
            hessian,
            ..
        } = self;

        assert_eq!(predictor_impulse.len(), r_n.len());

        // Convert to physical space.
        //let mut diff: Chunked3<Vec<f64>> = contact_basis.from_normal_space(r_n.view().into()).collect();
        //*diff.as_mut_tensor() -= *predictor_impulse.view().as_tensor();
        //let rhs = mass_inv_mtx.view() * *diff.view().as_tensor();
        //let rhs: Vec<f64> = contact_basis.to_normal_space(rhs.view().into_inner().into()).collect();
        let mut rhs = hessian.view() * (Tensor::new(r_n) * 0.5).view();
        let pred = mass_inv_mtx.view() * Tensor::new(predictor_impulse.view());
        let pred_n: Vec<f64> = contact_basis.to_normal_space(pred.view().into_inner().into()).collect();
        rhs -= pred_n.view().as_tensor();

        *obj = r_n.expr().dot(rhs.expr());

        true
    }

    fn objective_grad(&self, r_n: &[Number], grad_f_n: &mut [Number]) -> bool {
        let ContactProblem {
            predictor_impulse,
            contact_basis,
            mass_inv_mtx,
            hessian,
            ..
        } = self;

        //let diff: Chunked3<Vec<f64>> = contact_basis.from_normal_space(r_n.view().into()).collect();
        //let mut diff = Tensor::new(diff);
        //diff -= Tensor::new(predictor_impulse.view());
        //let grad = mass_inv_mtx.view() * diff.view();
        ////let grad = diff.view();
        //let grad_n = contact_basis.to_normal_space(grad.view().into_inner().into());

        let mut rhs = hessian.view() * Tensor::new(r_n);
        let pred = mass_inv_mtx.view() * Tensor::new(predictor_impulse.view());
        let pred_n: Vec<f64> = contact_basis.to_normal_space(pred.view().into_inner().into()).collect();
        rhs -= pred_n.view().as_tensor();

        assert_eq!(grad_f_n.len(), rhs.len());
        grad_f_n.copy_from_slice(rhs.view().into_inner());

        true
    }
}

impl ipopt::NewtonProblem for ContactProblem<'_> {
    fn num_hessian_non_zeros(&self) -> usize {
        self.hessian.num_non_zeros()
    }

    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // Diagonal objective Hessian.
        let mut idx = 0;
        for (row_idx, row) in self.hessian.data.iter().enumerate() {
            for col_idx in row.index_iter() {
                rows[idx] = row_idx as Index;
                cols[idx] = col_idx as Index;
                idx += 1;
            }
        }
        true
    }
    fn hessian_values(
        &self,
        _r: &[Number],
        vals: &mut [Number],
    ) -> bool {
        assert_eq!(self.hessian.num_non_zeros(), vals.len());
        vals.copy_from_slice(self.hessian.data.storage());
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
