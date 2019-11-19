#![allow(dead_code)]
use super::FrictionParams;
use crate::contact::*;
use ipopt::{self, Index, Ipopt, Number};

use super::FrictionSolveResult;
use unroll::unroll_for_loops;
use utils::soap::*;
use utils::zip;

use crate::Error;

// Convenience function for computing scales.
fn variable_scales<'a>(
    contact_impulse: &'a [f64],
    mu: f64,
) -> impl Iterator<Item = f64> + Clone + 'a {
    contact_impulse.iter().map(move |&cr| {
        let radius = mu * cr.abs();
        if radius > 0.0 {
            1.0 / radius
        } else {
            1.0
        }
    })
}

/// Friction solver.
pub struct FrictionSolver<'a> {
    /// Non-linear solver.
    solver: Ipopt<SemiImplicitFrictionProblem<'a>>,
}

impl<'a> FrictionSolver<'a> {
    pub fn new(
        predictor_impulse: &'a [[f64; 3]],
        prev_friction_impulse_t: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
    ) -> Result<FrictionSolver<'a>, Error> {
        let basis_mtx = contact_basis.tangent_basis_matrix();
        let hessian = mass_inv_mtx
            .clone()
            .diagonal_congruence_transform(basis_mtx.view());
        //let hessian = (basis_mtx.view().transpose() * basis_mtx.view()).into();

        let min_contact_impulse = contact_impulse
            .iter()
            .map(|&x| x.abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .expect("No contact impulses detected when computing friction");

        let mu = params.dynamic_friction;

        // Scale predictor and prev_friction_impulse to ensure that the minimization variables are
        // well scaled at the minimum.
        let predictor_impulse: Chunked3<Vec<f64>> = variable_scales(contact_impulse, mu)
            .zip(predictor_impulse.iter())
            .map(|(s, &p)| (Tensor::new(p) * s).into_inner())
            .collect();

        let prev_friction_impulse_t: Chunked2<Vec<f64>> = variable_scales(contact_impulse, mu)
            .zip(prev_friction_impulse_t.iter())
            .map(|(s, &p)| (Tensor::new(p) * s).into_inner())
            .collect();

        let problem = SemiImplicitFrictionProblem(FrictionProblem {
            predictor_impulse,
            prev_friction_impulse_t,
            contact_impulse,
            contact_basis,
            mu,
            mass_inv_mtx,
            hessian,
            objective_scale: min_contact_impulse * min_contact_impulse,
        });

        let mut ipopt = Ipopt::new(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        ipopt.set_option("nlp_scaling_max_gradient", 1.0);
        //ipopt.set_option("nlp_scaling_method", "user-scaling");
        //ipopt.set_option("derivative_test", "second-order");
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("hessian_constant", "yes");
        ipopt.set_option("max_iter", params.inner_iterations as i32);

        Ok(FrictionSolver { solver: ipopt })
    }

    pub(crate) fn problem(&self) -> &SemiImplicitFrictionProblem<'a> {
        self.solver.solver_data().problem
    }

    /// Solve one step.
    pub fn step(&mut self) -> Result<FrictionSolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            objective_value,
            status,
            ..
        } = self.solver.solve();

        let (cr, mu) = (
            solver_data.problem.0.contact_impulse,
            solver_data.problem.0.mu,
        );
        let solution: Vec<[f64; 2]> = variable_scales(cr, mu)
            .zip(Chunked2::from_flat(solver_data.solution.primal_variables).iter())
            .map(|(s, &r)| (Tensor::new(r) / s).into_inner())
            .collect();

        let result = FrictionSolveResult {
            objective_value,
            solution,
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded
            | ipopt::SolveStatus::SolvedToAcceptableLevel
            | ipopt::SolveStatus::MaximumIterationsExceeded => Ok(result),
            e => Err(Error::FrictionSolveError { status: e }),
        }
    }
}

pub(crate) struct FrictionProblem<'a> {
    predictor_impulse: Chunked3<Vec<f64>>,
    /// Tangentaial components of the previous friction impulse.
    prev_friction_impulse_t: Chunked2<Vec<f64>>,
    /// A set of contact forces for each contact point.
    contact_impulse: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Friction coefficient.
    mu: f64,
    /// Vertex masses.
    mass_inv_mtx: EffectiveMassInvView<'a>,
    /// Workspace Hessian matrix.
    hessian: DSBlockMatrix2,
    /// Scale the objective.
    objective_scale: f64,
}

impl<'a> FrictionProblem<'a> {
    pub fn num_contacts(&self) -> usize {
        self.predictor_impulse.len()
    }

    pub fn num_variables(&self) -> usize {
        2 * self.num_contacts()
    }

    pub fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        let bound = 2e19;
        x_l.iter_mut().for_each(|x| *x = -bound);
        x_u.iter_mut().for_each(|x| *x = bound);
        true
    }

    pub fn initial_point(&self, r: &mut [Number]) -> bool {
        for (i, (r, &p, &cr)) in zip!(
            Chunked2::from_flat(r).iter_mut(),
            self.predictor_impulse.iter(),
            self.contact_impulse.iter(),
        )
        .enumerate()
        {
            let p_norm = Tensor::new(p).norm();
            let pred = self.contact_basis.to_contact_coordinates(p, i);
            let radius = cr.abs() * self.mu;
            if radius > 0.0 {
                let constraint_radius = 1.0;
                if p_norm > constraint_radius {
                    *r = (Tensor::new([pred[1], pred[2]]) * (constraint_radius / p_norm))
                        .into_inner();
                } else {
                    *r = (Tensor::new([pred[1], pred[2]])).into_inner();
                }
            } else {
                *r = [0.0; 2];
            }
        }
        true
    }
}

pub(crate) struct SemiImplicitFrictionProblem<'a>(FrictionProblem<'a>);

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for SemiImplicitFrictionProblem<'_> {
    fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        self.0.bounds(x_l, x_u)
    }

    fn initial_point(&self, r: &mut [Number]) -> bool {
        self.0.initial_point(r)
    }

    fn objective(&self, r_t: &[Number], obj: &mut Number) -> bool {
        let r_t = Chunked2::from_flat(r_t);
        let FrictionProblem {
            predictor_impulse,
            prev_friction_impulse_t: prev_r_t,
            contact_basis,
            mass_inv_mtx,
            ..
        } = &self.0;

        assert_eq!(predictor_impulse.len(), r_t.len());

        // Compute (negative of) frictional dissipation.

        // Compute the difference between current and previous impulses in tangent space.
        let diff_t: Chunked2<Vec<_>> = (r_t.expr() - prev_r_t.expr()).eval();

        // Convert to physical space.
        let mut diff: Chunked3<Vec<f64>> = contact_basis
            .from_tangent_space(diff_t.view().into())
            .collect();

        *&mut diff.expr_mut() -= predictor_impulse.expr();

        let rhs = mass_inv_mtx.view() * *diff.view().as_tensor();
        //let rhs = diff.view();

        *obj = 0.5 * diff.expr().dot::<f64, _>(rhs.expr()) * self.0.objective_scale;

        true
    }

    fn objective_grad(&self, r_t: &[Number], grad_f_t: &mut [Number]) -> bool {
        let r_t = Chunked2::from_flat(r_t);
        let FrictionProblem {
            predictor_impulse,
            prev_friction_impulse_t: prev_r_t,
            contact_basis,
            mass_inv_mtx,
            ..
        } = &self.0;

        assert_eq!(predictor_impulse.len(), r_t.len());

        // Compute derivative of (negative of) frictional dissipation.

        // Compute the difference between current and previous impulses in tangent space.
        //let diff_t: Chunked2<Vec<_>> = (r_t.expr() - prev_r_t.expr()).eval();

        //let diff: Chunked3<Vec<f64>> = contact_basis.from_tangent_space(diff_t.view().into()).collect();
        //let mut diff = Tensor::new(diff);

        //diff -= Tensor::new(predictor_impulse.view());

        let diff_t: Chunked2<Vec<_>> = (r_t.expr() - prev_r_t.expr()).eval();

        // Convert to physical space.
        let mut diff: Chunked3<Vec<f64>> = contact_basis
            .from_tangent_space(diff_t.view().into())
            .collect();

        *&mut diff.expr_mut() -= predictor_impulse.expr();

        let grad = mass_inv_mtx.view() * *diff.view().as_tensor();
        //let grad = diff.view();

        let grad_t: Vec<_> = contact_basis
            .to_tangent_space(grad.view().into_inner().into())
            .collect();

        let mut grad_f_t = Chunked2::from_flat(grad_f_t);
        for (g_out, &g) in grad_f_t.iter_mut().zip(grad_t.iter()) {
            //*g_out = g;
            *g_out = (Tensor::new(g) * self.0.objective_scale).into_inner();
            //dbg!(f64::max(g_out[0].abs(), g_out[1].abs()));
        }

        true
    }
    //fn variable_scaling(&self, r_scaling: &mut [Number]) -> bool {
    //    let mut r_scaling = Chunked2::from_flat(r_scaling);
    //    for (out, s) in r_scaling
    //        .iter_mut()
    //        .zip(variable_scales(self.0.contact_impulse, self.0.mu))
    //    {
    //        out[0] = s;
    //        out[1] = s;
    //    }
    //    false
    //}
}

impl ipopt::ConstrainedProblem for SemiImplicitFrictionProblem<'_> {
    fn num_constraints(&self) -> usize {
        self.0.contact_impulse.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2 * self.num_constraints()
    }

    fn constraint(&self, r: &[Number], g: &mut [Number]) -> bool {
        let r = Chunked2::from_flat(r);
        assert_eq!(r.len(), g.len());
        for (c, &r) in zip!(g.iter_mut(), r.iter()) {
            *c = r.as_tensor().norm_squared();
            //dbg!(*c);
        }
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        for ((l, u), &cr) in g_l
            .iter_mut()
            .zip(g_u.iter_mut())
            .zip(self.0.contact_impulse.iter())
        {
            *l = -2e19; // inner product can never be negative, so leave this unconstrained.
            let radius = cr.abs() * self.0.mu;
            if radius > 0.0 {
                *u = 1.0;
            } else {
                // Turn into an equality constraint.
                *l = 0.0;
                *u = 0.0;
            }
        }
        true
    }

    #[unroll_for_loops]
    fn constraint_jacobian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        for constraint_idx in 0..self.num_constraints() {
            for j in 0..2 {
                rows[2 * constraint_idx + j] = constraint_idx as Index;
                cols[2 * constraint_idx + j] = (2 * constraint_idx + j) as Index;
            }
        }
        true
    }

    fn constraint_jacobian_values(&self, r: &[Number], vals: &mut [Number]) -> bool {
        let mut jacobian = Chunked2::from_flat(vals);
        let r = Chunked2::from_flat(r);
        for (jac, &r) in zip!(jacobian.iter_mut(), r.iter()) {
            *jac.as_mut_tensor() = Tensor::new(r) * 2.0;
            //dbg!(f64::max(jac[0].abs(), jac[1].abs()));
        }
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        // Constraint Hessian is diagonal.
        let mut idx = 0;
        for (row_idx, row) in self.0.hessian.data.iter().enumerate() {
            for col_idx in row.index_iter() {
                if row_idx > col_idx {
                    idx += 4;
                } else if row_idx == col_idx {
                    idx += 3;
                }
            }
        }
        2 * self.num_constraints() + idx
    }

    #[unroll_for_loops]
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // Diagonal objective Hessian.
        let offset = {
            let mut idx = 0;
            for (row_idx, row) in self.0.hessian.data.iter().enumerate() {
                for col_idx in row.index_iter() {
                    if row_idx < col_idx {
                        continue;
                    }
                    rows[idx] = (2 * row_idx) as Index;
                    cols[idx] = (2 * col_idx) as Index;
                    idx += 1;

                    if row_idx > col_idx {
                        rows[idx] = (2 * row_idx) as Index;
                        cols[idx] = (2 * col_idx + 1) as Index;
                        idx += 1;
                    }

                    rows[idx] = (2 * row_idx + 1) as Index;
                    cols[idx] = (2 * col_idx) as Index;
                    idx += 1;

                    rows[idx] = (2 * row_idx + 1) as Index;
                    cols[idx] = (2 * col_idx + 1) as Index;
                    idx += 1;
                }
            }
            idx
        };
        // Diagonal Constraint matrix.
        let num_diagonal_entries = 2 * self.num_constraints();
        for idx in 0..num_diagonal_entries {
            rows[offset + idx] = idx as Index;
            cols[offset + idx] = idx as Index;
        }
        true
    }
    fn hessian_values(
        &self,
        _r: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let mut idx = 0;
        for (row_idx, row) in self.0.hessian.data.iter().enumerate() {
            for (col_idx, block) in row.indexed_source_iter() {
                if row_idx < col_idx {
                    continue;
                }

                let hess_block =
                    (*block.into_arrays().as_tensor() * obj_factor * self.0.objective_scale)
                        .into_inner();

                vals[idx] = hess_block[0][0];
                idx += 1;

                if row_idx > col_idx {
                    vals[idx] = hess_block[0][1];
                    idx += 1;
                }

                vals[idx] = hess_block[1][0];
                idx += 1;

                vals[idx] = hess_block[1][1];
                idx += 1;
            }
        }

        let mut hess = Chunked2::from_flat(&mut vals[idx..]);
        for (&l, h) in zip!(lambda.iter(), hess.iter_mut()) {
            *h.as_mut_tensor() = Tensor::new([2.0; 2]) * l;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use utils::soap::Vector2;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(0.000001, mass)?;

        // Check that the point still has velocity in the positive x direction
        assert!(velocity[1] > 0.8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[0], 0.0, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn sticking_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(1.5, mass)?;

        // Check that the point gets stuck
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6, epsilon = 1e-5);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[0], 0.0, max_relative = 1e-6);
        Ok(())
    }

    fn sliding_point_tester(mu: f64, mass: f64) -> Result<(Vector2<f64>, Vector2<f64>), Error> {
        let params = FrictionParams {
            dynamic_friction: mu,
            inner_iterations: 100,
            tolerance: 1e-15,
            print_level: 0,
        };

        let prev_friction_impulse_t = vec![[0.0, 0.0]];
        let predictor_impulse = vec![[-1.0 * mass, 0.0, 0.0]]; // one point sliding right.
        let contact_impulse = vec![10.0 * mass];
        let mass_inv_mtx: DSBlockMatrix3 =
            DiagonalBlockMatrix::new(Chunked3::from_flat(vec![1.0 / mass; 3])).into();

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = FrictionSolver::new(
            &predictor_impulse,
            &prev_friction_impulse_t,
            &contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        )?;
        let result = solver.step()?;
        let FrictionSolveResult { solution, .. } = result;

        let impulse = Vector2::new(solution[0]);
        let new_vel = impulse / mass;
        let p_imp_t: Vec<_> = contact_basis.to_tangent_space(&predictor_impulse).collect();
        let prev_vel = -Vector2::new(p_imp_t[0]) / mass;
        // Add the previous velocity to the new impulse.
        let final_velocity = prev_vel + new_vel;

        dbg!(&p_imp_t);
        dbg!(&impulse);
        dbg!(&final_velocity);

        Ok((final_velocity, impulse))
    }

    /// A tetrahedron sliding on a slanted surface.
    #[test]
    fn sliding_tet() -> Result<(), Error> {
        let params = FrictionParams {
            dynamic_friction: 0.001,
            inner_iterations: 40,
            tolerance: 1e-7,
            print_level: 0,
        };

        let masses = vec![0.0003720701030949866, 0.0003720701030949866];
        let mass_inv_mtx: DSBlockMatrix3 =
            DiagonalBlockMatrix::new(Chunked3::from_array_vec(vec![
                [1.0 / masses[0]; 3],
                [1.0 / masses[1]; 3],
            ]))
            .into();

        let prev_friction_impulse_t = vec![[0.0, 0.0], [0.0, 0.0]];
        let predictor_impulse = vec![
            [
                0.0,
                -0.07225747944670913 * masses[0],
                -0.0000001280108566301736 * masses[0],
            ],
            [
                0.0,
                -0.06185827187696774 * masses[1],
                0.0060040275393186595 * masses[1],
            ],
        ]; // tet vertex velocities
        let contact_impulse = vec![-0.0000000018048827573828247, -0.00003259055555607145];

        let mut contact_basis = ContactBasis::new();
        let normals = vec![
            [-0.0, -0.7071067811865476, -0.7071067811865476],
            [-0.0, -0.7071067811865476, -0.7071067811865476],
        ];
        contact_basis.update_from_normals(normals);

        let mut solver = FrictionSolver::new(
            &predictor_impulse,
            &prev_friction_impulse_t,
            &contact_impulse,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        )?;
        let result = solver.step()?;
        let FrictionSolveResult { solution, .. } = result;

        let p_imp_t: Vec<_> = contact_basis.to_tangent_space(&predictor_impulse).collect();
        let final_velocity: Vec<_> = zip!(p_imp_t.iter(), solution.iter(), masses.iter())
            .map(|(&pr, &r, &m)| (-Vector2::new(pr) + Vector2::new(r)) / m)
            .collect();

        dbg!(&solution);
        dbg!(&final_velocity);

        Ok(())
    }
}
