use super::FrictionParams;
use geo::math::{Matrix3, Vector2};
use ipopt::{self, Index, Ipopt, Number};
use reinterpret::*;
use crate::contact::*;

use unroll::unroll_for_loops;
use utils::zip;

use crate::Error;

/// Result from one inner friction step.
#[derive(Clone, Debug, PartialEq)]
pub struct FrictionSolveResult {
    /// The value of the dissipation objective at the end of the step.
    pub objective_value: f64,
    /// Resultant friction force in contact space.
    pub solution: Vec<[f64; 2]>,
}

/// Friction solver.
pub struct FrictionSolver<'a, CJI> {
    /// Non-linear solver.
    solver: Ipopt<SemiImplicitFrictionProblem<'a, CJI>>,
}

impl<'a> FrictionSolver<'a, std::iter::Empty<(usize, usize)>> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_force` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn without_contact_jacobian(
        velocity: &'a [[f64; 2]],
        contact_force: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
    ) -> Result<FrictionSolver<'a, std::iter::Empty<(usize, usize)>>, Error> {
        Self::new_impl(velocity, contact_force, contact_basis, masses, params, None)
    }
}

impl<'a, CJI: Iterator<Item=(usize, usize)>> FrictionSolver<'a, CJI> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_force` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [[f64; 2]],
        contact_force: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: (&'a [Matrix3<f64>], CJI),
    ) -> Result<FrictionSolver<'a, CJI>, Error> {
        Self::new_impl(velocity, contact_force, contact_basis, masses, params, Some(contact_jacobian))
    }

    fn new_impl(
        velocity: &'a [[f64; 2]],
        contact_force: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
    ) -> Result<FrictionSolver<'a, CJI>, Error> {
        let problem = SemiImplicitFrictionProblem(FrictionProblem {
            velocity: reinterpret_slice(velocity),
            contact_force,
            contact_basis,
            mu: params.dynamic_friction,
            contact_jacobian,
            masses,
        });

        let mut ipopt = Ipopt::new(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        ipopt.set_option("nlp_scaling_max_gradient", 1e-3);
        //ipopt.set_option("derivative_test", "second-order");
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("max_iter", params.inner_iterations as i32);

        Ok(FrictionSolver { solver: ipopt })
    }

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn step(&mut self) -> Result<FrictionSolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            objective_value,
            status,
            ..
        } = self.solver.solve();

        let result = FrictionSolveResult {
            objective_value,
            solution: reinterpret_vec(solver_data.solution.primal_variables.to_vec()),
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::FrictionSolveError(e)),
        }
    }
}

pub(crate) struct FrictionProblem<'a, CJI> {
    /// A set of tangential velocities in contact space for active contacts. These are used to
    /// determine the applied frictional force.
    velocity: &'a [Vector2<f64>],
    /// A set of contact forces for each contact point.
    contact_force: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Friction coefficient.
    mu: f64,
    /// Contact Jacobian is a sparse matrix that maps vectors from vertices to contact points.
    /// If the `None` is specified, it is assumed that the contact Jacobian is the identity matrix,
    /// meaning that contacts occur at vertex positions.
    contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
    /// Vertex masses.
    masses: &'a [f64],
}

impl<CJI> FrictionProblem<'_, CJI> {
    pub fn num_contacts(&self) -> usize {
        self.velocity.len()
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

    pub fn initial_point(&self, f: &mut [Number]) -> bool {
        let forces: &mut [Vector2<f64>] = reinterpret_mut_slice(f);
        for (f, &cf, &v) in zip!(
            forces.iter_mut(),
            self.contact_force.iter(),
            self.velocity.iter()
        ) {
            let v_norm = v.norm();
            if v_norm > 0.0 {
                *f = v * ((-self.mu * cf.abs()) / v_norm)
            } else {
                *f = Vector2::zeros();
            }
        }
        true
    }
}

pub(crate) struct ExplicitFrictionProblem<'a, CJI>(FrictionProblem<'a, CJI>);

/// Prepare the problem for Newton iterations.
impl<CJI> ipopt::BasicProblem for ExplicitFrictionProblem<'_, CJI> {
    fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        self.0.bounds(x_l, x_u)
    }

    fn initial_point(&self, f: &mut [Number]) -> bool {
        self.0.initial_point(f)
    }

    fn objective(&self, f: &[Number], obj: &mut Number) -> bool {
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(self.0.velocity.len(), forces.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &f) in self.0.velocity.iter().zip(forces.iter()) {
            *obj += v.dot(f)
        }

        true
    }

    fn objective_grad(&self, _f: &[Number], grad_f: &mut [Number]) -> bool {
        let velocity_values: &[f64] = reinterpret_slice(self.0.velocity);
        assert_eq!(velocity_values.len(), grad_f.len());

        for g in grad_f.iter_mut() {
            *g = 0.0;
        }

        for (g, v) in grad_f.iter_mut().zip(velocity_values) {
            *g += v;
        }

        true
    }
}

impl<CJI> ipopt::ConstrainedProblem for ExplicitFrictionProblem<'_, CJI> {
    fn num_constraints(&self) -> usize {
        self.0.contact_force.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2 * self.num_constraints()
    }

    fn constraint(&self, f: &[Number], g: &mut [Number]) -> bool {
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(forces.len(), g.len());
        for (c, f) in g.iter_mut().zip(forces.iter()) {
            *c = f.dot(*f);
        }
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        for ((l, u), &cf) in g_l
            .iter_mut()
            .zip(g_u.iter_mut())
            .zip(self.0.contact_force.iter())
        {
            *l = -2e19; // inner product can never be negative, so leave this unconstrained.
            *u = self.0.mu * cf.abs();
            *u *= *u; // square the radius
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

    fn constraint_jacobian_values(&self, f: &[Number], vals: &mut [Number]) -> bool {
        let jacobian: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        for (jac, &f) in jacobian.iter_mut().zip(forces.iter()) {
            *jac = f * 2.0;
        }
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        // Objective Hessian is zero.
        // Constraint hessian is diagonal.
        2 * self.num_constraints()
    }

    #[unroll_for_loops]
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // Diagonal matrix
        for idx in 0..self.num_hessian_non_zeros() {
            rows[idx] = idx as Index;
            cols[idx] = idx as Index;
        }
        true
    }
    fn hessian_values(
        &self,
        _f: &[Number],
        _obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let hess_vals: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        assert_eq!(hess_vals.len(), lambda.len());
        for (h, &l) in hess_vals.iter_mut().zip(lambda.iter()) {
            *h = Vector2([2.0, 2.0]) * l;
        }
        true
    }
}

pub(crate) struct SemiImplicitFrictionProblem<'a, CJI>(FrictionProblem<'a, CJI>);

/// Prepare the problem for Newton iterations.
impl<CJI> ipopt::BasicProblem for SemiImplicitFrictionProblem<'_, CJI> {
    fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        self.0.bounds(x_l, x_u)
    }

    fn initial_point(&self, f: &mut [Number]) -> bool {
        self.0.initial_point(f)
    }

    fn objective(&self, f: &[Number], obj: &mut Number) -> bool {
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(self.0.velocity.len(), forces.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &f) in self.0.velocity.iter().zip(forces.iter()) {
            *obj += v.dot(f);
        }

        if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
            // A contact jacobian is provided. This makes computing the non-linear contribution a
            // tad more expensive.
            // TODO: implement this
        } else {
            // No constraint Jacobian is provided, the non-linear part is a standard inner product
            // scaled by the mass
            for (m, &f) in self.0.masses.iter().zip(forces.iter()) {
                *obj += f.dot(f) * ( 0.5 / m);
            }
        }

        true
    }

    fn objective_grad(&self, f: &[Number], grad_f: &mut [Number]) -> bool {
        let velocities: &[Vector2<f64>] = reinterpret_slice(self.0.velocity);
        let gradient: &mut [Vector2<f64>] = reinterpret_mut_slice(grad_f);
        assert_eq!(velocities.len(), gradient.len());

        for g in gradient.iter_mut() {
            *g = Vector2::zeros();
        }

        for (g, &v) in gradient.iter_mut().zip(velocities.iter()) {
            *g += v;
        }

        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
            // A contact jacobian is provided. This makes computing the non-linear contribution a
            // tad more expensive.
            // TODO: implement this
        } else {
            for (g, &m, &f) in zip!(gradient.iter_mut(), self.0.masses.iter(), forces.iter()) {
                *g += f * (1.0 / m);
            }
        }

        true
    }
}

impl<CJI> ipopt::ConstrainedProblem for SemiImplicitFrictionProblem<'_, CJI> {
    fn num_constraints(&self) -> usize {
        self.0.contact_force.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2 * self.num_constraints()
    }

    fn constraint(&self, f: &[Number], g: &mut [Number]) -> bool {
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        assert_eq!(forces.len(), g.len());
        for (c, f) in g.iter_mut().zip(forces.iter()) {
            *c = f.dot(*f);
        }
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        for ((l, u), &cf) in g_l
            .iter_mut()
            .zip(g_u.iter_mut())
            .zip(self.0.contact_force.iter())
        {
            *l = -2e19; // inner product can never be negative, so leave this unconstrained.
            *u = self.0.mu * cf.abs();
            *u *= *u; // square the radius
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

    fn constraint_jacobian_values(&self, f: &[Number], vals: &mut [Number]) -> bool {
        let jacobian: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        let forces: &[Vector2<f64>] = reinterpret_slice(f);
        for (jac, &f) in jacobian.iter_mut().zip(forces.iter()) {
            *jac = f * 2.0;
        }
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        // Objective Hessian is diagonal.
        // Constraint Hessian is diagonal.
        2 * self.num_constraints()
            + if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
                0 // TODO: Implement this
            } else {
                2 * self.0.masses.len()
            }
    }

    #[unroll_for_loops]
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // Diagonal objective Hessian.
        let num_diagonal_entries = 2*self.num_constraints();
        let offset = if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
            0
        } else {
            for idx in 0..num_diagonal_entries {
                rows[idx] = idx as Index;
                cols[idx] = idx as Index;
            }
            2 * self.num_constraints()
        };
        // Diagonal Constraint matrix.
        for idx in 0..num_diagonal_entries {
            rows[offset + idx] = idx as Index;
            cols[offset + idx] = idx as Index;
        }
        true
    }
    fn hessian_values(
        &self,
        _f: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let hess_vals: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        let num_hess_vals = hess_vals.len();
        let mut hess_iter_mut = hess_vals.iter_mut();
        if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
            assert_eq!(num_hess_vals, lambda.len());
            // A contact jacobian is provided. This makes computing the non-linear contribution a
            // tad more expensive.
            // TODO: implement this
        } else {
            assert_eq!(num_hess_vals, self.0.masses.len() + lambda.len());
            for (&m, h) in self.0.masses.iter().zip(&mut hess_iter_mut) {
                *h = Vector2::ones() * (1.0 * obj_factor / m);
            }
        }
        for (&l, h) in lambda.iter().zip(&mut hess_iter_mut) {
            *h = Vector2([2.0, 2.0]) * l;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let params = FrictionParams {
            dynamic_friction: 0.1,
            inner_iterations: 30,
            tolerance: 1e-5,
            print_level: 5,
            density: 1000.0,
        };
        let mass = 0.1;

        let velocity = vec![[1.0, 0.0]]; // one point sliding right.
        let contact_force = vec![10.0 * mass];
        let masses = vec![mass; 1];

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = FrictionSolver::without_contact_jacobian(&velocity, &contact_force, &contact_basis, &masses, params)?;
        let result = solver.step()?;
        let FrictionSolveResult {
            solution,
            objective_value,
        } = result;

        dbg!(Vector2(velocity[0]) + Vector2(solution[0]) / mass);
        assert_relative_eq!(solution[0][0], -15.0, max_relative = 1e-6);
        assert_relative_eq!(solution[0][1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(objective_value, -15.0, max_relative = 1e-6);


        Ok(())
    }
}
