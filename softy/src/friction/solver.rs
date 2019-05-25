#![allow(dead_code)]
use super::FrictionParams;
use geo::math::{Matrix3, Vector2};
use ipopt::{self, Index, Ipopt, Number};
use reinterpret::*;
use crate::contact::*;

use unroll::unroll_for_loops;
use utils::zip;
use super::FrictionSolveResult;

use crate::Error;

/// Friction solver.
pub struct FrictionSolver<'a, CJI> {
    /// Non-linear solver.
    solver: Ipopt<SemiImplicitFrictionProblem<'a, CJI>>,
}

impl<'a> FrictionSolver<'a, std::iter::Empty<(usize, usize)>> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_impulse` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn without_contact_jacobian(
        velocity: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
    ) -> Result<FrictionSolver<'a, std::iter::Empty<(usize, usize)>>, Error> {
        Self::new_impl(velocity, contact_impulse, contact_basis, masses, params, None)
    }
}

impl<'a, CJI: Iterator<Item=(usize, usize)>> FrictionSolver<'a, CJI> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_impulse` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: (&'a [Matrix3<f64>], CJI),
    ) -> Result<FrictionSolver<'a, CJI>, Error> {
        Self::new_impl(velocity, contact_impulse, contact_basis, masses, params, Some(contact_jacobian))
    }

    fn new_impl(
        velocity: &'a [[f64; 2]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: Option<(&'a [Matrix3<f64>], CJI)>,
    ) -> Result<FrictionSolver<'a, CJI>, Error> {
        let scale = 100.0;
            //zip!(contact_impulse.iter(), self.0.masses.iter())
            //.map(move |(&cr, &m)| m / (cr.abs()))
            //.max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            //.unwrap_or(1.0);

        let problem = SemiImplicitFrictionProblem(FrictionProblem {
            velocity: reinterpret_slice(velocity),
            contact_impulse,
            contact_basis,
            mu: params.dynamic_friction,
            contact_jacobian,
            masses,
            scale,
        });

        let mut ipopt = Ipopt::new(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        //ipopt.set_option("nlp_scaling_method", "user-scaling");
        //ipopt.set_option("nlp_scaling_max_gradient", 1e-3);
        //ipopt.set_option("derivative_test", "second-order");
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("max_iter", params.inner_iterations as i32);

        Ok(FrictionSolver { solver: ipopt })
    }

    pub(crate) fn scale(&self) -> f64 {
        self.problem().0.scale
    }

    pub(crate) fn problem(&self) -> &SemiImplicitFrictionProblem<'a, CJI> {
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
    contact_impulse: &'a [f64],
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
    pub scale: f64,
}

impl<'a, CJI> FrictionProblem<'a, CJI> {
    pub fn variable_scales(&'a self) -> impl Iterator<Item=f64> + Clone + 'a {
        self.contact_impulse.iter().map(move |&cr| 1.0 / (self.mu * cr.abs()))
    }

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

    pub fn initial_point(&self, r: &mut [Number]) -> bool {
        let impulses: &mut [Vector2<f64>] = reinterpret_mut_slice(r);
        for (r, &v, &cr) in zip!(
            impulses.iter_mut(),
            self.velocity.iter(),
            self.contact_impulse.iter()
        ) {
            let v_norm = v.norm();
            if v_norm > 0.0 {
                *r = v * (-self.mu * cr.abs() * self.scale / v_norm)
            } else {
                *r = Vector2::zeros();
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

    fn objective(&self, r: &[Number], obj: &mut Number) -> bool {
        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        assert_eq!(self.0.velocity.len(), impulses.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &r) in zip!(self.0.velocity.iter(), impulses.iter()) {
            *obj += v.dot(r) * self.0.scale;
        }

        *obj *= self.0.scale;

        true
    }

    fn objective_grad(&self, _r: &[Number], grad_f: &mut [Number]) -> bool {
        let velocity_values: &[f64] = reinterpret_slice(self.0.velocity);
        assert_eq!(velocity_values.len(), grad_f.len());

        for g in grad_f.iter_mut() {
            *g = 0.0;
        }

        for (g, v) in zip!(grad_f.iter_mut(), velocity_values) {
            *g += v;
        }

        true
    }

    //fn variable_scaling(&self, r_scaling: &mut [Number]) -> bool {
    //    for (out, s) in r_scaling.iter_mut().zip(self.0.variable_scales()) {
    //        *out = s;
    //    }
    //    true
    //}
}

impl<CJI> ipopt::ConstrainedProblem for ExplicitFrictionProblem<'_, CJI> {
    fn num_constraints(&self) -> usize {
        self.0.contact_impulse.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2 * self.num_constraints()
    }

    fn constraint(&self, r: &[Number], g: &mut [Number]) -> bool {
        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        assert_eq!(impulses.len(), g.len());
        for (c, r) in zip!(g.iter_mut(), impulses.iter()) {
            *c = r.dot(*r) * self.0.scale * self.0.scale;
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
            *u = self.0.mu * cr.abs();
            *u *= *u;
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
        let jacobian: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        for (jac, &r) in zip!(jacobian.iter_mut(), impulses.iter()) {
            *jac = r * 2.0 * self.0.scale;
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
        _r: &[Number],
        _obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let hess_vals: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        assert_eq!(hess_vals.len(), lambda.len());
        for (h, &l) in zip!(hess_vals.iter_mut(), lambda.iter()) {
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

    fn initial_point(&self, r: &mut [Number]) -> bool {
        self.0.initial_point(r)
    }

    fn objective(&self, r: &[Number], obj: &mut Number) -> bool {
        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        assert_eq!(self.0.velocity.len(), impulses.len());

        // Clear objective value.
        *obj = 0.0;

        // Compute (negative of) frictional dissipation.
        for (&v, &r) in zip!(self.0.velocity.iter(), impulses.iter()) {
            *obj += v.dot(r) * self.0.scale;
        }

        if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
            // A contact jacobian is provided. This makes computing the non-linear contribution a
            // tad more expensive.
            // TODO: implement this
        } else {
            // No constraint Jacobian is provided, the non-linear part is a standard inner product
            // scaled by the mass
            for (m, &r) in zip!(self.0.masses.iter(), impulses.iter()) {
                *obj += r.dot(r) * (0.5 * self.0.scale * self.0.scale / m);
            }
        }

        true
    }

    fn objective_grad(&self, r: &[Number], grad_f: &mut [Number]) -> bool {
        let velocities: &[Vector2<f64>] = reinterpret_slice(self.0.velocity);
        let gradient: &mut [Vector2<f64>] = reinterpret_mut_slice(grad_f);
        assert_eq!(velocities.len(), gradient.len());

        for g in gradient.iter_mut() {
            *g = Vector2::zeros();
        }

        for (g, &v) in zip!(gradient.iter_mut(), velocities.iter()) {
            *g += v;
        }

        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        if let Some(ref _contact_jacobian) = self.0.contact_jacobian {
            // A contact jacobian is provided. This makes computing the non-linear contribution a
            // tad more expensive.
            // TODO: implement this
        } else {
            for (g, &m, &r) in zip!(gradient.iter_mut(), self.0.masses.iter(), impulses.iter()) {
                *g += r * (self.0.scale * 1.0 / m);
            }
        }

        for g in gradient.iter_mut() {
            *g *= self.0.scale;
        }

        true
    }

    //fn objective_scaling(&self) -> f64 {
    //}

    //fn variable_scaling(&self, r_scaling: &mut [Number]) -> bool {
    //    let max_cr = 
    //        zip!(self.0.contact_impulse.iter(), self.0.masses.iter())
    //        .map(move |(&cr, &m)| m / (cr.abs()))
    //        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
    //        .unwrap_or(1.0);
    //    for s in r_scaling.iter_mut() {
    //        *s = max_cr;
    //    }
    //    true
    //}
}

impl<CJI> ipopt::ConstrainedProblem for SemiImplicitFrictionProblem<'_, CJI> {
    fn num_constraints(&self) -> usize {
        self.0.contact_impulse.len()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2 * self.num_constraints()
    }

    fn constraint(&self, r: &[Number], g: &mut [Number]) -> bool {
        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        assert_eq!(impulses.len(), g.len());
        for (c, &r) in zip!(g.iter_mut(), impulses.iter()) {
            *c = r.dot(r) * self.0.scale * self.0.scale;
            dbg!(*c);
        }
        true
    }

    //fn constraint_scaling(&self, g_scaling: &mut [Number]) -> bool {
    //    for (gs, &cr) in zip!(g_scaling.iter_mut(), self.0.contact_impulse.iter()) {
    //        *gs = 1.0/cr;
    //    }
    //    true
    //}

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        for ((l, u), &cr) in g_l
            .iter_mut()
            .zip(g_u.iter_mut())
            .zip(self.0.contact_impulse.iter())
        {
            *l = -2e19; // inner product can never be negative, so leave this unconstrained.
            *u = self.0.mu * cr.abs();
            *u *= *u;
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
        let jacobian: &mut [Vector2<f64>] = reinterpret_mut_slice(vals);
        let impulses: &[Vector2<f64>] = reinterpret_slice(r);
        for (jac, &r) in zip!(jacobian.iter_mut(), impulses.iter()) {
            *jac = r * 2.0 * self.0.scale * self.0.scale;
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
        _r: &[Number],
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
            for (&m, h) in zip!(self.0.masses.iter(), &mut hess_iter_mut) {
                *h = Vector2::ones() * (1.0 * obj_factor / m);
            }
        }
        for (&l, h) in zip!(lambda.iter(), &mut hess_iter_mut) {
            *h = Vector2([2.0, 2.0]) * l;
        }

        for h in hess_vals.iter_mut() {
            *h *= self.0.scale * self.0.scale;
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
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(0.000001, mass)?;

        // Check that the point still has velocity in the positive x direction
        dbg!(&velocity);
        dbg!(&impulse);
        assert!(velocity[0] > 0.8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn sticking_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(1.5, mass)?;
        // Check that the point gets stuck
        dbg!(&impulse);
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6, epsilon = 1e-8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);
        Ok(())
    }

    fn sliding_point_tester(mu: f64, mass: f64) -> Result<(Vector2<f64>, Vector2<f64>), Error> {
        let params = FrictionParams {
            dynamic_friction: mu,
            inner_iterations: 30,
            tolerance: 1e-10,
            print_level: 0,
        };

        let velocity = vec![[1.0, 0.0]]; // one point sliding right.
        let contact_impulse = vec![10.0 * mass];
        let masses = vec![mass; 1];

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = FrictionSolver::without_contact_jacobian(&velocity, &contact_impulse, &contact_basis, &masses, params)?;
        let result = solver.step()?;
        let FrictionSolveResult {
            solution,
            ..
        } = result;

        let impulse = Vector2(solution[0]) * solver.scale();
        let final_velocity = Vector2(velocity[0]) + impulse / mass;

        Ok((final_velocity, impulse))
    }

    /// A tetrahedron sliding on a slanted surface.
    #[test]
    fn sliding_tet() -> Result<(), Error> {
        let params = FrictionParams {
            dynamic_friction: 0.001,
            inner_iterations: 40,
            tolerance: 1e-10,
            print_level: 0,
        };

        let velocity = vec![
            [
                0.07225747944670913,
                0.0000001280108566301736
            ],
            [
                0.06185827187696774,
                -0.0060040275393186595
            ]
        ]; // tet vertex velocities
        let contact_impulse = vec![
            -0.0000000018048827573828247,
            -0.00003259055555607145
        ];
        let masses = vec![
            0.0003720701030949866,
            0.0003720701030949866,
        ];

        let mut contact_basis = ContactBasis::new();
        let normals = vec![
            [
                -0.0,
                -0.7071067811865476,
                -0.7071067811865476
            ],
            [
                -0.0,
                -0.7071067811865476,
                -0.7071067811865476
            ]
        ];
        contact_basis.update_from_normals(normals);

        let mut solver = FrictionSolver::without_contact_jacobian(&velocity, &contact_impulse, &contact_basis, &masses, params)?;
        let result = solver.step()?;
        let FrictionSolveResult {
            solution,
            ..
        } = result;

        let final_velocity: Vec<_> =
            zip!(velocity.iter(), solution.iter(), masses.iter())
            .map(|(&v, &r, &m)| Vector2(v) + Vector2(r) * (solver.scale() / m))
            .collect();

        dbg!(&solution);
        dbg!(&final_velocity);

        Ok(())
    }
}
