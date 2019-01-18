use crate::attrib_defines::*;
use crate::constraint::*;
use crate::energy::*;
use crate::energy_models::{
    gravity::Gravity, momentum::MomentumPotential, volumetric_neohookean::ElasticTetMeshEnergy,
};
//use geo::io::save_tetmesh_ascii;
use crate::constraints::{smooth_contact::LinearSmoothContactConstraint, volume::VolumeConstraint};
use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::*, Attrib, VertexPositions};
use crate::matrix::*;
use ipopt::{self, Index, Number};
use reinterpret::*;
use std::fmt;
use std::{cell::RefCell, rc::Rc};

use crate::TetMesh;
use crate::TriMesh;

pub struct Solution {
    /// This is the solution of the solve.
    pub primal_variables: Vec<f64>,
    /// Lower bound multipliers.
    pub lower_bound_multipliers: Vec<f64>,
    /// Upper bound multipliers.
    pub upper_bound_multipliers: Vec<f64>,
    /// Constraint (Lagrange) multipliers.
    pub constraint_multipliers: Vec<f64>,
}

/// Create an empty solution.
impl Default for Solution {
    fn default() -> Solution {
        Solution {
            primal_variables: Vec::new(),
            lower_bound_multipliers: Vec::new(),
            upper_bound_multipliers: Vec::new(),
            constraint_multipliers: Vec::new(),
        }
    }
}

/// Materialize a solution from the references got from Ipopt.
impl<'a> From<ipopt::Solution<'a>> for Solution {
    fn from(sol: ipopt::Solution<'a>) -> Solution {
        let mut mysol = Solution::default();
        mysol.update(sol);
        mysol
    }
}

impl Solution {
    /// Initialize a solution with all variables and multipliers set to zero.
    pub fn reset(&mut self, num_variables: usize, num_constraints: usize) -> &mut Self {
        self.clear();
        let x = vec![0.0; num_variables];
        self.primal_variables.extend_from_slice(&x);
        self.lower_bound_multipliers.extend_from_slice(&x);
        self.upper_bound_multipliers.extend_from_slice(&x);
        self.constraint_multipliers
            .extend_from_slice(&vec![0.0; num_constraints]);
        self
    }

    /// Clear all solution vectors.
    pub fn clear(&mut self) {
        self.primal_variables.clear();
        self.lower_bound_multipliers.clear();
        self.upper_bound_multipliers.clear();
        self.constraint_multipliers.clear();
    }

    /// Update allocated solution vectors with new data from Ipopt.
    pub fn update<'a>(&mut self, sol: ipopt::Solution<'a>) -> &mut Self {
        self.clear();
        self.primal_variables
            .extend_from_slice(sol.primal_variables);
        self.lower_bound_multipliers
            .extend_from_slice(sol.lower_bound_multipliers);
        self.upper_bound_multipliers
            .extend_from_slice(sol.upper_bound_multipliers);
        self.constraint_multipliers
            .extend_from_slice(sol.constraint_multipliers);
        self
    }
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NonLinearProblem {
    /// Position from the previous time step. We need to keep track of previous positions
    /// explicitly since we are doing a displacement solve. This vector is updated between steps
    /// and shared with other solver components like energies and constraints.
    pub prev_pos: Rc<RefCell<Vec<Vector3<f64>>>>,
    /// Velocity from the previous time step.
    pub prev_vel: Rc<RefCell<Vec<Vector3<f64>>>>,
    /// Tetrahedron mesh representing a soft solid computational domain.
    pub tetmesh: Rc<RefCell<TetMesh>>,
    /// Static or animated collision object represented by a triangle mesh.
    pub kinematic_object: Option<Rc<RefCell<TriMesh>>>,
    /// Elastic energy model.
    pub energy_model: ElasticTetMeshEnergy,
    /// Gravitational potential energy.
    pub gravity: Gravity,
    /// Momentum potential. The energy responsible for inertia.
    pub momentum_potential: Option<MomentumPotential>,
    /// Constraint on the total volume.
    pub volume_constraint: Option<VolumeConstraint>,
    /// Contact constraint on the smooth solid representation against the kinematic object.
    pub smooth_contact_constraint: Option<LinearSmoothContactConstraint>,
    /// Displacement bounds. This controlls how big of a step we can take per vertex position
    /// component. In otherwords the bounds on the inf norm for each vertex displacement.
    pub displacement_bound: Option<f64>,
    /// The time step defines the amount of time elapsed between steps (calls to `advance`).
    pub time_step: f64,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Solution data. This is kept around for warm starts.
    pub solution: Solution,
}

impl fmt::Debug for NonLinearProblem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "NonLinearProblem {{ energy_model: {:?}, volume_constraint: {:?}, \
             iterations: {:?} }}",
            self.energy_model, self.volume_constraint, self.iterations
        )
    }
}

impl NonLinearProblem {
    /// Save Ipopt solution for warm starts.
    pub fn save_solution(&mut self, solution: ipopt::Solution) {
        self.solution.update(solution);
    }

    /// Reset solution used for warm starts.
    pub fn reset_solution(&mut self) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        self.solution
            .reset(self.num_variables(), self.num_constraints());
    }

    /// Get the current iteration count and reset it.
    pub fn pop_iteration_count(&mut self) -> usize {
        let iter = self.iterations;
        // Reset caunt
        self.iterations = 0;
        if let Some(ref mut scc) = self.smooth_contact_constraint.as_mut() {
            scc.reset_iter_count();
        }
        iter
    }

    /// Intermediate callback for `Ipopt`.
    #[allow(clippy::too_many_arguments)] // TODO: Improve on the ipopt interface
    pub fn intermediate_cb(&mut self, _data: ipopt::IntermediateCallbackData) -> bool {
        self.iterations += 1;
        !(self.interrupt_checker)()
    }

    /// Commit displacement by advancing the internal state by the given displacement `dx`.
    pub fn advance(&mut self, disp: impl Iterator<Item = Vector3<f64>>) {
        let dt_inv = 1.0 / self.time_step;

        let mut prev_vel = self.prev_vel.borrow_mut();
        let mut prev_pos = self.prev_pos.borrow_mut();

        disp.zip(prev_pos.iter_mut().zip(prev_vel.iter_mut()))
            .for_each(|(dp, (prev_p, prev_v))| {
                *prev_p += dp;
                *prev_v = dp * dt_inv;
            });

        let mut tetmesh = self.tetmesh.borrow_mut();
        let verts = tetmesh.vertex_positions_mut();
        verts.copy_from_slice(reinterpret_slice(prev_pos.as_slice()));

        // Since we transformed the mesh, we need to invalidate its neighbour data so it's
        // recomputed at the next time step (if applicable).
        if let Some(ref mut scc) = self.smooth_contact_constraint {
            let mesh = self.kinematic_object.as_ref().unwrap().borrow();
            scc.update_cache(reinterpret_slice::<_, [f64; 3]>(mesh.vertex_positions()));
        }
    }

    /// Linearized constraint true violation measure.
    pub fn linearized_constraint_violation_l1(&self, dx: &[f64]) -> f64 {
        let mut value = 0.0;

        if let Some(ref scc) = self.smooth_contact_constraint {
            let prev_pos = self.prev_pos.borrow();
            let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

            let n = scc.constraint_size();

            let mut g = vec![0.0; n];
            scc.0.constraint(x, dx, &mut g);

            let (g_l, g_u) = scc.0.constraint_bounds();
            assert_eq!(g_l.len(), n);
            assert_eq!(g_u.len(), n);

            value += g
                .into_iter()
                .zip(g_l.into_iter().zip(g_u.into_iter()))
                .map(|(c, (l, u))| {
                    assert!(l <= u);
                    if c < l {
                        // below lower bound
                        l - c
                    } else if c > u {
                        // above upper bound
                        c - u
                    } else {
                        // Constraint not violated
                        0.0
                    }
                })
                .sum::<f64>();
        }

        value
    }

    /// Linearized constraint model violation measure.
    pub fn linearized_constraint_violation_model_l1(&self, dx: &[f64]) -> f64 {
        let mut value = 0.0;

        if let Some(ref scc) = self.smooth_contact_constraint {
            let prev_pos = self.prev_pos.borrow();
            let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

            let n = scc.constraint_size();

            let mut g = vec![0.0; n];
            scc.constraint(x, dx, &mut g);

            let (g_l, g_u) = scc.constraint_bounds();
            assert_eq!(g_l.len(), n);
            assert_eq!(g_u.len(), n);

            value += g
                .into_iter()
                .zip(g_l.into_iter().zip(g_u.into_iter()))
                .map(|(c, (l, u))| {
                    assert!(l <= u);
                    if c < l {
                        // below lower bound
                        l - c
                    } else if c > u {
                        // above upper bound
                        c - u
                    } else {
                        // Constraint not violated
                        0.0
                    }
                })
                .sum::<f64>();
        }

        value
    }

    /// Compute and return the objective value.
    pub fn objective_value(&self, dx: &[Number]) -> f64 {
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

        let mut obj = self.energy_model.energy(x, dx);

        obj += self.gravity.energy(x, dx);

        if let Some(ref mp) = self.momentum_potential {
            let prev_vel = self.prev_pos.borrow();
            let v: &[Number] = reinterpret_slice(prev_vel.as_slice());
            obj += mp.energy(v, dx);
        }
        obj
    }

    //pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    //    a.iter().zip(b.iter()).map(|(&a,&b)| a*b).sum()
    //}
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for NonLinearProblem {
    fn num_variables(&self) -> usize {
        self.tetmesh.borrow().num_vertices() * 3
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        let tetmesh = self.tetmesh.borrow();
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        let bound = self.displacement_bound.unwrap_or(2e19);
        x_l.iter_mut().for_each(|x| *x = -bound);
        x_u.iter_mut().for_each(|x| *x = bound);

        if let Ok(fixed_verts) = tetmesh.attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
        {
            // Find and set fixed vertices.
            let pos_lo: &mut [[Number; 3]] = reinterpret_mut_slice(x_l);
            let pos_hi: &mut [[Number; 3]] = reinterpret_mut_slice(x_u);
            pos_lo
                .iter_mut()
                .zip(pos_hi.iter_mut())
                .zip(fixed_verts.iter())
                .filter(|&(_, &fixed)| fixed != 0)
                .for_each(|((l, u), _)| {
                    *l = [0.0; 3];
                    *u = [0.0; 3];
                });
        }
        true
    }

    fn initial_point(&self, x: &mut [Number]) {
        x.copy_from_slice(self.solution.primal_variables.as_slice());
    }

    fn initial_bounds_multipliers(&self, z_l: &mut [Number], z_u: &mut [Number]) {
        z_l.copy_from_slice(self.solution.lower_bound_multipliers.as_slice());
        z_u.copy_from_slice(self.solution.upper_bound_multipliers.as_slice());
    }

    fn objective(&self, dx: &[Number], obj: &mut Number) -> bool {
        *obj = self.objective_value(dx);
        true
    }

    fn objective_grad(&self, dx: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f.iter_mut().for_each(|x| *x = 0.0); // clear gradient vector
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());
        self.energy_model.add_energy_gradient(x, dx, grad_f);
        self.gravity.add_energy_gradient(x, dx, grad_f);
        if let Some(ref mp) = self.momentum_potential {
            let prev_vel = self.prev_pos.borrow();
            let v: &[Number] = reinterpret_slice(prev_vel.as_slice());
            mp.add_energy_gradient(v, dx, grad_f);
        }

        true
    }
}

impl ipopt::ConstrainedProblem for NonLinearProblem {
    fn num_constraints(&self) -> usize {
        let mut num = 0;
        if let Some(ref vc) = self.volume_constraint {
            num += vc.constraint_size();
        }
        if let Some(ref scc) = self.smooth_contact_constraint {
            num += scc.constraint_size();
            //println!("num constraints  = {:?}", num);
        }
        num
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        let mut num = 0;
        if let Some(ref vc) = self.volume_constraint {
            num += vc.constraint_jacobian_size();
        }
        if let Some(ref scc) = self.smooth_contact_constraint {
            num += scc.constraint_jacobian_size();
            //println!("scc jac size = {:?}", num);
        }
        num
    }

    fn initial_constraint_multipliers(&self, lambda: &mut [Number]) {
        lambda.copy_from_slice(self.solution.constraint_multipliers.as_slice());
    }

    fn constraint(&self, dx: &[Number], g: &mut [Number]) -> bool {
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

        let mut i = 0; // Counter.

        if let Some(ref vc) = self.volume_constraint {
            let n = vc.constraint_size();
            vc.constraint(x, dx, &mut g[i..i + n]);
            i += n;
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let n = scc.constraint_size();
            scc.constraint(x, dx, &mut g[i..i + n]);
        }

        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let mut i = 0;
        if let Some(ref vc) = self.volume_constraint {
            let mut bounds = vc.constraint_bounds();
            let n = vc.constraint_size();
            g_l[i..i + n].swap_with_slice(&mut bounds.0);
            g_u[i..i + n].swap_with_slice(&mut bounds.1);
            i += n;
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let mut bounds = scc.constraint_bounds();
            let n = scc.constraint_size();
            g_l[i..i + n].swap_with_slice(&mut bounds.0);
            g_u[i..i + n].swap_with_slice(&mut bounds.1);
        }
        true
    }

    fn constraint_jacobian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut i = 0; // counter

        let mut row_offset = 0;
        if let Some(ref vc) = self.volume_constraint {
            for MatrixElementIndex { row, col } in vc.constraint_jacobian_indices_iter() {
                rows[i] = row as Index;
                cols[i] = col as Index;
                i += 1;
            }
            row_offset += vc.constraint_jacobian_size();
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            for MatrixElementIndex { row, col } in scc.constraint_jacobian_indices_iter() {
                rows[i] = (row + row_offset) as Index;
                cols[i] = col as Index;
                i += 1;
            }
            //row_offset += scc.constraint_jacobian_size();
        }

        true
    }

    fn constraint_jacobian_values(&self, dx: &[Number], vals: &mut [Number]) -> bool {
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

        let mut i = 0;

        if let Some(ref vc) = self.volume_constraint {
            let n = vc.constraint_jacobian_size();
            vc.constraint_jacobian_values(x, dx, &mut vals[i..i + n]);
            i += n;
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let n = scc.constraint_jacobian_size();
            scc.constraint_jacobian_values(x, dx, &mut vals[i..i + n]);
            //println!("jac g vals = {:?}", &vals[i..i+n]);
            //i += n;
        }

        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        let mut num = self.energy_model.energy_hessian_size();
        if let Some(ref mp) = self.momentum_potential {
            num += mp.energy_hessian_size();
        }
        if let Some(ref vc) = self.volume_constraint {
            num += vc.constraint_hessian_size();
        }
        //if let Some(ref scc) = self.smooth_contact_constraint {
        //    num += scc.constraint_hessian_size();
        //}
        num
    }

    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut i = 0;

        // Add energy indices
        let n = self.energy_model.energy_hessian_size();
        self.energy_model
            .energy_hessian_rows_cols(&mut rows[i..i + n], &mut cols[i..i + n]);
        i += n;

        if let Some(ref mp) = self.momentum_potential {
            let n = mp.energy_hessian_size();
            mp.energy_hessian_rows_cols(&mut rows[i..i + n], &mut cols[i..i + n]);
            i += n;
        }

        // Add volume constraint indices
        if let Some(ref vc) = self.volume_constraint {
            for MatrixElementIndex { row, col } in vc.constraint_hessian_indices_iter() {
                rows[i] = row as Index;
                cols[i] = col as Index;
                i += 1;
            }
        }
        //if let Some(ref scc) = self.smooth_contact_constraint {
        //    for MatrixElementIndex { row, col } in scc.constraint_hessian_indices_iter() {
        //        rows[i] = row as Index;
        //        cols[i] = col as Index;
        //        i += 1;
        //    }
        //}

        true
    }
    fn hessian_values(
        &self,
        dx: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

        let mut i = 0;
        let n = self.energy_model.energy_hessian_size();
        self.energy_model
            .energy_hessian_values(x, dx, &mut vals[i..i + n]);
        i += n;

        if let Some(ref mp) = self.momentum_potential {
            let n = mp.energy_hessian_size();
            let prev_vel = self.prev_pos.borrow();
            let v: &[Number] = reinterpret_slice(prev_vel.as_slice());
            mp.energy_hessian_values(v, dx, &mut vals[i..i + n]);
            i += n;
        }

        // Multiply energy hessian by objective factor.
        for v in vals[0..i].iter_mut() {
            *v *= obj_factor;
        }

        let coff = 0;

        if let Some(ref vc) = self.volume_constraint {
            let nc = vc.constraint_size();
            let nh = vc.constraint_hessian_size();
            vc.constraint_hessian_values(x, dx, &lambda[coff..coff + nc], &mut vals[i..i + nh]);
            //i += nh;
            //coff += nc
        }

        //if let Some(ref scc) = self.smooth_contact_constraint {
        //    let nc = scc.constraint_size();
        //    let nh = scc.constraint_hessian_size();
        //    scc.constraint_hessian_values(x, &lambda[coff..coff + nc], &mut vals[i..i+nh]);
        //    //i += nh;
        //    //coff += nc;
        //}

        true
    }
}
