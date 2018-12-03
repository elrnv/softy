use crate::attrib_defines::*;
use crate::constraint::*;
use crate::energy::*;
use crate::energy_models::{
    gravity::Gravity, momentum::MomentumPotential, volumetric_neohookean::ElasticTetMeshEnergy,
};
//use geo::io::save_tetmesh_ascii;
use crate::constraints::{smooth_contact::LinearSmoothContactConstraint, volume::VolumeConstraint};
use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::*, VertexPositions, Attrib};
use crate::matrix::*;
use ipopt::{self, Index, Number};
use reinterpret::*;
use std::fmt;
use std::{cell::RefCell, rc::Rc};

use crate::TetMesh;
use crate::TriMesh;

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NonLinearProblem {
    /// Position from the previous time step. We need to keep track of previous positions
    /// explicitly since we are doing a displacement solve. This vector is updated between steps
    /// and shared with other solver components like energies and constraints.
    pub prev_pos: Rc<RefCell<Vec<Vector3<f64>>>>,
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
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
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
    pub fn intermediate_cb(
        &mut self,
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
    ) -> bool {
        self.iterations += 1;
        !(self.interrupt_checker)()
    }

    /// Update the tetmesh vertex positions with the given displacment field.
    /// `dx` is expected to contain contiguous triplets of coordinates (x,y,z) for each vertex.
    pub fn update(&mut self, dx: &[f64]) {
        let displacement: &[Vector3<f64>] = reinterpret_slice(dx);
        let mut tetmesh = self.tetmesh.borrow_mut();
        let verts = tetmesh.vertex_positions_mut();
        let prev_pos = self.prev_pos.borrow();
        verts
            .iter_mut()
            .zip(prev_pos.iter())
            .zip(displacement.iter())
            .for_each(|((p, prev_p), disp)| *p = (*prev_p + *disp).into());
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
        //println!("SETTING BOUNDS!! {:?}", bound);

        if let Ok(fixed_verts) = tetmesh.attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB) {
            // Find and set fixed vertices.
            let pos_lo: &mut [[Number; 3]] = reinterpret_mut_slice(x_l);
            let pos_hi: &mut [[Number; 3]] = reinterpret_mut_slice(x_u);
            pos_lo.iter_mut()
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

    fn initial_point(&self) -> Vec<Number> {
        vec![0.0; self.num_variables()]
    }

    fn objective(&mut self, dx: &[Number], obj: &mut Number) -> bool {
        self.update(dx);
        let tetmesh = self.tetmesh.borrow();
        let pos: &[Number] = reinterpret_slice(tetmesh.vertex_positions());
        *obj = self.energy_model.energy(dx);
        *obj += self.gravity.energy(pos);
        if let Some(ref mut mp) = self.momentum_potential {
            *obj += mp.energy(dx);
        }
        true
    }

    fn objective_grad(&mut self, dx: &[Number], grad_f: &mut [Number]) -> bool {
        self.update(dx);
        grad_f.iter_mut().for_each(|x| *x = 0.0); // clear gradient vector
        let tetmesh = self.tetmesh.borrow();
        let pos: &[Number] = reinterpret_slice(tetmesh.vertex_positions());
        self.energy_model.add_energy_gradient(dx, grad_f);
        self.gravity.add_energy_gradient(pos, grad_f);
        if let Some(ref mut mp) = self.momentum_potential {
            mp.add_energy_gradient(dx, grad_f);
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
        if let Some(ref scc) = self.smooth_contact_constraint{
            num += scc.constraint_size();
            //println!("num constraints  = {:?}", num);
        }
        num
    }

    fn num_constraint_jac_non_zeros(&self) -> usize {
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

    fn constraint(&mut self, dx: &[Number], g: &mut [Number]) -> bool {
        self.update(dx);
        let tetmesh = self.tetmesh.borrow();
        let x: &[Number] = reinterpret_slice(tetmesh.vertex_positions());

        let mut i = 0; // Counter.

        if let Some(ref mut vc) = self.volume_constraint {
            let n = vc.constraint_size();
            vc.constraint(x, &mut g[i..i+n]);
            i += n;
        }

        if let Some(ref mut scc) = self.smooth_contact_constraint {
            let n = scc.constraint_size();
            scc.constraint(dx, &mut g[i..i+n]);
            //println!("g = {:?}", &g[i..i+n]);
            //i += n;
        }

        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let mut i = 0;
        if let Some(ref vc) = self.volume_constraint {
            let mut bounds = vc.constraint_bounds();
            let n = vc.constraint_size();
            g_l[i..i+n].swap_with_slice(&mut bounds.0);
            g_u[i..i+n].swap_with_slice(&mut bounds.1);
            i += n;
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let mut bounds = scc.constraint_bounds();
            let n = scc.constraint_size();
            g_l[i..i+n].swap_with_slice(&mut bounds.0);
            g_u[i..i+n].swap_with_slice(&mut bounds.1);
        }
        true
    }

    fn constraint_jac_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool {
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

    fn constraint_jac_values(&mut self, dx: &[Number], vals: &mut [Number]) -> bool {
        self.update(dx);
        let tetmesh = self.tetmesh.borrow();
        let x: &[Number] = reinterpret_slice(tetmesh.vertex_positions());

        let mut i = 0;

        if let Some(ref vc) = self.volume_constraint {
            let n = vc.constraint_jacobian_size();
            vc.constraint_jacobian_values(x, &mut vals[i..i+n]);
            i += n;
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let n = scc.constraint_jacobian_size();
            scc.constraint_jacobian_values(dx, &mut vals[i..i+n]);
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

    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut i = 0;

        // Add energy indices
        for MatrixElementIndex { ref row, ref col } in
            self.energy_model.energy_hessian_indices().iter()
        {
            rows[i] = *row as Index;
            cols[i] = *col as Index;
            i += 1;
        }

        if let Some(ref mut mp) = self.momentum_potential {
            for MatrixElementIndex { ref row, ref col } in mp.energy_hessian_indices().iter() {
                rows[i] = *row as Index;
                cols[i] = *col as Index;
                i += 1;
            }
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
        &mut self,
        dx: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        self.update(dx);
        let tetmesh = self.tetmesh.borrow();
        let x: &[Number] = reinterpret_slice(tetmesh.vertex_positions());

        let mut i = 0;

        for val in self.energy_model.energy_hessian_values(dx).iter() {
            vals[i] = obj_factor * (*val as Number);
            i += 1;
        }

        if let Some(ref mut mp) = self.momentum_potential {
            for val in mp.energy_hessian_values(dx).iter() {
                vals[i] = obj_factor * (*val as Number);
                i += 1;
            }
        }

        let coff = 0;

        if let Some(ref vc) = self.volume_constraint {
            let nc = vc.constraint_size();
            let nh = vc.constraint_hessian_size();
            vc.constraint_hessian_values(x, &lambda[coff..coff + nc], &mut vals[i..i+nh]);
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
