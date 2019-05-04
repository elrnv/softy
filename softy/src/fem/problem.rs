use crate::attrib_defines::*;
use crate::constraint::*;
use crate::constraints::{volume::VolumeConstraint, ContactConstraint};
use crate::energy::*;
use crate::energy_models::{
    gravity::Gravity, momentum::MomentumPotential, volumetric_neohookean::ElasticTetMeshEnergy,
};
use crate::matrix::*;
use geo::math::Vector3;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use ipopt::{self, Number};
use reinterpret::*;
use std::fmt;
use std::{cell::RefCell, rc::Rc};

use crate::TetMesh;
use crate::TriMesh;
use utils::zip;

#[derive(Clone)]
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

    ///// The number of constraints may change between saving the warm start solution and using it
    ///// for the next solve. For this reason we must remap the old multipliers to the new set of
    ///// constraints. Constraint multipliers that are new in the next solve will have a zero value.
    ///// This function works on a subset of all multipliers. The caller gives a slice of the
    ///// multipliers for which this function produces a new Vec of multipliers correspnding to the
    ///// new constraints, where the old multipliers are copied as available.
    ///// The values in `new_indices` and `old_indices` are required to be sorted.
    /////
    ///// NOTE: Most efficient to replace the entire constraint_multipliers vector in the warm start.
    //pub fn remap_constraint_multipliers(
    //    constraint_multipliers: &[f64],
    //    old_indices: &[usize],
    //    new_indices: &[usize],
    //) -> Vec<f64> {
    //    remap_values(constraint_multipliers, 0.0, old_indices, new_indices)
    //}
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
    pub smooth_contact_constraint: Option<Box<dyn ContactConstraint>>,
    /// Displacement bounds. This controlls how big of a step we can take per vertex position
    /// component. In other words the bounds on the inf norm for each vertex displacement.
    pub displacement_bound: Option<f64>,
    /// The time step defines the amount of time elapsed between steps (calls to `advance`).
    pub time_step: f64,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Solution data. This is kept around for warm starts.
    pub warm_start: Solution,
    pub initial_residual_error: f64,
    pub iter_counter: RefCell<usize>,
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
    pub fn update_warm_start(&mut self, solution: ipopt::Solution) {
        self.warm_start.update(solution);
    }

    ///// Clear the warm start using the sizes in the given solution.
    //pub fn clear_warm_start(&mut self, solution: ipopt::Solution) {
    //    self.warm_start.reset(
    //        solution.primal_variables.len(),
    //        solution.constraint_multipliers.len(),
    //    );
    //}

    /// Reset solution used for warm starts. Note that if the number of constraints has changed,
    /// then this method will set the warm start to have the new number of constraints.
    pub fn reset_warm_start(&mut self) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        self.warm_start
            .reset(self.num_variables(), self.num_constraints());
    }

    /// Get the current iteration count and reset it.
    pub fn pop_iteration_count(&mut self) -> usize {
        let iter = self.iterations;
        // Reset caunt
        self.iterations = 0;
        iter
    }

    /// Intermediate callback for `Ipopt`.
    pub fn intermediate_cb(&mut self, data: ipopt::IntermediateCallbackData) -> bool {
        if data.iter_count == 0 {
            // Record the initial max of dual and primal infeasibility.
            if data.inf_du > 0.0 {
                self.initial_residual_error = data.inf_du;
            }
        }

        self.iterations += 1;
        !(self.interrupt_checker)()
    }

    /// Get the set of currently active constraints.
    pub fn active_constraint_set(&self) -> Vec<usize> {
        let mut active_set = Vec::new();

        if self.volume_constraint.is_some() {
            active_set.push(0);
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let scc_active_constraints = scc.active_constraint_indices().unwrap_or_default();
            let offset = active_set.len();
            for c in scc_active_constraints.into_iter() {
                active_set.push(c + offset);
            }
        }

        active_set
    }

    pub fn clear_friction_impulses(&mut self) {
        if let Some(ref mut scc) = self.smooth_contact_constraint {
            scc.clear_friction_force();
        }
    }

    /// Restore the constraint set.
    pub fn reset_constraint_set(&mut self) -> bool {
        self.update_constraint_set(None)
    }

    /// Update all stateful constraints with the most recent data.
    /// Return an estimate if any constraints have changed. This estimate may have false negatives.
    /// Also return the mapping from new constraint indices to old constraint indices via a `Vec`
    /// of possibly invalid indices.
    pub fn update_constraint_set(&mut self, solution: Option<ipopt::Solution>) -> bool {
        let mut changed = false; // Report if anything has changed to the caller.

        if let Some(ref mut scc) = self.smooth_contact_constraint {
            let prev_pos = self.prev_pos.borrow_mut();
            let x: &[Number] = reinterpret_slice(prev_pos.as_slice());
            changed |= scc.update_cache(Some(x), solution.map(|sol| sol.primal_variables));
        }

        changed
    }

    /// Build a new set of multipliers from the old set and replace warm start multipliers with the
    /// new set.
    pub fn remap_contacts(&mut self, mut old_constraint_set: impl Iterator<Item = usize> + Clone) {
        use crate::constraints::remap_values;
        let mut new_constraint_set = self.active_constraint_set().into_iter();

        // Remap multipliers
        let new_multipliers = remap_values(
            self.warm_start.constraint_multipliers.iter().cloned(),
            0.0,
            old_constraint_set.clone(),
            new_constraint_set.clone(),
        );
        std::mem::replace(&mut self.warm_start.constraint_multipliers, new_multipliers);

        // Remap friction forces (if any)
        if let Some(ref mut scc) = self.smooth_contact_constraint {
            if self.volume_constraint.is_some() {
                // Consume the first constraint if any.
                old_constraint_set.next();
                new_constraint_set.next();
            }
            let old_set: Vec<_> = old_constraint_set.collect();
            let new_set: Vec<_> = new_constraint_set.collect();
            scc.remap_friction(&old_set, &new_set);
        }
    }

    ///// Update all stateful constraints with the most recent data. This also involves remapping any
    ///// multipliers that may have changed.
    ///// Return an estimate if any constraints have changed. This estimate may have false negatives.
    //pub fn update_constraints_and_remap(&mut self, solution: ipopt::Solution) -> bool {
    //    let mut changed = false; // Report if anything has changed to the caller.

    //    let offset = if self.volume_constraint.is_some() {
    //        1
    //    } else {
    //        0
    //    };

    //    // Build a new set of multipliers from the old set.
    //    let mut new_multipliers = self.warm_start.constraint_multipliers.clone();

    //    if let Some(ref mut scc) = self.smooth_contact_constraint {
    //        let (cache_changed, mapping) = scc.update_cache_with_mapping()
    //            .expect("Failed to update cached neighbourhoods");
    //        changed |= cache_changed;
    //        new_multipliers.resize(offset + mapping.len(), 0.0);
    //        dbg!(&new_multipliers);
    //        dbg!(&self.warm_start.constraint_multipliers);
    //        for (old_idx, new_mult) in zip!(mapping.into_iter(), new_multipliers.iter_mut())
    //            .filter_map(|(idx, mult)| idx.into_option().map(|i| (i, mult)))
    //        {
    //            *new_mult = self.warm_start.constraint_multipliers[offset + old_idx];
    //        }
    //        dbg!(&new_multipliers);
    //    }

    //    std::mem::replace(&mut self.warm_start.constraint_multipliers, new_multipliers);
    //    dbg!(&self.warm_start.constraint_multipliers);

    //    changed
    //}

    /// Commit displacement by advancing the internal state by the given displacement `dx`.
    pub fn advance(
        &mut self,
        solution: ipopt::Solution,
        and_warm_start: bool,
    ) -> (Solution, Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let (old_warm_start, old_prev_pos, old_prev_vel) = {
            // Reinterpret solver variables as positions in 3D space.
            let disp: &[Vector3<f64>] = reinterpret_slice(&solution.primal_variables);

            // In static simulations, velocity is simply displacement.
            let dt = self.time_step;
            let dt_inv = 1.0 / if dt > 0.0 { dt } else { 1.0 };

            let mut prev_pos = self.prev_pos.borrow_mut();
            let mut prev_vel = self.prev_vel.borrow_mut();

            let old_prev_pos = prev_pos.clone();
            let old_prev_vel = prev_vel.clone();

            zip!(disp.iter(), prev_pos.iter_mut(), prev_vel.iter_mut()).for_each(
                |(&dp, prev_p, prev_v)| {
                    *prev_p += dp;
                    *prev_v = dp * dt_inv;
                },
            );

            {
                let mut tetmesh = self.tetmesh.borrow_mut();
                let verts = tetmesh.vertex_positions_mut();
                verts.copy_from_slice(reinterpret_slice(prev_pos.as_slice()));
            }

            let old_warm_start = self.warm_start.clone();

            (old_warm_start, old_prev_pos, old_prev_vel)
        };

        if !and_warm_start {
            self.reset_warm_start();
        }

        //if and_warm_start {
        //    self.update_warm_start(solution);
        //} else {
        //    self.clear_warm_start(solution);
        //}

        //// Since we transformed the mesh, we need to invalidate its neighbour data so it's
        //// recomputed at the next time step (if applicable).
        //self.update_constraints();

        (old_warm_start, old_prev_pos, old_prev_vel)
    }

    pub fn update_max_step(&mut self, step: f64) {
        if let Some(ref mut scc) = self.smooth_contact_constraint {
            scc.update_max_step(step);
        }
    }
    pub fn update_radius_multiplier(&mut self, rad_mult: f64) {
        if let Some(ref mut scc) = self.smooth_contact_constraint {
            scc.update_radius_multiplier(rad_mult);
        }
    }

    /// Revert to the given old solution by the given displacement.
    pub fn revert_to(
        &mut self,
        solution: Solution,
        old_prev_pos: Vec<Vector3<f64>>,
        old_prev_vel: Vec<Vector3<f64>>,
    ) {
        {
            // Reinterpret solver variables as positions in 3D space.
            let mut prev_pos = self.prev_pos.borrow_mut();
            let mut prev_vel = self.prev_vel.borrow_mut();

            std::mem::replace(&mut *prev_vel, old_prev_vel);
            std::mem::replace(&mut *prev_pos, old_prev_pos);

            let mut tetmesh = self.tetmesh.borrow_mut();
            let verts = tetmesh.vertex_positions_mut();
            verts.copy_from_slice(reinterpret_slice(prev_pos.as_slice()));

            std::mem::replace(&mut self.warm_start, solution);
        }

        // Since we transformed the mesh, we need to invalidate its neighbour data so it's
        // recomputed at the next time step (if applicable).
        //self.update_constraints();
    }

    fn compute_constraint_violation(&self, displacement: &[f64], constraint: &mut [f64]) {
        use ipopt::ConstrainedProblem;
        let mut lower = vec![0.0; constraint.len()];
        let mut upper = vec![0.0; constraint.len()];
        self.constraint_bounds(&mut lower, &mut upper);
        assert!(self.constraint(displacement, constraint));
        for (c, (l, u)) in constraint
            .iter_mut()
            .zip(lower.into_iter().zip(upper.into_iter()))
        {
            assert!(l <= u); // sanity check

            // Subtract the appropriate bound from the constraint function:
            // If the constraint is lower than the lower bound, then the multiplier will be
            // non-zero and the result of the constraint force must be balanced by the objective
            // gradient under convergence. The same goes for the upper bound.
            if *c < l {
                *c -= l;
            } else if *c > u {
                *c -= u;
            } else {
                *c = 0.0; // Otherwise the constraint is satisfied so we set it to zero.
            }
        }
    }

    fn constraint_violation_norm(&self, displacement: &[f64]) -> f64 {
        use ipopt::ConstrainedProblem;
        let mut g = vec![0.0; self.num_constraints()];
        self.compute_constraint_violation(displacement, &mut g);
        crate::inf_norm(&g)
    }

    /// Return the constraint violation and whether the neighbourhood data (sparsity) would be
    /// changed if we took this step.
    pub fn probe_contact_constraint_violation(&mut self, solution: ipopt::Solution) -> f64 {
        self.constraint_violation_norm(solution.primal_variables)
    }

    /// Linearized constraint true violation measure.
    //pub fn linearized_constraint_violation_l1(&self, dx: &[f64]) -> f64 {
    //    let mut value = 0.0;

    //    if let Some(ref scc) = self.smooth_contact_constraint {
    //        let prev_pos = self.prev_pos.borrow();
    //        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

    //        let n = scc.constraint_size();

    //        let mut g = vec![0.0; n];
    //        scc.0.constraint(x, dx, &mut g);

    //        let (g_l, g_u) = scc.0.constraint_bounds();
    //        assert_eq!(g_l.len(), n);
    //        assert_eq!(g_u.len(), n);

    //        value += g
    //            .into_iter()
    //            .zip(g_l.into_iter().zip(g_u.into_iter()))
    //            .map(|(c, (l, u))| {
    //                assert!(l <= u);
    //                if c < l {
    //                    // below lower bound
    //                    l - c
    //                } else if c > u {
    //                    // above upper bound
    //                    c - u
    //                } else {
    //                    // Constraint not violated
    //                    0.0
    //                }
    //            })
    //            .sum::<f64>();
    //    }

    //    value
    //}

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
            let prev_vel = self.prev_vel.borrow();
            let v: &[Number] = reinterpret_slice(prev_vel.as_slice());
            obj += mp.energy(v, dx);
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            if self.time_step > 0.0 {
                obj -= scc.frictional_dissipation(dx);
            }
        }
        obj
    }

    /// Return true if the friction impulse was updated, and false otherwise.
    pub fn update_friction_impulse(&mut self, solution: ipopt::Solution, constraint_values: &[f64]) -> bool {
        if let Some(ref mut scc) = self.smooth_contact_constraint {
            let prev_pos = self.prev_pos.borrow();
            let position: &[[f64; 3]] = reinterpret::reinterpret_slice(prev_pos.as_slice());
            let displacement: &[[f64; 3]] =
                reinterpret::reinterpret_slice(solution.primal_variables);
            let offset = if self.volume_constraint.is_some() {
                1
            } else {
                0
            };
            let contact_force = &solution.constraint_multipliers[offset..];
            let potential_values = &constraint_values[offset..];
            scc.update_friction_force(contact_force, position, displacement, potential_values)
        } else {
            false
        }
    }

    /// Return the stacked friction impulses: one for each vertex.
    pub fn friction_impulse(&self) -> Vec<[f64; 3]> {
        use ipopt::BasicProblem;
        let mut impulse = vec![0.0; self.num_variables()];
        if let Some(ref scc) = self.smooth_contact_constraint {
            scc.subtract_friction_force(&mut impulse);
            for f in impulse.iter_mut() {
                *f *= -self.time_step; // convert force to impulse
            }
        }
        reinterpret::reinterpret_vec(impulse)
    }

    /// Return the stacked contact impulses: one for each vertex.
    pub fn contact_impulse(&self) -> Vec<[f64; 3]> {
        let mut impulse = vec![[0.0; 3]; self.tetmesh.borrow().num_vertices()];
        if let Some(ref scc) = self.smooth_contact_constraint {
            let prev_pos = self.prev_pos.borrow();
            let position: &[f64] = reinterpret::reinterpret_slice(prev_pos.as_slice());
            // Get contact force from the warm start.
            let offset = if self.volume_constraint.is_some() {
                1
            } else {
                0
            };
            let contact_force = &self.warm_start.constraint_multipliers[offset..];
            scc.compute_contact_impulse(position, contact_force, self.time_step, &mut impulse);
        }
        reinterpret::reinterpret_vec(impulse)
    }

    /*
     * The followin functions are there for debugging jacobians and hessians
     */

    #[allow(dead_code)]
    fn output_mesh(&self, x: &[Number], dx: &[Number], name: &str) -> Result<(), crate::Error> {
        let mut iter_counter = self.iter_counter.borrow_mut();
        let mut mesh = self.tetmesh.borrow().clone();
        let all_displacements: &[Vector3<f64>] = reinterpret_slice(dx);
        let all_positions: &[Vector3<f64>] = reinterpret_slice(x);
        mesh.vertex_positions_mut()
            .iter_mut()
            .zip(all_displacements.iter())
            .zip(all_positions.iter())
            .for_each(|((mp, d), p)| *mp = (*p + *d).into());
        *iter_counter += 1;
        geo::io::save_tetmesh(
            &mesh,
            &std::path::PathBuf::from(format!("out/{}_{}.vtk", name, *iter_counter)),
        )?;

        if let Some(tri_mesh_ref) = self.kinematic_object.as_ref() {
            let obj = geo::mesh::PolyMesh::from(tri_mesh_ref.borrow().clone());
            geo::io::save_polymesh(
                &obj,
                &std::path::PathBuf::from(format!("out/tri_{}_{}.vtk", name, *iter_counter)),
            )?;
        } else {
            return Err(crate::Error::NoKinematicMesh);
        }

        dbg!(*iter_counter);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_jacobian_img(&self, jac: &na::DMatrix<f64>) {
        use image::ImageBuffer;

        let nrows = jac.nrows();
        let ncols = jac.ncols();

        let ciel = 1.0; //jac.max();
        let floor = -1.0; //jac.min();

        let img = ImageBuffer::from_fn(ncols as u32, nrows as u32, |c, r| {
            let val = jac[(r as usize, c as usize)];
            let color = if val > 0.0 {
                [255, (255.0 * val / ciel) as u8, 0]
            } else if val < 0.0 {
                [0, (255.0 * (1.0 + val / floor)) as u8, 255]
            } else {
                [255, 0, 255]
            };
            image::Rgb(color)
        });

        img.save(format!("out/jac_{}.png", self.iter_counter.borrow()))
            .expect("Failed to save Jacobian Image");
    }

    #[allow(dead_code)]
    pub fn print_jacobian_svd(&self, values: &[Number]) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        use na::{base::storage::Storage, DMatrix};

        if values.is_empty() {
            return;
        }

        let mut rows = vec![0; values.len()];
        let mut cols = vec![0; values.len()];
        assert!(self.constraint_jacobian_indices(&mut rows, &mut cols));

        let nrows = self.num_constraints();
        let ncols = self.num_variables();
        let mut jac = DMatrix::<f64>::zeros(nrows, ncols);
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            jac[(row as usize, col as usize)] += v;
        }

        self.write_jacobian_img(&jac);

        use std::io::Write;

        let mut f =
            std::fs::File::create(format!("out/jac_{}.txt", self.iter_counter.borrow())).unwrap();
        writeln!(&mut f, "jac = ").ok();
        for r in 0..nrows {
            for c in 0..ncols {
                if jac[(r, c)] != 0.0 {
                    write!(&mut f, "{:9.5}", jac[(r, c)]).ok();
                } else {
                    write!(&mut f, "    .    ",).ok();
                }
            }
            writeln!(&mut f).ok();
        }
        writeln!(&mut f).ok();

        let svd = na::SVD::new(jac, false, false);
        let s: &[Number] = svd.singular_values.data.as_slice();
        let cond = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        dbg!(cond);
    }

    #[allow(dead_code)]
    pub fn print_hessian_svd(&self, values: &[Number]) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        use na::{base::storage::Storage, DMatrix};

        if values.is_empty() {
            return;
        }

        let mut rows = vec![0; values.len()];
        let mut cols = vec![0; values.len()];
        assert!(self.hessian_indices(&mut rows, &mut cols));

        let mut hess = DMatrix::zeros(self.num_variables(), self.num_variables());
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            hess[(row as usize, col as usize)] += v;
        }

        let svd = na::SVD::new(hess, false, false);
        let s: &[Number] = svd.singular_values.data.as_slice();
        let cond_hess = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        dbg!(cond_hess);
    }

    /*
     * End of debugging functions
     */
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

    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.copy_from_slice(self.warm_start.primal_variables.as_slice());
        true
    }

    fn initial_bounds_multipliers(&self, z_l: &mut [Number], z_u: &mut [Number]) -> bool {
        z_l.copy_from_slice(self.warm_start.lower_bound_multipliers.as_slice());
        z_u.copy_from_slice(self.warm_start.upper_bound_multipliers.as_slice());
        true
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
            let prev_vel = self.prev_vel.borrow();
            let v: &[Number] = reinterpret_slice(prev_vel.as_slice());
            mp.add_energy_gradient(v, dx, grad_f);
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            scc.subtract_friction_force(grad_f);
        }

        true
    }

    fn objective_scaling(&self) -> f64 {
        1.0 //e-4
    }

    /// Scaling the variables: `x_scaling` is the pre-allocated slice of scales, one for
    /// each constraint.
    fn variable_scaling(&self, _x_scaling: &mut [Number]) -> bool {
        false
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

    fn initial_constraint_multipliers(&self, lambda: &mut [Number]) -> bool {
        // The constrained points may change between updating the warm start and using it here.
        assert_eq!(lambda.len(), self.warm_start.constraint_multipliers.len());
        lambda.copy_from_slice(self.warm_start.constraint_multipliers.as_slice());
        true
    }

    fn constraint(&self, dx: &[Number], g: &mut [Number]) -> bool {
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));

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

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        let mut i = 0; // counter

        let mut row_offset = 0;
        if let Some(ref vc) = self.volume_constraint {
            if let Ok(iter) = vc.constraint_jacobian_indices_iter() {
                for MatrixElementIndex { row, col } in iter {
                    rows[i] = row as ipopt::Index;
                    cols[i] = col as ipopt::Index;
                    i += 1;

                    row_offset = row_offset.max(row + 1);
                }
                assert_eq!(row_offset, 1); // volume constraint should be just one constraint
            }
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            use ipopt::BasicProblem;
            let nrows = scc.constraint_size();
            let ncols = self.num_variables();
            let mut jac = vec![vec![0; nrows]; ncols]; // col major

            if let Ok(iter) = scc.constraint_jacobian_indices_iter() {
                for MatrixElementIndex { row, col } in iter {
                    rows[i] = (row + row_offset) as ipopt::Index;
                    cols[i] = col as ipopt::Index;
                    jac[col][row] = 1;
                    i += 1;
                }

                //println!("jac = ");
                //for r in 0..nrows {
                //    for c in 0..ncols {
                //        if jac[c][r] == 1 {
                //            print!(".");
                //        } else {
                //            print!(" ");
                //        }
                //    }
                //    println!("");
                //}
                //println!("");
            }
        }

        true
    }

    fn constraint_jacobian_values(&self, dx: &[Number], vals: &mut [Number]) -> bool {
        let prev_pos = self.prev_pos.borrow();
        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

        let mut i = 0;

        if let Some(ref vc) = self.volume_constraint {
            let n = vc.constraint_jacobian_size();
            vc.constraint_jacobian_values(x, dx, &mut vals[i..i + n])
                .ok();
            i += n;
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let n = scc.constraint_jacobian_size();
            scc.constraint_jacobian_values(x, dx, &mut vals[i..i + n])
                .ok();
            //println!("jac g vals = {:?}", &vals[i..i+n]);
        }

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));
        //self.print_jacobian_svd(vals);

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
        if let Some(ref scc) = self.smooth_contact_constraint {
            num += scc.constraint_hessian_size();
        }
        num
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
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
                rows[i] = row as ipopt::Index;
                cols[i] = col as ipopt::Index;
                i += 1;
            }
        }
        if let Some(ref scc) = self.smooth_contact_constraint {
            if let Ok(iter) = scc.constraint_hessian_indices_iter() {
                for MatrixElementIndex { row, col } in iter {
                    rows[i] = row as ipopt::Index;
                    cols[i] = col as ipopt::Index;
                    i += 1;
                }
            }
        }

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

        if cfg!(debug_assertions) {
            // Initialize vals in debug builds.
            for v in vals.iter_mut() {
                *v = 0.0;
            }
        }

        let mut i = 0;
        let n = self.energy_model.energy_hessian_size();
        self.energy_model
            .energy_hessian_values(x, dx, &mut vals[i..i + n]);
        i += n;

        if let Some(ref mp) = self.momentum_potential {
            let n = mp.energy_hessian_size();
            let prev_vel = self.prev_vel.borrow();
            let v: &[Number] = reinterpret_slice(prev_vel.as_slice());
            mp.energy_hessian_values(v, dx, &mut vals[i..i + n]);
            i += n;
        }

        // Multiply energy hessian by objective factor.
        for v in vals[0..i].iter_mut() {
            *v *= obj_factor;
        }

        let mut coff = 0;

        if let Some(ref vc) = self.volume_constraint {
            let nc = vc.constraint_size();
            let nh = vc.constraint_hessian_size();
            if let Ok(()) =
                vc.constraint_hessian_values(x, dx, &lambda[coff..coff + nc], &mut vals[i..i + nh])
            {
                i += nh;
                coff += nc;
            }
        }

        if let Some(ref scc) = self.smooth_contact_constraint {
            let nc = scc.constraint_size();
            let nh = scc.constraint_hessian_size();
            if let Ok(()) =
                scc.constraint_hessian_values(x, dx, &lambda[coff..coff + nc], &mut vals[i..i + nh])
            {
                i += nh;
                coff += nc;
            }
        }

        assert_eq!(i, vals.len());
        assert_eq!(coff, lambda.len());
        //self.print_hessian_svd(vals);

        true
    }

    /// Scaling the constraint function: `g_scaling` is the pre-allocated slice of scales, one for
    /// each constraint.
    fn constraint_scaling(&self, _g_scaling: &mut [Number]) -> bool {
        // Here we scale down the constraint function to prevent the solver from aggressively
        // avoiding the infeasible region. This has been observed only for the contact constraint
        // so far.
        // TODO: Figure out a more robust way to do this. From the Ipopt docs we are recommended:
        //  "As a guideline, we suggest to scale the optimization problem (either directly in the
        //  original formulation, or after using scaling factors) so that all sensitivities, i.e.,
        //  all non-zero first partial derivatives, are typically of the order $ 0.1-10$."
        //for g in g_scaling.iter_mut() {
        //    *g = 1e-4;
        //}
        //true
        false
    }
}
