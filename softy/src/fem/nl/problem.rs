use std::cell::RefCell;

use crate::fem::object_data::*;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use num_traits::Zero;
use tensr::*;

use crate::attrib_defines::*;
use crate::constraint::*;
use crate::constraints::{
    point_contact::{MassData, PointContactConstraint},
    volume::VolumeConstraint,
    ContactConstraint,
};
use crate::contact::{ContactJacobian, ContactJacobianView, TripletContactJacobian};
use crate::energy::{EnergyGradient, EnergyHessian, EnergyHessianTopology};
use crate::energy_models::{elasticity::*, gravity::*, inertia::*};
use crate::matrix::*;
use crate::objects::*;
use crate::{PointCloud, TriMesh};

#[derive(Clone)]
pub struct Solution {
    /// This is the solution of the solve.
    pub u: Vec<f64>,
}

impl Solution {
    pub fn new(num_variables: usize) -> Solution {
        Solution {
            u: vec![0.0; num_variables],
        }
    }
    pub fn reset(&mut self) {
        self.u.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// A struct that keeps track of which objects are being affected by the contact
/// constraints.
#[derive(Debug)]
pub struct FrictionalContactConstraint {
    pub object_index: SourceObject,
    pub collider_index: SourceObject,
    pub constraint: RefCell<PointContactConstraint>,
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NLProblem<T> {
    /// A data model of the current problem.
    ///
    /// This includes all primal variables and any additional mesh data required
    /// for simulation.
    pub object_data: ObjectData<T>,
    /// One way contact constraints between a pair of objects.
    pub frictional_contacts: Vec<FrictionalContactConstraint>,
    /// Constraint on the total volume.
    pub volume_constraints: Vec<(usize, RefCell<VolumeConstraint>)>,
    /// Gravitational potential energy.
    pub gravity: [f64; 3],
    /// The time step defines the amount of time elapsed between steps (calls to `advance`).
    /// If the time step is zero, objects don't exhibit inertia.
    pub time_step: f64,
    /// Counts the number of iterations.
    pub iterations: usize,
    /// The initial error measure used for relative stopping conditions.
    pub initial_residual_error: f64,
    /// Global iteration counter used for debug output.
    pub iter_counter: RefCell<usize>,
    /// The maximum size (diameter) of a simulated object (deformable or rigid).
    pub max_size: f64,
    /// The scale of the energy gradient intended to be used for rescaling the objective gradient.
    pub force_scale: f64,
}

impl<T: Real64> NLProblem<T> {
    pub fn variable_scale(&self) -> f64 {
        // This scaling makes variables unitless.
        utils::approx_power_of_two64(0.1 * self.max_size / self.time_step())
    }

    fn impulse_inv_scale(&self) -> f64 {
        utils::approx_power_of_two64(100.0 / (self.time_step() * self.force_scale))
    }

    fn volume_constraint_scale(&self) -> f64 {
        1.0
    }

    fn contact_constraint_scale(&self) -> f64 {
        1.0
    }

    fn time_step(&self) -> f64 {
        if self.is_static() {
            1.0
        } else {
            self.time_step
        }
    }

    /// Check if this problem represents a static simulation.
    ///
    /// In this case
    /// inertia is ignored and velocities are treated as displacements.
    fn is_static(&self) -> bool {
        self.time_step == 0.0
    }

    /// Produce an iterator over the given slice of scaled variables.
    pub fn scaled_variables_iter<'a>(&self, unscaled_var: &'a [T]) -> impl Iterator<Item = T> + 'a {
        ObjectData::scaled_variables_iter(unscaled_var, T::from(self.variable_scale()).unwrap())
    }

    /// Get the current iteration count and reset it.
    pub fn pop_iteration_count(&mut self) -> usize {
        let iter = self.iterations;
        // Reset count
        self.iterations = 0;
        iter
    }

    /// Get the minimum contact radius among all contact problems.
    ///
    /// If there are no contacts, simply return `None`.
    pub fn min_contact_radius(&self) -> Option<f64> {
        self.frictional_contacts
            .iter()
            .map(|fc| fc.constraint.borrow().contact_radius())
            .min_by(|a, b| a.partial_cmp(b).expect("Detected NaN contact radius"))
    }

    /// Save an intermediate state of the solve. This is used for debugging.
    #[allow(dead_code)]
    pub fn save_intermediate(&mut self, uv: &[T], step: usize) {
        self.update_current_velocity(uv);
        self.compute_step();
        let ws = self.object_data.workspace.borrow();
        let mut solids = self.object_data.solids.clone();
        let mut shells = self.object_data.shells.clone();
        ObjectData::update_simulated_meshes_with(
            &mut solids,
            &mut shells,
            ws.dof.view(),
            ws.vtx.view().map_storage(|vtx| vtx.state),
        );
        geo::io::save_tetmesh(
            &solids[0].tetmesh,
            &std::path::PathBuf::from(format!("./out/predictor_{}.vtk", step)),
        )
        .unwrap();
    }

    /// Update the solid meshes with the given points.
    pub fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        let new_pos = Chunked3::from_array_slice(pts.vertex_positions());
        self.object_data
            .update_solid_vertices(new_pos.view(), self.time_step())
    }

    /// Update the shell meshes with the given points.
    pub fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        let new_pos = Chunked3::from_array_slice(pts.vertex_positions());
        self.object_data
            .update_shell_vertices(new_pos.view(), self.time_step())
    }

    /// Update the underlying scaled velocity data with the given unscaled velocity.
    ///
    /// If `full` is true, then all vertex velocities are updated including ones for
    /// rigid bodies. Note that `full` is only usually necessary when computing contact
    /// constraints.
    pub fn update_current_velocity(&self, uv: &[T]) {
        self.object_data
            .update_workspace_velocity(uv, self.variable_scale());
    }

    /// Compute the set of currently active constraints into the given `Chunked` `Vec`.
    pub fn compute_active_constraint_set(&self, active_set: &mut Chunked<Vec<usize>>) {
        // Disassemble chunked collection.
        let (offsets, active_set) = active_set.as_inner_mut();

        for i in 0..self.volume_constraints.len() {
            active_set.push(i);
            offsets.push(active_set.len());
        }

        let mut offset = active_set.len();
        for FrictionalContactConstraint { ref constraint, .. } in self.frictional_contacts.iter() {
            let fc_active_constraints = constraint.borrow().active_constraint_indices();
            for c in fc_active_constraints.into_iter() {
                active_set.push(c + offset);
            }
            offset += constraint.borrow().num_potential_contacts();
            offsets.push(active_set.len());
        }
    }

    /// Check if all contact constraints are linear.
    pub fn all_contacts_linear(&self) -> bool {
        self.frictional_contacts
            .iter()
            .all(|contact_constraint| contact_constraint.constraint.borrow().is_linear())
    }

    pub(crate) fn is_rigid(&self, src_idx: SourceObject) -> bool {
        if let SourceObject::Shell(idx) = src_idx {
            if let ShellData::Rigid { .. } = self.object_data.shells[idx].data {
                return true;
            }
        }
        false
    }

    pub fn has_rigid(&self) -> bool {
        for fc in self.frictional_contacts.iter() {
            if self.is_rigid(fc.collider_index) | self.is_rigid(fc.object_index) {
                return true;
            }
        }
        false
    }

    /// Get the set of currently active constraints.
    pub fn active_constraint_set(&self) -> Chunked<Vec<usize>> {
        let mut active_set = Chunked::new();
        self.compute_active_constraint_set(&mut active_set);
        active_set
    }

    #[allow(dead_code)]
    pub fn clear_friction_impulses(&mut self) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint
                .borrow_mut()
                .clear_frictional_contact_impulse();
        }
    }

    //    /// Restore the constraint set.
    //    pub fn reset_constraint_set(&mut self) -> bool {
    //        let updated = self.update_constraint_set(None);
    //
    //        // Add forwarded friction impulse. This is the applied force for the next time step.
    //        let friction_impulse = self.friction_impulse();
    //        for (idx, solid) in self.object_data.solids.iter_mut().enumerate() {
    //            solid
    //                .tetmesh
    //                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                    "forward_friction",
    //                    friction_impulse.view().at(0).at(idx).view().into(),
    //                )
    //                .ok();
    //        }
    //        for (idx, shell) in self.object_data.shells.iter_mut().enumerate() {
    //            shell
    //                .trimesh
    //                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                    "forward_friction",
    //                    friction_impulse.view().at(1).at(idx).view().into(),
    //                )
    //                .ok();
    //        }
    //
    //        updated
    //    }
    //
    //    /// Check if the given constraint set is the same as the current one.
    //    pub fn is_same_as_constraint_set(&self, other_set: ChunkedView<&[usize]>) -> bool {
    //        let cur_set = self.active_constraint_set().into_storage();
    //        let other_set = other_set.into_storage();
    //        cur_set.len() == other_set.len()
    //            && cur_set
    //                .into_iter()
    //                .zip(other_set.iter())
    //                .all(|(cur, &other)| cur == other)
    //    }
    //
    //    /// Update all stateful constraints with the most recent data.
    //    ///
    //    /// Return an estimate if any constraints have changed, though this estimate may have false
    //    /// negatives.
    //    pub fn update_constraint_set(&mut self, solution: Option<Solution>) -> bool {
    //        let mut changed = false; // Report if anything has changed to the caller.
    //
    //        let scale = self.variable_scale();
    //        let time_step = self.time_step();
    //
    //        let NLProblem {
    //            ref mut frictional_contacts,
    //            ref object_data,
    //            ..
    //        } = *self;
    //
    //        // Update positions with the given solution (if any).
    //        let solution_is_some = solution.is_some();
    //        if let Some(sol) = solution {
    //            object_data.update_workspace_velocity(sol.primal_variables, scale);
    //            object_data.integrate_step(time_step);
    //            let mut ws = object_data.workspace.borrow_mut();
    //            let WorkspaceData { dof, vtx, .. } = &mut *ws;
    //            ObjectData::sync_pos(
    //                &object_data.shells,
    //                dof.view().map_storage(|dof| dof.q),
    //                vtx.view_mut().map_storage(|vtx| vtx.state.pos),
    //            );
    //        }
    //
    //        let ws = object_data.workspace.borrow_mut();
    //
    //        for FrictionalContactConstraint {
    //            object_index,
    //            collider_index,
    //            constraint,
    //        } in frictional_contacts.iter_mut()
    //        {
    //            if solution_is_some {
    //                let q = ws.dof.view().map_storage(|dof| dof.q);
    //                let pos = ws.vtx.view().map_storage(|vtx| vtx.state.pos);
    //                let object_pos = object_data.next_pos(q, pos, *object_index);
    //                let collider_pos = object_data.next_pos(q, pos, *collider_index);
    //                changed |= constraint
    //                    .borrow_mut()
    //                    .update_neighbors(object_pos.view(), collider_pos.view());
    //            } else {
    //                let object_pos = object_data.cur_pos(*object_index);
    //                let collider_pos = object_data.cur_pos(*collider_index);
    //                changed |= constraint
    //                    .borrow_mut()
    //                    .update_neighbors(object_pos.view(), collider_pos.view());
    //            }
    //        }
    //
    //        changed
    //    }

    //pub fn apply_frictional_contact_impulse(&mut self) {
    //    let NLProblem {
    //        frictional_contacts,
    //        object_data,
    //        ..
    //    } = self;

    //    let mut ws = object_data.workspace.borrow_mut();
    //    let WorkspaceData { dof, vtx, .. } = &mut *ws;

    //    for fc in frictional_contacts.iter() {
    //        let [obj_idx, coll_idx] = [fc.object_index, fc.collider_index];
    //        let fc = fc.constraint.borrow();

    //        let dq = dof.view_mut().map_storage(|dof| dof.dq);
    //        let vtx_vel = vtx.view_mut().map_storage(|vtx| vtx.state.vel);
    //        let mut obj_vel = object_data.mesh_vertex_subset(dq, vtx_vel, obj_idx);
    //        fc.add_mass_weighted_frictional_contact_impulse_to_object(obj_vel.view_mut());

    //        let dq = dof.view_mut().map_storage(|dof| dof.dq);
    //        let vtx_vel = vtx.view_mut().map_storage(|vtx| vtx.state.vel);
    //        let mut coll_vel = object_data.mesh_vertex_subset(dq, vtx_vel, coll_idx);
    //        fc.add_mass_weighted_frictional_contact_impulse_to_collider(coll_vel.view_mut());
    //    }
    //}

    /// Commit velocity by advancing the internal state by the given unscaled velocity `uv`.
    ///
    /// If `and_velocity` is `false`, then only positions are advanced, and velocities are reset.
    /// This emulates a critically damped, or quasi-static simulation.
    pub fn advance(&mut self, uv: &[T], and_velocity: bool) {
        self.update_current_velocity(uv);
        {
            //self.apply_frictional_contact_impulse();

            self.object_data.workspace.borrow();
            self.compute_step();
        }

        self.object_data.advance(and_velocity);
    }

    /// Advance object data one step back.
    pub fn revert_prev_step(&mut self) {
        self.object_data.revert_prev_step();
        // Clear any frictional impulsesl
        for fc in self.frictional_contacts.iter() {
            if let Some(friction_data) = fc.constraint.borrow_mut().frictional_contact_mut() {
                friction_data
                    .collider_impulse
                    .source_iter_mut()
                    .for_each(|(x, y)| {
                        *x = [0.0; 3];
                        *y = [0.0; 3]
                    });
                friction_data.object_impulse.iter_mut().for_each(|(x, y)| {
                    *x = [0.0; 3];
                    *y = [0.0; 3]
                });
            }
        }
    }

    pub fn update_max_step(&mut self, step: f64) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint.borrow_mut().update_max_step(step);
        }
    }
    pub fn update_radius_multiplier(&mut self, rad_mult: f64) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint
                .borrow_mut()
                .update_radius_multiplier(rad_mult);
        }
    }

    /// A convenience function to integrate the given velocity by the internal time step.
    ///
    /// For implicit integration this boils down to a simple multiply by the time step.
    pub fn compute_step(&self) {
        self.object_data.integrate_step(self.time_step());
    }

    /// Convert a given array of contact forces to impulses.
    fn contact_impulse_magnitudes(forces: &[f64], scale: f64) -> Vec<f64> {
        forces.iter().map(|&cf| cf * scale).collect()
    }

    /// Construct the global contact Jacobian matrix.
    ///
    /// The contact Jacobian consists of blocks represeting contacts between pairs of objects.
    /// Each block represets a particular coupling. Given two objects A and B, there can be two
    /// types of coupling:
    /// A is an implicit surface in contact with vertices of B and
    /// B is an implicit surface in contact with vertices of A.
    /// Both of these are valid for solids since they represent a volume, while cloth can only
    /// collide against implicit surfaces (for now) and not vice versa.
    pub fn construct_contact_jacobian(
        &self,
        solution: &[T],
        constraint_values: &[f64],
        // TODO: Move to GlobalContactJacobian, which combines these two outputs.
    ) -> (ContactJacobian, Chunked<Offsets<Vec<usize>>>) {
        let NLProblem {
            ref frictional_contacts,
            ref volume_constraints,
            ref object_data,
            ..
        } = *self;

        let mut jac_triplets = TripletContactJacobian::new();

        if frictional_contacts.is_empty() {
            return (
                jac_triplets.into(),
                Chunked::from_offsets(vec![0], Offsets::new(vec![])),
            );
        }

        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();

        let dof_view = object_data.dof.view();

        // A set of offsets indexing the beginnings of surface vertex slices for each object.
        // This is different than their generalized coordinate offsets.
        let mut surface_object_offsets = vec![0; 3];
        let mut surface_vertex_offsets = vec![0; dof_view.data().len() + 1];
        let mut surface_vertex_offset = 0;
        let mut idx = 1;

        for object_dofs in dof_view.iter() {
            for (solid_idx, _) in object_dofs.iter().enumerate() {
                surface_vertex_offsets[idx] = surface_vertex_offset;
                surface_vertex_offset += object_data.solids[solid_idx]
                    .entire_surface()
                    .trimesh
                    .num_vertices();
                idx += 1;
            }
            surface_object_offsets[SOLIDS_INDEX + 1] = surface_vertex_offset;
            for (shell_idx, _) in object_dofs.iter().enumerate() {
                surface_vertex_offsets[idx] = surface_vertex_offset;
                surface_vertex_offset += object_data.shells[shell_idx].trimesh.num_vertices();
                idx += 1;
            }
            surface_object_offsets[SHELLS_INDEX + 1] = surface_vertex_offset;
        }

        // Separate offsets by type of mesh for easier access.
        let surface_vertex_offsets =
            Chunked::from_offsets(surface_object_offsets, Offsets::new(surface_vertex_offsets));
        let surface_vertex_offsets_view = surface_vertex_offsets.view();

        let mut contact_offset = 0;

        //for fc in frictional_contacts.iter() {
        //    let n = fc.constraint.borrow().constraint_size();
        //    let constraint_offset = contact_offset + volume_constraints.len();

        //    // Get the normal component of the contact impulse.
        //    let contact_impulse = Self::contact_impulse_magnitudes(
        //        &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
        //        multiplier_impulse_scale,
        //    );

        //    log::debug!(
        //        "Maximum contact impulse: {}",
        //        crate::inf_norm(contact_impulse.iter().cloned())
        //    );

        //    let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

        //    let (_, active_contact_indices, _) = fc
        //        .constraint
        //        .borrow()
        //        .in_contact_indices(&contact_impulse, potential_values);

        //    let object_vertex_offset = match fc.object_index {
        //        SourceObject::Solid(i, _) => *surface_vertex_offsets_view.at(SOLIDS_INDEX).at(i),
        //        SourceObject::Shell(i) => *surface_vertex_offsets_view.at(SHELLS_INDEX).at(i),
        //    };
        //    let collider_vertex_offset = match fc.collider_index {
        //        SourceObject::Solid(i, _) => *surface_vertex_offsets_view.at(SOLIDS_INDEX).at(i),
        //        SourceObject::Shell(i) => *surface_vertex_offsets_view.at(SHELLS_INDEX).at(i),
        //    };

        //    fc.constraint.borrow().append_contact_jacobian_triplets(
        //        &mut jac_triplets,
        //        &active_contact_indices,
        //        contact_offset,
        //        object_vertex_offset,
        //        collider_vertex_offset,
        //    );

        //    contact_offset += n;
        //}

        jac_triplets.num_rows = contact_offset;
        jac_triplets.num_cols = surface_vertex_offsets_view.data().last_offset();

        let jac: ContactJacobian = jac_triplets.into();
        (
            jac.into_tensor()
                .pruned(|_, _, block| !block.is_zero())
                .into_data(),
            surface_vertex_offsets,
        )
    }

    pub fn construct_effective_mass_inv(
        &self,
        solution: &[T],
        constraint_values: &[f64],
        jac: ContactJacobianView,
        surface_vertex_offsets: ChunkedView<Offsets<&[usize]>>,
    ) -> Tensor![f64; S S 3 3] {
        let NLProblem {
            ref frictional_contacts,
            ref volume_constraints,
            ref object_data,
            ..
        } = *self;

        // TODO: improve this computation by avoiding intermediate mass matrix computation.

        // Size of the effective mass matrix in each dimension.
        let size = jac.into_tensor().num_cols();

        let mut blocks = Vec::with_capacity(size);
        let mut block_indices = Vec::with_capacity(size);

        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();

        let mut contact_offset = 0;

        //for fc in frictional_contacts.iter() {
        //    let FrictionalContactConstraint {
        //        object_index,
        //        collider_index,
        //        constraint,
        //    } = fc;

        //    let constraint = constraint.borrow();

        //    let n = constraint.constraint_size();
        //    let constraint_offset = contact_offset + volume_constraints.len();

        //    // Get the normal component of the contact impulse.
        //    let contact_impulse = Self::contact_impulse_magnitudes(
        //        &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
        //        multiplier_impulse_scale,
        //    );

        //    log::debug!(
        //        "Maximum contact impulse: {}",
        //        crate::inf_norm(contact_impulse.iter().cloned())
        //    );

        //    let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

        //    let (_, active_contact_indices, _) =
        //        constraint.in_contact_indices(&contact_impulse, potential_values);

        //    let PointContactConstraint {
        //        implicit_surface: ref surf,
        //        ref contact_points,
        //        ref object_mass_data,
        //        ref collider_mass_data,
        //        ..
        //    } = *constraint;

        //    let start_object = match object_index {
        //        SourceObject::Solid(i, _) => *surface_vertex_offsets.at(SOLIDS_INDEX).at(i),
        //        SourceObject::Shell(i) => *surface_vertex_offsets.at(SHELLS_INDEX).at(i),
        //    };
        //    let start_collider = match collider_index {
        //        SourceObject::Solid(i, _) => *surface_vertex_offsets.at(SOLIDS_INDEX).at(i),
        //        SourceObject::Shell(i) => *surface_vertex_offsets.at(SHELLS_INDEX).at(i),
        //    };

        //    let surf_neigh_indices = surf.neighborhood_vertex_indices();

        //    match object_mass_data {
        //        MassData::Sparse(vertex_masses) => {
        //            blocks.extend(
        //                surf_neigh_indices
        //                    .iter()
        //                    .map(|&i| Matrix3::diag(&vertex_masses[i][..]).into_data()),
        //            );
        //            block_indices.extend(
        //                surf_neigh_indices
        //                    .iter()
        //                    .map(|&i| (start_object + i, start_object + i)),
        //            );
        //        }
        //        MassData::Dense(mass, inertia) => {
        //            let [translation, rotation] = object_data
        //                .rigid_motion(*object_index)
        //                .expect("Object type doesn't match precomputed mass data");
        //            let eff_mass_inv = TriMeshShell::rigid_effective_mass_inv(
        //                *mass,
        //                translation.into_tensor(),
        //                rotation.into_tensor(),
        //                *inertia,
        //                Select::new(
        //                    surf_neigh_indices.as_slice(),
        //                    Chunked3::from_array_slice(surf.surface_vertex_positions()),
        //                ),
        //            );

        //            blocks.extend(
        //                eff_mass_inv
        //                    .iter()
        //                    .flat_map(|row| row.into_iter().map(|block| *block.into_arrays())),
        //            );
        //            block_indices.extend(surf_neigh_indices.iter().flat_map(|&row_idx| {
        //                surf_neigh_indices
        //                    .iter()
        //                    .map(move |&col_idx| (row_idx, col_idx))
        //            }));
        //        }
        //        MassData::Zero => {}
        //    }

        //    match collider_mass_data {
        //        MassData::Sparse(vertex_masses) => {
        //            blocks.extend(
        //                surf_neigh_indices
        //                    .iter()
        //                    .map(|&i| Matrix3::diag(&vertex_masses[i][..]).into_data()),
        //            );
        //            block_indices.extend(
        //                surf_neigh_indices
        //                    .iter()
        //                    .map(|&i| (start_object + i, start_object + i)),
        //            );
        //        }
        //        MassData::Dense(mass, inertia) => {
        //            let [translation, rotation] = object_data
        //                .rigid_motion(fc.collider_index)
        //                .expect("Object type doesn't match precomputed mass data");
        //            let eff_mass_inv = TriMeshShell::rigid_effective_mass_inv(
        //                *mass,
        //                translation.into_tensor(),
        //                rotation.into_tensor(),
        //                *inertia,
        //                Select::new(&active_contact_indices, contact_points.view()),
        //            );
        //            blocks.extend(
        //                eff_mass_inv
        //                    .iter()
        //                    .flat_map(|row| row.into_iter().map(|block| *block.into_arrays())),
        //            );
        //            block_indices.extend(active_contact_indices.iter().flat_map(|&row_idx| {
        //                active_contact_indices
        //                    .iter()
        //                    .map(move |&col_idx| (row_idx, col_idx))
        //            }));
        //        }
        //        MassData::Zero => {}
        //    }

        //    contact_offset += n;
        //}

        let blocks = Chunked3::from_flat(Chunked3::from_array_vec(
            Chunked3::from_array_vec(blocks).into_storage(),
        ));

        let mass_inv = SSBlockMatrix3::<f64>::from_index_iter_and_data(
            block_indices.into_iter(),
            size,
            size,
            blocks,
        );

        let jac_mass = jac.view().into_tensor() * mass_inv.view().transpose();
        (jac_mass.view() * jac.view().into_tensor().transpose()).into_data()
    }

    /// Returns true if all friction solves have been completed/converged.
    ///
    /// This should be the case if and only if all elements in `friction_steps`
    /// are zero, which makes the return value simply a convenience.
    pub fn update_friction_impulse_global(
        &mut self,
        solution: &[T],
        constraint_values: &[f64],
        friction_steps: &mut [u32],
    ) -> bool {
        if self.frictional_contacts.is_empty() {
            return true;
        }

        self.update_current_velocity(solution);
        let q_cur = self.object_data.dof.view().map_storage(|dof| dof.cur.q);
        let mut ws = self.object_data.workspace.borrow_mut();
        let WorkspaceData { dof, vtx } = &mut *ws;
        ObjectData::sync_vel(
            &self.object_data.shells,
            dof.view().map_storage(|dof| dof.dq),
            q_cur,
            vtx.view_mut().map_storage(|vtx| vtx.state.vel),
        );

        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();

        let (jac, surface_vertex_offsets) =
            self.construct_contact_jacobian(solution, constraint_values);

        let eff_mass_inv = self.construct_effective_mass_inv(
            solution,
            constraint_values,
            jac.view(),
            surface_vertex_offsets.view(),
        );

        let NLProblem {
            ref mut frictional_contacts,
            ref volume_constraints,
            ref object_data,
            ..
        } = *self;

        let mut is_finished = true;

        let mut constraint_offset = volume_constraints.len();

        // Update normals

        //        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
        //            let obj_cur_pos = object_data.cur_pos(fc.object_index);
        //            let col_cur_pos = object_data.cur_pos(fc.collider_index);
        //            let dq_next = dof.view().map_storage(|dof| dof.dq);
        //            let vtx_vel_next = vtx.view().map_storage(|vtx| vtx.state.vel);
        //            let obj_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.object_index);
        //            let col_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.collider_index);
        //
        //            let n = fc.constraint.borrow().constraint_size();
        //            let contact_impulse = Self::contact_impulse_magnitudes(
        //                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
        //                multiplier_impulse_scale,
        //            );
        //
        //            log::debug!(
        //                "Maximum contact impulse: {}",
        //                crate::inf_norm(contact_impulse.iter().cloned())
        //            );
        //            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];
        //
        //            // TODO: Refactor this low level code out. There needs to be a mechanism to pass rigid
        //            // motion data to the constraints since rigid bodies have a special effective mass.
        //            let q_cur = q_cur.at(SHELLS_INDEX);
        //            let rigid_motion = [
        //                object_data.rigid_motion(fc.object_index),
        //                object_data.rigid_motion(fc.collider_index),
        //            ];
        //            friction_steps[fc_idx] = fc
        //                .constraint
        //                .borrow_mut()
        //                .update_frictional_contact_impulse(
        //                    &contact_impulse,
        //                    [obj_cur_pos.view(), col_cur_pos.view()],
        //                    [obj_vel.view(), col_vel.view()],
        //                    rigid_motion,
        //                    potential_values,
        //                    friction_steps[fc_idx],
        //                );
        //
        //            is_finished &= friction_steps[fc_idx] == 0;
        //            constraint_offset += n;
        //        }

        //let normals = self.contact_normals();

        is_finished
    }

    /// Returns true if all friction solves have been completed/converged.
    ///
    /// This should be the case if and only if all elements in `friction_steps`
    /// are zero, which makes the return value simply a convenience.
    pub fn update_friction_impulse(
        &mut self,
        solution: &[T],
        constraint_values: &[f64],
        friction_steps: &mut [u32],
    ) -> bool {
        if self.frictional_contacts.is_empty() {
            return true;
        }

        self.update_current_velocity(solution);
        let q_cur = self.object_data.dof.view().map_storage(|dof| dof.cur.q);
        let mut ws = self.object_data.workspace.borrow_mut();
        let WorkspaceData { dof, vtx } = &mut *ws;
        ObjectData::sync_vel(
            &self.object_data.shells,
            dof.view().map_storage(|dof| dof.dq),
            q_cur,
            vtx.view_mut().map_storage(|vtx| vtx.state.vel),
        );

        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
        let NLProblem {
            ref mut frictional_contacts,
            ref volume_constraints,
            ref object_data,
            ..
        } = *self;

        let mut is_finished = true;

        let mut constraint_offset = volume_constraints.len();

        // TODO: This is not the right way to compute friction forces since it decouples each pair
        //       of colliding objects. We should construct a global jacobian matrix instead to
        //       resolve all friction forces simultaneously. We may use the block nature of
        //       contacts to construct a blockwise sparse matrix here.

        //        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
        //            let obj_cur_pos = object_data.cur_pos(fc.object_index);
        //            let col_cur_pos = object_data.cur_pos(fc.collider_index);
        //            let dq_next = dof.view().map_storage(|dof| dof.dq);
        //            let vtx_vel_next = vtx.view().map_storage(|vtx| vtx.state.vel);
        //            let obj_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.object_index);
        //            let col_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.collider_index);
        //
        //            let n = fc.constraint.borrow().constraint_size();
        //            let contact_impulse = Self::contact_impulse_magnitudes(
        //                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
        //                multiplier_impulse_scale,
        //            );
        //
        //            log::debug!(
        //                "Maximum contact impulse: {}",
        //                crate::inf_norm(contact_impulse.iter().cloned())
        //            );
        //            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];
        //
        //            let rigid_motion = [
        //                object_data.rigid_motion(fc.object_index),
        //                object_data.rigid_motion(fc.collider_index),
        //            ];
        //            friction_steps[fc_idx] = fc
        //                .constraint
        //                .borrow_mut()
        //                .update_frictional_contact_impulse(
        //                    &contact_impulse,
        //                    [obj_cur_pos.view(), col_cur_pos.view()],
        //                    [obj_vel.view(), col_vel.view()],
        //                    rigid_motion,
        //                    potential_values,
        //                    friction_steps[fc_idx],
        //                );
        //
        //            is_finished &= friction_steps[fc_idx] == 0;
        //            constraint_offset += n;
        //        }

        is_finished
    }

    //    pub fn contact_potential(&self) -> VertexData<Vec<f64>> {
    //        let NLProblem {
    //            object_data,
    //            frictional_contacts,
    //            ..
    //        } = self;
    //
    //        let mut pot = object_data.build_vertex_data();
    //
    //        let mut workspace_pot = Vec::new();
    //
    //        for fc in frictional_contacts.iter() {
    //            let obj_x0 = object_data.cur_pos(fc.object_index);
    //            let coll_x0 = object_data.cur_pos(fc.collider_index);
    //
    //            let active_constraint_indices = fc.constraint.borrow().active_constraint_indices();
    //            workspace_pot.clear();
    //            workspace_pot.resize(active_constraint_indices.len(), 0.0);
    //
    //            fc.constraint.borrow_mut().constraint(
    //                [obj_x0.view(), coll_x0.view()],
    //                workspace_pot.as_mut_slice(),
    //            );
    //
    //            let mut pot_view_mut =
    //                object_data.mesh_vertex_subset(pot.view_mut(), None, fc.collider_index);
    //            for (&aci, &pot) in active_constraint_indices.iter().zip(workspace_pot.iter()) {
    //                pot_view_mut[aci] += pot;
    //            }
    //        }
    //
    //        pot
    //    }

    /// Return the stacked friction corrector impulses: one for each vertex.
    pub fn friction_corrector_impulse(&self) -> VertexData3<Vec<f64>> {
        let NLProblem {
            object_data,
            frictional_contacts,
            ..
        } = self;

        // Create a chunked collection for the output. This essentially combines
        // the structure in `pos`, which involved vertices that are not degrees
        // of freedom and `prev_x` which involves vertices that ARE degrees of
        // freedom.
        let mut impulse = object_data.build_vertex_data3();

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in frictional_contacts.iter() {
            obj_imp.clear();
            obj_imp.resize(
                object_data.mesh_surface_vertex_count(fc.object_index),
                [0.0; 3],
            );

            coll_imp.clear();
            coll_imp.resize(
                object_data.mesh_surface_vertex_count(fc.collider_index),
                [0.0; 3],
            );

            let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
            let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());

            fc.constraint.borrow().add_friction_corrector_impulse(
                [
                    Subset::all(obj_imp.view_mut()),
                    Subset::all(coll_imp.view_mut()),
                ],
                1.0,
            );

            let mut imp = object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp.iter()) {
                *imp.as_mut_tensor() += *obj_imp.as_tensor();
            }

            let mut imp =
                object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp.iter()) {
                *imp.as_mut_tensor() += *coll_imp.as_tensor();
            }
        }
        impulse
    }

    /// Return the stacked friction impulses: one for each vertex.
    pub fn friction_impulse(&self) -> VertexData3<Vec<f64>> {
        let NLProblem {
            object_data,
            frictional_contacts,
            ..
        } = self;

        // Create a chunked collection for the output. This essentially combines
        // the structure in `pos`, which involved vertices that are not degrees
        // of freedom and `prev_x` which involves vertices that ARE degrees of
        // freedom.
        let mut impulse = object_data.build_vertex_data3();

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in frictional_contacts.iter() {
            obj_imp.clear();
            obj_imp.resize(
                object_data.mesh_surface_vertex_count(fc.object_index),
                [0.0; 3],
            );

            coll_imp.clear();
            coll_imp.resize(
                object_data.mesh_surface_vertex_count(fc.collider_index),
                [0.0; 3],
            );

            let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
            fc.constraint
                .borrow()
                .add_friction_impulse_to_object(Subset::all(obj_imp.view_mut()), 1.0);

            let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());
            fc.constraint
                .borrow()
                .add_friction_impulse_to_collider(Subset::all(coll_imp.view_mut()), 1.0);

            let mut imp = object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp.iter()) {
                *imp.as_mut_tensor() += *obj_imp.as_tensor();
            }

            let mut imp =
                object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp.iter()) {
                *imp.as_mut_tensor() += *coll_imp.as_tensor();
            }
        }
        impulse
    }

    pub fn collider_normals(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
        let NLProblem {
            object_data,
            frictional_contacts,
            ..
        } = self;

        let mut normals = object_data.build_vertex_data3();

        let mut coll_nml = Vec::new();

        for fc in frictional_contacts.iter() {
            coll_nml.clear();
            coll_nml.resize(
                object_data.mesh_surface_vertex_count(fc.collider_index),
                [0.0; 3],
            );

            let mut coll_nml = Chunked3::from_array_slice_mut(coll_nml.as_mut_slice());

            fc.constraint
                .borrow_mut()
                .collider_contact_normals(coll_nml.view_mut());

            let coll_nml_view = coll_nml.view();

            let mut nml =
                object_data.mesh_vertex_subset(normals.view_mut(), None, fc.collider_index);
            for (nml, coll_nml) in nml.iter_mut().zip(coll_nml_view.iter()) {
                *nml.as_mut_tensor() += *coll_nml.as_tensor();
            }
        }
        normals
    }

    /// Return the stacked contact impulses: one for each vertex.
    //pub fn contact_impulse(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
    //    let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
    //    let NLProblem {
    //        object_data,
    //        frictional_contacts,
    //        volume_constraints,
    //        ..
    //    } = self;

    //    let mut impulse = object_data.build_vertex_data3();

    //    let mut offset = volume_constraints.len();

    //    let mut obj_imp = Vec::new();
    //    let mut coll_imp = Vec::new();

    //    for fc in frictional_contacts.iter() {
    //        // Get contact force from the warm start.
    //        let n = fc.constraint.borrow().constraint_size();
    //        let contact_impulse = Self::contact_impulse_magnitudes(
    //            &warm_start.constraint_multipliers[offset..offset + n],
    //            multiplier_impulse_scale,
    //        );

    //        offset += n;

    //        obj_imp.clear();
    //        obj_imp.resize(
    //            object_data.mesh_surface_vertex_count(fc.object_index),
    //            [0.0; 3],
    //        );

    //        coll_imp.clear();
    //        coll_imp.resize(
    //            object_data.mesh_surface_vertex_count(fc.collider_index),
    //            [0.0; 3],
    //        );

    //        let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
    //        let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());

    //        let obj_x0 = object_data.cur_pos(fc.object_index);
    //        let coll_x0 = object_data.cur_pos(fc.collider_index);

    //        fc.constraint.borrow_mut().add_contact_impulse(
    //            [obj_x0.view(), coll_x0.view()],
    //            &contact_impulse,
    //            [obj_imp.view_mut(), coll_imp.view_mut()],
    //        );

    //        let obj_imp_view = obj_imp.view();
    //        let coll_imp_view = coll_imp.view();

    //        let mut imp = object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
    //        for (imp, obj_imp) in imp.iter_mut().zip(obj_imp_view.iter()) {
    //            *imp.as_mut_tensor() += *obj_imp.as_tensor();
    //        }

    //        let mut imp =
    //            object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
    //        for (imp, coll_imp) in imp.iter_mut().zip(coll_imp_view.iter()) {
    //            *imp.as_mut_tensor() += *coll_imp.as_tensor();
    //        }
    //    }
    //    assert_eq!(offset, warm_start.constraint_multipliers.len());
    //    impulse
    //}

    fn surface_vertex_areas(&self) -> VertexData<Vec<f64>> {
        use geo::ops::Area;
        use geo::prim::Triangle;
        let mut vertex_areas = self.object_data.build_vertex_data();
        for (idx, solid) in self.object_data.solids.iter().enumerate() {
            let TetMeshSurface { trimesh, indices } = &solid.entire_surface();
            for face in trimesh.face_iter() {
                let area_third =
                    Triangle::from_indexed_slice(face, trimesh.vertex_positions()).area() / 3.0;
                *vertex_areas
                    .view_mut()
                    .isolate(SOLIDS_INDEX)
                    .isolate(idx)
                    .isolate(indices[face[0]]) += area_third;
                *vertex_areas
                    .view_mut()
                    .isolate(SOLIDS_INDEX)
                    .isolate(idx)
                    .isolate(indices[face[1]]) += area_third;
                *vertex_areas
                    .view_mut()
                    .isolate(SOLIDS_INDEX)
                    .isolate(idx)
                    .isolate(indices[face[2]]) += area_third;
            }
        }
        for (idx, shell) in self.object_data.shells.iter().enumerate() {
            for face in shell.trimesh.face_iter() {
                let area_third =
                    Triangle::from_indexed_slice(face, shell.trimesh.vertex_positions()).area()
                        / 3.0;
                *vertex_areas
                    .view_mut()
                    .isolate(SHELLS_INDEX)
                    .isolate(idx)
                    .isolate(face[0]) += area_third;
                *vertex_areas
                    .view_mut()
                    .isolate(SHELLS_INDEX)
                    .isolate(idx)
                    .isolate(face[1]) += area_third;
                *vertex_areas
                    .view_mut()
                    .isolate(SHELLS_INDEX)
                    .isolate(idx)
                    .isolate(face[2]) += area_third;
            }
        }
        vertex_areas
    }

    fn pressure(&self, contact_impulse: VertexView3<&[f64]>) -> VertexData<Vec<f64>> {
        let mut pressure = self.object_data.build_vertex_data();
        let vertex_areas = self.surface_vertex_areas();
        for obj_type in 0..2 {
            for (imp, areas, pres) in zip!(
                contact_impulse.view().at(obj_type).iter(),
                vertex_areas.view().at(obj_type).iter(),
                pressure.view_mut().isolate(obj_type).iter_mut()
            ) {
                for (&i, &a, p) in zip!(imp.iter(), areas.iter(), pres.iter_mut()) {
                    if a > 0.0 {
                        *p += i.as_tensor().norm() / a;
                    }
                }
            }
        }

        //// DEBUG CODE:
        //if self.frictional_contacts.len() == 1 {
        //    let fc = &self.frictional_contacts[0];
        //    let ObjectData { solids, shells, .. } = &self.object_data;
        //    let [_, mut coll_p] = ObjectData::mesh_vertex_subset_split_mut_impl(
        //        pressure.view_mut(),
        //        None,
        //        [fc.object_index, fc.collider_index],
        //        solids,
        //        shells
        //    );

        //    fc.constraint
        //        .borrow()
        //        .smooth_collider_values(coll_p.view_mut());
        //}
        pressure
    }

    /// Update the solid and shell meshes with relevant simulation data.
    //pub fn update_mesh_data(&mut self) {
    //    let contact_impulse = self.contact_impulse();
    //    let friction_impulse = self.friction_impulse();
    //    let friction_corrector_impulse = self.friction_corrector_impulse();
    //    let pressure = self.pressure(contact_impulse.view());
    //    let potential = self.contact_potential();
    //    let normals = self.collider_normals();
    //    for (idx, solid) in self.object_data.solids.iter_mut().enumerate() {
    //        // Write back friction and contact impulses
    //        solid
    //            .tetmesh
    //            .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                "friction_corrector",
    //                friction_corrector_impulse
    //                    .view()
    //                    .at(0)
    //                    .at(idx)
    //                    .view()
    //                    .into(),
    //            )
    //            .ok();
    //        solid
    //            .tetmesh
    //            .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                FRICTION_ATTRIB,
    //                friction_impulse
    //                    .view()
    //                    .at(SOLIDS_INDEX)
    //                    .at(idx)
    //                    .view()
    //                    .into(),
    //            )
    //            .ok();
    //        solid
    //            .tetmesh
    //            .set_attrib_data::<ContactImpulseType, VertexIndex>(
    //                CONTACT_ATTRIB,
    //                contact_impulse
    //                    .view()
    //                    .at(SOLIDS_INDEX)
    //                    .at(idx)
    //                    .view()
    //                    .into(),
    //            )
    //            .ok();
    //        solid
    //            .tetmesh
    //            .set_attrib_data::<PotentialType, VertexIndex>(
    //                POTENTIAL_ATTRIB,
    //                potential.view().at(SOLIDS_INDEX).at(idx).view().into(),
    //            )
    //            .ok();
    //        solid
    //            .tetmesh
    //            .set_attrib_data::<PressureType, VertexIndex>(
    //                PRESSURE_ATTRIB,
    //                pressure.view().at(SOLIDS_INDEX).at(idx).view().into(),
    //            )
    //            .ok();

    //        solid
    //            .tetmesh
    //            .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                "collider_normals",
    //                normals.view().at(SOLIDS_INDEX).at(idx).view().into(),
    //            )
    //            .ok();

    //        // Write back elastic strain energy for visualization.
    //        Self::compute_strain_energy_attrib(solid);

    //        // Write back elastic forces on each node.
    //        Self::compute_elastic_forces_attrib(solid);
    //    }
    //    for (idx, shell) in self.object_data.shells.iter_mut().enumerate() {
    //        // Write back friction and contact impulses
    //        shell
    //            .trimesh
    //            .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                "friction_corrector",
    //                friction_corrector_impulse
    //                    .view()
    //                    .at(1)
    //                    .at(idx)
    //                    .view()
    //                    .into(),
    //            )
    //            .ok();
    //        shell
    //            .trimesh
    //            .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                FRICTION_ATTRIB,
    //                friction_impulse.view().at(1).at(idx).view().into(),
    //            )
    //            .ok();
    //        shell
    //            .trimesh
    //            .set_attrib_data::<ContactImpulseType, VertexIndex>(
    //                CONTACT_ATTRIB,
    //                contact_impulse.view().at(1).at(idx).view().into(),
    //            )
    //            .ok();
    //        shell
    //            .trimesh
    //            .set_attrib_data::<PotentialType, VertexIndex>(
    //                POTENTIAL_ATTRIB,
    //                potential.view().at(1).at(idx).view().into(),
    //            )
    //            .ok();
    //        shell
    //            .trimesh
    //            .set_attrib_data::<PressureType, VertexIndex>(
    //                PRESSURE_ATTRIB,
    //                pressure.view().at(1).at(idx).view().into(),
    //            )
    //            .ok();

    //        shell
    //            .trimesh
    //            .set_attrib_data::<FrictionImpulseType, VertexIndex>(
    //                "collider_normals",
    //                normals.view().at(1).at(idx).view().into(),
    //            )
    //            .ok();

    //        if let ShellData::Soft { .. } = shell.data {
    //            // Write back elastic strain energy for visualization.
    //            shell.compute_strain_energy_attrib();

    //            // Write back elastic forces on each node.
    //            shell.compute_elastic_forces_attrib();
    //        }
    //    }
    //}

    /*
     * The following functions are there for debugging jacobians and hessians
     */

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

        img.save(format!("./out/jac_{}.png", self.iter_counter.borrow()))
            .expect("Failed to save Jacobian Image");
    }

    //#[allow(dead_code)]
    //pub fn print_jacobian_svd(&self, values: &[f64]) {
    //    use ipopt::{BasicProblem, ConstrainedProblem};
    //    use na::{base::storage::Storage, DMatrix};

    //    if values.is_empty() {
    //        return;
    //    }

    //    let mut rows = vec![0; values.len()];
    //    let mut cols = vec![0; values.len()];
    //    assert!(self.constraint_jacobian_indices(&mut rows, &mut cols));

    //    let nrows = self.num_constraints();
    //    let ncols = self.num_variables();
    //    let mut jac = DMatrix::<f64>::zeros(nrows, ncols);
    //    for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
    //        jac[(row as usize, col as usize)] += v;
    //    }

    //    self.write_jacobian_img(&jac);

    //    use std::io::Write;

    //    let mut f =
    //        std::fs::File::create(format!("./out/jac_{}.txt", self.iter_counter.borrow())).unwrap();
    //    writeln!(&mut f, "jac = ").ok();
    //    for r in 0..nrows {
    //        for c in 0..ncols {
    //            if jac[(r, c)] != 0.0 {
    //                write!(&mut f, "{:9.5}", jac[(r, c)]).ok();
    //            } else {
    //                write!(&mut f, "    .    ",).ok();
    //            }
    //        }
    //        writeln!(&mut f).ok();
    //    }
    //    writeln!(&mut f).ok();

    //    let svd = na::SVD::new(jac, false, false);
    //    let s: &[f64] = Storage::as_slice(&svd.singular_values.data);
    //    let cond = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
    //        / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    //    log::debug!("Condition number of jacobian is: {}", cond);
    //}

    #[allow(dead_code)]
    fn output_first_solid(&self, x: VertexView3<&[f64]>, name: &str) -> Result<(), crate::Error> {
        let mut iter_counter = self.iter_counter.borrow_mut();
        for (idx, solid) in self.object_data.solids.iter().enumerate() {
            let mut mesh = solid.tetmesh.clone();
            mesh.vertex_positions_mut()
                .iter_mut()
                .zip(x.at(0).at(idx).iter())
                .for_each(|(out_p, p)| *out_p = *p);
            *iter_counter += 1;
            geo::io::save_tetmesh(
                &mesh,
                &std::path::PathBuf::from(format!("./out/{}_{}.vtk", name, *iter_counter)),
            )?;
            log::trace!("Iter counter: {}", *iter_counter);
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn print_jacobian_svd(&self, values: &[f64]) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        use na::{base::storage::Storage, DMatrix};

        if values.is_empty() {
            return;
        }

        let (rows, cols) = self.jacobian_indices();

        let mut hess = DMatrix::<f64>::zeros(self.num_variables(), self.num_variables());
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            hess[(row as usize, col as usize)] += v;
        }

        let svd = na::SVD::new(hess, false, false);
        let s: &[f64] = Storage::as_slice(&svd.singular_values.data);
        let cond_hess = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        log::debug!("Condition number of hessian is {}", cond_hess);
    }

    /*
     * End of debugging functions
     */

    fn jacobian_nnz(&self) -> usize {
        let mut num = 0;
        for solid in self.object_data.solids.iter() {
            num += solid.elasticity::<T>().energy_hessian_size()
                + if !self.is_static() {
                    solid.inertia().energy_hessian_size()
                } else {
                    0
                };
        }
        for shell in self.object_data.shells.iter() {
            num += shell.elasticity::<T>().energy_hessian_size()
                + if !self.is_static() {
                    shell.inertia().energy_hessian_size()
                } else {
                    0
                };
        }
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.borrow().constraint_hessian_size();
        }
        for fc in self.frictional_contacts.iter() {
            //TODO: add frictional contact counts
            //if !self.is_rigid(fc.object_index) {
            //    num += fc.constraint.borrow().object_constraint_hessian_size();
            //}
            //if !self.is_rigid(fc.collider_index) {
            //    num += fc.constraint.borrow().collider_constraint_hessian_size();
            //}
        }
        num
    }
}

pub trait NonLinearProblem<T: Real> {
    /// Returns the number of unknowns for the problem.
    fn num_variables(&self) -> usize;
    /// Constructs the initial point for the problem.
    ///
    /// The returned `Vec` must have `num_variables` elements.
    fn initial_point(&self) -> Vec<T> {
        vec![T::zero(); self.num_variables()]
    }
    /// Initializes the lower and upper bounds of the problem respectively.
    ///
    /// If bounds change between solves, then it is sufficient to override `update_bounds` only.
    /// This function calls `update_bounds` by default.
    /// If the bounds are not updated between solves, override this function
    /// instead to avoid wasted work between solves.
    fn initial_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut l = vec![std::f64::NEG_INFINITY; self.num_variables()];
        let mut u = vec![std::f64::INFINITY; self.num_variables()];
        self.update_bounds(l.as_mut_slice(), u.as_mut_slice());
        (l, u)
    }
    /// Updates the lower and upper bounds of the problem.
    ///
    /// If the bounds change between solves, this function should be defined instead of `initial_bounds`.
    fn update_bounds(&self, l: &mut [f64], u: &mut [f64]) {}

    /// The vector function `r` whose roots we want to find.
    ///
    /// `r(x) = 0`.
    fn residual(&self, x: &[T], r: &mut [T]);

    /*
     * The sparse Jacobian of `r`.
     */

    /// Returns the sparsity structure of the Jacobian.
    ///
    /// This function returns a pair of owned `Vec`s containing row and column
    /// indices corresponding to the (potentially) "non-zero" entries of the Jacobian.
    /// The returned vectors must have the same size.
    fn jacobian_indices(&self) -> (Vec<usize>, Vec<usize>);
    /// Updates the given set of values corresponding to each non-zero of the
    /// Jacobian as defined by `j_indices`.
    ///
    /// The size of `values` is the same as the size of one of the index vectors
    /// returned by `jacobian_indices`.
    fn jacobian_values(&self, x: &[T], values: &mut [T]);
}

/// Prepare the problem for Newton iterations.
impl<T: Real64> NonLinearProblem<T> for NLProblem<T> {
    fn num_variables(&self) -> usize {
        self.object_data.dof.storage().len()
    }

    fn initial_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let uv_l = vec![std::f64::NEG_INFINITY; self.num_variables()];
        let uv_u = vec![std::f64::INFINITY; self.num_variables()];

        // Fixed vertices have a predetermined velocity which is specified in the dof variable.
        // Unscale velocities so we can set the unscaled bounds properly.
        let uv_flat_view = self.object_data.dof.view().into_storage().cur.dq;
        self.object_data
            .update_workspace_velocity(uv_flat_view, 1.0 / self.variable_scale());
        let ws = self.object_data.workspace.borrow();
        let unscaled_dq = ws.dof.view().map_storage(|dof| dof.dq);
        let solid_prev_uv = unscaled_dq.isolate(SOLIDS_INDEX);
        let shell_prev_uv = unscaled_dq.isolate(SHELLS_INDEX);

        // Copy data structure over to uv_l and uv_u
        let mut uv_l = self.object_data.dof.view().map_storage(move |_| uv_l);
        let mut uv_u = self.object_data.dof.view().map_storage(move |_| uv_u);

        for (i, (solid, uv)) in self
            .object_data
            .solids
            .iter()
            .zip(solid_prev_uv.iter())
            .enumerate()
        {
            if let Ok(fixed_verts) = solid
                .tetmesh
                .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            {
                let mut uv_l = uv_l.view_mut().isolate(SOLIDS_INDEX).isolate(i);
                let mut uv_u = uv_u.view_mut().isolate(SOLIDS_INDEX).isolate(i);
                // Find and set fixed vertices.
                uv_l.iter_mut()
                    .zip(uv_u.iter_mut())
                    .zip(uv.iter())
                    .zip(fixed_verts.iter())
                    .filter(|&(_, &fixed)| fixed != 0)
                    .for_each(|(((l, u), uv), _)| {
                        *l = uv.as_tensor().cast::<f64>().into();
                        *u = *l;
                    });
            }
        }

        for (i, (shell, uv)) in self
            .object_data
            .shells
            .iter()
            .zip(shell_prev_uv.iter())
            .enumerate()
        {
            if let Ok(fixed_verts) = shell
                .trimesh
                .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            {
                let mut uv_l = uv_l.view_mut().isolate(1).isolate(i);
                let mut uv_u = uv_u.view_mut().isolate(1).isolate(i);
                // Find and set fixed vertices.
                uv_l.iter_mut()
                    .zip(uv_u.iter_mut())
                    .zip(uv.iter())
                    .zip(fixed_verts.iter())
                    .filter(|&(_, &fixed)| fixed != 0)
                    .for_each(|(((l, u), uv), _)| {
                        *l = uv.as_tensor().cast::<f64>().into();
                        *u = *l;
                    });
            }
        }
        let uv_l = uv_l.into_storage();
        let uv_u = uv_u.into_storage();
        debug_assert!(uv_l.iter().all(|&x| x.is_finite()) && uv_u.iter().all(|&x| x.is_finite()));
        (uv_l, uv_u)
    }

    fn residual(&self, uv: &[T], grad_f: &mut [T]) {
        log::trace!(
            "Unscaled variable norm: {:?}",
            crate::inf_norm(uv.iter().cloned())
        );
        grad_f.iter_mut().for_each(|x| *x = T::zero()); // clear gradient vector

        self.update_current_velocity(uv);
        self.compute_step();

        let dof_cur = self.object_data.dof.view().map_storage(|dof| dof.cur);

        // Copy the chunked structure from our object_data.
        // TODO: Refactor this into a function for Chunked and UniChunked types.
        let mut grad = dof_cur.map_storage(|_| grad_f);

        {
            let ws = self.object_data.workspace.borrow();
            let q_next = ws.dof.view().map_storage(|dof| dof.q);
            let q_cur = dof_cur.map_storage(|dof| dof.q);

            for (i, solid) in self.object_data.solids.iter().enumerate() {
                let q_cur = q_cur.at(SOLIDS_INDEX).at(i).into_storage();
                let q_next = q_next.at(SOLIDS_INDEX).at(i).into_storage();
                let g = grad
                    .view_mut()
                    .isolate(SOLIDS_INDEX)
                    .isolate(i)
                    .into_storage();
                solid.elasticity().add_energy_gradient(q_cur, q_next, g);
                solid
                    .gravity(self.gravity)
                    .add_energy_gradient(q_cur, q_next, g);
            }

            for (i, shell) in self.object_data.shells.iter().enumerate() {
                let q_cur = q_cur.at(SHELLS_INDEX).at(i).into_storage();
                let q_next = q_next.at(SHELLS_INDEX).at(i).into_storage();
                let g = grad
                    .view_mut()
                    .isolate(SHELLS_INDEX)
                    .isolate(i)
                    .into_storage();
                shell.elasticity().add_energy_gradient(q_cur, q_next, g);
                shell
                    .gravity(self.gravity)
                    .add_energy_gradient(q_cur, q_next, g);
            }
        } // Drop borrows into object_data.workspace

        if !self.is_static() {
            {
                let dq_cur = dof_cur.map_storage(|dof| dof.dq);
                let ws = self.object_data.workspace.borrow();
                let dq_next = ws.dof.view().map_storage(|dof| dof.dq);
                {
                    let grad_flat = grad.view_mut().into_storage();
                    // This is a correction to transform the above energy derivatives to
                    // velocity gradients from position gradients.
                    grad_flat.iter_mut().for_each(|g| *g *= self.time_step);
                }

                // Finally add inertia terms
                for (i, solid) in self.object_data.solids.iter().enumerate() {
                    let dq_cur = dq_cur.at(SOLIDS_INDEX).at(i).into_storage();
                    let dq_next = dq_next.at(SOLIDS_INDEX).at(i).into_storage();
                    let g = grad
                        .view_mut()
                        .isolate(SOLIDS_INDEX)
                        .isolate(i)
                        .into_storage();
                    solid.inertia().add_energy_gradient(dq_cur, dq_next, g);
                }

                for (i, shell) in self.object_data.shells.iter().enumerate() {
                    let dq_cur = dq_cur.at(SHELLS_INDEX).at(i).into_storage();
                    let dq_next = dq_next.at(SHELLS_INDEX).at(i).into_storage();
                    let g = grad
                        .view_mut()
                        .isolate(SHELLS_INDEX)
                        .isolate(i)
                        .into_storage();
                    shell.inertia().add_energy_gradient(dq_cur, dq_next, g);
                }
            } // Drop object_data.workspace borrow

            //            for fc in self.frictional_contacts.iter() {
            //                // Since add_friction_impulse is looking for a valid gradient, this
            //                // must involve only vertices that can change.
            //                //assert!(match fc.object_index {
            //                //    SourceObject::Solid(_) => true,
            //                //    SourceObject::Shell(i) => match self.object_data.shells[i].data {
            //                //        ShellData::Fixed { .. } => match fc.collider_index {
            //                //            SourceObject::Solid(_) => true,
            //                //            SourceObject::Shell(i) => {
            //                //                match self.object_data.shells[i].data {
            //                //                    ShellData::Fixed { .. } => false,
            //                //                    ShellData::Rigid { .. } => true,
            //                //                    _ => true,
            //                //                }
            //                //            }
            //                //        },
            //                //        ShellData::Rigid { .. } => true,
            //                //        _ => true,
            //                //    },
            //                //});
            //
            //                let fc_constraint = fc.constraint.borrow();
            //                if let Some(fc_contact) = fc_constraint.frictional_contact.as_ref() {
            //                    // ws.grad is a zero initialized memory slice to which we can write the gradient to.
            //                    // This may be different than `grad.view_mut()` if the object is rigid and has
            //                    // different degrees of freedom.
            //                    let mut ws = self.object_data.workspace.borrow_mut();
            //                    if fc_contact.params.friction_forwarding > 0.0 {
            //                        let mut obj_g = self.object_data.mesh_vertex_subset(
            //                            grad.view_mut(),
            //                            ws.vtx.view_mut().map_storage(|vtx| vtx.grad),
            //                            fc.object_index,
            //                        );
            //                        fc_constraint.add_friction_impulse_to_object(obj_g.view_mut(), -1.0);
            //
            //                        let mut coll_g = self.object_data.mesh_vertex_subset(
            //                            grad.view_mut(),
            //                            ws.vtx.view_mut().map_storage(|vtx| vtx.grad),
            //                            fc.collider_index,
            //                        );
            //                        fc_constraint.add_friction_impulse_to_collider(coll_g.view_mut(), -1.0);
            //                    }
            //                }
            //
            //                // Update `grad.view_mut()` with the newly computed gradients. This is a noop
            //                // unless at least one of the objects is rigid.
            //                self.object_data.sync_grad(fc.object_index, grad.view_mut());
            //                self.object_data
            //                    .sync_grad(fc.collider_index, grad.view_mut());
            //            }
        }

        let grad_f = grad.into_storage();

        let scale = self.variable_scale() * self.impulse_inv_scale();
        grad_f.iter_mut().for_each(|g| *g *= scale);
        log::trace!(
            "Objective gradient norm: {:?}",
            crate::inf_norm(grad_f.iter().cloned())
        );

        debug_assert!(grad_f.iter().all(|&g| g.is_finite()));
    }

    fn jacobian_indices(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = vec![0; self.jacobian_nnz()];
        let mut cols = vec![0; self.jacobian_nnz()];
        // This is used for counting offsets.
        let dq_cur_solid = self
            .object_data
            .dof
            .view()
            .at(SOLIDS_INDEX)
            .map_storage(|dof| dof.cur.dq);
        let dq_cur_shell = self
            .object_data
            .dof
            .view()
            .at(SHELLS_INDEX)
            .map_storage(|dof| dof.cur.dq);

        let mut count = 0; // Constraint counter

        // Add energy indices
        for (solid_idx, solid) in self.object_data.solids.iter().enumerate() {
            let offset = dq_cur_solid.offset_value(solid_idx) * 3;
            let elasticity = solid.elasticity::<T>();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_rows_cols_offset(
                (offset, offset).into(),
                &mut rows[count..count + n],
                &mut cols[count..count + n],
            );

            count += n;

            if !self.is_static() {
                let inertia = solid.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_rows_cols_offset(
                    (offset, offset).into(),
                    &mut rows[count..count + n],
                    &mut cols[count..count + n],
                );
                count += n;
            }
        }

        for (shell_idx, shell) in self.object_data.shells.iter().enumerate() {
            let offset = dq_cur_shell.offset_value(shell_idx) * 3;
            let elasticity = shell.elasticity::<T>();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_rows_cols_offset(
                (offset, offset).into(),
                &mut rows[count..count + n],
                &mut cols[count..count + n],
            );
            count += n;

            if !self.is_static() {
                let inertia = shell.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_rows_cols_offset(
                    (offset, offset).into(),
                    &mut rows[count..count + n],
                    &mut cols[count..count + n],
                );
                count += n;
            }
        }

        // Add volume constraint indices
        for (solid_idx, vc) in self.volume_constraints.iter() {
            let offset = dq_cur_solid.offset_value(*solid_idx) * 3;
            for MatrixElementIndex { row, col } in vc.borrow().constraint_hessian_indices_iter() {
                rows[count] = row + offset;
                cols[count] = col + offset;
                count += 1;
            }
        }

        // TODO: Add frictional contact indices

        assert_eq!(count, rows.len());
        assert_eq!(count, cols.len());
        (rows, cols)
    }

    fn jacobian_values(&self, uv: &[T], vals: &mut [T]) {
        //let lambda = Vec::new();
        self.update_current_velocity(uv);
        self.compute_step();

        let mut count = 0; // Values counter
        let mut coff = 0; // Constraint offset

        // Correction to make the above hessian wrt velocity instead of displacement.
        let dt = T::from(self.time_step()).unwrap();

        // Constraint scaling
        let c_scale = T::from(self.variable_scale() * self.variable_scale()).unwrap() * dt * dt;

        let dof_cur = self.object_data.dof.view().map_storage(|dof| dof.cur);

        let mut ws = self.object_data.workspace.borrow_mut();
        let WorkspaceData { dof, vtx, .. } = &mut *ws;
        ObjectData::sync_pos(
            &self.object_data.shells,
            dof.view().map_storage(|dof| dof.q),
            vtx.view_mut().map_storage(|vtx| vtx.state.pos),
        );

        let dof_next = dof.view();

        if cfg!(debug_assertions) {
            // Initialize vals in debug builds.
            for v in vals.iter_mut() {
                *v = T::zero();
            }
        }

        // Multiply energy hessian by objective factor and scaling factors.
        let factor =
            T::from(self.variable_scale() * self.variable_scale() * self.impulse_inv_scale())
                .unwrap();

        for (solid_idx, solid) in self.object_data.solids.iter().enumerate() {
            let dof_cur = dof_cur.at(SOLIDS_INDEX).at(solid_idx).into_storage();
            let dof_next = dof_next.at(SOLIDS_INDEX).at(solid_idx).into_storage();
            let elasticity = solid.elasticity::<T>();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_values(
                dof_cur.q,
                dof_next.q,
                dt * dt * factor,
                &mut vals[count..count + n],
            );
            count += n;

            if !self.is_static() {
                let inertia = solid.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_values(
                    dof_cur.dq,
                    dof_next.dq,
                    factor,
                    &mut vals[count..count + n],
                );
                count += n;
            }
        }

        for (shell_idx, shell) in self.object_data.shells.iter().enumerate() {
            let dof_cur = dof_cur.at(SHELLS_INDEX).at(shell_idx).into_storage();
            let dof_next = dof_next.at(SHELLS_INDEX).at(shell_idx).into_storage();
            let elasticity = shell.elasticity::<T>();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_values(
                dof_cur.q,
                dof_next.q,
                dt * dt * factor,
                &mut vals[count..count + n],
            );
            count += n;

            if !self.is_static() {
                let inertia = shell.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_values(
                    dof_cur.dq,
                    dof_next.dq,
                    factor,
                    &mut vals[count..count + n],
                );
                count += n;
            }
        }

        //for (solid_idx, vc) in self.volume_constraints.iter() {
        //    let q_cur = dof_cur.at(SOLIDS_INDEX).at(*solid_idx).into_storage().q;
        //    let q_next = dof_next.at(SOLIDS_INDEX).at(*solid_idx).into_storage().q;
        //    let nc = vc.borrow().constraint_size();
        //    let nh = vc.borrow().constraint_hessian_size();
        //    vc.borrow_mut()
        //        .constraint_hessian_values(
        //            q_cur,
        //            q_next,
        //            &lambda[coff..coff + nc],
        //            c_scale * self.volume_constraint_scale(),
        //            &mut vals[count..count + nh],
        //        )
        //        .unwrap();

        //    count += nh;
        //    coff += nc;
        //}

        // TODO: Add frictional contact values

        assert_eq!(count, vals.len());
        //assert_eq!(coff, lambda.len());
        //self.print_hessian_svd(vals);

        debug_assert!(vals.iter().all(|x| x.is_finite()));
    }
}
