use std::cell::RefCell;

use autodiff as ad;
use flatk::*;
use geo::mesh::VertexPositions;
use num_traits::{Float, Zero};
use tensr::{AsMutTensor, IntoData, IntoTensor, Matrix, Tensor};

use super::state::*;
use crate::constraint::*;
use crate::constraints::volume::VolumeConstraint;
use crate::contact::ContactJacobianView;
use crate::energy::{EnergyGradient, EnergyHessian, EnergyHessianTopology};
use crate::energy_models::{gravity::*, inertia::*};
use crate::matrix::*;
use crate::objects::tetsolid::*;
use crate::objects::trishell::*;
use crate::PointCloud;
use crate::{Real, Real64};

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

// Debug code to write the jacobian as an image.
#[allow(dead_code)]
pub fn write_jacobian_img(jac: &na::DMatrix<f64>, iter: u32) {
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

    img.save(format!("./out/jac_{}.png", iter))
        .expect("Failed to save Jacobian Image");
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
#[derive(Clone, Debug)]
pub struct NLProblem<T: Real> {
    /// A data model of the current problem.
    ///
    /// This includes all primal variables and any additional mesh data required
    /// for simulation.
    pub state: RefCell<State<T, ad::FT<T>>>,
    /// Contact penalty multiplier.
    ///
    /// This quantity is used to dynamically enforce non-penetration.
    pub kappa: f64,
    /// Contact tolerance.
    // TODO: move this to FrictionalContactConstraint.
    pub delta: f64,
    ///// One way contact constraints between a pair of objects.
    //pub frictional_contacts: Vec<FrictionalContactConstraint<T>>,
    ///// One way contact constraints between a pair of objects (used for automatic differentiation).
    //pub frictional_contacts_ad: Vec<FrictionalContactConstraint<ad::FT<T>>>,
    /// Constraint on the volume of different regions of the simulation.
    pub volume_constraints: Vec<RefCell<VolumeConstraint>>,
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
    /// The maximum scale of the energy gradient intended to be used for rescaling the objective gradient.
    pub max_element_force_scale: f64,
    /// The minimum scale of the energy gradient intended to be used for rescaling the objective gradient.
    pub min_element_force_scale: f64,
}

impl<T: Real> NLProblem<T> {
    pub fn impulse_inv_scale(&self) -> f64 {
        1.0 //utils::approx_power_of_two64(100.0 / (self.time_step() * self.max_element_force_scale))
    }

    //fn volume_constraint_scale(&self) -> f64 {
    //    1.0
    //}

    //fn contact_constraint_scale(&self) -> f64 {
    //    1.0
    //}

    pub fn time_step(&self) -> f64 {
        if self.is_static() {
            1.0
        } else {
            self.time_step
        }
    }

    /// Check if this problem represents a static simulation.
    ///
    /// In this case inertia is ignored and velocities are treated as
    /// displacements.
    fn is_static(&self) -> bool {
        self.time_step == 0.0
    }

    /// Returns the solved positions of the vertices.
    pub fn vertex_positions(&self) -> std::cell::Ref<[[T; 3]]> {
        let state = self.state.borrow();
        std::cell::Ref::map(state, |s| s.vtx.cur.pos.as_arrays())
    }
}

impl<T: Real64> NLProblem<T> {
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
        None
        //self.frictional_contacts
        //    .iter()
        //    .map(|fc| fc.constraint.borrow().contact_radius())
        //    .min_by(|a, b| a.partial_cmp(b).expect("Detected NaN contact radius"))
    }

    //        /// Save an intermediate state of the solve.
    //        ///
    //        /// This is used for debugging.
    //        #[allow(dead_code)]
    //        pub fn save_intermediate(&mut self, v: &[T], step: usize) {
    //            self.integrate_step(v);
    //            let mut ws = self.state.workspace.borrow_mut();
    //            // Copy v into the workspace to be used in update_simulated_meshes.
    //            ws.dof
    //                .view_mut()
    //                .storage_mut()
    //                .state
    //                .dq
    //                .iter_mut()
    //                .zip(v.iter())
    //                .for_each(|(out, &v)| *out = v);
    //
    //            let mut solids = self.state.solids.clone();
    //            let mut shells = self.state.shells.clone();
    //            State::update_simulated_meshes_with(
    //                &mut solids,
    //                &mut shells,
    //                ws.dof.view().map_storage(|dof| dof.state),
    //                ws.vtx.view().map_storage(|vtx| vtx.state),
    //            );
    //            geo::io::save_tetmesh(
    //                &solids[0].tetmesh,
    //                &std::path::PathBuf::from(format!("./out/predictor_{}.vtk", step)),
    //            )
    //            .unwrap();
    //        }

    /// Update the state with the given points.
    pub fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        let new_pos = Chunked3::from_array_slice(pts.vertex_positions());
        self.state
            .borrow_mut()
            .update_fixed_vertices(new_pos.view(), self.time_step())
    }

    /// Compute the set of currently active constraints into the given `Chunked` `Vec`.
    pub fn compute_active_constraint_set(&self, active_set: &mut Chunked<Vec<usize>>) {
        // Disassemble chunked collection.
        let (offsets, active_set) = active_set.as_inner_mut();

        for i in 0..self.volume_constraints.len() {
            active_set.push(i);
            offsets.push(active_set.len());
        }

        //let mut offset = active_set.len();
        //for FrictionalContactConstraint { ref constraint, .. } in self.frictional_contacts.iter() {
        //    let fc_active_constraints = constraint.borrow().active_constraint_indices();
        //    for c in fc_active_constraints.into_iter() {
        //        active_set.push(c + offset);
        //    }
        //    offset += constraint.borrow().num_potential_contacts();
        //    offsets.push(active_set.len());
        //}
    }

    ///// Check if all contact constraints are linear.
    //pub fn all_contacts_linear(&self) -> bool {
    //    self.frictional_contacts
    //        .iter()
    //        .all(|contact_constraint| contact_constraint.constraint.borrow().is_linear())
    //}

    /// Get the set of currently active constraints.
    pub fn active_constraint_set(&self) -> Chunked<Vec<usize>> {
        let mut active_set = Chunked::new();
        self.compute_active_constraint_set(&mut active_set);
        active_set
    }

    /// Check if the given constraint set is the same as the current one.
    pub fn is_same_as_constraint_set(&self, other_set: ChunkedView<&[usize]>) -> bool {
        let cur_set = self.active_constraint_set().into_storage();
        let other_set = other_set.into_storage();
        cur_set.len() == other_set.len()
            && cur_set
                .into_iter()
                .zip(other_set.iter())
                .all(|(cur, &other)| cur == other)
    }

    /// Updates all stateful constraints with the most recent data.
    ///
    /// Return an estimate if any constraints have changed, though this estimate may have false
    /// negatives.
    pub fn update_constraint_set(&mut self) -> bool {
        let changed = false; // Report if anything has changed to the caller.
        changed

        //    let NLProblem {
        //        //ref mut frictional_contacts,
        //        //ref mut frictional_contacts_ad,
        //        ref state,
        //        ..
        //    } = *self;

        //    let q = state.dof.view().storage().cur.q;
        //    let pos = state.vtx.view().storage().cur.pos;

        //    let q_ad: Vec<_> = q.iter().map(|&x| ad::F::cst(x)).collect();
        //    let pos_ad: Vec<_> = pos.iter().map(|&x| ad::F::cst(x)).collect();

        //    let q = state.dof.view().map_storage(|_| q);
        //    let pos = state.vtx.view().map_storage(|_| pos);

        //    for FrictionalContactConstraint {
        //        object_index,
        //        collider_index,
        //        constraint,
        //    } in frictional_contacts.iter_mut()
        //    {
        //        let object_pos = state.mesh_vertex_subset(q, pos, *object_index);
        //        let collider_pos = state.mesh_vertex_subset(q, pos, *collider_index);

        //        changed |= constraint
        //            .borrow_mut()
        //            .update_neighbors(object_pos.view(), collider_pos.view());
        //    }

        //    let q_ad = state.dof.view().map_storage(|_| q_ad.view());
        //    let pos_ad = state.vtx.view().map_storage(|_| pos_ad.view());

        //    for FrictionalContactConstraint {
        //        object_index,
        //        collider_index,
        //        constraint,
        //    } in frictional_contacts_ad.iter_mut()
        //    {
        //        let object_pos = state.mesh_vertex_subset(q_ad, pos_ad, *object_index);
        //        let collider_pos = state.mesh_vertex_subset(q_ad, pos_ad, *collider_index);

        //        changed |= constraint
        //            .borrow_mut()
        //            .update_neighbors(object_pos.view(), collider_pos.view());
        //    }

        //    changed
    }

    //pub fn apply_frictional_contact_impulse(&mut self) {
    //    let NLProblem {
    //        frictional_contacts,
    //        state,
    //        ..
    //    } = self;

    //    let mut ws = state.workspace.borrow_mut();
    //    let WorkspaceData { dof, vtx, .. } = &mut *ws;

    //    for fc in frictional_contacts.iter() {
    //        let [obj_idx, coll_idx] = [fc.object_index, fc.collider_index];
    //        let fc = fc.constraint.borrow();

    //        let dq = dof.view_mut().map_storage(|dof| dof.dq);
    //        let vtx_vel = vtx.view_mut().map_storage(|vtx| vtx.state.vel);
    //        let mut obj_vel = state.mesh_vertex_subset(dq, vtx_vel, obj_idx);
    //        fc.add_mass_weighted_frictional_contact_impulse_to_object(obj_vel.view_mut());

    //        let dq = dof.view_mut().map_storage(|dof| dof.dq);
    //        let vtx_vel = vtx.view_mut().map_storage(|vtx| vtx.state.vel);
    //        let mut coll_vel = state.mesh_vertex_subset(dq, vtx_vel, coll_idx);
    //        fc.add_mass_weighted_frictional_contact_impulse_to_collider(coll_vel.view_mut());
    //    }
    //}

    /// Commit velocity by advancing the internal state by the given unscaled velocity `uv`.
    pub fn advance(&mut self, v: &[T]) {
        self.integrate_step(v);
        self.state.borrow_mut().advance(v);
    }

    //    /// Advance object data one step back.
    //    pub fn retreat(&mut self) {
    //        self.state.borrow_mut().retreat();
    //        //        // Clear any frictional impulses
    //        //        for fc in self.frictional_contacts.iter() {
    //        //            if let Some(friction_data) = fc.constraint.borrow_mut().frictional_contact_mut() {
    //        //                friction_data
    //        //                    .collider_impulse
    //        //                    .source_iter_mut()
    //        //                    .for_each(|(x, y)| {
    //        //                        *x = [T::zero(); 3];
    //        //                        *y = [T::zero(); 3]
    //        //                    });
    //        //                friction_data.object_impulse.iter_mut().for_each(|(x, y)| {
    //        //                    *x = [T::zero(); 3];
    //        //                    *y = [T::zero(); 3]
    //        //                });
    //        //            }
    //        //        }
    //        //        for fc in self.frictional_contacts_ad.iter() {
    //        //            if let Some(friction_data) = fc.constraint.borrow_mut().frictional_contact_mut() {
    //        //                friction_data
    //        //                    .collider_impulse
    //        //                    .source_iter_mut()
    //        //                    .for_each(|(x, y)| {
    //        //                        *x = [ad::F::zero(); 3];
    //        //                        *y = [ad::F::zero(); 3]
    //        //                    });
    //        //                friction_data.object_impulse.iter_mut().for_each(|(x, y)| {
    //        //                    *x = [ad::F::zero(); 3];
    //        //                    *y = [ad::F::zero(); 3]
    //        //                });
    //        //            }
    //        //        }
    //    }

    pub fn update_max_step(&mut self, _step: f64) {
        //for fc in self.frictional_contacts.iter_mut() {
        //    fc.constraint.borrow_mut().update_max_step(step);
        //}
        //for fc in self.frictional_contacts_ad.iter_mut() {
        //    fc.constraint.borrow_mut().update_max_step(step);
        //}
    }
    pub fn update_radius_multiplier(&mut self, _rad_mult: f64) {
        //for fc in self.frictional_contacts.iter_mut() {
        //    fc.constraint
        //        .borrow_mut()
        //        .update_radius_multiplier(rad_mult);
        //}
        //for fc in self.frictional_contacts_ad.iter_mut() {
        //    fc.constraint
        //        .borrow_mut()
        //        .update_radius_multiplier(rad_mult);
        //}
    }

    /// A convenience function to integrate the given velocity by the internal time step.
    ///
    /// For implicit integration this boils down to a simple multiply by the time step.
    pub fn integrate_step(&self, v: &[T]) {
        let mut state = self.state.borrow_mut();
        State::be_step(state.step_state(v), self.time_step());
        // or self.state.tr_step(v, self.time_step());
        // or self.state.bdf2_step(v, self.time_step(), gamma);
    }

    /// Convert a given array of contact forces to impulses.
    //fn contact_impulse_magnitudes(forces: &[f64], scale: f64) -> Vec<f64> {
    //    forces.iter().map(|&cf| cf * scale).collect()
    //}

    ///// Construct the global contact Jacobian matrix.
    /////
    ///// The contact Jacobian consists of blocks representing contacts between pairs of objects.
    ///// Each block represents a particular coupling. Given two objects A and B, there can be two
    ///// types of coupling:
    ///// A is an implicit surface in contact with vertices of B and
    ///// B is an implicit surface in contact with vertices of A.
    ///// Both of these are valid for solids since they represent a volume, while cloth can only
    ///// collide against implicit surfaces (for now) and not vice versa.
    //pub fn construct_contact_jacobian(
    //    &self,
    //    solution: &[T],
    //    constraint_values: &[f64],
    //    // TODO: Move to GlobalContactJacobian, which combines these two outputs.
    //) -> (ContactJacobian, Chunked<Offsets<Vec<usize>>>) {
    //    let NLProblem {
    //        ref frictional_contacts,
    //        ref volume_constraints,
    //        ref state,
    //        ..
    //    } = *self;

    //    let mut jac_triplets = TripletContactJacobian::new();

    //    if frictional_contacts.is_empty() {
    //        return (
    //            jac_triplets.into(),
    //            Chunked::from_offsets(vec![0], Offsets::new(vec![])),
    //        );
    //    }

    //    let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();

    //    let dof_view = state.dof.view();

    //    // A set of offsets indexing the beginnings of surface vertex slices for each object.
    //    // This is different than their generalized coordinate offsets.
    //    let mut surface_object_offsets = vec![0; 3];
    //    let mut surface_vertex_offsets = vec![0; dof_view.data().len() + 1];
    //    let mut surface_vertex_offset = 0;
    //    let mut idx = 1;

    //    for object_dofs in dof_view.iter() {
    //        for (solid_idx, _) in object_dofs.iter().enumerate() {
    //            surface_vertex_offsets[idx] = surface_vertex_offset;
    //            surface_vertex_offset += state.solids[solid_idx]
    //                .entire_surface()
    //                .trimesh
    //                .num_vertices();
    //            idx += 1;
    //        }
    //        surface_object_offsets[SOLIDS_INDEX + 1] = surface_vertex_offset;
    //        for (shell_idx, _) in object_dofs.iter().enumerate() {
    //            surface_vertex_offsets[idx] = surface_vertex_offset;
    //            surface_vertex_offset += state.shells[shell_idx].trimesh.num_vertices();
    //            idx += 1;
    //        }
    //        surface_object_offsets[SHELLS_INDEX + 1] = surface_vertex_offset;
    //    }

    //    // Separate offsets by type of mesh for easier access.
    //    let surface_vertex_offsets =
    //        Chunked::from_offsets(surface_object_offsets, Offsets::new(surface_vertex_offsets));
    //    let surface_vertex_offsets_view = surface_vertex_offsets.view();

    //    let mut contact_offset = 0;

    //    //for fc in frictional_contacts.iter() {
    //    //    let n = fc.constraint.borrow().constraint_size();
    //    //    let constraint_offset = contact_offset + volume_constraints.len();

    //    //    // Get the normal component of the contact impulse.
    //    //    let contact_impulse = Self::contact_impulse_magnitudes(
    //    //        &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
    //    //        multiplier_impulse_scale,
    //    //    );

    //    //    log::debug!(
    //    //        "Maximum contact impulse: {}",
    //    //        crate::inf_norm(contact_impulse.iter().cloned())
    //    //    );

    //    //    let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

    //    //    let (_, active_contact_indices, _) = fc
    //    //        .constraint
    //    //        .borrow()
    //    //        .in_contact_indices(&contact_impulse, potential_values);

    //    //    let object_vertex_offset = match fc.object_index {
    //    //        SourceObject::Solid(i, _) => *surface_vertex_offsets_view.at(SOLIDS_INDEX).at(i),
    //    //        SourceObject::Shell(i) => *surface_vertex_offsets_view.at(SHELLS_INDEX).at(i),
    //    //    };
    //    //    let collider_vertex_offset = match fc.collider_index {
    //    //        SourceObject::Solid(i, _) => *surface_vertex_offsets_view.at(SOLIDS_INDEX).at(i),
    //    //        SourceObject::Shell(i) => *surface_vertex_offsets_view.at(SHELLS_INDEX).at(i),
    //    //    };

    //    //    fc.constraint.borrow().append_contact_jacobian_triplets(
    //    //        &mut jac_triplets,
    //    //        &active_contact_indices,
    //    //        contact_offset,
    //    //        object_vertex_offset,
    //    //        collider_vertex_offset,
    //    //    );

    //    //    contact_offset += n;
    //    //}

    //    jac_triplets.num_rows = contact_offset;
    //    jac_triplets.num_cols = surface_vertex_offsets_view.data().last_offset();

    //    let jac: ContactJacobian = jac_triplets.into();
    //    (
    //        jac.into_tensor()
    //            .pruned(|_, _, block| !block.is_zero())
    //            .into_data(),
    //        surface_vertex_offsets,
    //    )
    //}

    pub fn construct_effective_mass_inv(
        &self,
        _solution: &[T],
        _constraint_values: &[f64],
        jac: ContactJacobianView,
        _surface_vertex_offsets: ChunkedView<Offsets<&[usize]>>,
    ) -> Tensor![f64; S S 3 3] {
        //let NLProblem {
        //    //ref frictional_contacts,
        //    ref volume_constraints,
        //    ref state,
        //    ..
        //} = *self;

        // TODO: improve this computation by avoiding intermediate mass matrix computation.

        // Size of the effective mass matrix in each dimension.
        let size = jac.into_tensor().num_cols();

        let blocks = Vec::with_capacity(size);
        let block_indices = Vec::with_capacity(size);

        //let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();

        //let mut contact_offset = 0;

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
        //            let [translation, rotation] = state
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
        //            let [translation, rotation] = state
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

        let mass_inv = tensr::SSBlockMatrix3::<f64>::from_index_iter_and_data(
            block_indices.into_iter(),
            size,
            size,
            blocks,
        );

        let jac_mass = jac.view().into_tensor() * mass_inv.view().transpose();
        (jac_mass.view() * jac.view().into_tensor().transpose()).into_data()
    }

    //    /// Returns true if all friction solves have been completed/converged.
    //    ///
    //    /// This should be the case if and only if all elements in `friction_steps`
    //    /// are zero, which makes the return value simply a convenience.
    //    pub fn update_friction_impulse_global(
    //        &mut self,
    //        solution: &[T],
    //        constraint_values: &[f64],
    //        friction_steps: &mut [u32],
    //    ) -> bool {
    //        if self.frictional_contacts.is_empty() {
    //            return true;
    //        }
    //
    //        //self.update_current_velocity(solution);
    //        let q_cur = self.state.dof.view().map_storage(|dof| dof.cur.q);
    //        let mut ws = self.state.workspace.borrow_mut();
    //        let WorkspaceData { dof, vtx } = &mut *ws;
    //        sync_vel(
    //            &self.state.shells,
    //            dof.view().map_storage(|dof| dof.state.dq),
    //            q_cur,
    //            vtx.view_mut().map_storage(|vtx| vtx.state.vel),
    //        );
    //
    //        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
    //
    //        let (jac, surface_vertex_offsets) =
    //            self.construct_contact_jacobian(solution, constraint_values);
    //
    //        let eff_mass_inv = self.construct_effective_mass_inv(
    //            solution,
    //            constraint_values,
    //            jac.view(),
    //            surface_vertex_offsets.view(),
    //        );
    //
    //        let NLProblem {
    //            ref mut frictional_contacts,
    //            ref volume_constraints,
    //            ref state,
    //            ..
    //        } = *self;
    //
    //        let mut is_finished = true;
    //
    //        let mut constraint_offset = volume_constraints.len();
    //
    //        // Update normals
    //
    //        //        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
    //        //            let obj_cur_pos = state.cur_pos(fc.object_index);
    //        //            let col_cur_pos = state.cur_pos(fc.collider_index);
    //        //            let dq_next = dof.view().map_storage(|dof| dof.dq);
    //        //            let vtx_vel_next = vtx.view().map_storage(|vtx| vtx.state.vel);
    //        //            let obj_vel = state.next_vel(dq_next, vtx_vel_next, fc.object_index);
    //        //            let col_vel = state.next_vel(dq_next, vtx_vel_next, fc.collider_index);
    //        //
    //        //            let n = fc.constraint.borrow().constraint_size();
    //        //            let contact_impulse = Self::contact_impulse_magnitudes(
    //        //                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
    //        //                multiplier_impulse_scale,
    //        //            );
    //        //
    //        //            log::debug!(
    //        //                "Maximum contact impulse: {}",
    //        //                crate::inf_norm(contact_impulse.iter().cloned())
    //        //            );
    //        //            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];
    //        //
    //        //            // TODO: Refactor this low level code out. There needs to be a mechanism to pass rigid
    //        //            // motion data to the constraints since rigid bodies have a special effective mass.
    //        //            let q_cur = q_cur.at(SHELLS_INDEX);
    //        //            let rigid_motion = [
    //        //                state.rigid_motion(fc.object_index),
    //        //                state.rigid_motion(fc.collider_index),
    //        //            ];
    //        //            friction_steps[fc_idx] = fc
    //        //                .constraint
    //        //                .borrow_mut()
    //        //                .update_frictional_contact_impulse(
    //        //                    &contact_impulse,
    //        //                    [obj_cur_pos.view(), col_cur_pos.view()],
    //        //                    [obj_vel.view(), col_vel.view()],
    //        //                    rigid_motion,
    //        //                    potential_values,
    //        //                    friction_steps[fc_idx],
    //        //                );
    //        //
    //        //            is_finished &= friction_steps[fc_idx] == 0;
    //        //            constraint_offset += n;
    //        //        }
    //
    //        //let normals = self.contact_normals();
    //
    //        is_finished
    //    }

    //    pub fn contact_potential(&self) -> VertexData<Vec<f64>> {
    //        let NLProblem {
    //            state,
    //            frictional_contacts,
    //            ..
    //        } = self;
    //
    //        let mut pot = state.build_vertex_data();
    //
    //        let mut workspace_pot = Vec::new();
    //
    //        for fc in frictional_contacts.iter() {
    //            let obj_x0 = state.cur_pos(fc.object_index);
    //            let coll_x0 = state.cur_pos(fc.collider_index);
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
    //                state.mesh_vertex_subset(pot.view_mut(), None, fc.collider_index);
    //            for (&aci, &pot) in active_constraint_indices.iter().zip(workspace_pot.iter()) {
    //                pot_view_mut[aci] += pot;
    //            }
    //        }
    //
    //        pot
    //    }

    //    /// Return the stacked friction corrector impulses: one for each vertex.
    //    pub fn friction_corrector_impulse(&self) -> VertexData3<Vec<f64>> {
    //        let NLProblem {
    //            state,
    //            frictional_contacts,
    //            ..
    //        } = self;
    //
    //        // Create a chunked collection for the output. This essentially combines
    //        // the structure in `pos`, which involved vertices that are not degrees
    //        // of freedom and `prev_x` which involves vertices that ARE degrees of
    //        // freedom.
    //        let mut impulse = state.build_vertex_data3();
    //
    //        let mut obj_imp = Vec::new();
    //        let mut coll_imp = Vec::new();
    //
    //        for fc in frictional_contacts.iter() {
    //            obj_imp.clear();
    //            obj_imp.resize(
    //                state.mesh_surface_vertex_count(fc.object_index),
    //                [T::zero(); 3],
    //            );
    //
    //            coll_imp.clear();
    //            coll_imp.resize(
    //                state.mesh_surface_vertex_count(fc.collider_index),
    //                [T::zero(); 3],
    //            );
    //
    //            let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
    //            let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());
    //
    //            fc.constraint.borrow().add_friction_corrector_impulse(
    //                [
    //                    Subset::all(obj_imp.view_mut()),
    //                    Subset::all(coll_imp.view_mut()),
    //                ],
    //                T::one(),
    //            );
    //
    //            let mut imp = state.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
    //            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp.iter()) {
    //                *imp.as_mut_tensor() += obj_imp.as_tensor().cast::<f64>();
    //            }
    //
    //            let mut imp = state.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
    //            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp.iter()) {
    //                *imp.as_mut_tensor() += coll_imp.as_tensor().cast::<f64>();
    //            }
    //        }
    //        impulse
    //    }

    //    /// Return the stacked friction impulses: one for each vertex.
    //    pub fn friction_impulse(&self) -> VertexData3<Vec<f64>> {
    //        let NLProblem {
    //            state,
    //            frictional_contacts,
    //            ..
    //        } = self;
    //
    //        // Create a chunked collection for the output. This essentially combines
    //        // the structure in `pos`, which involved vertices that are not degrees
    //        // of freedom and `prev_x` which involves vertices that ARE degrees of
    //        // freedom.
    //        let mut impulse = state.build_vertex_data3();
    //
    //        let mut obj_imp = Vec::new();
    //        let mut coll_imp = Vec::new();
    //
    //        for fc in frictional_contacts.iter() {
    //            obj_imp.clear();
    //            obj_imp.resize(
    //                state.mesh_surface_vertex_count(fc.object_index),
    //                [T::zero(); 3],
    //            );
    //
    //            coll_imp.clear();
    //            coll_imp.resize(
    //                state.mesh_surface_vertex_count(fc.collider_index),
    //                [T::zero(); 3],
    //            );
    //
    //            let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
    //            fc.constraint
    //                .borrow()
    //                .add_friction_impulse_to_object(Subset::all(obj_imp.view_mut()), T::one());
    //
    //            let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());
    //            fc.constraint
    //                .borrow()
    //                .add_friction_impulse_to_collider(Subset::all(coll_imp.view_mut()), T::one());
    //
    //            let mut imp = state.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
    //            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp.iter()) {
    //                *imp.as_mut_tensor() += obj_imp.as_tensor().cast::<f64>();
    //            }
    //
    //            let mut imp = state.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
    //            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp.iter()) {
    //                *imp.as_mut_tensor() += coll_imp.as_tensor().cast::<f64>();
    //            }
    //        }
    //        impulse
    //    }

    //    pub fn collider_normals(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
    //        let NLProblem {
    //            state,
    //            frictional_contacts,
    //            ..
    //        } = self;
    //
    //        let mut normals = state.build_vertex_data3();
    //
    //        let mut coll_nml = Vec::new();
    //
    //        for fc in frictional_contacts.iter() {
    //            coll_nml.clear();
    //            coll_nml.resize(
    //                state.mesh_surface_vertex_count(fc.collider_index),
    //                [T::zero(); 3],
    //            );
    //
    //            let mut coll_nml = Chunked3::from_array_slice_mut(coll_nml.as_mut_slice());
    //
    //            fc.constraint
    //                .borrow_mut()
    //                .collider_contact_normals(coll_nml.view_mut());
    //
    //            let coll_nml_view = coll_nml.view();
    //
    //            let mut nml = state.mesh_vertex_subset(normals.view_mut(), None, fc.collider_index);
    //            for (nml, coll_nml) in nml.iter_mut().zip(coll_nml_view.iter()) {
    //                *nml.as_mut_tensor() += coll_nml.as_tensor().cast::<f64>();
    //            }
    //        }
    //        normals
    //    }

    /// Return the stacked contact impulses: one for each vertex.
    //pub fn contact_impulse(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
    //    let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
    //    let NLProblem {
    //        state,
    //        frictional_contacts,
    //        volume_constraints,
    //        ..
    //    } = self;

    //    let mut impulse = state.build_vertex_data3();

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
    //            state.mesh_surface_vertex_count(fc.object_index),
    //            [0.0; 3],
    //        );

    //        coll_imp.clear();
    //        coll_imp.resize(
    //            state.mesh_surface_vertex_count(fc.collider_index),
    //            [0.0; 3],
    //        );

    //        let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
    //        let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());

    //        let obj_x0 = state.cur_pos(fc.object_index);
    //        let coll_x0 = state.cur_pos(fc.collider_index);

    //        fc.constraint.borrow_mut().add_contact_impulse(
    //            [obj_x0.view(), coll_x0.view()],
    //            &contact_impulse,
    //            [obj_imp.view_mut(), coll_imp.view_mut()],
    //        );

    //        let obj_imp_view = obj_imp.view();
    //        let coll_imp_view = coll_imp.view();

    //        let mut imp = state.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
    //        for (imp, obj_imp) in imp.iter_mut().zip(obj_imp_view.iter()) {
    //            *imp.as_mut_tensor() += *obj_imp.as_tensor();
    //        }

    //        let mut imp =
    //            state.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
    //        for (imp, coll_imp) in imp.iter_mut().zip(coll_imp_view.iter()) {
    //            *imp.as_mut_tensor() += *coll_imp.as_tensor();
    //        }
    //    }
    //    assert_eq!(offset, warm_start.constraint_multipliers.len());
    //    impulse
    //}

    //    fn surface_vertex_areas(&self) -> VertexData<Vec<f64>> {
    //        use geo::ops::Area;
    //        use geo::prim::Triangle;
    //        let mut vertex_areas = self.state.build_vertex_data();
    //        for (idx, solid) in self.state.solids.iter().enumerate() {
    //            let TetMeshSurface { trimesh, indices } = &solid.entire_surface();
    //            for face in trimesh.face_iter() {
    //                let area_third =
    //                    Triangle::from_indexed_slice(face, trimesh.vertex_positions()).area() / 3.0;
    //                *vertex_areas
    //                    .view_mut()
    //                    .isolate(SOLIDS_INDEX)
    //                    .isolate(idx)
    //                    .isolate(indices[face[0]]) += area_third;
    //                *vertex_areas
    //                    .view_mut()
    //                    .isolate(SOLIDS_INDEX)
    //                    .isolate(idx)
    //                    .isolate(indices[face[1]]) += area_third;
    //                *vertex_areas
    //                    .view_mut()
    //                    .isolate(SOLIDS_INDEX)
    //                    .isolate(idx)
    //                    .isolate(indices[face[2]]) += area_third;
    //            }
    //        }
    //        for (idx, shell) in self.state.shells.iter().enumerate() {
    //            for face in shell.trimesh.face_iter() {
    //                let area_third =
    //                    Triangle::from_indexed_slice(face, shell.trimesh.vertex_positions()).area()
    //                        / 3.0;
    //                *vertex_areas
    //                    .view_mut()
    //                    .isolate(SHELLS_INDEX)
    //                    .isolate(idx)
    //                    .isolate(face[0]) += area_third;
    //                *vertex_areas
    //                    .view_mut()
    //                    .isolate(SHELLS_INDEX)
    //                    .isolate(idx)
    //                    .isolate(face[1]) += area_third;
    //                *vertex_areas
    //                    .view_mut()
    //                    .isolate(SHELLS_INDEX)
    //                    .isolate(idx)
    //                    .isolate(face[2]) += area_third;
    //            }
    //        }
    //        vertex_areas
    //    }
    //
    //    fn pressure(&self, contact_impulse: VertexView3<&[f64]>) -> VertexData<Vec<f64>> {
    //        let mut pressure = self.state.build_vertex_data();
    //        let vertex_areas = self.surface_vertex_areas();
    //        for obj_type in 0..2 {
    //            for (imp, areas, pres) in zip!(
    //                contact_impulse.view().at(obj_type).iter(),
    //                vertex_areas.view().at(obj_type).iter(),
    //                pressure.view_mut().isolate(obj_type).iter_mut()
    //            ) {
    //                for (&i, &a, p) in zip!(imp.iter(), areas.iter(), pres.iter_mut()) {
    //                    if a > 0.0 {
    //                        *p += i.as_tensor().norm() / a;
    //                    }
    //                }
    //            }
    //        }
    //
    //        //// DEBUG CODE:
    //        //if self.frictional_contacts.len() == 1 {
    //        //    let fc = &self.frictional_contacts[0];
    //        //    let ObjectData { solids, shells, .. } = &self.state;
    //        //    let [_, mut coll_p] = ObjectData::mesh_vertex_subset_split_mut_impl(
    //        //        pressure.view_mut(),
    //        //        None,
    //        //        [fc.object_index, fc.collider_index],
    //        //        solids,
    //        //        shells
    //        //    );
    //
    //        //    fc.constraint
    //        //        .borrow()
    //        //        .smooth_collider_values(coll_p.view_mut());
    //        //}
    //        pressure
    //    }

    /// Update the solid and shell meshes with relevant simulation data.
    //pub fn update_mesh_data(&mut self) {
    //    let contact_impulse = self.contact_impulse();
    //    let friction_impulse = self.friction_impulse();
    //    let friction_corrector_impulse = self.friction_corrector_impulse();
    //    let pressure = self.pressure(contact_impulse.view());
    //    let potential = self.contact_potential();
    //    let normals = self.collider_normals();
    //    for (idx, solid) in self.state.solids.iter_mut().enumerate() {
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
    //    for (idx, shell) in self.state.shells.iter_mut().enumerate() {
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
    pub fn print_jacobian_svd(&self, values: &[T]) {
        use na::DMatrix;

        if values.is_empty() {
            return;
        }

        let (rows, cols) = self.jacobian_indices();

        let n = self.num_variables();
        let nrows = n;
        let ncols = n;
        let mut jac = DMatrix::<f64>::zeros(nrows, ncols);
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            if row < nrows && col < ncols {
                jac[(row as usize, col as usize)] += v.to_f64().unwrap();
            }
        }

        write_jacobian_img(&jac, *self.iter_counter.borrow() as u32);

        use std::io::Write;

        let mut f =
            std::fs::File::create(format!("./out/jac_{}.jl", self.iter_counter.borrow())).unwrap();
        writeln!(&mut f, "jac = [").ok();
        for r in 0..nrows {
            for c in 0..ncols {
                if jac[(r, c)] != 0.0 {
                    write!(&mut f, "{:17.9e}", jac[(r, c)]).ok();
                } else {
                    write!(&mut f, "    0    ",).ok();
                }
            }
            writeln!(&mut f, ";").ok();
        }
        writeln!(&mut f, "]").ok();

        let svd = na::SVD::new(jac, false, false);
        let s: &[f64] = svd.singular_values.data.as_slice();
        let cond = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        writeln!(&mut f, "cond_jac = {}", cond).ok();
    }

    //#[allow(dead_code)]
    //pub fn print_jacobian_svd(&self, values: &[f64]) {
    //    use na::{base::storage::Storage, DMatrix};

    //    if values.is_empty() {
    //        return;
    //    }

    //    let (rows, cols) = self.jacobian_indices();

    //    let mut hess = DMatrix::<f64>::zeros(self.num_variables(), self.num_variables());
    //    for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
    //        hess[(row as usize, col as usize)] += v;
    //    }

    //    let svd = na::SVD::new(hess, false, false);
    //    let s: &[f64] = Storage::as_slice(&svd.singular_values.data);
    //    let cond_hess = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
    //        / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    //    log::debug!("Condition number of hessian is {}", cond_hess);
    //}

    /*
     * End of debugging functions
     */

    /// Conservatively estimates the number of non-zeros in the Jacobian.
    fn jacobian_nnz(&self) -> usize {
        let mut num = 0;
        {
            let solid = &self.state.borrow().solid;
            num += 2 * solid.elasticity::<T>().energy_hessian_size()
                - solid.elasticity::<T>().num_hessian_diagonal_nnz()
                + if !self.is_static() {
                    2 * solid.inertia().energy_hessian_size()
                        - solid.inertia().num_hessian_diagonal_nnz()
                } else {
                    0
                };
        }
        {
            let shell = &self.state.borrow().shell;
            num += 2 * shell.elasticity::<T>().energy_hessian_size()
                - shell.elasticity::<T>().num_hessian_diagonal_nnz()
                + if !self.is_static() {
                    2 * shell.inertia().energy_hessian_size()
                        - shell.inertia().num_hessian_diagonal_nnz()
                } else {
                    0
                };
        }
        for vc in self.volume_constraints.iter() {
            num +=
                2 * vc.borrow().constraint_hessian_size() - vc.borrow().num_hessian_diagonal_nnz()
        }

        //for fc in self.frictional_contacts.iter() {
        //    //TODO: add frictional contact counts
        //    //if !self.is_rigid(fc.object_index) {
        //    //    num += fc.constraint.borrow().object_constraint_hessian_size();
        //    //}
        //    //if !self.is_rigid(fc.collider_index) {
        //    //    num += fc.constraint.borrow().collider_constraint_hessian_size();
        //    //}
        //}
        num
    }

    /// Compute the momentum difference of the problem.
    ///
    /// For the velocity part of the force balance equation `M dv/dt - f(q, v) = 0`,
    /// this represents `M dv`.
    /// Strictly speaking this is equal to the momentum difference `M (v_+ - v_-)`.
    fn add_momentum_diff<S: Real>(
        &self,
        state: ResidualState<&[T], &[S], &mut [S]>,
        solid: &TetSolid,
        shell: &TriShell,
    ) {
        let ResidualState { cur, next, r } = state;
        solid.inertia().add_energy_gradient(cur.vel, next.vel, r);
        shell.inertia().add_energy_gradient(cur.vel, next.vel, r);
    }

    /// Compute contact violation: `max(0, -min(d(q)))`
    pub fn contact_violation(&self, constraint: &[T]) -> T {
        Float::max(
            T::zero(),
            -*constraint
                .iter()
                // If there is a NaN, propagate it down so it will appear in the violation.
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                .unwrap_or(&T::infinity()),
        )
    }
    pub fn contact_constraint(&self, _v: &[T]) -> Chunked<Vec<T>> {
        return Chunked::from_offsets(vec![0], vec![]);
        //let WorkspaceData { dof, vtx } = &mut *self.state.workspace.borrow_mut();
        //let GeneralizedWorkspace {
        //    state: GeneralizedState { q, .. },
        //    ..
        //} = dof.view_mut().into_storage();
        //let VertexWorkspace {
        //    state: VertexState { pos, .. },
        //    ..
        //} = vtx.view_mut().into_storage();

        //// Integrate position.
        //let q_cur = self.state.dof.storage().cur.q.as_slice();
        //self.state.be_step(v, self.time_step(), q_cur, q);

        //let dof = self.state.dof.view();
        //let vtx = self.state.vtx.view();
        //let q = dof.map_storage(|_| &*q);
        //let mut pos = vtx.map_storage(|_| &mut *pos);

        //// Transfer position data to vertex position state (relevant for rigid bodies)
        //sync_pos(&self.state.shells, q, pos.view_mut());

        //let pos = pos.view(); // Convert to a read-only reference.

        //let mut constraint = Chunked::new();
        //for (i, fc) in self.frictional_contacts.iter().enumerate() {
        //    let mut fc_constraint = fc.constraint.borrow_mut();

        //    // Compute constraint (i.e. distance).
        //    let n = fc_constraint.constraint_size();
        //    constraint.push_iter(std::iter::repeat(T::zero()).take(n));

        //    let obj_pos = self.state.mesh_vertex_subset(q, pos, fc.object_index);
        //    let col_pos = self.state.mesh_vertex_subset(q, pos, fc.collider_index);
        //    fc_constraint.constraint([obj_pos, col_pos], constraint.view_mut().isolate(i));
        //}
        //constraint
    }

    //fn subtract_friction_and_contact_forces<S: Real>(
    //    &self,
    //    q: &[S],
    //    dq: &[S],
    //    pos: &mut [S],
    //    vel: &mut [S],
    //    f: &mut [S],
    //    vfc: &mut [S],
    //    frictional_contacts: &[FrictionalContactConstraint<S>],
    //    local_lambda: &mut Vec<S>,
    //) {
    //    let dof = self.state.dof.view();
    //    let vtx = self.state.vtx.view();
    //    let mut f = dof.map_storage(|_| f);
    //    let q = dof.map_storage(|_| q);
    //    let dq = dof.map_storage(|_| dq);
    //    let pos = vtx.map_storage(|_| &*pos);
    //    let vel = vtx.map_storage(|_| &*vel);
    //    let mut vfc = vtx.map_storage(|_| vfc);

    //    let delta = self.delta as f32;
    //    let kappa = self.kappa as f32;
    //    let mut dist = Vec::new();

    //    for fc in frictional_contacts.iter() {
    //        // Since add_friction_impulse is looking for a valid gradient, this
    //        // must involve only vertices that can change.
    //        debug_assert!(match fc.object_index {
    //            SourceObject::Solid(_, _) => true,
    //            SourceObject::Shell(i) => match self.state.shells[i].data {
    //                ShellData::Fixed { .. } => match fc.collider_index {
    //                    SourceObject::Solid(_, _) => true,
    //                    SourceObject::Shell(i) => {
    //                        !matches!(self.state.shells[i].data, ShellData::Fixed { .. })
    //                    }
    //                },
    //                ShellData::Rigid { .. } => true,
    //                _ => true,
    //            },
    //        });

    //        let mut fc_constraint = fc.constraint.borrow_mut();

    //        // Compute Contact forces.

    //        // Compute constraint (i.e. distance).
    //        let n = fc_constraint.constraint_size();
    //        local_lambda.clear();
    //        local_lambda.resize(n, S::zero());

    //        let obj_pos = self.state.mesh_vertex_subset(q, pos, fc.object_index);
    //        let col_pos = self.state.mesh_vertex_subset(q, pos, fc.collider_index);

    //        fc_constraint.constraint([obj_pos, col_pos], local_lambda);

    //        dist.clear();
    //        dist.extend_from_slice(&local_lambda);

    //        compute_contact_force_magnitude(local_lambda, delta, kappa);

    //        // Compute Friction forces.
    //        let obj_vel = self.state.mesh_vertex_subset(dq, vel, fc.object_index);
    //        let col_vel = self.state.mesh_vertex_subset(dq, vel, fc.collider_index);
    //        let friction = fc_constraint.compute_friction_impulse(
    //            local_lambda,
    //            [obj_pos, col_pos],
    //            [obj_vel, col_vel],
    //            &dist,
    //        );

    //        // Compute lambda * constraint Jacobian (notice the order) and
    //        // *subtract* it from f (or vfc).
    //        let obj_j_blocks = fc_constraint
    //            .object_constraint_jacobian_blocks_par_iter()
    //            .collect::<Vec<_>>();
    //        let mut obj_fc =
    //            self.state
    //                .mesh_vertex_subset(f.view_mut(), vfc.view_mut(), fc.object_index);

    //        // Subtract contact force on object.
    //        for &(row, col, j) in obj_j_blocks.iter() {
    //            *obj_fc[col].as_mut_tensor() -= *j.as_tensor() * local_lambda[row];
    //        }

    //        // Subtract friction force force on object.
    //        if let Some((obj_f, _)) = friction.as_ref() {
    //            for (obj_f, fc) in zip!(obj_f.iter(), obj_fc.iter_mut()) {
    //                *fc.as_mut_tensor() -= obj_f.as_tensor();
    //            }
    //        }

    //        let col_j_blocks = fc_constraint
    //            .collider_constraint_jacobian_blocks_par_iter()
    //            .collect::<Vec<_>>();
    //        let mut col_fc =
    //            self.state
    //                .mesh_vertex_subset(f.view_mut(), vfc.view_mut(), fc.collider_index);

    //        // Subtract contact force on collider.
    //        for &(row, col, j) in col_j_blocks.iter() {
    //            *col_fc[col].as_mut_tensor() -= *j.as_tensor() * local_lambda[row];
    //        }

    //        // Subtract friction force force on collider.
    //        if let Some((_, col_f)) = friction.as_ref() {
    //            for (i, col_f, _) in col_f.iter() {
    //                *col_fc[i].as_mut_tensor() -= col_f.as_tensor();
    //            }
    //        }

    //        // Update f with results in vfc.
    //        self.state
    //            .transfer_velocity_vtx_to_dof(fc.object_index, vfc.view_mut(), f.view_mut());
    //        self.state.transfer_velocity_vtx_to_dof(
    //            fc.collider_index,
    //            vfc.view_mut(),
    //            f.view_mut(),
    //        );
    //    }
    //}

    /// Compute the acting force for the problem.
    ///
    /// For the velocity part of the force balance equation
    /// `M dv/dt - f(q,v) = 0`, this function subtracts `f(q,v)`.
    fn subtract_force<S: Real>(
        &self,
        state: ResidualState<&[T], &[S], &mut [S]>,
        solid: &TetSolid,
        shell: &TriShell,
        //vfc: &mut [S],
        //frictional_contacts: &[FrictionalContactConstraint<S>],
        //lambda: &mut Vec<S>,
    ) {
        let ResidualState { cur, next, r } = state;

        solid.elasticity().add_energy_gradient(cur.pos, next.pos, r);
        solid
            .gravity(self.gravity)
            .add_energy_gradient(cur.pos, next.pos, r);
        shell.elasticity().add_energy_gradient(cur.pos, next.pos, r);
        shell
            .gravity(self.gravity)
            .add_energy_gradient(cur.pos, next.pos, r);

        //self.subtract_friction_and_contact_forces(
        //    q,
        //    dq,
        //    pos,
        //    vel,
        //    f.storage_mut(),
        //    vfc,
        //    frictional_contacts,
        //    lambda,
        //);

        debug_assert!(r.iter().all(|r| r.is_finite()));
    }

    /// Compute the backward Euler residual with automatic differentiation.
    fn be_residual_autodiff(&self) {
        let state = &mut *self.state.borrow_mut();

        {
            // Integrate position.
            State::be_step(state.step_state_ad(), self.time_step());
        }
        state.update_vertices_ad();

        {
            let State {
                vtx, solid, shell, ..
            } = state;

            // Clear residual vector.
            vtx.residual_ad
                .storage_mut()
                .iter_mut()
                .for_each(|x| *x = ad::F::zero());

            self.subtract_force(
                vtx.residual_state_ad().into_storage(),
                solid,
                shell, //vtx_state, vfc, &self.frictional_contacts_ad, lambda,
            );
        }

        let mut res_state = state.vtx.residual_state_ad();
        let r = &mut res_state.storage_mut().r;
        *r.as_mut_tensor() *= ad::FT::cst(T::from(self.time_step()).unwrap());

        if !self.is_static() {
            self.add_momentum_diff(res_state.into_storage(), &state.solid, &state.shell);
        }

        // Transfer residual to degrees of freedom.
        state
            .dof
            .view_mut()
            .isolate(VERTEX_DOFS)
            .r_ad
            .iter_mut()
            .zip(state.vtx.residual_ad.storage().iter())
            .for_each(|(dof_r, vtx_r)| {
                *dof_r = *vtx_r;
            })
    }

    /// Compute the backward Euler residual.
    fn be_residual(&self, dq: &[T], r: &mut [T]) {
        let state = &mut *self.state.borrow_mut();

        {
            let step_state = state.step_state(dq);
            // Integrate position.
            State::be_step(step_state, self.time_step());
        }

        state.update_vertices(dq);

        // Clear residual vector.
        state
            .vtx
            .residual
            .storage_mut()
            .iter_mut()
            .for_each(|x| *x = T::zero());

        self.subtract_force(
            state.vtx.residual_state().into_storage(),
            &state.solid,
            &state.shell, //vfc, &self.frictional_contacts, lambda,
        );

        let mut res_state = state.vtx.residual_state();
        *res_state.storage_mut().r.as_mut_tensor() *= T::from(self.time_step()).unwrap();

        if !self.is_static() {
            self.add_momentum_diff(res_state.into_storage(), &state.solid, &state.shell);
        }

        // Transfer residual to degrees of freedom.
        r.iter_mut()
            .take(state.dof.view().isolate(VERTEX_DOFS).len())
            .zip(state.vtx.residual.storage().iter())
            .for_each(|(dof_r, vtx_r)| {
                *dof_r = *vtx_r;
            })
    }

    fn jacobian_indices(&self) -> (Vec<usize>, Vec<usize>) {
        let mut rows = vec![0; self.jacobian_nnz()];
        let mut cols = vec![0; self.jacobian_nnz()];
        let mut count = 0; // Constraint counter

        let state = self.state.borrow();

        // Add energy indices
        let solid = &state.solid;
        let elasticity = solid.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity
            .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);

        count += n;

        if !self.is_static() {
            let inertia = solid.inertia();
            let n = inertia.energy_hessian_size();
            inertia
                .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
            count += n;
        }

        let shell = &state.shell;
        let elasticity = shell.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity
            .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);

        count += n;

        if !self.is_static() {
            let inertia = shell.inertia();
            let n = inertia.energy_hessian_size();
            inertia
                .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
            count += n;
        }

        // Add volume constraint indices
        for vc in self.volume_constraints.iter() {
            for MatrixElementIndex { row, col } in vc.borrow().constraint_hessian_indices_iter() {
                rows[count] = row;
                cols[count] = col;
                count += 1;
            }
        }

        // Duplicate off-diagonal indices to form a complete matrix.
        let (rows_begin, rows_end) = rows.split_at_mut(count);
        let (cols_begin, cols_end) = cols.split_at_mut(count);

        // TODO: Parallelize this function
        let num_off_diagonals = rows_begin
            .iter()
            .zip(cols_begin.iter())
            .filter(|(&r, &c)| r != c)
            .zip(rows_end.iter_mut().zip(cols_end.iter_mut()))
            .map(|((&r, &c), (out_r, out_c))| {
                *out_c = r;
                *out_r = c;
            })
            .count();
        count += num_off_diagonals;

        // TODO: Add frictional contact indices

        assert_eq!(count, rows.len());
        assert_eq!(count, cols.len());

        (rows, cols)
    }

    fn be_jacobian_values(
        &self,
        dq: &[T],
        _r: &[T],
        rows: &[usize],
        cols: &[usize],
        vals: &mut [T],
    ) {
        let state = &mut *self.state.borrow_mut();
        State::be_step(state.step_state(dq), self.time_step());
        state.update_vertices(dq);

        let mut count = 0; // Values counter
                           //let mut coff = 0; // Constraint offset

        let dt = T::from(self.time_step()).unwrap();

        // Constraint scaling
        //let c_scale = dt * dt;

        // Multiply energy hessian by objective factor and scaling factors.
        let factor = T::from(self.impulse_inv_scale()).unwrap();

        let State {
            vtx, solid, shell, ..
        } = state;

        let ResidualState { cur, next, .. } = vtx.residual_state().into_storage();
        let elasticity = solid.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity.energy_hessian_values(
            cur.pos,
            next.pos,
            dt * dt * factor,
            &mut vals[count..count + n],
        );
        count += n;

        if !self.is_static() {
            let inertia = solid.inertia();
            let n = inertia.energy_hessian_size();
            inertia.energy_hessian_values(cur.vel, next.vel, factor, &mut vals[count..count + n]);
            count += n;
        }

        let elasticity = shell.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity.energy_hessian_values(
            cur.pos,
            next.pos,
            dt * dt * factor,
            &mut vals[count..count + n],
        );
        count += n;

        if !self.is_static() {
            let inertia = shell.inertia();
            let n = inertia.energy_hessian_size();
            inertia.energy_hessian_values(cur.vel, next.vel, factor, &mut vals[count..count + n]);
            count += n;
        }

        // Duplicate off-diagonal entries.
        let (vals_begin, vals_end) = vals.split_at_mut(count);
        let mut i = 0;
        for (&r, &c, &val) in zip!(rows.iter(), cols.iter(), vals_begin.iter()) {
            if r != c {
                vals_end[i] = val;
                i += 1;
            }
        }
        count += i;

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
        //self.print_jacobian_svd(vals);
        *self.iter_counter.borrow_mut() += 1;

        debug_assert!(vals.iter().all(|x| x.is_finite()));
    }

    fn be_jacobian_product(&self, v: &[T], p: &[T], jp: &mut [T]) {
        {
            let State { dof, .. } = &mut *self.state.borrow_mut();
            let dq_ad = &mut dof.storage_mut().next_ad.dq;
            // Construct automatically differentiable vector of velocities.
            // The residual will be computed into the state workspace vector r_ad.
            for (&v, &p, dq_ad) in zip!(v.iter(), p.iter(), dq_ad.iter_mut()) {
                *dq_ad = ad::FT::new(v, p);
            }
            // End of dynamic mutable borrow.
        }

        self.be_residual_autodiff();

        let State { dof, .. } = &mut *self.state.borrow_mut();
        for (jp, r_ad) in jp.iter_mut().zip(dof.storage_mut().r_ad.iter()) {
            *jp = T::from(r_ad.deriv()).unwrap();
        }
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
    fn jacobian_values(&self, x: &[T], r: &[T], rows: &[usize], cols: &[usize], values: &mut [T]);

    /// Compute the residual and simultaneously its Jacobian product with the given vector `p`.
    fn jacobian_product(&self, x: &[T], p: &[T], r: &[T], jp: &mut [T]);
}

/// A Mixed complementarity problem is a non-linear problem subject to box constraints.
pub trait MixedComplementarityProblem<T: Real>: NonLinearProblem<T> {
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
    fn update_bounds(&self, _l: &mut [f64], _u: &mut [f64]) {}
}

/// Prepare the problem for Newton iterations.
impl<T: Real64> NonLinearProblem<T> for NLProblem<T> {
    #[inline]
    fn num_variables(&self) -> usize {
        self.state.borrow().dof.storage().len()
    }

    #[inline]
    fn residual(&self, x: &[T], r: &mut [T]) {
        self.be_residual(x, r);
    }

    #[inline]
    fn jacobian_indices(&self) -> (Vec<usize>, Vec<usize>) {
        NLProblem::jacobian_indices(self)
    }

    #[inline]
    fn jacobian_values(&self, v: &[T], r: &[T], rows: &[usize], cols: &[usize], vals: &mut [T]) {
        self.be_jacobian_values(v, r, rows, cols, vals);
    }
    #[inline]
    fn jacobian_product(&self, v: &[T], p: &[T], _r: &[T], jp: &mut [T]) {
        self.be_jacobian_product(v, p, jp);
    }
}

impl<T: Real64> MixedComplementarityProblem<T> for NLProblem<T> {
    fn initial_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let uv_l = vec![std::f64::NEG_INFINITY; self.num_variables()];
        let uv_u = vec![std::f64::INFINITY; self.num_variables()];

        //debug_assert!(uv_l.iter().all(|&x| x.is_finite()));
        //debug_assert!(uv_u.iter().all(|&x| x.is_finite()));
        (uv_l, uv_u)
    }
}

impl<T: Real> NLProblem<T> {
    /// Constructs a clone of this problem with autodiff variables.
    pub fn clone_as_autodiff(&self) -> NLProblem<ad::F1> {
        let Self {
            state,
            kappa,
            delta,
            volume_constraints,
            gravity,
            time_step,
            iterations,
            initial_residual_error,
            iter_counter,
            max_size,
            max_element_force_scale,
            min_element_force_scale,
            ..
        } = self.clone();

        let state = state.borrow();
        let state = RefCell::new(state.clone_as_autodiff());
        NLProblem {
            state,
            kappa,
            delta,
            volume_constraints,
            gravity,
            time_step,
            iterations,
            initial_residual_error,
            iter_counter,
            max_size,
            max_element_force_scale,
            min_element_force_scale,
        }
    }

    /// Checks that the given problem has a consistent Jacobian imlementation.
    pub(crate) fn check_jacobian(&self, perturb_initial: bool) -> bool {
        log::debug!("Checking Jacobian..");
        use ad::F1 as F;
        let problem = self.clone_as_autodiff();
        let n = problem.num_variables();
        let mut x0 = problem.initial_point();
        if perturb_initial {
            perturb(&mut x0);
        }

        let mut r = vec![F::zero(); n];
        problem.residual(&x0, &mut r);

        let (jac_rows, jac_cols) = problem.jacobian_indices();

        let mut jac_values = vec![F::zero(); jac_rows.len()];
        problem.jacobian_values(&x0, &r, &jac_rows, &jac_cols, &mut jac_values);

        // Build a dense Jacobian.
        let mut jac_ad = vec![vec![0.0; n]; n];
        let mut jac = vec![vec![0.0; n]; n];
        for (&row, &col, &val) in zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter()) {
            if row < n && col < n {
                jac[row][col] += val.value();
            }
        }

        let mut success = true;
        for i in 0..n {
            x0[i] = F::var(x0[i]);
            problem.residual(&x0, &mut r);
            for j in 0..n {
                let res = approx::relative_eq!(
                    jac[i][j],
                    r[j].deriv(),
                    max_relative = 1e-6,
                    epsilon = 1e-10
                );
                jac_ad[i][j] = r[j].deriv();
                if !res {
                    success = false;
                    log::debug!("({}, {}): {} vs. {}", i, j, jac[i][j], r[j].deriv());
                }
            }
            x0[i] = F::cst(x0[i]);
        }

        if !success && n < 15 {
            // Print dense hessian if its small
            log::debug!("Actual:");
            for row in 0..n {
                for col in 0..=row {
                    log::debug!("{:10.2e}", jac[row][col]);
                }
                log::debug!("");
            }

            log::debug!("Expected:");
            for row in 0..n {
                for col in 0..=row {
                    log::debug!("{:10.2e}", jac_ad[row][col]);
                }
                log::debug!("");
            }
        }
        if success {
            log::debug!("No errors during Jacobian check.");
        }
        success
    }
}

pub(crate) fn perturb(x: &mut [ad::F1]) {
    use rand::distributions::Uniform;
    use rand::prelude::*;
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-0.1, 0.1);
    x.iter_mut().for_each(move |x| *x += rng.sample(range));
}

#[cfg(test)]
mod tests {
    use crate::fem::nl::*;
    use crate::objects::*;
    use crate::test_utils::*;
    use crate::Mesh;

    /// Verifies that the problem jacobian is implemented correctly.
    #[test]
    fn nl_problem_jacobian_one_tet() {
        init_logger();
        let mut solver_builder = SolverBuilder::new(sample_params());
        solver_builder
            .set_mesh(Mesh::from(make_one_tet_mesh()))
            .set_materials(vec![solid_material().into()]);
        let problem = solver_builder.build_problem::<f64>().unwrap();
        assert!(problem.check_jacobian(true));
    }

    #[test]
    fn nl_problem_jacobian_three_tets() {
        init_logger();
        let mut solver_builder = SolverBuilder::new(sample_params());
        solver_builder
            .set_mesh(Mesh::from(make_three_tet_mesh()))
            .set_materials(vec![solid_material().into()]);
        let problem = solver_builder.build_problem::<f64>().unwrap();
        assert!(problem.check_jacobian(true));
    }

    fn solid_material() -> SolidMaterial {
        SolidMaterial::new(0)
            .with_elasticity(ElasticityParameters {
                lambda: 5.4,
                mu: 263.1,
                model: ElasticityModel::NeoHookean,
            })
            .with_density(10.0)
            .with_damping(1.0, 0.01)
    }

    fn sample_params() -> SimParams {
        SimParams {
            gravity: [0.0, -9.81, 0.0],
            time_step: Some(0.01),
            clear_velocity: false,
            tolerance: 1e-2,
            max_iterations: 1,
            linsolve_tolerance: 1e-3,
            max_linsolve_iterations: 300,
            line_search: LineSearch::default_backtracking(),
            jacobian_test: false,
            contact_tolerance: 0.001,
            friction_tolerance: 0.001,
        }
    }
}
