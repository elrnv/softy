use crate::attrib_defines::*;
use crate::constraint::*;
use crate::constraints::{volume::VolumeConstraint, ContactConstraint};
use crate::energy::*;
use crate::energy_models::{elasticity::*, gravity::Gravity, inertia::Inertia};
use crate::matrix::*;
use crate::objects::*;
use crate::PointCloud;
use geo::math::Vector3;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use ipopt::{self, Number};
use std::cell::RefCell;
use utils::{aref::*, soap::*, zip};

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

/// The index of the object subject to the appropriate contact constraint.
/// This enum helps us map from the particular contact constraint to the
/// originating simulation object (shell or solid).
#[derive(Copy, Clone, Debug)]
pub enum SourceIndex {
    Solid(usize),
    Shell(usize),
}

impl SourceIndex {
    pub fn get(&self) -> usize {
        match self {
            SourceIndex::Solid(idx) | SourceIndex::Shell(idx) => *idx,
        }
    }
}

/// A struct that keeps track of which objects are being affected by the contact
/// constraints.
pub struct FrictionalContactConstraint {
    pub object_index: SourceIndex,
    pub collider_index: SourceIndex,
    pub constraint: Box<dyn ContactConstraint>,
}

/// A `Vertex` is a single element of the `VertexSet`.
#[derive(Clone, Debug)]
pub struct Vertex {
    prev_pos: [f64; 3],
    prev_vel: [f64; 3],
    cur_pos: [f64; 3],
    cur_vel: [f64; 3],
}

/// This set collects all simulated vertex data. Meshes that are not simulated are excluded.
/// The data is chunked into solids/shells, then into a subset of individual
/// meshes, and finally into x,y,z coordinates. Rigid shells have 6 degrees of
/// freedom, 3 for position and 3 for rotation, so a rigid shell will correspond
/// to 6 floats in each of these vectors. This can be more generally
/// interpreted as generalized coordinates.
#[derive(Clone, Debug)]
pub struct VertexSet {
    /// Generalized coordinates from the previous time step. Sometimes referred
    /// to as `q` in literature.
    pub prev_x: Chunked<Chunked<Chunked3<Vec<f64>>>>,
    /// Generalized coordinate derivative from the previous time step. Referred
    /// to as `\dot{q}` in literature.
    pub prev_v: Chunked<Chunked<Chunked3<Vec<f64>>>>,
    /// Workspace vector to compute intermediate displacements.
    pub cur_x: RefCell<Chunked<Chunked<Chunked3<Vec<f64>>>>>,
    /// Workspace vector to rescale variable values before performing computations on them.
    pub cur_v: RefCell<Chunked<Chunked<Chunked3<Vec<f64>>>>>,

    /// Workspace positions for all meshes. These are used to pass concrete
    /// positions (as opposed to generalized coordinates) to constraint functions.
    /// If the positions coincide with generalized coordinates for a particular
    /// object, then these variables may be omitted and `prev_x` or `cur_x` are
    /// used directly.
    pub pos: RefCell<Chunked<Chunked<Chunked3<Vec<f64>>>>>,
    /// Workspace velocities for all meshes. These are used to pass concrete
    /// velocities (as opposed to generalized coordinates) to constraint functions.
    /// If the positions coincide with generalized coordinates for a particular
    /// object, then these variables may be omitted and `prev_v` or `cur_v` are
    /// used directly.
    pub vel: RefCell<Chunked<Chunked<Chunked3<Vec<f64>>>>>,
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NonLinearProblem {
    /// A set of vertices with attached data per vertex for all simulated meshes combined.
    pub vertex_set: VertexSet,
    /// Tetrahedron mesh representing a soft solid computational domain.
    pub solids: Vec<TetMeshSolid>,
    /// Shell object represented by a triangle mesh.
    pub shells: Vec<TriMeshShell>,
    /// One way contact constraints between a pair of objects.
    pub frictional_contacts: Vec<FrictionalContactConstraint>,
    /// Constraint on the total volume.
    pub volume_constraints: Vec<(usize, VolumeConstraint)>,
    /// Gravitational potential energy.
    pub gravity: [f64; 3],
    /// The time step defines the amount of time elapsed between steps (calls to `advance`).
    pub time_step: f64,
    /// Displacement bounds. This controls how big of a step we can take per vertex position
    /// component. In other words the bounds on the inf norm for each vertex displacement.
    pub displacement_bound: Option<f64>,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<dyn FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Solution data. This is kept around for warm starts.
    pub warm_start: Solution,
    pub initial_residual_error: f64,
    pub iter_counter: RefCell<usize>,
}

impl NonLinearProblem {
    pub fn scale(&self) -> f64 {
        if self.time_step > 0.0 {
            1.0 / self.time_step
        } else {
            1.0
        }
    }

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
        // Reset count
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

    /// Get the minimum contact radius among all contact problems. If there are
    /// no contacts, simply return `None`.
    pub fn min_contact_radius(&self) -> Option<f64> {
        self.frictional_contacts
            .iter()
            .map(|fc| fc.constraint.contact_radius())
            .min_by(|a, b| a.partial_cmp(b).expect("Detected NaN contact radius"))
    }

    /// Update the solid meshes with the given points.
    pub fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        // All solids have prev_x coincident with pos so we use prev_x directly here.
        let mut prev_pos = self.vertex_set.prev_x.view_mut().at_mut(0);

        // All solids are simulated, so the input point set must have the same
        // size as our internal vertex set. If these are mismatched, then there
        // was an issue with constructing the solid meshes. This may not
        // necessarily be an error, we are just being conservative here.
        if pts.num_vertices() != prev_pos.view().data().len() {
            // We got an invalid point cloud
            return Err(crate::Error::SizeMismatch);
        }

        let new_pos = pts.vertex_positions();

        // Get the tetmesh and prev_pos so we can update the fixed vertices.
        for (solid, mut prev_pos) in self.solids.iter().zip(prev_pos.iter_mut()) {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = solid
                .tetmesh
                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let pts_iter = source_index_iter.map(|&idx| new_pos[idx]);

            // Only update fixed vertices, if no such attribute exists, return an error.
            let fixed_iter = solid
                .tetmesh
                .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
            prev_pos
                .iter_mut()
                .zip(pts_iter)
                .zip(fixed_iter)
                .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
                .for_each(|(pos, new_pos)| *pos = new_pos);
        }
        Ok(())
    }

    /// Update the shell meshes with the given points.
    pub fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        let mut prev_pos = self.vertex_set.prev_pos.view_mut().at_mut(1);

        let new_pos = pts.vertex_positions();

        // Get the tetmesh and prev_pos so we can update the fixed vertices.
        for (shell, mut prev_pos) in self.shells.iter().zip(prev_pos.iter_mut()) {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = shell
                .trimesh
                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let pts_iter = source_index_iter.map(|&idx| new_pos[idx]);

            // Only update fixed vertices, if no such attribute exists, return an error.
            let fixed_iter = shell
                .trimesh
                .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
            prev_pos
                .iter_mut()
                .zip(pts_iter)
                .zip(fixed_iter)
                .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
                .for_each(|(pos, new_pos)| *pos = new_pos);
        }
        Ok(())
    }

    /// Compute the set of currently active constraints into the given `Vec`.
    pub fn compute_active_constraint_set(&self, active_set: &mut Vec<usize>) {
        for i in 0..self.volume_constraints.len() {
            active_set.push(i);
        }

        for FrictionalContactConstraint { ref constraint, .. } in self.frictional_contacts.iter() {
            let fc_active_constraints = constraint.active_constraint_indices().unwrap_or_default();
            let offset = active_set.len();
            for c in fc_active_constraints.into_iter() {
                active_set.push(c + offset);
            }
        }
    }
    /// Get the set of currently active constraints.
    pub fn active_constraint_set(&self) -> Vec<usize> {
        let mut active_set = Vec::new();
        self.compute_active_constraint_set(&mut active_set);
        active_set
    }

    pub fn clear_friction_impulses(&mut self) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint.clear_frictional_contact_impulse();
        }
    }

    /// Restore the constraint set.
    pub fn reset_constraint_set(&mut self) -> bool {
        self.update_constraint_set(None)
    }

    /// Check if the given constraint set is the same as the current one.
    pub fn is_same_as_constraint_set(&self, other_set: &[usize]) -> bool {
        let cur_set = self.active_constraint_set();
        cur_set.len() == other_set.len()
            && cur_set
                .into_iter()
                .zip(other_set.iter())
                .all(|(cur, &other)| cur == other)
    }

    /// Translate a surface mesh coordinate index into the corresponding
    /// simulation coordinate index in our global array.
    fn source_coordinates(
        &self,
        object_index: SourceIndex,
        collider_index: SourceIndex,
        coord: usize,
    ) -> usize {
        let surf_vtx_idx = coord / 3;

        // Retrieve tetmesh coordinates when it's determined that the source is a solid tetmesh.
        let tetmesh_coordinates = |mesh_index| {
            let offset = self
                .vertex_set
                .prev_vel
                .view()
                .at(0)
                .offset_value(mesh_index);
            3 * (offset + self.solids[mesh_index].surface().indices[surf_vtx_idx]) + coord % 3
        };

        // Retrieve trimesh coordinates when it's determined that the source is a shell trimesh.
        let trimesh_coordinates = |mesh_index| {
            3 * self
                .vertex_set
                .prev_vel
                .view()
                .at(1)
                .offset_value(mesh_index)
                + coord
        };

        let num_object_surface_indices: usize;

        match object_index {
            SourceIndex::Solid(obj_i) => {
                num_object_surface_indices = self.solids[obj_i].surface().indices.len();
                if surf_vtx_idx < num_object_surface_indices {
                    return tetmesh_coordinates(obj_i);
                }
            }
            SourceIndex::Shell(obj_i) => {
                num_object_surface_indices = self.shells[obj_i].trimesh.num_vertices();
                if surf_vtx_idx < num_object_surface_indices {
                    return trimesh_coordinates(obj_i);
                }
            }
        }

        // If we haven't returned yet, this means that surf_vtx_idx corresponds
        // to collider surface.

        // Adjust surf_vtx_idx to be in the right range:
        let surf_vtx_idx = surf_vtx_idx - num_object_surface_indices;

        match collider_index {
            SourceIndex::Solid(coll_i) => {
                assert!(surf_vtx_idx < self.solids[coll_i].surface().indices.len());
                tetmesh_coordinates(coll_i)
            }
            SourceIndex::Shell(coll_i) => {
                assert!(surf_vtx_idx < self.shells[coll_i].trimesh.num_vertices());
                trimesh_coordinates(coll_i)
            }
        }
    }

    /// Index into a chunked set of per vertex data as defined in `vertex_set` for a single mesh.
    fn mesh_vertex_subset<'s, 'x, T: Clone + 'x>(
        solids: &'s [TetMeshSolid],
        x: ChunkedView<'x, ChunkedView<'x, Chunked3<&'x [T]>>>,
        source: SourceIndex,
    ) -> Subset<Chunked3<&'x [T]>> {
        match source {
            SourceIndex::Solid(i) => Subset::from_unique_ordered_indices(
                solids[i].surface().indices.to_vec(),
                x.at(0).at(i),
            ),
            SourceIndex::Shell(i) => Subset::all(x.at(1).at(i)),
        }
    }

    // TODO: refactor this function together with the function above.
    fn mesh_vertex_subset_mut<'s, 'x, T: Clone + 'x>(
        solids: &'s [TetMeshSolid],
        mut x: ChunkedView<'x, ChunkedView<'x, Chunked3<&'x mut [T]>>>,
        source: SourceIndex,
    ) -> Subset<Chunked3<&'x mut [T]>> {
        match source {
            SourceIndex::Solid(i) => Subset::from_unique_ordered_indices(
                solids[i].surface().indices.to_vec(),
                x.at_mut(0).at_mut(i),
            ),
            SourceIndex::Shell(i) => Subset::all(x.at_mut(1).at_mut(i)),
        }
    }

    fn mesh_surface_vertex_count(
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
        source: SourceIndex,
    ) -> usize {
        match source {
            SourceIndex::Solid(i) => solids[i].surface().indices.len(),
            SourceIndex::Shell(i) => shells[i].trimesh.num_vertices(),
        }
    }

    /// Update all stateful constraints with the most recent data.
    /// Return an estimate if any constraints have changed, though this estimate may have false
    /// negatives.
    pub fn update_constraint_set(&mut self, solution: Option<ipopt::Solution>) -> bool {
        let mut changed = false; // Report if anything has changed to the caller.

        // Update positions with the given solution (if any).
        let x = if let Some(uv) = solution.map(|sol| sol.primal_variables) {
            let v = self.update_current_velocity(uv);
            let mut cur_pos = self.vertex_set.cur_pos.borrow_mut();

            let v = v.view().into_flat();
            let cur_pos = cur_pos.view_mut().into_flat();
            let prev_pos = self.vertex_set.prev_pos.view().into_flat();
            self.integrate_step(v, prev_pos, cur_pos);
            ARef::Cell(self.vertex_set.cur_pos.borrow())
        } else {
            ARef::Plain(&self.vertex_set.prev_pos)
        };

        let NonLinearProblem {
            solids,
            frictional_contacts,
            ..
        } = self;

        for FrictionalContactConstraint {
            object_index,
            collider_index,
            constraint,
        } in frictional_contacts.iter_mut()
        {
            let object_pos = Self::mesh_vertex_subset(solids, x.view(), *object_index);
            let collider_pos = Self::mesh_vertex_subset(solids, x.view(), *collider_index);
            changed |= constraint.update_cache(object_pos.view(), collider_pos.view());
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
        for _ in self.volume_constraints.iter() {
            // Consume the volume constraints if any.
            old_constraint_set.next();
            new_constraint_set.next();
        }

        for fc in self.frictional_contacts.iter_mut() {
            let mut old_set = Vec::new();
            let mut new_set = Vec::new();
            for _ in 0..fc.constraint.num_contacts() {
                old_set.push(old_constraint_set.next().unwrap());
                new_set.push(new_constraint_set.next().unwrap());
            }
            fc.constraint.remap_frictional_contact(&old_set, &new_set);
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

    pub fn apply_frictional_contact_impulse(
        &self,
        mut vel: ChunkedView<ChunkedView<Chunked3<&mut [f64]>>>,
    ) {
        for fc in self.frictional_contacts.iter() {
            let mut object_vel =
                Self::mesh_vertex_subset_mut(&self.solids, vel.view_mut(), fc.object_index);
            fc.constraint
                .add_mass_weighted_frictional_contact_impulse(object_vel.view_mut());
        }
    }

    /// Helper function to compute new positions from a given slice of unscaled velocities.
    /// This function is used for debugging.
    #[allow(dead_code)]
    pub fn compute_step_from_unscaled_velocities(
        &self,
        uv: &[f64],
    ) -> std::cell::Ref<'_, Chunked<Chunked<Chunked3<Vec<f64>>>>> {
        let cur_vel = self.update_current_velocity(uv);
        let mut x1 = self.vertex_set.cur_pos.borrow_mut();

        let cur_vel = cur_vel.view().into_flat();
        let x1 = x1.view_mut().into_flat();
        let x0 = self.vertex_set.prev_pos.view().into_flat();
        self.integrate_step(cur_vel, x0, x1);
        self.vertex_set.cur_pos.borrow()
    }

    /// Commit velocity by advancing the internal state by the given unscaled velocity `uv`.
    /// If `and_velocity` is `false`, then only positions are advance, and velocities are reset.
    /// This emulates a critically damped, or quasi-static simulation.
    pub fn advance(&mut self, uv: &[f64], and_velocity: bool, and_warm_start: bool) {
        let (_old_warm_start, _old_prev_pos, _old_prev_vel) = {
            self.update_current_velocity(uv);
            {
                let mut cur_vel = self.vertex_set.cur_vel.borrow_mut();
                self.apply_frictional_contact_impulse(cur_vel.view_mut());

                self.compute_step(cur_vel.view());
            }

            let VertexSet {
                prev_pos,
                prev_vel,
                cur_vel,
                cur_pos,
                ..
            } = &mut self.vertex_set;

            let cur_vel = cur_vel.borrow();
            let cur_pos = cur_pos.borrow();

            let old_prev_pos = prev_pos.clone();
            let old_prev_vel = prev_vel.clone();

            {
                let prev_pos_flat_view = prev_pos.view_mut().into_flat();
                let prev_vel_flat_view = prev_vel.view_mut().into_flat();

                // Update prev pos
                prev_pos_flat_view
                    .iter_mut()
                    .zip(cur_pos.view().into_flat().iter())
                    .for_each(|(prev, &cur)| *prev = cur);

                // Update prev vel
                if and_velocity {
                    prev_vel_flat_view
                        .iter_mut()
                        .zip(cur_vel.view().into_flat().iter())
                        .for_each(|(prev, &cur)| *prev = cur);
                } else {
                    // Clear velocities. This ensures that any non-zero initial velocities are cleared
                    // for subsequent steps.
                    prev_vel_flat_view.iter_mut().for_each(|v| *v = 0.0);
                }
            }

            let prev_pos = prev_pos.view().view();

            // Update mesh vertex positions
            for (i, solid) in self.solids.iter_mut().enumerate() {
                let verts = solid.tetmesh.vertex_positions_mut();
                verts.copy_from_slice(prev_pos.at(0).at(i).into());
            }
            for (i, shell) in self.shells.iter_mut().enumerate() {
                match shell.material.properties {
                    ShellProperties::Deformable { .. } => {
                        let verts = shell.trimesh.vertex_positions_mut();
                        verts.copy_from_slice(prev_pos.at(1).at(i).into());
                    }
                    ShellProperties::Rigid { .. } => {
                        unimplemented!();
                    }
                    ShellProperties::Fixed => {
                        // Nothing to do here, fixed meshes aren't moved by
                        // simulation.
                    }
                }
            }

            let old_warm_start = self.warm_start.clone();

            (old_warm_start, old_prev_pos, old_prev_vel)
        };

        if !and_warm_start {
            self.reset_warm_start();
        }
    }

    pub fn update_max_step(&mut self, step: f64) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint.update_max_step(step);
        }
    }
    pub fn update_radius_multiplier(&mut self, rad_mult: f64) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint.update_radius_multiplier(rad_mult);
        }
    }

    ///// Revert to the given old solution by the given displacement.
    //pub fn revert_to(
    //    &mut self,
    //    solution: Solution,
    //    old_prev_pos: Vec<Vector3<f64>>,
    //    old_prev_vel: Vec<Vector3<f64>>,
    //) {
    //    {
    //        // Reinterpret solver variables as positions in 3D space.
    //        let mut prev_pos = self.vertex_set.prev_pos.data().borrow_mut();
    //        let mut prev_vel = self.vertex_set.prev_vel.data().borrow_mut();

    //        std::mem::replace(&mut *prev_vel, old_prev_vel);
    //        std::mem::replace(&mut *prev_pos, old_prev_pos);

    //        let mut tetmesh = self.tetmesh.borrow_mut();
    //        let verts = tetmesh.vertex_positions_mut();
    //        verts.copy_from_slice(reinterpret_slice(prev_pos.as_slice()));

    //        std::mem::replace(&mut self.warm_start, solution);
    //    }

    //    // Since we transformed the mesh, we need to invalidate its neighbour data so it's
    //    // recomputed at the next time step (if applicable).
    //    //self.update_constraints();
    //}

    fn compute_constraint_violation(&self, unscaled_vel: &[f64], constraint: &mut [f64]) {
        use ipopt::ConstrainedProblem;
        let mut lower = vec![0.0; constraint.len()];
        let mut upper = vec![0.0; constraint.len()];
        self.constraint_bounds(&mut lower, &mut upper);
        assert!(self.constraint(unscaled_vel, constraint));
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

    fn constraint_violation_norm(&self, unscaled_vel: &[f64]) -> f64 {
        use ipopt::ConstrainedProblem;
        let mut g = vec![0.0; self.num_constraints()];
        self.compute_constraint_violation(unscaled_vel, &mut g);
        crate::inf_norm(g)
    }

    /// Return the constraint violation and whether the neighbourhood data (sparsity) would be
    /// changed if we took this step.
    pub fn probe_contact_constraint_violation(&mut self, solution: ipopt::Solution) -> f64 {
        self.constraint_violation_norm(solution.primal_variables)
    }

    ///// Linearized constraint true violation measure.
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

    ///// Linearized constraint model violation measure.
    //pub fn linearized_constraint_violation_model_l1(&self, dx: &[f64]) -> f64 {
    //    let mut value = 0.0;

    //    if let Some(ref scc) = self.smooth_contact_constraint {
    //        let prev_pos = self.vertex_set.prev_pos.data().borrow();
    //        let x: &[Number] = prev_pos.as_slice();

    //        let n = scc.constraint_size();

    //        let mut g = vec![0.0; n];
    //        scc.constraint(x, dx, &mut g);

    //        let (g_l, g_u) = scc.constraint_bounds();
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

    fn integrate_step(&self, v: &[Number], x0: &[Number], x: &mut [Number]) {
        // In static simulations, velocity is simply displacement.
        let dt = if self.time_step > 0.0 {
            self.time_step
        } else {
            1.0
        };
        assert_eq!(v.len(), x.len());
        assert_eq!(x0.len(), x.len());
        x.iter_mut()
            .zip(x0.iter().zip(v.iter()))
            .for_each(|(x1, (&x0, &v))| *x1 = x0 + v * dt);
    }

    /// A convenience function to integrate the given velocity by the internal time step. For
    /// implicit itegration this boils down to a simple multiply by the time step.
    pub fn compute_step(
        &self,
        v: ChunkedView<ChunkedView<Chunked3<&[Number]>>>,
    ) -> std::cell::Ref<'_, Chunked<Chunked<Chunked3<Vec<f64>>>>> {
        {
            let mut cur_pos = self.vertex_set.cur_pos.borrow_mut();
            let x1 = cur_pos.view_mut().into_flat();
            let x0 = self.vertex_set.prev_pos.view().into_flat();
            self.integrate_step(v.into_flat(), x0, x1);
        }
        self.vertex_set.cur_pos.borrow()
    }

    /// Produce an iterator over the given slice of scaled variables.
    pub fn scaled_variables_iter<'a>(
        &self,
        unscaled_var: &'a [Number],
    ) -> impl Iterator<Item = Number> + 'a {
        let scale = self.scale();
        unscaled_var.iter().map(move |&val| val * scale)
    }

    pub fn update_current_velocity(
        &self,
        v: &[Number],
    ) -> std::cell::Ref<'_, Chunked<Chunked<Chunked3<Vec<f64>>>>> {
        {
            let mut cur_vel = self.vertex_set.cur_vel.borrow_mut();
            let sv = cur_vel.view_mut().into_flat();
            for (output, input) in sv.iter_mut().zip(self.scaled_variables_iter(v)) {
                *output = input;
            }
        }
        self.vertex_set.cur_vel.borrow()
    }

    /// Compute and return the objective value.
    pub fn objective_value(&self, v: ChunkedView<ChunkedView<Chunked3<&[Number]>>>) -> f64 {
        let cur_pos = self.compute_step(v);
        let x1 = cur_pos.view();
        let x0 = self.vertex_set.prev_pos.view();
        let v0 = self.vertex_set.prev_vel.view();

        let mut obj = 0.0;

        for (i, solid) in self.solids.iter().enumerate() {
            let x0 = x0.at(0).at(i).into_flat();
            let x1 = x1.at(0).at(i).into_flat();
            let v0 = v0.at(0).at(i).into_flat();
            let v = v.at(0).at(i).into_flat();
            obj += solid.elasticity().energy(x0, x1);
            obj += solid.gravity(self.gravity).energy(x0, x1);
            obj += solid.inertia().energy(v0, v);
        }

        for (i, shell) in self.shells.iter().enumerate() {
            let x0 = x0.at(1).at(i).into_flat();
            let x1 = x1.at(1).at(i).into_flat();
            //let v0 = v0.at(1).at(i).into_flat();
            //obj += shell.elasticity().energy(x0, x1);
            obj += shell.gravity(self.gravity).energy(x0, x1);
            //obj += shell.inertia().energy(v0, v);
        }

        // If time_step is 0.0, this is a pure static solve, which means that
        // there cannot be friction.
        if self.time_step > 0.0 {
            for fc in self.frictional_contacts.iter() {
                let obj_v = Self::mesh_vertex_subset(&self.solids, v, fc.object_index);
                let col_v = Self::mesh_vertex_subset(&self.solids, v, fc.collider_index);
                obj -= fc
                    .constraint
                    .frictional_dissipation([obj_v.view(), col_v.view()]);
            }
        }

        obj
    }

    /// Convert a given array of contact forces to impulses.
    fn contact_impulse_magnitudes(forces: &[f64], dt: f64) -> Vec<f64> {
        let dt = if dt > 0.0 { dt } else { 1.0 };
        forces.iter().map(|&cf| cf * dt).collect()
    }

    pub fn num_frictional_contacts(&self) -> usize {
        self.frictional_contacts.len()
    }

    /// Returns true if all friction solves have been completed/converged.
    /// This should be the case if and only if all elements in `friction_steps`
    /// are zero, which makes the return type simply a convenience.
    pub fn update_friction_impulse(
        &mut self,
        solution: ipopt::Solution,
        constraint_values: &[f64],
        friction_steps: &mut [u32],
    ) -> bool {
        if self.frictional_contacts.is_empty() {
            return true;
        }

        self.update_current_velocity(solution.primal_variables);

        let NonLinearProblem {
            ref mut frictional_contacts,
            ref volume_constraints,
            ref solids,
            ref vertex_set,
            time_step,
            ..
        } = *self;

        let prev_pos = vertex_set.prev_pos.view();
        let cur_vel = vertex_set.cur_vel.borrow();

        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
            let vel = cur_vel.view();
            let obj_prev_pos = Self::mesh_vertex_subset(&solids, prev_pos, fc.object_index);
            let col_prev_pos = Self::mesh_vertex_subset(&solids, prev_pos, fc.collider_index);
            let obj_vel = Self::mesh_vertex_subset(&solids, vel, fc.object_index);
            let col_vel = Self::mesh_vertex_subset(&solids, vel, fc.collider_index);

            let offset = volume_constraints.len();

            let contact_impulse = Self::contact_impulse_magnitudes(
                &solution.constraint_multipliers[offset..],
                time_step,
            );

            dbg!(crate::inf_norm(contact_impulse.iter().cloned()));
            let potential_values = &constraint_values[offset..];
            fc.constraint.update_frictional_contact_impulse(
                &contact_impulse,
                [obj_prev_pos.view(), col_prev_pos.view()],
                [obj_vel.view(), col_vel.view()],
                potential_values,
                friction_steps[fc_idx],
            );
        }

        false
    }

    /// Given a tetmesh, compute the strain energy per tetrahedron.
    fn compute_strain_energy_attrib(solid: &mut TetMeshSolid) {
        use geo::ops::ShapeMatrix;
        // Overwrite the "strain_energy" attribute.
        let mut strain = solid
            .tetmesh
            .remove_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB)
            .unwrap();
        strain
            .iter_mut::<f64>()
            .unwrap()
            .zip(zip!(
                solid.tetmesh.attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                    .unwrap(),
                solid.tetmesh.attrib_iter::<MuType, CellIndex>(MU_ATTRIB).unwrap(),
                solid.tetmesh.attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
                solid.tetmesh.attrib_iter::<RefShapeMtxInvType, CellIndex>(
                    REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                )
                .unwrap(),
                solid.tetmesh.tet_iter()
            ))
            .for_each(|(strain, (&lambda, &mu, &vol, &ref_shape_mtx_inv, tet))| {
                *strain =
                    NeoHookeanTetEnergy::new(tet.shape_matrix(), ref_shape_mtx_inv, vol, lambda, mu)
                        .elastic_energy()
            });

        solid
            .tetmesh
            .insert_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB, strain)
            .unwrap();
    }

    /// Given a tetmesh, compute the elastic forces per vertex, and save it at a vertex attribute.
    fn compute_elastic_forces_attrib(solid: &mut TetMeshSolid) {
        use geo::ops::ShapeMatrix;
        let mut forces_attrib = solid
            .tetmesh
            .remove_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB)
            .unwrap();

        let mut forces =
            Chunked3::from_grouped_mut_slice(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

        // Reset forces
        for f in forces.iter_mut() {
            *f = [0.0; 3];
        }

        let grad_iter = zip!(
            solid
                .tetmesh
                .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                .unwrap(),
            solid
                .tetmesh
                .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                .unwrap(),
            solid
                .tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            solid
                .tetmesh
                .attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB,)
                .unwrap(),
            solid.tetmesh.tet_iter()
        )
        .map(|(&lambda, &mu, &vol, &ref_shape_mtx_inv, tet)| {
            NeoHookeanTetEnergy::new(tet.shape_matrix(), ref_shape_mtx_inv, vol, lambda, mu)
                .elastic_energy_gradient()
        });

        for (grad, cell) in grad_iter.zip(solid.tetmesh.cells().iter()) {
            for j in 0..4 {
                let f = Vector3(forces[cell[j]]);
                forces[cell[j]] = (f - grad[j]).into();
            }
        }

        // Reinsert forces back into the attrib map
        solid
            .tetmesh
            .insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
    }

    /// Return the stacked friction impulses: one for each vertex.
    pub fn friction_impulse(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
        let prev_pos = self.vertex_set.prev_pos.view();

        // Copy vertex structure from the `vertex_set`
        let mut impulse = Chunked::from_offsets(
            prev_pos.offsets().to_vec(),
            Chunked::from_offsets(
                prev_pos.data().offsets().to_vec(),
                Chunked3::from_flat(vec![0.0; prev_pos.into_flat().len()]),
            ),
        );

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in self.frictional_contacts.iter() {
            obj_imp.clear();
            obj_imp.resize(
                Self::mesh_surface_vertex_count(&self.solids, &self.shells, fc.object_index),
                0.0,
            );

            coll_imp.clear();
            coll_imp.resize(
                Self::mesh_surface_vertex_count(&self.solids, &self.shells, fc.collider_index),
                0.0,
            );
            // TODO: Finish this
            //fc.constraint.add_friction_impulse(
            //    [Chunked3::from_flat(obj_imp.as_mut_slice()), Chunked3::from_flat(coll_imp.as_mut_slice())], 1.0);

            let mut imp =
                Self::mesh_vertex_subset_mut(&self.solids, impulse.view_mut(), fc.object_index);
            for (imp, obj_imp) in imp
                .iter_mut()
                .zip(Chunked3::from_flat(obj_imp.view_mut()).iter())
            {
                *imp = *obj_imp;
            }

            let mut imp =
                Self::mesh_vertex_subset_mut(&self.solids, impulse.view_mut(), fc.collider_index);
            for (imp, coll_imp) in imp
                .iter_mut()
                .zip(Chunked3::from_flat(coll_imp.view_mut()).iter())
            {
                *imp = *coll_imp;
            }
        }
        impulse
    }

    /// Return the stacked contact impulses: one for each vertex.
    pub fn contact_impulse(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
        let prev_pos = self.vertex_set.prev_pos.view();

        let mut offset = self.volume_constraints.len();

        // Copy vertex structure from the `vertex_set`
        let mut impulse = Chunked::from_offsets(
            prev_pos.offsets().to_vec(),
            Chunked::from_offsets(
                prev_pos.data().offsets().to_vec(),
                Chunked3::from_flat(vec![0.0; prev_pos.into_flat().len()]),
            ),
        );

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in self.frictional_contacts.iter() {
            // Get contact force from the warm start.
            let n = fc.constraint.num_contacts();
            let contact_impulse = Self::contact_impulse_magnitudes(
                &self.warm_start.constraint_multipliers[offset..offset + n],
                self.time_step,
            );

            offset += n;

            obj_imp.clear();
            obj_imp.resize(
                Self::mesh_surface_vertex_count(&self.solids, &self.shells, fc.object_index),
                0.0,
            );

            coll_imp.clear();
            coll_imp.resize(
                Self::mesh_surface_vertex_count(&self.solids, &self.shells, fc.collider_index),
                0.0,
            );

            let obj_x0 = Self::mesh_vertex_subset(&self.solids, prev_pos, fc.object_index);
            let coll_x0 = Self::mesh_vertex_subset(&self.solids, prev_pos, fc.collider_index);
            fc.constraint.add_contact_impulse(
                [obj_x0.view(), coll_x0.view()],
                &contact_impulse,
                [
                    Chunked3::from_flat(obj_imp.as_mut_slice()),
                    Chunked3::from_flat(coll_imp.as_mut_slice()),
                ],
            );

            let mut imp =
                Self::mesh_vertex_subset_mut(&self.solids, impulse.view_mut(), fc.object_index);
            for (imp, obj_imp) in imp
                .iter_mut()
                .zip(Chunked3::from_flat(obj_imp.view_mut()).iter())
            {
                *imp = *obj_imp;
            }

            let mut imp =
                Self::mesh_vertex_subset_mut(&self.solids, impulse.view_mut(), fc.collider_index);
            for (imp, coll_imp) in imp
                .iter_mut()
                .zip(Chunked3::from_flat(coll_imp.view_mut()).iter())
            {
                *imp = *coll_imp;
            }
        }
        impulse
    }

    /// Update the solid and shell meshes with relevant simulation data.
    pub fn update_mesh_data(&mut self) {
        let contact_impulse = self.contact_impulse();
        let friction_impulse = self.friction_impulse();
        for (idx, solid) in self.solids.iter_mut().enumerate() {
            // Write back friction and contact impulses
            solid
                .tetmesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    FRICTION_ATTRIB,
                    friction_impulse.view().at(0).at(idx).view().into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<ContactImpulseType, VertexIndex>(
                    CONTACT_ATTRIB,
                    contact_impulse.view().at(0).at(idx).view().into(),
                )
                .ok();

            // Write back elastic strain energy for visualization.
            Self::compute_strain_energy_attrib(solid);

            // Write back elastic forces on each node.
            Self::compute_elastic_forces_attrib(solid);
        }
        // TODO: Do the same for shells
    }

    /*
     * The following functions are there for debugging jacobians and hessians
     */
    /*

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
    */

    /*
     * End of debugging functions
     */
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for NonLinearProblem {
    fn num_variables(&self) -> usize {
        self.vertex_set.prev_vel.view().into_flat().len()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        let dt = if self.time_step > 0.0 {
            self.time_step
        } else {
            1.0
        };

        let bound = self.displacement_bound.unwrap_or(2e19) / (self.scale() * dt);

        x_l.iter_mut().for_each(|x| *x = -bound);
        x_u.iter_mut().for_each(|x| *x = bound);

        let x0 = self.vertex_set.prev_pos.view();
        let mut x_l = Chunked::from_offsets(
            *x0.offsets(),
            Chunked::from_offsets(*x0.data().offsets(), Chunked3::from_flat(x_l)),
        );
        let mut x_u = Chunked::from_offsets(
            *x0.offsets(),
            Chunked::from_offsets(*x0.data().offsets(), Chunked3::from_flat(x_u)),
        );

        for (i, solid) in self.solids.iter().enumerate() {
            if let Ok(fixed_verts) = solid
                .tetmesh
                .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            {
                let mut x_l = x_l.at_mut(0).at_mut(i);
                let mut x_u = x_u.at_mut(0).at_mut(i);
                // Find and set fixed vertices.
                x_l.iter_mut()
                    .zip(x_u.iter_mut())
                    .zip(fixed_verts.iter())
                    .filter(|&(_, &fixed)| fixed != 0)
                    .for_each(|((l, u), _)| {
                        *l = [0.0; 3];
                        *u = [0.0; 3];
                    });
            }
        }

        for (i, shell) in self.shells.iter().enumerate() {
            if let Ok(fixed_verts) = shell
                .trimesh
                .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            {
                let mut x_l = x_l.at_mut(1).at_mut(i);
                let mut x_u = x_u.at_mut(1).at_mut(i);
                // Find and set fixed vertices.
                x_l.iter_mut()
                    .zip(x_u.iter_mut())
                    .zip(fixed_verts.iter())
                    .filter(|&(_, &fixed)| fixed != 0)
                    .for_each(|((l, u), _)| {
                        *l = [0.0; 3];
                        *u = [0.0; 3];
                    });
            }
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

    fn objective(&self, uv: &[Number], obj: &mut Number) -> bool {
        let v = self.update_current_velocity(uv);
        *obj = self.objective_value(v.view());
        true
    }

    fn objective_grad(&self, uv: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f.iter_mut().for_each(|x| *x = 0.0); // clear gradient vector

        let v = self.update_current_velocity(uv);
        let v = v.view();

        // Copy the chunked structure from our vertex_set.
        // TODO: Refactor this into a function for Chunked and UniChunked types.
        let mut grad = Chunked::from_offsets(
            *v.offsets(),
            Chunked::from_offsets(*v.data().offsets(), Chunked3::from_flat(grad_f)),
        );

        let cur_pos = self.compute_step(v.view());
        let x1 = cur_pos.view();
        let x0 = self.vertex_set.prev_pos.view();
        let v0 = self.vertex_set.prev_vel.view();

        for (i, solid) in self.solids.iter().enumerate() {
            let x0 = x0.at(0).at(i).into_flat();
            let x1 = x1.at(0).at(i).into_flat();
            let v0 = v0.at(0).at(i).into_flat();
            let v = v.at(0).at(i).into_flat();
            let g = grad.at_mut(0).at_mut(i).into_flat();
            solid.elasticity().add_energy_gradient(x0, x1, g);
            solid.gravity(self.gravity).add_energy_gradient(x0, x1, g);
            solid.inertia().add_energy_gradient(v0, v, g);
        }

        for (i, shell) in self.shells.iter().enumerate() {
            let x0 = x0.at(1).at(i).into_flat();
            let x1 = x1.at(1).at(i).into_flat();
            //let v0 = v0.at(1).at(i).into_flat();
            let g = grad.at_mut(1).at_mut(i).into_flat();
            //shell.elasticity().energy(x0, x1, g);
            shell.gravity(self.gravity).add_energy_gradient(x0, x1, g);
            //shell.inertia().energy(v0, v, g);
        }

        let grad_f = grad.into_flat();

        let dt = self.time_step;
        if dt > 0.0 {
            // This is a correction to transform the above energy gradients to velocity gradients
            // from displacement or position gradients.
            grad_f.iter_mut().for_each(|g| *g *= dt);
        }

        let mut grad = Chunked::from_offsets(
            *v.offsets(),
            Chunked::from_offsets(*v.data().offsets(), Chunked3::from_flat(grad_f)),
        );

        for fc in self.frictional_contacts.iter() {
            let mut obj_g =
                Self::mesh_vertex_subset_mut(&self.solids, grad.view_mut(), fc.object_index);
            fc.constraint.add_friction_impulse(obj_g.view_mut(), -1.0);
        }

        let scale = self.scale();
        grad.into_flat().iter_mut().for_each(|g| *g *= scale);

        true
    }
}

impl ipopt::ConstrainedProblem for NonLinearProblem {
    fn num_constraints(&self) -> usize {
        let mut num = 0;
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.constraint_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.constraint_size();
            //println!("num constraints  = {:?}", num);
        }
        num
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        let mut num = 0;
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.constraint_jacobian_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.constraint_jacobian_size();
            //println!("scc jac size = {:?}", num);
        }
        num
    }

    fn initial_constraint_multipliers(&self, lambda: &mut [Number]) -> bool {
        // TODO: Move this is_empty logic to remapping contacts: The important observation here is
        // that contact set != constraint set because some methods don't enforce contacts via a
        // constraint.
        if !lambda.is_empty() {
            // The constrained points may change between updating the warm start and using it here.
            assert_eq!(lambda.len(), self.warm_start.constraint_multipliers.len());
            lambda.copy_from_slice(self.warm_start.constraint_multipliers.as_slice());
        }
        true
    }

    fn constraint(&self, uv: &[Number], g: &mut [Number]) -> bool {
        let cur_vel = self.update_current_velocity(uv);
        let cur_pos = self.compute_step(cur_vel.view());
        let x = cur_pos.view();
        let x0 = self.vertex_set.prev_pos.view();

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));

        let mut count = 0; // Constraint counter

        for (_, vc) in self.volume_constraints.iter() {
            let n = vc.constraint_size();
            vc.constraint(x0.into_flat(), x.into_flat(), &mut g[count..count + n]);
            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let n = fc.constraint.constraint_size();
            let obj_x0 = Self::mesh_vertex_subset(&self.solids, x0, fc.object_index);
            let coll_x0 = Self::mesh_vertex_subset(&self.solids, x0, fc.collider_index);
            let obj_x = Self::mesh_vertex_subset(&self.solids, x, fc.object_index);
            let coll_x = Self::mesh_vertex_subset(&self.solids, x, fc.collider_index);

            fc.constraint.constraint(
                [obj_x0.view(), coll_x0.view()],
                [obj_x.view(), coll_x.view()],
                &mut g[count..count + n],
            );
            count += n;
        }

        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let mut count = 0; // Constraint counter
        for (_, vc) in self.volume_constraints.iter() {
            let mut bounds = vc.constraint_bounds();
            let n = vc.constraint_size();
            g_l[count..count + n].swap_with_slice(&mut bounds.0);
            g_u[count..count + n].swap_with_slice(&mut bounds.1);
            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let mut bounds = fc.constraint.constraint_bounds();
            let n = fc.constraint.constraint_size();
            g_l[count..count + n].swap_with_slice(&mut bounds.0);
            g_u[count..count + n].swap_with_slice(&mut bounds.1);
            count += n;
        }
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        let mut count = 0; // Constraint counter

        let mut row_offset = 0;
        for (_, vc) in self.volume_constraints.iter() {
            let iter = vc.constraint_jacobian_indices_iter().unwrap();
            for MatrixElementIndex { row, col } in iter {
                rows[count] = (row + row_offset) as ipopt::Index;
                cols[count] = col as ipopt::Index;
                count += 1;
            }
            row_offset += 1;
        }

        for fc in self.frictional_contacts.iter() {
            use ipopt::BasicProblem;
            let nrows = fc.constraint.constraint_size();
            let ncols = self.num_variables();
            let mut jac = vec![vec![0; nrows]; ncols]; // col major

            let iter = fc.constraint.constraint_jacobian_indices_iter().unwrap();
            for MatrixElementIndex { row, col } in iter {
                rows[count] = (row + row_offset) as ipopt::Index;
                cols[count] = self.source_coordinates(fc.object_index, fc.collider_index, col)
                    as ipopt::Index;
                jac[col][row] = 1;
                count += 1;
            }
            row_offset += nrows;

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

        true
    }

    fn constraint_jacobian_values(&self, uv: &[Number], vals: &mut [Number]) -> bool {
        let v = self.update_current_velocity(uv);
        let cur_pos = self.compute_step(v.view());
        let x = cur_pos.view();
        let x0 = self.vertex_set.prev_pos.view();

        let mut count = 0; // Constraint counter

        for (_, vc) in self.volume_constraints.iter() {
            let n = vc.constraint_jacobian_size();
            vc.constraint_jacobian_values(
                x0.into_flat(),
                x.into_flat(),
                &mut vals[count..count + n],
            )
            .ok();
            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let n = fc.constraint.constraint_jacobian_size();
            let obj_x0 = Self::mesh_vertex_subset(&self.solids, x0, fc.object_index);
            let coll_x0 = Self::mesh_vertex_subset(&self.solids, x0, fc.collider_index);
            let obj_x = Self::mesh_vertex_subset(&self.solids, x, fc.object_index);
            let coll_x = Self::mesh_vertex_subset(&self.solids, x, fc.collider_index);

            fc.constraint
                .constraint_jacobian_values(
                    [obj_x0.view(), coll_x0.view()],
                    [obj_x.view(), coll_x.view()],
                    &mut vals[count..count + n],
                )
                .ok();
            //println!("jac g vals = {:?}", &vals[count..count+n]);
        }
        let dt = if self.time_step > 0.0 {
            self.time_step
        } else {
            1.0
        };
        let scale = dt * self.scale();
        vals.iter_mut().for_each(|v| *v *= scale);

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));
        //self.print_jacobian_svd(vals);

        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        let mut num = 0;
        for solid in self.solids.iter() {
            num += solid.elasticity().energy_hessian_size() + solid.inertia().energy_hessian_size();
        }
        //for shell in self.shells.iter() {
        //    num += shell.inertia().energy_hessian_size();
        //}
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.constraint_hessian_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.constraint_hessian_size();
        }
        num
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
        let mut count = 0; // Constraint counter

        // Add energy indices
        for solid in self.solids.iter() {
            let elasticity = solid.elasticity();
            let n = elasticity.energy_hessian_size();
            elasticity
                .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
            count += n;

            let inertia = solid.inertia();
            let n = inertia.energy_hessian_size();
            inertia
                .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
            count += n;
        }

        //for shell in self.shells.iter() {
        //    let inertia = shell.inertia();
        //    let n = inertia.energy_hessian_size();
        //    inertia.energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
        //    count += n;
        //}

        // Add volume constraint indices
        for (solid_idx, vc) in self.volume_constraints.iter() {
            let offset = self
                .vertex_set
                .prev_vel
                .view()
                .at(0)
                .offset_value(*solid_idx);
            for MatrixElementIndex { row, col } in vc.constraint_hessian_indices_iter() {
                rows[count] = (row + offset) as ipopt::Index;
                cols[count] = (col + offset) as ipopt::Index;
                count += 1;
            }
        }

        for fc in self.frictional_contacts.iter() {
            let iter = fc.constraint.constraint_hessian_indices_iter().unwrap();
            for MatrixElementIndex { row, col } in iter {
                rows[count] = self.source_coordinates(fc.object_index, fc.collider_index, row)
                    as ipopt::Index;
                cols[count] = self.source_coordinates(fc.object_index, fc.collider_index, col)
                    as ipopt::Index;
                count += 1;
            }
        }

        true
    }
    fn hessian_values(
        &self,
        uv: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let cur_vel = self.update_current_velocity(uv);
        let v = cur_vel.view();
        let cur_pos = self.compute_step(v.view());
        let x1 = cur_pos.view();
        let x0 = self.vertex_set.prev_pos.view();
        let v0 = self.vertex_set.prev_vel.view();

        if cfg!(debug_assertions) {
            // Initialize vals in debug builds.
            for v in vals.iter_mut() {
                *v = 0.0;
            }
        }

        // Correction to make the above hessian wrt velocity instead of displacement.
        let dt = if self.time_step > 0.0 {
            self.time_step
        } else {
            1.0
        };

        let mut count = 0; // Constraint counter

        for (solid_idx, solid) in self.solids.iter().enumerate() {
            let x0 = x0.at(0).at(solid_idx).into_flat();
            let x1 = x1.at(0).at(solid_idx).into_flat();
            let v0 = v0.at(0).at(solid_idx).into_flat();
            let v = v.at(0).at(solid_idx).into_flat();
            let elasticity = solid.elasticity();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_values(x0, x1, dt * dt, &mut vals[count..count + n]);
            count += n;

            let inertia = solid.inertia();
            let n = inertia.energy_hessian_size();
            inertia.energy_hessian_values(v0, v, 1.0, &mut vals[count..count + n]);
            count += n;
        }

        //for (shell_idx, shell) in self.shells.iter().enumerate() {
        //    let v0 = v0.at(1).at(shell_idx).into_flat();
        //    let v = v.at(1).at(shell_idx).into_flat();
        //    let inertia = shell.inertia();
        //    let n = inertia.energy_hessian_size();
        //    inertia.energy_hessian_values(v0, v, 1.0, &mut vals[count..count + n]);
        //    count += n;
        //}

        // Multiply energy hessian by objective factor.
        let factor = obj_factor * self.scale() * self.scale();
        for v in vals[0..count].iter_mut() {
            *v *= factor;
        }

        let mut coff = 0;

        let c_scale = self.scale() * self.scale() * dt * dt;

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let x0 = x0.at(0).at(*solid_idx).into_flat();
            let x1 = x1.at(0).at(*solid_idx).into_flat();
            let nc = vc.constraint_size();
            let nh = vc.constraint_hessian_size();
            vc.constraint_hessian_values(
                x0,
                x1,
                &lambda[coff..coff + nc],
                c_scale,
                &mut vals[count..count + nh],
            )
            .unwrap();

            count += nh;
            coff += nc;
        }

        for fc in self.frictional_contacts.iter() {
            let obj_x0 = Self::mesh_vertex_subset(&self.solids, x0, fc.object_index);
            let coll_x0 = Self::mesh_vertex_subset(&self.solids, x0, fc.collider_index);
            let obj_x = Self::mesh_vertex_subset(&self.solids, x1, fc.object_index);
            let coll_x = Self::mesh_vertex_subset(&self.solids, x1, fc.collider_index);
            let nc = fc.constraint.constraint_size();
            let nh = fc.constraint.constraint_hessian_size();
            fc.constraint
                .constraint_hessian_values(
                    [obj_x0.view(), coll_x0.view()],
                    [obj_x.view(), coll_x.view()],
                    &lambda[coff..coff + nc],
                    c_scale,
                    &mut vals[count..count + nh],
                )
                .unwrap();

            count += nh;
            coff += nc;
        }

        assert_eq!(count, vals.len());
        assert_eq!(coff, lambda.len());
        //self.print_hessian_svd(vals);

        true
    }
}
