use std::cell::{Ref, RefCell, RefMut};
use std::time::Instant;

use autodiff as ad;
use flatk::*;
use geo::attrib::*;
use geo::index::CheckedIndex;
use geo::mesh::{topology::*, VertexPositions};
use geo::Index;
use num_traits::Zero;
use rayon::prelude::*;
use tensr::{
    AsMutTensor, AsTensor, DSMatrix, IndexedExpr, IntoData, IntoExpr, IntoTensor, Matrix, Norm,
    Vector2, Vector3,
};

use super::state::*;
use crate::attrib_defines::*;
use crate::constraints::{
    volume_change_penalty::VolumeChangePenalty, ContactPenalty, FrictionJacobianTimings,
    MappedDistanceGradient,
};
use crate::energy::{Energy, EnergyGradient, EnergyHessian, EnergyHessianTopology};
use crate::energy_models::{gravity::*, inertia::*};
use crate::matrix::*;
use crate::nl_fem::{state, Preconditioner, ResidualTimings, SingleStepTimeIntegration};
use crate::objects::tetsolid::*;
use crate::objects::trishell::*;
use crate::Mesh;
use crate::PointCloud;
use crate::{Real, Real64};

mod assisted_line_search;
mod frictional_contact_constraint;

pub use assisted_line_search::*;
pub use frictional_contact_constraint::*;

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

/// Workspace variables for line search assist.
#[derive(Clone, Debug)]
pub struct LineSearchWorkspace<T> {
    // pub pos_cur: Chunked3<Vec<T>>,
    pub pos_next: Chunked3<Vec<T>>,
    pub dq: Vec<T>,
    pub vel: Chunked3<Vec<T>>,
    pub search_dir: Chunked3<Vec<T>>,
    pub f1vtx: Chunked3<Vec<T>>,
    pub f2vtx: Chunked3<Vec<T>>,
}

/// Workspace variables Jacobian product computation.
#[derive(Clone, Debug)]
pub struct JacobianWorkspace<T> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<T>,
    pub jac: DSMatrix<T>,
    pub mapping: Vec<Index>,
    /// Indicates if the values need to be recomputed.
    pub stale: bool,
}

impl<T: Real> Default for JacobianWorkspace<T> {
    fn default() -> Self {
        JacobianWorkspace {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
            jac: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
            mapping: Vec::new(),
            stale: true,
        }
    }
}

/// Workspace used for computing preconditiners.
#[derive(Clone, Debug, Default)]
pub struct PreconditionerWorkspace<T> {
    pub buffer: Vec<T>,
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
    /// The index of state vertices for each vertex from the original mesh used to create this problem.
    ///
    /// This vector is useful for debugging and optimization.
    pub state_vertex_indices: Vec<Index>,
    pub frictional_contact_constraints: Vec<FrictionalContactConstraint<T>>,
    pub frictional_contact_constraints_ad: Vec<FrictionalContactConstraint<ad::FT<T>>>,
    // pub self_contact_constraint: SelfContactConstraint<T>,
    /// Constraint on the volume of different regions of the simulation.
    pub volume_constraints: Vec<RefCell<VolumeChangePenalty>>,
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
    /// Original mesh used to create this problem.
    pub original_mesh: Mesh,

    pub prev_force: Vec<T>,
    pub candidate_force: RefCell<Vec<T>>,

    /// Workspace entries used for computing Jacobian products.
    pub jacobian_workspace: RefCell<JacobianWorkspace<T>>,

    /// Workspace for computing preconditioners.
    pub preconditioner_workspace: RefCell<PreconditionerWorkspace<T>>,

    pub preconditioner: Preconditioner,

    pub time_integration: SingleStepTimeIntegration,

    pub line_search_ws: RefCell<LineSearchWorkspace<T>>,

    pub debug_friction: RefCell<Vec<T>>,

    pub timings: RefCell<ResidualTimings>,
    pub jac_timings: RefCell<FrictionJacobianTimings>,
    pub project_element_hessians: bool,
    // pub candidate_alphas: RefCell<MinMaxHeap>,
}

impl<T: Real> NLProblem<T> {
    /// Get the current iteration count and reset it.
    pub fn pop_iteration_count(&mut self) -> usize {
        let iter = self.iterations;
        // Reset count
        self.iterations = 0;
        iter
    }

    pub fn impulse_inv_scale(&self) -> f64 {
        1.0 //utils::approx_power_of_two64(100.0 / (self.time_step() * self.max_element_force_scale))
    }

    fn volume_constraint_scale(&self) -> f64 {
        1.0
    }

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

    /// Returns the solved positions of the vertices in the original order.
    pub fn vertex_positions(&self) -> Vec<[T; 3]> {
        let State {
            vtx: VertexWorkspace {
                orig_index, next, ..
            },
            ..
        } = &*self.state.borrow();
        let pos = next.pos.as_arrays();
        let mut out = vec![[T::zero(); 3]; pos.len()];
        // TODO: add original_order to state so we can iterate (in parallel) over out insated here.
        orig_index
            .iter()
            .zip(pos.iter())
            .for_each(|(&i, pos)| out[i] = *pos);
        out
    }

    fn num_in_proximity(&self) -> usize {
        self.frictional_contact_constraints
            .iter()
            .map(|fc| {
                let constraint = fc.constraint.borrow();
                constraint.contact_state.constraint_size()
            })
            .sum()
    }
    fn num_contacts(&self) -> usize {
        self.frictional_contact_constraints
            .iter()
            .map(|fc| {
                let constraint = &*fc.constraint.borrow();
                constraint
                    .contact_state
                    .lambda
                    .iter()
                    .filter(|&&l| l > T::zero())
                    .count()
            })
            .sum()
    }
}

impl<T: Real64> NLProblem<T> {
    /// Returns a reference to the original mesh used to create this problem with updated position values from the given velocity degrees of freedom.
    pub fn mesh_with(&self, dq: &[T]) -> Mesh {
        let mut mesh = self.original_mesh.clone();
        let out = mesh.vertex_positions_mut();

        assert!(dq.len() % 3 == 0);
        {
            self.integrate_step(dq);
            self.state.borrow_mut().update_vertices(dq);
        }

        // Update positions
        {
            let State {
                vtx: VertexWorkspace {
                    orig_index, next, ..
                },
                ..
            } = &*self.state.borrow();
            let pos = next.pos.as_arrays();
            // TODO: add original_order to state so we can iterate (in parallel) over out instead here.
            orig_index
                .iter()
                .zip(pos.iter())
                .for_each(|(&i, pos)| out[i] = pos.as_tensor().cast::<f64>().into_data());
        }

        // TODO: add additional attributes.
        //self.compute_residual(&mut mesh);
        mesh
    }

    /// Returns a reference to the original mesh used to create this problem with updated values.
    pub fn mesh(&self) -> Mesh {
        let mut mesh = self.original_mesh.clone();
        let out_pos = mesh.vertex_positions_mut();
        let mut out_vel = vec![[0.0; 3]; out_pos.len()];
        let mut out_force = vec![[0.0; 3]; out_pos.len()];
        let mut out_mass = vec![0.0; out_pos.len()];

        // Update positions, velocities and net force
        {
            let State {
                vtx:
                    VertexWorkspace {
                        orig_index,
                        next,
                        mass_inv,
                        ..
                    },
                ..
            } = &*self.state.borrow();

            let force: Vec<[f32; 3]> = self
                .prev_force
                .chunks_exact(3)
                .map(|f| {
                    [
                        f[0].to_f32().unwrap(),
                        f[1].to_f32().unwrap(),
                        f[2].to_f32().unwrap(),
                    ]
                })
                .collect();

            let pos = next.pos.as_arrays();
            let vel = next.vel.as_arrays();
            // TODO: add original_order to state so we can iterate (in parallel) over out instead here.
            orig_index
                .iter()
                .zip(
                    pos.iter()
                        .zip(vel.iter())
                        .zip(force.iter())
                        .zip(mass_inv.iter()),
                )
                .for_each(|(&i, (((pos, vel), &force), &mass_inv))| {
                    out_pos[i] = pos.as_tensor().cast::<f64>().into_data();
                    out_vel[i] = vel.as_tensor().cast::<f64>().into_data();
                    out_force[i] = force;
                    out_mass[i] = 1.0 / mass_inv.to_f64().unwrap();
                });
        }

        let pos64 = out_pos.to_vec();
        // Add a position attribute guaranteed to be 64bit
        mesh.remove_attrib::<VertexIndex>(POSITION64_ATTRIB).ok(); // Removing attrib
        mesh.insert_attrib_data::<Pos64Type, VertexIndex>(POSITION64_ATTRIB, pos64)
            .unwrap(); // No panic: removed above.

        mesh.remove_attrib::<VertexIndex>(VELOCITY_ATTRIB).ok(); // Removing attrib
        mesh.insert_attrib_data::<VelType, VertexIndex>(VELOCITY_ATTRIB, out_vel)
            .unwrap(); // No panic: removed above.

        mesh.remove_attrib::<VertexIndex>(NET_FORCE_ATTRIB).ok(); // Removing attrib
        mesh.insert_attrib_data::<NetForceType, VertexIndex>(NET_FORCE_ATTRIB, out_force)
            .unwrap(); // No panic: removed above.

        debug_assert_eq!(mesh.num_vertices(), self.state_vertex_indices.len());

        mesh.remove_attrib::<VertexIndex>(STATE_INDEX_ATTRIB).ok(); // Removing attrib
        mesh.insert_attrib_data::<StateIndexType, VertexIndex>(
            STATE_INDEX_ATTRIB,
            self.state_vertex_indices
                .iter()
                .map(|&x| x.unwrap() as i32)
                .collect(),
        )
        .unwrap(); // No panic: removed above.

        mesh.remove_attrib::<VertexIndex>(MASS_ATTRIB).ok(); // Removing attrib
        mesh.insert_attrib_data::<MassType, VertexIndex>(MASS_ATTRIB, out_mass)
            .unwrap(); // No panic: removed above.

        self.compute_residual_on_mesh(&mut mesh);
        self.compute_distance_potential(&mut mesh);
        self.compute_frictional_contact_data(&mut mesh);
        mesh
    }

    /// Returns a vector of lumped vertex masses.
    fn lumped_mass_inv(&self) -> Ref<'_, [T]> {
        Ref::map(self.state.borrow(), |state| state.vtx.mass_inv.as_slice())
    }

    /// Returns a vector of lumped vertex stiffnesses.
    fn lumped_stiffness(&self) -> Ref<'_, [T]> {
        Ref::map(self.state.borrow(), |state| {
            state.vtx.lumped_stiffness.as_slice()
        })
    }

    fn epsilon_iter(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(
            self.frictional_contact_constraints
                .iter()
                .map(|fc| fc.constraint.borrow().params.friction_params.epsilon),
        )
    }

    fn epsilon_iter_mut(&mut self) -> Box<dyn Iterator<Item = RefMut<'_, f64>> + '_> {
        Box::new(self.frictional_contact_constraints.iter_mut().map(|fc| {
            RefMut::map(fc.constraint.borrow_mut(), |m| {
                &mut m.params.friction_params.epsilon
            })
        }))
    }

    fn contact_violation(&self, x: &[T]) -> ContactViolation {
        let constraint = self.contact_constraint(x);
        let constraint_index = self.contact_constraint_index();
        let (deepest, deepest_constraint_index) = constraint
            .iter()
            .zip(constraint_index.iter())
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .map(|(&a, &i)| (a, Some(i)))
            .unwrap_or_else(|| (T::zero(), None));

        if let Some(deepest_constraint_index) = deepest_constraint_index {
            let deepest = deepest.to_f64().unwrap();

            let delta = self.frictional_contact_constraints[deepest_constraint_index]
                .constraint
                .borrow()
                .params
                .tolerance as f64;
            let largest_penalty = ContactPenalty::new(delta).b(deepest);
            let bump_ratio: f64 =
                ContactPenalty::new(delta).db(deepest) / ContactPenalty::new(delta).db(0.5 * delta);

            ContactViolation {
                bump_ratio,
                violation: 0.0_f64.max(-deepest),
                penetration: deepest,
                largest_penalty,
                violating_constraint_index: deepest_constraint_index,
            }
        } else {
            ContactViolation::default()
        }
    }

    fn compute_distance_potential(&self, mesh: &mut Mesh) {
        let state = &*self.state.borrow();
        let mut orig_order_distance_potential = vec![0.0; mesh.num_vertices()];
        for (i, fc) in self.frictional_contact_constraints.iter().enumerate() {
            let constraint = fc.constraint.borrow();
            let dist = constraint.cached_distance_potential(mesh.num_vertices());
            orig_order_distance_potential
                .iter_mut()
                .for_each(|d| *d = 0.0); // zero out.
            state
                .vtx
                .orig_index
                .iter()
                .zip(dist.iter())
                .for_each(|(&i, d)| {
                    orig_order_distance_potential[i] = d.to_f64().unwrap();
                });

            let potential_attrib_name = format!("{}{}", POTENTIAL_ATTRIB, i);
            // Should not panic since distance potential should have the same number of elements as vertices.
            mesh.set_attrib_data::<PotentialType, VertexIndex>(
                &potential_attrib_name,
                orig_order_distance_potential.clone(),
            )
            .unwrap();
        }
    }

    fn compute_residual_on_mesh(&self, mesh: &mut Mesh) {
        //TODO: determine if we actually need to recompute the residual here.
        //self.compute_vertex_residual();
        let state = &*self.state.borrow();
        let vertex_residuals = state.vtx.view().map_storage(|state| state.residual);
        let mut orig_order_vertex_residuals = vec![[0.0; 3]; vertex_residuals.len()];
        state
            .vtx
            .orig_index
            .iter()
            .zip(vertex_residuals.iter())
            .for_each(|(&i, v)| {
                orig_order_vertex_residuals[i] = v.as_tensor().cast::<f64>().into_data()
            });
        // Should not panic since vertex_residuals should have the same number of elements as vertices.
        mesh.set_attrib_data::<ResidualType, VertexIndex>(
            RESIDUAL_ATTRIB,
            orig_order_vertex_residuals,
        )
        .unwrap();
    }

    // This includes friction and contact forces individually
    fn compute_frictional_contact_data(&self, mesh: &mut Mesh) {
        let State { vtx, .. } = &mut *self.state.borrow_mut();

        // Clear residual vector.
        vtx.residual
            .storage_mut()
            .iter_mut()
            .for_each(|x| *x = T::zero());

        // let mut rel_vel = Chunked3::from_flat(vec![T::zero(); vtx.residual.storage().len()]);
        // let mut normals = Chunked3::from_flat(vec![T::zero(); vtx.residual.storage().len()]);
        // let mut tangents = Chunked3::from_flat(vec![T::zero(); vtx.residual.storage().len()]);
        // let mut bitangents = Chunked3::from_flat(vec![T::zero(); vtx.residual.storage().len()]);
        let mut contact_force = Chunked3::from_flat(vec![T::zero(); vtx.residual.storage().len()]);
        let mut friction_force = Chunked3::from_flat(vec![T::zero(); vtx.residual.storage().len()]);

        {
            let ResidualState { next, .. } = vtx.residual_state().into_storage();
            let frictional_contacts = self.frictional_contact_constraints.as_slice();
            for fc in frictional_contacts.iter() {
                let mut fc_constraint = fc.constraint.borrow_mut();
                fc_constraint.update_state(Chunked3::from_flat(next.pos));
                fc_constraint.update_distance_potential();
                fc_constraint.update_multipliers();

                fc_constraint.subtract_constraint_force_par(contact_force.view_mut());
                fc_constraint.subtract_friction_force(
                    friction_force.view_mut(),
                    Chunked3::from_flat(next.vel),
                    false,
                );

                //fc_constraint.add_contact_data(
                //    Chunked3::from_flat(next.vel),
                //    normals.view_mut(),
                //    tangents.view_mut(),
                //    bitangents.view_mut(),
                //    rel_vel.view_mut(),
                //);
            }
        }

        // let mut orig_order_relative_velocities = vec![[0.0; 3]; rel_vel.len()];
        // let mut orig_order_normals = vec![[0.0; 3]; normals.len()];
        // let mut orig_order_tangents = vec![[0.0; 3]; tangents.len()];
        // let mut orig_order_bitangents = vec![[0.0; 3]; bitangents.len()];
        let mut orig_order_contact_forces = vec![[0.0; 3]; contact_force.len()];
        let mut orig_order_friction_forces = vec![[0.0; 3]; friction_force.len()];
        zip!(
            vtx.orig_index.iter(),
            contact_force.iter(),
            friction_force.iter(),
            // rel_vel.iter(),
            // normals.iter(),
            // tangents.iter(),
            // bitangents.iter(),
        )
        .for_each(|(&i, c, f /*, v, n, t, b*/)| {
            // orig_order_normals[i] = (-n.as_tensor().cast::<f64>()).into_data();
            // orig_order_tangents[i] = (-t.as_tensor().cast::<f64>()).into_data();
            // orig_order_bitangents[i] = (-b.as_tensor().cast::<f64>()).into_data();
            // orig_order_relative_velocities[i] = (-v.as_tensor().cast::<f64>()).into_data();
            orig_order_contact_forces[i] = (-c.as_tensor().cast::<f64>()).into_data();
            orig_order_friction_forces[i] = (-f.as_tensor().cast::<f64>()).into_data();
        });
        // Should not panic since vertex_forces should have the same number of elements as vertices.
        mesh.set_attrib_data::<ContactForceType, VertexIndex>(
            CONTACT_ATTRIB,
            orig_order_contact_forces,
        )
        .unwrap();
        mesh.set_attrib_data::<FrictionForceType, VertexIndex>(
            FRICTION_ATTRIB,
            orig_order_friction_forces,
        )
        .unwrap();
        // mesh.set_attrib_data::<VelType, VertexIndex>(
        //     "rel_vel",
        //     orig_order_relative_velocities,
        // )
        // .unwrap();
        // mesh.set_attrib_data::<[f64; 3], VertexIndex>("contact_normals", orig_order_normals)
        //     .unwrap();
        // mesh.set_attrib_data::<[f64; 3], VertexIndex>("contact_tangents", orig_order_tangents)
        //     .unwrap();
        // mesh.set_attrib_data::<[f64; 3], VertexIndex>("contact_bitangents", orig_order_bitangents)
        //     .unwrap();
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

    /// Update the fixed vertex state with the given points.
    pub fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        self.update_vertex_positions(pts.vertex_positions())
    }

    /// Update the fixed vertex state with the given vertex positions.
    pub fn update_vertex_positions(&mut self, pos: &[[f64; 3]]) -> Result<(), crate::Error> {
        let new_pos = Chunked3::from_array_slice(pos);
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

    /// Checks if the max_step is violated
    pub fn max_step_violation(&self) -> bool {
        let dt = self.time_step();
        let state = self.state.borrow();
        let vel = state.vtx.next.vel.view();
        self.frictional_contact_constraints
            .iter()
            .any(|fc| fc.constraint.borrow().max_step_violated(vel, dt))
    }

    /// Updates all stateful constraints with the most recent data.
    ///
    /// Return an estimate if any constraints have changed, though this estimate may have false
    /// negatives.
    pub fn update_constraint_set(&mut self, and_contact_hessian: bool, and_autodiff: bool) -> bool {
        let mut changed = false; // Report if anything has changed to the caller.

        let dt = self.time_step();

        let NLProblem {
            ref mut frictional_contact_constraints,
            ref mut frictional_contact_constraints_ad,
            ref state,
            ..
        } = *self;

        let state = &*state.borrow();
        let pos = state.vtx.next.pos.view();
        let vel = state.vtx.next.vel.view();
        let pos_ad: Chunked3<Vec<_>> = pos
            .iter()
            .map(|&[x, y, z]| [ad::F::cst(x), ad::F::cst(y), ad::F::cst(z)])
            .collect();

        for (fc, fc_ad) in frictional_contact_constraints
            .iter_mut()
            .zip(frictional_contact_constraints_ad.iter_mut())
        {
            let new_max_step = fc.constraint.borrow_mut().compute_max_step(vel, dt);
            fc.constraint.borrow_mut().set_max_step(new_max_step);
            fc_ad.constraint.borrow_mut().set_max_step(new_max_step);
            changed |= fc
                .constraint
                .borrow_mut()
                .update_neighbors(pos, and_contact_hessian);

            if and_autodiff {
                changed |= fc_ad
                    .constraint
                    .borrow_mut()
                    .update_neighbors(pos_ad.view(), and_contact_hessian);
            }
        }

        // TODO: REMOVE THE BELOW DEBUG CODE
        //for fc in frictional_contact_constraints.iter() {
        //    let mut fc_constraint = fc.constraint.borrow_mut();
        //    fc_constraint.update_state(pos.view());
        //    fc_constraint.update_constraint_gradient();
        //    fc_constraint.update_multipliers(
        //        self.delta as f32,
        //        self.kappa as f32,
        //    );
        //}
        //for fc in frictional_contact_constraints_ad.iter() {
        //    let mut fc_constraint = fc.constraint.borrow_mut();
        //    fc_constraint.update_state(pos_ad.view());
        //    fc_constraint.update_constraint_gradient();
        //    fc_constraint.update_multipliers(
        //        self.delta as f32,
        //        self.kappa as f32,
        //    );
        //}

        changed
    }

    /// Commit velocity by advancing the internal state by the given unscaled velocity `uv`.
    ///
    /// If `and_autodiff` is true, then the autodiff state is also advanced, since it is used to
    /// verify derivatives for lagged friction solutions.
    pub fn advance(&mut self, v: &[T]) {
        self.integrate_step(v);
        self.state.borrow_mut().update_vertices(v);

        // Advance to next state

        self.state.borrow_mut().advance(v);

        // Commit candidate forces. This is used for TR and SDIRK2 integration.
        self.prev_force.clone_from(&*self.candidate_force.borrow());
    }

    pub fn update_constraint_state(
        &mut self,
        v: &[T],
        explicit_jacobian: bool,
        and_autodiff: bool,
    ) {
        self.update_state(v, true, explicit_jacobian, false);
        // This caches current state so we don't have to recompute a lot of things during the next step.
        for fc in self.frictional_contact_constraints.iter_mut() {
            fc.constraint.borrow_mut().advance_state();
        }

        if and_autodiff {
            self.update_state_ad_cst(v, true);
            // First we need to make sure that the autodiff state is up to date, then advance.
            for fc in self.frictional_contact_constraints_ad.iter_mut() {
                fc.constraint.borrow_mut().advance_state();
            }
        }
    }

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

    /// A convenience function to integrate the given AD velocity by the internal time step.
    pub fn integrate_step_ad(&self) {
        let mut state = self.state.borrow_mut();

        let step_state = state.step_state_ad();
        let dt = self.time_step();

        match self.time_integration {
            SingleStepTimeIntegration::BE => State::be_step(step_state, dt),
            SingleStepTimeIntegration::TR => State::tr_step(step_state, dt),
            SingleStepTimeIntegration::BDF2 => State::bdf2_step(step_state, dt),
            SingleStepTimeIntegration::MixedBDF2(t) => {
                State::mixed_bdf2_step(step_state, dt, 1.0 - t as f64)
            }
            SingleStepTimeIntegration::SDIRK2 => {
                let alpha = 1.0 - 0.5 * 2.0_f64.sqrt();
                // let alpha = 1.0;
                State::sdirk2_step(step_state, dt, alpha)
            }
        }
    }

    /// A convenience function to integrate the given velocity by the internal time step.
    pub fn integrate_step(&self, v: &[T]) {
        let mut state = self.state.borrow_mut();

        let step_state = state.step_state(v);
        let dt = self.time_step();

        match self.time_integration {
            SingleStepTimeIntegration::BE => State::be_step(step_state, dt),
            SingleStepTimeIntegration::TR => State::tr_step(step_state, dt),
            SingleStepTimeIntegration::BDF2 => State::bdf2_step(step_state, dt),
            SingleStepTimeIntegration::MixedBDF2(t) => {
                State::mixed_bdf2_step(step_state, dt, 1.0 - t as f64)
            }
            SingleStepTimeIntegration::SDIRK2 => {
                let alpha = 1.0 - 0.5 * 2.0_f64.sqrt();
                // let alpha = 1.0;
                State::sdirk2_step(step_state, dt, alpha)
            }
        }
        // let step_state = state.step_state(v);
        // eprintln!("after q: {:?}", &step_state.data.next.q);
    }

    /*
     * The following functions are there for debugging jacobians and hessians
     */
    #[allow(dead_code)]
    pub fn print_jacobian_svd(&self, values: &[T]) {
        use na::DMatrix;

        if values.is_empty() {
            return;
        }

        let (rows, cols) = self.jacobian_indices(true);

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
    fn jacobian_nnz(&self, with_constraints: bool) -> usize {
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

        if with_constraints {
            for vc in self.volume_constraints.iter() {
                num += vc.borrow().penalty_hessian_size();
            }

            // let State {
            //     vtx,..
            // } = &*self.state.borrow();

            let num_active_coords = self.num_variables();
            for fc in self.frictional_contact_constraints.iter() {
                let nh = fc
                    .constraint
                    .borrow()
                    .constraint_hessian_size(num_active_coords / 3);
                let ndh = fc
                    .constraint
                    .borrow()
                    .num_hessian_diagonal_nnz(num_active_coords / 3);
                num += 2 * nh - ndh;
            }

            // dbg!(num);

            for fc in self.frictional_contact_constraints.iter() {
                // Add friction jacobian counts
                let mut constraint = fc.constraint.borrow_mut();
                // constraint.update_state(vtx.cur.pos.view());
                // constraint.update_distance_potential();
                // constraint.update_constraint_gradient();
                constraint.update_multipliers();
                // TODO: Refactor this to just compute the count.
                let dt = T::from(self.time_step()).unwrap();
                let State { vtx, .. } = &*self.state.borrow();
                let f_jac_count = constraint
                    .friction_jacobian_indexed_value_iter(
                        vtx.next.vel.view(),
                        dt,
                        num_active_coords / 3,
                    )
                    .map(|iter| iter.count())
                    .unwrap_or(0);
                // dbg!(f_jac_count);
                num += f_jac_count;
            }
        }

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
        dqdv: S,
    ) {
        let ResidualState { cur, next, r } = state;
        // eprintln!("cur vel : {:?}", &cur.vel);
        solid
            .inertia()
            .add_energy_gradient(cur.vel, next.vel, r, dqdv);
        shell
            .inertia()
            .add_energy_gradient(cur.vel, next.vel, r, dqdv);
    }

    pub fn contact_constraint(&self, _v: &[T]) -> Vec<T> {
        // self.integrate_step(v);
        // self.state.borrow_mut().update_vertices(v);
        let State { vtx, .. } = &*self.state.borrow_mut();
        let pos = vtx.next.pos.view();
        let mut constraint = Vec::new();
        for fc in self.frictional_contact_constraints.iter() {
            let fc_constraint = fc.constraint.borrow_mut();
            // fc_constraint.update_state(pos);
            // fc_constraint.update_distance_potential();
            constraint.extend(fc_constraint.cached_distance_potential(pos.len()));
        }
        constraint
    }

    // Index of the contact constraint for every constraint value returned by `contact_constraint`.
    pub fn contact_constraint_index(&self) -> Vec<usize> {
        let State { vtx, .. } = &*self.state.borrow_mut();
        let pos = vtx.next.pos.view();
        let mut constraint_index = Vec::new();
        for (i, _) in self.frictional_contact_constraints.iter().enumerate() {
            for _ in 0..pos.len() {
                constraint_index.push(i);
            }
        }
        constraint_index
    }

    fn inertia(
        &self,
        state: ResidualState<&[T], &[T], &mut [T]>,
        solid: &TetSolid,
        shell: &TriShell,
        dqdv: T,
    ) -> T {
        let ResidualState { cur, next, .. } = state;
        solid.inertia().energy(cur.vel, next.vel, dqdv)
            + shell.inertia().energy(cur.vel, next.vel, dqdv)
    }

    /// Energy with lagged friction.
    fn energy(
        &self,
        state: ResidualState<&[T], &[T], &mut [T]>,
        solid: &TetSolid,
        shell: &TriShell,
        frictional_contacts: &[FrictionalContactConstraint<T>],
        dqdv: T,
    ) -> T {
        let ResidualState { cur, next, .. } = state;

        let mut energy = solid.elasticity().energy(next.pos, next.vel, dqdv);
        energy += solid.gravity(self.gravity).energy(next.pos, next.vel, dqdv);
        energy += shell.elasticity().energy(next.pos, next.vel, dqdv);
        energy += shell.gravity(self.gravity).energy(next.pos, next.vel, dqdv);

        for vc in self.volume_constraints.iter() {
            energy += vc.borrow().compute_penalty(cur.pos, next.pos);
        }

        // Compute contact potential and lagged friction potential
        let ResidualState { next, .. } = state;
        for fc in frictional_contacts.iter() {
            let fc_constraint = fc.constraint.borrow();
            // fc_constraint.update_state(Chunked3::from_flat(next.pos));
            // fc_constraint.update_distance_potential();
            energy += fc_constraint.contact_constraint();

            energy += fc_constraint.lagged_friction_potential(Chunked3::from_flat(next.vel), dqdv);
        }

        energy
    }

    /// Computes and subtracts constraint forces from the given residual vector `r`.
    ///
    /// `pos` are the stacked position coordinates of all vertices.
    /// `vel` are the stacked velocity coordinates of all vertices.
    /// `lambda` is the workspace per constraint constraint force magnitude.
    /// `r` is the output stacked force vector.
    fn subtract_constraint_forces<S: Real>(
        &self,
        prev_pos: &[T],
        pos: &[S],
        vel: &[S],
        r: &mut [S],
        frictional_contact_constraints: &[FrictionalContactConstraint<S>],
        lagged_friction: bool,
    ) {
        assert_eq!(pos.len(), vel.len());
        assert_eq!(r.len(), pos.len());

        // Add volume constraint indices
        for vc in self.volume_constraints.iter() {
            let timings = &mut *self.timings.borrow_mut();
            add_time!(timings.volume_force; vc.borrow().subtract_pressure_force(prev_pos, pos, r));
        }

        // let mut f = vec![S::zero(); r.len()];

        // Compute contact lambda.
        for fc in frictional_contact_constraints.iter() {
            let fc_constraint = fc.constraint.borrow_mut();

            let timings = &mut *self.timings.borrow_mut();

            add_time!(timings.contact_force; fc_constraint.subtract_constraint_force(Chunked3::from_flat(r)));
            // add_time!(timings.contact_force; fc_constraint.subtract_constraint_force(Chunked3::from_flat(f.as_mut_slice())));
            // fc_constraint.subtract_constraint_force(Chunked3::from_flat(dbgr.as_mut_slice()));
        }
        // eprintln!("f = {:?}", &f);

        self.subtract_friction_forces(
            prev_pos,
            pos,
            vel,
            r,
            frictional_contact_constraints,
            lagged_friction,
        );
    }

    /// Computes and subtracts friction forces from the given residual vector `r`.
    /// This function assumes that the frictional_contact_constraints were already updated
    /// for state, distance potential and multipliers.
    ///
    /// `prev_pos` are the stacked position coordinates of all vertices from the previous step.
    /// `pos` are the stacked position coordinates of all vertices.
    /// `vel` are the stacked velocity coordinates of all vertices.
    /// `lambda` is the workspace per constraint constraint force magnitude.
    /// `r` is the output stacked force vector.
    fn subtract_friction_forces<S: Real>(
        &self,
        _prev_pos: &[T],
        _pos: &[S],
        vel: &[S],
        r: &mut [S],
        frictional_contact_constraints: &[FrictionalContactConstraint<S>],
        lagged: bool,
    ) {
        // assert_eq!(pos.len(), vel.len());
        // assert_eq!(r.len(), pos.len());

        // Compute contact lambda.
        for fc in frictional_contact_constraints.iter() {
            let mut fc_constraint = fc.constraint.borrow_mut();
            let timings = &mut *self.timings.borrow_mut();
            fc_constraint.friction_timings.borrow_mut().clear();
            fc_constraint.subtract_friction_force(
                Chunked3::from_flat(r),
                Chunked3::from_flat(vel),
                lagged,
            );
            timings.friction_force += *fc_constraint.friction_timings.borrow();
        }
    }

    /// Compute the acting force for the problem.
    ///
    /// For the velocity part of the force balance equation
    /// `M dv/dt - f(q,v) = 0`, this function subtracts `f(q,v)`.
    ///
    /// If symmetric is true, lagged friction is enforced.
    fn subtract_force<S: Real>(
        &self,
        state: ResidualState<&[T], &[S], &mut [S]>,
        solid: &TetSolid,
        shell: &TriShell,
        frictional_contacts: &[FrictionalContactConstraint<S>],
        dqdv: S,
        lagged_friction: bool,
    ) {
        let ResidualState { cur, next, r } = state;

        add_time!(self.timings.borrow_mut().energy_gradient; {
            solid.elasticity().add_energy_gradient(next.pos, next.vel, r, dqdv);
            solid
                .gravity(self.gravity)
                .add_energy_gradient(next.pos, next.vel, r, dqdv);
            shell.elasticity().add_energy_gradient(next.pos, next.vel, r, dqdv);
            shell
                .gravity(self.gravity)
                .add_energy_gradient(next.pos, next.vel, r, dqdv);
        });

        self.subtract_constraint_forces(
            cur.pos,
            next.pos,
            next.vel,
            r,
            frictional_contacts,
            lagged_friction,
        );

        debug_assert!(r.iter().all(|r| r.is_finite()));
    }

    /// Compute the residual on simulated vertices using dual numbers.
    ///
    /// This function takes a current force multiplier and previous force multiplier.
    /// For backward Euler, one should pass `force_mul = 1.0` and `prev_force_mul = 0.0`.
    fn compute_vertex_residual_ad_impl(&self, force_mul: f64, prev_force_mul: f64) {
        let State {
            vtx, solid, shell, ..
        } = &mut *self.state.borrow_mut();

        // Clear residual vector.
        vtx.residual_ad
            .storage_mut()
            .iter_mut()
            .for_each(|x| *x = ad::F::zero());

        let dt = ad::FT::cst(T::from(self.time_step()).unwrap());
        let force_mul = ad::FT::cst(T::from(force_mul).unwrap());

        self.subtract_force(
            vtx.residual_state_ad().into_storage(),
            solid,
            shell,
            self.frictional_contact_constraints_ad.as_slice(),
            dt * force_mul,
            false,
        );

        // Save the current force for when the step is advanced.
        // At that moment prev_force is updated to be candidate_force.
        // {
        //     let mut cf = self.candidate_force
        //         .borrow_mut();
        //     cf.clear();
        //     cf.extend(vtx.residual_ad.storage().iter().map(|&x| T::from(x).unwrap()));
        // }

        // If previous-force multiplier is zero, we don't need to keep track of forces between steps.
        if prev_force_mul > 0.0 {
            let prev_force_mul = ad::FT::cst(T::from(prev_force_mul).unwrap());

            // blend between cur and prev force
            vtx.residual_ad
                .storage_mut()
                .iter_mut()
                .zip(self.prev_force.iter())
                .for_each(|(cur_f, &prev_f)| {
                    *cur_f *= force_mul;
                    *cur_f += ad::FT::from(prev_f) * prev_force_mul;
                    *cur_f *= dt;
                });
        } else {
            *vtx.residual_ad.storage_mut().as_mut_tensor() *= force_mul * dt;
        }

        if !self.is_static() {
            let res_state = vtx.residual_state_ad();
            self.add_momentum_diff(res_state.into_storage(), &solid, &shell, dt);
        }
    }

    #[inline]
    fn residual_ad(&self) {
        self.compute_vertex_residual_ad();

        // Transfer residual to degrees of freedom.
        self.state.borrow_mut().dof_residual_ad_from_vertices();
    }

    /// Compute the residual on simulated vertices using dual numbers.
    #[inline]
    fn compute_vertex_residual_ad(&self) {
        let implicit_factor = self.time_integration.implicit_factor();
        let explicit_factor = self.time_integration.explicit_factor();
        self.compute_vertex_residual_ad_impl(implicit_factor as f64, explicit_factor as f64);
    }

    /// Update current vertex state to coincide with current dof state.
    pub fn update_cur_vertices(&mut self) {
        let state = &mut *self.state.borrow_mut();
        // eprintln!("during update cur prev dq = {:?}", &state.dof.data.prev.dq);
        match self.time_integration {
            SingleStepTimeIntegration::BDF2 => {
                state.update_cur_vertices_with_lerp(-1.0 / 3.0, 4.0 / 3.0)
            }
            SingleStepTimeIntegration::MixedBDF2(t) => {
                let t = t as f64;
                // Comments correspond to quantities directly from the TRBDF2 formula.
                // let gamma = 1.0 - t;
                // let a = 1.0 / (gamma * (2.0 - gamma));
                let a = 1.0 / (1.0 - t * t);
                // let b = (1.0 - gamma)^2 / (gamma * (2 - gamma));
                let b = t * t / (1.0 - t * t);
                state.update_cur_vertices_with_lerp(-b, a)
            }
            SingleStepTimeIntegration::SDIRK2 => {
                // For SDIRK2 we use the previous state since it's a two step method.
                state.update_cur_vertices_with_lerp(1.0, 0.0)
            }
            _ => state.update_cur_vertices_direct(),
        }
        // eprintln!("cur vel : {:?}", &state.vtx.cur.vel);
    }

    /// Clear current velocity dofs.
    pub fn clear_velocities(&mut self) {
        let state = &mut *self.state.borrow_mut();
        state.clear_velocities();
    }

    /// Compute the objective on simulated vertices.
    ///
    /// This is typically the total system energy with lagged friction potential.
    pub fn compute_objective(&self, force_mul: f64, prev_force_mul: f64) -> T {
        let State {
            vtx, solid, shell, ..
        } = &mut *self.state.borrow_mut();

        let force_mul = T::from(force_mul).unwrap();
        let dt = T::from(self.time_step()).unwrap();
        let dqdv = dt * force_mul;

        let mut objective = T::zero();

        if prev_force_mul > 0.0 {
            let prev_force_mul = T::from(prev_force_mul).unwrap();
            objective += self
                .prev_force
                .iter()
                .zip(vtx.next.vel.storage().iter())
                .map(|(&r, &v)| r * v)
                .sum::<T>()
                * prev_force_mul
                * dt;
        }

        objective += self.energy(
            vtx.residual_state().into_storage(),
            solid,
            shell,
            self.frictional_contact_constraints.as_slice(),
            dqdv,
        );

        if !self.is_static() {
            objective += self.inertia(vtx.residual_state().into_storage(), solid, shell, dqdv);
        }
        objective
    }

    /// Compute the residual on simulated vertices.
    ///
    /// This function takes a current force multiplier and previous force multiplier.
    /// For backward euler, one should pass `force_mul = 1.0` and `prev_force_mul = 0.0`.
    fn compute_vertex_residual_impl(&self, force_mul: f64, prev_force_mul: f64, symmetric: bool) {
        let State {
            vtx, solid, shell, ..
        } = &mut *self.state.borrow_mut();

        // Clear residual vector.
        vtx.residual
            .storage_mut()
            .iter_mut()
            .for_each(|x| *x = T::zero());

        let dt = T::from(self.time_step()).unwrap();
        let force_mul = T::from(force_mul).unwrap();

        let dqdv = dt * force_mul;

        self.subtract_force(
            vtx.residual_state().into_storage(),
            solid,
            shell,
            self.frictional_contact_constraints.as_slice(),
            dqdv,
            symmetric,
        );

        // eprintln!("dt = {:?}; force_mul = {:?}; prev_force_mul = {:?}", dt, force_mul, prev_force_mul);

        // Save the current force for when the step is advanced.
        // At that moment prev_force is updated to be candidate_force.
        self.candidate_force
            .borrow_mut()
            .clone_from(vtx.residual.storage());

        // If previous-force multiplier is zero, we don't need to keep track of forces between steps.
        if prev_force_mul > 0.0 {
            let prev_force_mul = T::from(prev_force_mul).unwrap();

            // Blend between prev and cur forces using the given multipliers
            vtx.residual
                .storage_mut()
                .iter_mut()
                .zip(self.prev_force.iter())
                .for_each(|(cur_f, &prev_f)| {
                    *cur_f *= force_mul;
                    *cur_f += prev_f * prev_force_mul;
                    *cur_f *= dt;
                });
        } else {
            *vtx.residual.storage_mut().as_mut_tensor() *= dqdv;
        }

        if !self.is_static() {
            let res_state = vtx.residual_state();
            self.add_momentum_diff(res_state.into_storage(), solid, shell, dqdv);
        }
    }

    /// Compute the residual on simulated vertices.
    ///
    /// If symmetric is false, lagged friction will be used which symmetrizes the system completely.
    #[inline]
    fn compute_vertex_residual(&self, symmetric: bool) {
        let implicit_factor = self.time_integration.implicit_factor();
        let explicit_factor = self.time_integration.explicit_factor();
        self.compute_vertex_residual_impl(
            implicit_factor as f64,
            explicit_factor as f64,
            symmetric,
        );
    }

    fn jacobian_indices(&self, with_constraints: bool) -> (Vec<usize>, Vec<usize>) {
        let jac_nnz = self.jacobian_nnz(with_constraints);
        let mut rows = vec![0; jac_nnz];
        let mut cols = vec![0; jac_nnz];
        let count = self.compute_jacobian_indices(&mut rows, &mut cols, with_constraints);

        rows.resize(count, 0);
        cols.resize(count, 0);

        (rows, cols)
    }

    /// Returns number of actual non-zeros in the Jacobian.
    fn compute_jacobian_indices(
        &self,
        rows: &mut [usize],
        cols: &mut [usize],
        with_constraints: bool,
    ) -> usize {
        let num_active_coords = self.num_variables();
        let mut count = 0; // Constraint counter

        let state = self.state.borrow();

        // Add energy indices
        let solid = &state.solid;
        let elasticity = solid.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity
            .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);

        count += n;

        let shell = &state.shell;
        let elasticity = shell.elasticity::<T>();
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

        if !self.is_static() {
            let inertia = shell.inertia();
            let n = inertia.energy_hessian_size();
            inertia
                .energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
            count += n;
        }

        // Duplicate off-diagonal indices to form a complete matrix.
        let (rows_begin, rows_end) = rows.split_at_mut(count);
        let (cols_begin, cols_end) = cols.split_at_mut(count);

        // Ensure there is nothing in the upper triangular part.
        debug_assert_eq!(
            rows_begin
                .iter()
                .zip(cols_begin.iter())
                .filter(|(&r, &c)| r < c)
                .count(),
            0
        );

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

        // Cache computed nonzeros without friction
        // Update jacobian sparsity for jacobian product computation.
        {
            let mut jws = self.jacobian_workspace.borrow_mut();
            jws.rows.resize(count, 0);
            jws.cols.resize(count, 0);
            jws.vals.resize(count, T::zero());
            jws.rows.clone_from_slice(&rows[..count]);
            jws.cols.clone_from_slice(&cols[..count]);
            jws.stale = true;

            let JacobianWorkspace {
                rows,
                cols,
                vals,
                jac,
                mapping,
                ..
            } = &mut *jws;
            (*jac, *mapping) = crate::fem::nl::newton::sparse_matrix_and_mapping(
                rows,
                cols,
                vals,
                num_active_coords,
                false,
            );
        }

        if with_constraints {
            // Add volume constraint indices
            for vc in self.volume_constraints.iter() {
                let mut nh = 0;
                for MatrixElementIndex { row, col } in vc.borrow().penalty_hessian_indices_iter() {
                    rows[count] = row;
                    cols[count] = col;
                    count += 1;
                    nh += 1;
                }
                assert_eq!(nh, vc.borrow().penalty_hessian_size());
            }

            // eprintln!("i pre contact = {count}");

            // Compute friction derivatives.
            // Note that friction Jacobian is non-symmetric and so must appear after the symmetrization above.

            // Add contact constraint Jacobian
            for fc in self.frictional_contact_constraints.iter() {
                let constraint = fc.constraint.borrow();
                // Indices for constraint Hessian first term (multipliers held constant)
                // let bcount = count;
                count += constraint
                    .constraint_hessian_indices_iter(num_active_coords / 3)
                    //.filter(|idx| idx.row < num_active_coords && idx.col < num_active_coords)
                    .zip(rows[count..].iter_mut().zip(cols[count..].iter_mut()))
                    .map(|(MatrixElementIndex { row, col }, (out_row, out_col))| {
                        *out_row = row;
                        *out_col = col;
                    })
                    .count();
                // use std::io::Write;
                // let mut f = std::fs::File::create("./out/jac_indices.jl").unwrap();
                // writeln!(f, "jrows = {:?}", &rows[bcount..count]).unwrap();
                // writeln!(f, "jcols = {:?}", &cols[bcount..count]).unwrap();
                // eprintln!("jrows = {:?}", &rows[bcount..]);
                // eprintln!("jcols = {:?}", &cols[bcount..]);
                // dbg!(count - bcount);
            }

            // let State { vtx, .. } = &*self.state.borrow();

            // dbg!(count);
            // dbg!(rows[count..].len());

            // Add Non-symmetric friction Jacobian entries.
            for fc in self.frictional_contact_constraints.iter() {
                let mut constraint = fc.constraint.borrow_mut();
                // constraint.update_state(vtx.cur.pos.view());
                // constraint.update_distance_potential();
                // constraint.update_constraint_gradient();
                // constraint.update_multipliers(delta, kappa);
                // Compute friction hessian second term (multipliers held constant)
                let dt = T::from(self.time_step()).unwrap();
                let State { vtx, .. } = &*self.state.borrow();
                let f_jac_count = constraint
                    .friction_jacobian_indexed_value_iter(
                        vtx.next.vel.view(),
                        dt,
                        num_active_coords / 3,
                    )
                    .map(|iter| {
                        iter.zip(rows[count..].iter_mut().zip(cols[count..].iter_mut()))
                            .map(|((row, col, _), (out_row, out_col))| {
                                *out_col = col;
                                *out_row = row;
                            })
                            .count()
                    })
                    .unwrap_or(0);
                // dbg!(f_jac_count);
                count += f_jac_count;
            }
        }
        count
    }

    /// Jacobian diagonal excluding constraints.
    #[inline]
    fn jacobian_incomplete_diagonal(&self, v: &[T], r: &[T], diag: &mut [T]) {
        // self.integrate_step(v);
        // self.state.borrow_mut().update_vertices(v);
        let implicit_factor = T::from(self.time_integration.implicit_factor()).unwrap();
        let dt = T::from(self.time_step()).unwrap();
        let h = dt * implicit_factor;
        self.jacobian_diagonal_impl(v, r, diag, h, h);
    }

    fn jacobian_diagonal_impl<'v>(
        &self,
        _dq: &'v [T],
        _r: &[T],
        diag: &mut [T],
        // Derivative of q+ with respect to v+
        dqdv: T,
        // Multiplier for force Jacobians
        force_multiplier: T,
    ) {
        let state = &mut *self.state.borrow_mut();

        // Multiply energy hessian by objective factor and scaling factors.
        let factor = T::from(self.impulse_inv_scale()).unwrap();

        let State {
            vtx, solid, shell, ..
        } = state;

        let ResidualState { cur, next, .. } = vtx.residual_state().into_storage();
        let elasticity = solid.elasticity::<T>();
        elasticity.add_energy_hessian_diagonal(
            next.pos,
            next.vel,
            dqdv * force_multiplier * factor,
            diag,
            dqdv,
        );

        let elasticity = shell.elasticity::<T>();
        elasticity.add_energy_hessian_diagonal(
            next.pos,
            next.vel,
            dqdv * force_multiplier * factor,
            diag,
            dqdv,
        );

        if !self.is_static() {
            let inertia = solid.inertia();
            inertia.add_energy_hessian_diagonal(cur.vel, next.vel, factor, diag, dqdv);
        }

        if !self.is_static() {
            let inertia = shell.inertia();
            inertia.add_energy_hessian_diagonal(cur.vel, next.vel, factor, diag, dqdv);
        }
    }

    #[inline]
    fn jacobian_values_with_constraints(
        &self,
        v: &[T],
        r: &[T],
        rows: &[usize],
        cols: &[usize],
        vals: &mut [T],
        with_constraints: bool,
    ) {
        // self.integrate_step(v);
        // self.state.borrow_mut().update_vertices(v);
        let implicit_factor = T::from(self.time_integration.implicit_factor()).unwrap();
        let dt = T::from(self.time_step()).unwrap();
        let h = dt * implicit_factor;
        self.jacobian_values(v, r, rows, cols, vals, h, h, with_constraints);
    }

    fn jacobian_values<'v>(
        &self,
        _dq: &'v [T],
        _r: &[T],
        rows: &[usize],
        cols: &[usize],
        vals: &mut [T],
        // Derivative of q+ with respect to v+
        dqdv: T,
        // Multiplier for force Jacobians
        force_multiplier: T,
        with_constraints: bool,
    ) {
        let num_active_coords = self.num_variables();
        let state = &mut *self.state.borrow_mut();

        let mut count = 0; // Values counter

        // Multiply energy hessian by objective factor and scaling factors.
        let factor = T::from(self.impulse_inv_scale()).unwrap();

        let State {
            vtx, solid, shell, ..
        } = state;

        let t_begin = Instant::now();
        let ResidualState { cur, next, .. } = vtx.residual_state().into_storage();
        let elasticity = solid.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity.energy_hessian_values(
            next.pos,
            next.vel,
            dqdv * force_multiplier * factor,
            &mut vals[count..count + n],
            dqdv,
        );
        count += n;

        let elasticity = shell.elasticity::<T>();
        let n = elasticity.energy_hessian_size();
        elasticity.energy_hessian_values(
            next.pos,
            next.vel,
            dqdv * force_multiplier * factor,
            &mut vals[count..count + n],
            dqdv,
        );
        count += n;

        // use std::io::Write;
        // let mut f = std::fs::File::create("./out/jac2_vals.jl").unwrap();
        // writeln!(f, "jvals2 = {:?}", &vals[..count]).unwrap();

        if !self.is_static() {
            let inertia = solid.inertia();
            let n = inertia.energy_hessian_size();
            inertia.energy_hessian_values(
                cur.vel,
                next.vel,
                factor,
                &mut vals[count..count + n],
                dqdv,
            );
            count += n;
        }

        if !self.is_static() {
            let inertia = shell.inertia();
            let n = inertia.energy_hessian_size();
            inertia.energy_hessian_values(
                cur.vel,
                next.vel,
                factor,
                &mut vals[count..count + n],
                dqdv,
            );
            count += n;
        }

        // eprintln!("lower triangular = {count}");

        // Duplicate off-diagonal entries.
        let (vals_lower, vals_upper) = vals.split_at_mut(count);
        let t_end_of_fem = Instant::now();

        // Check that there are the same number of off-diagonal elements in vals_lower as there are
        // total elements in vals_upper
        //debug_assert_eq!(
        //     rows.iter()
        //         .zip(cols.iter())
        //         .zip(vals_lower.iter())
        //         .filter(|((&r, &c), _)| r != c)
        //         .count(),
        //     vals_upper.len()
        // );

        // Fill upper trianular part.
        count += rows
            .iter()
            .zip(cols.iter())
            .zip(vals_lower.iter())
            .filter(|((&r, &c), _)| r != c)
            .zip(vals_upper.iter_mut())
            .map(|((_, &val), out_val)| {
                *out_val = val;
            })
            .count();

        let t_end_of_diag = Instant::now();

        // // DEBUG CODE
        // let mut hess = vec![vec![0.0; num_active_coords]; num_active_coords];
        // for vc in self.volume_constraints.iter() {
        //     let vc = vc.borrow_mut();
        //     for (idx, val) in vc.constraint_hessian_indices_iter().zip(vc.constraint_hessian_values_iter(
        //             cur.pos,
        //             next.pos,
        //             &[T::one()][..],
        //         )) {
        //         if idx.row < num_active_coords && idx.col < num_active_coords {
        //             hess[idx.row][idx.col] += val.to_f64().unwrap();
        //         }
        //     }
        // }
        // eprintln!("h = [");
        // for row in hess.iter() {
        //     for entry in row.iter() {
        //         eprint!("{entry:?} ");
        //     }
        //     eprintln!(";");
        // }
        // eprintln!("]");
        // // END OF DEBUG CODE

        if with_constraints {
            // Add volume constraint entries.
            for vc in self.volume_constraints.iter() {
                let nh = vc.borrow().penalty_hessian_size();
                vc.borrow_mut()
                    .penalty_hessian_values(
                        cur.pos,
                        next.pos,
                        &[T::one()][..],
                        dqdv * force_multiplier * self.volume_constraint_scale(),
                        &mut vals[count..count + nh],
                    )
                    .unwrap();

                count += nh;
            }
        }

        let t_end_of_volume = Instant::now();

        // eprintln!("pre contact = {count}");

        // Compute friction derivatives.
        // Note that friction Jacobian is non-symmetric and so must appear after the symmetrization above.

        // let mut jac = vec![vec![0.0; num_active_coords]; num_active_coords];

        if with_constraints {
            // Add contact constraint jacobian entries here.
            for fc in self.frictional_contact_constraints.iter() {
                let constraint = fc.constraint.borrow_mut();
                // constraint.update_state_with_rebuild(Chunked3::from_flat(next.pos), false);
                //constraint.update_distance_potential();
                // constraint.update_constraint_gradient();
                // constraint.update_multipliers(delta, kappa);
                // Compute constraint hessian first term (multipliers held constant)
                // let bcount = count;

                count += constraint
                    .constraint_hessian_indexed_values_iter(num_active_coords / 3)
                    .zip(vals[count..].iter_mut())
                    .map(|((_, val), out_val)| {
                        // if idx.row == 10 && idx.col == 10 {
                        // eprintln!(
                        //     "contact jac ({:?}, {:?}): {:?}",
                        //     idx.row,
                        //     idx.col,
                        //     -dqdv * force_multiplier * factor * val
                        // );
                        // }
                        *out_val = -dqdv * force_multiplier * factor * val;
                        // jac[idx.row][idx.col] += out_val.to_f64().unwrap();
                    })
                    .count();
                // use std::io::Write;
                // let mut f = std::fs::File::create("./out/jac_vals.jl").unwrap();
                // writeln!(f, "jvals = {:?}", &vals[bcount..count]).unwrap();
                // dbg!(count - bcount);
            }
        }

        let t_end_of_contact = Instant::now();

        // eprintln!("cj = [");
        // for row in jac.iter() {
        //     for entry in row.iter() {
        //         eprint!("{entry:?} ");
        //     }
        //     eprintln!(";");
        // }
        // eprintln!("]");
        //
        //let orig_vel = next.vel.to_vec();
        //let orig_pos = cur.pos.to_vec();

        // dbg!(&orig_pos);
        // dbg!(&next.pos);
        // dbg!(&orig_vel);

        // dbg!(count);
        // dbg!(vals[count..].len());

        if with_constraints {
            // Add Non-symmetric friction Jacobian entries.
            // let n = num_active_coords;
            for fc in self.frictional_contact_constraints.iter() {
                let mut constraint = fc.constraint.borrow_mut();
                // constraint.update_state(Chunked3::from_flat(cur.pos.view()));
                // constraint.update_distance_potential();
                // constraint.update_constraint_gradient();
                // constraint.update_multipliers(delta, kappa);

                // let mut jac = vec![vec![0.0; n]; n];
                // Compute friction hessian second term (multipliers held constant)
                constraint.jac_timings.borrow_mut().clear();
                // dbg!(num_active_coords);
                let f_jac_count = constraint
                    .friction_jacobian_indexed_value_iter(
                        Chunked3::from_flat(next.vel),
                        dqdv,
                        num_active_coords / 3,
                    )
                    .map(|iter| {
                        iter.zip(vals[count..].iter_mut())
                            .map(|((_r, _c, val), out_val)| {
                                // jac[_r][_c] = val.to_f64().unwrap();
                                // *out_val = T::zero();
                                // if _r == 29 && _c == 29 {
                                //     eprintln!(
                                //         "({_r},{_c}): {:?} -- {:?}",
                                //         val,
                                //         force_multiplier * factor * val
                                //     );
                                // }
                                *out_val = force_multiplier * factor * val;
                            })
                            .count()
                    })
                    .unwrap_or(0);
                // dbg!(f_jac_count);
                *self.jac_timings.borrow_mut() += *constraint.jac_timings.borrow();
                count += f_jac_count;

                // eprintln!("FRICTION JACOBIAN:");
                // for row in 0..n {
                //     for col in 0..n {
                //         eprint!("{:10.2e} ", jac[row][col]);
                //     }
                //     eprintln!("");
                // }
            }
        }

        let t_end = Instant::now();

        // let ResidualState { next, r, .. } = vtx.residual_state_ad().into_storage();
        // let mut vel = next.vel.to_vec();
        // let mut next_pos = next.pos.to_vec();
        // let mut cur_pos = next_pos.clone();
        //
        // // Clear duals
        // for i in 0..n {
        //     cur_pos[i].x = orig_pos[i];
        //     cur_pos[i].dx = T::zero();
        //     vel[i].x = orig_vel[i];
        //     vel[i].dx = T::zero();
        // }
        //
        // for fc in self.frictional_contact_constraints_ad.iter() {
        //     let mut constraint = fc.constraint.borrow_mut();
        //     let delta = self.delta as f32;
        //     let kappa = self.kappa as f32;
        //
        //     let mut jac = vec![vec![0.0; n]; n];
        //     for i in 0..n {
        //         eprintln!("FD AUTODIFF WRT {}", i);
        //         vel[i].dx = T::one();
        //
        //         for (next_pos, &cur_pos, &vel) in zip!(next_pos.iter_mut(), cur_pos.iter(), vel.iter()) {
        //             *next_pos = cur_pos + vel * dt;
        //         }
        //
        //         // Clear residual vector
        //         for j in 0..n {
        //             r[j] = autodiff::F::zero();
        //         }
        //
        //         constraint.update_state(Chunked3::from_flat(&next_pos));
        //         constraint.update_distance_potential();
        //         constraint.update_multipliers(delta, kappa);
        //         constraint.update_constraint_gradient();
        //
        //         constraint
        //             .subtract_friction_force(Chunked3::from_flat(r), Chunked3::from_flat(vel.view()));
        //
        //         for j in 0..n {
        //             jac[j][i] = r[j].deriv().to_f64().unwrap();
        //         }
        //
        //         vel[i].dx = T::zero();
        //     }
        //     eprintln!("FRICTION JACOBIAN AUTODIFF:");
        //     for row in 0..n {
        //         for col in 0..n {
        //             eprint!("{:10.2e} ", jac[row][col]);
        //         }
        //         eprintln!("");
        //     }
        // }

        let timings = &mut *self.timings.borrow_mut();
        timings.jacobian.fem += t_end_of_fem - t_begin;
        timings.jacobian.diag += t_end_of_diag - t_end_of_fem;
        timings.jacobian.volume += t_end_of_volume - t_end_of_diag;
        timings.jacobian.contact += t_end_of_contact - t_end_of_volume;
        timings.jacobian.friction += t_end - t_end_of_contact;
        timings.jacobian.total += t_end - t_begin;
        timings.total += t_end - t_begin;

        //self.print_jacobian_svd(vals);
        *self.iter_counter.borrow_mut() += 1;

        debug_assert!(vals.iter().take(count).all(|x| x.is_finite()));
    }

    /// Use automatic differentiation to compute the Jacobian product.
    #[allow(dead_code)]
    fn jacobian_product_ad(&self, v: &[T], p: &[T], jp: &mut [T]) {
        let t_begin = Instant::now();

        self.update_state_ad(v, p, false);

        // Compute residual using dual numbers, the result is stored in dof residual.
        self.residual_ad();

        let State { dof, .. } = &mut *self.state.borrow_mut();
        for (jp, r_ad) in jp.iter_mut().zip(dof.storage_mut().r_ad.iter()) {
            *jp = T::from(r_ad.deriv()).unwrap();
        }
        self.timings.borrow_mut().total += Instant::now() - t_begin;
    }

    /// Computes the jacobi preconditioner for the symmetric part only (already inverted and sqrt).
    #[allow(dead_code)]
    fn compute_incomplete_jacobi_preconditioner(&self, v: &[T], jacobi: &mut [T]) {
        let PreconditionerWorkspace { ref mut buffer } =
            &mut *self.preconditioner_workspace.borrow_mut();

        let nverts = self.state.borrow().vtx.next.len();
        buffer.clear();
        buffer.resize(3 * nverts, T::zero());

        self.jacobian_incomplete_diagonal(v, &[], buffer);

        crate::fem::nl::state::vtx_to_dof(
            buffer,
            ChunkedView::from_offsets(&[0, jacobi.len()], jacobi),
        );

        jacobi.iter_mut().for_each(|x| {
            // Guard against degeneracies (it's an incomplete preconditioner, this doesn't
            // necessarily mean that the jacobian is non-invertible).
            let xsqrt = num_traits::Float::sqrt(num_traits::Float::abs(*x));
            if xsqrt < T::from(f64::EPSILON).unwrap() {
                *x = T::one();
            } else {
                *x = T::one() / xsqrt
            }
        });

        // eprintln!("jacobi = {:?}", jacobi);
    }

    /// Computes a simple diagonal preconditioner. (inverted and sqrt).
    fn compute_physical_preconditioner(&self, v: &[T], precond: &mut [T]) {
        let PreconditionerWorkspace { ref mut buffer } =
            &mut *self.preconditioner_workspace.borrow_mut();

        // Compute FEM (non-constraint) part of the jacobi preconditioner.
        let nverts = self.state.borrow().vtx.next.len();
        buffer.clear();
        buffer.resize(3 * nverts, T::zero());

        self.jacobian_incomplete_diagonal(v, &[], buffer);

        crate::fem::nl::state::vtx_to_dof(
            buffer,
            ChunkedView::from_offsets(&[0, precond.len()], precond),
        );

        let dt = T::from(self.time_step()).unwrap();
        let h = T::from(self.time_integration.implicit_factor()).unwrap() * dt;

        // Next compute and add the constraint part.

        // For each potential contact, apply a scale in the normal and tangential direction
        // Then map it to vertices via sliding basis mapping.

        let nvert_dofs = v.len() / 3;

        // self.integrate_step(v);
        // self.state.borrow_mut().update_vertices(v);

        let State { vtx, .. } = &*self.state.borrow_mut();
        let vtx_vel = vtx.next.vel.view();

        for fc in self.frictional_contact_constraints.iter() {
            // {
            let constraint = &mut *fc.constraint.borrow_mut();

            let delta = constraint.params.tolerance;
            let kappa = T::from(constraint.params.stiffness).unwrap();

            let pot = constraint.contact_state.distance_potential.as_slice();
            let lambda = constraint.contact_state.lambda.as_slice();
            {
                constraint
                    .object_distance_potential_hessian_indexed_blocks_iter(lambda)
                    .for_each(|(row, col, mtx)| {
                        if row == col && row < nvert_dofs {
                            for k in 0..3 {
                                precond[3 * row + k] -= h * h * mtx[k][k];
                            }
                        }
                    });
                constraint
                    .collider_distance_potential_hessian_indexed_blocks_iter(lambda)
                    .for_each(|(idx, mtx)| {
                        if idx < nvert_dofs {
                            for k in 0..3 {
                                precond[3 * idx + k] -= h * h * mtx[k][k];
                            }
                        }
                    });

                // Contact
                let MappedDistanceGradient { matrix: g, .. } = constraint
                    .contact_state
                    .distance_gradient
                    .borrow()
                    .expect("Uninitialized constraint gradient.");
                let g_view = g.view();

                // Compute diagonal of distance gradient * distance gradient transpose
                g_view.into_iter().for_each(|(vtx_idx, lhs_row)| {
                    if vtx_idx < nvert_dofs {
                        lhs_row.into_expr().for_each(|IndexedExpr { index, expr }| {
                            let ddb = ContactPenalty::new(delta).ddb(pot[index]);
                            for i in 0..3 {
                                precond[3 * vtx_idx + i] +=
                                    kappa * h * h * ddb * expr[i][0] * expr[i][0];
                            }
                        });
                    }
                });
            }

            if constraint.params.friction_params.is_none() {
                continue;
            }
            let friction_params = constraint.params.friction_params;
            // }

            // Friction

            // if let Some(f_iter) = fc
            //     .constraint
            //     .borrow_mut()
            //     .friction_jacobian_indexed_value_iter(
            //         vtx_vel,
            //         self.delta as f32,
            //         self.kappa as f32,
            //         self.epsilon as f32,
            //         h,
            //         nvert_dofs,
            //         false,
            //     )
            // {
            //     f_iter.for_each(|(row, col, val)| {
            //         if row == col {
            //             precond[row] += h * val;
            //         }
            //     })
            // }

            let constrained_collider_vertices = constraint
                .contact_state
                .constrained_collider_vertices
                .as_slice();
            let contact_basis = &constraint.contact_state.contact_basis;

            let mu = T::from(friction_params.dynamic_friction).unwrap();
            let jac = constraint.contact_state.contact_jacobian.as_ref().unwrap();
            let mut vc = jac.mul(
                vtx_vel,
                &constrained_collider_vertices,
                &constraint.implicit_surface_vertex_indices,
                &constraint.collider_vertex_indices,
            );

            for (contact_idx, ((scale_contact, &d), (jac_row_idx, jac_row))) in vc
                .iter_mut()
                .zip(pot.iter())
                .zip(jac.matrix.view().into_iter())
                .enumerate()
            {
                assert_eq!(contact_idx, jac_row_idx); // jac should be dense.
                let vc = contact_basis.to_contact_coordinates(*scale_contact, contact_idx);
                let lambda = -ContactPenalty::new(delta).db(d);
                let v2 = Vector2::from([vc[1], vc[2]]);
                let mtx = friction_params.friction_profile.jacobian(
                    v2,
                    lambda,
                    T::from(friction_params.epsilon).unwrap(),
                ) * (kappa * mu * h * h);
                let eta_jac = [
                    [T::zero(); 3],
                    [T::zero(), mtx[0][0], mtx[0][1]],
                    [T::zero(), mtx[1][0], mtx[1][1]],
                ]
                .into_tensor();
                let b = contact_basis.contact_basis_matrix(contact_idx);
                let diag_block = b.transpose() * eta_jac * b;

                // Collider
                let vtx_idx =
                    constraint.collider_vertex_indices[constrained_collider_vertices[contact_idx]];
                if vtx_idx < nvert_dofs {
                    for i in 0..3 {
                        precond[3 * vtx_idx + i] += diag_block[i][i];
                    }
                }

                // // Implicit surface
                for (surf_vtx_idx, block) in jac_row.into_iter() {
                    let vtx_idx = constraint.implicit_surface_vertex_indices[surf_vtx_idx];
                    if vtx_idx < nvert_dofs {
                        let block = block.into_arrays().as_tensor().transpose();
                        for (i, &row) in block.data.iter().enumerate() {
                            precond[3 * vtx_idx + i] += row.dot(diag_block * row);
                        }
                    }
                }
            }
        }

        // Finally invert and square root the preconditioner so it can be multiplied directly
        // from the left and right.
        precond.iter_mut().for_each(|x| {
            // Guard against degeneracies.
            let xsqrt = num_traits::Float::sqrt(num_traits::Float::abs(*x));
            if xsqrt < T::from(f64::EPSILON).unwrap() {
                *x = T::one();
            } else {
                *x = T::one() / xsqrt
            }
        });

        // eprintln!("precond = {:?}", precond);
    }

    /// Computes the Jacobian product using forward mode AD for friction Jacobian.
    fn jacobian_product_constraint_ad(&self, v: &[T], p: &[T], jp: &mut [T]) {
        let t_begin = Instant::now();
        // Compute Jacobian product normally. Use AD for friction and contact only.

        // First pre-compute the Jacobian rows, cols and values.
        let JacobianWorkspace {
            rows,
            cols,
            vals,
            jac,
            mapping,
            stale,
            ..
        } = &mut *self.jacobian_workspace.borrow_mut();

        if *stale {
            self.jacobian_values_with_constraints(v, &[], rows, cols, vals, false);
            // Redistribute values
            jac.storage_mut().fill(T::zero());
            for (&pos, &j_val) in mapping.iter().zip(vals.iter()) {
                if let Some(pos) = pos.into_option() {
                    jac.storage_mut()[pos] += j_val;
                }
            }

            *stale = false;
        }

        // Compute the contact and friction Jacobians using AD

        self.update_state_ad(v, p, false);

        let t_update = Instant::now();

        let multiplier: f32 = self.time_integration.implicit_factor();

        {
            let State { vtx, .. } = &mut *self.state.borrow_mut();

            // Clear residual vector.
            vtx.residual_ad
                .storage_mut()
                .iter_mut()
                .for_each(|x| *x = ad::F::zero());

            let ResidualState { cur, next, r } = vtx.residual_state_ad().into_storage();
            self.subtract_constraint_forces(
                cur.pos,
                next.pos,
                next.vel,
                r,
                self.frictional_contact_constraints_ad.as_slice(),
                false,
            );
            let dt = ad::FT::<T>::cst(T::from(self.time_step()).unwrap());
            let force_mul = ad::FT::<T>::cst(T::from(multiplier).unwrap());
            *vtx.residual_ad.storage_mut().as_mut_tensor() *= force_mul * dt;
        }

        let t_force = Instant::now();

        // Transfer residual to degrees of freedom.
        self.state.borrow_mut().dof_residual_ad_from_vertices();

        let t_dof_to_vtx = Instant::now();

        let State { dof, .. } = &mut *self.state.borrow_mut();
        for (jp, r_ad) in jp.iter_mut().zip(dof.storage_mut().r_ad.iter()) {
            *jp = T::from(r_ad.deriv()).unwrap();
        }

        let t_read_deriv = Instant::now();

        // Add remaining Jacobian product
        // let num_active_variables = v.len();
        // for ((&row, &col), &val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        //     if row < num_active_variables && col < num_active_variables {
        //         test_jp1[row] += val * p[col];
        //     }
        // }
        jac.view()
            .into_data()
            .into_par_iter()
            .zip(jp.par_iter_mut())
            .for_each(|(row, jp)| {
                for (col_idx, &j_val) in row.into_iter() {
                    *jp += j_val * p[col_idx];
                }
            });

        let t_product = Instant::now();

        let timings = &mut *self.timings.borrow_mut();
        timings.force_ad += t_force - t_update;
        timings.dof_to_vtx_ad += t_dof_to_vtx - t_force;
        timings.read_deriv_ad += t_read_deriv - t_dof_to_vtx;
        timings.product_ad += t_product - t_read_deriv;
        timings.total += Instant::now() - t_begin;
    }

    /// Same as `update_state_ad` but uses an all zero vector for the dual part of `x`.
    fn update_state_ad_cst(&self, x: &[T], rebuild_tree: bool) {
        self.update_state_ad(x, std::iter::repeat(&T::zero()), rebuild_tree);
    }

    fn update_state_ad<'a, PI>(&self, v: &[T], p: PI, rebuild_tree: bool)
    where
        PI: IntoIterator<Item = &'a T>,
    {
        {
            let State { dof, .. } = &mut *self.state.borrow_mut();
            let dq_ad = &mut dof.storage_mut().next_ad.dq;
            // Construct automatically differentiable vector of velocities.
            // The residual will be computed into the state workspace vector r_ad.
            for (&v, &p, dq_ad) in zip!(v.iter(), p.into_iter(), dq_ad.iter_mut()) {
                *dq_ad = ad::FT::new(v, p);
            }
            // End of dynamic mutable borrow.
        }

        self.integrate_step_ad();
        self.state.borrow_mut().update_vertices_ad();
        let state = self.state.borrow();
        let pos = state.vtx.next_ad.pos.view();

        for fc in self.frictional_contact_constraints_ad.iter() {
            let mut fc_constraint = fc.constraint.borrow_mut();
            let timings = &mut *self.timings.borrow_mut();

            fc_constraint.update_timings.borrow_mut().clear();
            add_time!(timings.update_state; fc_constraint.update_state_with_rebuild(pos, rebuild_tree));
            add_time!(timings.update_distance_potential; fc_constraint.update_distance_potential() );
            add_time!(timings.update_constraint_gradient; fc_constraint.update_constraint_gradient() );
            add_time!(timings.update_multipliers; fc_constraint.update_multipliers());
            add_time!(timings.update_sliding_basis; fc_constraint.update_sliding_basis(false, false, pos.len()));
            timings.update_constraint_details += *fc_constraint.update_timings.borrow();
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ContactViolation {
    pub bump_ratio: f64,
    pub violation: f64,
    pub penetration: f64,
    pub largest_penalty: f64,
    pub violating_constraint_index: usize,
}

/// An api for a non-linear problem.
pub trait NonLinearProblem<T: Real> {
    fn residual_timings(&self) -> RefMut<'_, ResidualTimings>;
    fn jacobian_timings(&self) -> RefMut<'_, FrictionJacobianTimings>;

    fn debug_friction(&self) -> Ref<'_, Vec<T>>;

    /// Returns the time step size.
    fn time_step(&self) -> f64;

    /// Number of vertices in contact, or within `delta` of the surface.
    fn num_contacts(&self) -> usize;

    /// Number of vertices in close proximity to be detected by the spatial tree.
    fn num_in_proximity(&self) -> usize;

    /// Returns a mesh using current state data.
    fn mesh(&self) -> Mesh;

    fn save_contact_jac(&self, i: usize);

    /// Returns a mesh updated with the given velocity information.
    fn mesh_with(&self, dq: &[T]) -> Mesh;

    /// Returns a vector of lumped vertex masses.
    fn lumped_mass_inv(&self) -> Ref<'_, [T]>;

    /// Returns a vector of lumped vertex stiffnesses.
    fn lumped_stiffness(&self) -> Ref<'_, [T]>;

    /// Returns an iterator of epsilons for each constraint.
    fn epsilon_iter(&self) -> Box<dyn Iterator<Item = f64> + '_>;

    /// Returns a mutable iterator of epsilons for each constraint.
    fn epsilon_iter_mut(&mut self) -> Box<dyn Iterator<Item = RefMut<'_, f64>> + '_>;

    /// Computes the global maximum contact violation
    fn contact_violation(&self, x: &[T]) -> ContactViolation;

    /// Returns the number of unknowns for the problem.
    fn num_variables(&self) -> usize;
    /// Constructs the initial point for the problem.
    ///
    /// The returned `Vec` must have `num_variables` elements.
    fn initial_point(&self) -> Vec<T> {
        vec![T::zero(); self.num_variables()]
    }

    /// Checks if the problem is converged.
    ///
    /// This is the stopping condition implementation.
    fn converged(
        &self,
        x_prev: &[T],
        x: &[T],
        r: &[T],
        r_unscaled: &[T],
        merit: f64,
        x_tol: f32,
        r_tol: f32,
        a_tol: f32,
    ) -> bool;

    /// Returns true if we can use the objective as a merit function.
    fn use_obj_merit(&self) -> bool;

    /// Computes a warm start into `x`.
    ///
    /// The given `x` are the unknowns from the previous step, so doing nothing will use
    /// previous step's solution as a warm start.
    fn compute_warm_start(&self, _x: &mut [T]) {}

    /// An objective function defined by the problem.
    ///
    /// This can be an energy or a residual norm. It is used by the newton solver to guide the
    /// root finding method.
    fn objective(&self, x: &[T]) -> T;

    /// The vector function `r` whose roots we want to find.
    ///
    /// `r(x) = 0`.
    fn residual(&self, x: &[T], r: &mut [T], symmetric: bool);

    /// Update internal state according to the given velocity.
    ///
    /// This function automatically decides which state should be updated.
    fn update_state(
        &self,
        x: &[T],
        rebuild_tree: bool,
        explicit_jacobian: bool,
        stash_sliding_basis: bool,
    );

    /// Compute a diagonal preconditioner.
    fn diagonal_preconditioner(&self, v: &[T], precond: &mut [T]);

    /*
     * The sparse Jacobian of `r`.
     */

    /// Returns the sparsity structure of the Jacobian.
    ///
    /// This function returns a pair of owned `Vec`s containing row and column
    /// indices corresponding to the (potentially) "non-zero" entries of the Jacobian.
    /// The returned vectors must have the same size.
    fn jacobian_indices(&self, with_constraints: bool) -> (Vec<usize>, Vec<usize>);

    /// Updates the given set of values corresponding to each non-zero of the
    /// Jacobian as defined by `j_indices`.
    ///
    /// The size of `values` is the same as the size of one of the index vectors
    /// returned by `jacobian_indices`.
    fn jacobian_values(&self, x: &[T], r: &[T], rows: &[usize], cols: &[usize], values: &mut [T]);

    /// Compute the residual and simultaneously its Jacobian product with the given vector `p`.
    fn jacobian_product(&self, x: &[T], p: &[T], r: &[T], jp: &mut [T]);

    /// Invalidates any values that were cached.
    ///
    /// This function must be called whenever the unknown `x` is changed between subsequent calls to `jacobian_product`.
    fn invalidate_cached_jacobian_product_values(&self);
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
    fn residual_timings(&self) -> RefMut<'_, ResidualTimings> {
        self.timings.borrow_mut()
    }
    #[inline]
    fn jacobian_timings(&self) -> RefMut<'_, FrictionJacobianTimings> {
        self.jac_timings.borrow_mut()
    }
    #[inline]
    fn debug_friction(&self) -> Ref<'_, Vec<T>> {
        self.debug_friction.borrow()
    }
    #[inline]
    fn time_step(&self) -> f64 {
        NLProblem::time_step(self)
    }
    #[inline]
    fn num_contacts(&self) -> usize {
        NLProblem::num_contacts(self)
    }
    #[inline]
    fn num_in_proximity(&self) -> usize {
        NLProblem::num_in_proximity(self)
    }
    #[inline]
    fn mesh(&self) -> Mesh {
        NLProblem::mesh(self)
    }
    #[inline]
    fn mesh_with(&self, dq: &[T]) -> Mesh {
        NLProblem::mesh_with(self, dq)
    }

    fn save_contact_jac(&self, i: usize) {
        use std::io::Write;
        for (jaci, fc) in self.frictional_contact_constraints.iter().enumerate() {
            let mut file = std::fs::File::create(&format!("./out/jac/jac_{jaci}_{i}.py")).unwrap();
            let fc_constraint = fc.constraint.borrow();
            let jac = fc_constraint
                .contact_state
                .contact_jacobian
                .as_ref()
                .unwrap()
                .matrix
                .view();
            let constrained_collider_vertices = fc_constraint
                .contact_state
                .constrained_collider_vertices
                .as_slice();
            writeln!(file, "Jrows = [").unwrap();
            for (row_idx, row) in jac.into_iter() {
                let vtx_idx =
                    fc_constraint.collider_vertex_indices[constrained_collider_vertices[row_idx]];
                for (_, _) in row.into_iter() {
                    for i in 0..3 {
                        for _ in 0..3 {
                            write!(file, "{:?}, ", 3 * vtx_idx + i).unwrap();
                        }
                    }
                }
            }
            writeln!(file, "]").unwrap();
            writeln!(file, "Jcols = [").unwrap();
            for (_, row) in jac.into_iter() {
                for (col_idx, _) in row.into_iter() {
                    for _ in 0..3 {
                        for j in 0..3 {
                            write!(
                                file,
                                "{:?}, ",
                                3 * fc_constraint.implicit_surface_vertex_indices[col_idx] + j
                            )
                            .unwrap();
                        }
                    }
                }
            }
            writeln!(file, "]").unwrap();
            writeln!(file, "Jvals = [").unwrap();
            for (_, row) in jac.into_iter() {
                for (_, block) in row.into_iter() {
                    for i in 0..3 {
                        for j in 0..3 {
                            write!(file, "{:?}, ", block.into_arrays()[i][j].to_f64().unwrap())
                                .unwrap();
                        }
                    }
                }
            }
            writeln!(file, "]").unwrap();
            writeln!(file, "nrows = {:?}", self.num_variables()).unwrap();
            writeln!(file, "ncols = {:?}", self.num_variables()).unwrap();
            writeln!(
                file,
                "v = {:?}",
                &self
                    .state
                    .borrow()
                    .vtx
                    .next
                    .vel
                    .storage()
                    .iter()
                    .map(|x| x.to_f64().unwrap())
                    .collect::<Vec<_>>()
            )
            .unwrap();
        }
    }
    #[inline]
    fn lumped_mass_inv(&self) -> Ref<'_, [T]> {
        NLProblem::lumped_mass_inv(self)
    }
    #[inline]
    fn lumped_stiffness(&self) -> Ref<'_, [T]> {
        NLProblem::lumped_stiffness(self)
    }
    #[inline]
    fn epsilon_iter(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        NLProblem::epsilon_iter(self)
    }
    #[inline]
    fn epsilon_iter_mut(&mut self) -> Box<dyn Iterator<Item = RefMut<'_, f64>> + '_> {
        NLProblem::epsilon_iter_mut(self)
    }
    #[inline]
    fn contact_violation(&self, x: &[T]) -> ContactViolation {
        NLProblem::contact_violation(self, x)
    }
    #[inline]
    fn num_variables(&self) -> usize {
        self.state.borrow().dof.storage().len()
    }

    #[inline]
    fn objective(&self, _dq: &[T]) -> T {
        // self.integrate_step(dq);
        // self.state.borrow_mut().update_vertices(dq);
        //
        let implicit_factor = self.time_integration.implicit_factor();
        let explicit_factor = self.time_integration.explicit_factor();
        self.compute_objective(implicit_factor as f64, explicit_factor as f64)
    }

    /// Stopping condition.
    fn converged(
        &self,
        x_prev: &[T],
        x: &[T],
        r: &[T],
        r_unscaled: &[T],
        _merit: f64,
        x_tol: f32,
        r_tol: f32,
        a_tol: f32,
    ) -> bool {
        use tensr::LpNorm;
        let State { vtx, .. } = &*self.state.borrow();
        let mass_inv = vtx.mass_inv.as_slice();
        let rel_mesh_size = vtx.rel_mesh_size.as_slice();

        (r_tol > 0.0 && {
            let r_norm = r.as_tensor().lp_norm(LpNorm::Inf).to_f64().unwrap();
            r_norm < r_tol as f64
        }) || (x_tol > 0.0 && {
            let denom = x.as_tensor().norm() + T::one();

            let dx_norm = num_traits::Float::sqrt(
                x_prev
                    .iter()
                    .zip(x.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<T>(),
            );

            dx_norm < T::from(x_tol).unwrap() * denom
        }) || (a_tol > 0.0 && {
            let h_inv =
                1.0 / (NLProblem::time_step(self) * self.time_integration.implicit_factor() as f64);
            let a_tol = T::from(a_tol).unwrap();
            Chunked3::from_flat(&*r_unscaled)
                .iter()
                .zip(mass_inv.iter())
                .zip(rel_mesh_size.iter())
                .zip(
                    Chunked3::from_flat(&*x_prev)
                        .iter()
                        .zip(Chunked3::from_flat(&*x).iter()),
                )
                .all(|(((&r, &m_inv), &size), (&_v_prev, &v))| {
                    let a_abs = a_tol * T::from(size * h_inv).unwrap();
                    let v = Vector3::from(v);
                    // let v_prev = Vector3::from(v_prev);
                    let r = Vector3::from(r);
                    // let ah = v - v_prev;
                    (r * m_inv).norm() <= num_traits::Float::max(a_tol * v.norm(), a_abs)
                })
        })
    }

    fn use_obj_merit(&self) -> bool {
        self.frictional_contact_constraints.iter().all(|fc| {
            let params = fc.constraint.borrow().params.friction_params;
            if params.is_some() {
                params.lagged
            } else {
                true
            }
        }) && self.project_element_hessians
    }

    // TODO: Figure out a better warm start. FE does not work well.
    fn compute_warm_start(&self, dq: &mut [T]) {
        // Use FE for warm starts.
        eprintln!("before = {:?}", &*dq);

        // Take an explicit step
        //{
        //    let state = &mut *self.state.borrow_mut();
        //    State::be_step(state.step_state(dq), self.time_step());
        //    state.update_vertices(dq);
        //}

        let State {
            vtx, solid, shell, ..
        } = &mut *self.state.borrow_mut();

        // Clear residual vector. Using as a buffer. This shouldn't affect other solves.
        vtx.residual
            .storage_mut()
            .iter_mut()
            .for_each(|x| *x = T::zero());

        let dt = self.time_step();

        self.subtract_force(
            vtx.residual_state().into_storage(),
            solid,
            shell,
            self.frictional_contact_constraints.as_slice(),
            T::from(dt).unwrap(),
            false,
        );

        eprintln!("force = {:?}", vtx.residual.storage());

        vtx.mass_inv
            .iter()
            .zip(vtx.residual.iter())
            .zip(Chunked3::from_flat(&mut *dq).iter_mut())
            .for_each(|((&m_inv, f), v)| {
                *v.as_mut_tensor() -= *f.as_tensor() * m_inv * dt;
            });
        eprintln!("after = {:?}", &*dq);
    }

    fn update_state(
        &self,
        x: &[T],
        rebuild_tree: bool,
        explicit_jacobian: bool,
        stash_sliding_basis: bool,
    ) {
        self.integrate_step(x);
        // self.line_search_ws
        //     .borrow_mut()
        //     .vel
        //     .storage_mut()
        //     .clone_from(self.state.borrow().vtx.next.vel.storage());
        self.state.borrow_mut().update_vertices(x);
        let state = self.state.borrow();
        let pos = state.vtx.next.pos.view();

        for fc in self.frictional_contact_constraints.iter() {
            let mut fc_constraint = fc.constraint.borrow_mut();
            let timings = &mut *self.timings.borrow_mut();
            fc_constraint.update_timings.borrow_mut().clear();
            add_time!(timings.update_state; fc_constraint.update_state_with_rebuild(pos, rebuild_tree));
            add_time!(timings.update_distance_potential; fc_constraint.update_distance_potential() );
            add_time!(timings.update_constraint_gradient; fc_constraint.update_constraint_gradient() );
            add_time!(timings.update_multipliers; fc_constraint.update_multipliers());
            add_time!(timings.update_sliding_basis; fc_constraint.update_sliding_basis(explicit_jacobian, stash_sliding_basis, pos.len()));
            timings.update_constraint_details += *fc_constraint.update_timings.borrow();
        }
    }

    fn residual(&self, _x: &[T], r: &mut [T], symmetric: bool) {
        // eprintln!("dq = {dq:?}");
        let t_begin = Instant::now();
        // self.integrate_step(dq);
        // self.state.borrow_mut().update_vertices(dq);
        self.compute_vertex_residual(symmetric);

        // Transfer residual to degrees of freedom.
        let state = &*self.state.borrow();
        state.dof_residual_from_vertices(r);
        self.timings.borrow_mut().total += Instant::now() - t_begin;
    }

    fn diagonal_preconditioner(&self, v: &[T], precond: &mut [T]) {
        let t_begin = Instant::now();
        match self.preconditioner {
            Preconditioner::IncompleteJacobi => {
                NLProblem::compute_incomplete_jacobi_preconditioner(self, v, precond)
            }
            Preconditioner::ApproximateJacobi => {
                NLProblem::compute_physical_preconditioner(self, v, precond)
            }
            Preconditioner::None => {}
        }
        self.timings.borrow_mut().preconditioner += Instant::now() - t_begin;
    }

    #[inline]
    fn jacobian_indices(&self, with_constraints: bool) -> (Vec<usize>, Vec<usize>) {
        NLProblem::jacobian_indices(self, with_constraints)
    }

    #[inline]
    fn jacobian_values(&self, v: &[T], r: &[T], rows: &[usize], cols: &[usize], vals: &mut [T]) {
        self.jacobian_values_with_constraints(v, r, rows, cols, vals, true);
    }

    #[inline]
    fn jacobian_product(&self, v: &[T], p: &[T], _r: &[T], jp: &mut [T]) {
        self.jacobian_product_constraint_ad(v, p, jp);
        // self.jacobian_product_ad(v, p, jp);
    }

    #[inline]
    fn invalidate_cached_jacobian_product_values(&self) {
        self.jacobian_workspace.borrow_mut().stale = true;
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

impl<T: Real64> NLProblem<T> {
    /// Constructs a clone of this problem with autodiff variables.
    pub fn clone_as_autodiff(&self) -> NLProblem<ad::F1> {
        let Self {
            state,
            state_vertex_indices,
            volume_constraints,
            frictional_contact_constraints,
            gravity,
            time_step,
            iterations,
            initial_residual_error,
            iter_counter,
            max_size,
            max_element_force_scale,
            min_element_force_scale,
            original_mesh,
            candidate_force,
            prev_force,
            jacobian_workspace,
            preconditioner_workspace,
            preconditioner,
            time_integration,
            project_element_hessians,
            // candidate_alphas,
            ..
        } = self.clone();

        let state = state.borrow();
        let state = RefCell::new(state.clone_as_autodiff());
        let frictional_contact_constraints = frictional_contact_constraints
            .iter()
            .map(FrictionalContactConstraint::clone_as_autodiff::<f64>)
            .collect::<Vec<_>>();
        let frictional_contact_constraints_ad = frictional_contact_constraints
            .iter()
            .map(FrictionalContactConstraint::clone_as_autodiff::<ad::F1>)
            .collect::<Vec<_>>();
        let candidate_force = candidate_force.borrow();
        let jacobian_workspace = RefCell::new(JacobianWorkspace {
            rows: jacobian_workspace.borrow().rows.clone(),
            cols: jacobian_workspace.borrow().cols.clone(),
            vals: vec![ad::F1::zero(); jacobian_workspace.borrow().vals.len()],
            jac: DSMatrix::from_triplets_iter(std::iter::empty(), 0, 0),
            mapping: Vec::new(),
            stale: true,
        });
        let preconditioner_workspace = RefCell::new(PreconditionerWorkspace {
            buffer: vec![ad::F1::zero(); preconditioner_workspace.borrow().buffer.len()],
        });
        // let candidate_alphas = candidate_alphas.borrow();
        NLProblem {
            state,
            state_vertex_indices,
            volume_constraints,
            frictional_contact_constraints,
            frictional_contact_constraints_ad,
            gravity,
            time_step,
            iterations,
            initial_residual_error,
            iter_counter,
            max_size,
            max_element_force_scale,
            min_element_force_scale,
            original_mesh,
            candidate_force: RefCell::new(
                candidate_force
                    .iter()
                    .map(|&x| ad::F1::cst(x.to_f64().unwrap()))
                    .collect(),
            ),
            prev_force: prev_force
                .iter()
                .map(|&x| ad::F1::cst(x.to_f64().unwrap()))
                .collect(),
            jacobian_workspace,
            preconditioner_workspace,
            time_integration,
            preconditioner,
            debug_friction: RefCell::new(Vec::new()),
            line_search_ws: RefCell::new(LineSearchWorkspace {
                // pos_cur: Chunked3::default(),
                pos_next: Chunked3::default(),
                vel: Chunked3::default(),
                search_dir: Chunked3::default(),
                f1vtx: Chunked3::default(),
                f2vtx: Chunked3::default(),
                dq: Vec::new(),
            }),
            timings: RefCell::new(ResidualTimings::default()),
            jac_timings: RefCell::new(FrictionJacobianTimings::default()),
            project_element_hessians,
            // candidate_alphas: RefCell::new(candidate_alphas.clone()),
        }
    }

    /// Checks that the given problem has a consistent Jacobian implementation.
    pub(crate) fn check_jacobian(
        &self,
        level: u8,
        x: &[T],
        perturb_initial: bool,
    ) -> Result<(), crate::Error> {
        if level == 0 {
            return Ok(());
        }

        // let objective_result = Ok(());
        let objective_result = self.check_objective_gradient(x, perturb_initial);

        if level == 1 {
            return objective_result;
        }

        log::debug!("Checking Jacobian...");
        use ad::F1 as F;
        // Compute Jacobian
        let jac = {
            let problem_clone = self.clone();
            let n = problem_clone.num_variables();
            let mut x0 = x.to_vec();
            if perturb_initial {
                perturb(&mut x0);
            }

            let (jac_rows, jac_cols) = problem_clone.jacobian_indices(true);

            let mut r = vec![T::zero(); n];
            problem_clone.update_state(&x0, true, true, false);
            problem_clone.residual(&x0, &mut r, false);

            let mut jac_values = vec![T::zero(); jac_rows.len()];
            NonLinearProblem::jacobian_values(
                &problem_clone,
                &x0,
                &r,
                &jac_rows,
                &jac_cols,
                &mut jac_values,
            );

            // Build a dense Jacobian.
            let mut jac = vec![vec![0.0; n]; n];
            for (&row, &col, &val) in zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter()) {
                if row < n && col < n {
                    jac[row][col] += val.to_f64().unwrap();
                }
            }
            jac
        };

        // Check Jacobian and compute autodiff Jacobian.

        let problem = self.clone_as_autodiff();
        let n = problem.num_variables();
        let mut x0: Vec<_> = x.iter().map(|&x| F::cst(x.to_f64().unwrap())).collect();
        if perturb_initial {
            perturb(&mut x0);
        }

        let mut r = vec![F::zero(); n];
        problem.update_state(&x0, true, true, false);
        problem.residual(&x0, &mut r, false);

        let mut jac_ad = vec![vec![0.0; n]; n];

        let mut success = true;
        for col in 0..n {
            // eprintln!("CHECK JAC AUTODIFF WRT {}", col);
            x0[col] = F::var(x0[col]);
            problem.update_state(&x0, false, true, false);
            problem.residual(&x0, &mut r, false);
            let d: Vec<f64> = r.iter().map(|r| r.deriv()).collect();
            let avg_deriv = d.into_tensor().norm();
            for row in 0..n {
                let res = approx::relative_eq!(
                    jac[row][col],
                    r[row].deriv(),
                    max_relative = 1e-6,
                    epsilon = 1e-5 * 1.0_f64.max(avg_deriv)
                );
                jac_ad[row][col] = r[row].deriv();
                if !res {
                    success = false;
                    log::debug!(
                        "({}, {}): {} vs. {}",
                        row,
                        col,
                        jac[row][col],
                        r[row].deriv()
                    );
                }
            }
            x0[col] = F::cst(x0[col]);
        }

        if !success {
            geo::io::save_mesh(&problem.mesh(), "./out/deriv_mesh.vtk")?;
        }
        if !success && n < 15 {
            // Print dense hessian if its small
            log::debug!("Actual:");
            for (row_idx, row) in jac.iter().enumerate() {
                for entry in row.iter().take(row_idx + 1) {
                    log::debug!("{:10.2e}", entry);
                }
                log::debug!("");
            }

            log::debug!("Expected:");
            for (row_idx, row) in jac_ad.iter().enumerate() {
                for entry in row.iter().take(row_idx + 1) {
                    log::debug!("{:10.2e}", entry);
                }
                log::debug!("");
            }

            // {
            //     let vel: Vec<_> = x0.iter().map(|x| T::from(x.value()).unwrap()).collect();
            //     let state = &mut *self.state.borrow_mut();
            //     let step_state = state.step_state(&vel);
            //     // Integrate position.
            //     State::be_step(step_state, self.time_step());
            //
            //     state.update_vertices(&vel);
            // }
            // geo::io::save_mesh(&self.mesh(), "./out/problem.vtk").unwrap();
            //
            // eprintln!("Actual:");
            // for row in 0..n {
            //     for col in 0..n {
            //         eprint!("{:10.2e} ", jac[row][col]);
            //     }
            //     eprintln!("");
            // }
            //
            // eprintln!("Expected:");
            // for row in 0..n {
            //     for col in 0..n {
            //         eprint!("{:10.2e} ", jac_ad[row][col]);
            //     }
            //     eprintln!("");
            // }
        }
        if success {
            log::debug!("No errors during Jacobian check.");
            objective_result
        } else {
            Err(crate::Error::DerivativeCheckFailure)
        }
    }

    // Checks that the given problem has a consistent Jacobian implementation.
    pub(crate) fn check_objective_gradient(
        &self,
        x: &[T],
        perturb_initial: bool,
    ) -> Result<(), crate::Error> {
        log::debug!("Checking Objective Gradient...");
        use ad::F1 as F;
        // Compute Gradient
        let grad: Vec<f64> = {
            let problem_clone = self.clone();
            let n = problem_clone.num_variables();
            let mut x0 = x.to_vec();
            if perturb_initial {
                perturb(&mut x0);
            }

            let mut r = vec![T::zero(); n];
            problem_clone.update_state(&x0, true, false, false);
            problem_clone.residual(&x0, &mut r, true);
            r.iter().map(|&x| x.to_f64().unwrap()).collect()
        };

        let problem = self.clone_as_autodiff();
        let n = problem.num_variables();
        let mut x0: Vec<_> = x.iter().map(|&x| F::cst(x.to_f64().unwrap())).collect();
        if perturb_initial {
            perturb(&mut x0);
        }

        let mut grad_ad = vec![0.0; n];

        // Precompute constraints.
        let mut _r = vec![ad::F1::zero(); n];
        problem.update_state(&x0, true, false, false);
        problem.residual(&x0, &mut _r, true);

        let mut success = true;
        for i in 0..n {
            // eprintln!("CHECK GRAD AUTODIFF WRT {}", i);
            x0[i] = F::var(x0[i]);
            problem.update_state(&x0, true, false, false);
            let obj = problem.objective(&x0);
            let dobj = obj.deriv();
            let res = approx::relative_eq!(grad[i], dobj, max_relative = 1e-5, epsilon = 1e-6,);
            grad_ad[i] = dobj;
            if !res {
                success = false;
                log::debug!("({}): {} vs. {}", i, grad[i], dobj);
            }
            x0[i] = F::cst(x0[i]);
        }

        if !success && n < 15 {
            // Print dense grad if its small
            log::debug!("Actual:");
            for i in 0..n {
                log::debug!("{:10.2e}", grad[i]);
            }

            log::debug!("Expected:");
            for i in 0..n {
                log::debug!("{:10.2e}", grad_ad[i]);
            }
        }
        if success {
            log::debug!("No errors during Gradient check.");
            Ok(())
        } else {
            Err(crate::Error::DerivativeCheckFailure)
        }
    }
}

pub(crate) fn perturb<T: Real>(x: &mut [T]) {
    use rand::distributions::Uniform;
    use rand::prelude::*;
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-0.1, 0.1);
    x.iter_mut()
        .for_each(move |x| *x += T::from(rng.sample(range)).unwrap());
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
        use TimeIntegration::*;
        init_logger();
        for ti in [BE, BDF2, SDIRK2, TRBDF2(0.5)] {
            let mut params = sample_params();
            params.time_integration = ti;
            let mut solver_builder = SolverBuilder::new(params);
            solver_builder
                .set_mesh(Mesh::from(make_one_tet_mesh()))
                .set_materials(vec![solid_material().into()]);
            let problem = solver_builder.build_problem::<f64>().unwrap();
            assert!(problem
                .check_jacobian(2, &problem.initial_point(), true)
                .is_ok());
        }
    }

    #[test]
    fn nl_problem_jacobian_three_tets() {
        use TimeIntegration::*;
        init_logger();
        for ti in [BE, BDF2, SDIRK2, TRBDF2(0.5)] {
            let mut params = sample_params();
            params.time_integration = ti;
            let mut solver_builder = SolverBuilder::new(params);
            solver_builder
                .set_mesh(Mesh::from(make_three_tet_mesh()))
                .set_materials(vec![solid_material().into()]);
            let problem = solver_builder.build_problem::<f64>().unwrap();
            assert!(problem
                .check_jacobian(2, &problem.initial_point(), true)
                .is_ok());
        }
    }

    fn solid_material() -> SolidMaterial {
        SolidMaterial::new(0)
            .with_elasticity(Elasticity::from_lame(
                5.4,
                263.1,
                ElasticityModel::NeoHookean,
            ))
            .with_density(10.0)
            .with_damping(1.0)
    }

    fn sample_params() -> SimParams {
        SimParams {
            gravity: [0.0, -9.81, 0.0],
            time_step: Some(0.01),
            velocity_clear_frequency: 0.0,
            residual_tolerance: 1e-2.into(),
            acceleration_tolerance: None,
            velocity_tolerance: 1e-2.into(),
            max_iterations: 1,
            linsolve: LinearSolver::Direct,
            line_search: LineSearch::default_backtracking(),
            derivative_test: 2,
            time_integration: TimeIntegration::default(),
            preconditioner: Preconditioner::default(),
            contact_iterations: 5,
            log_file: None,
            project_element_hessians: false,
            solver_type: SolverType::Newton,
        }
    }
}
