use std::cell::RefCell;

use ipopt::{self, Number};

use flatk::Entity;
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
use crate::energy::*;
use crate::energy_models::{elasticity::*, gravity::Gravity, inertia::Inertia};
use crate::matrix::*;
use crate::objects::*;
use crate::{PointCloud, TriMesh};

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
    #[inline]
    fn from(sol: ipopt::Solution<'a>) -> Solution {
        let mut mysol = Solution::default();
        mysol.update(sol);
        mysol
    }
}

/// Integrate rotation axis-angle.
/// `k0` is previous axis-angle vector.
///
/// The idea here is taken from https://arxiv.org/pdf/1604.08139.pdf
#[inline]
fn integrate_rotation<T: Real>(k0: Vector3<T>, dw: Vector3<T>) -> Vector3<T> {
    (Quaternion::from_vector(k0) * Quaternion::from_vector(dw)).into_vector()
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
///
/// This enum helps us map from the particular contact constraint to the
/// originating simulation object (shell or solid).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SourceObject {
    /// Solid index along with a bool indicating whether the object uses fixed vertices for
    /// collision.
    Solid(usize, bool),
    Shell(usize),
}

/// A struct that keeps track of which objects are being affected by the contact
/// constraints.
#[derive(Debug)]
pub struct FrictionalContactConstraint {
    pub object_index: SourceObject,
    pub collider_index: SourceObject,
    pub constraint: RefCell<PointContactConstraint>,
}

/// An enum that tags data as static (Fixed) or changing (Variable).
/// `Var` is short for "variability".
#[derive(Copy, Clone, Debug)]
pub enum Var<M, T> {
    Fixed(M),
    Rigid(M, T, Matrix3<T>),
    Variable(M),
}

/// Object/Collider type.
///
/// Same as above but without the mesh.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Tag<T> {
    Fixed,
    Rigid(T, Matrix3<T>),
    Variable,
}

impl<M, T> Var<M, T> {
    #[inline]
    pub fn map<U, F: FnOnce(M) -> U>(self, f: F) -> Var<U, T> {
        match self {
            Var::Fixed(x) => Var::Fixed(f(x)),
            Var::Rigid(x, m, i) => Var::Rigid(f(x), m, i),
            Var::Variable(x) => Var::Variable(f(x)),
        }
    }

    /// Effectively untag the underlying data by converting into the inner type.
    #[inline]
    pub fn untag(self) -> M {
        match self {
            Var::Fixed(t) | Var::Variable(t) | Var::Rigid(t, _, _) => t,
        }
    }
}

impl<M, T: Clone> Var<M, T> {
    #[inline]
    pub fn tag(&self) -> Tag<T> {
        match self {
            Var::Fixed(_) => Tag::Fixed,
            Var::Rigid(_, mass, inertia) => Tag::Rigid(mass.clone(), inertia.clone()),
            Var::Variable(_) => Tag::Variable,
        }
    }
}

/// Index of solid objects in the global array of vertices and dofs.
const SOLIDS_INDEX: usize = 0;
/// Index of shell objects in the global array of vertices and dofs.
const SHELLS_INDEX: usize = 1;

/// The common generic state of a single vertex at some point in time.
///
/// `X` and `V` can be either a single component (x, y or z) like `f64`, a triplet like `[f64; 3]`
/// or a stacked collection of components like `Vec<f64>` depending on the context.
#[derive(Copy, Clone, Debug, PartialEq, Default, Entity)]
pub struct VertexState<X, V> {
    pub pos: X,
    pub vel: V,
}

#[derive(Copy, Clone, Debug, PartialEq, Default, Entity)]
pub struct VertexWorkspace<X, V, G> {
    #[entity]
    pub state: VertexState<X, V>,
    /// Gradient for all meshes for which the generalized coordinates don't coincide
    /// with vertex positions.
    pub grad: G,
}

/// A generic `Vertex` with past and present states.
///
/// A single vertex may have other attributes depending on the context.
/// This struct keeps track of the most fundamental attributes required for animation.
#[derive(Copy, Clone, Debug, PartialEq, Default, Entity)]
pub struct Vertex<X, V> {
    #[entity]
    pub prev: VertexState<X, V>,
    #[entity]
    pub cur: VertexState<X, V>,
}

/// Generalized coordinate `q`, and its time derivative `dq`, which is short for `dq/dt`.
///
/// This can be a vertex position and velocity as in `VertexState` or an axis angle and its rotation
/// differential representation for rigid body motion. Other coordinate representations are possible.
#[derive(Copy, Clone, Debug, PartialEq, Default, Entity)]
pub struct GeneralizedState<Q, D> {
    pub q: Q,
    pub dq: D,
}

/// Generalized coordinates with past and present states.
#[derive(Copy, Clone, Debug, PartialEq, Default, Entity)]
pub struct GeneralizedCoords<X, V> {
    #[entity]
    pub prev: GeneralizedState<X, V>,
    #[entity]
    pub cur: GeneralizedState<X, V>,
}

pub type SurfaceVertexData<D> = Subset<D>;
pub type SurfaceVertexView<'i, D> = SubsetView<'i, D>;

pub type VertexData<D> = Chunked<Chunked<D>>;
pub type VertexView<'i, D> = ChunkedView<'i, ChunkedView<'i, D>>;

pub type GeneralizedData<D> = Chunked<Chunked<D>>;
pub type GeneralizedView<'i, D> = ChunkedView<'i, ChunkedView<'i, D>>;

pub type SurfaceVertexData3<T> = SurfaceVertexData<Chunked3<T>>;
pub type SurfaceVertexView3<'i, T> = SurfaceVertexView<'i, Chunked3<T>>;

pub type VertexData3<D> = VertexData<Chunked3<D>>;
pub type VertexView3<'i, D> = VertexView<'i, Chunked3<D>>;

pub type GeneralizedData3<D> = GeneralizedData<Chunked3<D>>;
pub type GeneralizedView3<'i, D> = GeneralizedView<'i, Chunked3<D>>;

/// Variables and their integrated values precomputed for processing.
///
/// This is a helper struct for `ObjectData`.
#[derive(Clone, Debug)]
pub struct WorkspaceData {
    /// Next state in generalized coordinates.
    ///
    /// Currently all generalized coordinates used fit into triplets, however this may not always
    /// be the case.
    pub dof: GeneralizedData3<GeneralizedState<Vec<f64>, Vec<f64>>>,
    /// Vertex positions, velocities and gradients for all meshes for which the generalized coordinates
    /// don't coincide with vertex positions. These are used to pass concrete
    /// vertex quantities (as opposed to generalized coordinates) to constraint
    /// functions and compute intermediate vertex data.
    pub vtx: VertexData3<VertexWorkspace<Vec<f64>, Vec<f64>, Vec<f64>>>,
}

// TODO: update this doc:
/// Simulation vertex and mesh data.
///
/// Meshes that are not simulated are excluded.
/// The data is chunked into solids/shells, then into a subset of individual
/// meshes, and finally into x,y,z coordinates. Rigid shells have 6 degrees of
/// freedom, 3 for position and 3 for rotation, so a rigid shell will correspond
/// to 6 floats in each of these vectors. This can be more generally
/// interpreted as generalized coordinates.
/// This struct is responsible for mapping between input mesh vertices and
/// generalized coordinates.
#[derive(Clone, Debug)]
pub struct ObjectData {
    /// Generalized coordinates from the previous time step.
    ///
    /// Sometimes referred to as `q` in literature.
    pub dof: GeneralizedData3<GeneralizedCoords<Vec<f64>, Vec<f64>>>,

    /// Vertex positions from the previous time step.
    ///
    /// This contains only those positions and velocities that are not coincident with degrees of
    /// freedom, such as for rigid bodies.
    pub vtx: VertexData3<Vertex<Vec<f64>, Vec<f64>>>,

    /// Workspace data used to precompute variables and their integrated values.
    pub workspace: RefCell<WorkspaceData>,

    /// Tetrahedron mesh representing a soft solid computational domain.
    pub solids: Vec<TetMeshSolid>,
    /// Shell object represented by a triangle mesh.
    pub shells: Vec<TriMeshShell>,
}

impl ObjectData {
    /// Build a `VertexData` struct with a zero entry for each vertex of each mesh.
    pub fn build_vertex_data(&self) -> VertexData<Vec<f64>> {
        let mut mesh_sizes = Vec::new();
        mesh_sizes.extend(self.solids.iter().map(|solid| solid.tetmesh.num_vertices()));
        mesh_sizes.extend(self.shells.iter().map(|shell| shell.trimesh.num_vertices()));
        let out = vec![0.0; mesh_sizes.iter().sum::<usize>()];

        let out = Chunked::from_sizes(mesh_sizes, out);
        let num_solids = self.solids.len();
        let num_shells = self.shells.len();

        Chunked::from_offsets(vec![0, num_solids, num_solids + num_shells], out)
    }

    /// Build a `VertexData3` struct with a zero entry for each vertex of each mesh.
    pub fn build_vertex_data3(&self) -> VertexData3<Vec<f64>> {
        let mut mesh_sizes = Vec::new();
        mesh_sizes.extend(self.solids.iter().map(|solid| solid.tetmesh.num_vertices()));
        mesh_sizes.extend(self.shells.iter().map(|shell| shell.trimesh.num_vertices()));
        let out = Chunked3::from_array_vec(vec![[0.0; 3]; mesh_sizes.iter().sum::<usize>()]);

        let out = Chunked::from_sizes(mesh_sizes, out);
        let num_solids = self.solids.len();
        let num_shells = self.shells.len();

        Chunked::from_offsets(vec![0, num_solids, num_solids + num_shells], out)
    }

    #[inline]
    pub fn next_pos<'x>(
        &'x self,
        x: GeneralizedView3<'x, &'x [f64]>,
        pos: VertexView3<'x, &'x [f64]>,
        src_idx: SourceObject,
    ) -> SurfaceVertexView3<'x, &'x [f64]> {
        // Determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        self.mesh_vertex_subset(x, pos, src_idx)
    }

    #[inline]
    pub fn cur_pos(&self, src_idx: SourceObject) -> SurfaceVertexView3<&[f64]> {
        let ObjectData { dof, vtx, .. } = self;
        self.mesh_vertex_subset(
            dof.view().map_storage(|q| q.cur.q),
            vtx.view().map_storage(|v| v.cur.pos),
            src_idx,
        )
    }

    #[inline]
    pub fn next_vel<'x>(
        &'x self,
        v: GeneralizedView3<'x, &'x [f64]>,
        vel: VertexView3<'x, &'x [f64]>,
        src_idx: SourceObject,
    ) -> SurfaceVertexView3<'x, &'x [f64]> {
        // First determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        self.mesh_vertex_subset(v, vel, src_idx)
    }

    #[inline]
    pub fn prev_vel(&self, src_idx: SourceObject) -> SurfaceVertexView3<&[f64]> {
        let ObjectData { dof, vtx, .. } = self;
        self.mesh_vertex_subset(
            dof.view().map_storage(|q| q.cur.dq),
            vtx.view().map_storage(|v| v.cur.vel),
            src_idx,
        )
    }

    /// Transfer internally stored workspace gradient to the given array of degrees of freedom.
    /// This is a noop when degrees of freedom coincide with vertex velocities.
    pub fn sync_grad(&self, source: SourceObject, grad_x: VertexView3<&mut [f64]>) {
        let mut ws = self.workspace.borrow_mut();
        match source {
            SourceObject::Shell(i) => {
                if let ShellData::Rigid { .. } = self.shells[i].data {
                    let mut grad_dofs = grad_x.isolate(SHELLS_INDEX).isolate(i);
                    let grad_vtx = &mut ws
                        .vtx
                        .view_mut()
                        .map_storage(|vtx| vtx.grad)
                        .isolate(SHELLS_INDEX)
                        .isolate(i);
                    debug_assert_eq!(grad_dofs.len(), 2);
                    let r_iter = self.shells[i]
                        .trimesh
                        .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
                        .expect("Missing rigid body reference positions");
                    for (g_vtx, &r) in grad_vtx.iter_mut().zip(r_iter) {
                        // Transfer gradient from vertices to degrees of freedom.
                        *grad_dofs[0].as_mut_tensor() += *g_vtx.as_tensor();
                        let r = r.into_tensor();
                        *grad_dofs[1].as_mut_tensor() += r.cross((*g_vtx).into_tensor());
                        *g_vtx = [0.0; 3]; // Value moved over, reset it.
                    }
                }
            }
            _ => {} // Noop. Nothing to do since vertices and degrees of freedom are the same.
        }
    }

    pub fn mesh_vertex_subset<'x, D: 'x, Alt>(
        &'x self,
        x: GeneralizedView<'x, D>,
        alt: Alt,
        source: SourceObject,
    ) -> SurfaceVertexView<'x, D>
    where
        D: Set + RemovePrefix,
        std::ops::Range<usize>: IsolateIndex<D, Output = D>,
        Alt: Into<Option<VertexView<'x, D>>>,
    {
        match source {
            SourceObject::Solid(i, with_fixed) => Subset::from_unique_ordered_indices(
                &self.solids[i].surface(with_fixed).indices,
                x.isolate(SOLIDS_INDEX).isolate(i),
            ),
            SourceObject::Shell(i) => {
                // Determine source data.
                let x = match self.shells[i].data {
                    ShellData::Soft { .. } => x,
                    _ => alt.into().unwrap_or(x),
                };

                Subset::all(x.isolate(SHELLS_INDEX).isolate(i))
            }
        }
    }

    #[inline]
    pub fn mesh_surface_vertex_count(&self, source: SourceObject) -> usize {
        match source {
            SourceObject::Solid(i, with_fixed) => self.solids[i].surface(with_fixed).indices.len(),
            SourceObject::Shell(i) => self.shells[i].trimesh.num_vertices(),
        }
    }

    /// A utility function to find the coordinates of the given solid surface vertex index inside the
    /// global array of variables.
    #[inline]
    fn tetmesh_solid_vertex_coordinates(
        &self,
        mesh_index: usize,
        coord: usize,
        with_fixed: bool,
    ) -> usize {
        let surf_vtx_idx = coord / 3;
        let offset = self.dof.view().at(SOLIDS_INDEX).offset_value(mesh_index);
        3 * (offset + self.solids[mesh_index].surface(with_fixed).indices[surf_vtx_idx]) + coord % 3
    }

    /// A utility function to find the coordinates of the given soft shell surface vertex index inside the
    /// global array of variables.
    #[inline]
    fn soft_trimesh_shell_vertex_coordinates(&self, mesh_index: usize, coord: usize) -> usize {
        let surf_vtx_idx = coord / 3;
        let q = self.dof.view().at(1);
        assert!(!q.is_empty()); // Must be at least one shell.
        assert!(!q.at(mesh_index).is_empty()); // Verts should be coincident with dofs.
        let offset = q.offset_value(mesh_index);
        3 * (offset + surf_vtx_idx) + coord % 3
    }

    /// Translate a surface mesh coordinate index into the corresponding
    /// simulation coordinate index in our global array of generalized coordinates.
    ///
    /// If `None` is returned, it means the coordinate belongs to a rigid object vertex.
    #[inline]
    pub fn source_coordinate(&self, index: SourceObject, coord: usize) -> Option<usize> {
        match index {
            SourceObject::Solid(i, with_fixed) => {
                Some(self.tetmesh_solid_vertex_coordinates(i, coord, with_fixed))
            }
            SourceObject::Shell(i) => {
                if let ShellData::Rigid { .. } = self.shells[i].data {
                    return None;
                }
                Some(self.soft_trimesh_shell_vertex_coordinates(i, coord))
            }
        }
    }

    /// Produce an iterator over the given slice of scaled variables.
    #[inline]
    pub(crate) fn scaled_variables_iter<'a>(
        unscaled_var: &'a [Number],
        scale: Number,
    ) -> impl Iterator<Item = Number> + 'a {
        unscaled_var.iter().map(move |&val| val * scale)
    }

    #[inline]
    pub fn update_workspace_velocity(&self, uv: &[Number], scale: f64) {
        let mut ws = self.workspace.borrow_mut();
        let sv = ws.dof.view_mut().into_storage().dq;
        for (output, input) in sv.iter_mut().zip(Self::scaled_variables_iter(uv, scale)) {
            *output = input;
        }
    }

    /// Update vertex positions of non dof vertices.
    ///
    /// This ensures that vertex data queried by constraint functions is current.
    /// This function is only intended to sync pos with cur_x.
    fn sync_pos(
        shells: &[TriMeshShell],
        q: GeneralizedView3<&[f64]>,
        mut pos: VertexView3<&mut [f64]>,
    ) {
        for (i, shell) in shells.iter().enumerate() {
            if let ShellData::Rigid { .. } = shell.data {
                let q = q.isolate(SHELLS_INDEX).isolate(i);
                let mut pos = pos.view_mut().isolate(SHELLS_INDEX).isolate(i);
                debug_assert_eq!(q.len(), 2);
                let translation = Vector3::new(q[0]);
                let rotation = Vector3::new(q[1]);
                // Apply the translation and rotation from cur_x to the shell and save it to pos.
                // Previous values in pos are not used.
                for (out_p, &r) in pos.iter_mut().zip(
                    shell
                        .trimesh
                        .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
                        .expect("Missing rigid body reference positions"),
                ) {
                    *out_p.as_mut_tensor() = rotate(r.into_tensor(), rotation) + translation;
                }
            }
        }
    }

    /// Update vertex velocities of vertices not in q (dofs).
    fn sync_vel(
        shells: &[TriMeshShell],
        dq_next: GeneralizedView3<&[f64]>,
        q_cur: GeneralizedView3<&[f64]>,
        mut vel_next: VertexView3<&mut [f64]>,
    ) {
        for (i, shell) in shells.iter().enumerate() {
            if let ShellData::Rigid { .. } = shell.data {
                let dq_next = dq_next.isolate(SHELLS_INDEX).isolate(i);
                let q_cur = q_cur.isolate(SHELLS_INDEX).isolate(i);
                debug_assert_eq!(dq_next.len(), 2);
                let rotation = Vector3::new(q_cur[1]);
                let linear = Vector3::new(dq_next[0]);
                let angular = Vector3::new(dq_next[1]);
                let mut vel_next = vel_next.view_mut().isolate(SHELLS_INDEX).isolate(i);
                for (out_vel, &r) in vel_next.iter_mut().zip(
                    shell
                        .trimesh
                        .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
                        .expect("Missing rigid body reference positions"),
                ) {
                    *out_vel.as_mut_tensor() =
                        rotate(angular.cross(r.into_tensor()), rotation) + linear;
                }
            }
        }
    }

    // Update `cur_x` using implicit integration with the given velocity `v`.
    fn integrate_step(&self, dt: f64) {
        debug_assert!(dt > 0.0);
        let mut ws = self.workspace.borrow_mut();
        let WorkspaceData { dof, .. } = &mut *ws;

        {
            let mut dof_next = dof.view_mut().into_storage();
            let q_cur = self.dof.view().into_storage().cur.q;
            debug_assert_eq!(q_cur.len(), dof_next.len());

            // In static simulations, velocity is simply displacement.

            // Integrate all (positional) degrees of freedom using standard implicit euler.
            // Note this code includes rigid motion, but we will overwrite those below.
            dof_next
                .iter_mut()
                .zip(q_cur.iter())
                .for_each(|(GeneralizedState { q, dq }, &x0)| *q = dq.mul_add(dt, x0));
        }

        // Integrate rigid motion
        let mut dof_next = dof.view_mut().isolate(SHELLS_INDEX);
        let q_cur = self
            .dof
            .view()
            .isolate(SHELLS_INDEX)
            .map_storage(|dof| dof.cur.q);

        for (shell, q_cur, dof_next) in zip!(self.shells.iter(), q_cur.iter(), dof_next.iter_mut())
        {
            match shell.data {
                ShellData::Rigid { .. } => {
                    // We are only interested in rigid rotations.
                    assert_eq!(q_cur.len(), 2);
                    let dof = dof_next.isolate(SHELLS_INDEX);
                    *dof.q =
                        integrate_rotation(q_cur[1].into_tensor(), *dof.dq.as_mut_tensor() * dt)
                            .into_data();
                }
                _ => {}
            }
        }
    }

    /// Update the solid meshes with the given global array of vertex positions
    /// for all solids. Note that we set velocities only, since the positions will be updated
    /// automatically from the ipopt solution.
    pub fn update_solid_vertices(
        &mut self,
        new_pos: Chunked3<&[f64]>,
        time_step: f64,
    ) -> Result<(), crate::Error> {
        // All solids have dof coincident with vtx so we use dof directly here.
        let mut dof_cur = self
            .dof
            .view_mut()
            .isolate(SOLIDS_INDEX)
            .map_storage(|dof| dof.cur);

        // All solids are simulated, so the input point set must have the same
        // size as our internal vertex set. If these are mismatched, then there
        // was an issue with constructing the solid meshes. This may not
        // necessarily be an error, we are just being conservative here.
        if new_pos.len() != dof_cur.view().data().len() {
            // We got an invalid point cloud
            return Err(crate::Error::SizeMismatch);
        }

        debug_assert!(time_step > 0.0);
        let dt_inv = 1.0 / time_step;

        // Get the tetmesh and pos so we can update the fixed vertices.
        for (solid, mut dof_cur) in zip!(self.solids.iter(), dof_cur.iter_mut()) {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = solid
                .tetmesh
                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let new_pos_iter = source_index_iter.map(|&idx| new_pos.get(idx as usize));

            // Only update fixed vertices, if no such attribute exists, return an error.
            let fixed_iter = solid
                .tetmesh
                .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
            dof_cur
                .iter_mut()
                .zip(new_pos_iter)
                .zip(fixed_iter)
                .filter_map(|(data, &fixed)| if fixed != 0i8 { Some(data) } else { None })
                .for_each(|(GeneralizedState { q, dq }, new_pos)| {
                    // Update the vertices we find in the given `new_pos` collection, not all may
                    // still be there.
                    if let Some(&new_pos) = new_pos {
                        *dq.as_mut_tensor() = (*new_pos.as_tensor() - *(*q).as_tensor()) * dt_inv;
                        //*pos = new_pos; // automatically updated via solve.
                    }
                });
        }
        Ok(())
    }

    /// Update the shell meshes with the given global array of vertex positions
    /// and velocities for all shells.
    pub fn update_shell_vertices(
        &mut self,
        new_pos: Chunked3<&[f64]>,
        time_step: f64,
    ) -> Result<(), crate::Error> {
        // Some shells are simulated on a per vertex level, some are rigid or
        // fixed, so we will update `dof` for the former and `vtx` for the
        // latter.
        let ObjectData { dof, vtx, .. } = self;

        let mut dof_cur = dof.view_mut().isolate(SHELLS_INDEX).map_storage(|q| q.cur);
        let mut vtx_cur = vtx.view_mut().isolate(SHELLS_INDEX).map_storage(|v| v.cur);

        debug_assert!(time_step > 0.0);
        let dt_inv = 1.0 / time_step;

        // Get the trimesh and {dof,vtx}_cur so we can update the fixed vertices.
        for (shell, (mut dof_cur, mut vtx_cur)) in self
            .shells
            .iter()
            .zip(dof_cur.iter_mut().zip(vtx_cur.iter_mut()))
        {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = shell
                .trimesh
                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let new_pos_iter = source_index_iter.map(|&idx| new_pos.get(idx as usize));

            match shell.data {
                ShellData::Soft { .. } => {
                    // Generalized coordinates of Soft shells coincide with vertex coordinates.
                    // Only update fixed vertices, if no such attribute exists, return an error.
                    let fixed_iter = shell
                        .trimesh
                        .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
                    dof_cur
                        .iter_mut()
                        .zip(new_pos_iter)
                        .zip(fixed_iter)
                        .filter_map(|(data, &fixed)| if fixed != 0i8 { Some(data) } else { None })
                        .for_each(|(GeneralizedState { q, dq }, new_pos)| {
                            // It's possible that the new vector of positions is missing some
                            // vertices that were fixed before, so we try to update those we
                            // actually find in the `new_pos` collection.
                            if let Some(&new_pos) = new_pos {
                                *dq.as_mut_tensor() =
                                    (*new_pos.as_tensor() - *(*q).as_tensor()) * dt_inv;
                                //*pos = new_pos; // automatically updated via solve.
                            }
                        });
                }
                ShellData::Rigid { fixed, .. } => {
                    // Rigid bodies can have 0, 1, or 2 fixed vertices.
                    // With 3 fixed vertices these become completely fixed.
                    let source_indices = shell
                        .trimesh
                        .attrib_as_slice::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;

                    match fixed {
                        FixedVerts::Zero => {
                            // For 0 fixed vertices, nothing needs to be done here.
                        }
                        FixedVerts::One(vtx_idx) => {
                            // For 1 fixed vertex, assign the appropriate velocity to that vertex.
                            if let Some(&new_pos) = new_pos.get(source_indices[vtx_idx] as usize) {
                                let mut vtx_cur = vtx_cur.isolate(vtx_idx);
                                *(&mut vtx_cur.vel).as_mut_tensor() =
                                    (new_pos.into_tensor() - *vtx_cur.pos.as_tensor()) * dt_inv;
                                // Position will by updated by the solve automatically.
                            }
                        }
                        FixedVerts::Two(verts) => {
                            // For 2 fixed vertices.
                            // TODO: This may be a bad idea since it may generate infeasible configurations
                            //       quite easily. Resolve this.
                            if let Some(&p0) = new_pos.get(source_indices[verts[0]] as usize) {
                                if let Some(&p1) = new_pos.get(source_indices[verts[1]] as usize) {
                                    let mut v0 = vtx_cur.view_mut().isolate(verts[0]);
                                    *(&mut v0.vel).as_mut_tensor() =
                                        (p0.into_tensor() - *v0.pos.as_tensor()) * dt_inv;
                                    let mut v1 = vtx_cur.view_mut().isolate(verts[1]);
                                    *(&mut v1.vel).as_mut_tensor() =
                                        (p1.into_tensor() - *v1.pos.as_tensor()) * dt_inv;
                                }
                            }
                        }
                    }
                }
                ShellData::Fixed { .. } => {
                    // This mesh is fixed and doesn't obey any physics. Simply
                    // copy the positions and velocities over.
                    vtx_cur.iter_mut().zip(new_pos_iter).for_each(
                        |(VertexState { pos, vel }, new_pos)| {
                            if let Some(&new_pos) = new_pos {
                                *vel.as_mut_tensor() =
                                    (*new_pos.as_tensor() - *(*pos).as_tensor()) * dt_inv;
                                *pos = new_pos; // Not automatically updated since these are not part of the solve.
                            }
                        },
                    );
                }
            }
        }

        // Update current pos and vel as well
        let mut ws = self.workspace.borrow_mut();
        let vtx_state = &mut ws.vtx.storage_mut().state;
        vtx_state.pos.copy_from_slice(&vtx_cur.storage().pos);
        vtx_state.vel.copy_from_slice(&vtx_cur.storage().vel);

        Ok(())
    }

    /// Advance `*.prev` variables to `*.cur`, and those to current workspace variables and
    /// update the referenced meshes.
    pub fn advance(&mut self, and_velocity: bool) {
        let ObjectData {
            dof,
            vtx,
            workspace,
            solids,
            shells,
        } = self;

        let mut ws = workspace.borrow_mut();
        let WorkspaceData {
            dof: dof_next,
            vtx: vtx_next,
            ..
        } = &mut *ws;
        Self::sync_vel(
            &shells,
            dof_next.view().map_storage(|dof| dof.dq),
            dof.view().map_storage(|dof| dof.cur.q),
            vtx_next.view_mut().map_storage(|vtx| vtx.state.vel),
        );
        Self::sync_pos(
            &shells,
            dof_next.view().map_storage(|dof| dof.q),
            vtx_next.view_mut().map_storage(|vtx| vtx.state.pos),
        );

        {
            // Advance positional degrees of freedom
            zip!(
                dof.storage_mut().view_mut().iter_mut(),
                dof_next.storage().view().iter()
            )
            .for_each(|(GeneralizedCoords { prev, cur }, next)| {
                *prev.q = *cur.q;
                *cur.q = *next.q;
            });

            // Advance vertex positions
            zip!(
                vtx.storage_mut().view_mut().iter_mut(),
                vtx_next.storage().view().iter()
            )
            .for_each(|(Vertex { prev, cur }, next)| {
                *prev.pos = *cur.pos;
                *cur.pos = *next.state.pos;
            });

            // Update time derivatives
            if and_velocity {
                // Advance dq/dt
                zip!(
                    dof.storage_mut().view_mut().iter_mut(),
                    dof_next.storage().view().iter()
                )
                .for_each(|(GeneralizedCoords { prev, cur }, next)| {
                    *prev.dq = *cur.dq;
                    *cur.dq = *next.dq;
                });

                // Advance vertex velocities
                zip!(
                    vtx.storage_mut().view_mut().iter_mut(),
                    vtx_next.storage().view().iter()
                )
                .for_each(|(Vertex { prev, cur }, next)| {
                    *prev.vel = *cur.vel;
                    *cur.vel = *next.state.vel;
                });
            } else {
                // Clear velocities. This ensures that any non-zero initial velocities are cleared
                // for subsequent steps.
                // Clear dq/dt
                zip!(dof.storage_mut().view_mut().iter_mut()).for_each(
                    |GeneralizedCoords { prev, cur }| {
                        *prev.dq = *cur.dq;
                        *cur.dq = 0.0;
                    },
                );

                // Clear vertex velocities
                zip!(vtx.storage_mut().view_mut().iter_mut()).for_each(|Vertex { prev, cur }| {
                    *prev.vel = *cur.vel;
                    *cur.vel = 0.0;
                });
            }
        }

        let vtx_cur = vtx.view().map_storage(|vtx| vtx.cur);
        Self::update_simulated_meshes_with(
            solids,
            shells,
            dof.view().map_storage(|dof| dof.cur),
            vtx_cur,
        );
        Self::update_fixed_meshes(shells, vtx_cur);
    }

    pub fn revert_prev_step(&mut self) {
        let ObjectData {
            shells, dof, vtx, ..
        } = self;

        {
            dof.view_mut()
                .into_storage()
                .iter_mut()
                .for_each(|GeneralizedCoords { prev, cur }| {
                    *cur.q = *prev.q;
                    *cur.dq = *prev.dq;
                });
            vtx.view_mut()
                .into_storage()
                .iter_mut()
                .for_each(|Vertex { prev, cur }| {
                    *cur.pos = *prev.pos;
                    *cur.vel = *prev.vel;
                });
        }

        // We don't need to update all meshes in this case, just the interior edge angles since
        // these are actually used in simulation.
        for (i, shell) in shells.iter_mut().enumerate() {
            match shell.data {
                ShellData::Soft { .. } => {
                    shell.update_interior_edge_angles(
                        dof.view()
                            .at(SHELLS_INDEX)
                            .at(i)
                            .map_storage(|dof| dof.cur.q)
                            .into(),
                    );
                }
                _ => {}
            }
        }
    }

    pub fn update_simulated_meshes_with(
        solids: &mut [TetMeshSolid],
        shells: &mut [TriMeshShell],
        dof_next: GeneralizedView3<GeneralizedState<&[f64], &[f64]>>,
        vtx_next: VertexView3<VertexState<&[f64], &[f64]>>,
    ) {
        // Update mesh vertex positions and velocities.
        for (i, solid) in solids.iter_mut().enumerate() {
            let verts = solid.tetmesh.vertex_positions_mut();
            let dof_next = dof_next.at(SOLIDS_INDEX).at(i);
            verts.copy_from_slice(dof_next.map_storage(|dof| dof.q).into());
            solid
                .tetmesh
                .attrib_as_mut_slice::<_, VertexIndex>(VELOCITY_ATTRIB)
                .expect("Missing velocity attribute")
                .copy_from_slice(dof_next.map_storage(|dof| dof.dq).into());
        }
        for (i, shell) in shells.iter_mut().enumerate() {
            match shell.data {
                ShellData::Soft { .. } => {
                    let dof_next = dof_next.at(SHELLS_INDEX).at(i);
                    let verts = shell.trimesh.vertex_positions_mut();
                    verts.copy_from_slice(dof_next.map_storage(|dof| dof.q).into());
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<_, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .copy_from_slice(dof_next.map_storage(|dof| dof.dq).into());
                    shell.update_interior_edge_angles(dof_next.map_storage(|dof| dof.q).into());
                }
                ShellData::Rigid { .. } => {
                    let vtx_next = vtx_next.at(SHELLS_INDEX).at(i);
                    let verts = shell.trimesh.vertex_positions_mut();
                    verts.copy_from_slice(vtx_next.map_storage(|vtx| vtx.pos).into());
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<_, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .copy_from_slice(vtx_next.map_storage(|vtx| vtx.vel).into());
                }
                ShellData::Fixed { .. } => {
                    // Not simulated, nothing to do here.
                    // These meshes are updated only for visualization purposes, which is done in
                    // `update_fixed_meshes`, which doesn't need to happen for instance when
                    // reverting a solve.
                }
            }
        }
    }

    /// We need to update the internal meshes since these are used for output.
    ///
    /// Note that the update for fixed meshes is coming from point clouds, and this function is
    /// needed to copy that data onto the meshes that will be output by this solver even though
    /// they are not simulated.
    pub fn update_fixed_meshes(
        shells: &mut [TriMeshShell],
        vtx_cur: VertexView3<VertexState<&[f64], &[f64]>>,
    ) {
        // Update inferred velocities on fixed meshes
        for (i, shell) in shells.iter_mut().enumerate() {
            match shell.data {
                ShellData::Fixed { .. } => {
                    let vtx_cur = vtx_cur.at(SHELLS_INDEX).at(i);
                    shell
                        .trimesh
                        .vertex_positions_mut()
                        .copy_from_slice(vtx_cur.map_storage(|vtx| vtx.pos).into());
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<_, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .copy_from_slice(vtx_cur.map_storage(|vtx| vtx.vel).into());
                }
                _ => {}
            }
        }
    }

    /// Returns the rigid motion translation and rotation for the given source object if it is rigid.
    ///
    /// If the object is non-rigid, `None` is returned.
    #[inline]
    fn rigid_motion(&self, src: SourceObject) -> Option<[[f64; 3]; 2]> {
        let q_cur = self
            .dof
            .view()
            .map_storage(|dof| dof.cur.q)
            .at(SHELLS_INDEX);
        if let SourceObject::Shell(idx) = src {
            let q_cur = q_cur.at(idx);
            if let ShellData::Rigid { .. } = self.shells[idx].data {
                Some([q_cur[0], q_cur[1]])
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NonLinearProblem {
    /// A model of the current problem. This includes all primal variables and
    /// any additional mesh data required for simulation.
    pub object_data: ObjectData,
    /// One way contact constraints between a pair of objects.
    pub frictional_contacts: Vec<FrictionalContactConstraint>,
    /// Constraint on the total volume.
    pub volume_constraints: Vec<(usize, RefCell<VolumeConstraint>)>,
    /// Gravitational potential energy.
    pub gravity: [f64; 3],
    /// The time step defines the amount of time elapsed between steps (calls to `advance`).
    /// If the time step is zero, objects don't exhibit inertia.
    pub time_step: f64,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<dyn FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Solution data. This is kept around for warm starts.
    pub warm_start: Solution,
    pub initial_residual_error: f64,
    pub iter_counter: RefCell<usize>,

    /// The maximum size (diameter) of a simulated object (deformable or rigid).
    pub max_size: f64,
    /// The scale of the energy gradient intended to be used for rescaling the objective gradient.
    pub force_scale: f64,
}

impl NonLinearProblem {
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

    /// Check if this problem represents a static simulation. In this case
    /// inertia is ignored and velocities are treated as displacements.
    fn is_static(&self) -> bool {
        self.time_step == 0.0
    }

    /// Produce an iterator over the given slice of scaled variables.
    pub fn scaled_variables_iter<'a>(
        &self,
        unscaled_var: &'a [Number],
    ) -> impl Iterator<Item = Number> + 'a {
        ObjectData::scaled_variables_iter(unscaled_var, self.variable_scale())
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
            .map(|fc| fc.constraint.borrow().contact_radius())
            .min_by(|a, b| a.partial_cmp(b).expect("Detected NaN contact radius"))
    }

    /// Save an intermediate state of the solve. This is used for debugging.
    #[allow(dead_code)]
    pub fn save_intermediate(&mut self, uv: &[f64], step: usize) {
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
    pub fn update_current_velocity(&self, uv: &[Number]) {
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

    /// Restore the constraint set.
    pub fn reset_constraint_set(&mut self) -> bool {
        let updated = self.update_constraint_set(None);

        // Add forwarded friction impulse. This is the applied force for the next time step.
        let friction_impulse = self.friction_impulse();
        for (idx, solid) in self.object_data.solids.iter_mut().enumerate() {
            solid
                .tetmesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    "forward_friction",
                    friction_impulse.view().at(0).at(idx).view().into(),
                )
                .ok();
        }
        for (idx, shell) in self.object_data.shells.iter_mut().enumerate() {
            shell
                .trimesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    "forward_friction",
                    friction_impulse.view().at(1).at(idx).view().into(),
                )
                .ok();
        }

        updated
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

    /// Update all stateful constraints with the most recent data.
    /// Return an estimate if any constraints have changed, though this estimate may have false
    /// negatives.
    pub fn update_constraint_set(&mut self, solution: Option<ipopt::Solution>) -> bool {
        let mut changed = false; // Report if anything has changed to the caller.

        let scale = self.variable_scale();
        let time_step = self.time_step();

        let NonLinearProblem {
            ref mut frictional_contacts,
            ref object_data,
            ..
        } = *self;

        // Update positions with the given solution (if any).
        let solution_is_some = solution.is_some();
        if let Some(sol) = solution {
            object_data.update_workspace_velocity(sol.primal_variables, scale);
            object_data.integrate_step(time_step);
            let mut ws = object_data.workspace.borrow_mut();
            let WorkspaceData { dof, vtx, .. } = &mut *ws;
            ObjectData::sync_pos(
                &object_data.shells,
                dof.view().map_storage(|dof| dof.q),
                vtx.view_mut().map_storage(|vtx| vtx.state.pos),
            );
        }

        let ws = object_data.workspace.borrow_mut();

        for FrictionalContactConstraint {
            object_index,
            collider_index,
            constraint,
        } in frictional_contacts.iter_mut()
        {
            if solution_is_some {
                let q = ws.dof.view().map_storage(|dof| dof.q);
                let pos = ws.vtx.view().map_storage(|vtx| vtx.state.pos);
                let object_pos = object_data.next_pos(q, pos, *object_index);
                let collider_pos = object_data.next_pos(q, pos, *collider_index);
                changed |= constraint
                    .borrow_mut()
                    .update_neighbours(object_pos.view(), collider_pos.view());
            } else {
                let object_pos = object_data.cur_pos(*object_index);
                let collider_pos = object_data.cur_pos(*collider_index);
                changed |= constraint
                    .borrow_mut()
                    .update_neighbours(object_pos.view(), collider_pos.view());
            }
        }

        changed
    }

    /// Build a new set of multipliers from the old set and replace warm start multipliers with the
    /// new set.
    pub fn remap_warm_start(&mut self, old_constraint_set: ChunkedView<&[usize]>) {
        use crate::constraints::remap_values;
        let active_constraint_set = self.active_constraint_set();
        let new_values = active_constraint_set.data();
        let old_values = old_constraint_set.data();

        // Remap multipliers
        let new_multipliers = remap_values(
            self.warm_start.constraint_multipliers.iter().cloned(),
            0.0,
            old_values.iter().cloned(),
            new_values.iter().cloned(),
        );
        self.warm_start.constraint_multipliers = new_multipliers;
    }

    pub fn apply_frictional_contact_impulse(&mut self) {
        let NonLinearProblem {
            frictional_contacts,
            object_data,
            ..
        } = self;

        let mut ws = object_data.workspace.borrow_mut();
        let WorkspaceData { dof, vtx, .. } = &mut *ws;

        for fc in frictional_contacts.iter() {
            let [obj_idx, coll_idx] = [fc.object_index, fc.collider_index];
            let fc = fc.constraint.borrow();

            let dq = dof.view_mut().map_storage(|dof| dof.dq);
            let vtx_vel = vtx.view_mut().map_storage(|vtx| vtx.state.vel);
            let mut obj_vel = object_data.mesh_vertex_subset(dq, vtx_vel, obj_idx);
            fc.add_mass_weighted_frictional_contact_impulse_to_object(obj_vel.view_mut());

            let dq = dof.view_mut().map_storage(|dof| dof.dq);
            let vtx_vel = vtx.view_mut().map_storage(|vtx| vtx.state.vel);
            let mut coll_vel = object_data.mesh_vertex_subset(dq, vtx_vel, coll_idx);
            fc.add_mass_weighted_frictional_contact_impulse_to_collider(coll_vel.view_mut());
        }
    }

    /// Commit velocity by advancing the internal state by the given unscaled velocity `uv`.
    /// If `and_velocity` is `false`, then only positions are advanced, and velocities are reset.
    /// This emulates a critically damped, or quasi-static simulation.
    pub fn advance(&mut self, uv: &[f64], and_velocity: bool, and_warm_start: bool) {
        self.update_current_velocity(uv);
        {
            self.apply_frictional_contact_impulse();

            self.object_data.workspace.borrow();
            self.compute_step();
        }

        self.object_data.advance(and_velocity);

        if !and_warm_start {
            self.reset_warm_start();
        }
    }

    /// Advance object data one step back.
    pub fn revert_prev_step(&mut self) {
        self.object_data.revert_prev_step();
        self.reset_warm_start();
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

    /// A convenience function to integrate the given velocity by the internal time step.
    ///
    /// For implicit integration this boils down to a simple multiply by the time step.
    pub fn compute_step(&self) {
        self.object_data.integrate_step(self.time_step());
    }

    /// Compute and return the objective value.
    pub fn objective_value(&self, uv: &[Number]) -> f64 {
        self.update_current_velocity(uv);
        self.compute_step();
        let mut obj = 0.0;

        let ws = self.object_data.workspace.borrow();
        let dof_next = ws.dof.view();
        let dof_cur = self.object_data.dof.view().map_storage(|dof| dof.cur);

        for (i, solid) in self.object_data.solids.iter().enumerate() {
            let dof_cur = dof_cur.at(SOLIDS_INDEX).at(i).into_storage();
            let dof_next = dof_next.at(SOLIDS_INDEX).at(i).into_storage();
            obj += solid.elasticity().energy(dof_cur.q, dof_next.q);
            obj += solid.gravity(self.gravity).energy(dof_cur.q, dof_next.q);
            if !self.is_static() {
                obj += solid.inertia().energy(dof_cur.dq, dof_next.dq);
            }
        }

        for (i, shell) in self.object_data.shells.iter().enumerate() {
            let dof_cur = dof_cur.at(SHELLS_INDEX).at(i).into_storage();
            let dof_next = dof_next.at(SHELLS_INDEX).at(i).into_storage();
            obj += shell.elasticity().energy(dof_cur.q, dof_next.q);
            obj += shell.gravity(self.gravity).energy(dof_cur.q, dof_next.q);
            obj += shell.inertia().energy(dof_cur.dq, dof_next.dq);
        }

        // If time_step is 0.0, this is a pure static solve, which means that
        // there cannot be friction.
        if !self.is_static() {
            let vtx_next = ws.vtx.view();
            for fc in self.frictional_contacts.iter() {
                let fc_constraint = fc.constraint.borrow();
                if let Some(fc_contact) = fc_constraint.frictional_contact.as_ref() {
                    if fc_contact.params.friction_forwarding > 0.0 {
                        let dq_next = dof_next.map_storage(|dof| dof.dq);
                        let vel_next = vtx_next.map_storage(|vtx| vtx.state.vel);
                        let obj_v = self
                            .object_data
                            .next_vel(dq_next, vel_next, fc.object_index);
                        let col_v = self
                            .object_data
                            .next_vel(dq_next, vel_next, fc.collider_index);
                        obj -= fc
                            .constraint
                            .borrow()
                            .frictional_dissipation([obj_v.view(), col_v.view()]);
                    }
                }
            }
        }

        obj
    }

    /// Convert a given array of contact forces to impulses.
    fn contact_impulse_magnitudes(forces: &[f64], scale: f64) -> Vec<f64> {
        forces.iter().map(|&cf| cf * scale).collect()
    }

    #[inline]
    pub fn num_frictional_contacts(&self) -> usize {
        self.frictional_contacts.len()
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
        solution: ipopt::Solution,
        constraint_values: &[f64],
        // TODO: Move to GlobalContactJacobian, which combines these two outputs.
    ) -> (ContactJacobian, Chunked<Offsets<Vec<usize>>>) {
        let NonLinearProblem {
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

        for fc in frictional_contacts.iter() {
            let n = fc.constraint.borrow().constraint_size();
            let constraint_offset = contact_offset + volume_constraints.len();

            // Get the normal component of the contact impulse.
            let contact_impulse = Self::contact_impulse_magnitudes(
                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
                multiplier_impulse_scale,
            );

            log::debug!(
                "Maximum contact impulse: {}",
                crate::inf_norm(contact_impulse.iter().cloned())
            );

            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

            let (_, active_contact_indices, _) = fc
                .constraint
                .borrow()
                .in_contact_indices(&contact_impulse, potential_values);

            let object_vertex_offset = match fc.object_index {
                SourceObject::Solid(i, _) => *surface_vertex_offsets_view.at(SOLIDS_INDEX).at(i),
                SourceObject::Shell(i) => *surface_vertex_offsets_view.at(SHELLS_INDEX).at(i),
            };
            let collider_vertex_offset = match fc.collider_index {
                SourceObject::Solid(i, _) => *surface_vertex_offsets_view.at(SOLIDS_INDEX).at(i),
                SourceObject::Shell(i) => *surface_vertex_offsets_view.at(SHELLS_INDEX).at(i),
            };

            fc.constraint.borrow().append_contact_jacobian_triplets(
                &mut jac_triplets,
                &active_contact_indices,
                contact_offset,
                object_vertex_offset,
                collider_vertex_offset,
            );

            contact_offset += n;
        }

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
        solution: ipopt::Solution,
        constraint_values: &[f64],
        jac: ContactJacobianView,
        surface_vertex_offsets: ChunkedView<Offsets<&[usize]>>,
    ) -> Tensor![f64; S S 3 3] {
        let NonLinearProblem {
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

        for fc in frictional_contacts.iter() {
            let FrictionalContactConstraint {
                object_index,
                collider_index,
                constraint,
            } = fc;

            let constraint = constraint.borrow();

            let n = constraint.constraint_size();
            let constraint_offset = contact_offset + volume_constraints.len();

            // Get the normal component of the contact impulse.
            let contact_impulse = Self::contact_impulse_magnitudes(
                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
                multiplier_impulse_scale,
            );

            log::debug!(
                "Maximum contact impulse: {}",
                crate::inf_norm(contact_impulse.iter().cloned())
            );

            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

            let (_, active_contact_indices, _) =
                constraint.in_contact_indices(&contact_impulse, potential_values);

            let PointContactConstraint {
                implicit_surface: ref surf,
                ref contact_points,
                ref object_mass_data,
                ref collider_mass_data,
                ..
            } = *constraint;

            let start_object = match object_index {
                SourceObject::Solid(i, _) => *surface_vertex_offsets.at(SOLIDS_INDEX).at(i),
                SourceObject::Shell(i) => *surface_vertex_offsets.at(SHELLS_INDEX).at(i),
            };
            let start_collider = match collider_index {
                SourceObject::Solid(i, _) => *surface_vertex_offsets.at(SOLIDS_INDEX).at(i),
                SourceObject::Shell(i) => *surface_vertex_offsets.at(SHELLS_INDEX).at(i),
            };

            let surf_neigh_indices = surf.neighbourhood_vertex_indices();

            match object_mass_data {
                MassData::Sparse(vertex_masses) => {
                    blocks.extend(
                        surf_neigh_indices
                            .iter()
                            .map(|&i| Matrix3::diag(&vertex_masses[i][..]).into_data()),
                    );
                    block_indices.extend(
                        surf_neigh_indices
                            .iter()
                            .map(|&i| (start_object + i, start_object + i)),
                    );
                }
                MassData::Dense(mass, inertia) => {
                    let [translation, rotation] = object_data
                        .rigid_motion(*object_index)
                        .expect("Object type doesn't match precomputed mass data");
                    let eff_mass_inv = TriMeshShell::rigid_effective_mass_inv(
                        *mass,
                        translation.into_tensor(),
                        rotation.into_tensor(),
                        *inertia,
                        Select::new(
                            surf_neigh_indices.as_slice(),
                            Chunked3::from_array_slice(surf.surface_vertex_positions()),
                        ),
                    );

                    blocks.extend(
                        eff_mass_inv
                            .iter()
                            .flat_map(|row| row.into_iter().map(|block| *block.into_arrays())),
                    );
                    block_indices.extend(surf_neigh_indices.iter().flat_map(|&row_idx| {
                        surf_neigh_indices
                            .iter()
                            .map(move |&col_idx| (row_idx, col_idx))
                    }));
                }
                MassData::Zero => {}
            }

            match collider_mass_data {
                MassData::Sparse(vertex_masses) => {
                    blocks.extend(
                        surf_neigh_indices
                            .iter()
                            .map(|&i| Matrix3::diag(&vertex_masses[i][..]).into_data()),
                    );
                    block_indices.extend(
                        surf_neigh_indices
                            .iter()
                            .map(|&i| (start_object + i, start_object + i)),
                    );
                }
                MassData::Dense(mass, inertia) => {
                    let [translation, rotation] = object_data
                        .rigid_motion(fc.collider_index)
                        .expect("Object type doesn't match precomputed mass data");
                    let eff_mass_inv = TriMeshShell::rigid_effective_mass_inv(
                        *mass,
                        translation.into_tensor(),
                        rotation.into_tensor(),
                        *inertia,
                        Select::new(&active_contact_indices, contact_points.view()),
                    );
                    blocks.extend(
                        eff_mass_inv
                            .iter()
                            .flat_map(|row| row.into_iter().map(|block| *block.into_arrays())),
                    );
                    block_indices.extend(active_contact_indices.iter().flat_map(|&row_idx| {
                        active_contact_indices
                            .iter()
                            .map(move |&col_idx| (row_idx, col_idx))
                    }));
                }
                MassData::Zero => {}
            }

            contact_offset += n;
        }

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
        solution: ipopt::Solution,
        constraint_values: &[f64],
        friction_steps: &mut [u32],
    ) -> bool {
        if self.frictional_contacts.is_empty() {
            return true;
        }

        self.update_current_velocity(solution.primal_variables);
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
            self.construct_contact_jacobian(solution.clone(), constraint_values);

        let eff_mass_inv = self.construct_effective_mass_inv(
            solution.clone(),
            constraint_values,
            jac.view(),
            surface_vertex_offsets.view(),
        );

        let NonLinearProblem {
            ref mut frictional_contacts,
            ref volume_constraints,
            ref object_data,
            ..
        } = *self;

        let mut is_finished = true;

        let mut constraint_offset = volume_constraints.len();

        // Update normals

        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
            let obj_cur_pos = object_data.cur_pos(fc.object_index);
            let col_cur_pos = object_data.cur_pos(fc.collider_index);
            let dq_next = dof.view().map_storage(|dof| dof.dq);
            let vtx_vel_next = vtx.view().map_storage(|vtx| vtx.state.vel);
            let obj_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.object_index);
            let col_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.collider_index);

            let n = fc.constraint.borrow().constraint_size();
            let contact_impulse = Self::contact_impulse_magnitudes(
                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
                multiplier_impulse_scale,
            );

            log::debug!(
                "Maximum contact impulse: {}",
                crate::inf_norm(contact_impulse.iter().cloned())
            );
            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

            // TODO: Refactor this low level code out. There needs to be a mechanism to pass rigid
            // motion data to the constraints since rigid bodies have a special effective mass.
            let q_cur = q_cur.at(SHELLS_INDEX);
            let rigid_motion = [
                object_data.rigid_motion(fc.object_index),
                object_data.rigid_motion(fc.collider_index),
            ];
            friction_steps[fc_idx] = fc
                .constraint
                .borrow_mut()
                .update_frictional_contact_impulse(
                    &contact_impulse,
                    [obj_cur_pos.view(), col_cur_pos.view()],
                    [obj_vel.view(), col_vel.view()],
                    rigid_motion,
                    potential_values,
                    friction_steps[fc_idx],
                );

            is_finished &= friction_steps[fc_idx] == 0;
            constraint_offset += n;
        }

        //let normals = self.contact_normals();

        is_finished
    }

    /// Returns true if all friction solves have been completed/converged.
    /// This should be the case if and only if all elements in `friction_steps`
    /// are zero, which makes the return value simply a convenience.
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
        let NonLinearProblem {
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

        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
            let obj_cur_pos = object_data.cur_pos(fc.object_index);
            let col_cur_pos = object_data.cur_pos(fc.collider_index);
            let dq_next = dof.view().map_storage(|dof| dof.dq);
            let vtx_vel_next = vtx.view().map_storage(|vtx| vtx.state.vel);
            let obj_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.object_index);
            let col_vel = object_data.next_vel(dq_next, vtx_vel_next, fc.collider_index);

            let n = fc.constraint.borrow().constraint_size();
            let contact_impulse = Self::contact_impulse_magnitudes(
                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
                multiplier_impulse_scale,
            );

            log::debug!(
                "Maximum contact impulse: {}",
                crate::inf_norm(contact_impulse.iter().cloned())
            );
            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];

            let rigid_motion = [
                object_data.rigid_motion(fc.object_index),
                object_data.rigid_motion(fc.collider_index),
            ];
            friction_steps[fc_idx] = fc
                .constraint
                .borrow_mut()
                .update_frictional_contact_impulse(
                    &contact_impulse,
                    [obj_cur_pos.view(), col_cur_pos.view()],
                    [obj_vel.view(), col_vel.view()],
                    rigid_motion,
                    potential_values,
                    friction_steps[fc_idx],
                );

            is_finished &= friction_steps[fc_idx] == 0;
            constraint_offset += n;
        }

        is_finished
    }

    /// Given a trimesh, compute the strain energy per triangle.
    fn compute_trimesh_strain_energy_attrib(mesh: &mut TriMesh) {
        use geo::ops::ShapeMatrix;
        // Overwrite the "strain_energy" attribute.
        let mut strain = mesh
            .remove_attrib::<FaceIndex>(STRAIN_ENERGY_ATTRIB)
            .unwrap();
        strain
            .direct_iter_mut::<f64>()
            .unwrap()
            .zip(zip!(
                mesh.attrib_iter::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)
                    .unwrap()
                    .map(|&x| f64::from(x)),
                mesh.attrib_iter::<MuType, FaceIndex>(MU_ATTRIB)
                    .unwrap()
                    .map(|&x| f64::from(x)),
                mesh.attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                    .unwrap(),
                mesh.attrib_iter::<RefTriShapeMtxInvType, FaceIndex>(
                    REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                )
                .unwrap(),
                mesh.tri_iter()
            ))
            .for_each(|(strain, (lambda, mu, &vol, &ref_shape_mtx_inv, tri))| {
                let shape_mtx = Matrix2x3::new(tri.shape_matrix());
                *strain =
                    NeoHookeanTriEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu).energy()
            });

        mesh.insert_attrib::<FaceIndex>(STRAIN_ENERGY_ATTRIB, strain)
            .unwrap();
    }

    /// Given a trimesh, compute the elastic forces per vertex, and save it at a vertex attribute.
    fn compute_trimesh_elastic_forces_attrib(trimesh: &mut TriMesh) {
        use geo::ops::ShapeMatrix;
        let mut forces_attrib = trimesh
            .remove_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB)
            .unwrap();

        let mut forces =
            Chunked3::from_array_slice_mut(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

        // Reset forces
        for f in forces.iter_mut() {
            *f = [0.0; 3];
        }

        let grad_iter = zip!(
            trimesh
                .attrib_iter::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh
                .attrib_iter::<MuType, FaceIndex>(MU_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<RefTriShapeMtxInvType, FaceIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                .unwrap(),
            trimesh.tri_iter()
        )
        .map(|(lambda, mu, &vol, &ref_shape_mtx_inv, tri)| {
            let shape_mtx = Matrix2x3::new(tri.shape_matrix());
            NeoHookeanTriEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu)
                .energy_gradient()
        });

        for (grad, face) in grad_iter.zip(trimesh.faces().iter()) {
            for j in 0..3 {
                let f = Vector3::new(forces[face[j]]);
                forces[face[j]] = (f - grad[j]).into();
            }
        }

        // Reinsert forces back into the attrib map
        trimesh
            .insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
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
            .direct_iter_mut::<f64>()
            .unwrap()
            .zip(zip!(
                solid
                    .tetmesh
                    .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                    .unwrap()
                    .map(|&x| f64::from(x)),
                solid
                    .tetmesh
                    .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                    .unwrap()
                    .map(|&x| f64::from(x)),
                solid
                    .tetmesh
                    .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
                solid
                    .tetmesh
                    .attrib_iter::<RefTetShapeMtxInvType, CellIndex>(
                        REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                    )
                    .unwrap(),
                solid.tetmesh.tet_iter()
            ))
            .for_each(|(strain, (lambda, mu, &vol, &ref_shape_mtx_inv, tet))| {
                let shape_mtx = Matrix3::new(tet.shape_matrix());
                *strain =
                    NeoHookeanTetEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu).energy()
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
            Chunked3::from_array_slice_mut(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

        // Reset forces
        for f in forces.iter_mut() {
            *f = [0.0; 3];
        }

        let grad_iter = zip!(
            solid
                .tetmesh
                .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            solid
                .tetmesh
                .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            solid
                .tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            solid
                .tetmesh
                .attrib_iter::<RefTetShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB,)
                .unwrap(),
            solid.tetmesh.tet_iter()
        )
        .map(|(lambda, mu, &vol, &ref_shape_mtx_inv, tet)| {
            let shape_mtx = Matrix3::new(tet.shape_matrix());
            NeoHookeanTetEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu)
                .energy_gradient()
        });

        for (grad, cell) in grad_iter.zip(solid.tetmesh.cells().iter()) {
            for j in 0..4 {
                let f = Vector3::new(forces[cell[j]]);
                forces[cell[j]] = (f - grad[j]).into();
            }
        }

        // Reinsert forces back into the attrib map
        solid
            .tetmesh
            .insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
    }

    pub fn contact_potential(&self) -> VertexData<Vec<f64>> {
        let NonLinearProblem {
            object_data,
            frictional_contacts,
            ..
        } = self;

        let mut pot = object_data.build_vertex_data();

        let mut workspace_pot = Vec::new();

        for fc in frictional_contacts.iter() {
            let obj_x0 = object_data.cur_pos(fc.object_index);
            let coll_x0 = object_data.cur_pos(fc.collider_index);

            let active_constraint_indices = fc.constraint.borrow().active_constraint_indices();
            workspace_pot.clear();
            workspace_pot.resize(active_constraint_indices.len(), 0.0);

            fc.constraint.borrow_mut().constraint(
                [obj_x0.view(), coll_x0.view()],
                workspace_pot.as_mut_slice(),
            );

            let mut pot_view_mut =
                object_data.mesh_vertex_subset(pot.view_mut(), None, fc.collider_index);
            for (&aci, &pot) in active_constraint_indices.iter().zip(workspace_pot.iter()) {
                pot_view_mut[aci] += pot;
            }
        }

        pot
    }

    /// Return the stacked friction corrector impulses: one for each vertex.
    pub fn friction_corrector_impulse(&self) -> VertexData3<Vec<f64>> {
        let NonLinearProblem {
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
        let NonLinearProblem {
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
        let NonLinearProblem {
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
    pub fn contact_impulse(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
        let NonLinearProblem {
            object_data,
            frictional_contacts,
            volume_constraints,
            warm_start,
            ..
        } = self;

        let mut impulse = object_data.build_vertex_data3();

        let mut offset = volume_constraints.len();

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in frictional_contacts.iter() {
            // Get contact force from the warm start.
            let n = fc.constraint.borrow().constraint_size();
            let contact_impulse = Self::contact_impulse_magnitudes(
                &warm_start.constraint_multipliers[offset..offset + n],
                multiplier_impulse_scale,
            );

            offset += n;

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

            let obj_x0 = object_data.cur_pos(fc.object_index);
            let coll_x0 = object_data.cur_pos(fc.collider_index);

            fc.constraint.borrow_mut().add_contact_impulse(
                [obj_x0.view(), coll_x0.view()],
                &contact_impulse,
                [obj_imp.view_mut(), coll_imp.view_mut()],
            );

            let obj_imp_view = obj_imp.view();
            let coll_imp_view = coll_imp.view();

            let mut imp = object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.object_index);
            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp_view.iter()) {
                *imp.as_mut_tensor() += *obj_imp.as_tensor();
            }

            let mut imp =
                object_data.mesh_vertex_subset(impulse.view_mut(), None, fc.collider_index);
            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp_view.iter()) {
                *imp.as_mut_tensor() += *coll_imp.as_tensor();
            }
        }
        assert_eq!(offset, warm_start.constraint_multipliers.len());
        impulse
    }

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
    pub fn update_mesh_data(&mut self) {
        let contact_impulse = self.contact_impulse();
        let friction_impulse = self.friction_impulse();
        let friction_corrector_impulse = self.friction_corrector_impulse();
        let pressure = self.pressure(contact_impulse.view());
        let potential = self.contact_potential();
        let normals = self.collider_normals();
        for (idx, solid) in self.object_data.solids.iter_mut().enumerate() {
            // Write back friction and contact impulses
            solid
                .tetmesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    "friction_corrector",
                    friction_corrector_impulse
                        .view()
                        .at(0)
                        .at(idx)
                        .view()
                        .into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    FRICTION_ATTRIB,
                    friction_impulse
                        .view()
                        .at(SOLIDS_INDEX)
                        .at(idx)
                        .view()
                        .into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<ContactImpulseType, VertexIndex>(
                    CONTACT_ATTRIB,
                    contact_impulse
                        .view()
                        .at(SOLIDS_INDEX)
                        .at(idx)
                        .view()
                        .into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<PotentialType, VertexIndex>(
                    POTENTIAL_ATTRIB,
                    potential.view().at(SOLIDS_INDEX).at(idx).view().into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<PressureType, VertexIndex>(
                    PRESSURE_ATTRIB,
                    pressure.view().at(SOLIDS_INDEX).at(idx).view().into(),
                )
                .ok();

            solid
                .tetmesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    "collider_normals",
                    normals.view().at(SOLIDS_INDEX).at(idx).view().into(),
                )
                .ok();

            // Write back elastic strain energy for visualization.
            Self::compute_strain_energy_attrib(solid);

            // Write back elastic forces on each node.
            Self::compute_elastic_forces_attrib(solid);
        }
        for (idx, shell) in self.object_data.shells.iter_mut().enumerate() {
            // Write back friction and contact impulses
            shell
                .trimesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    "friction_corrector",
                    friction_corrector_impulse
                        .view()
                        .at(1)
                        .at(idx)
                        .view()
                        .into(),
                )
                .ok();
            shell
                .trimesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    FRICTION_ATTRIB,
                    friction_impulse.view().at(1).at(idx).view().into(),
                )
                .ok();
            shell
                .trimesh
                .set_attrib_data::<ContactImpulseType, VertexIndex>(
                    CONTACT_ATTRIB,
                    contact_impulse.view().at(1).at(idx).view().into(),
                )
                .ok();
            shell
                .trimesh
                .set_attrib_data::<PotentialType, VertexIndex>(
                    POTENTIAL_ATTRIB,
                    potential.view().at(1).at(idx).view().into(),
                )
                .ok();
            shell
                .trimesh
                .set_attrib_data::<PressureType, VertexIndex>(
                    PRESSURE_ATTRIB,
                    pressure.view().at(1).at(idx).view().into(),
                )
                .ok();

            shell
                .trimesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    "collider_normals",
                    normals.view().at(1).at(idx).view().into(),
                )
                .ok();

            if let ShellData::Soft { .. } = shell.data {
                // Write back elastic strain energy for visualization.
                Self::compute_trimesh_strain_energy_attrib(&mut shell.trimesh);

                // Write back elastic forces on each node.
                Self::compute_trimesh_elastic_forces_attrib(&mut shell.trimesh);
            }
        }
    }

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
            std::fs::File::create(format!("./out/jac_{}.txt", self.iter_counter.borrow())).unwrap();
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
        let s: &[Number] = Storage::as_slice(&svd.singular_values.data);
        let cond = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        log::debug!("Condition number of jacobian is: {}", cond);
    }

    #[allow(dead_code)]
    fn output_first_solid(
        &self,
        x: VertexView3<&[Number]>,
        name: &str,
    ) -> Result<(), crate::Error> {
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
    pub fn print_hessian_svd(&self, values: &[Number]) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        use na::{base::storage::Storage, DMatrix};

        if values.is_empty() {
            return;
        }

        let mut rows = vec![0; values.len()];
        let mut cols = vec![0; values.len()];
        assert!(self.hessian_indices(&mut rows, &mut cols));

        let mut hess = DMatrix::<f64>::zeros(self.num_variables(), self.num_variables());
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            hess[(row as usize, col as usize)] += v;
        }

        let svd = na::SVD::new(hess, false, false);
        let s: &[Number] = Storage::as_slice(&svd.singular_values.data);
        let cond_hess = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        log::debug!("Condition number of hessian is {}", cond_hess);
    }

    /*
     * End of debugging functions
     */
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for NonLinearProblem {
    fn num_variables(&self) -> usize {
        self.object_data.dof.view().into_storage().len()
    }

    fn bounds(&self, uv_l: &mut [Number], uv_u: &mut [Number]) -> bool {
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        let bound = 2e19;

        uv_l.iter_mut().for_each(|x| *x = -bound);
        uv_u.iter_mut().for_each(|x| *x = bound);

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
                        *l = *uv;
                        *u = *uv;
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
                        *l = *uv;
                        *u = *uv;
                    });
            }
        }
        let uv_l = uv_l.into_storage();
        let uv_u = uv_u.into_storage();
        debug_assert!(uv_l.iter().all(|&x| x.is_finite()) && uv_u.iter().all(|&x| x.is_finite()));
        true
    }

    fn initial_point(&self, uv: &mut [Number]) -> bool {
        uv.copy_from_slice(self.warm_start.primal_variables.as_slice());
        debug_assert!(uv.iter().all(|&uv| uv.is_finite()));
        true
    }

    fn initial_bounds_multipliers(&self, z_l: &mut [Number], z_u: &mut [Number]) -> bool {
        z_l.copy_from_slice(self.warm_start.lower_bound_multipliers.as_slice());
        z_u.copy_from_slice(self.warm_start.upper_bound_multipliers.as_slice());
        debug_assert!(z_l.iter().all(|&z| z.is_finite()) && z_u.iter().all(|&z| z.is_finite()));
        true
    }

    fn objective(&self, uv: &[Number], obj: &mut Number) -> bool {
        *obj = self.objective_value(uv) * self.impulse_inv_scale();
        //debug_assert!(obj.is_finite());
        log::trace!("Objective value = {}", *obj);
        obj.is_finite()
    }

    fn objective_grad(&self, uv: &[Number], grad_f: &mut [Number]) -> bool {
        log::trace!(
            "Unscaled variable norm: {}",
            crate::inf_norm(uv.iter().cloned())
        );
        grad_f.iter_mut().for_each(|x| *x = 0.0); // clear gradient vector

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

            for fc in self.frictional_contacts.iter() {
                // Since add_fricton_impulse is looking for a valid gradient, this
                // must involve only vertices that can change.
                //assert!(match fc.object_index {
                //    SourceObject::Solid(_) => true,
                //    SourceObject::Shell(i) => match self.object_data.shells[i].data {
                //        ShellData::Fixed { .. } => match fc.collider_index {
                //            SourceObject::Solid(_) => true,
                //            SourceObject::Shell(i) => {
                //                match self.object_data.shells[i].data {
                //                    ShellData::Fixed { .. } => false,
                //                    ShellData::Rigid { .. } => true,
                //                    _ => true,
                //                }
                //            }
                //        },
                //        ShellData::Rigid { .. } => true,
                //        _ => true,
                //    },
                //});

                let fc_constraint = fc.constraint.borrow();
                if let Some(fc_contact) = fc_constraint.frictional_contact.as_ref() {
                    // ws.grad is a zero initialized memory slice to which we can write the gradient to.
                    // This may be different than `grad.view_mut()` if the object is rigid and has
                    // different degrees of freedom.
                    let mut ws = self.object_data.workspace.borrow_mut();
                    if fc_contact.params.friction_forwarding > 0.0 {
                        let mut obj_g = self.object_data.mesh_vertex_subset(
                            grad.view_mut(),
                            ws.vtx.view_mut().map_storage(|vtx| vtx.grad),
                            fc.object_index,
                        );
                        fc_constraint.add_friction_impulse_to_object(obj_g.view_mut(), -1.0);

                        let mut coll_g = self.object_data.mesh_vertex_subset(
                            grad.view_mut(),
                            ws.vtx.view_mut().map_storage(|vtx| vtx.grad),
                            fc.collider_index,
                        );
                        fc_constraint.add_friction_impulse_to_collider(coll_g.view_mut(), -1.0);
                    }
                }

                // Update `grad.view_mut()` with the newly computed gradients. This is a noop
                // unless at least one of the objects is rigid.
                self.object_data.sync_grad(fc.object_index, grad.view_mut());
                self.object_data
                    .sync_grad(fc.collider_index, grad.view_mut());
            }
        }

        let grad_f = grad.into_storage();

        let scale = self.variable_scale() * self.impulse_inv_scale();
        grad_f.iter_mut().for_each(|g| *g *= scale);
        log::trace!(
            "Objective gradient norm: {}",
            crate::inf_norm(grad_f.iter().cloned())
        );

        debug_assert!(grad_f.iter().all(|&g| g.is_finite()));
        true
    }
}

impl ipopt::ConstrainedProblem for NonLinearProblem {
    fn num_constraints(&self) -> usize {
        let mut num = 0;
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.borrow().constraint_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.borrow().constraint_size();
            //println!("num constraints  = {:?}", num);
        }
        num
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        let mut num = 0;
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.borrow().constraint_jacobian_size();
        }
        for fc in self.frictional_contacts.iter() {
            let constraint = fc.constraint.borrow();
            if self.is_rigid(fc.object_index) {
                num += constraint.constraint_size() * 6;
            } else {
                num += constraint.object_constraint_jacobian_size();
            }

            if self.is_rigid(fc.collider_index) {
                num += constraint.constraint_size() * 6;
            } else {
                num += constraint.collider_constraint_jacobian_size();
            }
        }
        num
    }

    fn initial_constraint_multipliers(&self, lambda: &mut [Number]) -> bool {
        // TODO: Move this is_empty logic to remapping contacts: The important observation here is
        // that contact set != constraint set because some methods don't enforce contacts via a
        // constraint.
        if !lambda.is_empty() {
            // The constrained points may change between updating the warm start and using it here.
            if lambda.len() != self.warm_start.constraint_multipliers.len() {
                // This is sometimes not caught for some reason, so we ouput an explicit error.
                log::error!(
                    "Number of multipliers ({}) does not match warm start ({})",
                    lambda.len(),
                    self.warm_start.constraint_multipliers.len()
                );
                assert_eq!(lambda.len(), self.warm_start.constraint_multipliers.len());
            }
            lambda.copy_from_slice(self.warm_start.constraint_multipliers.as_slice());
        }

        debug_assert!(lambda.iter().all(|l| l.is_finite()));
        true
    }

    fn constraint(&self, uv: &[Number], g: &mut [Number]) -> bool {
        self.update_current_velocity(uv);
        self.compute_step();

        let mut count = 0; // Constraint counter

        let mut ws = self.object_data.workspace.borrow_mut();
        let WorkspaceData { dof, vtx, .. } = &mut *ws;
        let q_next = dof.view().map_storage(|dof| dof.q);
        ObjectData::sync_pos(
            &self.object_data.shells,
            q_next,
            vtx.view_mut().map_storage(|vtx| vtx.state.pos),
        );

        let q_cur = self.object_data.dof.view().map_storage(|dof| dof.cur.q);

        let q_cur_solid = q_cur.at(SOLIDS_INDEX);
        let q_next_solid = q_next.at(SOLIDS_INDEX);
        let vtx_pos_next = vtx.view().map_storage(|vtx| vtx.state.pos);

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let n = vc.borrow().constraint_size();
            vc.borrow_mut().constraint(
                q_cur_solid.at(*solid_idx).into_storage(),
                q_next_solid.at(*solid_idx).into_storage(),
                &mut g[count..count + n],
            );

            let scale = self.volume_constraint_scale();
            g[count..count + n].iter_mut().for_each(|x| *x *= scale);

            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let n = fc.constraint.borrow().constraint_size();
            let obj_pos = self
                .object_data
                .next_pos(q_next, vtx_pos_next.view(), fc.object_index);
            let coll_pos =
                self.object_data
                    .next_pos(q_next, vtx_pos_next.view(), fc.collider_index);

            fc.constraint
                .borrow_mut()
                .constraint([obj_pos.view(), coll_pos.view()], &mut g[count..count + n]);

            let scale = self.contact_constraint_scale();
            g[count..count + n].iter_mut().for_each(|x| *x *= scale);

            count += n;
        }

        assert_eq!(count, g.len());
        log::trace!("Constraint norm: {}", crate::inf_norm(g.iter().cloned()));

        debug_assert!(g.iter().all(|g| g.is_finite()));
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let mut count = 0; // Constraint counter
        for (_, vc) in self.volume_constraints.iter() {
            let mut bounds = vc.borrow().constraint_bounds();
            let n = vc.borrow().constraint_size();
            g_l[count..count + n].swap_with_slice(&mut bounds.0);
            g_u[count..count + n].swap_with_slice(&mut bounds.1);
            // In theory we need to divide constraint bounds by time step to be consistent with the
            // units of the constraint. However all constraint bounds are either zero or infinite,
            // so this is not actually necessary here.
            //let scale = self.volume_constraint_scale();
            //g_l[count..count + n].iter_mut().for_each(|x| *x *= scale);
            //g_u[count..count + n].iter_mut().for_each(|x| *x *= scale);
            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let mut bounds = fc.constraint.borrow().constraint_bounds();
            let n = fc.constraint.borrow().constraint_size();
            g_l[count..count + n].swap_with_slice(&mut bounds.0);
            g_u[count..count + n].swap_with_slice(&mut bounds.1);

            // In theory we need to divide constraint bounds by time step to be consistent with the
            // units of the constraint. However all constraint bounds are either zero or infinite,
            // so this is not actually necessary here.
            //let scale = self.contact_constraint_scale();
            //g_l[count..count + n].iter_mut().for_each(|x| *x *= scale);
            //g_u[count..count + n].iter_mut().for_each(|x| *x *= scale);

            count += n;
        }

        assert_eq!(count, g_l.len());
        assert_eq!(count, g_u.len());
        debug_assert!(g_l.iter().all(|x| x.is_finite()) && g_u.iter().all(|x| x.is_finite()));
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        // This is used for counting offsets.
        let dof_solid = self.object_data.dof.view().at(SOLIDS_INDEX);

        let mut count = 0; // Constraint counter

        let mut row_offset = 0;
        for (solid_idx, vc) in self.volume_constraints.iter() {
            let vc = vc.borrow();
            let iter = vc.constraint_jacobian_indices_iter().unwrap();
            let col_offset = dof_solid.offset_value(*solid_idx) * 3;
            for MatrixElementIndex { row, col } in iter {
                rows[count] = (row + row_offset) as ipopt::Index;
                cols[count] = (col + col_offset) as ipopt::Index;
                count += 1;
            }
            row_offset += 1;
        }

        // Contact constraints

        // Special handling for rigid Jacobians since these are dense.
        let process_rigid = |src_idx,
                             nrows,
                             row_offset,
                             rows: &mut [ipopt::Index],
                             cols: &mut [ipopt::Index]|
         -> usize {
            let mut count = 0;
            if let SourceObject::Shell(idx) = src_idx {
                if let ShellData::Rigid { .. } = self.object_data.shells[idx].data {
                    let dof_shells = self.object_data.dof.view().at(SHELLS_INDEX);
                    assert_eq!(dof_shells.at(idx).len(), 2); // The trimesh should have 6 degrees of freedom
                    let offset = dof_shells.offset_value(idx);

                    for row in 0..nrows {
                        for i in 0..6 {
                            rows[count] = (row + row_offset) as ipopt::Index;
                            cols[count] = (3 * offset + i) as ipopt::Index;
                            count += 1;
                        }
                    }
                    assert_eq!(count, 6 * nrows);
                }
            }
            count
        };

        for fc in self.frictional_contacts.iter() {
            //use ipopt::BasicProblem;
            let nrows = fc.constraint.borrow().constraint_size();
            //let ncols = self.num_variables();
            //let mut jac = vec![vec![0; nrows]; ncols]; // col major

            count += process_rigid(
                fc.object_index,
                nrows,
                row_offset,
                &mut rows[count..],
                &mut cols[count..],
            );

            let constraint = fc.constraint.borrow();
            for MatrixElementIndex { row, col } in
                constraint.object_constraint_jacobian_indices_iter()
            {
                if let Some(coord) = self.object_data.source_coordinate(fc.object_index, col) {
                    debug_assert!(row < nrows);
                    rows[count] = (row + row_offset) as ipopt::Index;
                    cols[count] = coord as ipopt::Index;
                    debug_assert!(cols[count] < ipopt::BasicProblem::num_variables(self) as i32);
                    //jac[col][row] = 1;
                    count += 1;
                }
            }

            count += process_rigid(
                fc.collider_index,
                nrows,
                row_offset,
                &mut rows[count..],
                &mut cols[count..],
            );

            for MatrixElementIndex { row, col } in
                constraint.collider_constraint_jacobian_indices_iter()
            {
                if let Some(coord) = self.object_data.source_coordinate(fc.collider_index, col) {
                    debug_assert!(row < nrows);
                    rows[count] = (row + row_offset) as ipopt::Index;
                    cols[count] = coord as ipopt::Index;
                    debug_assert!(cols[count] < ipopt::BasicProblem::num_variables(self) as i32);
                    //jac[col][row] = 1;
                    count += 1;
                }
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

        assert_eq!(count, rows.len());
        assert_eq!(count, cols.len());

        true
    }

    fn constraint_jacobian_values(&self, uv: &[Number], vals: &mut [Number]) -> bool {
        self.update_current_velocity(uv);
        self.compute_step();

        let dt = self.time_step();

        let mut count = 0; // Constraint counter

        let mut ws = self.object_data.workspace.borrow_mut();
        let WorkspaceData { dof, vtx, .. } = &mut *ws;
        let q_next = dof.view().map_storage(|dof| dof.q);
        ObjectData::sync_pos(
            &self.object_data.shells,
            q_next,
            vtx.view_mut().map_storage(|vtx| vtx.state.pos),
        );

        let dof_next = dof.view();
        let q_cur = self.object_data.dof.view().map_storage(|dof| dof.cur.q);

        let q_cur_solid = q_cur.at(SOLIDS_INDEX);
        let q_next_solid = dof_next.at(SOLIDS_INDEX).map_storage(|dof| dof.q);

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let n = vc.borrow().constraint_jacobian_size();
            vc.borrow_mut()
                .constraint_jacobian_values(
                    q_cur_solid.at(*solid_idx).into_storage(),
                    q_next_solid.at(*solid_idx).into_storage(),
                    &mut vals[count..count + n],
                )
                .ok();
            let scale = self.volume_constraint_scale();
            vals[count..count + n].iter_mut().for_each(|v| *v *= scale);
            count += n;
        }

        // TODO: Refactor rigid handling:
        //       Either put it all into the constraint itself (probably not good)
        //       Or refactor rigid handling as a remapping of the constraint jacobian to generalized
        //       coordinates.
        //       Because constraints can be linearized it makes sense to bake this in to the constraint
        //       so that jacobians don't need to be recomputed every time.
        // Special handling for rigid Jacobians since these are dense.
        let process_rigid_object = |constraint: &PointContactConstraint,
                                    pos: SurfaceVertexView3<&[f64]>,
                                    x: GeneralizedView3<&[f64]>,
                                    src_idx,
                                    nrows,
                                    scale,
                                    vals: &mut [Number]|
         -> usize {
            let mut count = 0;
            if let SourceObject::Shell(idx) = src_idx {
                if let ShellData::Rigid { .. } = self.object_data.shells[idx].data {
                    vals[..6 * nrows].iter_mut().for_each(|v| *v = 0.0); // initialize vals, since will be adding to it.
                    let cm = x.at(1).at(idx)[0].into_tensor();
                    for (row, col, block) in constraint.object_constraint_jacobian_blocks_iter() {
                        let scaled_block = block.into_tensor() * scale;
                        let linear = &mut vals[6 * row..6 * row + 3];
                        *linear.as_mut_tensor() += scaled_block;

                        let r = pos[col].into_tensor() - cm;
                        //let r = pos[col].into_tensor();
                        let angular = &mut vals[6 * row + 3..6 * (row + 1)];
                        *angular.as_mut_tensor() += r.skew() * scaled_block;
                    }
                    count += 6 * nrows;
                }
            }
            count
        };
        let process_rigid_collider = |constraint: &PointContactConstraint,
                                      x0: GeneralizedView3<&[f64]>,
                                      v: GeneralizedView3<&[f64]>,
                                      src_idx,
                                      nrows,
                                      scale,
                                      vals: &mut [Number]|
         -> usize {
            let mut count = 0;
            if let SourceObject::Shell(idx) = src_idx {
                if let ShellData::Rigid { .. } = self.object_data.shells[idx].data {
                    vals[..6 * nrows].iter_mut().for_each(|v| *v = 0.0); // initialize vals, since will be adding to it.

                    let pos = self.object_data.shells[idx]
                        .trimesh
                        .attrib_as_slice::<RigidRefPosType, VertexIndex>(
                            REFERENCE_VERTEX_POS_ATTRIB,
                        )
                        .unwrap();
                    let x0 = x0.view().at(1).at(idx)[1].into_tensor();
                    let w_new = v.view().at(1).at(idx)[1].into_tensor();
                    let dw = w_new * dt;
                    for (row, col, block) in constraint.collider_constraint_jacobian_blocks_iter() {
                        let scaled_block = block.into_tensor() * scale;
                        let linear = &mut vals[6 * row..6 * row + 3];
                        *linear.as_mut_tensor() += scaled_block;

                        let r = pos[col].into_tensor();
                        let angular = &mut vals[6 * row + 3..6 * (row + 1)];
                        // Compute dR(w)r/dw as per https://arxiv.org/pdf/1312.0788.pdf
                        let rotdw = rotation(dw);
                        let dw2 = dw.norm_squared();
                        let dpdw = if dw2 > 0.0 {
                            (dw * dw.transpose() - dw.skew() * (rotdw - Matrix3::identity()))
                                * (r / dw2).skew()
                                * rotdw.transpose()
                        } else {
                            r.skew()
                        };
                        *angular.as_mut_tensor() += dpdw * rotate(scaled_block, -x0);
                    }
                    count += 6 * nrows;
                }
            }
            count
        };

        let vtx_pos_next = vtx.view().map_storage(|vtx| vtx.state.pos);
        let q_next = dof_next.map_storage(|dof| dof.q);
        for fc in self.frictional_contacts.iter() {
            let nrows = fc.constraint.borrow().constraint_size();
            let scale = self.contact_constraint_scale();

            let obj_pos = self
                .object_data
                .next_pos(q_next, vtx_pos_next, fc.object_index);
            let coll_pos = self
                .object_data
                .next_pos(q_next, vtx_pos_next, fc.collider_index);

            let mut constraint = fc.constraint.borrow_mut();
            if !constraint.is_linear() {
                constraint.update_surface_with_mesh_pos(obj_pos.view());
                constraint.update_contact_points(coll_pos.view());
            }

            let obj_rigid_count = process_rigid_object(
                &*constraint,
                obj_pos.view(),
                dof_next.map_storage(|dof| dof.q),
                fc.object_index,
                nrows,
                scale,
                &mut vals[count..],
            );

            if obj_rigid_count > 0 {
                count += obj_rigid_count;
            } else {
                let n = constraint.object_constraint_jacobian_size();
                for (v, out_v) in constraint
                    .object_constraint_jacobian_values_iter()
                    .zip(vals[count..count + n].iter_mut())
                {
                    *out_v = v * scale;
                }
                count += n;
            }

            let coll_rigid_count = process_rigid_collider(
                &*constraint,
                q_cur,
                dof_next.map_storage(|dof| dof.dq),
                fc.collider_index,
                nrows,
                scale,
                &mut vals[count..],
            );

            if coll_rigid_count > 0 {
                count += coll_rigid_count;
            } else {
                let n = constraint.collider_constraint_jacobian_size();
                for (v, out_v) in constraint
                    .collider_constraint_jacobian_values_iter()
                    .zip(vals[count..count + n].iter_mut())
                {
                    *out_v = v * scale;
                }
                count += n;
            }
        }

        assert_eq!(count, vals.len());

        let scale = self.time_step() * self.variable_scale();
        vals.iter_mut().for_each(|v| *v *= scale);

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));
        //self.print_jacobian_svd(vals);
        log::trace!(
            "Constraint Jacobian norm: {}",
            crate::inf_norm(vals.iter().cloned())
        );

        debug_assert!(vals.iter().all(|x| x.is_finite()));
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        let mut num = 0;
        for solid in self.object_data.solids.iter() {
            num += solid.elasticity().energy_hessian_size()
                + if !self.is_static() {
                    solid.inertia().energy_hessian_size()
                } else {
                    0
                };
        }
        for shell in self.object_data.shells.iter() {
            num += shell.elasticity().energy_hessian_size()
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
            if !self.is_rigid(fc.object_index) {
                num += fc.constraint.borrow().object_constraint_hessian_size();
            }
            if !self.is_rigid(fc.collider_index) {
                num += fc.constraint.borrow().collider_constraint_hessian_size();
            }
        }
        num
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
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
            let elasticity = solid.elasticity();
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
            let elasticity = shell.elasticity();
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
                rows[count] = (row + offset) as ipopt::Index;
                cols[count] = (col + offset) as ipopt::Index;
                count += 1;
            }
        }

        for fc in self.frictional_contacts.iter() {
            let constraint = fc.constraint.borrow();
            for MatrixElementIndex { row, col } in
                constraint.object_constraint_hessian_indices_iter()
            {
                if let Some(row_coord) = self.object_data.source_coordinate(fc.object_index, row) {
                    if let Some(col_coord) =
                        self.object_data.source_coordinate(fc.object_index, col)
                    {
                        rows[count] = row_coord as ipopt::Index;
                        cols[count] = col_coord as ipopt::Index;
                        count += 1;
                    }
                }
            }

            for MatrixElementIndex { row, col } in
                constraint.collider_constraint_hessian_indices_iter()
            {
                if let Some(row_coord) = self.object_data.source_coordinate(fc.collider_index, row)
                {
                    if let Some(col_coord) =
                        self.object_data.source_coordinate(fc.collider_index, col)
                    {
                        rows[count] = row_coord as ipopt::Index;
                        cols[count] = col_coord as ipopt::Index;
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, rows.len());
        assert_eq!(count, cols.len());

        true
    }
    fn hessian_values(
        &self,
        uv: &[Number],
        mut obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        self.update_current_velocity(uv);
        self.compute_step();

        let mut count = 0; // Values counter
        let mut coff = 0; // Constraint offset

        // Correction to make the above hessian wrt velocity instead of displacement.
        let dt = self.time_step();

        // Constraint scaling
        let c_scale = self.variable_scale() * self.variable_scale() * dt * dt;

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
                *v = 0.0;
            }
        }

        // Multiply energy hessian by objective factor and scaling factors.
        obj_factor *= self.variable_scale() * self.variable_scale() * self.impulse_inv_scale();

        for (solid_idx, solid) in self.object_data.solids.iter().enumerate() {
            let dof_cur = dof_cur.at(SOLIDS_INDEX).at(solid_idx).into_storage();
            let dof_next = dof_next.at(SOLIDS_INDEX).at(solid_idx).into_storage();
            let elasticity = solid.elasticity();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_values(
                dof_cur.q,
                dof_next.q,
                dt * dt * obj_factor,
                &mut vals[count..count + n],
            );
            count += n;

            if !self.is_static() {
                let inertia = solid.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_values(
                    dof_cur.dq,
                    dof_next.dq,
                    obj_factor,
                    &mut vals[count..count + n],
                );
                count += n;
            }
        }

        for (shell_idx, shell) in self.object_data.shells.iter().enumerate() {
            let dof_cur = dof_cur.at(SHELLS_INDEX).at(shell_idx).into_storage();
            let dof_next = dof_next.at(SHELLS_INDEX).at(shell_idx).into_storage();
            let elasticity = shell.elasticity();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_values(
                dof_cur.q,
                dof_next.q,
                dt * dt * obj_factor,
                &mut vals[count..count + n],
            );
            count += n;

            if !self.is_static() {
                let inertia = shell.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_values(
                    dof_cur.dq,
                    dof_next.dq,
                    obj_factor,
                    &mut vals[count..count + n],
                );
                count += n;
            }
        }

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let q_cur = dof_cur.at(SOLIDS_INDEX).at(*solid_idx).into_storage().q;
            let q_next = dof_next.at(SOLIDS_INDEX).at(*solid_idx).into_storage().q;
            let nc = vc.borrow().constraint_size();
            let nh = vc.borrow().constraint_hessian_size();
            vc.borrow_mut()
                .constraint_hessian_values(
                    q_cur,
                    q_next,
                    &lambda[coff..coff + nc],
                    c_scale * self.volume_constraint_scale(),
                    &mut vals[count..count + nh],
                )
                .unwrap();

            count += nh;
            coff += nc;
        }

        for fc in self.frictional_contacts.iter() {
            let obj_pos = self.object_data.next_pos(
                dof_next.map_storage(|dof| dof.q),
                vtx.view().map_storage(|vtx| vtx.state.pos),
                fc.object_index,
            );
            let coll_pos = self.object_data.next_pos(
                dof_next.map_storage(|dof| dof.q),
                vtx.view().map_storage(|vtx| vtx.state.pos),
                fc.collider_index,
            );
            let nc = fc.constraint.borrow().constraint_size();

            let mut constraint = fc.constraint.borrow_mut();
            if !constraint.is_linear() {
                constraint.update_surface_with_mesh_pos(obj_pos.view());
                constraint.update_contact_points(coll_pos.view());
            }

            if !self.is_rigid(fc.object_index) {
                let noh = constraint.object_constraint_hessian_size();
                constraint
                    .object_constraint_hessian_values_iter(&lambda[coff..coff + nc])
                    .zip(vals[count..count + noh].iter_mut())
                    .for_each(|(v, out_v)| *out_v = v * c_scale * self.contact_constraint_scale());
                count += noh;
            }

            if !self.is_rigid(fc.collider_index) {
                let nch = constraint.collider_constraint_hessian_size();
                constraint
                    .collider_constraint_hessian_values_iter(&lambda[coff..coff + nc])
                    .zip(vals[count..count + nch].iter_mut())
                    .for_each(|(v, out_v)| *out_v = v * c_scale * self.contact_constraint_scale());
                count += nch;
            }

            coff += nc;
        }

        assert_eq!(count, vals.len());
        assert_eq!(coff, lambda.len());
        //self.print_hessian_svd(vals);

        debug_assert!(vals.iter().all(|x| x.is_finite()));
        true
    }
}
