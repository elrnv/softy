use std::cell::RefCell;

use autodiff as ad;
use flatk::Component;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use num_traits::Zero;
use tensr::*;

use crate::attrib_defines::*;
use crate::objects::*;

/// Integrate rotation axis-angle.
/// `k0` is previous axis-angle vector.
///
/// The idea here is taken from https://arxiv.org/pdf/1604.08139.pdf
#[inline]
fn integrate_rotation<T: Real>(k0: Vector3<T>, dw: Vector3<T>) -> Vector3<T> {
    (Quaternion::from_vector(k0) * Quaternion::from_vector(dw)).into_vector()
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

/// Index of solid objects in the global array of vertices and dofs.
pub const SOLIDS_INDEX: usize = 0;
/// Index of shell objects in the global array of vertices and dofs.
pub const SHELLS_INDEX: usize = 1;

/// The common generic state of a single vertex at some point in time.
///
/// `X` and `V` can be either a single component (x, y or z) like `f64`, a triplet like `[f64; 3]`
/// or a stacked collection of values like `Vec<f64>` depending on the context.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct VertexState<X, V> {
    pub pos: X,
    pub vel: V,
}

#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct VertexWorkspaceComponent<X, V, XAD, VAD, G, L, FC, LAD, FCAD> {
    #[component]
    pub state: VertexState<X, V>,
    #[component]
    pub state_ad: VertexState<XAD, VAD>,
    /// Gradient for all meshes for which the generalized coordinates don't coincide
    /// with vertex positions.
    pub grad: G,
    pub lambda: L,
    pub vfc: FC,
    pub lambda_ad: LAD,
    pub vfc_ad: FCAD,
}

pub type VertexWorkspace<T, F> = VertexWorkspaceComponent<T, T, F, F, T, T, T, F, F>;

#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct GeneralizedWorkspaceComponent<X, V, XAD, VAD, RAD> {
    #[component]
    pub state: GeneralizedState<X, V>,
    #[component]
    pub state_ad: GeneralizedState<XAD, VAD>,
    pub r_ad: RAD,
}

pub type GeneralizedWorkspace<T, F> = GeneralizedWorkspaceComponent<T, T, F, F, F>;

/// A generic `Vertex` with past and present states.
///
/// A single vertex may have other attributes depending on the context.
/// This struct keeps track of the most fundamental attributes required for animation.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct Vertex<X, V> {
    #[component]
    pub prev: VertexState<X, V>,
    #[component]
    pub cur: VertexState<X, V>,
}

/// Generalized coordinate `q`, and its time derivative `dq`, which is short for `dq/dt`.
///
/// This can be a vertex position and velocity as in `VertexState` or an axis angle and its rotation
/// differential representation for rigid body motion. Other coordinate representations are possible.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct GeneralizedState<Q, D> {
    pub q: Q,
    pub dq: D,
}

/// Generalized coordinates with past and present states.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct GeneralizedCoords<X, V> {
    #[component]
    pub prev: GeneralizedState<X, V>,
    #[component]
    pub cur: GeneralizedState<X, V>,
}

// Data for a subset of vertices on the surface of the object.
pub type SurfaceVertexData<D> = Subset<D>;
/// View of the data for a subset of vertices on the surface of the object.
pub type SurfaceVertexView<'i, D> = SubsetView<'i, D>;

/// Per vertex values.
pub type VertexData<D> = Chunked<Chunked<D>>;
/// View of per vertex values.
pub type VertexView<'i, D> = ChunkedView<'i, ChunkedView<'i, D>>;

/// Per generalized coordinate values.
pub type GeneralizedData<D> = Chunked<Chunked<D>>;
/// View of per generalized coordinate values.
pub type GeneralizedView<'i, D> = ChunkedView<'i, ChunkedView<'i, D>>;

// Data for a subset of vertices on the surface of the object in triplets.
pub type SurfaceVertexData3<T> = SurfaceVertexData<Chunked3<T>>;
/// View of the data for a subset of vertices on the surface of the object in triplets.
pub type SurfaceVertexView3<'i, T> = SurfaceVertexView<'i, Chunked3<T>>;

/// Per vertex triplets.
pub type VertexData3<D> = VertexData<Chunked3<D>>;
/// View of per vertex triplets.
pub type VertexView3<'i, D> = VertexView<'i, Chunked3<D>>;

/// Per generalized coordinate triplets.
pub type GeneralizedData3<D> = GeneralizedData<Chunked3<D>>;
/// View of per generalized coordinate triplets.
pub type GeneralizedView3<'i, D> = GeneralizedView<'i, Chunked3<D>>;

/// Variables and their integrated values precomputed for processing.
///
/// This is a helper struct for `ObjectData`.
#[derive(Clone, Debug)]
pub struct WorkspaceData<T, F> {
    /// Next state in generalized coordinates.
    ///
    /// Currently all generalized coordinates fit into triplets, however this may not always
    /// be the case.
    pub dof: GeneralizedData3<GeneralizedWorkspace<Vec<T>, Vec<F>>>,
    /// Vertex positions, velocities and gradients.
    ///
    /// These are stored for all meshes where the generalized coordinates
    /// don't coincide with vertex degrees of freedom. These are used to pass concrete
    /// vertex quantities (as opposed to generalized coordinates) to constraint
    /// functions and compute intermediate vertex data.
    pub vtx: VertexData3<VertexWorkspace<Vec<T>, Vec<F>>>,
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
pub struct State<T, F> {
    /// Generalized coordinates from the previous time step.
    ///
    /// Sometimes referred to as `q` in literature.
    pub dof: GeneralizedData3<GeneralizedCoords<Vec<T>, Vec<T>>>,

    /// Vertex positions from the previous time step.
    ///
    /// This contains only those positions and velocities that are not coincident with degrees of
    /// freedom, such as for rigid bodies.
    pub vtx: VertexData3<Vertex<Vec<T>, Vec<T>>>,

    /// Workspace data used to precompute variables and their integrated values.
    pub workspace: RefCell<WorkspaceData<T, F>>,

    /// Tetrahedron mesh representing a soft solid computational domain.
    pub solids: Vec<TetMeshSolid>,

    /// Shell object represented by a triangle mesh.
    pub shells: Vec<TriMeshShell>,
}

impl<T: Real> State<T, ad::FT<T>> {
    pub fn clone_as_autodiff(&self) -> State<ad::FT<f64>, ad::FT<ad::FT<f64>>> {
        let State {
            dof,
            vtx,
            workspace,
            solids,
            shells,
        } = self;

        let convert = |v: &[T]| {
            v.iter()
                .cloned()
                .map(|x| ad::F::cst(x.to_f64().unwrap()))
                .collect::<Vec<_>>()
        };
        let convert_ad = |v: &[ad::FT<T>]| {
            v.iter()
                .cloned()
                .map(|x| {
                    ad::F::new(
                        ad::F::new(x.value().to_f64().unwrap(), x.deriv().to_f64().unwrap()),
                        ad::F::zero(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let convert_state = |state: &GeneralizedState<&[T], &[T]>| GeneralizedState {
            q: convert(state.q),
            dq: convert(state.dq),
        };
        let convert_coords = |&GeneralizedCoords { ref prev, ref cur }| GeneralizedCoords {
            prev: convert_state(prev),
            cur: convert_state(cur),
        };
        let convert_vtx_state = |state: &VertexState<&[T], &[T]>| VertexState {
            pos: convert(state.pos),
            vel: convert(state.vel),
        };
        let convert_vtx_workspace = |ws: &VertexWorkspace<&[T], &[ad::FT<T>]>| VertexWorkspace {
            state: convert_vtx_state(&ws.state),
            state_ad: VertexState {
                pos: convert_ad(&ws.state_ad.pos),
                vel: convert_ad(&ws.state_ad.vel),
            },
            grad: convert(ws.grad),
            lambda: convert(ws.lambda),
            vfc: convert(ws.vfc),
            lambda_ad: convert_ad(ws.lambda_ad),
            vfc_ad: convert_ad(ws.vfc_ad),
        };
        let convert_dof_workspace =
            |ws: &GeneralizedWorkspace<&[T], &[ad::FT<T>]>| GeneralizedWorkspace {
                state: convert_state(&ws.state),
                state_ad: GeneralizedState {
                    q: convert_ad(ws.state_ad.q),
                    dq: convert_ad(ws.state_ad.dq),
                },
                r_ad: convert_ad(ws.r_ad),
            };
        let convert_vtx_coords = |&Vertex { ref prev, ref cur }| Vertex {
            prev: convert_vtx_state(prev),
            cur: convert_vtx_state(cur),
        };
        let dof_storage = convert_coords(dof.view().storage());
        let vtx_storage = convert_vtx_coords(vtx.view().storage());
        let dof = dof.clone_with_storage(dof_storage);
        let vtx = vtx.clone_with_storage(vtx_storage);
        let ws = workspace.borrow();
        let ws_dof_storage = convert_dof_workspace(ws.dof.view().storage());
        let ws_vtx_storage = convert_vtx_workspace(ws.vtx.view().storage());
        let workspace = RefCell::new(WorkspaceData {
            dof: ws.dof.clone_with_storage(ws_dof_storage),
            vtx: ws.vtx.clone_with_storage(ws_vtx_storage),
        });
        State {
            dof,
            vtx,
            workspace,
            solids: solids.clone(),
            shells: shells.clone(),
        }
    }
}

/// Update vertex positions of non dof vertices.
///
/// This ensures that vertex data queried by constraint functions is current.
/// This function is only intended to sync pos with cur_x.
pub fn sync_pos<S: Real>(
    shells: &[TriMeshShell],
    q: GeneralizedView3<&[S]>,
    mut pos: VertexView3<&mut [S]>,
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
                *out_p.as_mut_tensor() =
                    rotate(r.into_tensor().cast::<S>(), rotation) + translation;
            }
        }
    }
}

/// Update vertex velocities of vertices not in q (dofs).
pub fn sync_vel<S: Real>(
    shells: &[TriMeshShell],
    dq_next: GeneralizedView3<&[S]>,
    q_cur: GeneralizedView3<&[S]>,
    mut vel_next: VertexView3<&mut [S]>,
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
                    rotate(angular.cross(r.into_tensor().cast::<S>()), rotation) + linear;
            }
        }
    }
}

impl<T: Real> State<T, ad::FT<T>> {
    /// Build a `VertexView` struct with the given data.
    pub fn vtx_view<D>(&self, data: D) -> VertexView<D> {
        Chunked {
            chunks: self.vtx.chunks.view(),
            data: Chunked {
                chunks: self.vtx.data.chunks.view(),
                data,
            },
        }
    }
    /// Build a `GeneralizedView` struct with the given data.
    pub fn dof_view<D>(&self, data: D) -> GeneralizedView<D> {
        Chunked {
            chunks: self.dof.chunks.view(),
            data: Chunked {
                chunks: self.dof.data.chunks.view(),
                data,
            },
        }
    }

    /// Build a `VertexData` struct with the given data collection.
    ///
    /// To build a `VertexData3` simply ensure that `data` is a `Chunked3` type.
    pub fn vertex_data_with<D>(&self, data: D) -> VertexData<D> {
        Chunked {
            chunks: self.vtx.chunks.clone(),
            data: Chunked {
                chunks: self.vtx.data.chunks.clone(),
                data,
            },
        }
    }

    /// Build a `VertexData` struct with a zero entry for each vertex of each mesh.
    pub fn build_vertex_data<S: Zero + Copy>(&self) -> VertexData<Vec<S>> {
        let mut mesh_sizes = Vec::new();
        mesh_sizes.extend(self.solids.iter().map(|solid| solid.tetmesh.num_vertices()));
        mesh_sizes.extend(self.shells.iter().map(|shell| shell.trimesh.num_vertices()));
        let out = vec![S::zero(); mesh_sizes.iter().sum::<usize>()];

        let out = Chunked::from_sizes(mesh_sizes, out);
        let num_solids = self.solids.len();
        let num_shells = self.shells.len();

        Chunked::from_offsets(vec![0, num_solids, num_solids + num_shells], out)
    }

    /// Build a `VertexData3` struct with a zero entry for each vertex of each mesh.
    pub fn build_vertex_data3<S: bytemuck::Pod + Zero + Copy>(&self) -> VertexData3<Vec<S>> {
        let mut mesh_sizes = Vec::new();
        mesh_sizes.extend(self.solids.iter().map(|solid| solid.tetmesh.num_vertices()));
        mesh_sizes.extend(self.shells.iter().map(|shell| shell.trimesh.num_vertices()));
        let out = Chunked3::from_array_vec(vec![[S::zero(); 3]; mesh_sizes.iter().sum::<usize>()]);

        let out = Chunked::from_sizes(mesh_sizes, out);
        let num_solids = self.solids.len();
        let num_shells = self.shells.len();

        Chunked::from_offsets(vec![0, num_solids, num_solids + num_shells], out)
    }

    #[inline]
    pub fn next_pos<'x>(
        &'x self,
        x: GeneralizedView3<'x, &'x [T]>,
        pos: VertexView3<'x, &'x [T]>,
        src_idx: SourceObject,
    ) -> SurfaceVertexView3<'x, &'x [T]> {
        // Determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        self.mesh_vertex_subset(x, pos, src_idx)
    }

    #[inline]
    pub fn cur_pos(&self, src_idx: SourceObject) -> SurfaceVertexView3<&[T]> {
        let State { dof, vtx, .. } = self;
        self.mesh_vertex_subset(
            dof.view().map_storage(|q| q.cur.q),
            vtx.view().map_storage(|v| v.cur.pos),
            src_idx,
        )
    }

    #[inline]
    pub fn next_vel<'x>(
        &'x self,
        v: GeneralizedView3<'x, &'x [T]>,
        vel: VertexView3<'x, &'x [T]>,
        src_idx: SourceObject,
    ) -> SurfaceVertexView3<'x, &'x [T]> {
        // First determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        self.mesh_vertex_subset(v, vel, src_idx)
    }

    #[inline]
    pub fn prev_vel(&self, src_idx: SourceObject) -> SurfaceVertexView3<&[T]> {
        let State { dof, vtx, .. } = self;
        self.mesh_vertex_subset(
            dof.view().map_storage(|q| q.cur.dq),
            vtx.view().map_storage(|v| v.cur.vel),
            src_idx,
        )
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
        unscaled_var: &'a [T],
        scale: T,
    ) -> impl Iterator<Item = T> + 'a {
        unscaled_var.iter().map(move |&val| val * scale)
    }

    #[inline]
    pub fn update_workspace_velocity(&self, uv: &[T], scale: f64) {
        let mut ws = self.workspace.borrow_mut();
        let sv = ws.dof.view_mut().into_storage().state.dq;
        for (output, input) in sv
            .iter_mut()
            .zip(Self::scaled_variables_iter(uv, T::from(scale).unwrap()))
        {
            *output = input;
        }
    }

    /// Transfer internally stored workspace gradient to the given array of degrees of freedom.
    /// This is a noop when degrees of freedom coincide with vertex velocities.
    pub fn sync_grad(&self, source: SourceObject, grad_x: VertexView3<&mut [T]>) {
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
                        *grad_dofs[1].as_mut_tensor() +=
                            r.cast::<T>().cross((*g_vtx).into_tensor());
                        *g_vtx = [T::zero(); 3]; // Value moved over, reset it.
                    }
                }
            }
            _ => {} // Noop. Nothing to do since vertices and degrees of freedom are the same.
        }
    }

    /// Add vertex vectors to the given array of degrees of freedom.
    ///
    /// This is a noop when degrees of freedom coincide with vertex velocities.
    /// This function defines how non vertex degrees of freedom are related to
    /// vertex DoFs for velocity vector quantities.
    pub fn transfer_velocity_vtx_to_dof<I: Real, O: Real>(
        &self,
        source: SourceObject,
        vtx: VertexView3<&mut [I]>,
        dof: GeneralizedView3<&mut [O]>,
    ) {
        match source {
            SourceObject::Shell(i) => {
                let mut dof = dof.isolate(SHELLS_INDEX).isolate(i);
                let mut vtx = vtx.isolate(SHELLS_INDEX).isolate(i);
                if let ShellData::Rigid { .. } = self.shells[i].data {
                    debug_assert_eq!(dof.len(), 2);
                    let r_iter = self.shells[i]
                        .trimesh
                        .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
                        .expect("Missing rigid body reference positions");
                    for (vtx, &r) in vtx.iter_mut().zip(r_iter) {
                        // Transfer gradient from vertices to degrees of freedom.
                        *dof[0].as_mut_tensor() += vtx.as_tensor().cast::<O>();
                        let r = r.into_tensor();
                        *dof[1].as_mut_tensor() +=
                            r.cast::<O>().cross((*vtx).into_tensor().cast::<O>());
                        // Transfer complete, zero out input.
                        *vtx = [I::zero(); 3];
                    }
                }
            }
            _ => {}
        }
    }

    // Update `cur_x` using backward Euler integration with the given velocity `v`.
    pub fn integrate_step(&self, dt: f64) {
        debug_assert!(dt > 0.0);
        let mut ws = self.workspace.borrow_mut();
        let WorkspaceData { dof, .. } = &mut *ws;

        {
            let mut dof_next = dof.view_mut().into_storage().state;
            let q_cur = self.dof.view().into_storage().cur.q;
            debug_assert_eq!(q_cur.len(), dof_next.len());

            // In static simulations, velocity is simply displacement.

            // Integrate all (positional) degrees of freedom using standard implicit euler.
            // Note this code includes rigid motion, but we will overwrite those below.
            dof_next
                .iter_mut()
                .zip(q_cur.iter())
                .for_each(|(GeneralizedState { q, dq }, &x0)| {
                    *q = dq.mul_add(T::from(dt).unwrap(), x0)
                });
        }

        // Integrate rigid rotation.
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
                    let dof = dof_next.isolate(1).state;
                    *dof.q = integrate_rotation(
                        q_cur[1].into_tensor(),
                        *dof.dq.as_mut_tensor() * T::from(dt).unwrap(),
                    )
                    .into_data();
                }
                _ => {}
            }
        }
    }

    /// Take a backward Euler step computed in `q_next`.
    pub fn be_step<S: Real>(&self, v: &[S], dt: f64, q_cur: &[T], q_next: &mut [S]) {
        debug_assert!(dt > 0.0);

        // Source of topology information.
        let dof = self.dof.view();

        // In static simulations, velocity is simply displacement.

        // Integrate all (positional) degrees of freedom using standard implicit Euler.
        // Note this code includes rigid motion, but we will overwrite those below.
        zip!(q_next.iter_mut(), q_cur.iter(), v.iter()).for_each(|(q_next, &q_cur, v_next)| {
            *q_next = v_next.mul_add(S::from(dt).unwrap(), S::from(q_cur).unwrap())
        });

        // Integrate rigid rotation.
        let mut q_next = dof.isolate(SHELLS_INDEX).map_storage(|_| q_next);
        let v_next = dof.isolate(SHELLS_INDEX).map_storage(|_| v);
        let q_cur = dof.isolate(SHELLS_INDEX).map_storage(|_| q_cur);

        for (shell, q_cur, v_next, q_next) in zip!(
            self.shells.iter(),
            q_cur.iter(),
            v_next.iter(),
            q_next.iter_mut()
        ) {
            match shell.data {
                ShellData::Rigid { .. } => {
                    // We are only interested in rigid rotations.
                    assert_eq!(q_cur.len(), 2);
                    let q_next = q_next.isolate(1);
                    let v_next = v_next.isolate(1);
                    *q_next = integrate_rotation(
                        q_cur[1].into_tensor().cast::<S>(),
                        *v_next.as_tensor() * S::from(dt).unwrap(),
                    )
                    .into_data();
                }
                _ => {}
            }
        }
    }

    /// Take a blended step in `q` between current and previous `v` values.
    ///
    /// This type of step is used in trapezoidal rule, implicit Newmark
    /// integration and other SDIRK variants.
    ///
    /// `q_next = q_cur + h*((1-alpha)*v_next + alpha*v_cur)`
    pub fn lerp_step(&self, v: &[T], dt: f64, alpha: f64) {
        debug_assert!(dt > 0.0);

        let dt = T::from(dt).unwrap();
        let alpha = T::from(alpha).unwrap();

        let dof = self.dof.view();

        let GeneralizedState {
            q: q_cur,
            dq: v_cur,
        } = dof.storage().cur;

        let mut ws = self.workspace.borrow_mut();
        let q_next = ws.dof.view_mut().into_storage().state.q;

        // In static simulations, velocity is simply displacement.

        // Note this code includes rigid rotation, but we will overwrite that below.
        zip!(q_next.iter_mut(), q_cur.iter(), v_cur.iter(), v.iter()).for_each(
            |(q_next, &q_cur, &v_cur, &v_next)| {
                // `q_next = q_cur + h*((1-alpha)*v_next + alpha*v_cur)`
                *q_next = v_next
                    .mul_add(T::one() - alpha, v_cur * alpha)
                    .mul_add(dt, q_cur);
            },
        );

        // Integrate rigid rotation.
        let mut dof_next = ws.dof.view_mut().isolate(SHELLS_INDEX);
        let dof_cur = dof.isolate(SHELLS_INDEX).map_storage(|dof| dof.cur);

        for (shell, dof_cur, dof_next) in
            zip!(self.shells.iter(), dof_cur.iter(), dof_next.iter_mut())
        {
            match shell.data {
                ShellData::Rigid { .. } => {
                    assert_eq!(dof_cur.len(), 2);

                    // We are only interested in rigid rotations.
                    // Translations have been integrated correctly above.
                    let cur = dof_cur.isolate(1);

                    let dof = dof_next.isolate(SHELLS_INDEX);
                    let v_next = dof.state.dq.as_tensor();
                    *dof.state.q = (Quaternion::from_vector(*cur.q.as_tensor())
                        * Quaternion::from_vector(*cur.dq.as_tensor() * dt * alpha)
                        * Quaternion::from_vector(*v_next * dt * (T::one() - alpha)))
                    .into_vector()
                    .into_data()
                }
                _ => {}
            }
        }
    }

    /// Take a `gamma` parametrized BDF2 step in `q` computed in the state workspace.
    ///
    /// Use `gamma = 1/2` for vanilla BDF2. Other values of `gamma` are useful for BDF2 variants like TR-BDF2.
    pub fn bdf2_step(&self, v: &[T], dt: f64, gamma: f64) {
        debug_assert!(dt > 0.0);

        let dt = T::from(dt).unwrap();
        let gamma = T::from(gamma).unwrap();

        let dof = self.dof.view();

        let q_cur = dof.storage().cur.q;
        let q_prev = dof.storage().prev.q;

        let mut ws = self.workspace.borrow_mut();
        let q_next = ws.dof.view_mut().into_storage().state.q;

        // Compute coefficients
        let _1 = T::one();
        let _2 = T::from(2.0).unwrap();
        let a = _1 / (gamma * (_2 - gamma));
        let b = (_1 - gamma) * (_1 - gamma) / (gamma * (_2 - gamma));
        let c = (_1 - gamma) / (_2 - gamma);

        // Integrate all (positional) degrees of freedom using standard implicit euler.
        // Note this code includes rigid motion, but we will overwrite those below.
        zip!(q_next.iter_mut(), q_prev.iter(), q_cur.iter(), v.iter()).for_each(
            |(q_next, &q_prev, &q_cur, &v_next)| {
                *q_next = v_next * c * dt + q_cur * a - q_prev * b;
            },
        );

        // Integrate rigid rotation.
        let mut dof_next = ws.dof.view_mut().isolate(SHELLS_INDEX);
        let q_cur = dof.isolate(SHELLS_INDEX).map_storage(|dof| dof.cur.q);
        let q_prev = dof.isolate(SHELLS_INDEX).map_storage(|dof| dof.prev.q);

        for (shell, q_prev, q_cur, dof_next) in zip!(
            self.shells.iter(),
            q_prev.iter(),
            q_cur.iter(),
            dof_next.iter_mut()
        ) {
            match shell.data {
                ShellData::Rigid { .. } => {
                    // We are only interested in rigid rotations.
                    assert_eq!(q_cur.len(), 2);
                    assert_eq!(q_prev.len(), 2);
                    let dof = dof_next.isolate(SHELLS_INDEX);
                    let v_next = dof.state.dq.as_tensor();
                    *dof.state.q = (Quaternion::from_vector(*q_cur[1].as_tensor() * a)
                        * Quaternion::from_vector(-*q_prev[1].as_tensor() * b)
                        * Quaternion::from_vector(*v_next * dt * c))
                    .into_vector()
                    .into_data()
                }
                _ => {}
            }
        }
    }

    /// Updates the solid meshes with the given global array of vertex positions
    /// for all solids.
    ///
    /// Set velocities only, since the positions will be updated
    /// automatically from the solution.
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
        let dt_inv = T::one() / T::from(time_step).unwrap();

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
                        *dq.as_mut_tensor() =
                            (new_pos.as_tensor().cast::<T>() - *(*q).as_tensor()) * dt_inv;
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
        let State { dof, vtx, .. } = self;

        let mut dof_cur = dof.view_mut().isolate(SHELLS_INDEX).map_storage(|q| q.cur);
        let mut vtx_cur = vtx.view_mut().isolate(SHELLS_INDEX).map_storage(|v| v.cur);

        debug_assert!(time_step > 0.0);
        let dt_inv = T::one() / T::from(time_step).unwrap();

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
                                    (new_pos.as_tensor().cast::<T>() - *(*q).as_tensor()) * dt_inv;
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
                                    (new_pos.into_tensor().cast::<T>() - *vtx_cur.pos.as_tensor())
                                        * dt_inv;
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
                                    *(&mut v0.vel).as_mut_tensor() = (p0.into_tensor().cast::<T>()
                                        - *v0.pos.as_tensor())
                                        * dt_inv;
                                    let mut v1 = vtx_cur.view_mut().isolate(verts[1]);
                                    *(&mut v1.vel).as_mut_tensor() = (p1.into_tensor().cast::<T>()
                                        - *v1.pos.as_tensor())
                                        * dt_inv;
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
                                let new_pos_t = new_pos.as_tensor().cast::<T>();
                                *vel.as_mut_tensor() = (new_pos_t - *(*pos).as_tensor()) * dt_inv;
                                *pos = new_pos_t.into(); // Not automatically updated since these are not part of the solve.
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
    ///
    /// it's assumed that the vtx and dof quantities are already synced.
    pub fn advance(&mut self, and_velocity: bool) {
        let State {
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
        //Self::sync_vel(
        //    &shells,
        //    dof_next.view().map_storage(|dof| dof.dq),
        //    dof.view().map_storage(|dof| dof.cur.q),
        //    vtx_next.view_mut().map_storage(|vtx| vtx.state.vel),
        //);
        //Self::sync_pos(
        //    &shells,
        //    dof_next.view().map_storage(|dof| dof.q),
        //    vtx_next.view_mut().map_storage(|vtx| vtx.state.pos),
        //);

        {
            // Advance positional degrees of freedom
            zip!(
                dof.storage_mut().view_mut().iter_mut(),
                dof_next.storage().view().iter()
            )
            .for_each(|(GeneralizedCoords { prev, cur }, next)| {
                *prev.q = *cur.q;
                *cur.q = *next.state.q;
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
                    *cur.dq = *next.state.dq;
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
                        *cur.dq = T::zero();
                    },
                );

                // Clear vertex velocities
                zip!(vtx.storage_mut().view_mut().iter_mut()).for_each(|Vertex { prev, cur }| {
                    *prev.vel = *cur.vel;
                    *cur.vel = T::zero();
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
        let State {
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
        dof_next: GeneralizedView3<GeneralizedState<&[T], &[T]>>,
        vtx_next: VertexView3<VertexState<&[T], &[T]>>,
    ) {
        // Update mesh vertex positions and velocities.
        for (i, solid) in solids.iter_mut().enumerate() {
            let dof_next = dof_next.at(SOLIDS_INDEX).at(i);
            solid
                .tetmesh
                .vertex_positions_mut()
                .iter_mut()
                .zip(dof_next.map_storage(|x| x.q))
                .for_each(|(out, new)| *out.as_mut_tensor() = new.as_tensor().cast::<f64>());
            solid
                .tetmesh
                .attrib_as_mut_slice::<VelType, VertexIndex>(VELOCITY_ATTRIB)
                .expect("Missing velocity attribute")
                .iter_mut()
                .zip(dof_next.map_storage(|x| x.dq))
                .for_each(|(out, new)| *out.as_mut_tensor() = new.as_tensor().cast::<f64>());
        }
        for (i, shell) in shells.iter_mut().enumerate() {
            match shell.data {
                ShellData::Soft { .. } => {
                    let dof_next = dof_next.at(SHELLS_INDEX).at(i);
                    shell
                        .trimesh
                        .vertex_positions_mut()
                        .iter_mut()
                        .zip(dof_next.map_storage(|x| x.q))
                        .for_each(|(out, new)| {
                            *out.as_mut_tensor() = new.as_tensor().cast::<f64>()
                        });
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<VelType, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .iter_mut()
                        .zip(dof_next.map_storage(|x| x.dq))
                        .for_each(|(out, new)| {
                            *out.as_mut_tensor() = new.as_tensor().cast::<f64>()
                        });
                    shell.update_interior_edge_angles(dof_next.map_storage(|dof| dof.q).into());
                }
                ShellData::Rigid { .. } => {
                    let vtx_next = vtx_next.at(SHELLS_INDEX).at(i);
                    shell
                        .trimesh
                        .vertex_positions_mut()
                        .iter_mut()
                        .zip(vtx_next.map_storage(|x| x.pos))
                        .for_each(|(out, new)| {
                            *out.as_mut_tensor() = new.as_tensor().cast::<f64>()
                        });
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<VelType, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .iter_mut()
                        .zip(vtx_next.map_storage(|x| x.vel))
                        .for_each(|(out, new)| {
                            *out.as_mut_tensor() = new.as_tensor().cast::<f64>()
                        });
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
        vtx_cur: VertexView3<VertexState<&[T], &[T]>>,
    ) {
        // Update inferred velocities on fixed meshes
        for (i, shell) in shells.iter_mut().enumerate() {
            match shell.data {
                ShellData::Fixed { .. } => {
                    let vtx_cur = vtx_cur.at(SHELLS_INDEX).at(i);
                    shell
                        .trimesh
                        .vertex_positions_mut()
                        .iter_mut()
                        .zip(vtx_cur.map_storage(|x| x.pos))
                        .for_each(|(out, new)| {
                            *out.as_mut_tensor() = new.as_tensor().cast::<f64>()
                        });
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<VelType, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .iter_mut()
                        .zip(vtx_cur.map_storage(|x| x.vel))
                        .for_each(|(out, new)| {
                            *out.as_mut_tensor() = new.as_tensor().cast::<f64>()
                        });
                }
                _ => {}
            }
        }
    }

    /// Returns the rigid motion translation and rotation for the given source object if it is rigid.
    ///
    /// If the object is non-rigid, `None` is returned.
    #[inline]
    pub fn rigid_motion(&self, src: SourceObject) -> Option<[[T; 3]; 2]> {
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

    /// Returns the total number of degrees of freedom.
    #[inline]
    pub fn num_dofs(&self) -> usize {
        self.dof.storage().len()
    }
}
