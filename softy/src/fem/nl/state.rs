use autodiff as ad;
use flatk::Component;
use geo::attrib::Attrib;
use geo::mesh::{topology::*, VertexPositions};
use num_traits::Zero;
use tensr::*;

use crate::attrib_defines::*;
use crate::objects::tetsolid::*;
use crate::objects::trishell::*;
use crate::objects::Material;
use crate::Error;
use crate::Mesh;
use crate::Real;

/// Index for the part of dofs corresponding to vertex degrees of freedom.
pub const VERTEX_DOFS: usize = 0;
pub const RIGID_DOFS: usize = 1;

/// Integrate rotation axis-angle.
/// `k0` is previous axis-angle vector.
///
/// The idea here is taken from https://arxiv.org/pdf/1604.08139.pdf
#[inline]
fn integrate_rotation<T: Real>(k0: Vector3<T>, dw: Vector3<T>) -> Vector3<T> {
    (Quaternion::from_vector(k0) * Quaternion::from_vector(dw)).into_vector()
}

/// The common generic state of a single particle at some point in time.
/// This can be a vertex or the translational components of a rigid body.
///
/// `X` and `V` can be either a single component (x, y or z) like `f64`, a triplet like `[f64; 3]`
/// or a stacked collection of values like `Vec<f64>` depending on the context.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct ParticleState<X, V> {
    pub pos: X,
    pub vel: V,
}

impl<X: Real, V: Real> ParticleState<Chunked3<&mut [X]>, Chunked3<&mut [V]>> {
    /// Updates vertex positions from given degrees of freedom.
    ///
    /// This ensures that vertex data queried by constraint functions is current.
    pub fn update<S: Real>(&mut self, dof: Chunked<GeneralizedState<&[S], &[S]>>) {
        let vtx_dofs = dof.isolate(VERTEX_DOFS);
        for (ParticleState { pos, vel }, GeneralizedState { q, dq }) in
            self.iter_mut().zip(vtx_dofs)
        {
            *pos.as_mut_tensor() = q.as_tensor().cast::<X>();
            *vel.as_mut_tensor() = dq.as_tensor().cast::<V>();
        }
        //rigid.sync_pos(q, pos);
        //if let ShellData::Rigid { .. } = shell.data {
        //    let q = q.isolate(SHELLS_INDEX).isolate(i);
        //    let mut pos = pos.view_mut().isolate(SHELLS_INDEX).isolate(i);
        //    debug_assert_eq!(q.len(), 2);
        //    let translation = Vector3::new(q[0]);
        //    let rotation = Vector3::new(q[1]);
        //    // Apply the translation and rotation from cur_x to the shell and save it to pos.
        //    // Previous values in pos are not used.
        //    for (out_p, &r) in pos.iter_mut().zip(
        //        shell
        //            .trimesh
        //            .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
        //            .expect("Missing rigid body reference positions"),
        //    ) {
        //        *out_p.as_mut_tensor() =
        //            rotate(r.into_tensor().cast::<S>(), rotation) + translation;
        //    }
        //}
    }
}

/// Generalized coordinate `q`, and its time derivative `dq`, which is short for `dq/dt`.
///
/// This can be a vertex position and velocity as in `VertexState` or an axis angle and its
/// rotation differential representation for rigid body motion. Other coordinate representations
/// are possible.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct GeneralizedState<Q, D> {
    pub q: Q,
    pub dq: D,
}

impl<Q: Real, D: Real> GeneralizedState<Vec<Q>, Vec<D>> {
    /// Constructs a new generalized state from the given set of objects.
    ///
    /// Assume that vertices were previously sorted such that all the free
    /// vertices are at the beginning.
    pub fn new_chunked<X: Real, V: Real>(
        solid: &TetSolid,
        shell: &TriShell,
        // rigid: &RigidBody,
        vtx_pos: &[X],
        vtx_vel: &[V],
        vertex_type: &[VertexType],
    ) -> Chunked<Self> /* Chunked<Vec<Index>> */ {
        // All the free vertices are considered as degrees of freedom.
        // We sorted the vertices such that free vertices come first.

        let num_verts = vtx_pos.len();
        let num_free_verts = vertex_type
            .binary_search(&VertexType::Free.next_variant())
            .unwrap_or(num_verts);

        let num_vtx_dofs = num_free_verts * 3;

        let q = vtx_pos
            .iter()
            .take(num_vtx_dofs)
            .map(|&x| Q::from(x).unwrap())
            //.chain(rigid.translation())
            //.chain(rigid.orientation())
            .collect();
        let dq = vtx_vel
            .iter()
            .take(num_vtx_dofs)
            .map(|&v| D::from(v).unwrap())
            //.chain(rigid.translation())
            //.chain(rigid.orientation())
            .collect();

        Chunked::from_offsets(vec![0, num_vtx_dofs], GeneralizedState { q, dq })
    }
}

///// Orientation and angular velocity state of a rigid object or an oriented particle.
//#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
//pub struct RotationalState<X, V> {
//    pub orientation: X,
//    pub angular_velocity: V,
//}

//#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
//pub struct RigidState<X, V, R, W> {
//    pub linear: ParticleState<X, V>,
//    /// Orientation and angular velocity state.
//    pub angular: RotationalState<R, W>,
//}

#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct VertexWorkspaceComponent<X, V, XAD, VAD, G, L, FC, LAD, FCAD, M, I, VT> {
    #[component]
    pub state: ParticleState<X, V>,
    #[component]
    pub state_ad: ParticleState<XAD, VAD>,
    /// Gradient for all meshes for which the generalized coordinates don't coincide
    /// with vertex positions.
    pub grad: G,
    pub lambda: L,
    pub vfc: FC,
    pub lambda_ad: LAD,
    pub vfc_ad: FCAD,
    pub mass_inv: M,
    pub orig_index: I,
    pub vertex_type: VT,
}

pub type VertexWorkspace<V, VF, S, I, VT> =
    VertexWorkspaceComponent<V, V, VF, VF, V, V, V, VF, VF, S, I, VT>;

///// A generic `Particle` with past and present states.
/////
///// A single vertex may have other attributes depending on the context.
///// This struct keeps track of the most fundamental attributes required for animation.
//#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
//pub struct Particle<X, V> {
//    #[component]
//    pub prev: ParticleState<X, V>,
//    #[component]
//    pub cur: ParticleState<X, V>,
//}

/// Generalized coordinates with past and present states.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct GeneralizedCoords<X, V, XAD, VAD, RAD> {
    #[component]
    pub prev: GeneralizedState<X, V>,
    #[component]
    pub cur: GeneralizedState<X, V>,
    pub next_q: X,
    #[component]
    pub next_ad: GeneralizedState<XAD, VAD>,
    pub r_ad: RAD,
}

impl<Q: Real, D: Real, QAD: Real, DAD: Real, RAD: Real>
    GeneralizedCoords<Vec<Q>, Vec<D>, Vec<QAD>, Vec<DAD>, Vec<RAD>>
{
    /// Constructs a new generalized state from the given set of objects.
    pub fn new_chunked<T: Real>(
        solid: &TetSolid,
        shell: &TriShell,
        vtx_pos: &[T],
        vtx_vel: &[T],
        vertex_types: &[VertexType],
    ) -> Chunked<Self> {
        let (offsets, cur) =
            GeneralizedState::new_chunked(solid, shell, vtx_pos, vtx_vel, vertex_types)
                .into_inner();
        let prev = cur.clone();
        let next_q = cur.q.clone();
        let next_ad = GeneralizedState {
            q: cur.q.iter().map(|&x| QAD::from(x).unwrap()).collect(),
            dq: cur.dq.iter().map(|&x| DAD::from(x).unwrap()).collect(),
        };
        Chunked::from_offsets(
            offsets.into_inner(),
            GeneralizedCoords {
                prev,
                cur,
                next_q,
                next_ad,
                r_ad: Vec::new(),
            },
        )
    }

    /// Advance all degrees of freedom forward using the given velocities
    /// `dq_next` and `next_q` stored within.
    pub fn advance(&mut self, dq_next: &[D]) {
        dq_next.iter().zip(self.iter_mut()).for_each(
            |(
                dq_next,
                GeneralizedCoords {
                    prev, cur, next_q, ..
                },
            )| {
                *prev.q = *cur.q;
                *cur.q = *next_q;
                *prev.dq = *cur.dq;
                *cur.dq = *dq_next;
            },
        );
    }

    /// Retreat all degrees of freedom forward.
    ///
    /// This is the opposite of `advance`.
    pub fn retreat(&mut self) {
        self.iter_mut().for_each(
            |GeneralizedCoords {
                 prev, cur, next_q, ..
             }| {
                *next_q = *cur.q;
                *cur.q = *prev.q;
                *cur.dq = *prev.dq;
                // TODO: Do we need to keep around a prev prev step?
            },
        );
    }

    pub fn step_state_ad(
        &mut self,
    ) -> StepStateComponent<&[Q], &[Q], &[D], &mut [QAD], &[DAD], &mut [RAD]> {
        StepStateComponent {
            prev_q: self.prev.q.view(),
            cur: GeneralizedState {
                q: self.cur.q.view(),
                dq: self.cur.dq.view(),
            },
            next: GeneralizedState {
                q: self.next_ad.q.view_mut(),
                dq: self.next_ad.dq.view(),
            },
            r: self.r_ad.view_mut(),
        }
    }

    pub fn step_state<'dq, 'r>(
        &mut self,
        dq_next: &'dq [D],
        r: &'r mut [D],
    ) -> StepStateComponent<&[Q], &[Q], &[D], &mut [Q], &'dq [D], &'r mut [D]> {
        StepStateComponent {
            prev_q: self.prev.q.view(),
            cur: GeneralizedState {
                q: self.cur.q.view(),
                dq: self.cur.dq.view(),
            },
            next: GeneralizedState {
                q: self.next_q.view_mut(),
                dq: dq_next,
            },
            r,
        }
    }
}

///// Rigid degrees of freedom with past and present states.
//#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
//pub struct Rigid<X, V, R, W, XAD, RAD> {
//    #[component]
//    pub prev: RigidState<X, V, R, W>,
//    #[component]
//    pub cur: RigidState<X, V, R, W>,
//    #[component]
//    pub next_pos: X,
//    #[component]
//    pub next_orientation: R,
//    #[component]
//    pub next_pos_ad: XAD,
//    #[component]
//    pub next_orientation_ad: RAD,
//}

// Data for a subset of vertices on the surface of the object.
pub type SurfaceVertexData<D> = Subset<D>;
/// View of the data for a subset of vertices on the surface of the object.
pub type SurfaceVertexView<'i, D> = SubsetView<'i, D>;

// Data for a subset of vertices on the surface of the object in triplets.
pub type SurfaceVertexData3<T> = SurfaceVertexData<Chunked3<T>>;
/// View of the data for a subset of vertices on the surface of the object in triplets.
pub type SurfaceVertexView3<'i, T> = SurfaceVertexView<'i, Chunked3<T>>;

/// Simulation vertex and mesh data.
///
/// Each simulated object should know which dofs and vertices it affects.
#[derive(Clone, Debug)]
pub struct State<T, F> {
    /// Generalized coordinates from previous and for current time step.
    ///
    /// The coordinates are chunked by the type of object they control.
    ///
    /// These correspond to free variables of the problem or quantities with the
    /// same topology as the free variables (e.g. residuals). The topology of
    /// how these affect the simulation is encoded by each simulated object.
    pub dof: Chunked<GeneralizedCoords<Vec<T>, Vec<T>, Vec<F>, Vec<F>, Vec<F>>>,

    /// Per vertex positions, velocities and other workspace quantities.
    pub vtx:
        VertexWorkspace<Chunked3<Vec<T>>, Chunked3<Vec<F>>, Vec<T>, Vec<usize>, Vec<VertexType>>,

    pub shell: TriShell,
    pub solid: TetSolid,
    //pub rigid: RigidBody,
    ///// Rigid object state.
    // If orientation is specified in quaternions, then we could place the chunking
    // structure around the Vecs instead of around Rigid.
    //pub rigid: Chunked3<Rigid<Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<F>, Vec<F>>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VertexType {
    Free,
    /* Reduced? */
    Rigid,
    Fixed,
    Ignored,
}

impl VertexType {
    /// Returns the next varient in the order of definition.
    const fn next_variant(&self) -> Self {
        use VertexType::*;
        match *self {
            Free => Rigid,
            Rigid => Fixed,
            Fixed => Ignored,
            Ignored => Free,
        }
    }
}

/// Generalized state variables used for time integration.
#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
pub struct StepStateComponent<P, CQ, CD, NQ, ND, R> {
    prev_q: P,
    #[component]
    cur: GeneralizedState<CQ, CD>,
    #[component]
    next: GeneralizedState<NQ, ND>,
    r: R,
}

pub type StepState<Const, Mut> = StepStateComponent<Const, Const, Const, Mut, Const, Mut>;

impl<T: Real> State<T, ad::FT<T>> {
    /// Constructs a state struct from a global mesh and a set of materials.
    pub fn try_from_mesh_and_materials(
        mesh: &Mesh,
        materials: &[Material],
    ) -> Result<State<T, ad::FT<T>>, Error> {
        // Enumerate all vertices so that fixed vertices are put at the end.
        let num_verts = mesh.num_vertices();
        let mut vertex_indices: Vec<usize> = (0..num_verts).collect();

        // Mark Fixed vertices
        let mut vertex_type = mesh
            .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            .map(|attrib| {
                attrib
                    .map(|&x| {
                        if x > 0 {
                            VertexType::Fixed
                        } else {
                            VertexType::Ignored
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|_| vec![VertexType::Ignored; num_verts]);

        if let Ok(mtl_id) = mesh.attrib_as_slice::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB) {
            // Mark vertices attached to rigid objects as rigid.
            for (cell, &mtl_id) in mesh.cell_iter().zip(mtl_id.iter()) {
                if let Material::Rigid(_) = materials[mtl_id.max(0) as usize] {
                    for ty in cell
                        .iter()
                        .map(|&idx| &mut vertex_type[idx])
                        .filter(|&ty| *ty == VertexType::Ignored)
                    {
                        *ty = VertexType::Rigid;
                    }
                }
            }

            // Mark vertices attached to soft objects as free.
            for ((cell, cell_type), &mtl_id) in mesh
                .cell_iter()
                .zip(mesh.cell_type_iter())
                .zip(mtl_id.iter())
            {
                if let Material::SoftShell(_) | Material::Solid(_) =
                    materials[mtl_id.max(0) as usize]
                {
                    if TetSolid::is_valid_cell(cell, cell_type)
                        || TriShell::is_valid_cell(cell, cell_type)
                    {
                        for ty in cell
                            .iter()
                            .map(|&idx| &mut vertex_type[idx])
                            .filter(|&ty| *ty == VertexType::Ignored)
                        {
                            *ty = VertexType::Free;
                        }
                    }
                }
            }
        }

        let num_free_verts = vertex_type
            .iter()
            .filter(|&x| *x == VertexType::Free)
            .count();
        mesh.sort_vertices_by_key(|vtx_idx| vertex_type[vtx_idx]);
        vertex_type.sort();

        let solid = TetSolid::try_from_mesh_and_materials(&mesh, materials, &vertex_type)?;
        let shell = TriShell::try_from_mesh_and_materials(&mesh, materials, &vertex_type)?;
        //let rigid = RigidBody::try_from_mesh_and_materials(&mesh, materials)?;

        let vel = mesh
            .attrib::<VelType, VertexIndex>(VELOCITY_ATTRIB)
            .unwrap_or_else(|_| vec![[0.0; 3]; num_verts]);

        // Degrees of freedom.
        // All non-fixed vertices that are not part of a rigid body are degrees
        // of freedom in this scheme.
        let dof = GeneralizedCoords::new_chunked(
            &solid,
            &shell,
            // rigid,
            Chunked3::from_arrays(mesh.vertex_positions()),
            Chunked3::from_arrays(vel),
            vertex_type,
        );

        // This is the vertex workspace which gets updated on demand.
        let vtx = VertexWorkspace::default();

        // Masses don't usually change so we can initialize them now.
        let mass = Self::compute_vertex_masses(&solid, &shell, num_verts);
        vtx.mass_inv = mass
            .into_iter()
            .take(num_free_verts)
            .map(|m| T::from(1.0 / m).unwrap())
            .chain(std::iter::repeat(T::zero()).take(num_free_verts))
            .collect();

        // Save the map to original indices, which is needed when we update
        // vertex positions.
        vtx.orig_index = mesh
            .attrib_as_slice::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?
            .iter()
            .map(|&i| usize::from(i))
            .collect();

        vtx.vertex_type = vertex_type;

        assert_eq!(vtx.mass_inv.len(), num_verts);

        //let rigid = Self::build_rigid_bodies(trimesh)?;
        Ok(State {
            dof,
            vtx,
            shell,
            solid,
            //rigid
        })
    }

    pub fn clone_as_autodiff(&self) -> State<ad::FT<f64>, ad::FT<ad::FT<f64>>> {
        let State {
            dof,
            vtx,
            shell,
            solid,
            //rigid,
        } = self;

        let convert = |v: &[T]| {
            v.iter()
                .cloned()
                .map(|x| ad::F::cst(x.to_f64().unwrap()))
                .collect::<Vec<_>>()
        };
        let convert3 = |v: Chunked3<&[T]>| {
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
        let convert_state_ad =
            |state: &GeneralizedState<&[ad::FT<T>], &[ad::FT<T>]>| GeneralizedState {
                q: convert_ad(state.q),
                dq: convert_ad(state.dq),
            };
        let convert_coords =
            |coords: &GeneralizedCoords<&[T], &[T], &[ad::FT<T>], &[ad::FT<T>], &[ad::FT<T>]>| {
                GeneralizedCoords {
                    prev: convert_state(&coords.prev),
                    cur: convert_state(&coords.cur),
                    next_q: convert(&coords.next_q),
                    next_ad: convert_state_ad(&coords.next_ad),
                    r_ad: convert_ad(&coords.r_ad),
                    //active_dofs: Vec::new(),
                }
            };
        let convert_vtx_state =
            |state: &ParticleState<Chunked3<&[T]>, Chunked3<&[T]>>| ParticleState {
                pos: convert(state.pos),
                vel: convert(state.vel),
            };
        let convert_vtx_workspace = |ws: &VertexWorkspace<
            Chunked3<&[T]>,
            Chunked3<&[ad::FT<T>]>,
            &[T],
            &[usize],
            &[VertexType],
        >| VertexWorkspace {
            state: convert_vtx_state(&ws.state),
            state_ad: ParticleState {
                pos: convert_ad(&ws.state_ad.pos),
                vel: convert_ad(&ws.state_ad.vel),
            },
            grad: convert(ws.grad),
            lambda: convert(ws.lambda),
            vfc: convert(ws.vfc),
            lambda_ad: convert_ad(ws.lambda_ad),
            vfc_ad: convert_ad(ws.vfc_ad),
            mass_inv: convert(ws.mass_inv),
            orig_index: ws.orig_index.to_vec(),
            vertex_type: ws.vertex_type.to_vec(),
        };
        let dof_storage = convert_coords(dof.view().storage());
        let dof = dof.clone_with_storage(dof_storage);
        let vtx_storage = convert_vtx_workspace(vtx.view().storage());
        let vtx = vtx.clone_with_storage(vtx_storage);
        State {
            dof,
            vtx,
            shell: shell.clone(),
            solid: solid.clone(),
            //rigid: rigid.clone(),
        }
    }

    /// Transfer state from degrees of freedom to vertex state.
    ///
    /// This ensures that vertex data queried by constraint functions is current.
    pub fn update_vertices<S: Real>(&mut self, dq_next: &[S]) {
        let Self { dof, vtx, .. } = self;
        let dof_next = dof.map_storage(|dof| GeneralizedState {
            q: dof.next_q,
            dq: dq_next,
        });
        vtx.state.update(dof_next);
    }

    /// Transfer state from degrees of freedom to vertex state.
    ///
    /// This ensures that vertex data queried by constraint functions is current.
    pub fn update_vertices_ad<S: Real>(&mut self) {
        let Self { dof, vtx, .. } = self;
        let dof_next = dof.map_storage(|dof| GeneralizedState {
            q: dof.next_ad.q,
            dq: dof.next_ad.dq,
        });
        vtx.state_ad.update(dof_next);
    }

    ///// Update vertex velocities of vertices not in q (dofs).
    //pub fn sync_vel<S: Real>(
    //    shells: &[TriMeshShell],
    //    dq_next: GeneralizedView3<&[S]>,
    //    q_cur: GeneralizedView3<&[S]>,
    //    mut vel_next: VertexView3<&mut [S]>,
    //) {
    //    for (i, shell) in shells.iter().enumerate() {
    //        if let ShellData::Rigid { .. } = shell.data {
    //            let dq_next = dq_next.isolate(SHELLS_INDEX).isolate(i);
    //            let q_cur = q_cur.isolate(SHELLS_INDEX).isolate(i);
    //            debug_assert_eq!(dq_next.len(), 2);
    //            let rotation = Vector3::new(q_cur[1]);
    //            let linear = Vector3::new(dq_next[0]);
    //            let angular = Vector3::new(dq_next[1]);
    //            let mut vel_next = vel_next.view_mut().isolate(SHELLS_INDEX).isolate(i);
    //            for (out_vel, &r) in vel_next.iter_mut().zip(
    //                shell
    //                    .trimesh
    //                    .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
    //                    .expect("Missing rigid body reference positions"),
    //            ) {
    //                *out_vel.as_mut_tensor() =
    //                    rotate(angular.cross(r.into_tensor().cast::<S>()), rotation) + linear;
    //            }
    //        }
    //    }
    //}

    ///// Transfer internally stored workspace gradient to the given array of degrees of freedom.
    ///// This is a noop when degrees of freedom coincide with vertex velocities.
    //pub fn sync_grad(&self, source: SourceObject, grad_x: VertexView3<&mut [T]>) {
    //    let mut ws = self.workspace.borrow_mut();
    //    match source {
    //        SourceObject::Shell(i) => {
    //            if let ShellData::Rigid { .. } = self.shells[i].data {
    //                let mut grad_dofs = grad_x.isolate(SHELLS_INDEX).isolate(i);
    //                let grad_vtx = &mut ws
    //                    .vtx
    //                    .view_mut()
    //                    .map_storage(|vtx| vtx.grad)
    //                    .isolate(SHELLS_INDEX)
    //                    .isolate(i);
    //                debug_assert_eq!(grad_dofs.len(), 2);
    //                let r_iter = self.shells[i]
    //                    .trimesh
    //                    .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
    //                    .expect("Missing rigid body reference positions");
    //                for (g_vtx, &r) in grad_vtx.iter_mut().zip(r_iter) {
    //                    // Transfer gradient from vertices to degrees of freedom.
    //                    *grad_dofs[0].as_mut_tensor() += *g_vtx.as_tensor();
    //                    let r = r.into_tensor();
    //                    *grad_dofs[1].as_mut_tensor() +=
    //                        r.cast::<T>().cross((*g_vtx).into_tensor());
    //                    *g_vtx = [T::zero(); 3]; // Value moved over, reset it.
    //                }
    //            }
    //        }
    //        _ => {} // Noop. Nothing to do since vertices and degrees of freedom are the same.
    //    }
    //}

    ///// Add vertex vectors to the given array of degrees of freedom.
    /////
    ///// This is a noop when degrees of freedom coincide with vertex velocities.
    ///// This function defines how non vertex degrees of freedom are related to
    ///// vertex DoFs for velocity vector quantities.
    //pub fn transfer_velocity_vtx_to_dof<I: Real, O: Real>(
    //    &self,
    //    source: SourceObject,
    //    vtx: VertexView3<&mut [I]>,
    //    dof: GeneralizedView3<&mut [O]>,
    //) {
    //    match source {
    //        SourceObject::Shell(i) => {
    //            let mut dof = dof.isolate(SHELLS_INDEX).isolate(i);
    //            let mut vtx = vtx.isolate(SHELLS_INDEX).isolate(i);
    //            if let ShellData::Rigid { .. } = self.shells[i].data {
    //                debug_assert_eq!(dof.len(), 2);
    //                let r_iter = self.shells[i]
    //                    .trimesh
    //                    .attrib_iter::<RigidRefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
    //                    .expect("Missing rigid body reference positions");
    //                for (vtx, &r) in vtx.iter_mut().zip(r_iter) {
    //                    // Transfer gradient from vertices to degrees of freedom.
    //                    *dof[0].as_mut_tensor() += vtx.as_tensor().cast::<O>();
    //                    let r = r.into_tensor();
    //                    *dof[1].as_mut_tensor() +=
    //                        r.cast::<O>().cross((*vtx).into_tensor().cast::<O>());
    //                    // Transfer complete, zero out input.
    //                    *vtx = [I::zero(); 3];
    //                }
    //            }
    //        }
    //        _ => {}
    //    }
    //}

    //// Update `cur_x` using backward Euler integration with the given velocity `v`.
    //pub fn integrate_step(&self, dt: f64) {
    //    debug_assert!(dt > 0.0);
    //    let mut ws = self.workspace.borrow_mut();
    //    let WorkspaceData { dof, .. } = &mut *ws;

    //    {
    //        let mut dof_next = dof.view_mut().into_storage().state;
    //        let q_cur = self.dof.view().into_storage().cur.q;
    //        debug_assert_eq!(q_cur.len(), dof_next.len());

    //        // In static simulations, velocity is simply displacement.

    //        // Integrate all (positional) degrees of freedom using standard implicit euler.
    //        // Note this code includes rigid motion, but we will overwrite those below.
    //        dof_next
    //            .iter_mut()
    //            .zip(q_cur.iter())
    //            .for_each(|(GeneralizedState { q, dq }, &x0)| {
    //                *q = dq.mul_add(T::from(dt).unwrap(), x0)
    //            });
    //    }

    //    // Integrate rigid rotation.
    //    let mut dof_next = dof.view_mut().isolate(SHELLS_INDEX);
    //    let q_cur = self
    //        .dof
    //        .view()
    //        .isolate(SHELLS_INDEX)
    //        .map_storage(|dof| dof.cur.q);

    //    for (shell, q_cur, dof_next) in zip!(self.shells.iter(), q_cur.iter(), dof_next.iter_mut())
    //    {
    //        match shell.data {
    //            ShellData::Rigid { .. } => {
    //                // We are only interested in rigid rotations.
    //                assert_eq!(q_cur.len(), 2);
    //                let dof = dof_next.isolate(1).state;
    //                *dof.q = integrate_rotation(
    //                    q_cur[1].into_tensor(),
    //                    *dof.dq.as_mut_tensor() * T::from(dt).unwrap(),
    //                )
    //                .into_data();
    //            }
    //            _ => {}
    //        }
    //    }
    //}

    /// Take a backward Euler step computed in `q_next`.
    pub fn be_step<S: Real>(state: ChunkedView<StepState<&[S], &mut [S]>>, dt: f64) {
        debug_assert!(dt > 0.0);

        // In static simulations, velocity is simply displacement.

        // Integrate all (positional) degrees of freedom using standard implicit Euler.
        // Note this code includes rigid motion, but we will overwrite those below.
        state
            .isolate(VERTEX_DOFS)
            .into_iter()
            .for_each(|StepState { cur, next, .. }| {
                *next.q = num_traits::Float::mul_add(next.dq, S::from(dt).unwrap(), cur.q)
            });

        // Integrate rigid rotation.
        //let mut q_next = dof.isolate(SHELLS_INDEX).map_storage(|_| q_next);
        //let v_next = dof.isolate(SHELLS_INDEX).map_storage(|_| v);
        //let q_cur = dof.isolate(SHELLS_INDEX).map_storage(|_| q_cur);

        //for (shell, q_cur, v_next, q_next) in zip!(
        //    self.shells.iter(),
        //    q_cur.iter(),
        //    v_next.iter(),
        //    q_next.iter_mut()
        //) {
        //    match shell.data {
        //        ShellData::Rigid { .. } => {
        //            // We are only interested in rigid rotations.
        //            assert_eq!(q_cur.len(), 2);
        //            let q_next = q_next.isolate(1);
        //            let v_next = v_next.isolate(1);
        //            *q_next = integrate_rotation(
        //                q_cur[1].into_tensor().cast::<S>(),
        //                *v_next.as_tensor() * S::from(dt).unwrap(),
        //            )
        //            .into_data();
        //        }
        //        _ => {}
        //    }
        //}
    }

    /// Take a blended step in `q` between current and previous `v` values.
    ///
    /// This type of step is used in trapezoidal rule, implicit Newmark
    /// integration and other SDIRK variants.
    ///
    /// `q_next = q_cur + h*((1-alpha)*v_next + alpha*v_cur)`
    pub fn lerp_step<S: Real>(state: ChunkedView<StepState<&[S], &mut [S]>>, dt: f64, alpha: f64) {
        debug_assert!(dt > 0.0);

        let dt = S::from(dt).unwrap();
        let alpha = S::from(alpha).unwrap();

        // In static simulations, velocity is simply displacement.

        // Note this code includes rigid rotation, but we will overwrite that below.
        state
            .isolate(VERTEX_DOFS)
            .into_iter()
            .for_each(|StepState { cur, next, .. }| {
                // `q_next = q_cur + h*((1-alpha)*v_next + alpha*dq_cur)`
                *next.q = *next
                    .dq
                    .mul_add(S::one() - alpha, cur.dq * alpha)
                    .mul_add(dt, cur.q);
            });

        // Integrate rigid rotation.
        //let mut dof_next = ws.dof.view_mut().isolate(SHELLS_INDEX);
        //let dof_cur = dof.isolate(SHELLS_INDEX).map_storage(|dof| dof.cur);

        //for (shell, dof_cur, dof_next) in
        //    zip!(self.shells.iter(), dof_cur.iter(), dof_next.iter_mut())
        //{
        //    match shell.data {
        //        ShellData::Rigid { .. } => {
        //            assert_eq!(dof_cur.len(), 2);

        //            // We are only interested in rigid rotations.
        //            // Translations have been integrated correctly above.
        //            let cur = dof_cur.isolate(1);

        //            let dof = dof_next.isolate(SHELLS_INDEX);
        //            let v_next = dof.state.dq.as_tensor();
        //            *dof.state.q = (Quaternion::from_vector(*cur.q.as_tensor())
        //                * Quaternion::from_vector(*cur.dq.as_tensor() * dt * alpha)
        //                * Quaternion::from_vector(*v_next * dt * (T::one() - alpha)))
        //            .into_vector()
        //            .into_data()
        //        }
        //        _ => {}
        //    }
        //}
    }

    /// Take a `gamma` parametrized BDF2 step in `q` computed in the state workspace.
    ///
    /// Use `gamma = 1/2` for vanilla BDF2. Other values of `gamma` are useful for BDF2 variants like TR-BDF2.
    pub fn bdf2_step<S: Real>(
        &mut self,
        state: ChunkedView<StepState<&[S], &mut [S]>>,
        dt: f64,
        gamma: f64,
    ) {
        debug_assert!(dt > 0.0);

        let dt = S::from(dt).unwrap();
        let gamma = S::from(gamma).unwrap();

        // Compute coefficients
        let _1 = S::one();
        let _2 = S::from(2.0).unwrap();
        let a = _1 / (gamma * (_2 - gamma));
        let b = (_1 - gamma) * (_1 - gamma) / (gamma * (_2 - gamma));
        let c = (_1 - gamma) / (_2 - gamma);

        // Integrate all (positional) degrees of freedom using standard implicit euler.
        // Note this code includes rigid motion, but we will overwrite those below.
        state.isolate(VERTEX_DOFS).into_iter().for_each(
            |StepState {
                 prev_q, cur, next, ..
             }| {
                next.q = next.dq * c * dt + cur.q * a - prev_q * b;
            },
        );

        // Integrate rigid rotation.
        //let mut dof_next = ws.dof.view_mut().isolate(SHELLS_INDEX);
        //let q_cur = dof.isolate(SHELLS_INDEX).map_storage(|dof| dof.cur.q);
        //let q_prev = dof.isolate(SHELLS_INDEX).map_storage(|dof| dof.prev.q);

        //for (shell, q_prev, q_cur, dof_next) in zip!(
        //    self.shells.iter(),
        //    q_prev.iter(),
        //    q_cur.iter(),
        //    dof_next.iter_mut()
        //) {
        //    match shell.data {
        //        ShellData::Rigid { .. } => {
        //            // We are only interested in rigid rotations.
        //            assert_eq!(q_cur.len(), 2);
        //            assert_eq!(q_prev.len(), 2);
        //            let dof = dof_next.isolate(SHELLS_INDEX);
        //            let v_next = dof.state.dq.as_tensor();
        //            *dof.state.q = (Quaternion::from_vector(*q_cur[1].as_tensor() * a)
        //                * Quaternion::from_vector(-*q_prev[1].as_tensor() * b)
        //                * Quaternion::from_vector(*v_next * dt * c))
        //            .into_vector()
        //            .into_data()
        //        }
        //        _ => {}
        //    }
        //}
    }

    /// Updates fixed vertices(
    pub fn update_fixed_vertices(
        &mut self,
        new_pos: Chunked3<&[f64]>,
        time_step: f64,
    ) -> Result<(), crate::Error> {
        debug_assert!(time_step > 0.0);
        let dt_inv = T::one() / T::from(time_step).unwrap();

        // Get the vertex index of the original vertex (point in given point cloud).
        // This is done because meshes can be reordered when building the
        // solver.
        let new_pos_iter = self.vtx.orig_index.iter().map(|&idx| new_pos.get(idx));

        let Self {
            vtx: VertexWorkspace {
                state, vertex_type, ..
            },
            ..
        } = self;

        // Only update fixed vertices, if no such attribute exists, return an error.
        state
            .iter_mut()
            .zip(new_pos_iter)
            .zip(vertex_type.iter())
            .filter_map(|(data, &vt)| {
                if vt == VertexType::Fixed {
                    Some(data)
                } else {
                    None
                }
            })
            .for_each(|(ParticleState { pos, vel }, new_pos)| {
                // Update the vertices we find in the given `new_pos` collection.
                if let Some(&new_pos) = new_pos {
                    *vel.as_mut_tensor() =
                        (new_pos.as_tensor().cast::<T>() - *(*pos).as_tensor()) * dt_inv;
                    *pos.as_mut_tensor() = new_pos.as_tensor().cast::<T>();
                }
            });
        Ok(())
    }

    //    /// Update the shell meshes with the given global array of vertex positions
    //    /// and velocities for all shells.
    //    pub fn update_shell_vertices(
    //        &mut self,
    //        new_pos: Chunked3<&[f64]>,
    //        time_step: f64,
    //    ) -> Result<(), crate::Error> {
    //        // Some shells are simulated on a per vertex level, some are rigid or
    //        // fixed, so we will update `dof` for the former and `vtx` for the
    //        // latter.
    //        let State { dof, vtx, .. } = self;
    //
    //        let mut dof_cur = dof.view_mut().isolate(SHELLS_INDEX).map_storage(|q| q.cur);
    //        let mut vtx_cur = vtx.view_mut().isolate(SHELLS_INDEX).map_storage(|v| v.cur);
    //
    //        debug_assert!(time_step > 0.0);
    //        let dt_inv = T::one() / T::from(time_step).unwrap();
    //
    //        // Get the trimesh and {dof,vtx}_cur so we can update the fixed vertices.
    //        for (shell, (mut dof_cur, mut vtx_cur)) in self
    //            .shells
    //            .iter()
    //            .zip(dof_cur.iter_mut().zip(vtx_cur.iter_mut()))
    //        {
    //            // Get the vertex index of the original vertex (point in given point cloud).
    //            // This is done because meshes can be reordered when building the
    //            // solver. This attribute maintains the link between the caller and
    //            // the internal mesh representation. This way the user can still
    //            // update internal meshes as needed between solves.
    //            let source_index_iter = shell
    //                .trimesh
    //                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
    //            let new_pos_iter = source_index_iter.map(|&idx| new_pos.get(idx as usize));
    //
    //            match shell.data {
    //                ShellData::Soft { .. } => {
    //                    // Generalized coordinates of Soft shells coincide with vertex coordinates.
    //                    // Only update fixed vertices, if no such attribute exists, return an error.
    //                    let fixed_iter = shell
    //                        .trimesh
    //                        .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
    //                    dof_cur
    //                        .iter_mut()
    //                        .zip(new_pos_iter)
    //                        .zip(fixed_iter)
    //                        .filter_map(|(data, &fixed)| if fixed != 0i8 { Some(data) } else { None })
    //                        .for_each(|(GeneralizedState { q, dq }, new_pos)| {
    //                            // It's possible that the new vector of positions is missing some
    //                            // vertices that were fixed before, so we try to update those we
    //                            // actually find in the `new_pos` collection.
    //                            if let Some(&new_pos) = new_pos {
    //                                *dq.as_mut_tensor() =
    //                                    (new_pos.as_tensor().cast::<T>() - *(*q).as_tensor()) * dt_inv;
    //                                //*pos = new_pos; // automatically updated via solve.
    //                            }
    //                        });
    //                }
    //                ShellData::Rigid { fixed, .. } => {
    //                    // Rigid bodies can have 0, 1, or 2 fixed vertices.
    //                    // With 3 fixed vertices these become completely fixed.
    //                    let source_indices = shell
    //                        .trimesh
    //                        .attrib_as_slice::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
    //
    //                    match fixed {
    //                        FixedVerts::Zero => {
    //                            // For 0 fixed vertices, nothing needs to be done here.
    //                        }
    //                        FixedVerts::One(vtx_idx) => {
    //                            // For 1 fixed vertex, assign the appropriate velocity to that vertex.
    //                            if let Some(&new_pos) = new_pos.get(source_indices[vtx_idx] as usize) {
    //                                let mut vtx_cur = vtx_cur.isolate(vtx_idx);
    //                                *(&mut vtx_cur.vel).as_mut_tensor() =
    //                                    (new_pos.into_tensor().cast::<T>() - *vtx_cur.pos.as_tensor())
    //                                        * dt_inv;
    //                                // Position will by updated by the solve automatically.
    //                            }
    //                        }
    //                        FixedVerts::Two(verts) => {
    //                            // For 2 fixed vertices.
    //                            // TODO: This may be a bad idea since it may generate infeasible configurations
    //                            //       quite easily. Resolve this.
    //                            if let Some(&p0) = new_pos.get(source_indices[verts[0]] as usize) {
    //                                if let Some(&p1) = new_pos.get(source_indices[verts[1]] as usize) {
    //                                    let mut v0 = vtx_cur.view_mut().isolate(verts[0]);
    //                                    *(&mut v0.vel).as_mut_tensor() = (p0.into_tensor().cast::<T>()
    //                                        - *v0.pos.as_tensor())
    //                                        * dt_inv;
    //                                    let mut v1 = vtx_cur.view_mut().isolate(verts[1]);
    //                                    *(&mut v1.vel).as_mut_tensor() = (p1.into_tensor().cast::<T>()
    //                                        - *v1.pos.as_tensor())
    //                                        * dt_inv;
    //                                }
    //                            }
    //                        }
    //                    }
    //                }
    //                ShellData::Fixed { .. } => {
    //                    // This mesh is fixed and doesn't obey any physics. Simply
    //                    // copy the positions and velocities over.
    //                    vtx_cur.iter_mut().zip(new_pos_iter).for_each(
    //                        |(ParticleState { pos, vel }, new_pos)| {
    //                            if let Some(&new_pos) = new_pos {
    //                                let new_pos_t = new_pos.as_tensor().cast::<T>();
    //                                *vel.as_mut_tensor() = (new_pos_t - *(*pos).as_tensor()) * dt_inv;
    //                                *pos = new_pos_t.into(); // Not automatically updated since these are not part of the solve.
    //                            }
    //                        },
    //                    );
    //                }
    //            }
    //        }
    //
    //        // Update current pos and vel as well
    //        let mut ws = self.workspace.borrow_mut();
    //        let vtx_state = &mut ws.vtx.storage_mut().state;
    //        vtx_state.pos.copy_from_slice(&vtx_cur.storage().pos);
    //        vtx_state.vel.copy_from_slice(&vtx_cur.storage().vel);
    //
    //        Ok(())
    //    }

    /// Advance `*.prev` variables to `*.cur`, and those to current workspace variables and
    /// update the referenced meshes.
    ///
    /// If `dq_next` is given, then velocities are also advanced.
    pub fn advance(&mut self, dq_next: &[T]) {
        let Self {
            dof, vtx, shell, ..
        } = self;

        // Advance degrees of freedom.
        dof.storage_mut().advance(dq_next);
        // Update vertex data with the new state.
        vtx.state.update(dof.storage().next_q, dq_next);

        // Update edge angles for shells.
        let pos_next = vtx.state.map_storage(|state| state.pos).into();
        shell.update_dihedral_angles(pos_next);
    }

    /// Reverts to previous step.
    ///
    /// This is the opposite of `advance`.
    pub fn retreat(&mut self) {
        let State {
            dof, vtx, shell, ..
        } = self;

        // Retreat degrees of freedom.
        dof.storage_mut().retreat();
        // Update vertex data with the old state.
        vtx.state.update(dof.storage().cur.q, dof.storage().cur.dq);

        // Update edge angles for shells.
        let pos_prev = self.vtx.state.map_storage(|state| state.pos).into();
        shell.update_dihedral_angles(pos_prev);
    }

    /// Updates the given input vertex positions and velocities with the current vertex state.
    ///
    /// Note that to get the latest state into vertex quantities, `update` must be called first.
    pub fn update_input(&self, input_vtx_pos: &mut [[f64; 3]], input_vtx_vel: &mut [[f64; 3]]) {
        self.vtx
            .state
            .iter()
            .zip(self.vtx.orig_index.iter())
            .for_each(|(ParticleState { pos, vel }, &orig_index)| {
                input_vtx_pos[orig_index].as_mut_tensor() = pos.as_tensor().cast::<f64>();
                input_vtx_vel[orig_index].as_mut_tensor() = vel.as_tensor().cast::<f64>();
            });
    }

    ///// Returns the rigid motion translation and rotation for the given source object if it is rigid.
    /////
    ///// If the object is non-rigid, `None` is returned.
    //#[inline]
    //pub fn rigid_motion(&self, src: SourceObject) -> Option<[[T; 3]; 2]> {
    //    let q_cur = self
    //        .dof
    //        .view()
    //        .map_storage(|dof| dof.cur.q)
    //        .at(SHELLS_INDEX);
    //    if let SourceObject::Shell(idx) = src {
    //        let q_cur = q_cur.at(idx);
    //        if let ShellData::Rigid { .. } = self.shells[idx].data {
    //            Some([q_cur[0], q_cur[1]])
    //        } else {
    //            None
    //        }
    //    } else {
    //        None
    //    }
    //}

    /// Returns the total number of degrees of freedom.
    #[inline]
    pub fn num_dofs(&self) -> usize {
        self.dof.len()
    }

    /// Compute vertex masses on the given solid and shell elements.
    pub fn compute_vertex_masses(
        solid: &TetSolid,
        shell: &TriShell,
        num_vertices: usize,
    ) -> Vec<MassType> {
        let mut masses = vec![0.0; num_vertices];

        for (&vol, &density, cell) in zip!(
            solid.nh_tet_elements.ref_volume.iter(),
            solid.nh_tet_elements.density.iter(),
            solid.nh_tet_elements.tets.iter(),
        )
        .chain(zip!(
            solid.snh_tet_elements.ref_volume.iter(),
            solid.snh_tet_elements.density.iter(),
            solid.snh_tet_elements.tets.iter(),
        )) {
            for i in 0..4 {
                masses[cell[i]] += 0.25 * vol * f64::from(density);
            }
        }

        for (&area, &density, cell) in zip!(
            shell.triangle_elements.ref_area.iter(),
            shell.triangle_elements.density.iter(),
            shell.triangle_elements.triangles.iter(),
        ) {
            for i in 0..3 {
                masses[cell[i]] += area * f64::from(density) / 3.0;
            }
        }

        masses
    }
}