use std::cell::RefCell;

use num_traits::Zero;

use autodiff as ad;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use geo::ops::{ShapeMatrix, Volume};
use tensr::*;

use super::mcp::*;
use super::newton::*;
use super::problem::{FrictionalContactConstraint, NLProblem, NonLinearProblem};
use super::{NLSolver, SimParams, SolveResult, Status};
use crate::attrib_defines::*;
use crate::constraints::point_contact::compute_contact_penalty;
use crate::constraints::*;
use crate::contact::*;
use crate::fem::{ref_tet, state::*};
use crate::inf_norm;
use crate::objects::*;
use crate::Real64;
use crate::{Error, PointCloud, PolyMesh, TetMesh, TriMesh};

#[derive(Clone, Debug)]
pub struct SolverBuilder {
    sim_params: SimParams,
    solids: Vec<(TetMesh, SolidMaterial)>,
    soft_shells: Vec<(PolyMesh, SoftShellMaterial)>,
    rigid_shells: Vec<(PolyMesh, RigidMaterial)>,
    fixed_shells: Vec<(PolyMesh, FixedMaterial)>,
    frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
}

/// The index of the object subject to the appropriate contact constraint.
///
/// This enum helps us map from the particular contact constraint to the
/// originating simulation object (shell or solid).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SourceIndex {
    Solid(usize),
    Shell(usize),
}

impl SourceIndex {
    #[inline]
    fn into_source(self, with_fixed: bool) -> SourceObject {
        match self {
            SourceIndex::Solid(i) => SourceObject::Solid(i, with_fixed),
            SourceIndex::Shell(i) => SourceObject::Shell(i),
        }
    }

    #[inline]
    pub fn get(&self) -> usize {
        match self {
            SourceIndex::Solid(idx) | SourceIndex::Shell(idx) => *idx,
        }
    }
}

impl SolverBuilder {
    /// Create a `SolverBuilder` with the minimum required parameters, which are the simulation
    /// parameters, `SimParams`.
    pub fn new(sim_params: SimParams) -> Self {
        SolverBuilder {
            sim_params,
            solids: Vec::new(),
            soft_shells: Vec::new(),
            rigid_shells: Vec::new(),
            fixed_shells: Vec::new(),
            frictional_contacts: Vec::new(),
        }
    }

    /// Add a tetmesh representing a soft solid (e.g. soft tissue).
    pub fn add_solid(&mut self, mesh: TetMesh, mat: SolidMaterial) -> &mut Self {
        self.solids.push((mesh, mat));
        self
    }

    /// Add a polygon mesh representing a shell (e.g. cloth).
    pub fn add_shell<M>(&mut self, shell: PolyMesh, mat: M) -> &mut Self
    where
        M: Into<ShellMaterial>,
    {
        match mat.into() {
            ShellMaterial::Fixed(m) => self.fixed_shells.push((shell, m)),
            ShellMaterial::Rigid(m) => self.rigid_shells.push((shell, m)),
            ShellMaterial::Soft(m) => self.soft_shells.push((shell, m)),
        }
        self
    }

    /// Add a polygon mesh representing a soft shell (e.g. cloth).
    pub fn add_soft_shell(&mut self, shell: PolyMesh, mat: SoftShellMaterial) -> &mut Self {
        self.soft_shells.push((shell, mat));
        self
    }

    /// Add a static polygon mesh representing an immovable solid.
    pub fn add_fixed(&mut self, shell: PolyMesh, id: usize) -> &mut Self {
        self.fixed_shells.push((shell, FixedMaterial::new(id)));
        self
    }

    /// Add a rigid polygon mesh representing an rigid solid.
    pub fn add_rigid(&mut self, shell: PolyMesh, density: f32, id: usize) -> &mut Self {
        self.rigid_shells
            .push((shell, RigidMaterial::new(id, density)));
        self
    }

    /// Set parameters for frictional contact problems. The given two material IDs determine which
    /// materials should experience frictional contact described by the given parameters. To add
    /// self-contact, simply set the two ids to be equal. For one-directional
    /// models, the first index corresponds to the object (affected) while the
    /// second index corresponds to the collider (unaffected). Some
    /// bi-directional constraints treat the two objects differently, and
    /// changing the order of the indices may change the behaviour. In these
    /// cases, the first index corresponds to the `object` (primary) and the second to the
    /// `collider` (secondary).
    pub fn add_frictional_contact(
        &mut self,
        params: FrictionalContactParams,
        mat_ids: (usize, usize),
    ) -> &mut Self {
        // We can already weed out frictional contacts for pure static sims
        // since we already have the `SimParams`.
        if params.friction_params.is_none() || self.sim_params.time_step.is_some() {
            self.frictional_contacts.push((params, mat_ids));
        }
        self
    }

    /// Construct a chunked vector of source indices where each chunk represents a unique material
    /// id. This means that the first chunk contains all objects whose material id is 0, second
    /// chunk contains all objects with material id being 1 and so on.
    pub(crate) fn build_material_sources(
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
    ) -> Chunked<Vec<SourceIndex>> {
        // Build a mapping to mesh indices in their respective slices.
        let mut material_source = Vec::new();
        material_source.extend((0..solids.len()).map(|i| SourceIndex::Solid(i)));
        material_source.extend((0..shells.len()).map(|i| SourceIndex::Shell(i)));

        // Sort materials by material id.
        let to_mat_id = |k: &SourceIndex| match *k {
            SourceIndex::Solid(i) => solids[i].material.id,
            SourceIndex::Shell(i) => shells[i].material_id(),
        };
        material_source.sort_by_key(to_mat_id);

        // Assemble material source indices into a chunked array where each
        // chunk represents a unique id. This means some chunks are empty.
        let mut off = 0;
        let mut offsets = vec![0];
        for cur_id in material_source.iter().map(to_mat_id) {
            while offsets.len() - 1 < cur_id {
                offsets.push(off);
            }
            off += 1;
        }
        offsets.push(off);

        Chunked::from_offsets(offsets, material_source)
    }

    /// Helper function to compute a set of frictional contacts.
    fn build_frictional_contacts<T: Real>(
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
        frictional_contacts: &[(FrictionalContactParams, (usize, usize))],
    ) -> Vec<FrictionalContactConstraint<T>> {
        let material_source = Self::build_material_sources(solids, shells);

        // Convert frictional contact parameters into frictional contact constraints.
        // This function creates a frictional contact for every pair of matching material ids.
        // In other words if two objects share the same material id for which a frictional contact
        // is assigned, then at least two frictional contact constraints are created.

        frictional_contacts
            .iter()
            .flat_map(|&(params, (obj_id, coll_id))| {
                let material_source_obj = material_source.view().get(obj_id);
                let material_source_coll = material_source.view().get(coll_id);
                log::info!(
                    "Building constraints between materials: ({}, {})",
                    obj_id,
                    coll_id
                );
                material_source_obj
                    .into_iter()
                    .flat_map(move |material_source_obj| {
                        material_source_coll
                            .into_iter()
                            .flat_map(move |material_source_coll| {
                                SolverBuilder::build_contact_constraint(
                                    solids,
                                    shells,
                                    params,
                                    material_source_coll,
                                    material_source_obj,
                                )
                            })
                    })
            })
            .collect()
    }

    fn build_contact_constraint<'a, T: Real>(
        solids: &'a [TetMeshSolid],
        shells: &'a [TriMeshShell],
        mut params: FrictionalContactParams,
        material_source_coll: &'a [SourceIndex],
        material_source_obj: &'a [SourceIndex],
    ) -> impl Iterator<Item = FrictionalContactConstraint<T>> + 'a {
        let construct_friction_constraint =
            move |m0: SourceIndex, m1: SourceIndex, constraint| FrictionalContactConstraint {
                object_index: m0.into_source(params.use_fixed),
                collider_index: m1.into_source(params.use_fixed),
                constraint,
            };

        material_source_obj.into_iter().flat_map(move |&m0| {
            let object = match m0 {
                SourceIndex::Solid(i) => {
                    ContactSurface::deformable(&solids[i].surface(params.use_fixed).trimesh)
                }
                SourceIndex::Shell(i) => shells[i].contact_surface(),
            };

            material_source_coll.into_iter().flat_map(move |&m1| {
                let collider = match m1 {
                    SourceIndex::Solid(j) => {
                        ContactSurface::deformable(&solids[j].surface(params.use_fixed).trimesh)
                    }
                    SourceIndex::Shell(j) => shells[j].contact_surface(),
                };
                params.contact_type = ContactType::Point; // Linearized not supported for nl solvers.
                build_contact_constraint(object, collider, params)
                    .ok()
                    .map(|c| construct_friction_constraint(m0, m1, c))
                    .into_iter()
            })
        })
    }

    /// Helper function to initialize volume constraints from a set of solids.
    fn build_volume_constraints(
        solids: &[TetMeshSolid],
    ) -> Vec<(usize, RefCell<VolumeConstraint>)> {
        // Initialize volume constraint
        solids
            .iter()
            .enumerate()
            .filter(|&(_, solid)| solid.material.volume_preservation())
            .map(|(idx, solid)| (idx, RefCell::new(VolumeConstraint::new(&solid.tetmesh))))
            .collect()
    }

    /// Helper function to build a global array of vertex data. This is stacked
    /// vertex positions and velocities used by the solver to deform all meshes
    /// at the same time.
    fn build_state<T: Real>(
        solids: Vec<TetMeshSolid>,
        shells: Vec<TriMeshShell>,
    ) -> Result<State<T, ad::FT<T>>, Error> {
        // Generalized coordinates and their derivatives.
        let mut dof = Chunked::<Chunked3<GeneralizedState<Vec<T>, Vec<T>>>>::default();

        // Vertex position and velocities.
        let mut vtx = Chunked::<Chunked3<VertexState<Vec<T>, Vec<T>>>>::default();

        for TetMeshSolid {
            tetmesh: ref mesh, ..
        } in solids.iter()
        {
            // Get previous position vector from the tetmesh.
            dof.push_iter(
                GeneralizedState {
                    q: mesh.vertex_positions(),
                    dq: mesh.attrib_as_slice::<VelType, VertexIndex>(VELOCITY_ATTRIB)?,
                }
                .iter()
                .map(|dof| GeneralizedState {
                    q: *dof.q.as_tensor().cast::<T>().as_data(),
                    dq: *dof.dq.as_tensor().cast::<T>().as_data(),
                }),
            );
            vtx.push_iter(std::iter::empty());
        }

        for TriMeshShell {
            trimesh: ref mesh,
            data,
        } in shells.iter()
        {
            match data {
                ShellData::Rigid { cm, .. } => {
                    let mesh_vel =
                        mesh.direct_attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;

                    let mesh_pos = mesh.vertex_positions();

                    let translation: [f64; 3] = *cm.as_data();
                    // NOTE: if rotation is not [0.0; 3], then we must unrotate when computing rt
                    // below.
                    let q = [translation, [0.0; 3]];

                    // Average linear velocity.
                    let mut linear = Vector3::zero();
                    for &v in mesh_vel.iter() {
                        linear += v.into_tensor();
                    }
                    linear /= mesh_vel.len() as f64;

                    // Compute least squares angular velocity.
                    // Compute r^T matrix. The matrix of rigid body positions.
                    let rt = na::DMatrix::from_iterator(
                        3,
                        3 * mesh_pos.len(),
                        mesh_pos.iter().cloned().flat_map(|p| {
                            let pskew = (p.into_tensor() - *cm).skew().into_data();
                            (0..3).flat_map(move |r| (0..3).map(move |c| pskew[r][c]))
                        }),
                    );

                    // Collect all velocities
                    let v = na::DVector::from_iterator(
                        3 * mesh_vel.len(),
                        mesh_vel.iter().cloned().flat_map(|v| {
                            // Subtract linear part
                            let w = v.into_tensor() - linear;
                            (0..3).map(move |i| w[i])
                        }),
                    );

                    // Solve normal equations:
                    let rtv = -&rt * v;
                    let rtr = &rt * rt.transpose();
                    let angular = rtr
                        .qr()
                        .solve(&rtv)
                        .unwrap_or_else(|| na::DVector::zeros(3));
                    let angular = Vector3::new([angular[0], angular[1], angular[2]]);

                    let dq = [linear.into_data(), angular.into_data()];
                    dof.push_iter(q.iter().zip(dq.iter()).map(|(q, dq)| GeneralizedState {
                        q: *q.as_tensor().cast::<T>().as_data(),
                        dq: *dq.as_tensor().cast::<T>().as_data(),
                    }));

                    vtx.push_iter(mesh.vertex_position_iter().zip(mesh_vel.iter()).map(
                        |(&pos, &vel)| VertexState {
                            pos: *pos.as_tensor().cast::<T>().as_data(),
                            vel: *vel.as_tensor().cast::<T>().as_data(),
                        },
                    ));
                }
                ShellData::Soft { .. } => {
                    let mesh_prev_vel =
                        mesh.direct_attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    dof.push_iter(mesh.vertex_position_iter().zip(mesh_prev_vel.iter()).map(
                        |(&pos, &vel)| GeneralizedState {
                            q: *pos.as_tensor().cast::<T>().as_data(),
                            dq: *vel.as_tensor().cast::<T>().as_data(),
                        },
                    ));
                    vtx.push_iter(std::iter::empty());
                }
                ShellData::Fixed { .. } => {
                    dof.push_iter(std::iter::empty());
                    let mesh_vel =
                        mesh.direct_attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    vtx.push_iter(mesh.vertex_position_iter().zip(mesh_vel.iter()).map(
                        |(&pos, &vel)| VertexState {
                            pos: *pos.as_tensor().cast::<T>().as_data(),
                            vel: *vel.as_tensor().cast::<T>().as_data(),
                        },
                    ));
                }
            }
        }

        let num_meshes = solids.len() + shells.len();
        let dof_state = Chunked::from_offsets(vec![0, solids.len(), num_meshes], dof);
        let vtx_state = Chunked::from_offsets(vec![0, solids.len(), num_meshes], vtx);

        let dof = dof_state.clone().map_storage(|dof| GeneralizedCoords {
            prev: dof.clone(),
            cur: dof,
        });
        let vtx = vtx_state.clone().map_storage(|vtx| Vertex {
            prev: vtx.clone(),
            cur: vtx.clone(),
        });
        let vtx_ws_next = vtx_state.clone().map_storage(|vtx| {
            let n = vtx.len();
            let state_ad = VertexState {
                pos: vtx.pos.iter().map(|&x| ad::F::cst(x)).collect(),
                vel: vtx.vel.iter().map(|&x| ad::F::cst(x)).collect(),
            };
            VertexWorkspace {
                state: vtx,
                state_ad,
                grad: vec![T::zero(); n],
                lambda: vec![T::zero(); n],
                vfc: vec![T::zero(); n],
                lambda_ad: vec![ad::F::zero(); n],
                vfc_ad: vec![ad::F::zero(); n],
            }
        });

        let dof_ws_next = dof_state.clone().map_storage(|dof| {
            let m = dof.len();
            GeneralizedWorkspace {
                state: dof.clone(),
                state_ad: GeneralizedState {
                    q: dof.q.iter().cloned().map(ad::F::cst).collect(),
                    dq: dof.dq.iter().cloned().map(ad::F::cst).collect(),
                },
                r_ad: vec![ad::F::zero(); m],
            }
        });

        Ok(State {
            dof: dof.clone(),
            vtx,

            workspace: RefCell::new(WorkspaceData {
                dof: dof_ws_next,
                vtx: vtx_ws_next,
            }),

            solids,
            shells,
        })
    }

    /// Helper function to build an array of solids with associated material properties and attributes.
    fn build_solids(solids: Vec<(TetMesh, SolidMaterial)>) -> Result<Vec<TetMeshSolid>, Error> {
        // Equip `TetMesh`es with physics parameters, making them bona-fide solids.
        let mut out = Vec::new();
        for (tetmesh, material) in solids.into_iter() {
            // Prepare deformable solid for simulation.
            out.push(Self::prepare_solid_attributes(TetMeshSolid::new(
                tetmesh, material,
            ))?);
        }

        Ok(out)
    }

    /// Helper function to build a list of shells with associated material properties and attributes.
    fn build_shells(
        soft_shells: Vec<(PolyMesh, SoftShellMaterial)>,
        rigid_shells: Vec<(PolyMesh, RigidMaterial)>,
        fixed_shells: Vec<(PolyMesh, FixedMaterial)>,
    ) -> Result<Vec<TriMeshShell>, Error> {
        // Equip `PolyMesh`es with physics parameters, making them bona-fide shells.
        let mut out = Vec::new();
        for (polymesh, material) in soft_shells.into_iter() {
            let trimesh = TriMesh::from(polymesh);
            // Prepare shell for simulation.
            out.push(TriMeshShell::soft(trimesh, material).with_simulation_attributes()?)
        }
        for (polymesh, material) in rigid_shells.into_iter() {
            let trimesh = TriMesh::from(polymesh);
            // Prepare shell for simulation.
            out.push(TriMeshShell::rigid(trimesh, material).with_simulation_attributes()?)
        }
        for (polymesh, material) in fixed_shells.into_iter() {
            let trimesh = TriMesh::from(polymesh);
            // Prepare shell for simulation.
            out.push(TriMeshShell::fixed(trimesh, material).with_simulation_attributes()?)
        }

        Ok(out)
    }

    /// Helper function to compute the maximum element mass in the problem.
    fn compute_max_element_mass(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
        use std::cmp::Ordering;
        let mut mass = solids
            .iter()
            .map(|TetMeshSolid { ref tetmesh, .. }| {
                tetmesh
                    .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .ok()
                    .and_then(|ref_vol_iter| {
                        tetmesh
                            .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                            .ok()
                            .and_then(|density_iter| {
                                ref_vol_iter
                                    .zip(density_iter)
                                    .map(|(vol, &density)| vol * f64::from(density))
                                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                            })
                    })
                    .unwrap_or(0.0)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap_or(0.0);

        mass = mass.max(
            shells
                .iter()
                .map(
                    |TriMeshShell {
                         ref trimesh,
                         ref data,
                         ..
                     }| {
                        match data {
                            ShellData::Rigid { mass, .. } => *mass,
                            ShellData::Soft { .. } => trimesh
                                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                                .expect("Missing reference area attribute")
                                .zip(
                                    trimesh
                                        .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                                        .expect("Missing density attribute on trimesh"),
                                )
                                .map(|(area, &density)| area * f64::from(density))
                                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                                .unwrap_or(0.0),
                            _ => 0.0,
                        }
                    },
                )
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                .unwrap_or(0.0),
        );

        mass
    }

    /// Helper function to compute the minimum element mass in the problem.
    fn compute_min_element_mass(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
        use std::cmp::Ordering;
        let mut mass = solids
            .iter()
            .map(|TetMeshSolid { ref tetmesh, .. }| {
                tetmesh
                    .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .ok()
                    .and_then(|ref_vol_iter| {
                        tetmesh
                            .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                            .ok()
                            .and_then(|density_iter| {
                                ref_vol_iter
                                    .zip(density_iter)
                                    .map(|(vol, &density)| vol * f64::from(density))
                                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                            })
                    })
                    .unwrap_or(1.0)
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap_or(1.0);

        mass = mass.min(
            shells
                .iter()
                .map(
                    |TriMeshShell {
                         ref trimesh,
                         ref data,
                         ..
                     }| {
                        match data {
                            ShellData::Rigid { mass, .. } => *mass,
                            ShellData::Soft { .. } => trimesh
                                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                                .expect("Missing reference area attribute")
                                .zip(
                                    trimesh
                                        .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                                        .expect("Missing density attribute on trimesh"),
                                )
                                .map(|(area, &density)| area * f64::from(density))
                                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                                .unwrap_or(1.0),
                            _ => 1.0,
                        }
                    },
                )
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                .unwrap_or(1.0),
        );

        mass
    }

    /// Helper to compute max object size (diameter) over all deformable or rigid objects.
    /// This is used for normalizing the problem.
    fn compute_max_size(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
        use geo::ops::*;
        let mut bbox = geo::bbox::BBox::<f64>::empty();

        for TetMeshSolid { ref tetmesh, .. } in solids.iter() {
            bbox.absorb(tetmesh.bounding_box());
        }

        for TriMeshShell {
            ref trimesh,
            ref data,
        } in shells.iter()
        {
            match data {
                ShellData::Rigid { .. } | ShellData::Soft { .. } => {
                    bbox.absorb(trimesh.bounding_box());
                }
                _ => {}
            }
        }

        bbox.diameter()
    }

    /// Helper to compute max element size. This is used for normalizing tolerances.
    fn compute_max_element_size(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
        let mut max_size = 0.0_f64;

        for TetMeshSolid { ref tetmesh, .. } in solids.iter() {
            max_size = max_size.max(
                tetmesh
                    .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .expect("Reference volume missing")
                    .max_by(|a, b| a.partial_cmp(b).expect("Degenerate tetrahedron detected"))
                    .expect("Given TetMesh is empty")
                    .cbrt(),
            );
        }

        for TriMeshShell { ref trimesh, .. } in shells.iter() {
            max_size = max_size.max(
                trimesh
                    .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                    .ok()
                    .and_then(|area_iter| {
                        area_iter
                            .max_by(|a, b| a.partial_cmp(b).expect("Degenerate triangle detected"))
                    })
                    .unwrap_or(&0.0)
                    .sqrt(),
            );
        }

        max_size
    }

    /// Helper function to compute the maximum elastic modulus of all given meshes.
    /// This aids in figuring out the correct scaling for the problem.
    fn compute_max_modulus(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> Result<f32, Error> {
        let mut max_modulus = 0.0_f32;

        for TetMeshSolid { ref tetmesh, .. } in solids.iter() {
            max_modulus = max_modulus
                .max(
                    *tetmesh
                        .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)?
                        .max_by(|a, b| a.partial_cmp(b).expect("Invalid First Lame Parameter"))
                        .expect("Given TetMesh is empty"),
                )
                .max(
                    *tetmesh
                        .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)?
                        .max_by(|a, b| a.partial_cmp(b).expect("Invalid Shear Modulus"))
                        .expect("Given TetMesh is empty"),
                );
        }

        for TriMeshShell { ref trimesh, data } in shells.iter() {
            match data {
                ShellData::Soft { .. } => {
                    max_modulus = max_modulus
                        .max(
                            *trimesh
                                .attrib_iter::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)?
                                .max_by(|a, b| {
                                    a.partial_cmp(b).expect("Invalid First Lame Parameter")
                                })
                                .expect("Given TriMesh is empty"),
                        )
                        .max(
                            *trimesh
                                .attrib_iter::<MuType, FaceIndex>(MU_ATTRIB)?
                                .max_by(|a, b| a.partial_cmp(b).expect("Invalid Shear Modulus"))
                                .expect("Given TriMesh is empty"),
                        );
                }
                _ => {}
            }
        }

        Ok(max_modulus)
    }

    /// Helper function to compute the maximum elastic bending stiffness.
    fn compute_max_bending_stiffness(shells: &[TriMeshShell]) -> f64 {
        let mut max_bending_stiffness = 0.0_f64;

        for TriMeshShell { data, .. } in shells.iter() {
            match data {
                ShellData::Soft {
                    interior_edge_bending_stiffness,
                    ..
                } => {
                    max_bending_stiffness = max_bending_stiffness.max(
                        interior_edge_bending_stiffness
                            .iter()
                            .cloned()
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                            .unwrap_or(0.0),
                    )
                }
                _ => {}
            }
        }

        max_bending_stiffness
    }

    /// Helper to compute min element size. This is used for normalizing tolerances.
    fn compute_min_element_size(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
        let mut min_size = f64::INFINITY;

        for TetMeshSolid { ref tetmesh, .. } in solids.iter() {
            min_size = min_size.min(
                tetmesh
                    .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .expect("Reference volume missing")
                    .min_by(|a, b| a.partial_cmp(b).expect("Degenerate tetrahedron detected"))
                    .expect("Given TetMesh is empty")
                    .cbrt(),
            );
        }

        for TriMeshShell { ref trimesh, .. } in shells.iter() {
            min_size = min_size.min(
                trimesh
                    .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                    .ok()
                    .and_then(|area_iter| {
                        area_iter
                            .min_by(|a, b| a.partial_cmp(b).expect("Degenerate triangle detected"))
                    })
                    .unwrap_or(&f64::INFINITY)
                    .sqrt(),
            );
        }

        min_size
    }

    /// Helper function to compute the minimum elastic modulus of all given meshes.
    /// This aids in figuring out the correct scaling for the problem.
    fn compute_min_modulus(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> Result<f32, Error> {
        let mut min_modulus = f32::INFINITY;

        let ignore_zero = |x| if x == 0.0 { f32::INFINITY } else { x };

        for TetMeshSolid { ref tetmesh, .. } in solids.iter() {
            min_modulus = min_modulus
                .min(ignore_zero(
                    *tetmesh
                        .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)?
                        .min_by(|a, b| a.partial_cmp(b).expect("Invalid First Lame Parameter"))
                        .unwrap_or(&f32::INFINITY),
                ))
                .min(ignore_zero(
                    *tetmesh
                        .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)?
                        .min_by(|a, b| a.partial_cmp(b).expect("Invalid Shear Modulus"))
                        .unwrap_or(&f32::INFINITY),
                ));
        }

        for TriMeshShell { ref trimesh, data } in shells.iter() {
            match data {
                ShellData::Soft { .. } => {
                    min_modulus = min_modulus
                        .min(ignore_zero(
                            *trimesh
                                .attrib_iter::<LambdaType, FaceIndex>(LAMBDA_ATTRIB)?
                                .min_by(|a, b| {
                                    a.partial_cmp(b).expect("Invalid First Lame Parameter")
                                })
                                .unwrap_or(&f32::INFINITY),
                        ))
                        .min(ignore_zero(
                            *trimesh
                                .attrib_iter::<MuType, FaceIndex>(MU_ATTRIB)?
                                .min_by(|a, b| a.partial_cmp(b).expect("Invalid Shear Modulus"))
                                .unwrap_or(&f32::INFINITY),
                        ));
                }
                _ => {}
            }
        }

        Ok(min_modulus)
    }

    /// Helper function to compute the minimum elastic bending stiffness.
    fn compute_min_bending_stiffness(shells: &[TriMeshShell]) -> f64 {
        let mut min_bending_stiffness = f64::INFINITY;

        for TriMeshShell { data, .. } in shells.iter() {
            match data {
                ShellData::Soft {
                    interior_edge_bending_stiffness,
                    ..
                } => {
                    min_bending_stiffness = min_bending_stiffness.min(
                        interior_edge_bending_stiffness
                            .iter()
                            .cloned()
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                            .unwrap_or(f64::INFINITY),
                    )
                }
                _ => {}
            }
        }

        min_bending_stiffness
    }

    pub(crate) fn build_problem<T: Real>(&self) -> Result<NLProblem<T>, Error> {
        let SolverBuilder {
            sim_params: params,
            solids,
            soft_shells,
            rigid_shells,
            fixed_shells,
            frictional_contacts,
        } = self.clone();

        let solids = Self::build_solids(solids)?;
        let shells = Self::build_shells(soft_shells, rigid_shells, fixed_shells)?;

        let state = Self::build_state(solids, shells)?;

        let volume_constraints = Self::build_volume_constraints(&state.solids);

        let gravity = [
            f64::from(params.gravity[0]),
            f64::from(params.gravity[1]),
            f64::from(params.gravity[2]),
        ];

        let time_step = f64::from(params.time_step.unwrap_or(0.0f32));

        if frictional_contacts.iter().any(|(_, (i, j))| i == j) {
            return Err(Error::UnimplementedFeature {
                description: String::from("Self contacts"),
            });
        }

        let frictional_contacts_ad = Self::build_frictional_contacts::<ad::FT<T>>(
            &state.solids,
            &state.shells,
            &frictional_contacts,
        );
        let frictional_contacts = Self::build_frictional_contacts::<T>(
            &state.solids,
            &state.shells,
            &frictional_contacts,
        );

        // Compute maxes

        let max_size = Self::compute_max_size(&state.solids, &state.shells);

        let max_element_size = Self::compute_max_element_size(&state.solids, &state.shells);

        // The following scales have units of force (N).
        let max_element_modulus_scale = Self::compute_max_modulus(&state.solids, &state.shells)?
            as f64
            * max_element_size
            * max_element_size;

        let max_element_mass = Self::compute_max_element_mass(&state.solids, &state.shells);

        let max_element_gravity_scale = max_element_mass * gravity.as_tensor().norm();

        let max_element_inertia_scale = if time_step > 0.0 {
            max_element_mass * max_element_size / (time_step * time_step)
        } else {
            0.0
        };

        let max_element_bending_scale =
            Self::compute_max_bending_stiffness(&state.shells) / max_element_size;

        log::trace!("max_element_modulus_scale = {}", max_element_modulus_scale);
        log::trace!("max_element_inertia_scale = {}", max_element_inertia_scale);
        log::trace!("max_element_gravity_scale = {}", max_element_gravity_scale);
        log::trace!("max_element_bending_scale = {}", max_element_bending_scale);

        // Determine the most likely dominant force.
        let mut max_scale = max_element_modulus_scale
            .max(max_element_inertia_scale)
            .max(max_element_gravity_scale)
            .max(max_element_bending_scale);

        // max_scale is a denominator. Ensure that it is never zero.
        if max_scale == 0.0 {
            log::warn!("All scaling factors are zero");
            max_scale = 1.0;
        }

        // Compute mins

        let ignore_zero = |x| if x == 0.0 { f64::INFINITY } else { x };

        let min_element_size = Self::compute_min_element_size(&state.solids, &state.shells);

        let min_element_modulus_scale =
            ignore_zero(Self::compute_min_modulus(&state.solids, &state.shells)? as f64)
                * min_element_size
                * min_element_size;

        let min_element_mass = Self::compute_min_element_mass(&state.solids, &state.shells);

        let min_element_gravity_scale = ignore_zero(min_element_mass * gravity.as_tensor().norm());

        let min_element_inertia_scale = if time_step > 0.0 {
            min_element_mass * min_element_size / (time_step * time_step)
        } else {
            f64::INFINITY
        };

        let min_element_bending_scale =
            Self::compute_min_bending_stiffness(&state.shells) / min_element_size;

        log::trace!("min_element_modulus_scale = {}", min_element_modulus_scale);
        log::trace!("min_element_inertia_scale = {}", min_element_inertia_scale);
        log::trace!("min_element_gravity_scale = {}", min_element_gravity_scale);
        log::trace!("min_element_bending_scale = {}", min_element_bending_scale);

        // Determine the least dominant force.
        let min_scale = min_element_modulus_scale
            .min(min_element_inertia_scale)
            .min(min_element_gravity_scale)
            .min(min_element_bending_scale);

        let num_variables = state.dof.storage().len();

        Ok(NLProblem {
            state,
            lambda: RefCell::new(Vec::new()),
            lambda_ad: RefCell::new(Vec::new()),
            kappa: 1.0 / params.contact_tolerance as f64,
            delta: params.contact_tolerance as f64,
            frictional_contacts,
            frictional_contacts_ad,
            volume_constraints,
            time_step,
            gravity,
            iterations: 0,
            initial_residual_error: std::f64::INFINITY,
            iter_counter: RefCell::new(0),
            max_size,
            max_element_force_scale: max_scale,
            min_element_force_scale: min_scale,
        })
    }

    /// Build the simulation solver.
    pub fn build<T: Real64>(&self) -> Result<Solver<impl NLSolver<NLProblem<T>, T>, T>, Error> {
        self.build_with_interrupt(|| true)
    }

    /// Build the simulation solver with a given interrupter.
    ///
    /// This version of build allows the caller to conditionally interrupt the solver.
    pub fn build_with_interrupt<T: Real64>(
        &self,
        mut interrupt_checker: impl FnMut() -> bool + Send + 'static,
    ) -> Result<Solver<impl NLSolver<NLProblem<T>, T>, T>, Error> {
        let problem = self.build_problem::<T>()?;

        let num_variables = problem.num_variables();

        // Setup Ipopt parameters using the input simulation params.
        let params = self.sim_params.clone();

        let r_scale = problem.min_element_force_scale;
        let r_tol = params.tolerance * r_scale as f32;

        let x_tol = params.tolerance;

        log::info!("Simulation Parameters:\n{:#?}", params);
        log::info!("r_tol: {:?}", r_tol);
        log::info!("x_tol: {:?}", x_tol);
        log::trace!("r-scale: {:?}", r_scale);

        // Construct the non-linear equation solver.
        let solver = MCPSolver::newton(
            problem,
            NewtonParams {
                r_tol,
                x_tol,
                max_iter: params.max_iterations,
                linsolve_tol: params.linsolve_tolerance,
                linsolve_max_iter: params.max_linsolve_iterations,
                line_search: params.line_search,
            },
            Box::new(move |_| interrupt_checker()),
            Box::new(|_| true),
        );

        Ok(Solver {
            solver,
            sim_params: params,
            max_step: 0.0,
            solution: vec![T::zero(); num_variables],
        })
    }

    /// Compute signed volume for reference elements in the given `TetMesh`.
    fn compute_ref_tet_signed_volumes(mesh: &mut TetMesh) -> Result<Vec<f64>, Error> {
        use rayon::iter::Either;
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;
        let ref_volumes: Vec<f64> = ref_pos
            .chunks_exact(4)
            .map(|tet| ref_tet(tet).signed_volume())
            .collect();
        let inverted: Vec<_> = ref_volumes
            .iter()
            // Ignore fixed elements, since they will not be part of the simulation anyways.
            .zip(Either::from(
                mesh.attrib_iter::<FixedIntType, CellIndex>(FIXED_ATTRIB)
                    .map(|i| i.cloned())
                    .map_err(|_| std::iter::repeat(0)),
            ))
            .enumerate()
            .filter_map(|(i, (&v, fixed))| {
                if v <= 0.0 && fixed == 0 {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if !inverted.is_empty() {
            return Err(Error::InvertedReferenceElement { inverted });
        }
        Ok(ref_volumes)
    }

    /// Compute shape matrix inverses for reference elements in the given `TetMesh`.
    fn compute_ref_tet_shape_matrix_inverses(
        mesh: &mut TetMesh,
    ) -> Result<Vec<Matrix3<f64>>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;
        // Compute reference shape matrix inverses
        Ok(ref_pos
            .chunks_exact(4)
            .map(|tet| {
                let ref_shape_matrix = ref_tet(tet).shape_matrix();
                // We assume that ref_shape_matrices are non-singular.
                // This should have been checked in `compute_ref_tet_signed_volumes`.
                Matrix3::new(ref_shape_matrix).inverse().unwrap()
            })
            .collect())
    }

    /// A helper function to populate the vertex reference position attribute.
    pub(crate) fn prepare_cell_vertex_ref_pos_attribute(mesh: &mut TetMesh) -> Result<(), Error> {
        let mut ref_pos = vec![[0.0; 3]; mesh.num_cell_vertices()];
        let pos = if let Ok(vtx_ref_pos) =
            mesh.attrib_as_slice::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
        {
            // There is a reference attribute on the vertices themselves, just transfer these
            // to cell vertices instead of using mesh position.
            vtx_ref_pos.to_vec()
        } else {
            mesh.vertex_position_iter()
                .map(|&x| Vector3::new(x).cast::<f32>().into())
                .collect()
        };

        for (cell_idx, cell) in mesh.cell_iter().enumerate() {
            for i in 0..4 {
                let cell_vtx_idx: usize = mesh.cell_vertex(cell_idx, i).unwrap().into();
                for j in 0..3 {
                    ref_pos[cell_vtx_idx][j] = pos[cell[i]][j];
                }
            }
        }

        mesh.attrib_or_add_data::<RefPosType, CellVertexIndex>(
            REFERENCE_CELL_VERTEX_POS_ATTRIB,
            ref_pos.as_slice(),
        )?;
        Ok(())
    }

    ///// A helper function to populate the edge reference angles used to compute bending energies.
    //#[unroll_for_loops]
    //pub(crate) fn prepare_edge_ref_angle_attribute(mesh: &mut TriMesh) -> Result<(), Error>
    //{
    //    for (face_idx, face) in mesh.edge_iter().enumerate() {
    //        for i in 0..3 {
    //            let face_vtx_idx: usize = mesh.face_vertex(face_idx, i).unwrap().into();
    //            for j in 0..3 {
    //                ref_pos[face_vtx_idx][j] = pos[face[i]][j];
    //            }
    //        }
    //    }

    //    mesh.attrib_or_add_data::<RefAngleType, EdgeIndex>(
    //        REFERENCE_POSITION_ATTRIB,
    //        ref_pos.as_slice(),
    //    )?;
    //    Ok(())
    //}

    /// Compute vertex masses on the given solid. The solid is assumed to have
    /// volume and density attributes already.
    pub fn compute_solid_vertex_masses(solid: &mut TetMeshSolid) {
        let tetmesh = &mut solid.tetmesh;
        let mut masses = vec![0.0; tetmesh.num_vertices()];

        for (&vol, density, cell) in zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tetmesh.cell_iter()
        ) {
            for i in 0..4 {
                masses[cell[i]] += 0.25 * vol * density;
            }
        }

        tetmesh
            .set_attrib_data::<MassType, VertexIndex>(MASS_ATTRIB, &masses)
            .unwrap();
    }

    pub(crate) fn prepare_deformable_tetmesh_attributes(mesh: &mut TetMesh) -> Result<(), Error> {
        Self::prepare_cell_vertex_ref_pos_attribute(mesh)?;
        let ref_volumes = Self::compute_ref_tet_signed_volumes(mesh)?;
        mesh.set_attrib_data::<RefVolType, CellIndex>(
            REFERENCE_VOLUME_ATTRIB,
            ref_volumes.as_slice(),
        )?;

        let ref_shape_mtx_inverses = Self::compute_ref_tet_shape_matrix_inverses(mesh)?;
        mesh.set_attrib_data::<_, CellIndex>(
            REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
            ref_shape_mtx_inverses.as_slice(),
        )?;
        Ok(())
    }

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    pub(crate) fn prepare_solid_attributes(mut solid: TetMeshSolid) -> Result<TetMeshSolid, Error> {
        solid.init_source_index_attribute()?;

        solid.init_deformable_vertex_attributes()?;

        solid.init_fixed_element_attribute()?;

        Self::prepare_deformable_tetmesh_attributes(&mut solid.tetmesh)?;

        {
            // Add elastic strain energy and elastic force attributes.
            // These will be computed at the end of the time step.
            solid
                .tetmesh
                .set_attrib::<StrainEnergyType, CellIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
        }

        // Below we prepare attributes that give elasticity and density parameters. If such were
        // already provided on the mesh, then any given global parameters are ignored. This
        // behaviour is justified because variable material properties are most likely more
        // accurate and probably determined from a data driven method.

        solid.init_elasticity_attributes()?;

        solid.init_density_attribute()?;

        // Compute vertex masses.
        Self::compute_solid_vertex_masses(&mut solid);

        Ok(solid)
    }
}

/// Finite element engine.
pub struct Solver<S, T> {
    /// Non-linear solver.
    solver: S,
    /// Simulation parameters. This is kept around for convenience.
    sim_params: SimParams,
    /// Maximal displacement length.
    ///
    /// Used to limit displacement which is necessary in contact scenarios
    /// because it defines how far a step we can take before the constraint
    /// Jacobian sparsity pattern changes. If zero, then no limit is applied but
    /// constraint Jacobian is kept sparse.
    max_step: f64,
    /// Structured solution to the problem in the solver.
    ///
    /// This is also used as warm start for subsequent steps.
    solution: Vec<T>,
}

impl<S, T> Solver<S, T>
where
    T: Real64 + na::RealField,
    S: NLSolver<NLProblem<T>, T>,
{
    /// Sets the interrupt checker to be called at every outer iteration.
    pub fn set_coarse_interrupter(&self, mut interrupted: impl FnMut() -> bool + Send + 'static) {
        *self.solver.outer_callback().borrow_mut() = Box::new(move |_| !interrupted());
    }

    /// Sets the interrupt checker to be called at every inner iteration.
    pub fn set_fine_interrupter(&self, mut interrupted: impl FnMut() -> bool + Send + 'static) {
        *self.solver.inner_callback().borrow_mut() = Box::new(move |_| !interrupted());
    }

    /// If the time step was not specified or specified to be zero, then this function will return
    /// zero.
    pub fn time_step(&self) -> f64 {
        self.sim_params.time_step.unwrap_or(0.0).into()
    }
    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &NLProblem<T> {
        self.solver.problem()
    }

    /// Get a mutable reference to the underlying problem.
    fn problem_mut(&mut self) -> &mut NLProblem<T> {
        self.solver.problem_mut()
    }

    /// Get a slice of solid objects represented in this solver.
    pub fn solids(&self) -> &[TetMeshSolid] {
        &self.problem().state.solids
    }

    /// Get a slice of shell objects represented in this solver.
    pub fn shells(&self) -> &[TriMeshShell] {
        &self.problem().state.shells
    }

    /// Get an immutable borrow for the underlying `TetMeshSolid` at the given index.
    pub fn solid(&self, index: usize) -> &TetMeshSolid {
        &self.problem().state.solids[index]
    }

    /// Get a mutable borrow for the underlying `TetMeshSolid` at the given index.
    pub fn solid_mut(&mut self, index: usize) -> &mut TetMeshSolid {
        &mut self.problem_mut().state.solids[index]
    }

    /// Get an immutable borrow for the underlying `TriMeshShell` at the given index.
    pub fn shell(&self, index: usize) -> &TriMeshShell {
        &self.problem().state.shells[index]
    }

    /// Get a mutable borrow for the underlying `TriMeshShell` at the given index.
    pub fn shell_mut(&mut self, index: usize) -> &mut TriMeshShell {
        &mut self.problem_mut().state.shells[index]
    }

    /// Get simulation parameters.
    pub fn params(&self) -> SimParams {
        self.sim_params.clone()
    }

    /// Update the maximal displacement allowed. If zero, no limit is applied.
    pub fn update_max_step(&mut self, step: f64) {
        self.max_step = step;
        self.problem_mut().update_max_step(step);
    }

    pub fn update_radius_multiplier(&mut self, rad: f64) {
        self.problem_mut().update_radius_multiplier(rad);
    }

    /// Update the solid meshes with the given points.
    pub fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.problem_mut().update_solid_vertices(pts)
    }

    /// Update the shell meshes with the given points.
    pub fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.problem_mut().update_shell_vertices(pts)
    }

    /// Update the `mesh` and `prev_pos` with the current solution.
    fn commit_solution(&mut self, relax_max_step: bool) {
        {
            let and_velocity = !self.sim_params.clear_velocity;
            let Self {
                solver, solution, ..
            } = self;

            // Advance internal state (positions and velocities) of the problem.
            solver.problem_mut().advance(&solution, and_velocity);
        }

        // Reduce max_step for next iteration if the solution was a good one.
        if relax_max_step {
            let dt = self.time_step();
            if let Some(radius) = self.problem().min_contact_radius() {
                let step =
                    inf_norm(self.solution.iter().cloned()) * if dt > 0.0 { dt } else { 1.0 };
                let new_max_step = (step.to_f64().unwrap() - radius).max(self.max_step * 0.5);
                if self.max_step != new_max_step {
                    log::info!(
                        "Relaxing max step from {} to {}",
                        self.max_step,
                        new_max_step
                    );
                    self.max_step = new_max_step;
                    self.problem_mut().update_max_step(new_max_step);
                }
            }
        }
    }

    /// Revert previously committed solution. We just advance in the opposite direction.
    fn revert_solution(&mut self) {
        self.problem_mut().revert_prev_step();
    }

    fn initial_residual_error(&self) -> f64 {
        self.problem().initial_residual_error
    }

    //fn save_current_active_constraint_set(&mut self) {
    //    let Solver {
    //        ref solver,
    //        ref mut old_active_constraint_set,
    //        ..
    //    } = self;
    //    old_active_constraint_set.clear();
    //    solver
    //        .solver_data()
    //        .problem
    //        .compute_active_constraint_set(old_active_constraint_set);
    //}

    //fn remap_warm_start(&mut self) {
    //    let Solver {
    //        solver,
    //        old_active_constraint_set,
    //        ..
    //    } = self;

    //    solver
    //        .solver_data_mut()
    //        .problem
    //        .remap_warm_start(old_active_constraint_set.view());
    //}

    fn all_contacts_linear(&self) -> bool {
        self.problem().all_contacts_linear()
    }
}

impl<S, T> Solver<S, T>
where
    T: Real64 + na::RealField,
    S: NLSolver<NLProblem<T>, T>,
{
    /// Run the non-linear solver on one time step.
    pub fn step(&mut self) -> Result<SolveResult, Error> {
        let Self {
            sim_params,
            solver,
            solution,
            ..
        } = self;

        if sim_params.jacobian_test {
            solver.problem().check_jacobian(true);
        }

        solver.problem_mut().update_constraint_set();

        let mut contact_iterations = 5i32;

        // Loop to resolve all contacts.
        loop {
            let result = solver.solve_with(solution.as_mut_slice());
            log::trace!("Solve Result: {}", &result);
            match result.status {
                Status::Success | Status::MaximumIterationsExceeded => {
                    // Compute contact violation.
                    let constraint = solver
                        .problem_mut()
                        .contact_constraint(solution.as_slice())
                        .into_storage();
                    //let mut orig_lambda = constraint.clone();
                    //let mut shifted_lambda = constraint.clone();
                    let smallest = constraint
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                        .copied()
                        .unwrap_or(T::zero())
                        .to_f64()
                        .unwrap();
                    //shifted_lambda.iter_mut().for_each(|x| *x -= smallest);
                    //compute_contact_penalty(
                    //    shifted_lambda.as_mut_slice(),
                    //    sim_params.contact_tolerance,
                    //);
                    //compute_contact_penalty(
                    //    orig_lambda.as_mut_slice(),
                    //    sim_params.contact_tolerance,
                    //);

                    let delta = sim_params.contact_tolerance as f64;
                    let denom: f64 = (compute_contact_penalty(0.0, delta as f32)
                        - sim_params.tolerance as f64 / solver.problem().kappa)
                        .min(0.5 * delta)
                        .max(f64::EPSILON);

                    let bump_ratio: f64 =
                        compute_contact_penalty(smallest, sim_params.contact_tolerance) / denom;
                    log::trace!("Bump ratio: {}", bump_ratio);
                    log::trace!("Kappa: {}", solver.problem().kappa);
                    let violation = solver
                        .problem_mut()
                        .contact_violation(constraint.storage().as_slice());
                    log::trace!("Contact violation: {}", violation);

                    contact_iterations -= 1;

                    if contact_iterations < 0 {
                        break Err(Error::NLSolveError {
                            result: SolveResult {
                                status: Status::MaximumContactIterationsExceeded,
                                ..result
                            },
                        });
                    }

                    if violation > T::zero() {
                        solver.problem_mut().kappa *= bump_ratio.max(2.0);
                        continue;
                    } else {
                        // Relax kappa
                        if solver.problem().kappa > 1.0 / sim_params.contact_tolerance as f64 {
                            solver.problem_mut().kappa /= 2.0;
                        }
                        self.commit_solution(false);
                        // On success, update the mesh with useful metrics.
                        //self.problem_mut().update_mesh_data();
                        break Ok(result);
                    }
                }
                _ => {
                    break Err(Error::NLSolveError { result });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the above function with moc data. This also serves as an example of usage.
    #[test]
    fn build_material_sources_test() {
        // Convenience functions
        let new_solid = |&id| TetMeshSolid::new(TetMesh::default(), SolidMaterial::new(id));
        let new_shell = |&id| TriMeshShell::soft(TriMesh::default(), SoftShellMaterial::new(id));
        let solid = |id| SourceIndex::Solid(id);
        let shell = |id| SourceIndex::Shell(id);

        // No shells
        let solids: Vec<_> = [1, 0, 0, 2].iter().map(new_solid).collect();
        let srcs = SolverBuilder::build_material_sources(&solids, &[]);
        assert_eq!(
            srcs,
            Chunked::from_offsets(
                vec![0, 2, 3, 4],
                vec![solid(1), solid(2), solid(0), solid(3)]
            )
        );

        // No solids
        let shells: Vec<_> = [3, 4, 3, 1].iter().map(new_shell).collect();
        let srcs = SolverBuilder::build_material_sources(&[], &shells);
        assert_eq!(
            srcs,
            Chunked::from_offsets(
                vec![0, 0, 1, 1, 3, 4],
                vec![shell(3), shell(0), shell(2), shell(1)]
            )
        );

        // Complex test
        let solids: Vec<_> = [0, 0, 0, 1, 2, 1, 0, 1].iter().map(new_solid).collect();
        let shells: Vec<_> = [3, 4, 0, 1, 0, 6, 10].iter().map(new_shell).collect();
        let srcs = SolverBuilder::build_material_sources(&solids, &shells);
        assert_eq!(
            srcs,
            Chunked::from_offsets(
                vec![0, 6, 10, 11, 12, 13, 13, 14, 14, 14, 14, 15],
                vec![
                    solid(0),
                    solid(1),
                    solid(2),
                    solid(6),
                    shell(2),
                    shell(4),
                    solid(3),
                    solid(5),
                    solid(7),
                    shell(3),
                    solid(4),
                    shell(0),
                    shell(1),
                    shell(5),
                    shell(6)
                ]
            )
        );
    }
}
