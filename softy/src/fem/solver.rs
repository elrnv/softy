use crate::attrib_defines::*;
use crate::constraints::*;
use crate::contact::*;
use crate::fem::problem::{FrictionalContactConstraint, Var};
use crate::objects::*;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use geo::ops::{ShapeMatrix, Volume};
use geo::prim::Tetrahedron;
use ipopt::{self, Ipopt, SolverData, SolverDataMut};
use num_traits::Zero;
use std::cell::RefCell;
use tensr::*;
use utils::zip;

use crate::inf_norm;

use super::{
    MuStrategy, NonLinearProblem, ObjectData, SimParams, Solution, SourceIndex, WorkspaceData,
};
use crate::{Error, PointCloud, PolyMesh, TetMesh, TriMesh};

/// Result from one inner simulation step.
#[derive(Clone, Debug, PartialEq)]
pub struct InnerSolveResult {
    /// Number of inner iterations in one step.
    pub iterations: u32,
    /// The value of the objective at the end of the step.
    pub objective_value: f64,
    /// Constraint values at the solution of the inner step.
    pub constraint_values: Vec<f64>,
}

/// Result from one simulation step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolveResult {
    /// Maximum number of inner iterations during one outer step.
    pub max_inner_iterations: u32,
    /// Number of total accumulated inner iterations.
    pub inner_iterations: u32,
    /// Number of outer iterations of the step.
    pub iterations: u32,
    /// The value of the objective at the end of the time step.
    pub objective_value: f64,
}

impl SolveResult {
    fn combine_inner_step_data(self, iterations: u32, objective_value: f64) -> SolveResult {
        SolveResult {
            // Aggregate max number of iterations.
            max_inner_iterations: iterations.max(self.max_inner_iterations),

            inner_iterations: iterations + self.inner_iterations,

            // Adding a new inner solve result means we have completed another inner solve: +1
            // outer iterations.
            iterations: self.iterations + 1,

            // The objective value of the solution is the objective value of the last inner solve.
            objective_value,
        }
    }
    /// Add an inner solve result to this solve result.
    fn combine_inner_result(self, inner: &InnerSolveResult) -> SolveResult {
        self.combine_inner_step_data(inner.iterations, inner.objective_value)
    }
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Iterations: {}\nObjective: {}\nMax Inner Iterations: {}",
            self.iterations, self.objective_value, self.max_inner_iterations
        )
    }
}
/// Get reference tetrahedron.
/// This routine assumes that there is a vertex attribute called `ref` of type `[f32;3]`.
pub fn ref_tet(ref_pos: &[RefPosType]) -> Tetrahedron<f64> {
    Tetrahedron::new([
        Vector3::new(ref_pos[0]).cast::<f64>().into(),
        Vector3::new(ref_pos[1]).cast::<f64>().into(),
        Vector3::new(ref_pos[2]).cast::<f64>().into(),
        Vector3::new(ref_pos[3]).cast::<f64>().into(),
    ])
}

#[derive(Clone, Debug)]
pub struct SolverBuilder {
    sim_params: SimParams,
    solids: Vec<(TetMesh, SolidMaterial)>,
    soft_shells: Vec<(PolyMesh, SoftShellMaterial)>,
    rigid_shells: Vec<(PolyMesh, RigidMaterial)>,
    fixed_shells: Vec<(PolyMesh, FixedMaterial)>,
    frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
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
    fn build_frictional_contacts(
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
        frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
    ) -> Vec<FrictionalContactConstraint> {
        let material_source = Self::build_material_sources(solids, shells);

        // Convert frictional contact parameters into frictional contact constraints.
        // This function creates a frictional contact for every pair of matching material ids.
        // In other words if two objects share the same material id for which a frictional contact
        // is assigned, then at least two frictional contact constraints are created.

        frictional_contacts
            .into_iter()
            .flat_map(|(params, (obj_id, coll_id))| {
                let material_source_obj = material_source.view().get(obj_id);
                let material_source_coll = material_source.view().get(coll_id);
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

    fn build_contact_constraint<'a>(
        solids: &'a [TetMeshSolid],
        shells: &'a [TriMeshShell],
        params: FrictionalContactParams,
        material_source_coll: &'a [SourceIndex],
        material_source_obj: &'a [SourceIndex],
    ) -> impl Iterator<Item = FrictionalContactConstraint> + 'a {
        let construct_friction_constraint = |m0, m1, constraint| FrictionalContactConstraint {
            object_index: m0,
            collider_index: m1,
            constraint,
        };

        material_source_obj.into_iter().flat_map(move |&m0| {
            let object = match m0 {
                SourceIndex::Solid(i) => Var::Variable(&solids[i].surface().trimesh),
                SourceIndex::Shell(i) => shells[i].tagged_mesh(),
            };

            material_source_coll.into_iter().flat_map(move |&m1| {
                let collider = match m1 {
                    SourceIndex::Solid(j) => Var::Variable(&solids[j].surface().trimesh),
                    SourceIndex::Shell(j) => shells[j].tagged_mesh(),
                };
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
    fn build_object_data(
        solids: Vec<TetMeshSolid>,
        shells: Vec<TriMeshShell>,
    ) -> Result<ObjectData, Error> {
        // Generalized coordinates and their derivatives.
        let mut prev_x = Chunked::<Chunked3<Vec<f64>>>::new();
        let mut prev_v = Chunked::<Chunked3<Vec<f64>>>::new();

        // Vertex position and velocities.
        let mut pos = Chunked::<Chunked3<Vec<f64>>>::new();
        let mut vel = Chunked::<Chunked3<Vec<f64>>>::new();

        for TetMeshSolid {
            tetmesh: ref mesh, ..
        } in solids.iter()
        {
            // Get previous position vector from the tetmesh.
            prev_x.push(mesh.vertex_positions().to_vec());
            prev_v
                .push(mesh.direct_attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?);
            pos.push(vec![]);
            vel.push(vec![]);
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
                    prev_x.push(vec![translation, [0.0; 3]]);

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

                    prev_v.push(vec![linear.into_data(), angular.into_data()]);

                    pos.push(mesh.vertex_positions().to_vec());
                    vel.push(mesh_vel);
                }
                ShellData::Soft { .. } => {
                    prev_x.push(mesh.vertex_positions().to_vec());
                    let mesh_prev_vel =
                        mesh.direct_attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    prev_v.push(mesh_prev_vel);
                    pos.push(vec![]);
                    vel.push(vec![]);
                }
                ShellData::Fixed { .. } => {
                    prev_x.push(vec![]);
                    prev_v.push(vec![]);
                    pos.push(mesh.vertex_positions().to_vec());
                    let mesh_vel =
                        mesh.direct_attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    vel.push(mesh_vel);
                }
            }
        }

        let num_meshes = solids.len() + shells.len();
        let prev_x = Chunked::from_offsets(vec![0, solids.len(), num_meshes], prev_x);
        let prev_v = Chunked::from_offsets(vec![0, solids.len(), num_meshes], prev_v);
        let pos = Chunked::from_offsets(vec![0, solids.len(), num_meshes], pos);
        let vel = Chunked::from_offsets(vec![0, solids.len(), num_meshes], vel);
        let mut grad = vel.clone();
        grad.view_mut()
            .into_storage()
            .iter_mut()
            .for_each(|v| *v = 0.0); // Zero out gradient.

        Ok(ObjectData {
            prev_x: prev_x.clone(),
            prev_v: prev_v.clone(),
            prev_prev_x: prev_x.clone().into_storage(),
            prev_prev_v: prev_v.clone().into_storage(),
            prev_pos: pos.clone(),
            prev_vel: vel.clone(),
            prev_prev_pos: pos.clone().into_storage(),
            prev_prev_vel: vel.clone().into_storage(),

            workspace: RefCell::new(WorkspaceData {
                x: prev_x,
                v: prev_v,
                pos,
                vel,
                grad,
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
    #[cfg(ignore)]
    fn compute_element_max_size(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
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

    fn build_problem(&self) -> Result<NonLinearProblem, Error> {
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

        let object_data = Self::build_object_data(solids, shells)?;

        let volume_constraints = Self::build_volume_constraints(&object_data.solids);

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

        let frictional_contacts = Self::build_frictional_contacts(
            &object_data.solids,
            &object_data.shells,
            frictional_contacts,
        );

        let max_size = Self::compute_max_size(&object_data.solids, &object_data.shells);

        // The following scales have units of force (N).
        let max_modulus_scale = Self::compute_max_modulus(&object_data.solids, &object_data.shells)?
            as f64
            * max_size
            * max_size;

        let max_element_mass_scale = if time_step > 0.0 {
            Self::compute_max_element_mass(&object_data.solids, &object_data.shells) * max_size
                / (time_step * time_step)
        } else {
            0.0
        };

        let max_element_bending_scale =
            Self::compute_max_bending_stiffness(&object_data.shells) / max_size;

        // Determine the most likely dominant force.
        let mut max_scale = max_modulus_scale
            .max(max_element_mass_scale)
            .max(max_element_bending_scale);

        // max_scale is a denominator. Ensure that it is never zero.
        if max_scale == 0.0 {
            log::warn!("All scaling factors are zero");
            max_scale = 1.0;
        }

        Ok(NonLinearProblem {
            object_data,
            frictional_contacts,
            volume_constraints,
            time_step,
            gravity,
            interrupt_checker: Box::new(|| false),
            iterations: 0,
            warm_start: Solution::default(),
            initial_residual_error: std::f64::INFINITY,
            iter_counter: RefCell::new(0),
            max_size,
            force_scale: max_scale,
        })
    }

    /// Build the simulation solver.
    pub fn build(&self) -> Result<Solver, Error> {
        let mut problem = self.build_problem()?;
        let time_step = problem.time_step;

        // Note that we don't need the solution field to get the number of variables and
        // constraints. This means we can use these functions to initialize solution.
        problem.reset_warm_start();

        let all_contacts_linear = problem.all_contacts_linear();
        let has_rigid = problem.has_rigid();

        // Construct the Ipopt solver.
        let mut ipopt = Ipopt::new(problem)?;

        // Setup ipopt paramters using the input simulation params.
        let params = self.sim_params.clone();

        log::info!("Simulation Parameters:\n{:#?}", params);

        let tol = params.tolerance as f64;

        ipopt.set_option("tol", tol);
        ipopt.set_option("acceptable_tol", tol);
        ipopt.set_option("max_iter", params.max_iterations as i32);
        //ipopt.set_option("bound_relax_factor", 0.0);

        match params.mu_strategy {
            MuStrategy::Monotone => ipopt.set_option("mu_strategy", "monotone"),
            MuStrategy::Adaptive => ipopt.set_option("mu_strategy", "adaptive"),
        };

        ipopt.set_option("sb", "yes"); // removes the Ipopt welcome message
        ipopt.set_option("print_level", params.print_level as i32);
        //ipopt.set_option("nlp_scaling_method", "user-scaling");
        ipopt.set_option("warm_start_init_point", "yes");
        if all_contacts_linear && !has_rigid {
            ipopt.set_option("jac_d_constant", "yes");
        }
        ipopt.set_option(
            "nlp_scaling_max_gradient",
            f64::from(params.max_gradient_scaling),
        );
        ipopt.set_option("print_timing_statistics", "yes");
        //ipopt.set_option("hessian_approximation", "limited-memory");
        if params.derivative_test > 0 {
            ipopt.set_option("derivative_test_tol", 1e-4);
            ipopt.set_option("point_perturbation_radius", 0.01);
            if params.derivative_test == 1 {
                ipopt.set_option("derivative_test", "first-order");
            } else if params.derivative_test == 2 {
                ipopt.set_option("derivative_test", "second-order");
            } else {
                return Err(Error::InvalidParameter {
                    name: "derivative_test".to_string(),
                });
            }
        }
        if let Some(ref log_file) = params.log_file {
            ipopt.set_option(
                "output_file",
                log_file
                    .as_os_str()
                    .to_str()
                    .expect("Invalid log file path."),
            );
        }
        ipopt.set_intermediate_callback(Some(NonLinearProblem::intermediate_cb));

        Ok(Solver {
            solver: ipopt,
            step_count: 0,
            inner_iterations: 0,
            sim_params: params,
            max_step: 0.0,
            old_active_constraint_set: Chunked::<Vec<usize>>::new(),
            max_time_step: time_step,
            time_step_remaining: time_step,
            minimum_time_step: time_step,
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
pub struct Solver {
    /// Non-linear solver.
    solver: Ipopt<NonLinearProblem>,
    /// Step count (outer iterations). These count the number of times the function `step` was
    /// called.
    step_count: usize,
    /// Total number of Ipopt iterations taken by this solver.
    inner_iterations: usize,
    /// Simulation parameters. This is kept around for convenience.
    sim_params: SimParams,
    /// Maximal displacement length. Used to limit displacement which is necessary in contact
    /// scenarios because it defines how far a step we can take before the constraint Jacobian
    /// sparsity pattern changes. If zero, then no limit is applied but constraint Jacobian is kept
    /// sparse.
    max_step: f64,
    /// A set of active constraints from the previous time step. This set is used to update the set
    /// of constraints for the new time steps after vertex positions have changed.
    /// The set is chunked by constraints. Each constraint chunk represents the
    /// active set for that constraint.
    old_active_constraint_set: Chunked<Vec<usize>>,
    /// Maximum time step.
    max_time_step: f64,
    /// The remainder of the time step left to simulate before we are done with this time step.
    time_step_remaining: f64,
    /// Record the current minimum time step used. This is useful for tweaking timesteps.
    minimum_time_step: f64,
}

impl Solver {
    /// If the time step was not specified or specified to be zero, then this function will return
    /// zero.
    pub fn time_step(&self) -> f64 {
        self.sim_params.time_step.unwrap_or(0.0).into()
    }
    /// Set the interrupt checker to the given function.
    pub fn set_interrupter(&mut self, checker: Box<dyn FnMut() -> bool>) {
        self.problem_mut().interrupt_checker = checker;
    }

    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &NonLinearProblem {
        self.solver.solver_data().problem
    }

    /// Get a mutable reference to the underlying problem.
    fn problem_mut(&mut self) -> &mut NonLinearProblem {
        self.solver.solver_data_mut().problem
    }

    /// Get a slice of solid objects represented in this solver.
    pub fn solids(&self) -> &[TetMeshSolid] {
        &self.problem().object_data.solids
    }

    /// Get a slice of shell objects represented in this solver.
    pub fn shells(&self) -> &[TriMeshShell] {
        &self.problem().object_data.shells
    }

    /// Get an immutable borrow for the underlying `TetMeshSolid` at the given index.
    pub fn solid(&self, index: usize) -> &TetMeshSolid {
        &self.problem().object_data.solids[index]
    }

    /// Get a mutable borrow for the underlying `TetMeshSolid` at the given index.
    pub fn solid_mut(&mut self, index: usize) -> &mut TetMeshSolid {
        &mut self.problem_mut().object_data.solids[index]
    }

    /// Get an immutable borrow for the underlying `TriMeshShell` at the given index.
    pub fn shell(&self, index: usize) -> &TriMeshShell {
        &self.problem().object_data.shells[index]
    }

    /// Get a mutable borrow for the underlying `TriMeshShell` at the given index.
    pub fn shell_mut(&mut self, index: usize) -> &mut TriMeshShell {
        &mut self.problem_mut().object_data.shells[index]
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

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn inner_step(&mut self) -> Result<InnerSolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            constraint_values,
            objective_value,
            status,
        } = self.solver.solve();

        let iterations = solver_data.problem.pop_iteration_count() as u32;
        solver_data.problem.update_warm_start(solver_data.solution);

        let result = InnerSolveResult {
            iterations,
            objective_value,
            constraint_values: constraint_values.to_vec(),
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::InnerSolveError {
                status: e,
                objective_value,
                iterations,
            }),
        }
    }

    /// Compute and add the value of the constraint function minus constraint bounds.
    pub fn compute_constraint_violation(&self, constraint: &mut [f64]) {
        use ipopt::ConstrainedProblem;
        let SolverData {
            problem,
            solution:
                ipopt::Solution {
                    primal_variables,
                    constraint_multipliers,
                    ..
                },
            ..
        } = self.solver.solver_data();

        assert_eq!(constraint.len(), constraint_multipliers.len());
        let mut lower = vec![0.0; constraint.len()];
        let mut upper = vec![0.0; constraint.len()];
        problem.constraint_bounds(&mut lower, &mut upper);
        assert!(problem.constraint(primal_variables, constraint));
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

    /// Compute the gradient of the objective. We only consider unfixed vertices.  Panic if this fails.
    pub fn compute_objective_gradient_product(&self, grad: &mut [f64]) -> f64 {
        use ipopt::BasicProblem;
        let SolverData {
            problem,
            solution: ipopt::Solution {
                primal_variables, ..
            },
            ..
        } = self.solver.solver_data();

        assert_eq!(grad.len(), primal_variables.len());
        assert!(problem.objective_grad(primal_variables, grad));

        grad.iter()
            .zip(primal_variables.iter())
            .map(|(g, dx)| g * dx)
            .sum()
    }

    ///// Compute the gradient of the objective. We only consider unfixed vertices.  Panic if this fails.
    //pub fn compute_objective_gradient(&self, grad: &mut [f64]) {
    //    use ipopt::BasicProblem;
    //    let SolverData {
    //        problem,
    //        solution: ipopt::Solution {
    //            primal_variables, ..
    //        },
    //        ..
    //    } = self.solver.solver_data();

    //    assert_eq!(grad.len(), primal_variables.len());
    //    assert!(problem.objective_grad(primal_variables, grad));

    //    // Erase fixed vert data. This doesn't contribute to the solve.
    //    let mesh = problem.tetmesh.borrow();
    //    let fixed_iter = mesh
    //        .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
    //        .expect("Missing fixed verts attribute")
    //        .map(|&x| x != 0);
    //    let vert_grad: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);
    //    for g in vert_grad.iter_mut().filter_masked(fixed_iter) {
    //        *g = Vector3::zeros();
    //    }
    //}

    ///// Compute and add the Jacobian and constraint multiplier product to the given vector.
    //pub fn add_constraint_jacobian_product(&self, jac_prod: &mut [f64]) {
    //    use ipopt::ConstrainedProblem;
    //    let SolverData {
    //        problem,
    //        solution:
    //            ipopt::Solution {
    //                primal_variables,
    //                constraint_multipliers,
    //                ..
    //            },
    //        ..
    //    } = self.solver.solver_data();

    //    let jac_nnz = problem.num_constraint_jacobian_non_zeros();
    //    let mut rows = vec![0; jac_nnz];
    //    let mut cols = vec![0; jac_nnz];
    //    assert!(problem.constraint_jacobian_indices(&mut rows, &mut cols));

    //    let mut values = vec![0.0; jac_nnz];
    //    assert!(problem.constraint_jacobian_values(primal_variables, &mut values));

    //    // We don't consider values for fixed vertices.
    //    let mesh = problem.tetmesh.borrow();
    //    let fixed = mesh
    //        .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
    //        .expect("Missing fixed verts attribute");

    //    assert_eq!(jac_prod.len(), primal_variables.len());
    //    // Effectively this is jac.transpose() * constraint_multipliers
    //    for ((row, col), val) in rows
    //        .into_iter()
    //        .zip(cols.into_iter())
    //        .zip(values.into_iter())
    //    {
    //        if fixed[(col as usize) / 3] == 0 {
    //            jac_prod[col as usize] += val * constraint_multipliers[row as usize];
    //        }
    //    }
    //}

    //fn dx(&self) -> &[f64] {
    //    self.solver.solver_data().solution.primal_variables
    //}

    //fn lambda(&self) -> &[f64] {
    //    self.solver.solver_data().constraint_multipliers
    //}

    /// Update the `mesh` and `prev_pos` with the current solution.
    fn commit_solution(&mut self, and_warm_start: bool, relax_max_step: bool) {
        {
            let and_velocity = !self.sim_params.clear_velocity;
            let SolverDataMut {
                problem, solution, ..
            } = self.solver.solver_data_mut();

            // Advance internal state (positions and velocities) of the problem.
            problem.advance(solution.primal_variables, and_velocity, and_warm_start);
        }

        // Reduce max_step for next iteration if the solution was a good one.
        if relax_max_step {
            let dt = self.time_step();
            let SolverDataMut {
                problem, solution, ..
            } = self.solver.solver_data_mut();
            if let Some(radius) = problem.min_contact_radius() {
                let step = inf_norm(problem.scaled_variables_iter(solution.primal_variables))
                    * if dt > 0.0 { dt } else { 1.0 };
                let new_max_step = (step - radius).max(self.max_step * 0.5);
                if self.max_step != new_max_step {
                    log::info!(
                        "Relaxing max step from {} to {}",
                        self.max_step,
                        new_max_step
                    );
                    self.max_step = new_max_step;
                    problem.update_max_step(new_max_step);
                }
            }
        }
    }

    /// Revert previously committed solution. We just advance in the opposite direction.
    fn revert_solution(&mut self) {
        self.problem_mut().revert_prev_step();
    }

    //fn output_meshes(&self, iter: u32) {
    //    let mesh = self.borrow_mesh();
    //    geo::io::save_tetmesh(
    //        &mesh,
    //        &std::path::PathBuf::from(format!("out/mesh_{}.vtk", iter + 1)),
    //    )
    //    .expect("Couldn't write to output tetrahedron mesh.");
    //    if let Some(mesh) = self.try_borrow_kinematic_mesh() {
    //        let polymesh = PolyMesh::from(mesh.clone());
    //        geo::io::save_polymesh(
    //            &polymesh,
    //            &std::path::PathBuf::from(format!("out/trimesh_{}.vtk", iter + 1)),
    //        )
    //        .expect("Couldn't write to output triangle lmesh.");
    //    }
    //}

    //fn compute_objective(&self) -> f64 {
    //    self.compute_objective_dx(self.solver.solver_data().solution.primal_variables)
    //}

    //fn compute_objective_dx(&self, dx: &[f64]) -> f64 {
    //    self.problem().objective_value(dx)
    //}

    //fn model_l1(&self) -> f64 {
    //    self.model_l1_dx(self.solver.solver_data().solution.primal_variables)
    //}

    //fn model_l1_dx(&self, dx: &[f64]) -> f64 {
    //    self.problem().linearized_constraint_violation_model_l1(dx)
    //}

    //fn merit_l1(&self) -> f64 {
    //    self.merit_l1_dx(self.solver.solver_data().solution.primal_variables)
    //}

    //fn merit_l1_dx(&self, _dx: &[f64]) -> f64 {
    //    0.0
    //    //self.problem().linearized_constraint_violation_l1(dx)
    //}

    //fn compute_residual(&self, residual: &mut Vec<f64>) -> f64 {
    //    use ipopt::{BasicProblem, ConstrainedProblem};
    //    let num_variables = self.problem().num_variables();
    //    let num_constraints = self.problem().num_constraints();

    //    // Reset the residual.
    //    residual.clear();
    //    residual.resize(num_variables + num_constraints, 0.0);

    //    let (objective_residual, constraint_violation) = residual.split_at_mut(num_variables);

    //    self.compute_objective_gradient(objective_residual);
    //    self.add_constraint_jacobian_product(objective_residual);
    //    self.compute_constraint_violation(constraint_violation);

    //    let constraint_violation_norm = inf_norm(constraint_violation.iter().cloned());
    //    let residual_norm = inf_norm(residual.iter().cloned());

    //    println!(
    //        "residual = {:?}, cv = {:?}",
    //        residual_norm, constraint_violation_norm
    //    );

    //    residual_norm
    //}

    fn initial_residual_error(&self) -> f64 {
        self.problem().initial_residual_error
    }

    fn save_current_active_constraint_set(&mut self) {
        let Solver {
            ref solver,
            ref mut old_active_constraint_set,
            ..
        } = self;
        old_active_constraint_set.clear();
        solver
            .solver_data()
            .problem
            .compute_active_constraint_set(old_active_constraint_set);
    }

    /// Determine if the inner step is acceptable. If unacceptable, adjust solver paramters and
    /// return false to indicate that the same step should be taken again.
    /// In either case this function modifies the configuration of the constraints and updates the
    /// constraint set. This means that the caller should take care to remap any constraint related
    /// values as needed. The configuration may also need to be reverted.
    fn check_inner_step(&mut self) -> bool {
        let step_acceptable = if self.problem().min_contact_radius().is_some() {
            let step = {
                let SolverData {
                    problem, solution, ..
                } = self.solver.solver_data();
                let dt = self.time_step();
                inf_norm(problem.scaled_variables_iter(solution.primal_variables))
                    * if dt > 0.0 { dt } else { 1.0 }
            };

            let constraint_violation = {
                let SolverDataMut {
                    problem, solution, ..
                } = self.solver.solver_data_mut();
                // Note: at the time of this writing, the sparsity_changed indicator can
                // have false negatives (no change detected, but really there was a
                // change). In which case we would be skipping an opportunity to do a more
                // accurate step.
                let _ = problem.update_constraint_set(Some(solution.clone()));

                problem.probe_contact_constraint_violation(solution)
            };

            let initial_error = self.initial_residual_error();
            let relative_tolerance = f64::from(self.sim_params.tolerance) / initial_error;
            log::debug!("Checking constraint violation: {}", constraint_violation);
            log::debug!(
                "Relative threshold for constraint violation: {}",
                relative_tolerance
            );
            // NOTE: Ipopt can't detect constraint values below 1e-7 in absolute value. It seems to
            // be a hardcoded threshold.
            if constraint_violation > 1e-7_f64.max(relative_tolerance) {
                // intersecting objects detected (allow leeway via relative_tolerance)
                if self.max_step < step {
                    // Increase the max_step to be slightly bigger than the current step to avoid
                    // floating point issues.
                    log::debug!("Increasing max step to {:e}", 1.1 * step);
                    self.update_max_step(1.1 * step);

                    // We don't commit the solution here because it may be far from the
                    // true solution, just redo the whole solve with the right
                    // neighbourhood information.
                    false
                } else {
                    log::debug!("Max step: {:e} is saturated, but constraint is still violated, continuing...", step);
                    // The step is smaller than max_step and the constraint is still violated.
                    // Nothing else we can do, just accept the solution and move on.
                    true
                }
            } else {
                // The solution is good, reset the max_step, and continue.
                // TODO: There is room for optimization here. It may be better to reduce the max
                // step but not to zero to prevent extra steps in the future.
                log::debug!("Solution accepted");
                true
            }
        } else {
            // No contact constraints, all solutions are good.
            true
        };

        // Restore the constraints to original configuration.
        self.problem_mut().reset_constraint_set();
        step_acceptable
    }

    /// Compute the friction impulse in the problem and return `true` if it has been updated and
    /// `false` otherwise. If friction is disabled, this function will return `false`.
    fn compute_friction_impulse(
        &mut self,
        constraint_values: &[f64],
        friction_steps: &mut [u32],
    ) -> bool {
        // Select constraint multipliers responsible for the contact force.
        let SolverDataMut {
            problem, solution, ..
        } = self.solver.solver_data_mut();

        problem.update_friction_impulse(solution, constraint_values, friction_steps)
    }

    fn remap_warm_start(&mut self) {
        let Solver {
            solver,
            old_active_constraint_set,
            ..
        } = self;

        solver
            .solver_data_mut()
            .problem
            .remap_warm_start(old_active_constraint_set.view());
    }

    fn all_contacts_linear(&self) -> bool {
        self.problem().all_contacts_linear()
    }

    /// Run the optimization solver on one time step.
    pub fn step(&mut self) -> Result<SolveResult, Error> {
        // Initialize the result of this function.
        let mut result = SolveResult {
            max_inner_iterations: 0,
            inner_iterations: 0,
            iterations: 0,
            objective_value: 0.0,
        };

        let all_contacts_linear = self.all_contacts_linear();

        // Recompute constraints since the active set may have changed if a collision mesh has moved.
        self.save_current_active_constraint_set();
        self.problem_mut().reset_constraint_set();

        log::info!(
            "Start step with time step remaining: {}",
            self.time_step_remaining
        );

        let mut recovery = false; // we are in recovery mode.

        // The number of friction solves to do.
        let mut friction_steps =
            vec![self.sim_params.friction_iterations; self.problem().num_frictional_contacts()];
        let total_friction_steps = friction_steps.iter().sum::<u32>();
        let mut iteration_count = 0;
        loop {
            if iteration_count >= self.sim_params.max_outer_iterations {
                log::warn!(
                    "Reached max outer iterations: {:?}",
                    self.sim_params.max_outer_iterations
                );
                break;
            }
            iteration_count += 1;

            // Clamp time step to what we actually need.
            if self.problem().time_step > self.time_step_remaining {
                self.problem_mut().time_step = self.time_step_remaining;
            }

            self.minimum_time_step = self.minimum_time_step.min(self.problem().time_step);

            // Remap warm start from the initial constraint reset above, or if the constraints were
            // updated after advection.
            self.remap_warm_start();
            self.save_current_active_constraint_set();

            let step_result = self.inner_step();

            let min_time_step = 1e-8;

            // The following block determines if after the inner step there were any changes
            // in the constraints where new points may have violated the constraint. If so we
            // update the constraint neighbourhoods and rerun the inner step.
            match step_result {
                Ok(step_result) => {
                    result = result.combine_inner_result(&step_result);
                    //{
                    //    let SolverDataMut {
                    //        problem, solution, ..
                    //    } = self.solver.solver_data_mut();
                    //    problem.save_intermediate(solution.primal_variables, self.step_count);
                    //}

                    if all_contacts_linear || self.check_inner_step() {
                        if !friction_steps.is_empty() && total_friction_steps > 0 {
                            debug_assert!(
                                all_contacts_linear
                                    // Check that the constraint set hasn't been modified in the check inner step.
                                    || self.problem().is_same_as_constraint_set(
                                        self.old_active_constraint_set.view()
                                    )
                            );
                            let is_finished = self.compute_friction_impulse(
                                &step_result.constraint_values,
                                &mut friction_steps,
                            );
                            if !is_finished {
                                continue;
                            }
                        }
                        self.commit_solution(true, !all_contacts_linear);
                        self.time_step_remaining -= self.problem().time_step;
                        log::info!("Time step remaining: {}", self.time_step_remaining);
                        recovery = false; // We have recovered.
                        if self.time_step_remaining < min_time_step {
                            // Reset time step progress
                            self.time_step_remaining = self.max_time_step;
                            break;
                        }
                        if !all_contacts_linear {
                            // Completing the time step, we need to reset constraint set for next step
                            self.problem_mut().reset_constraint_set();
                        }
                    }
                }
                Err(Error::InnerSolveError {
                    status,
                    iterations,
                    objective_value,
                }) => {
                    // Something went wrong, revert one step, reduce the time step and try again.

                    // Don't bother if there are no more max_outer_iterations left, the time step
                    // is too small or the user requested a stop.
                    if self.problem().time_step < min_time_step
                        || iteration_count + 1 >= self.sim_params.max_outer_iterations
                        || status == ipopt::SolveStatus::UserRequestedStop
                    {
                        log::warn!("Failed to recover");
                        // Can't recover, return with an error
                        result = result.combine_inner_step_data(iterations, objective_value);
                        self.commit_solution(true, !all_contacts_linear);
                        return Err(Error::SolveError { status, result });
                    }
                    if !recovery {
                        log::info!("Recovering: Revert previous step");

                        self.revert_solution();
                        self.problem_mut().reset_constraint_set();
                        // reset friction iterations.
                        friction_steps
                            .iter_mut()
                            .for_each(|n| *n = self.sim_params.friction_iterations);
                        // Since we reverted a step we should add that time step to the time
                        // remaining.
                        self.time_step_remaining += self.problem().time_step;
                    }

                    // Decrement iteration count since we reverted one iteration.
                    // Since we check for minimal time step we know this loop will eventually
                    // break even if the problem is not recoverable.
                    iteration_count -= 1;

                    recovery = true;
                    self.problem_mut().time_step *= 0.5;
                    log::info!("Reduce time step to {}", self.problem().time_step);
                }
                Err(e) => {
                    // Unknown error: Clear warm start and return.
                    self.commit_solution(false, false);
                    return Err(e);
                }
            }
        }

        // Gradually restore time_step if its lower than max_step
        if self.problem().time_step < self.max_time_step {
            self.problem_mut().time_step *= 2.0;
        }

        if !all_contacts_linear || recovery {
            // Remap warm start since after committing the solution, the constraint set may have
            // changed if we relaxed max_step.
            // (Or we existed while still in recovery for some reason).
            // We use warm start data to update the mesh
            self.remap_warm_start();
        }

        //self.output_meshes(self.step_count as u32);

        self.inner_iterations += result.inner_iterations as usize;
        self.step_count += 1;

        log::debug!("Inner iterations: {}", self.inner_iterations);
        log::info!("Minimum time step so far: {}", self.minimum_time_step);

        // On success, update the mesh with useful metrics.
        self.problem_mut().update_mesh_data();

        Ok(result)
    }

    //    /// Run the optimization solver on one time step. This method uses the trust region method to
    //    /// resolve linearized constraints.
    //    pub fn step_tr(&mut self) -> Result<SolveResult, Error> {
    //        use ipopt::BasicProblem;
    //
    //        println!("params = {:?}", self.sim_params);
    //        println!("material = {:?}", self.solid_material);
    //
    //        // Initialize the result of this function.
    //        let mut result = SolveResult {
    //            max_inner_iterations: 0,
    //            inner_iterations: 0,
    //            iterations: 0,
    //            objective_value: 0.0,
    //        };
    //
    //        let mut residual = Vec::new();
    //        let mut residual_norm = self.compute_residual(&mut residual);
    //        let mut objective_gradient = vec![0.0; self.problem().num_variables()];
    //
    //        let zero_dx = vec![0.0; self.dx().len()];
    //
    //        let rho = 0.1;
    //
    //        self.output_meshes(0);
    //
    //        // We should iterate until a relative residual goes to zero.
    //        for iter in 0..self.sim_params.max_outer_iterations {
    //            let step_result = self.inner_step();
    //
    //            let f_k = self.compute_objective_dx(&zero_dx);
    //            let c_k = self.merit_l1_dx(&zero_dx);
    //            assert_relative_eq!(c_k, self.model_l1_dx(&zero_dx));
    //
    //            let mu = if relative_eq!(c_k, 0.0) {
    //                0.0
    //            } else {
    //                let fdx = self.compute_objective_gradient_product(&mut objective_gradient);
    //                println!("fdx = {:?}", fdx);
    //                (fdx / ((1.0 - rho) * c_k)).max(0.0)
    //            };
    //            println!("mu = {:?}", mu);
    //
    //            let prev_merit = f_k + mu * c_k;
    //            let prev_model = prev_merit;
    //
    //            let f_k1 = self.compute_objective();
    //
    //            let merit_model = f_k1 + mu * self.model_l1();
    //            let merit_value = f_k1 + mu * self.merit_l1();
    //
    //            // Since we are using linearized constraints, we compute the true
    //            // residual and iterate until it vanishes. For unconstrained problems or problems with
    //            // no linearized constraints, only one step is needed.
    //            residual_norm = self.compute_residual(&mut residual);
    //
    //            // Commit the solution whether or not there is an error. In case of error we will be
    //            // able to investigate the result.
    //            let (old_sol, old_prev_pos, old_prev_vel) = self.commit_solution(true);
    //
    //            self.output_meshes(iter + 1);
    //
    //            // Update output result with new data.
    //            match step_result {
    //                Ok(step_result) => {
    //                    result = result.combine_inner_result(&step_result);
    //                }
    //                Err(Error::InnerSolveError {
    //                    status,
    //                    iterations,
    //                    objective_value,
    //                }) => {
    //                    // If the problem is infeasible, it may be that our step size is too small,
    //                    // Try to increase it (but not past max_step) and try again:
    //                    if status == ipopt::SolveStatus::InfeasibleProblemDetected {
    //                        let max_step = self.max_step;
    //                        let problem = self.problem_mut();
    //                        if let Some(disp) = problem.displacement_bound {
    //                            problem.displacement_bound.replace(max_step.min(1.5 * disp));
    //                            continue;
    //                        }
    //                    } else {
    //                        // Otherwise we don't know what else to do so return an error.
    //                        result = result.combine_inner_step_data(iterations, objective_value);
    //                        return Err(Error::SolveError(status, result));
    //                    }
    //                }
    //                Err(e) => {
    //                    // Unknown error: Reset warm start and return.
    //                    self.solver.set_option("warm_start_init_point", "no");
    //                    return Err(e);
    //                }
    //            }
    //
    //            if residual_norm <= f64::from(self.sim_params.outer_tolerance) {
    //                break;
    //            }
    //
    //            // Check that the objective is actually lowered. Trust region method for constraints.
    //            // The justification is that our constraint violation estimate is much worse than the
    //            // energy estimate from the solve of the inner problem. As such we will iterate on the
    //            // infinity norm of all violated linearized constraints (constraint points outside the
    //            // feasible domain).
    //            {
    //                let max_step = self.max_step;
    //                let SolverDataMut {
    //                    problem, solution, ..
    //                } = self.solver.solver_data_mut();
    //
    //                if max_step > 0.0 {
    //                    if let Some(disp) = problem.displacement_bound {
    //                        println!(
    //                            "f_k = {:?}, f_k+1 = {:?}, m_k = {:?}, m_k+1 = {:?}",
    //                            prev_merit, merit_value, prev_model, merit_model
    //                        );
    //                        let reduction = (prev_merit - merit_value) / (prev_model - merit_model);
    //                        println!("reduction = {:?}", reduction);
    //
    //                        let step_size = inf_norm(solution.primal_variables.iter().cloned());
    //
    //                        if reduction < 0.25 {
    //                            if step_size < 1e-5 * max_step {
    //                                break; // Reducing the step wont help at this point
    //                            }
    //                            // The constraint violation is not decreasing, roll back and reduce the step size.
    //                            println!("step size taken {:?}", step_size);
    //                            problem.displacement_bound.replace(0.25 * step_size);
    //                            println!("reducing step to {:?}", problem.displacement_bound.unwrap());
    //                        } else {
    //                            println!("step size = {:?} vs. disp = {:?}", step_size, disp);
    //                            if reduction > 0.75 && relative_eq!(step_size, disp) {
    //                                // we took a full step
    //                                // The linearized constraints are a good model of the actual constraints,
    //                                // increase the step size and continue.
    //                                problem.displacement_bound.replace(max_step.min(2.0 * disp));
    //                                println!(
    //                                    "increase step to {:?}",
    //                                    problem.displacement_bound.unwrap()
    //                                );
    //                            } // Otherwise keep the step size the same.
    //                        }
    //                        if reduction < 0.15 {
    //                            // Otherwise constraint violation is not properly decreasing
    //                            Self::revert_solution(problem, old_sol, old_prev_pos, old_prev_vel);
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //
    //        if result.iterations > self.sim_params.max_outer_iterations {
    //            eprintln!(
    //                "WARNING: Reached max outer iterations: {:?}\nResidual is: {:?}",
    //                result.iterations, residual_norm
    //            );
    //        }
    //
    //        self.step_count += 1;
    //
    //        // On success, update the mesh with new positions and useful metrics.
    //        let (lambda, mu) = self.solid_material.unwrap().elasticity.lame_parameters();
    //
    //        let mut mesh = self.borrow_mut_mesh();
    //
    //        // Write back elastic strain energy for visualization.
    //        Self::compute_strain_energy_attrib(&mut mesh, lambda, mu);
    //
    //        // Write back elastic forces on each node.
    //        Self::compute_elastic_forces_attrib(&mut mesh, lambda, mu);
    //
    //        Ok(result)
    //    }
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
