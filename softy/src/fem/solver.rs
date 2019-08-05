use crate::attrib_defines::*;
use crate::constraints::*;
use crate::contact::*;
use crate::fem::problem::FrictionalContactConstraint;
use crate::objects::*;
use geo::math::{Matrix3, Vector3};
use geo::mesh::{
    attrib::{self, VertexAttrib},
    topology::*,
    Attrib, VertexPositions,
};
use geo::ops::{Area, ShapeMatrix, Volume};
use geo::prim::{Tetrahedron, Triangle};
use ipopt::{self, Ipopt, SolverData, SolverDataMut};
use std::cell::RefCell;
use utils::{soap::*, zip};

use crate::inf_norm;

use super::{MuStrategy, NonLinearProblem, ObjectData, SimParams, Solution, SourceIndex};
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
/// This routine assumes that there is a vertex attribute called `ref` of type `[f64;3]`.
pub fn ref_tet(tetmesh: &TetMesh, indices: &[usize; 4]) -> Tetrahedron<f64> {
    let ref_pos = tetmesh
        .attrib::<VertexIndex>(REFERENCE_POSITION_ATTRIB)
        .unwrap()
        .as_slice::<[f64; 3]>()
        .unwrap();
    Tetrahedron::from_indexed_slice(indices, ref_pos)
}

#[derive(Clone, Debug)]
pub struct SolverBuilder {
    sim_params: SimParams,
    solids: Vec<(TetMesh, SolidMaterial)>,
    shells: Vec<(PolyMesh, ShellMaterial)>,
    frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
}

impl SolverBuilder {
    /// Create a `SolverBuilder` with the minimum required parameters, which are the simulation
    /// parameters, `SimParams`.
    pub fn new(sim_params: SimParams) -> Self {
        SolverBuilder {
            sim_params,
            solids: Vec::new(),
            shells: Vec::new(),
            frictional_contacts: Vec::new(),
        }
    }

    /// Add a tetmesh representing a soft solid (e.g. soft tissue).
    pub fn add_solid(&mut self, mesh: TetMesh, mat: SolidMaterial) -> &mut Self {
        self.solids.push((mesh, mat));
        self
    }

    /// Add a polygon mesh representing a shell (e.g. cloth).
    pub fn add_shell(&mut self, shell: PolyMesh, mat: ShellMaterial) -> &mut Self {
        self.shells.push((shell, mat));
        self
    }

    /// Add a static polygon mesh representing an immovable solid.
    pub fn add_fixed(&mut self, shell: PolyMesh, id: usize) -> &mut Self {
        self.shells.push((shell, ShellMaterial::fixed(id)));
        self
    }

    /// Add a rigid polygon mesh representing an rigid solid.
    pub fn add_rigid(&mut self, shell: PolyMesh, density: f64, id: usize) -> &mut Self {
        self.shells.push((shell, ShellMaterial::rigid(id, density)));
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
        if !params.friction_params.is_some() || self.sim_params.time_step.is_some() {
            self.frictional_contacts.push((params, mat_ids));
        }
        self
    }

    /// Helper function to compute a set of frictional contacts.
    fn build_frictional_contacts(
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
        frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
    ) -> Vec<FrictionalContactConstraint> {
        // Build a mapping to mesh indices in their respective slices.
        let mut material_source = Vec::new();
        material_source.extend((0..solids.len()).map(|i| SourceIndex::Solid(i)));
        material_source.extend((0..shells.len()).map(|i| SourceIndex::Shell(i)));

        // Sort materials by material id.
        let to_mat_id = |k: &SourceIndex| match *k {
            SourceIndex::Solid(i) => solids[i].material.id,
            SourceIndex::Shell(i) => shells[i].material.id,
        };
        material_source.sort_unstable_by_key(to_mat_id);

        // Assemble material source indices into a chunked array where each
        // chunk represents a unique id. This means some chunks are empty.
        let mut id = 0;
        let mut off = 0;
        let mut offsets = vec![0];
        for cur_id in material_source.view().into_flat().iter().map(to_mat_id) {
            while id < cur_id {
                offsets.push(off);
                id += 1;
            }
            off += 1;
        }
        offsets.push(off);

        let material_source = Chunked::from_offsets(offsets, material_source);

        let construct_friction_constraint = |m0, m1, constraint| FrictionalContactConstraint {
            object_index: m0,
            collider_index: m1,
            constraint,
        };

        // Convert frictional contact parameters into frictional contact constraints.
        frictional_contacts
            .into_iter()
            .flat_map(|(params, (obj_id, coll_id))| {
                let material_source_obj = &material_source[obj_id];
                let material_source_coll = &material_source[coll_id];
                material_source_obj.iter().flat_map(move |&m0| {
                    let (solid_iter, shell_iter) = match m0 {
                        SourceIndex::Solid(i) => (
                            Some(material_source_coll.iter().flat_map(move |&m1| {
                                match m1 {
                                    SourceIndex::Solid(j) => build_contact_constraint(
                                        &solids[i].surface().trimesh,
                                        &solids[j].surface().trimesh,
                                        params,
                                    ),
                                    SourceIndex::Shell(j) => build_contact_constraint(
                                        &solids[i].surface().trimesh,
                                        &shells[j].trimesh,
                                        params,
                                    ),
                                }
                                .map(|c| construct_friction_constraint(m0, m1, c))
                                .ok()
                                .into_iter()
                            })),
                            None,
                        ),
                        SourceIndex::Shell(i) => (
                            None,
                            Some(material_source_coll.iter().flat_map(move |&m1| {
                                match m1 {
                                    SourceIndex::Solid(j) => build_contact_constraint(
                                        &shells[i].trimesh,
                                        &solids[j].surface().trimesh,
                                        params,
                                    ),
                                    SourceIndex::Shell(j) => build_contact_constraint(
                                        &shells[i].trimesh,
                                        &shells[j].trimesh,
                                        params,
                                    ),
                                }
                                .map(|c| construct_friction_constraint(m0, m1, c))
                                .ok()
                                .into_iter()
                            })),
                        ),
                    };
                    solid_iter
                        .into_iter()
                        .flatten()
                        .chain(shell_iter.into_iter().flatten())
                })
            })
            .collect()
    }

    /// Helper function to initialize volume constraints from a set of solids.
    fn build_volume_constraints(solids: &[TetMeshSolid]) -> Vec<(usize, VolumeConstraint)> {
        // Initialize volume constraint
        solids
            .iter()
            .enumerate()
            .filter(|&(_, solid)| solid.material.volume_preservation())
            .map(|(idx, solid)| (idx, VolumeConstraint::new(&solid.tetmesh)))
            .collect()
    }

    /// Assuming `mesh` has prepopulated vertex masses, this function computes its center of mass.
    fn compute_centre_of_mass(mesh: &TriMesh) -> [f64; 3] {
        let mut com = Vector3::zeros();
        let mut total_mass = 0.0;

        for (&v, &m) in mesh.vertex_position_iter().zip(
            mesh.attrib_iter::<MassType, VertexIndex>(MASS_ATTRIB)
                .unwrap(),
        ) {
            com += Vector3(v) * m;
            total_mass += m;
        }
        (com / total_mass).into()
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
            prev_v.push(mesh.attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?);
            pos.push(vec![]);
            vel.push(vec![]);
        }

        for TriMeshShell {
            trimesh: ref mesh,
            material,
        } in shells.iter()
        {
            match material.properties {
                ShellProperties::Rigid { .. } => {
                    let translation = Self::compute_centre_of_mass(mesh);
                    let rotation = [0.0; 3];
                    prev_x.push(vec![translation, rotation]);
                    prev_v.push(vec![[0.0; 3], [0.0; 3]]);

                    pos.push(mesh.vertex_positions().to_vec());
                    let mesh_vel =
                        mesh.attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    vel.push(mesh_vel);
                }
                ShellProperties::Deformable { .. } => {
                    prev_x.push(mesh.vertex_positions().to_vec());
                    let mesh_prev_vel =
                        mesh.attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    prev_v.push(mesh_prev_vel);
                    pos.push(vec![]);
                    vel.push(vec![]);
                }
                ShellProperties::Fixed => {
                    prev_x.push(vec![]);
                    prev_v.push(vec![]);
                    pos.push(mesh.vertex_positions().to_vec());
                    let mesh_vel =
                        mesh.attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?;
                    vel.push(mesh_vel);
                }
            }
        }

        let num_meshes = solids.len() + shells.len();
        let prev_x = Chunked::from_offsets(vec![0, solids.len(), num_meshes], prev_x);
        let prev_v = Chunked::from_offsets(vec![0, solids.len(), num_meshes], prev_v);
        let pos = Chunked::from_offsets(vec![0, solids.len(), num_meshes], pos);
        let vel = Chunked::from_offsets(vec![0, solids.len(), num_meshes], vel);

        Ok(ObjectData {
            prev_x: prev_x.clone(),
            prev_v: prev_v.clone(),
            cur_x: RefCell::new(prev_x.clone()),
            cur_v: RefCell::new(prev_v.clone()),

            pos: pos.clone(),
            vel: vel.clone(),

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
    fn build_shells(shells: Vec<(PolyMesh, ShellMaterial)>) -> Result<Vec<TriMeshShell>, Error> {
        // Equip `PolyMesh`es with physics parameters, making them bona-fide shells.
        let mut out = Vec::new();
        for (polymesh, material) in shells.into_iter() {
            let trimesh = TriMesh::from(polymesh);
            // Prepare shell for simulation.
            out.push(Self::prepare_shell_attributes(TriMeshShell {
                trimesh,
                material,
            })?)
        }

        Ok(out)
    }

    /// Helper to compute max element size. This is used for normalizing tolerances.
    fn compute_max_size(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> f64 {
        let mut max_size = 0.0_f64;

        for TetMeshSolid { ref tetmesh, .. } in solids.iter() {
            max_size = max_size.max(
                tetmesh
                    .tet_iter()
                    .map(Volume::volume)
                    .max_by(|a, b| a.partial_cmp(b).expect("Degenerate tetrahedron detected"))
                    .expect("Given TetMesh is empty")
                    .cbrt(),
            );
        }

        for TriMeshShell { ref trimesh, .. } in shells.iter() {
            max_size = max_size.max(
                trimesh
                    .tri_iter()
                    .map(Area::area)
                    .max_by(|a, b| a.partial_cmp(b).expect("Degenerate triangle detected"))
                    .expect("Given TriMesh is empty")
                    .sqrt(),
            );
        }

        max_size
    }

    /// Helper function to compute the maximum elastic modulus of all given meshes.
    /// This aids in figuring out the correct scaling for the convergence tolerances.
    fn compute_max_modulus(solids: &[TetMeshSolid], shells: &[TriMeshShell]) -> Result<f64, Error> {
        let mut max_modulus = 0.0_f64;

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

        for TriMeshShell {
            ref trimesh,
            material,
        } in shells.iter()
        {
            match material.properties {
                ShellProperties::Deformable { .. } => {
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

    fn build_problem(&self) -> Result<NonLinearProblem, Error> {
        let SolverBuilder {
            sim_params: params,
            solids,
            shells,
            frictional_contacts,
        } = self.clone();

        let solids = Self::build_solids(solids)?;
        let shells = Self::build_shells(shells)?;

        let object_data = Self::build_object_data(solids, shells)?;

        let volume_constraints = Self::build_volume_constraints(&object_data.solids);

        let gravity = [
            f64::from(params.gravity[0]),
            f64::from(params.gravity[1]),
            f64::from(params.gravity[2]),
        ];

        let time_step = f64::from(params.time_step.unwrap_or(0.0f32));

        let frictional_contacts = Self::build_frictional_contacts(
            &object_data.solids,
            &object_data.shells,
            frictional_contacts,
        );

        let displacement_bound = None;
        //let displacement_bound = smooth_contact_params.map(|scp| {
        //    // Convert from a 2 norm bound (max_step) to an inf norm bound (displacement component
        //    // bound).
        //    scp.max_step / 3.0f64.sqrt()
        //});

        Ok(NonLinearProblem {
            object_data,
            frictional_contacts,
            volume_constraints,
            time_step,
            gravity,
            displacement_bound,
            interrupt_checker: Box::new(|| false),
            iterations: 0,
            warm_start: Solution::default(),
            initial_residual_error: std::f64::INFINITY,
            iter_counter: RefCell::new(0),
        })
    }

    /// Build the simulation solver.
    pub fn build(&self) -> Result<Solver, Error> {
        let mut problem = self.build_problem()?;

        let max_size =
            Self::compute_max_size(&problem.object_data.solids, &problem.object_data.shells);

        // Note that we don't need the solution field to get the number of variables and
        // constraints. This means we can use these functions to initialize solution.
        problem.reset_warm_start();

        let max_modulus =
            Self::compute_max_modulus(&problem.object_data.solids, &problem.object_data.shells)?;

        // Construct the Ipopt solver.
        let mut ipopt = Ipopt::new(problem)?;

        // Setup ipopt paramters using the input simulation params.
        let mut params = self.sim_params.clone();

        // Determine the true force tolerance. To start we base this tolerance
        // on the elastic response which depends on mu and lambda as well as per
        // tet volume: Larger stiffnesses and volumes cause proportionally
        // larger gradients. Thus our tolerance should reflect these properties.
        let max_area = max_size * max_size;
        let tol = f64::from(params.tolerance) * max_area * max_modulus;
        params.tolerance = tol as f32;
        params.outer_tolerance *= (max_area * max_modulus) as f32;

        ipopt.set_option("tol", tol);
        ipopt.set_option("acceptable_tol", tol);
        ipopt.set_option("max_iter", params.max_iterations as i32);

        match params.mu_strategy {
            MuStrategy::Monotone => ipopt.set_option("mu_strategy", "monotone"),
            MuStrategy::Adaptive => ipopt.set_option("mu_strategy", "adaptive"),
        };

        ipopt.set_option("sb", "yes"); // removes the Ipopt welcome message
        ipopt.set_option("print_level", params.print_level as i32);
        //ipopt.set_option("nlp_scaling_method", "user-scaling");
        ipopt.set_option("warm_start_init_point", "yes");
        //ipopt.set_option("jac_d_constant", "yes");
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
                return Err(Error::InvalidParameter("derivative_test".to_string()));
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
            old_active_set: Vec::new(),
        })
    }

    /// Compute signed volume for reference elements in the given `TetMesh`.
    fn compute_ref_tri_areas(mesh: &mut TriMesh) -> Result<Vec<f64>, Error> {
        let ref_pos = mesh
            .attrib::<VertexIndex>(REFERENCE_POSITION_ATTRIB)
            .unwrap()
            .as_slice::<[f64; 3]>()?;

        let ref_tri = |indices| Triangle::from_indexed_slice(indices, ref_pos);

        Ok(mesh.face_iter().map(|face| ref_tri(face).area()).collect())
    }

    /// Compute signed volume for reference elements in the given `TetMesh`.
    fn compute_ref_tet_signed_volumes(mesh: &mut TetMesh) -> Result<Vec<f64>, Error> {
        let ref_volumes: Vec<f64> = mesh
            .cell_iter()
            .map(|cell| ref_tet(&mesh, cell).signed_volume())
            .collect();
        if ref_volumes.iter().any(|&x| x <= 0.0) {
            return Err(Error::InvertedReferenceElement);
        }
        Ok(ref_volumes)
    }

    /// Compute shape matrix inverses for reference elements in the given `TetMesh`.
    fn compute_ref_tet_shape_matrix_inverses(mesh: &mut TetMesh) -> Vec<Matrix3<f64>> {
        // Compute reference shape matrix inverses
        mesh.cell_iter()
            .map(|cell| {
                let ref_shape_matrix = ref_tet(&mesh, cell).shape_matrix();
                // We assume that ref_shape_matrices are non-singular.
                ref_shape_matrix.inverse().unwrap()
            })
            .collect()
    }

    fn prepare_kinematic_mesh_vertex_attributes<M: NumVertices + VertexAttrib + Attrib>(
        mesh: &mut M,
    ) -> Result<(), Error> {
        mesh.attrib_or_add::<VelType, VertexIndex>(VELOCITY_ATTRIB, [0.0; 3])?;
        Ok(())
    }

    /// A helper function to populate vertex attributes for simulation on a dynamic mesh.
    fn prepare_dynamic_mesh_vertex_attributes<M: NumVertices + VertexAttrib + Attrib>(
        mesh: &mut M,
    ) -> Result<(), Error> {
        Self::prepare_kinematic_mesh_vertex_attributes(mesh)?;

        // If this attribute doesn't exist, assume no vertices are fixed. This function will
        // return an error if there is an existing Fixed attribute with the wrong type.
        {
            use geo::mesh::attrib::*;
            let fixed_buf = mesh
                .remove_attrib::<VertexIndex>(FIXED_ATTRIB)
                .unwrap_or_else(|_| {
                    Attribute::from_vec(vec![0 as FixedIntType; mesh.num_vertices()])
                })
                .into_buffer();
            let mut fixed = fixed_buf.cast_into_vec::<FixedIntType>();
            if fixed.is_empty() {
                // If non-numeric type detected, just fill it with zeros.
                fixed.resize(mesh.num_vertices(), 0);
            }
            mesh.insert_attrib::<VertexIndex>(FIXED_ATTRIB, Attribute::from_vec(fixed))?;
        }

        Ok(())
    }

    /// A helper function to populate vertex attributes for simulation on a deformable mesh.
    pub(crate) fn prepare_deformable_mesh_vertex_attributes<M>(mesh: &mut M) -> Result<(), Error>
    where
        M: NumVertices + VertexPositions<Element = [f64; 3]> + VertexAttrib + Attrib,
    {
        // Deformable meshes are dynamic. Prepare dynamic attributes first.
        Self::prepare_dynamic_mesh_vertex_attributes(mesh)?;

        let verts = mesh.vertex_positions().to_vec();

        mesh.attrib_or_add_data::<RefPosType, VertexIndex>(
            REFERENCE_POSITION_ATTRIB,
            verts.as_slice(),
        )?;

        {
            // Add elastic elastic force attributes.
            // These will be computed at the end of the time step.
            mesh.set_attrib::<ElasticForceType, VertexIndex>(ELASTIC_FORCE_ATTRIB, [0f64; 3])?;
        }

        Ok(())
    }

    /// Compute vertex masses on the given solid. The solid is assumed to have
    /// volume and density attributes already.
    pub fn compute_solid_vertex_masses(solid: &mut TetMeshSolid) {
        let tetmesh = &mut solid.tetmesh;
        let mut masses = vec![0.0; tetmesh.num_vertices()];

        for (&vol, &density, cell) in zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap(),
            tetmesh.cell_iter()
        ) {
            for i in 0..4 {
                masses[cell[i]] += 0.25 * vol * density;
            }
        }

        tetmesh
            .add_attrib_data::<MassType, VertexIndex>(MASS_ATTRIB, masses)
            .unwrap();
    }

    /// Compute vertex masses on the given shell. The shell is assumed to have
    /// area and density attributes already.
    pub fn compute_shell_vertex_masses(shell: &mut TriMeshShell) {
        let trimesh = &mut shell.trimesh;
        let mut masses = vec![0.0; trimesh.num_vertices()];

        for (&area, density, face) in zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap(),
            trimesh.face_iter()
        ) {
            for i in 0..3 {
                masses[face[i]] += area * density / 3.0;
            }
        }

        trimesh
            .add_attrib_data::<MassType, VertexIndex>(MASS_ATTRIB, masses)
            .unwrap();
    }

    pub(crate) fn prepare_density_attribute<Obj: Object>(object: &mut Obj) -> Result<(), Error> {
        // Prepare density parameter
        if let Some(density) = object.material().scaled_density() {
            let num_elements = object.num_elements();
            match object
                .mesh_mut()
                .add_attrib_data::<DensityType, Obj::ElementIndex>(
                    DENSITY_ATTRIB,
                    vec![density; num_elements],
                ) {
                // if ok or already exists, everything is ok.
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
        } else {
            // No global density parameter was given. Check that it exists on the mesh itself.
            if object
                .mesh()
                .attrib_check::<DensityType, Obj::ElementIndex>(DENSITY_ATTRIB)
                .is_err()
            {
                return Err(Error::MissingDensityParam);
            }

            // Scale the mesh parameter so that it is consistent with the rest
            // of the material.

            let scale = object.material().scale();
            for density in object
                .mesh_mut()
                .attrib_iter_mut::<DensityType, Obj::ElementIndex>(DENSITY_ATTRIB)?
            {
                *density *= scale;
            }
        }
        Ok(())
    }

    /// Transfer parameters `lambda` and `mu` from the object material to the
    /// mesh if it hasn't already been populated on the input.
    pub(crate) fn prepare_elasticity_attributes<Obj: Object>(obj: &mut Obj) -> Result<(), Error> {
        if let Some(elasticity) = obj.material().scaled_elasticity() {
            let num_elements = obj.num_elements();
            match obj
                .mesh_mut()
                .add_attrib_data::<LambdaType, Obj::ElementIndex>(
                    LAMBDA_ATTRIB,
                    vec![elasticity.lambda; num_elements],
                ) {
                // if ok or already exists, everything is ok.
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
            match obj.mesh_mut().add_attrib_data::<MuType, Obj::ElementIndex>(
                MU_ATTRIB,
                vec![elasticity.mu; num_elements],
            ) {
                // if ok or already exists, everything is ok.
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
        } else {
            // No global elasticity parameters were given. Check that the mesh has the right
            // parameters.
            if obj
                .mesh()
                .attrib_check::<LambdaType, Obj::ElementIndex>(LAMBDA_ATTRIB)
                .is_err()
                || obj
                    .mesh()
                    .attrib_check::<MuType, Obj::ElementIndex>(MU_ATTRIB)
                    .is_err()
            {
                return Err(Error::MissingElasticityParams);
            }

            // Scale the mesh parameters so that they are consistent with the
            // rest of the material.

            let scale = obj.material().scale();
            for lambda in obj
                .mesh_mut()
                .attrib_iter_mut::<LambdaType, Obj::ElementIndex>(LAMBDA_ATTRIB)?
            {
                *lambda *= scale;
            }
            for mu in obj
                .mesh_mut()
                .attrib_iter_mut::<MuType, Obj::ElementIndex>(MU_ATTRIB)?
            {
                *mu *= scale;
            }
        }
        Ok(())
    }

    pub(crate) fn prepare_deformable_tetmesh_attributes(mesh: &mut TetMesh) -> Result<(), Error> {
        let ref_volumes = Self::compute_ref_tet_signed_volumes(mesh)?;
        mesh.set_attrib_data::<RefVolType, CellIndex>(
            REFERENCE_VOLUME_ATTRIB,
            ref_volumes.as_slice(),
        )?;

        let ref_shape_mtx_inverses = Self::compute_ref_tet_shape_matrix_inverses(mesh);
        mesh.set_attrib_data::<_, CellIndex>(
            REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
            ref_shape_mtx_inverses.as_slice(),
        )?;
        Ok(())
    }

    pub(crate) fn prepare_source_index_attribute<M>(mesh: &mut M) -> Result<(), Error>
    where
        M: NumVertices + Attrib + VertexAttrib,
    {
        // Need source index for meshes so that their vertices can be updated.
        // If this index is missing, it is assumed that the source is coincident
        // with the provided mesh.
        let num_verts = mesh.num_vertices();
        match mesh.add_attrib_data::<SourceIndexType, VertexIndex>(
            SOURCE_INDEX_ATTRIB,
            (0..num_verts).collect::<Vec<_>>(),
        ) {
            Err(attrib::Error::AlreadyExists(_)) => Ok(()),
            Err(e) => Err(e.into()),
            _ => Ok(()),
        }
    }

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    pub(crate) fn prepare_solid_attributes(mut solid: TetMeshSolid) -> Result<TetMeshSolid, Error> {
        let TetMeshSolid {
            tetmesh: ref mut mesh,
            ref mut material,
            ..
        } = &mut solid;

        *material = material.normalized();

        Self::prepare_source_index_attribute(mesh)?;

        Self::prepare_deformable_mesh_vertex_attributes(mesh)?;

        Self::prepare_deformable_tetmesh_attributes(mesh)?;

        {
            // Add elastic strain energy and elastic force attributes.
            // These will be computed at the end of the time step.
            mesh.set_attrib::<StrainEnergyType, CellIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
        }

        // Below we prepare attributes that give elasticity and density parameters. If such were
        // already provided on the mesh, then any given global parameters are ignored. This
        // behaviour is justified because variable material properties are most likely more
        // accurate and probably determined from a data driven method.

        Self::prepare_elasticity_attributes(&mut solid)?;

        Self::prepare_density_attribute(&mut solid)?;

        // Compute vertex masses.
        Self::compute_solid_vertex_masses(&mut solid);

        Ok(solid)
    }

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    pub(crate) fn prepare_shell_attributes(mut shell: TriMeshShell) -> Result<TriMeshShell, Error> {
        let TriMeshShell {
            trimesh: ref mut mesh,
            material,
        } = &mut shell;

        *material = material.normalized();

        Self::prepare_source_index_attribute(mesh)?;

        match material.properties {
            ShellProperties::Fixed => {
                // Kinematic meshes don't have material properties.
                Self::prepare_kinematic_mesh_vertex_attributes(mesh)?;
            }
            ShellProperties::Rigid { .. } => {
                Self::prepare_dynamic_mesh_vertex_attributes(mesh)?;
            }
            ShellProperties::Deformable { .. } => {
                Self::prepare_deformable_mesh_vertex_attributes(mesh)?;

                let ref_areas = Self::compute_ref_tri_areas(mesh)?;
                mesh.set_attrib_data::<RefAreaType, FaceIndex>(
                    REFERENCE_AREA_ATTRIB,
                    ref_areas.as_slice(),
                )?;

                {
                    // Add elastic strain energy attribute.
                    // This will be computed at the end of the time step.
                    mesh.set_attrib::<StrainEnergyType, FaceIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
                }

                // Below we prepare attributes that give elasticity and density parameters. If such were
                // already provided on the mesh, then any given global parameters are ignored. This
                // behaviour is justified because variable material properties are most likely more
                // accurate and probably determined from a data driven method.

                Self::prepare_elasticity_attributes(&mut shell)?;

                Self::prepare_density_attribute(&mut shell)?;

                // Compute vertex masses.
                Self::compute_shell_vertex_masses(&mut shell);
            }
        };

        Ok(shell)
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
    old_active_set: Vec<usize>,
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
    fn commit_solution(&mut self, and_warm_start: bool) {
        {
            let and_velocity = !self.sim_params.clear_velocity;
            let SolverDataMut {
                problem, solution, ..
            } = self.solver.solver_data_mut();

            // Advance internal state (positions and velocities) of the problem.
            problem.advance(solution.primal_variables, and_velocity, and_warm_start);
        }

        // Comitting solution. Reduce max_step for next iteration.
        let dt = self.time_step();
        let SolverDataMut {
            problem, solution, ..
        } = self.solver.solver_data_mut();
        if let Some(radius) = problem.min_contact_radius() {
            if and_warm_start {
                let step = inf_norm(problem.scaled_variables_iter(solution.primal_variables))
                    * if dt > 0.0 { dt } else { 1.0 };
                // If warm start is selected, then this solution was good, which means we're not
                // comitting it just for debugging purposes.
                let new_max_step = (step - radius).max(self.max_step * 0.5);
                self.max_step = new_max_step;
                problem.update_max_step(new_max_step);
                problem.reset_constraint_set();
            }
        }
    }

    ///// Revert previously committed solution. We just subtract step here.
    //fn revert_solution(
    //    problem: &mut NonLinearProblem,
    //    solution: Solution,
    //    old_prev_pos: Vec<Vector3<f64>>,
    //    old_prev_vel: Vec<Vector3<f64>>,
    //) {
    //    problem.revert_to(solution, old_prev_pos, old_prev_vel);
    //    //problem.reset_warm_start();
    //}

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
            ref mut old_active_set,
            ..
        } = self;
        old_active_set.clear();
        solver
            .solver_data()
            .problem
            .compute_active_constraint_set(old_active_set);
    }

    /// Determine if the inner step is acceptable. If unacceptable, adjust solver paramters and
    /// return false to indicate that the same step should be taken again.
    /// In either case this function modifies the configuration of the constraints and updates the
    /// constraint set. This means that the caller should take care to remap any constraint related
    /// values as needed. The configuration may also need to be reverted.
    fn check_inner_step(&mut self) -> bool {
        if self.problem().min_contact_radius().is_some() {
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
            dbg!(constraint_violation);
            dbg!(relative_tolerance);
            // NOTE: Ipopt can't detect constraint values below 1e-7 in absolute value. It seems to
            // be a hardcoded threshold.
            if constraint_violation > 1e-7_f64.max(relative_tolerance) {
                // intersecting objects detected (allow leeway via relative_tolerance)
                if self.max_step < step {
                    // Increase the max_step to be slightly bigger than the current step to avoid
                    // floating point issues.
                    println!("[softy] Increasing max step to {:e}", 1.1 * step);
                    self.update_max_step(1.1 * step);

                    // We don't commit the solution here because it may be far from the
                    // true solution, just redo the whole solve with the right
                    // neighbourhood information.
                    false
                } else {
                    println!("[softy] Max step: {:e} is saturated, but constraint is still violated, continuing...", step);
                    // The step is smaller than max_step and the constraint is still violated.
                    // Nothing else we can do, just accept the solution and move on.
                    true
                }
            } else {
                // The solution is good, reset the max_step, and continue.
                // TODO: There is room for optimization here. It may be better to reduce the max
                // step but not to zero to prevent extra steps in the future.
                println!("[softy] Solution accepted");
                true
            }
        } else {
            // No contact constraints, all solutions are good.
            true
        }
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

    fn remap_contacts(&mut self) {
        let Solver {
            solver,
            old_active_set,
            ..
        } = self;

        solver
            .solver_data_mut()
            .problem
            .remap_contacts(old_active_set.iter().cloned());
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

        // Recompute constraints since the active set may have changed if a collision mesh has moved.
        self.save_current_active_constraint_set();
        self.problem_mut().reset_constraint_set();

        // The number of friction solves to do.
        let mut friction_steps =
            vec![self.sim_params.friction_iterations; self.problem().num_frictional_contacts()];
        let total_friction_steps = friction_steps.iter().sum::<u32>();
        for _ in 0..self.sim_params.max_outer_iterations {
            // Remap contacts from the initial constraint reset above, or if the constraints were
            // updated after advection.
            self.remap_contacts();
            self.save_current_active_constraint_set();
            let step_result = self.inner_step();

            // The following block determines if after the inner step there were any changes
            // in the constraints where new points may have violated the constraint. If so we
            // update the constraint neighbourhoods and rerun the inner step.
            match step_result {
                Ok(step_result) => {
                    result = result.combine_inner_result(&step_result);
                    //{
                    //    // Output intermediate mesh
                    //    let SolverData {
                    //        problem, solution, ..
                    //    } = self.solver.solver_data();
                    //    let pos = problem.compute_step_from_unscaled_velocities(solution.primal_variables);
                    //    let mut tetmesh = self.borrow_mut_mesh();
                    //    let v = problem.scale_variables(solution.primal_variables);
                    //    tetmesh.set_attrib_data::<VelType, VertexIndex>("inter_vel", reinterpret_slice(&v))?;
                    //    for (&new_p, prev_p) in pos.iter().zip(tetmesh.vertex_positions_mut()) {
                    //        *prev_p = new_p.into();
                    //    }
                    //    geo::io::save_tetmesh(
                    //        &tetmesh,
                    //        &std::path::PathBuf::from(format!("./out/mesh_before_{}.vtk", self.step_count)),
                    //    )?;
                    //}

                    let step_acceptable = self.check_inner_step();

                    // Restore the constraints to original configuration.
                    self.problem_mut().reset_constraint_set();

                    if step_acceptable {
                        if !friction_steps.is_empty() && total_friction_steps > 0 {
                            debug_assert!(self
                                .problem()
                                .is_same_as_constraint_set(&self.old_active_set));
                            let is_finished = self.compute_friction_impulse(
                                &step_result.constraint_values,
                                &mut friction_steps,
                            );
                            if !is_finished {
                                continue;
                            }
                        }
                        self.commit_solution(true);
                        break;
                    }
                }
                Err(Error::InnerSolveError {
                    status,
                    iterations,
                    objective_value,
                }) => {
                    result = result.combine_inner_step_data(iterations, objective_value);
                    self.commit_solution(true);
                    return Err(Error::SolveError(status, result));
                }
                Err(e) => {
                    // Unknown error: Clear warm start and return.
                    self.commit_solution(false);
                    return Err(e);
                }
            }
        }

        // Remap contacts since after committing the solution, the constraint set may have changed.
        self.remap_contacts();

        if result.iterations > self.sim_params.max_outer_iterations {
            eprintln!(
                "WARNING: Reached max outer iterations: {:?}",
                result.iterations
            );
        }

        //self.output_meshes(self.step_count as u32);

        self.inner_iterations += result.inner_iterations as usize;
        self.step_count += 1;

        dbg!(self.inner_iterations);

        // On success, update the mesh with useful metrics.
        self.problem_mut().update_mesh_data();

        // Clear friction forces.
        self.problem_mut().clear_friction_impulses();

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
    use crate::test_utils::*;
    use crate::KernelType;
    use approx::*;
    use geo;
    use std::path::PathBuf;
    use utils::*;

    /// Utility function to compare positions of two meshes.
    fn compare_meshes(solution: &TetMesh, expected: &TetMesh, tol: f64) {
        for (pos, expected_pos) in solution
            .vertex_positions()
            .iter()
            .zip(expected.vertex_positions().iter())
        {
            for j in 0..3 {
                assert_relative_eq!(pos[j], expected_pos[j], max_relative = tol, epsilon = 1e-7);
            }
        }
    }

    /*
     * One tet tests
     */

    /// Helper function to generate a simple solver for one initially deformed tet under gravity.
    fn one_tet_solver() -> Solver {
        let mesh = make_one_deformed_tet_mesh();

        SolverBuilder::new(SimParams {
            print_level: 0,
            derivative_test: 0,
            ..STATIC_PARAMS
        })
        .add_solid(mesh, SOLID_MATERIAL)
        .build()
        .expect("Failed to build a solver for a one tet test.")
    }

    /// Test that the solver produces no change for an equilibrium configuration.
    #[test]
    fn one_tet_equilibrium_test() {
        let params = SimParams {
            gravity: [0.0f32, 0.0, 0.0],
            outer_tolerance: 1e-10, // This is a fairly strict tolerance.
            ..STATIC_PARAMS
        };

        let mesh = make_one_tet_mesh();

        let mut solver = SolverBuilder::new(params)
            .add_solid(mesh.clone(), SOLID_MATERIAL)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());

        // Expect the tet to remain in original configuration
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &mesh, 1e-6);
    }

    /// Test that the solver produces no change for an equilibrium configuration for a
    /// tetrahedralized box. This example also uses a softer material and a momentum term
    /// (dynamics enabled), which is more sensitive to perturbations.
    #[test]
    fn box_equilibrium_test() {
        let params = SimParams {
            gravity: [0.0f32, 0.0, 0.0],
            outer_tolerance: 1e-10, // This is a fairly strict tolerance.
            ..DYNAMIC_PARAMS
        };

        let soft_material = SolidMaterial::new(0)
            .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.49))
            .with_volume_preservation(false)
            .with_density(1000.0);

        // Box in equilibrium configuration should stay in equilibrium configuration
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk")).unwrap();

        let mut solver = SolverBuilder::new(params)
            .add_solid(mesh.clone(), soft_material)
            .build()
            .expect("Failed to create solver for soft box equilibrium test");
        assert!(solver.step().is_ok());

        // Expect the box to remain in original configuration
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &mesh, 1e-6);
    }

    /// Test one deformed tet under gravity fixed at two vertices. This is not an easy test because
    /// the initial condition is far from the solution and this is a fully static solve.
    /// This test has a unique solution.
    #[test]
    fn one_deformed_tet_test() {
        let mut solver = one_tet_solver();
        assert!(solver.step().is_ok());
        let solution = &solver.solid(0).tetmesh;

        let verts = solution.vertex_positions();

        // Check that the free verts are below the horizontal.
        assert!(verts[2][1] < 0.0 && verts[3][1] < 0.0);

        // Check that they are approximately at the same altitude.
        assert_relative_eq!(verts[2][1], verts[3][1], max_relative = 1e-3);
    }

    /// Test that subsequent outer iterations don't change the solution when Ipopt has converged.
    /// This is not the case with linearized constraints.
    #[test]
    fn one_tet_outer_test() -> Result<(), Error> {
        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            ..STATIC_PARAMS
        };

        let mesh = make_one_deformed_tet_mesh();

        let mut solver = SolverBuilder::new(params)
            .add_solid(mesh.clone(), SOLID_MATERIAL)
            .build()?;
        solver.step()?;

        let solution = &solver.solid(0).tetmesh;
        let mut expected_solver = one_tet_solver();
        expected_solver.step()?;
        let expected = &expected_solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn one_tet_volume_constraint_test() -> Result<(), Error> {
        let mesh = make_one_deformed_tet_mesh();

        let material = SOLID_MATERIAL.with_volume_preservation(true);

        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .add_solid(mesh, material)
            .build()?;
        solver.step()?;
        Ok(())
    }

    /*
     * Three tet tests
     */

    #[test]
    fn three_tets_static_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .add_solid(mesh, SOLID_MATERIAL)
            .build()?;
        solver.step()?;
        let solution = &solver.solid(0).tetmesh;
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_static_expected.vtk"))?;
        compare_meshes(&solution, &expected, 1e-3);
        Ok(())
    }

    #[test]
    fn three_tets_dynamic_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .add_solid(mesh, SOLID_MATERIAL)
            .build()?;
        solver.step()?;
        let solution = &solver.solid(0).tetmesh;
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_dynamic_expected.vtk"))?;
        compare_meshes(&solution, &expected, 1e-2);
        Ok(())
    }

    #[test]
    fn three_tets_static_volume_constraint_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let material = SOLID_MATERIAL.with_volume_preservation(true);
        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .add_solid(mesh, material)
            .build()?;
        solver.step()?;
        let solution = &solver.solid(0).tetmesh;
        let exptected = geo::io::load_tetmesh(&PathBuf::from(
            "assets/three_tets_static_volume_constraint_expected.vtk",
        ))?;
        compare_meshes(&solution, &exptected, 1e-4);
        Ok(())
    }

    #[test]
    fn three_tets_dynamic_volume_constraint_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let material = SOLID_MATERIAL.with_volume_preservation(true);
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .add_solid(mesh, material)
            .build()?;
        solver.step()?;
        let solution = &solver.solid(0).tetmesh;
        let expected = geo::io::load_tetmesh(&PathBuf::from(
            "assets/three_tets_dynamic_volume_constraint_expected.vtk",
        ))?;
        compare_meshes(&solution, &expected, 1e-2);
        Ok(())
    }

    #[test]
    fn animation_test() -> Result<(), Error> {
        let mut verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let mesh = make_three_tet_mesh_with_verts(verts.clone());

        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .add_solid(mesh, SOLID_MATERIAL)
            .build()?;

        for frame in 1u32..100 {
            let offset = 0.01 * (if frame < 50 { frame } else { 0 } as f64);
            verts.iter_mut().for_each(|x| (*x)[1] += offset);
            let pts = PointCloud::new(verts.clone());
            solver.update_solid_vertices(&pts)?;
            solver.step()?;
        }
        Ok(())
    }

    #[test]
    fn animation_volume_constraint_test() -> Result<(), Error> {
        let mut verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let mesh = make_three_tet_mesh_with_verts(verts.clone());

        let incompressible_material = SOLID_MATERIAL.with_volume_preservation(true);

        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .add_solid(mesh, incompressible_material)
            .build()?;

        for frame in 1u32..100 {
            let offset = 0.01 * (if frame < 50 { frame } else { 0 } as f64);
            verts.iter_mut().for_each(|x| (*x)[1] += offset);
            let pts = PointCloud::new(verts.clone());
            //save_tetmesh_ascii(
            //    solver.borrow_mesh(),
            //    &PathBuf::from(format!("./out/mesh_{}.vtk", frame)),
            //);
            solver.update_solid_vertices(&pts)?;
            solver.step()?;
        }
        Ok(())
    }

    const STRETCH_PARAMS: SimParams = SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..STATIC_PARAMS
    };

    fn medium_solid_material() -> SolidMaterial {
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
    }

    #[test]
    fn box_stretch_test() -> Result<(), Error> {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk"))?;
        let mut solver = SolverBuilder::new(SimParams {
            print_level: 0,
            derivative_test: 0,
            ..STRETCH_PARAMS
        })
        .add_solid(mesh, medium_solid_material())
        .build()?;
        solver.step()?;
        let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched.vtk"))?;
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn box_stretch_volume_constraint_test() -> Result<(), Error> {
        let incompressible_material = medium_solid_material().with_volume_preservation(true);
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk"))?;
        let mut solver = SolverBuilder::new(SimParams {
            print_level: 0,
            derivative_test: 0,
            ..STRETCH_PARAMS
        })
        .add_solid(mesh, incompressible_material)
        .build()?;
        solver.step()?;
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_const_volume.vtk"))?;
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn box_twist_test() -> Result<(), Error> {
        let material = medium_solid_material()
            .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0));
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .add_solid(mesh, material)
            .build()?;
        solver.step()?;
        let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted.vtk"))?;
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn box_twist_dynamic_volume_constraint_test() -> Result<(), Error> {
        let material = medium_solid_material()
            .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0))
            .with_volume_preservation(true);

        // We use a large time step to get the simulation to settle to the static sim with less
        // iterations.
        let params = SimParams {
            time_step: Some(2.0),
            ..DYNAMIC_PARAMS
        };

        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
        let mut solver = SolverBuilder::new(params.clone())
            .add_solid(mesh, material)
            .build()?;

        // The dynamic sim needs to settle
        for _ in 1u32..15 {
            let result = solver.step()?;
            assert!(
                result.iterations <= params.max_outer_iterations,
                "Unconstrained solver ran out of outer iterations."
            );
        }

        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-4);
        Ok(())
    }

    #[test]
    fn box_twist_volume_constraint_test() -> Result<(), Error> {
        let material = medium_solid_material()
            .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0))
            .with_volume_preservation(true);
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .add_solid(mesh, material)
            .build()?;
        solver.step()?;
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    /// This test insures that a non-linearized constraint like volume doesn't cause multiple outer
    /// iterations, and converges after the first solve.
    #[test]
    fn box_twist_volume_constraint_outer_test() -> Result<(), Error> {
        let material = medium_solid_material()
            .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0))
            .with_volume_preservation(true);

        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            ..STRETCH_PARAMS
        };

        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
        let mut solver = SolverBuilder::new(params)
            .add_solid(mesh, material)
            .build()?;
        let solve_result = solver.step()?;
        assert_eq!(solve_result.iterations, 1);

        // This test should produce the exact same mesh as the original
        // box_twist_volume_constraint_test
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
        let solution = &solver.solid(0).tetmesh;
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    /*
     * Tests with contact constraints
     */

    fn compute_contact_constraint(
        sample_mesh: &PolyMesh,
        tetmesh: &TetMesh,
        kernel: KernelType,
    ) -> Vec<f32> {
        use implicits::*;

        // There are currently two different ways to compute the implicit function representing the
        // contact constraint. Since this is a test we do it both ways and make sure the result is
        // the same. This doubles as a test for the implicits package.

        let mut trimesh_copy = sample_mesh.clone();
        let surface_trimesh = tetmesh.surface_trimesh();

        let params = implicits::Params {
            kernel,
            background_field: BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            },
            sample_type: SampleType::Face,
            ..Default::default()
        };

        let mut surface_polymesh = PolyMesh::from(surface_trimesh.clone());
        compute_potential_debug(&mut trimesh_copy, &mut surface_polymesh, params, || false)
            .expect("Failed to compute constraint value");

        let pot_attrib = trimesh_copy
            .attrib_clone_into_vec::<f32, VertexIndex>("potential")
            .expect("Potential attribute doesn't exist");

        {
            let surf = surface_from_trimesh(&surface_trimesh, params)
                .expect("Failed to build implicit surface.");

            let mut pot_attrib64 = vec![0.0f64; sample_mesh.num_vertices()];
            surf.potential(sample_mesh.vertex_positions(), &mut pot_attrib64)
                .expect("Failed to compute contact constraint potential.");

            // Make sure the two potentials are identical.
            println!("potential = {:?}", pot_attrib);
            println!("potential64 = {:?}", pot_attrib64);
            for (&x, &y) in pot_attrib.iter().zip(pot_attrib64.iter()) {
                assert_relative_eq!(x, y as f32, max_relative = 1e-5);
            }
        }

        pot_attrib.into_iter().map(|x| x as f32).collect()
    }

    #[test]
    fn tet_push_test() -> Result<(), Error> {
        // A triangle is being pushed on top of a tet.
        let height = 1.18032;
        let mut tri_verts = vec![
            [0.1, height, 0.0],
            [-0.05, height, 0.0866026],
            [-0.05, height, -0.0866026],
        ];

        let tri = vec![3, 0, 2, 1];

        let mut tetmesh = make_regular_tet();

        // Set fixed vertices
        tetmesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 1, 1, 1])?;

        let trimesh = PolyMesh::new(tri_verts.clone(), &tri);

        // Set contact parameters
        let kernel = KernelType::Approximate {
            radius_multiplier: 1.59,
            tolerance: 0.001,
        };

        //compute_contact_constraint(&trimesh, &tetmesh, radius, tolerance);

        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            max_outer_iterations: 20,
            ..STRETCH_PARAMS
        };

        let mut solver = SolverBuilder::new(params.clone())
            .add_solid(tetmesh.clone(), medium_solid_material().with_id(0))
            .add_fixed(trimesh.clone(), 1)
            .add_frictional_contact(
                FrictionalContactParams {
                    contact_type: ContactType::Point,
                    kernel,
                    friction_params: None,
                },
                (0, 1),
            )
            .build()?;

        let solve_result = solver.step()?;
        assert_eq!(solve_result.iterations, 1); // should be no more than one outer iteration

        // Expect no push since the triangle is outside the surface.
        for (pos, exp_pos) in solver
            .solid(0)
            .tetmesh
            .vertex_position_iter()
            .zip(tetmesh.vertex_positions().iter())
        {
            for i in 0..3 {
                assert_relative_eq!(pos[i], exp_pos[i], max_relative = 1e-5, epsilon = 1e-6);
            }
        }

        // Verify constraint, should be positive before push
        let constraint = compute_contact_constraint(&trimesh, &solver.solid(0).tetmesh, kernel);
        assert!(constraint.iter().all(|&x| x >= 0.0f32));

        // Simulate push
        let offset = 0.34;
        tri_verts.iter_mut().for_each(|x| (*x)[1] -= offset);
        let pts = PointCloud::new(tri_verts.clone());
        assert!(solver.update_shell_vertices(&pts).is_ok());
        let solve_result = solver.step()?;
        assert!(solve_result.iterations <= params.max_outer_iterations);

        // Verify constraint, should be positive after push
        let constraint = compute_contact_constraint(&trimesh, &solver.solid(0).tetmesh, kernel);
        assert!(constraint.iter().all(|&x| x >= -params.outer_tolerance));

        // Expect only the top vertex to be pushed down.
        let offset_verts = vec![
            [0.0, 0.629, 0.0],
            tetmesh.vertex_position(1),
            tetmesh.vertex_position(2),
            tetmesh.vertex_position(3),
        ];

        for (pos, exp_pos) in solver
            .solid(0)
            .tetmesh
            .vertex_position_iter()
            .zip(offset_verts.iter())
        {
            for i in 0..3 {
                assert_relative_eq!(pos[i], exp_pos[i], epsilon = 1e-3);
            }
        }

        Ok(())
    }

    fn ball_tri_push_tester(
        material: SolidMaterial,
        fc_params: FrictionalContactParams,
    ) -> Result<(), Error> {
        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball_fixed.vtk")).unwrap();

        let params = SimParams {
            max_iterations: 100,
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            max_outer_iterations: 200,
            ..DYNAMIC_PARAMS
        };

        let polymesh = geo::io::load_polymesh(&PathBuf::from("assets/tri.vtk"))?;
        let mut solver = SolverBuilder::new(params.clone())
            .add_solid(tetmesh, material.with_id(0))
            .add_fixed(polymesh, 1)
            .add_frictional_contact(fc_params, (0, 1))
            .build()?;

        let res = solver.step()?;
        //println!("res = {:?}", res);
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );
        Ok(())
    }

    //#[test]
    fn ball_tri_push_test() -> Result<(), Error> {
        let material =
            SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4));
        let fc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.812,
                tolerance: 0.07,
            },
            friction_params: None,
        };

        ball_tri_push_tester(material, fc_params)
    }

    //#[test]
    fn ball_tri_push_volume_constraint_test() -> Result<(), Error> {
        let material = SOLID_MATERIAL
            .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
            .with_volume_preservation(true);
        let fc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.812,
                tolerance: 0.07,
            },
            friction_params: None,
        };

        ball_tri_push_tester(material, fc_params)
    }

    fn ball_bounce_tester(
        material: SolidMaterial,
        fc_params: FrictionalContactParams,
        tetmesh: TetMesh,
    ) -> Result<(), Error> {
        let friction_iterations = if fc_params.friction_params.is_some() {
            1
        } else {
            0
        };
        let params = SimParams {
            max_iterations: 200,
            outer_tolerance: 0.1,
            max_outer_iterations: 20,
            gravity: [0.0f32, -9.81, 0.0],
            time_step: Some(0.0208333),
            friction_iterations,
            ..DYNAMIC_PARAMS
        };

        let mut grid = make_grid(Grid {
            rows: 4,
            cols: 4,
            orientation: AxisPlaneOrientation::ZX,
        });

        scale(&mut grid, [3.0, 1.0, 3.0].into());
        translate(&mut grid, [0.0, -3.0, 0.0].into());

        let mut solver = SolverBuilder::new(params.clone())
            .add_solid(tetmesh, material.with_id(0))
            .add_fixed(grid, 1)
            .add_frictional_contact(fc_params, (0, 1))
            .build()?;

        for _ in 0..50 {
            let res = solver.step()?;
            //println!("res = {:?}", res);
            //geo::io::save_tetmesh(
            //    &solver.borrow_mesh(),
            //    &PathBuf::from(format!("./out/mesh_{}.vtk", i)),
            //    ).unwrap();
            assert!(
                res.iterations <= params.max_outer_iterations,
                "Exceeded max outer iterations."
            );
        }

        Ok(())
    }

    #[test]
    fn ball_bounce_on_points_test() -> Result<(), Error> {
        let material =
            SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4));

        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.1,
                tolerance: 0.01,
            },
            friction_params: None,
        };

        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

        ball_bounce_tester(material, sc_params, tetmesh)
    }

    #[test]
    fn ball_bounce_on_points_volume_constraint_test() -> Result<(), Error> {
        let material = SOLID_MATERIAL
            .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
            .with_volume_preservation(true);

        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.1,
                tolerance: 0.01,
            },
            friction_params: None,
        };

        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

        ball_bounce_tester(material, sc_params, tetmesh)
    }

    /// Tet bouncing on an implicit surface. This is an easy test where the tet sees the entire
    /// local implicit surface.
    #[test]
    fn tet_bounce_on_implicit_test() -> Result<(), Error> {
        let material =
            SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));

        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Implicit,
            kernel: KernelType::Approximate {
                radius_multiplier: 20.0, // deliberately large radius
                tolerance: 0.0001,
            },
            friction_params: None,
        };

        let tetmesh = make_regular_tet();

        ball_bounce_tester(material, sc_params, tetmesh)
    }

    /// Ball bouncing on an implicit surface.
    #[test]
    fn ball_bounce_on_implicit_test() -> Result<(), Error> {
        let material =
            SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));

        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Implicit,
            kernel: KernelType::Approximate {
                radius_multiplier: 2.0,
                tolerance: 0.0001,
            },
            friction_params: None,
        };

        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

        ball_bounce_tester(material, sc_params, tetmesh)
    }

    /// Ball with constant volume bouncing on an implicit surface.
    #[test]
    fn ball_bounce_on_implicit_volume_constraint_test() -> Result<(), Error> {
        let material = SOLID_MATERIAL
            .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
            .with_volume_preservation(true);

        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Implicit,
            kernel: KernelType::Approximate {
                radius_multiplier: 2.0,
                tolerance: 0.0001,
            },
            friction_params: None,
        };

        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

        ball_bounce_tester(material, sc_params, tetmesh)
    }

    /// Ball bouncing on an implicit surface with staggered projections friction.
    #[test]
    fn ball_bounce_on_sp_implicit_test() -> Result<(), Error> {
        let material =
            SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));

        let sc_params = FrictionalContactParams {
            contact_type: ContactType::SPImplicit,
            kernel: KernelType::Approximate {
                radius_multiplier: 2.0,
                tolerance: 0.0001,
            },
            friction_params: Some(crate::friction::FrictionParams {
                dynamic_friction: 0.2,
                inner_iterations: 100,
                tolerance: 1e-5,
                print_level: 0,
            }),
        };

        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

        ball_bounce_tester(material, sc_params, tetmesh)
    }

    /*
     * More complex tests
     */

    #[cfg(not(debug_assertions))]
    mod complex_tests {
        use super::*;

        fn stiff_material() -> SolidMaterial {
            SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(1750e6, 10e6))
        }

        #[test]
        fn torus_medium_test() -> Result<(), Error> {
            let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
            let mut solver = SolverBuilder::new(SimParams {
                print_level: 0,
                ..DYNAMIC_PARAMS
            })
            .add_solid(mesh, stiff_material())
            .build()
            .unwrap();
            solver.step()?;
            Ok(())
        }

        #[test]
        fn torus_long_test() -> Result<(), Error> {
            let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk"))?;
            let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
                .add_solid(mesh, stiff_material())
                .build()?;

            for _i in 0..10 {
                //geo::io::save_tetmesh_ascii(
                //    &solver.borrow_mesh(),
                //    &PathBuf::from(format!("./out/mesh_{}.vtk", 1)),
                //    ).unwrap();
                solver.step()?;
            }
            Ok(())
        }
    }
}
