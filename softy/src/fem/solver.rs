use crate::attrib_defines::*;
use crate::constraints::*;
use crate::contact::*;
use crate::energy::*;
use crate::energy_models::{
    gravity::TetMeshGravity,
    inertia::TetMeshInertia,
    elasticity::tet_nh::{TetMeshNeoHookean, NeoHookeanTetEnergy},
};
use geo::math::{Matrix3, Vector3};
use geo::mesh::{topology::*, Attrib, VertexPositions};
use geo::ops::{ShapeMatrix, Volume};
use geo::prim::Tetrahedron;
use ipopt::{self, Ipopt, SolverData, SolverDataMut};
use reinterpret::*;
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};
use utils::soap::*;

use approx::*;

use crate::inf_norm;
use crate::mask_iter::*;

use super::{MuStrategy, NonLinearProblem, SimParams, Solution};
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
    solids: Vec<(TetMesh, Material)>,
    shells: Vec<(PolyMesh, Material)>,
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
    pub fn add_solid(&mut self, mesh: TetMesh, mat: Material) -> &mut Self {
        self.solids.push((mesh, mat));
        self
    }

    /// Add a polygon mesh representing a shell (e.g. cloth).
    pub fn add_shell(&mut self, shell: PolyMesh, mat: Material) -> &mut Self {
        self.shells.push((mesh, mat));
        self
    }

    /// Set parameters for frictional contact problems. The given two material IDs determine which
    /// materials should experience frictional contact described by the given parameters. To add
    /// self-contact, simply set the two ids to be equal.
    pub fn add_frictional_contact(&mut self, params: FrictionalContactParams, mat_ids: (usize, usize)) -> &mut Self {
        self.frictional_contacts.push((params, mat_ids));
        self
    }

    /// Build the simulation solver.
    pub fn build(&self) -> Result<Solver, Error> {
        let SolverBuilder {
            sim_params: mut params,
            solids,
            shells,
            frictional_contacts,
        } = self.clone();

        let mut max_size = 0.0;

        // Prepare solids
        let (solids, solid_vertex_set) = {
            let mut prev_pos = Chunked::new();
            let mut prev_vel = Chunked::new();

            for (mesh, _) in solids.iter() {
                // We need this quantity to compute the tolerance.
                let mut max_vol = 0.0;
                max_vol = max_vol.max(mesh
                            .tet_iter()
                            .map(Volume::volume)
                            .max_by(|a, b| a.partial_cmp(b).expect("Degenerate tetrahedron detected"))
                            .expect("Given TetMesh is empty"));
                max_size = max_size.max(max_vol.cbrt());

                // Get previous position vector from the tetmesh.
                let mesh_prev_pos = UniChunked::from_grouped_vec(mesh.vertex_positions().to_vec());
                prev_pos.push(mesh_prev_pos);
                let mesh_prev_vel = UniChunked::from_grouped_vec(mesh.attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?);
                prev_vel.push(mesh_prev_vel);
            }

            // Initialize volume constraint
            let mut volume_constraints = Vec::new();
            for (id, (mesh, material)) in solids.iter().enumerate() {
                if material.volume_preservation {
                    volume_constraints.push((id, VolumeConstraint::new(mesh)));
                }
            }

            let vertex_set = VertexSet {
                prev_pos: prev_pos.clone(),
                prev_vel: prev_vel.clone(),
                cur_pos: RefCell::new(prev_pos.clone()),
                scaled_vel: RefCell::new(prev_vel.clone()),
            };

            // Equip `TetMesh`es with physics parameters, making them bona-fide solids.
            let solids = solids.into_iter().map(|(mut tetmesh, material)| {
                // Prepare deformable solid for simulation.
                Self::prepare_mesh_attributes(&mut tetmesh, material)?;
                TetMeshSolid { tetmesh, material }
            }).collect::Vec<_>();

            (solids, vertex_set)
        }

        // Prepare shells
        let shells = {
            let mut prev_pos = Chunked::new();
            let mut prev_vel = Chunked::new();

            for (mesh, _) in shells.iter() {
                let mesh = TriMesh::from(mesh); // Triangulate mesh.

                // We need this quantity to compute the tolerance.
                let mut max_area = 0.0;
                max_area = max_area.max(mesh
                            .tri_iter()
                            .map(Area::area)
                            .max_by(|a, b| a.partial_cmp(b).expect("Degenerate triangle detected"))
                            .expect("Given TriMesh is empty"));

                max_size = max_size.max(max_area.sqrt());


                // Get previous position vector from the tetmesh.
                let mesh_prev_pos = UniChunked::from_grouped_vec(mesh.vertex_positions().to_vec());
                prev_pos.push(mesh_prev_pos);
                let mesh_prev_vel = UniChunked::from_grouped_vec(mesh.attrib_clone_into_vec::<VelType, VertexIndex>(VELOCITY_ATTRIB)?);
                prev_vel.push(mesh_prev_vel);
            }

            // Initialize volume constraint
            let mut volume_constraints = Vec::new();
            for (id, (mesh, material)) in solids.iter().enumerate() {
                if material.volume_preservation {
                    volume_constraints.push((id, VolumeConstraint::new(mesh)));
                }
            }

            let vertex_set = VertexSet {
                prev_pos: prev_pos.clone(),
                prev_vel: prev_vel.clone(),
                cur_pos: RefCell::new(prev_pos.clone()),
                scaled_vel: RefCell::new(prev_vel.clone()),
            };

            // Equip `TetMesh`es with physics parameters, making them bona-fide solids.
            let solids = solids.into_iter().map(|(mut tetmesh, material)| {
                // Prepare deformable solid for simulation.
                Self::prepare_mesh_attributes(&mut tetmesh, material)?;
                TetMeshSolid { tetmesh, material }
            }).collect::Vec<_>();

            (solids, vertex_set)

        }

        let gravity = [
            f64::from(params.gravity[0]),
            f64::from(params.gravity[1]),
            f64::from(params.gravity[2]),
        ];

        let time_step = f64::from(params.time_step.unwrap_or(0.0f32));

        let mut solid_surfaces = vec![None; solids.len()];

        // Convert frictional contact parameters into frictional contact constraints.
        for frictional_contacts = frictional_contacts.into_iter().map(|(params, ids)| {
            if let Some(solid) = solids.iter().find(|&&solid| solid.material.id == ids.0) {
            crate::constraints::build_contact_constraint(solids, shell, params)

        }).collect();
        let = kinematic_object.as_ref().and_then(|trimesh| {
            crate::constraints::build_contact_constraint(
                &mesh,
                &trimesh,
                cparams,
                density,
                time_step,
            )
            .ok()
        });

        let displacement_bound = None;
        //let displacement_bound = smooth_contact_params.map(|scp| {
        //    // Convert from a 2 norm bound (max_step) to an inf norm bound (displacement component
        //    // bound).
        //    scp.max_step / 3.0f64.sqrt()
        //});
        //
        let mut problem = NonLinearProblem {
            vertex_set,
            time_step,
            solids,
            solid_surfaces,
            shells,
            volume_constraints,
            gravity,
            smooth_contact_constraint,
            displacement_bound,
            interrupt_checker: Box::new(|| false),
            iterations: 0,
            warm_start: Solution::default(),
            initial_residual_error: std::f64::INFINITY,
            iter_counter: RefCell::new(0),
        };

        // Note that we don't need the solution field to get the number of variables and
        // constraints. This means we can use these functions to initialize solution.
        problem.reset_warm_start();

        let mut ipopt = Ipopt::new(problem)?;

        // Determine the true force tolerance. To start we base this tolerance on the elastic
        // response which depends on mu and lambda as well as per tet volume:
        // Larger stiffnesses and volumes cause proportionally larger gradients. Thus our tolerance
        // should reflect these properties.
        let max_area = max_size * max_size;
        let tol = f64::from(params.tolerance) * max_area * lambda.max(mu);
        params.tolerance = tol as f32;
        params.outer_tolerance *= (max_area * lambda.max(mu)) as f32;

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
            solid_material: Some(solid_material),
            max_step: 0.0,
            old_active_set: Vec::new(),
        })
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

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    pub(crate) fn prepare_mesh_attributes(mesh: &mut TetMesh, material: Material)
        -> Result<&mut TetMesh, Error>
    {
        let verts = mesh.vertex_positions().to_vec();

        mesh.attrib_or_add_data::<RefPosType, VertexIndex>(
            REFERENCE_POSITION_ATTRIB,
            verts.as_slice(),
        )?;

        mesh.attrib_or_add::<VelType, VertexIndex>(VELOCITY_ATTRIB, [0.0; 3])?;

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

        {
            // Add elastic strain energy and elastic force attributes.
            // These will be computed at the end of the time step.
            mesh.set_attrib::<StrainEnergyType, CellIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
            mesh.set_attrib::<ElasticForceType, VertexIndex>(ELASTIC_FORCE_ATTRIB, [0f64; 3])?;
        }

        // Below we prepare attributes that give elasticity and density parameters. If such were
        // already provided on the mesh, then any given global parameters are ignored. This
        // behaviour is justified because variable material properties are most likely more
        // accurate and probably determined from a data driven method.

        // Prepare elasticity parameters
        if let Some(elasticity) = material.elasticity {
            match mesh.add_attrib_data::<LambdaType, CellIndex>(
                LAMBDA_ATTRIB,
                vec![elasticity.lambda; mesh.num_cells()]
            ) {
                Err(e) => return Err(e);
                // if ok or already exists, everything is ok.
                Err(Error::AlreadyExists(_)) => { }
                _ => {} 
            }
            match mesh.add_attrib_data::<MuType, CellIndex>(
                MU_ATTRIB,
                vec![elasticity.mu; mesh.num_cells()]
            ) {
                Err(e) => return Err(e);
                // if ok or already exists, everything is ok.
                Err(Error::AlreadyExists(_)) => { }
                _ => {} 
            }
        } else {
            // No global elasticity parameters were given. Check that the mesh has the right
            // parameters.
            if mesh.attrib_check::<LambdaType, CellIndex>(LAMBDA_ATTRIB).is_err()
            || mesh.attrib_check::<MuType, CellIndex>(MU_ATTRIB).is_err() {
                   return Err(Error::MissingElasticityParams)
            }
        }

        // Prepare density parameter
        if let Some(density) = material.density {
            match mesh.add_attrib_data::<LambdaType, CellIndex>(
                LAMBDA_ATTRIB,
                vec![elasticity.lambda; mesh.num_cells()]
            ) {
                Err(e) => return Err(e);
                // if ok or already exists, everything is ok.
                Err(Error::AlreadyExists(_)) => { }
                _ => {} 
            }
        } else {
            // No global density parameter was given. Check that it exists on the mesh itself.
            if mesh.attrib_check::<DensityType, CellIndex>(DENSITY_ATTRIB).is_err() {
                return Err(Error::MissingDensityParam)
            }
        }

        Ok(mesh)
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
    /// Simulation paramters. This is kept around for convenience.
    sim_params: SimParams,
    /// Solid material properties.
    solid_material: Option<Material>,
    /// Maximal displaement length. Used to limit displacement which is necessary in contact
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
    pub fn set_interrupter(&mut self, checker: Box<FnMut() -> bool>) {
        self.problem_mut().interrupt_checker = checker;
    }

    /// Get an immutable borrow for the underlying `TriMesh` of the kinematic object.
    pub fn try_borrow_kinematic_mesh(&self) -> Option<Ref<'_, TriMesh>> {
        self.problem().kinematic_object.as_ref().map(|x| x.borrow())
        //match &self.solver.solver_data().problem.kinematic_object {
        //    Some(x) => Some(x.borrow()),
        //    None => None,
        //}
    }

    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &NonLinearProblem {
        self.solver.solver_data().problem
    }

    /// Get a mutable reference to the underlying problem.
    fn problem_mut(&mut self) -> &mut NonLinearProblem {
        self.solver.solver_data_mut().problem
    }

    /// Get an immutable borrow for the underlying `TetMesh`.
    pub fn borrow_mesh(&self) -> Ref<'_, TetMesh> {
        self.problem().tetmesh.borrow()
    }

    /// Get a mutable borrow for the underlying `TetMesh`.
    pub fn borrow_mut_mesh(&self) -> RefMut<'_, TetMesh> {
        self.problem().tetmesh.borrow_mut()
    }

    /// Get simulation parameters.
    pub fn params(&self) -> SimParams {
        self.sim_params.clone()
    }

    /// Get the solid material model (if any).
    pub fn solid_material(&self) -> Option<Material> {
        self.solid_material
    }

    /// Update the maximal displacement allowed. If zero, no limit is applied.
    pub fn update_max_step(&mut self, step: f64) {
        self.max_step = step;
        self.problem_mut().update_max_step(step);
    }
    pub fn update_radius_multiplier(&mut self, rad: f64) {
        self.problem_mut().update_radius_multiplier(rad);
    }

    /// Check the contact radius (valid in the presence of a contact constraint)
    pub fn contact_radius(&mut self) -> Option<f64> {
        self.problem_mut()
            .smooth_contact_constraint
            .as_ref()
            .map(|x| &**x) // remove box wrapper
            .map(ContactConstraint::contact_radius)
    }

    /// Update the solid meshes with the given points.
    pub fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        // TODO: Move this implementation to the problem.
        let problem = self.problem_mut();

        let mut prev_pos = problem.solid_vertex_set.prev_pos.borrow_mut();

        // All solids are simulated, so the input point set must have the same
        // size as our internal vertex set. If these are mismatched, then there
        // was an issue with constructing the solid meshes. This may not
        // necessarily be an error, we are just being conservative here.
        if pts.num_vertices() != prev_pos.view().into_flat().len() {
            // We got an invalid point cloud
            return Err(Error::SizeMismatch);
        }

        let new_pos = pts.vertex_positions();

        // Get the tetmesh and prev_pos so we can update the fixed vertices.
        for (solid, prev_pos) in problem.solids.borrow().zip(prev_pos.iter_mut()) {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = solid.tetmesh.attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let pts_iter = source_index_iter.map(|&idx| new_pos[src_idx]);

            // Only update fixed vertices, if no such attribute exists, return an error.
            let fixed_iter = solid.tetmesh.attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
            prev_pos
                .iter_mut()
                .zip(pts_iter)
                .zip(fixed_iter)
                .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
                .for_each(|(pos, new_pos)| *pos = *new_pos);
        }
        Ok(())
    }

    /// Update the shell meshes with the given points.
    pub fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        // TODO: Move this implementation to the problem.
        let problem = self.problem_mut();

        let mut prev_pos = problem.shell_vertex_set.prev_pos.borrow_mut();

        let new_pos = pts.vertex_positions();

        // Get the tetmesh and prev_pos so we can update the fixed vertices.
        for (shell, prev_pos) in problem.shells.borrow().zip(prev_pos.iter_mut()) {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = shell.trimesh.attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let pts_iter = source_index_iter.map(|&idx| new_pos[src_idx]);

            // Only update fixed vertices, if no such attribute exists, return an error.
            let fixed_iter = shell.trimesh.attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
            prev_pos
                .iter_mut()
                .zip(pts_iter)
                .zip(fixed_iter)
                .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
                .for_each(|(pos, new_pos)| *pos = *new_pos);
        }
        Ok(())
    }

    /// Given a tetmesh, compute the strain energy per tetrahedron.
    fn compute_strain_energy_attrib(mesh: &mut TetMesh, lambda: f64, mu: f64) {
        // Overwrite the "strain_energy" attribute.
        let mut strain = mesh
            .remove_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB)
            .unwrap();
        strain
            .iter_mut::<f64>()
            .unwrap()
            .zip(
                mesh.attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
            )
            .zip(
                mesh.attrib_iter::<RefShapeMtxInvType, CellIndex>(
                    REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                )
                .unwrap(),
            )
            .zip(mesh.tet_iter())
            .for_each(|(((strain, &vol), &ref_shape_mtx_inv), tet)| {
                *strain =
                    NeoHookeanTetEnergy::new(tet.shape_matrix(), ref_shape_mtx_inv, vol, lambda, mu)
                        .elastic_energy()
            });

        mesh.insert_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB, strain)
            .unwrap();
    }

    /// Given a tetmesh, compute the elastic forces per vertex.
    fn compute_elastic_forces(forces: &mut [Vector3<f64>], mesh: &TetMesh, lambda: f64, mu: f64) {
        // Reset forces
        for f in forces.iter_mut() {
            *f = Vector3::zeros();
        }

        let grad_iter = mesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                mesh.attrib_iter::<RefShapeMtxInvType, CellIndex>(
                    REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                )
                .unwrap(),
            )
            .zip(mesh.tet_iter())
            .map(|((&vol, &ref_shape_mtx_inv), tet)| {
                NeoHookeanTetEnergy::new(tet.shape_matrix(), ref_shape_mtx_inv, vol, lambda, mu)
                    .elastic_energy_gradient()
            });

        for (grad, cell) in grad_iter.zip(mesh.cells().iter()) {
            for j in 0..4 {
                forces[cell[j]] -= grad[j];
            }
        }
    }

    /// Given a tetmesh, compute the elastic forces per vertex, and save it at a vertex attribute.
    fn compute_elastic_forces_attrib(mesh: &mut TetMesh, lambda: f64, mu: f64) {
        let mut forces_attrib = mesh
            .remove_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB)
            .unwrap();

        {
            let forces: &mut [Vector3<f64>] =
                reinterpret_mut_slice(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

            Self::compute_elastic_forces(forces, mesh, lambda, mu);
        }

        // Reinsert forces back into the attrib map
        mesh.insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
    }

    fn add_friction_impulses_attrib(&mut self) {
        let impulses = self.problem().friction_impulse();
        let mut mesh = self.borrow_mut_mesh();
        mesh.set_attrib_data::<FrictionImpulseType, VertexIndex>(FRICTION_ATTRIB, &impulses)
            .ok();
    }

    fn add_contact_impulses_attrib(&mut self) {
        let impulses = self.problem().contact_impulse();
        let mut mesh = self.borrow_mut_mesh();
        mesh.set_attrib_data::<ContactImpulseType, VertexIndex>(CONTACT_ATTRIB, &impulses)
            .ok();
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

    /// Compute the gradient of the objective. We only consider unfixed vertices.  Panic if this fails.
    pub fn compute_objective_gradient(&self, grad: &mut [f64]) {
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

        // Erase fixed vert data. This doesn't contribute to the solve.
        let mesh = problem.tetmesh.borrow();
        let fixed_iter = mesh
            .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            .expect("Missing fixed verts attribute")
            .map(|&x| x != 0);
        let vert_grad: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);
        for g in vert_grad.iter_mut().filter_masked(fixed_iter) {
            *g = Vector3::zeros();
        }
    }

    /// Compute and add the Jacobian and constraint multiplier product to the given vector.
    pub fn add_constraint_jacobian_product(&self, jac_prod: &mut [f64]) {
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

        let jac_nnz = problem.num_constraint_jacobian_non_zeros();
        let mut rows = vec![0; jac_nnz];
        let mut cols = vec![0; jac_nnz];
        assert!(problem.constraint_jacobian_indices(&mut rows, &mut cols));

        let mut values = vec![0.0; jac_nnz];
        assert!(problem.constraint_jacobian_values(primal_variables, &mut values));

        // We don't consider values for fixed vertices.
        let mesh = problem.tetmesh.borrow();
        let fixed = mesh
            .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            .expect("Missing fixed verts attribute");

        assert_eq!(jac_prod.len(), primal_variables.len());
        // Effectively this is jac.transpose() * constraint_multipliers
        for ((row, col), val) in rows
            .into_iter()
            .zip(cols.into_iter())
            .zip(values.into_iter())
        {
            if fixed[(col as usize) / 3] == 0 {
                jac_prod[col as usize] += val * constraint_multipliers[row as usize];
            }
        }
    }

    fn dx(&self) -> &[f64] {
        self.solver.solver_data().solution.primal_variables
    }

    //fn lambda(&self) -> &[f64] {
    //    self.solver.solver_data().constraint_multipliers
    //}

    /// Update the `mesh` and `prev_pos` with the current solution.
    fn commit_solution(
        &mut self,
        and_warm_start: bool,
    ) -> (Solution, Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let res = {
            let and_velocity = !self.sim_params.clear_velocity;
            let SolverDataMut {
                problem, solution, ..
            } = self.solver.solver_data_mut();

            // Advance internal state (positions and velocities) of the problem.
            problem.advance(solution.primal_variables, and_velocity, and_warm_start)
        };

        // Comitting solution. Reduce max_step for next iteration.
        let dt = self.time_step();
        if let Some(radius) = self.contact_radius() {
            let SolverDataMut {
                problem, solution, ..
            } = self.solver.solver_data_mut();
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
        res
    }

    /// Revert previously committed solution. We just subtract step here.
    fn revert_solution(
        problem: &mut NonLinearProblem,
        solution: Solution,
        old_prev_pos: Vec<Vector3<f64>>,
        old_prev_vel: Vec<Vector3<f64>>,
    ) {
        problem.revert_to(solution, old_prev_pos, old_prev_vel);
        //problem.reset_warm_start();
    }

    fn output_meshes(&self, iter: u32) {
        let mesh = self.borrow_mesh();
        geo::io::save_tetmesh(
            &mesh,
            &std::path::PathBuf::from(format!("out/mesh_{}.vtk", iter + 1)),
        )
        .expect("Couldn't write to output tetrahedron mesh.");
        if let Some(mesh) = self.try_borrow_kinematic_mesh() {
            let polymesh = PolyMesh::from(mesh.clone());
            geo::io::save_polymesh(
                &polymesh,
                &std::path::PathBuf::from(format!("out/trimesh_{}.vtk", iter + 1)),
            )
            .expect("Couldn't write to output triangle lmesh.");
        }
    }

    fn compute_objective(&self) -> f64 {
        self.compute_objective_dx(self.solver.solver_data().solution.primal_variables)
    }

    fn compute_objective_dx(&self, dx: &[f64]) -> f64 {
        self.problem().objective_value(dx)
    }

    fn model_l1(&self) -> f64 {
        self.model_l1_dx(self.solver.solver_data().solution.primal_variables)
    }

    fn model_l1_dx(&self, dx: &[f64]) -> f64 {
        self.problem().linearized_constraint_violation_model_l1(dx)
    }

    fn merit_l1(&self) -> f64 {
        self.merit_l1_dx(self.solver.solver_data().solution.primal_variables)
    }

    fn merit_l1_dx(&self, _dx: &[f64]) -> f64 {
        0.0
        //self.problem().linearized_constraint_violation_l1(dx)
    }

    fn compute_residual(&self, residual: &mut Vec<f64>) -> f64 {
        use ipopt::{BasicProblem, ConstrainedProblem};
        let num_variables = self.problem().num_variables();
        let num_constraints = self.problem().num_constraints();

        // Reset the residual.
        residual.clear();
        residual.resize(num_variables + num_constraints, 0.0);

        let (objective_residual, constraint_violation) = residual.split_at_mut(num_variables);

        self.compute_objective_gradient(objective_residual);
        self.add_constraint_jacobian_product(objective_residual);
        self.compute_constraint_violation(constraint_violation);

        let constraint_violation_norm = inf_norm(constraint_violation.iter().cloned());
        let residual_norm = inf_norm(residual.iter().cloned());

        println!(
            "residual = {:?}, cv = {:?}",
            residual_norm, constraint_violation_norm
        );

        residual_norm
    }

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
        if self.contact_radius().is_some() {
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
    fn compute_friction_impulse(&mut self, constraint_values: &[f64], friction_steps: u32) -> u32 {
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
        // TODO: This doesn't actually work because Ipopt holds a lock to the logfile, so we can
        // fix this by either writing to a different log file, or prepending this info to the
        // beginning of the solve before ipopt gets the lock since this info won't change between
        // solves.
        if let Some(ref log) = self.sim_params.log_file {
            if let Ok(ref mut file) = std::fs::OpenOptions::new()
                .write(true)
                .append(true)
                .open(log)
            {
                use std::io::Write;
                writeln!(file, "Params = {:#?}", self.sim_params)?;
                writeln!(file, "Material = {:#?}", self.solid_material)?;
            }
        }

        println!("Params = {:#?}", self.sim_params);
        println!("Material = {:#?}", self.solid_material);

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
        let mut friction_steps = self.sim_params.friction_iterations;
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
                        if friction_steps > 0 {
                            debug_assert!(self
                                .problem()
                                .is_same_as_constraint_set(&self.old_active_set));
                            friction_steps = self.compute_friction_impulse(
                                &step_result.constraint_values,
                                friction_steps,
                            );
                            if friction_steps > 0 {
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

        // On success, update the mesh with new positions and useful metrics.
        let (lambda, mu) = self.solid_material.unwrap().elasticity.lame_parameters();

        // Write back friction impulses
        self.add_friction_impulses_attrib();
        self.add_contact_impulses_attrib();

        // Clear friction forces.
        self.problem_mut().clear_friction_impulses();

        let mut mesh = self.borrow_mut_mesh();

        // Write back elastic strain energy for visualization.
        Self::compute_strain_energy_attrib(&mut mesh, lambda, mu);

        // Write back elastic forces on each node.
        Self::compute_elastic_forces_attrib(&mut mesh, lambda, mu);

        Ok(result)
    }

    /// Run the optimization solver on one time step. This method uses the trust region method to
    /// resolve linearized constraints.
    pub fn step_tr(&mut self) -> Result<SolveResult, Error> {
        use ipopt::BasicProblem;

        println!("params = {:?}", self.sim_params);
        println!("material = {:?}", self.solid_material);

        // Initialize the result of this function.
        let mut result = SolveResult {
            max_inner_iterations: 0,
            inner_iterations: 0,
            iterations: 0,
            objective_value: 0.0,
        };

        let mut residual = Vec::new();
        let mut residual_norm = self.compute_residual(&mut residual);
        let mut objective_gradient = vec![0.0; self.problem().num_variables()];

        let zero_dx = vec![0.0; self.dx().len()];

        let rho = 0.1;

        self.output_meshes(0);

        // We should iterate until a relative residual goes to zero.
        for iter in 0..self.sim_params.max_outer_iterations {
            let step_result = self.inner_step();

            let f_k = self.compute_objective_dx(&zero_dx);
            let c_k = self.merit_l1_dx(&zero_dx);
            assert_relative_eq!(c_k, self.model_l1_dx(&zero_dx));

            let mu = if relative_eq!(c_k, 0.0) {
                0.0
            } else {
                let fdx = self.compute_objective_gradient_product(&mut objective_gradient);
                println!("fdx = {:?}", fdx);
                (fdx / ((1.0 - rho) * c_k)).max(0.0)
            };
            println!("mu = {:?}", mu);

            let prev_merit = f_k + mu * c_k;
            let prev_model = prev_merit;

            let f_k1 = self.compute_objective();

            let merit_model = f_k1 + mu * self.model_l1();
            let merit_value = f_k1 + mu * self.merit_l1();

            // Since we are using linearized constraints, we compute the true
            // residual and iterate until it vanishes. For unconstrained problems or problems with
            // no linearized constraints, only one step is needed.
            residual_norm = self.compute_residual(&mut residual);

            // Commit the solution whether or not there is an error. In case of error we will be
            // able to investigate the result.
            let (old_sol, old_prev_pos, old_prev_vel) = self.commit_solution(true);

            self.output_meshes(iter + 1);

            // Update output result with new data.
            match step_result {
                Ok(step_result) => {
                    result = result.combine_inner_result(&step_result);
                }
                Err(Error::InnerSolveError {
                    status,
                    iterations,
                    objective_value,
                }) => {
                    // If the problem is infeasible, it may be that our step size is too small,
                    // Try to increase it (but not past max_step) and try again:
                    if status == ipopt::SolveStatus::InfeasibleProblemDetected {
                        let max_step = self.max_step;
                        let problem = self.problem_mut();
                        if let Some(disp) = problem.displacement_bound {
                            problem.displacement_bound.replace(max_step.min(1.5 * disp));
                            continue;
                        }
                    } else {
                        // Otherwise we don't know what else to do so return an error.
                        result = result.combine_inner_step_data(iterations, objective_value);
                        return Err(Error::SolveError(status, result));
                    }
                }
                Err(e) => {
                    // Unknown error: Reset warm start and return.
                    self.solver.set_option("warm_start_init_point", "no");
                    return Err(e);
                }
            }

            if residual_norm <= f64::from(self.sim_params.outer_tolerance) {
                break;
            }

            // Check that the objective is actually lowered. Trust region method for constraints.
            // The justification is that our constraint violation estimate is much worse than the
            // energy estimate from the solve of the inner problem. As such we will iterate on the
            // infinity norm of all violated linearized constraints (constraint points outside the
            // feasible domain).
            {
                let max_step = self.max_step;
                let SolverDataMut {
                    problem, solution, ..
                } = self.solver.solver_data_mut();

                if max_step > 0.0 {
                    if let Some(disp) = problem.displacement_bound {
                        println!(
                            "f_k = {:?}, f_k+1 = {:?}, m_k = {:?}, m_k+1 = {:?}",
                            prev_merit, merit_value, prev_model, merit_model
                        );
                        let reduction = (prev_merit - merit_value) / (prev_model - merit_model);
                        println!("reduction = {:?}", reduction);

                        let step_size = inf_norm(solution.primal_variables.iter().cloned());

                        if reduction < 0.25 {
                            if step_size < 1e-5 * max_step {
                                break; // Reducing the step wont help at this point
                            }
                            // The constraint violation is not decreasing, roll back and reduce the step size.
                            println!("step size taken {:?}", step_size);
                            problem.displacement_bound.replace(0.25 * step_size);
                            println!("reducing step to {:?}", problem.displacement_bound.unwrap());
                        } else {
                            println!("step size = {:?} vs. disp = {:?}", step_size, disp);
                            if reduction > 0.75 && relative_eq!(step_size, disp) {
                                // we took a full step
                                // The linearized constraints are a good model of the actual constraints,
                                // increase the step size and continue.
                                problem.displacement_bound.replace(max_step.min(2.0 * disp));
                                println!(
                                    "increase step to {:?}",
                                    problem.displacement_bound.unwrap()
                                );
                            } // Otherwise keep the step size the same.
                        }
                        if reduction < 0.15 {
                            // Otherwise constraint violation is not properly decreasing
                            Self::revert_solution(problem, old_sol, old_prev_pos, old_prev_vel);
                        }
                    }
                }
            }
        }

        if result.iterations > self.sim_params.max_outer_iterations {
            eprintln!(
                "WARNING: Reached max outer iterations: {:?}\nResidual is: {:?}",
                result.iterations, residual_norm
            );
        }

        self.step_count += 1;

        // On success, update the mesh with new positions and useful metrics.
        let (lambda, mu) = self.solid_material.unwrap().elasticity.lame_parameters();

        let mut mesh = self.borrow_mut_mesh();

        // Write back elastic strain energy for visualization.
        Self::compute_strain_energy_attrib(&mut mesh, lambda, mu);

        // Write back elastic forces on each node.
        Self::compute_elastic_forces_attrib(&mut mesh, lambda, mu);

        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use crate::KernelType;
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
        .solid_material(SOLID_MATERIAL)
        .add_solid(mesh)
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
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh.clone())
            .build()
            .unwrap();
        assert!(solver.step().is_ok());

        // Expect the tet to remain in original configuration
        let solution = solver.borrow_mesh();
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

        let soft_material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.49),
            incompressibility: false,
            density: 1000.0,
            damping: 0.0,
        };

        // Box in equilibrium configuration should stay in equilibrium configuration
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk")).unwrap();

        let mut solver = SolverBuilder::new(params)
            .solid_material(soft_material)
            .add_solid(mesh.clone())
            .build()
            .expect("Failed to create solver for soft box equilibrium test");
        assert!(solver.step().is_ok());

        // Expect the box to remain in original configuration
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &mesh, 1e-6);
    }

    /// Test one deformed tet under gravity fixed at two vertices. This is not an easy test because
    /// the initial condition is far from the solution and this is a fully static solve.
    /// This test has a unique solution.
    #[test]
    fn one_deformed_tet_test() {
        let mut solver = one_tet_solver();
        assert!(solver.step().is_ok());
        let solution = solver.borrow_mesh();
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
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh.clone())
            .build()?;
        solver.step()?;

        let solution = solver.borrow_mesh();
        let mut expected_solver = one_tet_solver();
        expected_solver.step()?;
        let expected = expected_solver.borrow_mesh();
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn one_tet_volume_constraint_test() -> Result<(), Error> {
        let mesh = make_one_deformed_tet_mesh();

        let material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };

        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
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
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()?;
        solver.step()?;
        let solution = solver.borrow_mesh();
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_static_expected.vtk"))?;
        compare_meshes(&solution, &expected, 1e-3);
        Ok(())
    }

    #[test]
    fn three_tets_dynamic_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()?;
        solver.step()?;
        let solution = solver.borrow_mesh();
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_dynamic_expected.vtk"))?;
        compare_meshes(&solution, &expected, 1e-2);
        Ok(())
    }

    #[test]
    fn three_tets_static_volume_constraint_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };
        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()?;
        solver.step()?;
        let solution = solver.borrow_mesh();
        let exptected = geo::io::load_tetmesh(&PathBuf::from(
            "assets/three_tets_static_volume_constraint_expected.vtk",
        ))?;
        compare_meshes(&solution, &exptected, 1e-4);
        Ok(())
    }

    #[test]
    fn three_tets_dynamic_volume_constraint_test() -> Result<(), Error> {
        let mesh = make_three_tet_mesh();
        let material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()?;
        solver.step()?;
        let solution = solver.borrow_mesh();
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
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
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

        let incompressible_material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };

        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(incompressible_material)
            .add_solid(mesh)
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

    const MEDIUM_SOLID_MATERIAL: Material = Material {
        elasticity: ElasticityParameters {
            bulk_modulus: 300e6,
            shear_modulus: 100e6,
        },
        ..SOLID_MATERIAL
    };

    #[test]
    fn box_stretch_test() -> Result<(), Error> {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk"))?;
        let mut solver = SolverBuilder::new(SimParams {
            print_level: 0,
            derivative_test: 0,
            ..STRETCH_PARAMS
        })
        .solid_material(MEDIUM_SOLID_MATERIAL)
        .add_solid(mesh)
        .build()?;
        solver.step()?;
        let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched.vtk"))?;
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn box_stretch_volume_constraint_test() -> Result<(), Error> {
        let incompressible_material = Material {
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk"))?;
        let mut solver = SolverBuilder::new(SimParams {
            print_level: 0,
            derivative_test: 0,
            ..STRETCH_PARAMS
        })
        .solid_material(incompressible_material)
        .add_solid(mesh)
        .build()?;
        solver.step()?;
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_const_volume.vtk"))?;
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn box_twist_test() -> Result<(), Error> {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            ..MEDIUM_SOLID_MATERIAL
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()?;
        solver.step()?;
        let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted.vtk"))?;
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    #[test]
    fn box_twist_dynamic_volume_constraint_test() -> Result<(), Error> {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };

        // We use a large time step to get the simulation to settle to the static sim with less
        // iterations.
        let params = SimParams {
            time_step: Some(2.0),
            ..DYNAMIC_PARAMS
        };

        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
        let mut solver = SolverBuilder::new(params.clone())
            .solid_material(material)
            .add_solid(mesh)
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
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected, 1e-4);
        Ok(())
    }

    #[test]
    fn box_twist_volume_constraint_test() -> Result<(), Error> {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()?;
        solver.step()?;
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected, 1e-6);
        Ok(())
    }

    /// This test insures that a non-linearized constraint like volume doesn't cause multiple outer
    /// iterations, and converges after the first solve.
    #[test]
    fn box_twist_volume_constraint_outer_test() -> Result<(), Error> {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };

        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            ..STRETCH_PARAMS
        };

        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
        let mut solver = SolverBuilder::new(params)
            .solid_material(material)
            .add_solid(mesh)
            .build()?;
        let solve_result = solver.step()?;
        assert_eq!(solve_result.iterations, 1);

        // This test should produce the exact same mesh as the original
        // box_twist_volume_constraint_test
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
        let solution = solver.borrow_mesh();
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
            .solid_material(MEDIUM_SOLID_MATERIAL)
            .add_solid(tetmesh.clone())
            .add_shell(trimesh.clone())
            .smooth_contact_params(FrictionalContactParams {
                contact_type: ContactType::Point,
                kernel,
                friction_params: None,
            })
            .build()?;

        let solve_result = solver.step()?;
        assert_eq!(solve_result.iterations, 1); // should be no more than one outer iteration

        // Expect no push since the triangle is outside the surface.
        for (pos, exp_pos) in solver
            .borrow_mesh()
            .vertex_position_iter()
            .zip(tetmesh.vertex_positions().iter())
        {
            for i in 0..3 {
                assert_relative_eq!(pos[i], exp_pos[i], max_relative = 1e-5, epsilon = 1e-6);
            }
        }

        // Verify constraint, should be positive before push
        let constraint = compute_contact_constraint(&trimesh, &solver.borrow_mesh(), kernel);
        assert!(constraint.iter().all(|&x| x >= 0.0f32));

        // Simulate push
        let offset = 0.34;
        tri_verts.iter_mut().for_each(|x| (*x)[1] -= offset);
        let pts = PointCloud::new(tri_verts.clone());
        assert!(solver.update_shell_vertices(&pts).is_ok());
        let solve_result = solver.step()?;
        assert!(solve_result.iterations <= params.max_outer_iterations);

        // Verify constraint, should be positive after push
        let constraint = compute_contact_constraint(&trimesh, &solver.borrow_mesh(), kernel);
        assert!(constraint.iter().all(|&x| x >= -params.outer_tolerance));

        // Expect only the top vertex to be pushed down.
        let offset_verts = vec![
            [0.0, 0.629, 0.0],
            tetmesh.vertex_position(1),
            tetmesh.vertex_position(2),
            tetmesh.vertex_position(3),
        ];

        for (pos, exp_pos) in solver
            .borrow_mesh()
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
        material: Material,
        sc_params: FrictionalContactParams,
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
            .solid_material(material)
            .add_solid(tetmesh)
            .add_shell(polymesh)
            .smooth_contact_params(sc_params)
            .build()?;

        let res = solver.step()?;
        //println!("res = {:?}", res);
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );
        Ok(())
    }

    #[test]
    fn ball_tri_push_test() -> Result<(), Error> {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e6, 0.4),
            ..SOLID_MATERIAL
        };
        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.812,
                tolerance: 0.07,
            },
            friction_params: None,
        };

        ball_tri_push_tester(material, sc_params)
    }

    #[test]
    fn ball_tri_push_volume_constraint_test() -> Result<(), Error> {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e6, 0.4),
            incompressibility: true,
            ..SOLID_MATERIAL
        };
        let sc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.812,
                tolerance: 0.07,
            },
            friction_params: None,
        };

        ball_tri_push_tester(material, sc_params)
    }

    fn ball_bounce_tester(
        material: Material,
        sc_params: FrictionalContactParams,
        tetmesh: TetMesh,
    ) -> Result<(), Error> {
        let friction_iterations = if sc_params.friction_params.is_some() {
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
            .solid_material(material)
            .add_solid(tetmesh)
            .add_shell(grid)
            .smooth_contact_params(sc_params)
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
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e6, 0.4),
            ..SOLID_MATERIAL
        };

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
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e6, 0.4),
            incompressibility: true,
            ..SOLID_MATERIAL
        };

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
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e5, 0.4),
            ..SOLID_MATERIAL
        };

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
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e5, 0.4),
            ..SOLID_MATERIAL
        };

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
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e6, 0.4),
            incompressibility: true,
            ..SOLID_MATERIAL
        };

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
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(10e5, 0.4),
            ..SOLID_MATERIAL
        };

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

    const STIFF_MATERIAL: Material = Material {
        elasticity: ElasticityParameters {
            bulk_modulus: 1750e6,
            shear_modulus: 10e6,
        },
        ..SOLID_MATERIAL
    };

    #[test]
    fn torus_medium_test() -> Result<(), Error> {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        let mut solver = SolverBuilder::new(SimParams {
            print_level: 0,
            ..DYNAMIC_PARAMS
        })
        .solid_material(STIFF_MATERIAL)
        .add_solid(mesh)
        .build()
        .unwrap();
        solver.step()?;
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn torus_long_test() -> Result<(), Error> {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk"))?;
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(STIFF_MATERIAL)
            .add_solid(mesh)
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
