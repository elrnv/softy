use crate::attrib_defines::*;
use crate::constraints::*;
use crate::energy::*;
use crate::energy_models::{
    gravity::Gravity,
    momentum::MomentumPotential,
    volumetric_neohookean::{ElasticTetMeshEnergyBuilder, NeoHookeanTetEnergy},
};
use crate::geo::math::{Matrix3, Vector3};
use crate::geo::mesh::{VertexPositions, tetmesh::TetCell, topology::*, Attrib};
use crate::geo::ops::{ShapeMatrix, Volume};
use crate::geo::prim::Tetrahedron;
use ipopt::{self, Ipopt, SolverDataMut};
use reinterpret::*;
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use crate::mask_iter::*;

use super::NonLinearProblem;
use super::SimParams;
use crate::Error;
use crate::PointCloud;
use crate::TetMesh;
use crate::PolyMesh;
use crate::TriMesh;

/// Result from one inner simulation step.
#[derive(Debug)]
pub struct InnerSolveResult {
    /// Number of inner iterations in one step.
    pub iterations: u32,
    /// The value of the objective at the end of the step.
    pub objective_value: f64,
}

/// Result from one simulation step.
#[derive(Debug)]
pub struct SolveResult {
    /// Maximum number of inner iterations during one outer step.
    pub max_inner_iterations: u32,
    /// Number of outer iterations of the step.
    pub iterations: u32,
    /// The value of the objective at the end of the time step.
    pub objective_value: f64,
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Iterations: {}\nObjective: {}\nMax Inner Iterations: {}",
               self.iterations, self.objective_value, self.max_inner_iterations)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ElasticityParameters {
    /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
    /// incompressibility. The larger the value, the more incompressible the material is.
    /// Think of this as "Volume Stiffness".
    pub bulk_modulus: f32,
    /// Shear modulus measures the material's resistance to shear deformation. The larger the
    /// value, the more it resists changes in shape. Think of this as "Shape Stiffness".
    pub shear_modulus: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material {
    /// Parameters determining the elastic behaviour of a simulated solid.
    pub elasticity: ElasticityParameters,
    /// Incompressibility sets the material to be globally incompressible, if set to `true`. In
    /// contrast to `elasticity.bulk_modulus`, this parameter affects global incompressibility,
    /// while `bulk_modulus` affects *local* incompressibility (on a per element level).
    pub incompressibility: bool,
    /// The density of the material.
    pub density: f32,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model.
    pub damping: f32,
}

impl ElasticityParameters {
    pub fn from_young_poisson(young: f32, poisson: f32) -> Self {
        ElasticityParameters {
            bulk_modulus: young / (3.0 * (1.0 - 2.0 * poisson)),
            shear_modulus: young / (2.0 * (1.0 + poisson)),
        }
    }

    /// Convert internal elasticity parameters to Lame parameters for Neo-Hookean energy.
    pub fn lame_parameters(&self) -> (f64, f64) {
        let lambda = self.bulk_modulus as f64 - 2.0 * self.shear_modulus as f64 / 3.0;
        let mu = self.shear_modulus as f64;
        (lambda, mu)
    }
}

/// Get reference tetrahedron.
/// This routine assumes that there is a vertex attribute called `ref` of type `[f64;3]`.
pub fn ref_tet(tetmesh: &TetMesh, indices: &TetCell) -> Tetrahedron<f64> {
    let ref_pos = tetmesh
        .attrib::<VertexIndex>(REFERENCE_POSITION_ATTRIB)
        .unwrap()
        .as_slice::<[f64; 3]>()
        .unwrap();
    Tetrahedron::from_indexed_slice(indices.get(), ref_pos)
}

#[derive(Clone, Debug)]
pub struct SolverBuilder {
    sim_params: SimParams,
    solid_material: Option<Material>,
    solids: Vec<TetMesh>,
    shells: Vec<PolyMesh>,
    smooth_contact_params: Option<SmoothContactParams>,
}

impl SolverBuilder {
    /// Create a `SolverBuilder` with the minimum required parameters, which are the simulation
    /// parameters, `SimParams`.
    pub fn new(sim_params: SimParams) -> Self {
        SolverBuilder {
            sim_params,
            solid_material: None,
            solids: Vec::new(),
            shells: Vec::new(),
            smooth_contact_params: None,
        }
    }

    /// Set the solid material properties. This will override any variable material properties set
    /// on the mesh itself.
    pub fn solid_material(&mut self, mat_props: Material) -> &mut Self {
        self.solid_material = Some(mat_props);
        self
    }

    /// Add a tetmesh representing a soft solid (e.g. soft tissue).
    pub fn add_solid(&mut self, solid: TetMesh) -> &mut Self {
        self.solids.push(solid);
        self
    }

    /// Add a polygon mesh representing a shell (e.g. cloth).
    pub fn add_shell(&mut self, shell: PolyMesh) -> &mut Self {
        self.shells.push(shell);
        self
    }

    /// Set parameters for smooth contact problems. This is necessary when a shell is provided.
    pub fn smooth_contact_params(&mut self, params: SmoothContactParams) -> &mut Self {
        self.smooth_contact_params = Some(params);
        self
    }

    /// Build the simulation solver.
    pub fn build(&self) -> Result<Solver, Error> {
        let SolverBuilder {
            sim_params: mut params,
            solid_material,
            solids,
            shells,
            smooth_contact_params,
            ..
        } = self;

        // Get kinematic shell
        let kinematic_object = if shells.len() > 0 {
            if smooth_contact_params.is_none() {
                return Err(Error::MissingContactParams);
            }

            Some(Rc::new(RefCell::new(TriMesh::from(shells[0].clone()))))
        } else {
            None
        };

        // TODO: add support for more solids.
        if solids.len() == 0 {
            return Err(Error::NoSimulationMesh);
        }

        // Get deformable solid.
        let mut mesh = solids[0].clone();

        // We need this quantity to compute the tolerance.
        let max_vol = mesh.tet_iter().map(|tet| tet.volume()).max_by(|a,b| a.partial_cmp(b)
                                                                     .expect("Degenerate tetrahedron detected"))
            .expect("Given TetMesh is empty");

        // Prepare deformable solid for simulation.
        Self::prepare_mesh_attributes(&mut mesh)?;

        // Get previous position vector from the tetmesh.
        let prev_pos = Rc::new(RefCell::new(
            reinterpret_slice(mesh.vertex_positions()).to_vec()
        ));

        let solid_material = solid_material.unwrap(); // TODO: implement variable material properties

        // Retrieve lame parameters.
        let (lambda, mu) = solid_material.elasticity.lame_parameters();

        // Normalize material parameters with mu.
        let lambda = lambda / mu;
        let density = solid_material.density as f64 / mu;
        // premultiply damping by timestep reciprocal.
        let damping = if let Some(dt) = params.time_step {
            if dt != 0.0 {
                1.0 / dt as f64
            } else {
                0.0
            }
        } else {
            0.0
        } * solid_material.damping as f64
            / mu;

        // All values are normalized by mu including mu itself so it becomes 1.0.
        let mu = 1.0;

        let gravity = [
            params.gravity[0] as f64,
            params.gravity[1] as f64,
            params.gravity[2] as f64,
        ];

        // Initialize volume constraint
        let volume_constraint = if solid_material.incompressibility {
            Some(VolumeConstraint::new(&mesh))
        } else {
            None
        };

        // Lift mesh into the heap to be shared among energies and constraints.
        let mesh = Rc::new(RefCell::new(mesh));

        let energy_model_builder =
            ElasticTetMeshEnergyBuilder::new(Rc::clone(&mesh)).material(lambda, mu, damping);

        let momentum_potential = params
            .time_step
            .map(|dt| MomentumPotential::new(Rc::clone(&mesh), density, dt as f64));

        let smooth_contact_constraint = kinematic_object.as_ref()
            .map(|trimesh| LinearSmoothContactConstraint::new(&mesh, &trimesh, smooth_contact_params.unwrap()));

        let displacement_bound = smooth_contact_params.map(|scp| {
            // Convert from a 2 norm bound (max_step) to an inf norm bound (displacement component
            // bound).
            scp.max_step / 3.0f64.sqrt()
        });

        let problem = NonLinearProblem {
            prev_pos,
            tetmesh: Rc::clone(&mesh),
            kinematic_object,
            energy_model: energy_model_builder.build(),
            gravity: Gravity::new(Rc::clone(&mesh), density, &gravity),
            momentum_potential,
            volume_constraint,
            smooth_contact_constraint,
            displacement_bound,
            interrupt_checker: Box::new(|| false),
            iterations: 0,
        };

        let mut ipopt = Ipopt::new(problem)?;

        // Determine the true force tolerance. To start we base this tolerance on the elastic
        // response which depends on mu and lambda as well as per tet volume:
        // Larger stiffnesses and volumes cause proportionally larger gradients. Thus our tolerance
        // should reflect these properties.
        let tol = params.tolerance as f64 * max_vol * lambda.max(mu);
        params.tolerance = tol as f32;
        params.outer_tolerance *= (max_vol * lambda.max(mu)) as f32;
        println!("tol = {:?}", params.tolerance);
        println!("outer_tol = {:?}", params.outer_tolerance);

        ipopt.set_option("tol", tol);
        ipopt.set_option("acceptable_tol", 10.0 * tol);
        ipopt.set_option("max_iter", params.max_iterations as i32);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("sb", "yes"); // removes the Ipopt welcome message
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("nlp_scaling_method", "none");
        //ipopt.set_option("nlp_scaling_max_gradient", 1e-5);
        //ipopt.set_option("print_timing_statistics", "yes");
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
        ipopt.set_intermediate_callback(Some(NonLinearProblem::intermediate_cb));

        Ok(Solver {
            solver: ipopt,
            step_count: 0,
            sim_params: params.clone(),
            solid_material: Some(solid_material),
            max_step: smooth_contact_params.map_or(0.0, |x| x.max_step),
        })
    }

    /// Compute signed volume for reference elements in the given `TetMesh`.
    fn compute_ref_tet_signed_volumes(mesh: &mut TetMesh) -> Result<Vec<f64>, Error> {
        let ref_volumes: Vec<f64> = mesh
            .cell_iter()
            .map(|cell| ref_tet(&mesh, cell).signed_volume())
            .collect();
        if ref_volumes.iter().find(|&&x| x <= 0.0).is_some() {
            return Err(Error::InvertedReferenceElement);
        }
        Ok(ref_volumes)
    }

    /// Compute shape matrix inverses for reference elements in the given `TetMesh`.
    fn compute_ref_tet_shape_matrix_inverses(mesh: &mut TetMesh) -> Vec<Matrix3<f64>> {
        // Compute reference shape matrix inverses
        mesh
            .cell_iter()
            .map(|cell| {
                let ref_shape_matrix = ref_tet(&mesh, cell).shape_matrix();
                // We assume that ref_shape_matrices are non-singular.
                ref_shape_matrix.inverse().unwrap()
            })
            .collect()
    }

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    fn prepare_mesh_attributes(mesh: &mut TetMesh) -> Result<&mut TetMesh, Error> {
        let verts = mesh.vertex_positions().to_vec();

        mesh.attrib_or_add_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts.as_slice())?;

        mesh.attrib_or_add::<DispType, VertexIndex>(DISPLACEMENT_ATTRIB, [0.0; 3])?;

        // If this attribute doesn't exist, assume no vertices are fixed. This function will
        // return an error if there is an existing Fixed attribute with the wrong type.
        {
            use crate::geo::mesh::attrib::*;
            let fixed_buf = mesh.remove_attrib::<VertexIndex>(FIXED_ATTRIB)
                .unwrap_or(Attribute::from_vec(vec![0 as FixedIntType; mesh.num_vertices()])).into_buffer();
            let mut fixed = fixed_buf.cast_into_vec::<FixedIntType>();
            if fixed.is_empty() { // If non-numeric type detected, just fill it with zeros.
                fixed.resize(mesh.num_vertices(), 0);
            }
            mesh.insert_attrib::<VertexIndex>(FIXED_ATTRIB, Attribute::from_vec(fixed))?;
        }

        let ref_volumes = Self::compute_ref_tet_signed_volumes(mesh)?;
        mesh.set_attrib_data::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB, ref_volumes.as_slice())?;

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
    /// Simulation paramters. This is kept around for convenience.
    sim_params: SimParams,
    /// Solid material properties.
    solid_material: Option<Material>,
    /// Maximal displaement length. Used to limit displacement which is necessary in contact
    /// scenarios because it defines how far a step we can take before the constraint Jacobian
    /// sparsity pattern changes. If zero, then no limit is applied but constraint Jacobian is kept
    /// sparse.
    max_step: f64,
}

impl Solver {

    /// Set the interrupt checker to the given function.
    pub fn set_interrupter(&mut self, checker: Box<FnMut() -> bool>) {
        self.solver.solver_data_mut().problem.interrupt_checker = checker;
    }

    /// Get an immutable borrow for the underlying `TriMesh` of the kinematic object.
    pub fn try_borrow_kinematic_mesh(&self) -> Option<Ref<'_, TriMesh>> {
        match &self.solver.solver_data_ref().problem.kinematic_object {
            Some(x) => Some(x.borrow()),
            None => None,
        }
    }

    /// Get an immutable borrow for the underlying `TetMesh`.
    pub fn borrow_mesh(&self) -> Ref<'_, TetMesh> {
        self.solver.solver_data_ref().problem.tetmesh.borrow()
    }

    /// Get a mutable borrow for the underlying `TetMesh`.
    pub fn borrow_mut_mesh(&mut self) -> RefMut<'_, TetMesh> {
        self.solver.solver_data_ref().problem.tetmesh.borrow_mut()
    }

    /// Get simulation parameters.
    pub fn params(&mut self) -> SimParams {
        self.sim_params
    }

    /// Get the solid material model (if any).
    pub fn solid_material(&mut self) -> Option<Material> {
        self.solid_material
    }

    /// Update the maximal displacement allowed. If zero, no limit is applied.
    pub fn update_max_step(&mut self, step: f64) {
        self.max_step = step;
        if let Some(ref mut scc) = self.solver.solver_data_mut().problem.smooth_contact_constraint {
            scc.update_max_step(step);
        }
    }

    /// Update the solver mesh with the given points.
    pub fn update_mesh_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        // Get solver data. We want to update the primal variables with new positions from `pts`.
        let SolverDataMut { problem, .. } = self.solver.solver_data_mut();

        // Get the tetmesh and prev_pos so we can update fixed vertices only.
        let tetmesh = &problem.tetmesh.borrow();
        let mut prev_pos = problem.prev_pos.borrow_mut();

        if pts.num_vertices() != prev_pos.len() || pts.num_vertices() != tetmesh.num_vertices() {
            // We got an invalid point cloud
            return Err(Error::SizeMismatch);
        }

        // Only update fixed vertices, if no such attribute exists, return an error.
        let fixed_iter = tetmesh.attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?; 
        prev_pos
            .iter_mut()
            .zip(pts.vertex_position_iter())
            .zip(fixed_iter)
            .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
            .for_each(|(pos, new_pos)| *pos = Vector3::from(*new_pos));
        Ok(())
    }

    /// Update the kinematic object mesh with the given points.
    pub fn update_kinematic_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        // Get solver data. We want to update the primal variables with new positions from `pts`.
        let SolverDataMut { problem, .. } = self.solver.solver_data_mut();

        // Get the kinematic object so we can update the vertices.
        let mut trimesh = match &problem.kinematic_object {
            Some(ref mesh) => mesh.borrow_mut(),
            None => return Err(Error::NoKinematicMesh),
        };

        if pts.num_vertices() != trimesh.num_vertices() {
            // We got an invalid point cloud
            return Err(Error::SizeMismatch);
        }

        // Update all the vertices.
        trimesh.vertex_position_iter_mut()
            .zip(pts.vertex_position_iter())
            .for_each(|(pos, new_pos)| *pos = *new_pos);
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
                mesh.attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
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
    fn compute_elastic_forces(
        forces: &mut [Vector3<f64>],
        mesh: &TetMesh,
        lambda: f64,
        mu: f64,
    ) {
        // Reset forces
        for f in forces.iter_mut() {
            *f = Vector3::zeros();
        }

        let grad_iter = mesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                mesh.attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
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

    /// Update the state of the system, which includes new positions and velocities stored on the
    /// mesh, as well as previous positions stored in `prev_pos`.
    fn advance(mesh: &mut TetMesh, prev_pos: &mut [Vector3<f64>], disp: &[Vector3<f64>]) {
        // Write back the velocity for the next iteration.
        for ((prev_dx, prev_x), &dx) in mesh
            .attrib_iter_mut::<DispType, VertexIndex>(DISPLACEMENT_ATTRIB)
            .unwrap()
            .zip(prev_pos.iter_mut())
            .zip(disp.iter())
        {
            *prev_dx = dx.into();
            *prev_x += dx;
        }

        mesh.vertex_position_iter_mut()
            .zip(prev_pos.iter())
            .for_each(|(pos, prev_pos)| *pos = (*prev_pos).into());
    }

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn inner_step(&mut self) -> Result<InnerSolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            objective_value,
            status,
            ..
        } = self.solver.solve();

        let iterations = solver_data.problem.pop_iteration_count() as u32;

        let result = InnerSolveResult {
            iterations,
            objective_value,
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::SolveError(e, SolveResult {
                max_inner_iterations: iterations,
                iterations,
                objective_value
            })),
        }
    }

    /// Compute and add the value of the constraint function minus constraint bounds.
    pub fn add_constraint_function(&mut self, constraint: &mut [f64]) {
        use ipopt::ConstrainedProblem;
        let SolverDataMut {
            problem,
            primal_variables,
            constraint_multipliers,
            ..
        } = self.solver.solver_data_mut();

        assert_eq!(constraint.len(), constraint_multipliers.len());
        let (lower, upper) = problem.constraint_bounds();
        assert!(problem.constraint(&vec![0.0; primal_variables.len()], constraint));
        for (c, (l, u)) in constraint.iter_mut().zip(lower.into_iter().zip(upper.into_iter())) {
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
    pub fn compute_objective_gradient(&mut self, grad: &mut [f64]) {
        use ipopt::BasicProblem;
        let SolverDataMut {
            problem,
            primal_variables,
            ..
        } = self.solver.solver_data_mut();

        assert_eq!(grad.len(), primal_variables.len());
        assert!(problem.objective_grad(&vec![0.0; primal_variables.len()], grad));

        // Erase fixed vert data. This doesn't contribute to the solve.
        let mesh = problem.tetmesh.borrow();
        let fixed_iter = mesh.attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            .expect("Missing fixed verts attribute").map(|&x| x != 0);
        let vert_grad: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);
        for g in vert_grad.iter_mut().filter_masked(fixed_iter) {
            *g = Vector3::zeros();
        }
    }

    /// Compute and add the Jacobian and constraint multiplier product to the given vector.
    pub fn add_constraint_jacobian_product(&mut self, jac_prod: &mut [f64]) {
        use ipopt::ConstrainedProblem;
        let SolverDataMut {
            problem,
            primal_variables,
            constraint_multipliers,
            ..
        } = self.solver.solver_data_mut();

        let jac_nnz = problem.num_constraint_jac_non_zeros();
        let mut rows = vec![0; jac_nnz];
        let mut cols = vec![0; jac_nnz];
        assert!(problem.constraint_jac_indices(&mut rows, &mut cols));

        let mut values = vec![0.0; jac_nnz];
        assert!(problem.constraint_jac_values(&vec![0.0; primal_variables.len()], &mut values));

        // We don't consider values for fixed vertices.
        let mesh = problem.tetmesh.borrow();
        let fixed = mesh.attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            .expect("Missing fixed verts attribute");

        assert_eq!(jac_prod.len(), primal_variables.len());
        // Effectively this is jac.transpose() * constraint_multipliers
        for ((row, col), val) in rows.into_iter().zip(cols.into_iter()).zip(values.into_iter()) {
            if fixed[(col as usize)/3] == 0 {
                jac_prod[col as usize] += val*constraint_multipliers[row as usize];
            }
        }
    }

    /// Update the `mesh` and `prev_pos` with the current solution.
    fn commit_solution(&mut self) {
        let SolverDataMut {
            problem,
            primal_variables,
            ..
        } = self.solver.solver_data_mut();

        // Reinterpret solver variables as positions in 3D space.
        let displacement: &[Vector3<f64>] = reinterpret_slice(primal_variables);

        // Update the mesh itself.
        {
            let mesh = &mut problem.tetmesh.borrow_mut();
            let mut prev_pos = problem.prev_pos.borrow_mut();

            Self::advance(mesh, prev_pos.as_mut_slice(), displacement);
        }

        // Since we advected the mesh, we need to invalidate its neighbour data so it's
        // recomputed at the next time step (if applicable).
        if let Some(ref mut scc) = problem.smooth_contact_constraint {
            scc.invalidate_neighbour_data();
            scc.update_surface();
        }
    }

    fn inf_norm(vec: &[f64]) -> f64 {
        vec.iter().map(|x| x.abs()).max_by(|a, b| a.partial_cmp(b).expect("Detected NaNs")).unwrap_or(0.0)
    }

    /// Run the optimization solver on one time step.
    pub fn step(&mut self) -> Result<SolveResult, Error> {

        // Initialize the result of this function.
        let mut result = SolveResult {
            max_inner_iterations: 0,
            iterations: 0,
            objective_value: 0.0,
        };

        let mut residual_norm = 0.0;

        let mut objective_residual = vec![0.0; self.solver.solver_data_ref().primal_variables.len()];
        let mut constraint_violation = vec![0.0; self.solver.solver_data_ref().constraint_multipliers.len()];

        // We should iterate until a relative residual goes to zero.
        for iter in 0..self.sim_params.max_outer_iterations {

            let step_result = self.inner_step();

            // Commit the solution whether or not there is an error. In case of error we will be
            // able to investigate the result.
            self.commit_solution();

            // Update output result with new data.
            match step_result {
                Ok(step_result) => {
                    result.max_inner_iterations =
                        step_result.iterations.max(result.max_inner_iterations);
                    result.iterations += 1;
                    result.objective_value = step_result.objective_value;
                }
                Err(Error::SolveError(status, step_result)) => {
                    // In case of solve error, update our result and return. We don't increment
                    // step count so we dont trigger warm start using these multipliers.

                    // Reset warm start after we encounter an error
                    self.solver.set_option("warm_start_init_point", "no");

                    result.max_inner_iterations =
                        step_result.iterations.max(result.max_inner_iterations);
                    result.iterations += 1;
                    result.objective_value = step_result.objective_value;

                    return Err(Error::SolveError(status, result));
                }
                Err(e) => {
                    // Unknown error: Reset warm start and return.
                    self.solver.set_option("warm_start_init_point", "no");
                    return Err(e);
                }
            }

            if self.step_count == 0 && iter == 0 {
                // After the first step lets set ipopt into warm start mode
                self.solver.set_option("warm_start_init_point", "yes");
            }

            // Since we are using linearized constraints, we compute the true constraint violation
            // residual and iterate until it vanishes.
            objective_residual.iter_mut().for_each(|x| *x = 0.0); // Reset the objective residual.
            self.compute_objective_gradient(&mut objective_residual);
            self.add_constraint_jacobian_product(&mut objective_residual);

            constraint_violation.iter_mut().for_each(|x| *x = 0.0); // Reset the constraint residual.
            self.add_constraint_function(&mut constraint_violation);

            residual_norm = Self::inf_norm(&objective_residual).max(Self::inf_norm(&constraint_violation));
            println!("residual = {:?}", residual_norm);

            if residual_norm <= self.sim_params.outer_tolerance as f64 {
                break;
            }
        }

        if result.iterations >= self.sim_params.max_outer_iterations {
            eprintln!("WARNING: Reached max outer iterations: {:?}\nResidual is: {:?}",
                      result.iterations, residual_norm);
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
    use crate::geo;
    use std::path::PathBuf;

    /*
     * Setup code
     */

    const STATIC_PARAMS: SimParams = SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        time_step: None,
        tolerance: 1e-9,
        max_iterations: 300,
        max_outer_iterations: 1,
        outer_tolerance: 0.001,
        print_level: 0,
        derivative_test: 0,
    };

    const DYNAMIC_PARAMS: SimParams = SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        time_step: Some(0.01),
        ..STATIC_PARAMS
    };

    // Note: The key to getting reliable simulations here is to keep bulk_modulus, shear_modulus
    // (mu) and density in the same range of magnitude. Higher stiffnesses compared to denisty will
    // produce highly oscillatory configurations and keep the solver from converging fast.
    // As an example if we increase the moduli below by 1000, the solver can't converge, even in
    // 300 steps.
    const SOLID_MATERIAL: Material = Material {
        elasticity: ElasticityParameters {
            bulk_modulus: 1750e3,
            shear_modulus: 10e3,
        },
        incompressibility: false,
        density: 1000.0,
        damping: 0.0,
    };

    fn make_one_tet_mesh() -> TetMesh {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 2, 1, 3];
        let mut mesh = TetMesh::new(verts.clone(), indices);
        mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0, 0])
            .unwrap();

        mesh.add_attrib_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts)
            .unwrap();
        mesh
    }

    fn make_one_deformed_tet_mesh() -> TetMesh {
        let mut mesh = make_one_tet_mesh();
        mesh.vertex_positions_mut()[3][2] = 2.0;
        mesh
    }

    fn make_three_tet_mesh_with_verts(verts: Vec<[f64; 3]>) -> TetMesh {
        let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
        let mut mesh = TetMesh::new(verts, indices);
        mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 0, 1, 1, 0, 0])
            .unwrap();

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, ref_verts)
            .unwrap();
        mesh
    }

    fn make_three_tet_mesh() -> TetMesh {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ];
        make_three_tet_mesh_with_verts(verts)
    }

    /// Utility function to compare positions of two meshes.
    fn compare_meshes(solution: &TetMesh, expected: &TetMesh) {
        for (pos, expected_pos) in solution
            .vertex_positions()
            .iter()
            .zip(expected.vertex_positions().iter())
        {
            for j in 0..3 {
                assert_relative_eq!(pos[j], expected_pos[j], max_relative = 1.0e-6);
            }
        }
    }

    /*
     * One tet tests
     */

    /// Helper function to generate a simple solver for one initially deformed tet under gravity.
    fn one_tet_solver() -> Solver {
        let mesh = make_one_deformed_tet_mesh();

        SolverBuilder::new(STATIC_PARAMS)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()
            .unwrap()
    }

    /// Test that the solver produces no change for an equilibrium configuration.
    #[test]
    fn one_tet_equilibrium_test() {
        let params = SimParams {
            gravity: [0.0f32, 0.0, 0.0],
            outer_tolerance: 1e-10, // This is a fairly strict tolerance.
            max_outer_iterations: 2,
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
        compare_meshes(&solution, &mesh);
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
    fn one_tet_outer_test() {
        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            max_outer_iterations: 2,
            ..STATIC_PARAMS
        };

        let mesh = make_one_deformed_tet_mesh();

        let mut solver = SolverBuilder::new(params)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh.clone())
            .build()
            .unwrap();
        assert!(solver.step().is_ok());

        let solution = solver.borrow_mesh();
        let mut expected_solver = one_tet_solver();
        expected_solver.step().unwrap();
        let expected = expected_solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn one_tet_volume_constraint_test() {
        let mesh = make_one_deformed_tet_mesh();

        let material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };

        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
    }

    /*
     * Three tet tests
     */

    #[test]
    fn three_tets_static_test() {
        let mesh = make_three_tet_mesh();
        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let solution = solver.borrow_mesh();
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_static_expected.vtk")).unwrap();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn three_tets_dynamic_test() {
        let mesh = make_three_tet_mesh();
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let solution = solver.borrow_mesh();
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_dynamic_expected.vtk"))
                .unwrap();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn three_tets_static_volume_constraint_test() {
        let mesh = make_three_tet_mesh();
        let material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };
        let mut solver = SolverBuilder::new(STATIC_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let solution = solver.borrow_mesh();
        let exptected = geo::io::load_tetmesh(&PathBuf::from(
            "assets/three_tets_static_volume_constraint_expected.vtk",
        ))
        .unwrap();
        compare_meshes(&solution, &exptected);
    }

    #[test]
    fn three_tets_dynamic_volume_constraint_test() {
        let mesh = make_three_tet_mesh();
        let material = Material {
            incompressibility: true,
            ..SOLID_MATERIAL
        };
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let solution = solver.borrow_mesh();
        let expected = geo::io::load_tetmesh(&PathBuf::from(
            "assets/three_tets_dynamic_volume_constraint_expected.vtk",
        ))
        .unwrap();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn animation_test() {
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
            .build()
            .unwrap();

        for frame in 1u32..100 {
            let offset = 0.01 * (if frame < 50 { frame } else { 0 } as f64);
            verts.iter_mut().for_each(|x| (*x)[1] += offset);
            let pts = PointCloud::new(verts.clone());
            assert!(solver.update_mesh_vertices(&pts).is_ok());
            assert!(solver.step().is_ok());
        }
    }

    #[test]
    fn animation_volume_constraint_test() {
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
            .build()
            .unwrap();

        for frame in 1u32..100 {
            let offset = 0.01 * (if frame < 50 { frame } else { 0 } as f64);
            verts.iter_mut().for_each(|x| (*x)[1] += offset);
            let pts = PointCloud::new(verts.clone());
            //save_tetmesh_ascii(
            //    solver.borrow_mesh(),
            //    &PathBuf::from(format!("./out/mesh_{}.vtk", frame)),
            //);
            assert!(solver.update_mesh_vertices(&pts).is_ok());
            assert!(solver.step().is_ok());
        }
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
    fn box_stretch_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk")).unwrap();
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .solid_material(MEDIUM_SOLID_MATERIAL)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn box_stretch_volume_constraint_test() {
        let incompressible_material = Material {
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk")).unwrap();
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .solid_material(incompressible_material)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_const_volume.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn box_twist_test() {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            ..MEDIUM_SOLID_MATERIAL
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn box_twist_volume_constraint_test() {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = SolverBuilder::new(STRETCH_PARAMS)
            .solid_material(material)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    /// This test insures that a non-linearized constraint like volume doesn't cause multiple outer
    /// iterations, and converges after the first solve.
    #[test]
    fn box_twist_volume_constraint_outer_test() {
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.0),
            incompressibility: true,
            ..MEDIUM_SOLID_MATERIAL
        };

        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            max_outer_iterations: 2,
            ..STRETCH_PARAMS
        };

        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = SolverBuilder::new(params)
            .solid_material(material)
            .add_solid(mesh)
            .build()
            .unwrap();
        let solve_result = solver.step().expect("Solve failed");
        assert_eq!(solve_result.iterations, 1);

        // This test should produce the exact same mesh as the original
        // box_twist_volume_constraint_test
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    /*
     * Tests with contact constraints
     */

    fn compute_contact_constraint(sample_mesh: &PolyMesh, tetmesh: &TetMesh, radius: f64, tolerance: f64) -> Vec<f32> {
        use implicits::*;
        // There are currently two different ways to compute the implicit function representing the
        // contact constraint. Since this is a test we do it both ways and make sure the result is
        // the same. This douples as a test for the implicits crate.

        let mut trimesh_copy = sample_mesh.clone();
        let surface_trimesh = tetmesh.surface_trimesh();
        let mut surface_polymesh = PolyMesh::from(surface_trimesh.clone());
        compute_potential(&mut trimesh_copy, &mut surface_polymesh,
                          implicits::Params {
                              kernel: KernelType::Approximate {
                                  tolerance,
                                  radius,
                              },
                              background_potential: BackgroundPotentialType::None,
                              sample_type: SampleType::Face,
                          }, || false)
        .expect("Failed to compute constraint value");
        
        let pot_attrib = trimesh_copy.attrib_clone_into_vec::<f32, VertexIndex>("potential")
            .expect("Potential attribute doesn't exist");


        {
            let mut builder = ImplicitSurfaceBuilder::new();
            builder.triangles(reinterpret::reinterpret_slice(surface_trimesh.faces()).to_vec())
                .vertices(surface_trimesh.vertex_positions().to_vec())
                .kernel(KernelType::Approximate { tolerance, radius })
                .sample_type(SampleType::Face)
                .background_potential(BackgroundPotentialType::None);

            let surf = builder.build().expect("Failed to build implicit surface.");
            
            let mut pot_attrib64 = vec![0.0f64; sample_mesh.num_vertices()];
            surf.potential(sample_mesh.vertex_positions(), &mut pot_attrib64)
                .expect("Failed to compute contact constraint potential.");

            // Make sure the two potentials are identical.
            println!("potential = {:?}", pot_attrib);
            println!("potential64 = {:?}", pot_attrib64);
            for (&x,&y) in pot_attrib.iter().zip(pot_attrib64.iter()) {
                assert_relative_eq!(x, y as f32, max_relative = 1e-5);
            }
        }

        pot_attrib.into_iter().map(|x| x as f32).collect()
    }

    #[test]
    fn tet_push_test() {
        // A triangle is being pushed on top of a tet.
        let height = 1.18032;
        let mut tri_verts = vec![
            [0.5, height, 0.0],
            [-0.25, height, 0.433013],
            [-0.25, height, -0.433013],
        ];

        let tri = vec![3, 0, 2, 1];

        let tet_verts = vec![
            [0.0, 1.0, 0.0],
            [-0.94281, -0.33333, 0.0],
            [0.471405, -0.33333, 0.816498],
            [0.471405, -0.33333, -0.816498],
        ];

        let tet = vec![3, 1, 0, 2];

        let fixed = vec![0, 1, 1, 1];
            
        let mut tetmesh = TetMesh::new(tet_verts.clone(), tet);
        tetmesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed).unwrap();

        let trimesh = PolyMesh::new(tri_verts.clone(), &tri);

        // Set contact parameters
        let radius = 2.0;
        let tolerance = 0.1;

        compute_contact_constraint(&trimesh, &tetmesh, radius, tolerance);

        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            max_outer_iterations: 100,
            ..STRETCH_PARAMS
        };

        let mut solver = SolverBuilder::new(params)
            .solid_material(MEDIUM_SOLID_MATERIAL)
            .add_solid(tetmesh.clone())
            .add_shell(trimesh.clone())
            .smooth_contact_params(SmoothContactParams { radius: 2.0, tolerance: 0.1, max_step: 2.0 })
            .build()
            .unwrap();

        let solve_result = solver.step().expect("Failed equilibrium solve.");
        assert_eq!(solve_result.iterations, 1); // should be no more than one outer iteration

        // Expect no push since the triangle is outside the surface.
        for (pos, exp_pos) in solver.borrow_mesh().vertex_position_iter().zip(tet_verts.iter()) {
            for i in 0..3 {
                assert_relative_eq!(pos[i], exp_pos[i], max_relative=1e-5, epsilon = 1e-6);
            }
        }

        // Verify constraint, should be positive before push
        let constraint = compute_contact_constraint(&trimesh, &solver.borrow_mesh(), radius, tolerance);
        assert!(constraint.iter().all(|&x| x > 0.0f32));

        // Simulate push
        let offset = 0.5;
        tri_verts.iter_mut().for_each(|x| (*x)[1] -= offset);
        let pts = PointCloud::new(tri_verts.clone());
        assert!(solver.update_kinematic_vertices(&pts).is_ok());
        let solve_result = solver.step().expect("Failed push solve.");
        assert!(solve_result.iterations < params.max_outer_iterations);

        // Verify constraint, should be positive after push
        let constraint = compute_contact_constraint(&trimesh, &solver.borrow_mesh(), radius, tolerance);
        assert!(constraint.iter().all(|&x| x > 0.0f32));

        // Expect only the top vertex to be pushed down.
        let offset_verts = vec![
            [0.0, 0.723103, 0.0],
            tet_verts[1],
            tet_verts[2],
            tet_verts[3],
        ];

        for (pos, exp_pos) in solver.borrow_mesh().vertex_position_iter().zip(offset_verts.iter()) {
            for i in 0..3 {
                assert_relative_eq!(pos[i], exp_pos[i], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn ball_tri_push_test() {
        let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk")).unwrap();
        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1000.0, 0.4),
            ..SOLID_MATERIAL
        };

        let params = SimParams {
            outer_tolerance: 1e-5, // This is a fairly strict tolerance.
            max_outer_iterations: 10,
            ..DYNAMIC_PARAMS
        };

        let polymesh = geo::io::load_polymesh(&PathBuf::from("assets/tri.vtk")).unwrap();
        let mut solver = SolverBuilder::new(params)
            .solid_material(material)
            .add_solid(tetmesh)
            .add_shell(polymesh)
            .smooth_contact_params(SmoothContactParams { radius: 1.0, tolerance: 1e-5, max_step: 1.5 })
            .build()
            .unwrap();

        let res = solver.step().expect("Failed push solve.");
        println!("res = {:?}", res);
        assert!(res.iterations < params.max_outer_iterations, "Exceeded max outer iterations.");
    }

    /*
     * More complex tests
     */

    #[test]
    fn torus_medium_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()
            .unwrap();
        assert!(solver.step().is_ok());
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn torus_long_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .solid_material(SOLID_MATERIAL)
            .add_solid(mesh)
            .build()
            .unwrap();

        for _i in 0..10 {
        //geo::io::save_tetmesh_ascii(
        //    &solver.borrow_mesh(),
        //    &PathBuf::from(format!("./out/mesh_{}.vtk", 1)),
        //    ).unwrap();
            let res = solver.step();
            assert!(res.is_ok());
        }
    }
}
