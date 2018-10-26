use energy::*;
use energy_model::{ElasticTetMeshEnergy, ElasticTetMeshEnergyBuilder, NeoHookeanTetEnergy};
use geo::math::{Matrix3, Vector3};
use geo::mesh::{attrib, tetmesh::TetCell, topology::*, Attrib};
use geo::ops::{ShapeMatrix, Volume};
use geo::prim::Tetrahedron;
use ipopt::{self, Index, Ipopt, Number, SolverData};
use reinterpret::*;
use attrib_names::*;
use std::fmt;

use PointCloud;
use TetMesh;
pub type Tet = Tetrahedron<f64>;

/// Result from one simulation step.
#[derive(Debug)]
pub struct SolveResult {
    /// Number of inner iterations of the step result.
    pub iterations: u32,
    /// The value of the objective at the end of the time step.
    pub objective_value: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialProperties {
    /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
    /// incompressibility. The larger the value, the more incompressible the material is.
    /// Think of this as "Volume Stiffness"
    pub bulk_modulus: f32,
    /// Shear modulus measures the material's resistance to shear deformation. The larger the
    /// value, the more it resists changes in shape. Think of this as "Shape Stiffness"
    pub shear_modulus: f32,
    /// The density of the material.
    pub density: f32,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model.
    pub damping: f32,
}

impl MaterialProperties {
    /// Convert internal elasticity parameters to Lame parameters for Neo-Hookean energy.
    pub fn lame_parameters(&self) -> (f64, f64) {
        let lambda = self.bulk_modulus as f64 - 2.0 * self.shear_modulus as f64 / 3.0;
        let mu = self.shear_modulus as f64;
        (lambda, mu)
    }
}

/// Simulation parameters.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimParams {
    pub material: MaterialProperties,
    pub gravity: [f32; 3],
    pub time_step: Option<f32>,
    pub tolerance: f32,
}

/// Get reference tetrahedron.
/// This routine assumes that there is a vertex attribute called `ref` of type `[f64;3]`.
pub fn ref_tet(tetmesh: &TetMesh, indices: &TetCell) -> Tet {
    let attrib = tetmesh.attrib::<VertexIndex>(REFERENCE_POSITION_ATTRIB).unwrap();
    Tetrahedron(
        (attrib.get::<[f64; 3], _>(indices[0]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[1]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[2]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[3]).unwrap()).into(),
    )
}

/// Finite element engine.
pub struct FemEngine {
    /// Non-linear solver.
    solver: Ipopt<NonLinearProblem>,
    /// Step count (outer iterations). These count the number of times the function `step` was
    /// called.
    step_count: usize,
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NonLinearProblem {
    /// Elastic energy model.
    pub energy_model: ElasticTetMeshEnergy,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Simulation parameters.
    pub params: SimParams,
}

impl fmt::Debug for NonLinearProblem{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NonLinearProblem {{ energy_model: {:?}, iterations: {:?}, params: {:?} }}",
               self.energy_model, self.iterations, self.params)
    }
}

impl FemEngine {
    /// Run the optimization solver on one time step.
    pub fn new(mut mesh: TetMesh, params: SimParams) -> Result<Self, Error> {
        // Prepare tet mesh for simulation.
        let verts = mesh.vertex_positions().to_vec();
        mesh.attrib_or_add_data::<_, VertexIndex>(
            REFERENCE_POSITION_ATTRIB,
            verts.as_slice(),
        )?;

        mesh.attrib_or_add::<_, VertexIndex>(DISPLACEMENT_ATTRIB, [0.0; 3])?;

        {
            // Compute reference element signed volumes
            let ref_volumes: Vec<f64> = mesh
                .cell_iter()
                .map(|cell| ref_tet(&mesh, cell).signed_volume())
                .collect();
            if ref_volumes.iter().find(|&&x| x <= 0.0).is_some() {
                return Err(Error::InvertedReferenceElement);
            }
            mesh.set_attrib_data::<_, CellIndex>(
                REFERENCE_VOLUME_ATTRIB,
                ref_volumes.as_slice(),
            )?;
        }

        {
            // Compute reference shape matrix inverses
            let ref_shape_mtx_inverses: Vec<Matrix3<f64>> = mesh
                .cell_iter()
                .map(|cell| {
                    let ref_shape_matrix = ref_tet(&mesh, cell).shape_matrix();
                    // We assume that ref_shape_matrices are non-singular.
                    ref_shape_matrix.inverse().unwrap()
                }).collect();
            mesh.set_attrib_data::<_, CellIndex>(
                REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                ref_shape_mtx_inverses.as_slice(),
            )?;
        }

        {
            // Add elastic strain energy and elastic force attributes.
            // These will be computed at the end of the time step.
            mesh.set_attrib::<_, CellIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
            mesh.set_attrib::<_, VertexIndex>(ELASTIC_FORCE_ATTRIB, [0f64; 3])?;
        }

        let (lambda, mu) = params.material.lame_parameters();

        let density = params.material.density as f64;
        let damping = params.material.damping as f64;
        let gravity = [
            params.gravity[0] as f64,
            params.gravity[1] as f64,
            params.gravity[2] as f64,
        ];
        let mut energy_model_builder = ElasticTetMeshEnergyBuilder::new(mesh)
            .material(lambda / mu, 1.0, density / mu, damping / mu)
            .gravity(gravity);

        if let Some(dt) = params.time_step {
            energy_model_builder = energy_model_builder.time_step(dt as f64);
        }

        let problem = NonLinearProblem {
            energy_model: energy_model_builder.build(),
            interrupt_checker: Box::new(|| false),
            iterations: 0,
            params,
        };

        let mut ipopt = Ipopt::new_newton(problem);

        ipopt.set_option("tol", params.tolerance as f64);
        ipopt.set_option("acceptable_tol", 10.0 * params.tolerance as f64);
        ipopt.set_option("max_iter", 800);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("sb", "yes"); // removes the Ipopt welcome message
        ipopt.set_option("print_level", 0);
        ipopt.set_option("nlp_scaling_max_gradient", 1e-5);
        //ipopt.set_option("print_timing_statistics", "yes");
        //ipopt.set_option("hessian_approximation", "limited-memory");
        //ipopt.set_option("derivative_test", "second-order");
        //ipopt.set_option("derivative_test_tol", 1e-4);
        //ipopt.set_option("point_perturbation_radius", 0.01);
        ipopt.set_intermediate_callback(Some(NonLinearProblem::intermediate_cb));

        Ok(FemEngine { solver: ipopt, step_count: 0 })
    }

    /// Set the interrupt checker to the given function.
    pub fn set_interrupter(&mut self, checker: Box<FnMut() -> bool>) {
        self.solver.get_solver_data().problem.interrupt_checker = checker;
    }

    /// Get an immutable reference to the solver `TetMesh`.
    pub fn mesh_ref(&mut self) -> &TetMesh {
        &self.solver.get_solver_data().problem.energy_model.solid
    }

    /// Get solver parameters.
    pub fn params(&mut self) -> SimParams {
        self.solver.get_solver_data().problem.params
    }

    /// Update the solver mesh with the given points.
    pub fn update_mesh_vertices(&mut self, pts: &PointCloud) -> bool {
        // Get solver data. We want to update the primal variables with new positions from `pts`.
        let SolverData {
            problem,
            ..
        } = self.solver.get_solver_data();

        // Get the tetmesh so we can update fixed vertices only.
        let ElasticTetMeshEnergy {
            solid: ref tetmesh,
            ref mut prev_pos,
            ..
        } = problem.energy_model;

        if pts.num_vertices() != prev_pos.len()
            || pts.num_vertices() != tetmesh.num_vertices() {
            // We got an invalid point cloud
            return false;
        }

        // Only update fixed vertices.
        prev_pos.iter_mut()
            .zip(pts.vertex_iter())
            .zip(
                tetmesh.attrib_iter::<i8, VertexIndex>(FIXED_ATTRIB)
                .unwrap()
                )
            .filter_map( |(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None } )
            .for_each( |(pos, new_pos)| *pos = Vector3::from(*new_pos));
        true
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
                mesh.attrib_iter::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
            ).zip(
                mesh.attrib_iter::<Matrix3<f64>, CellIndex>(
                    REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                ).unwrap(),
            ).zip(mesh.tet_iter())
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
        mesh: &mut TetMesh,
        lambda: f64,
        mu: f64,
    ) {
        // Reset forces
        for f in forces.iter_mut() {
            *f = Vector3::zeros();
        }

        let grad_iter = mesh
            .attrib_iter::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                mesh.attrib_iter::<Matrix3<f64>, CellIndex>(
                    REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                ).unwrap(),
            ).zip(mesh.tet_iter())
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
    fn update_state(mesh: &mut TetMesh, prev_pos: &mut [Vector3<f64>], disp: &[Vector3<f64>]) {
        // Write back the velocity for the next iteration.
        for ((prev_dx, prev_x), &dx) in mesh
            .attrib_iter_mut::<[f64; 3], VertexIndex>(DISPLACEMENT_ATTRIB)
            .unwrap()
            .zip(prev_pos.iter_mut())
            .zip(disp.iter())
        {
            *prev_dx = dx.into();
            *prev_x += dx;
        }

        mesh.vertex_iter_mut()
            .zip(prev_pos.iter())
            .for_each(|(pos, prev_pos)| *pos = (*prev_pos).into());
    }

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn solve_step(&mut self) -> Result<SolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            objective_value,
            status,
            ..
        } = self.solver.solve();

        let iterations = solver_data.problem.iteration_count() as u32;

        let result = SolveResult {
            iterations,
            objective_value,
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::SolveError(e, result)),
        }
    }

    /// Run the optimization solver on one time step.
    pub fn step(&mut self) -> Result<SolveResult, Error> {
        if self.step_count == 1 {
            // After the first step lets set ipopt into warm start mode
            self.solver.set_option("warm_start_init_point", "yes");
        }

        let step_result = self.solve_step();

        self.step_count += 1;

        // On success, update the mesh with new positions and useful metrics.
        if let Ok(_) = step_result {
            let (lambda, mu) = self.params().material.lame_parameters();

            let SolverData {
                ref mut problem,
                ref primal_variables,
                ..
            } = self.solver.get_solver_data();

            let displacement: &[Vector3<f64>] = reinterpret_slice(primal_variables);

            let ElasticTetMeshEnergy {
                solid: ref mut mesh,
                ref mut prev_pos,
                ..
            } = problem.energy_model;

            Self::update_state(mesh, prev_pos.as_mut_slice(), displacement);

            // Write back elastic strain energy for visualization.
            Self::compute_strain_energy_attrib(mesh, lambda, mu);

            // Write back elastic forces on each node.
            Self::compute_elastic_forces_attrib(mesh, lambda, mu);
        }

        step_result
    }
}

impl NonLinearProblem {
    /// Get information about the current simulation
    pub fn iteration_count(&self) -> usize {
        self.iterations
    }

    /// Intermediate callback for `Ipopt`.
    pub fn intermediate_cb(
        &mut self,
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
    ) -> bool {
        self.iterations += 1;
        !(self.interrupt_checker)()
    }
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for NonLinearProblem {
    fn num_variables(&self) -> usize {
        self.energy_model.solid.num_vertices() * 3
    }

    fn bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let solid = &self.energy_model.solid;
        let n = solid.num_vertices();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        lo.resize(n, [-2e19; 3]);
        hi.resize(n, [2e19; 3]);

        if let Ok(fixed_verts) = solid.attrib_as_slice::<i8, VertexIndex>(FIXED_ATTRIB) {
            // Find and set fixed vertices.
            lo.iter_mut()
                .zip(hi.iter_mut())
                .zip(fixed_verts.iter())
                .filter(|&(_, &fixed)| fixed != 0)
                .for_each(|((l, u), _)| {
                    *l = [0.0;3];
                    *u = [0.0;3];
                });
        }
        (reinterpret_vec(lo), reinterpret_vec(hi))
    }

    fn initial_point(&self) -> Vec<Number> {
        vec![0.0; self.num_variables()]
    }

    fn objective(&mut self, dx: &[Number], obj: &mut Number) -> bool {
        self.energy_model.update(dx);
        *obj = self.energy_model.energy();
        true
    }

    fn objective_grad(&mut self, dx: &[Number], grad_f: &mut [Number]) -> bool {
        self.energy_model.update(dx);
        grad_f.copy_from_slice(reinterpret_slice(self.energy_model.energy_gradient()));

        true
    }
}

impl ipopt::NewtonProblem for NonLinearProblem {
    fn num_hessian_non_zeros(&self) -> usize {
        self.energy_model.energy_hessian_size()
    }
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        for (i, &MatrixElementIndex { ref row, ref col }) in
            self.energy_model.energy_hessian_indices().iter().enumerate()
        {
            rows[i] = *row as Index;
            cols[i] = *col as Index;
        }

        true
    }
    fn hessian_values(&mut self, dx: &[Number], vals: &mut [Number]) -> bool {
        self.energy_model.update(dx);

        for (i, val) in self.energy_model.energy_hessian_values().iter().enumerate() {
            vals[i] = *val as Number;
        }

        true
    }
}

#[derive(Debug)]
pub enum Error {
    AttribError(attrib::Error),
    InvertedReferenceElement,
    SolveError(ipopt::SolveStatus, SolveResult), // Iterations and objective value
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo;
    use std::path::PathBuf;
    const DYNAMIC_PARAMS: SimParams = SimParams {
        material: MaterialProperties {
            bulk_modulus: 1e6,
            shear_modulus: 1e5,
            density: 1000.0,
            damping: 0.0,
        },
        gravity: [0.0f32, 0.0, 0.0],
        time_step: Some(0.01),
        tolerance: 1e-9,
    };

    const STATIC_PARAMS: SimParams = SimParams {
        material: MaterialProperties {
            bulk_modulus: 1750e6,
            shear_modulus: 10e6,
            density: 1000.0,
            damping: 0.0,
        },
        gravity: [0.0f32, -9.81, 0.0],
        time_step: None,
        tolerance: 1e-9,
    };

    #[test]
    fn simple_static_test() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ];
        let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
        let mut mesh = TetMesh::new(verts, indices);
        mesh.add_attrib_data::<i8, VertexIndex>(FIXED_ATTRIB, vec![0, 0, 1, 1, 0, 0])
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

        assert!(FemEngine::new(mesh, STATIC_PARAMS).unwrap().step().is_ok());
    }

    #[test]
    fn simple_dynamic_test() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ];
        let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
        let mut mesh = TetMesh::new(verts, indices);
        mesh.add_attrib_data::<i8, VertexIndex>(FIXED_ATTRIB, vec![0, 0, 1, 1, 0, 0])
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

        assert!(FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap().step().is_ok());
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
        let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
        let mut mesh = TetMesh::new(verts.clone(), indices);

        let fixed = vec![0, 0, 1, 1, 0, 0];
        mesh.add_attrib_data::<i8, VertexIndex>(FIXED_ATTRIB, fixed)
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

        let mut solver = FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap();

        for frame in 1u32..100 {
            let offset = 0.01*(if frame < 50 { frame } else { 0 } as f64);
            verts.iter_mut().for_each(|x| (*x)[1] += offset);
            let pts = PointCloud::new(verts.clone());
            assert!(solver.update_mesh_vertices(&pts));
            assert!(solver.step().is_ok());
        }
    }

    #[test]
    fn torus_medium_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        assert!(FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap().step().is_ok());
    }

    #[test]
    fn torus_large_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets_large.vtk")).unwrap();
        assert!(FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap().step().is_ok());
    }

    #[test]
    fn torus_long_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();

        let mut engine = FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap();
        for _i in 0..25 {
            assert!(engine.step().is_ok());
        }
    }
}
