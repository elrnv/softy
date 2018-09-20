use energy::*;
use energy_model::{ElasticTetMeshEnergy, NeoHookeanTetEnergy};
use geo::math::{Matrix3, Vector3};
use geo::mesh::{self, attrib, tetmesh::TetCell, topology::*, Attrib};
use geo::ops::{ShapeMatrix, Volume};
use geo::prim::Tetrahedron;
use ipopt::{self, Index, Ipopt, Number};
use reinterpret::*;
use std::cell::{Ref, RefCell, RefMut};

pub type TetMesh = mesh::TetMesh<f64>;
pub type Tet = Tetrahedron<f64>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolveResult {
    pub iterations: u32,
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
    let attrib = tetmesh.attrib::<VertexIndex>("ref").unwrap();
    Tetrahedron(
        (attrib.get::<[f64; 3], _>(indices[0]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[1]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[2]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[3]).unwrap()).into(),
    )
}

/// Finite element engine.
pub struct FemEngine<'a, F: FnMut() -> bool + Sync> {
    /// Non-linear solver.
    solver: Ipopt<NonLinearProblem<'a, F>>,
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
struct NonLinearProblem<'a, F: FnMut() -> bool + Sync> {
    /// Elastic energy model.
    pub energy_model: RefCell<ElasticTetMeshEnergy<'a>>,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: F,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Simulation parameters.
    pub params: SimParams,
}

impl<'a, F: FnMut() -> bool + Sync> FemEngine<'a, F> {
    const VELOCITY_ATTRIB: &'static str = "vel";
    const REFERENCE_POSITION_ATTRIB: &'static str = "ref";
    const REFERENCE_VOLUME_ATTRIB: &'static str = "ref_volume";
    const REFERENCE_SHAPE_MATRIX_INV_ATTRIB: &'static str = "ref_shape_mtx_inv";
    const STRAIN_ENERGY_ATTRIB: &'static str = "strain_energy";
    const ELASTIC_FORCE_ATTRIB: &'static str = "elastic_force";

    /// Run the optimization solver on one time step.
    pub fn new(
        mesh: &'a mut TetMesh,
        params: SimParams,
        interrupt_checker: F,
    ) -> Result<Self, Error> {
        // Prepare tet mesh for simulation.
        let verts = mesh.vertex_positions().to_vec();
        mesh.attrib_or_add_data::<_, VertexIndex>(
            Self::REFERENCE_POSITION_ATTRIB,
            verts.as_slice(),
        )?;

        mesh.attrib_or_add::<_, VertexIndex>(Self::VELOCITY_ATTRIB, [0.0; 3])?;

        {
            // Compute reference element signed volumes
            let ref_volumes: Vec<f64> = mesh
                .cell_iter()
                .map(|cell| ref_tet(mesh, cell).signed_volume())
                .collect();
            if ref_volumes.iter().find(|&&x| x <= 0.0).is_some() {
                return Err(Error::InvertedReferenceElement);
            }
            mesh.set_attrib_data::<_, CellIndex>(
                Self::REFERENCE_VOLUME_ATTRIB,
                ref_volumes.as_slice(),
            )?;
        }

        {
            // Compute reference shape matrix inverses
            let ref_shape_mtx_inverses: Vec<Matrix3<f64>> = mesh
                .cell_iter()
                .map(|cell| {
                    let ref_shape_matrix = ref_tet(mesh, cell).shape_matrix();
                    // We assume that ref_shape_matrices are non-singular.
                    ref_shape_matrix.inverse().unwrap()
                }).collect();
            mesh.set_attrib_data::<_, CellIndex>(
                Self::REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                ref_shape_mtx_inverses.as_slice(),
            )?;
        }

        {
            // Add elastic strain energy and elastic force attributes.
            // These will be computed at the end of the time step.
            mesh.set_attrib::<_, CellIndex>(Self::STRAIN_ENERGY_ATTRIB, 0f64)?;
            mesh.set_attrib::<_, VertexIndex>(Self::ELASTIC_FORCE_ATTRIB, [0f64; 3])?;
        }

        let (lambda, mu) = params.material.lame_parameters();

        let density = params.material.density as f64;
        let damping = params.material.damping as f64;
        let gravity = [
            params.gravity[0] as f64,
            params.gravity[1] as f64,
            params.gravity[2] as f64,
        ];
        let mut energy_model = ElasticTetMeshEnergy::new(mesh)
            .material(lambda / mu, 1.0, density / mu, damping / mu)
            .gravity(gravity);
        if let Some(dt) = params.time_step {
            energy_model = energy_model.time_step(dt as f64);
        }

        let problem = NonLinearProblem {
            energy_model: RefCell::new(energy_model),
            interrupt_checker,
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

        Ok(FemEngine { solver: ipopt })
    }

    pub fn mesh(&self) -> TetMesh {
        self.solver.problem().energy_model.borrow().solid.clone()
    }

    pub fn params(&self) -> SimParams {
        self.solver.problem().params
    }

    /// Given a tetmesh, compute the strain energy per tetrahedron.
    fn compute_strain_energy_attrib(mesh: &mut TetMesh, lambda: f64, mu: f64) {
        // Overwrite the "strain_energy" attribute.
        let mut strain = mesh
            .remove_attrib::<CellIndex>(Self::STRAIN_ENERGY_ATTRIB)
            .unwrap();
        strain
            .iter_mut::<f64>()
            .unwrap()
            .zip(
                mesh.attrib_iter::<f64, CellIndex>(Self::REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
            ).zip(
                mesh.attrib_iter::<Matrix3<f64>, CellIndex>(
                    Self::REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                ).unwrap(),
            ).zip(mesh.tet_iter())
            .for_each(|(((strain, &vol), &ref_shape_mtx_inv), tet)| {
                *strain =
                    NeoHookeanTetEnergy::new(tet.shape_matrix(), ref_shape_mtx_inv, vol, lambda, mu)
                        .elastic_energy()
            });

        mesh.insert_attrib::<CellIndex>(Self::STRAIN_ENERGY_ATTRIB, strain)
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
            .attrib_iter::<f64, CellIndex>(Self::REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                mesh.attrib_iter::<Matrix3<f64>, CellIndex>(
                    Self::REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
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
            .remove_attrib::<VertexIndex>(Self::ELASTIC_FORCE_ATTRIB)
            .unwrap();

        {
            let forces: &mut [Vector3<f64>] =
                reinterpret_mut_slice(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

            Self::compute_elastic_forces(forces, mesh, lambda, mu);
        }

        // Reinsert forces back into the attrib map
        mesh.insert_attrib::<VertexIndex>(Self::ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
    }

    /// Update the state of the system, which includes new positions and velocities stored on the
    /// mesh, as well as previous positions stored in `prev_pos`.
    fn update_state(mesh: &mut TetMesh, prev_pos: &mut [Vector3<f64>], new_pos: &[Vector3<f64>]) {
        // Write back the velocity for the next iteration.
        for ((vel, prev_x), &x) in mesh
            .attrib_iter_mut::<[f64; 3], VertexIndex>(Self::VELOCITY_ATTRIB)
            .unwrap()
            .zip(prev_pos.iter_mut())
            .zip(new_pos.iter())
        {
            *vel = (x - *prev_x).into();
            *prev_x = x;
        }
    }

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn solve_step(&mut self) -> Result<SolveResult, Error> {
        // Solve non-linear problem
        let (status, obj) = self.solver.solve();

        let result = SolveResult {
            iterations: self.solver.problem().iteration_count() as u32,
            objective_value: obj,
        };

        match status {
            ipopt::ReturnStatus::SolveSucceeded | ipopt::ReturnStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::SolveError(e, result)),
        }
    }

    /// Run the optimization solver on one time step.
    pub fn step(&mut self) -> Result<SolveResult, Error> {
        let step_result = self.solve_step();

        let FemEngine { ref mut solver } = *self;

        let (lambda, mu) = self.params().material.lame_parameters();

        let ElasticTetMeshEnergy {
            solid: ref mut mesh,
            ref mut prev_pos,
            ..
        } = *solver.problem().energy_model.borrow_mut();

        // On success, update the mesh with new positions and useful metrics.
        if let Ok(_) = step_result {
            let new_pos: &[Vector3<f64>] = reinterpret_slice(solver.solution());
            Self::update_state(mesh, prev_pos.as_mut_slice(), new_pos);

            // Write back elastic strain energy for visualization.
            Self::compute_strain_energy_attrib(mesh, lambda, mu);

            // Write back elastic forces on each node.
            Self::compute_elastic_forces_attrib(mesh, lambda, mu);
        }

        step_result
    }
}

impl<'a, F: FnMut() -> bool + Sync> NonLinearProblem<'a, F> {
    pub fn energy_model(&self) -> Ref<ElasticTetMeshEnergy<'a>> {
        self.energy_model.borrow()
    }
    pub fn energy_model_mut(&self) -> RefMut<ElasticTetMeshEnergy<'a>> {
        self.energy_model.borrow_mut()
    }

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
impl<'a, F: FnMut() -> bool + Sync> ipopt::BasicProblem for NonLinearProblem<'a, F> {
    fn num_variables(&self) -> usize {
        self.energy_model().solid.num_vertices() * 3
    }

    fn bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let solid = &self.energy_model().solid;
        let n = solid.num_vertices();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        lo.resize(n, [-2e19; 3]);
        hi.resize(n, [2e19; 3]);

        if let Ok(fixed_verts) = solid.attrib_as_slice::<i8, VertexIndex>("fixed") {
            // Find and set fixed vertices.
            lo.iter_mut()
                .zip(hi.iter_mut())
                .zip(fixed_verts.iter().zip(solid.vertex_positions().iter()))
                .filter(|&(_, (&fixed, _))| fixed != 0)
                .for_each(|((l, u), (_, p))| {
                    *l = *p;
                    *u = *p;
                });
        }
        (reinterpret_vec(lo), reinterpret_vec(hi))
    }

    fn initial_point(&self) -> Vec<Number> {
        let solid = &self.energy_model().solid;
        reinterpret_slice(solid.vertex_positions()).to_vec()
    }

    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        let mut model = self.energy_model_mut();
        model.update(x);
        *obj = model.energy();
        true
    }

    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        let mut model = self.energy_model_mut();
        model.update(x);
        grad_f.copy_from_slice(reinterpret_slice(model.energy_gradient()));

        true
    }
}

impl<'a, F: FnMut() -> bool + Sync> ipopt::NewtonProblem for NonLinearProblem<'a, F> {
    fn num_hessian_non_zeros(&self) -> usize {
        self.energy_model().energy_hessian_size()
    }
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut model = self.energy_model_mut();
        for (i, &MatrixElementTriplet { ref idx, .. }) in model.energy_hessian().iter().enumerate()
        {
            rows[i] = idx.row as Index;
            cols[i] = idx.col as Index;
        }

        true
    }
    fn hessian_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool {
        let mut model = self.energy_model_mut();
        model.update(x);

        for (i, elem) in model.energy_hessian().iter().enumerate() {
            vals[i] = elem.val as Number;
        }

        true
    }
}

#[derive(Debug)]
pub enum Error {
    AttribError(attrib::Error),
    InvertedReferenceElement,
    SolveError(ipopt::ReturnStatus, SolveResult),
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
    fn simple_tet_static_test() {
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
        mesh.add_attrib_data::<i8, VertexIndex>("fixed", vec![0, 0, 1, 1, 0, 0])
            .unwrap();

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
            .unwrap();

        assert!(
            FemEngine::new(&mut mesh, STATIC_PARAMS, || false)
                .unwrap()
                .step()
                .is_ok()
        );
    }

    #[test]
    fn simple_tet_dynamic_test() {
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
        mesh.add_attrib_data::<i8, VertexIndex>("fixed", vec![0, 0, 1, 1, 0, 0])
            .unwrap();

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
            .unwrap();

        assert!(
            FemEngine::new(&mut mesh, DYNAMIC_PARAMS, || false)
                .unwrap()
                .step()
                .is_ok()
        );
    }

    #[test]
    fn torus_medium_test() {
        let mut mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        assert!(
            FemEngine::new(&mut mesh, DYNAMIC_PARAMS, || false)
                .unwrap()
                .step()
                .is_ok()
        );
    }

    #[test]
    fn torus_large_test() {
        let mut mesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets_large.vtk")).unwrap();
        assert!(
            FemEngine::new(&mut mesh, DYNAMIC_PARAMS, || false)
                .unwrap()
                .step()
                .is_ok()
        );
    }

    #[test]
    fn torus_long_test() {
        let mut mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();

        let mut engine = FemEngine::new(&mut mesh, DYNAMIC_PARAMS, || false).unwrap();
        for _i in 0..50 {
            assert!(engine.step().is_ok());
        }
    }
}
