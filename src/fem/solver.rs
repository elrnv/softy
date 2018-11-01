use crate::attrib_names::*;
use crate::constraints::total_volume::VolumeConstraint;
use crate::energy::*;
use crate::energy_models::{
    gravity::Gravity,
    momentum::MomentumPotential,
    volumetric_neohookean::{ElasticTetMeshEnergyBuilder, NeoHookeanTetEnergy},
};
use crate::geo::math::{Matrix3, Vector3};
use crate::geo::mesh::{tetmesh::TetCell, topology::*, Attrib};
use crate::geo::ops::{ShapeMatrix, Volume};
use crate::geo::prim::Tetrahedron;
use ipopt::{self, Ipopt, SolverDataMut};
use reinterpret::*;
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use super::NonLinearProblem;
use super::SimParams;
use crate::Error;
use crate::PointCloud;
use crate::TetMesh;

/// Result from one simulation step.
#[derive(Debug)]
pub struct SolveResult {
    /// Number of inner iterations of the step result.
    pub iterations: u32,
    /// The value of the objective at the end of the time step.
    pub objective_value: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ElasticityProperties {
    /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
    /// incompressibility. The larger the value, the more incompressible the material is.
    /// Think of this as "Volume Stiffness"
    pub bulk_modulus: f32,
    /// Shear modulus measures the material's resistance to shear deformation. The larger the
    /// value, the more it resists changes in shape. Think of this as "Shape Stiffness"
    pub shear_modulus: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialProperties {
    /// Properties determining the elastic behaviour of a simulated solid.
    pub elasticity: ElasticityProperties,
    /// The density of the material.
    pub density: f32,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model.
    pub damping: f32,
}

impl ElasticityProperties {
    pub fn from_young_poisson(young: f32, poisson: f32) -> Self {
        ElasticityProperties {
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
    Tetrahedron::from_indexed_slice(indices, ref_pos)
}

/// Finite element engine.
pub struct Solver {
    /// Non-linear solver.
    solver: Ipopt<NonLinearProblem>,
    /// Step count (outer iterations). These count the number of times the function `step` was
    /// called.
    step_count: usize,
}

impl Solver {
    /// Run the optimization solver on one time step.
    pub fn new(mut mesh: TetMesh, params: SimParams) -> Result<Self, Error> {
        // Prepare tet mesh for simulation.
        let verts = mesh.vertex_positions().to_vec();
        mesh.attrib_or_add_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts.as_slice())?;

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
            mesh.set_attrib_data::<_, CellIndex>(REFERENCE_VOLUME_ATTRIB, ref_volumes.as_slice())?;
        }

        {
            // Compute reference shape matrix inverses
            let ref_shape_mtx_inverses: Vec<Matrix3<f64>> = mesh
                .cell_iter()
                .map(|cell| {
                    let ref_shape_matrix = ref_tet(&mesh, cell).shape_matrix();
                    // We assume that ref_shape_matrices are non-singular.
                    ref_shape_matrix.inverse().unwrap()
                })
                .collect();
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

        let (lambda, mu) = params.material.elasticity.lame_parameters();

        // Normalize material parameters with mu.
        let lambda = lambda / mu;
        let density = params.material.density as f64 / mu;
        // premultiply damping by timestep reciprocal.
        let damping = if let Some(dt) = params.time_step {
            if dt != 0.0 {
                1.0 / dt as f64
            } else {
                0.0
            }
        } else {
            0.0
        } * params.material.damping as f64
            / mu;
        let gravity = [
            params.gravity[0] as f64,
            params.gravity[1] as f64,
            params.gravity[2] as f64,
        ];

        // Get previous position vector from the tetmesh.
        let prev_pos = Rc::new(RefCell::new(
            reinterpret_slice(mesh.vertex_positions()).to_vec(),
        ));

        // Initialize volume constraint
        let volume_constraint = if params.volume_constraint {
            Some(VolumeConstraint::new(&mesh)) //, Rc::clone(&prev_pos)))
        } else {
            None
        };

        let mesh = Rc::new(RefCell::new(mesh));

        let energy_model_builder =
            ElasticTetMeshEnergyBuilder::new(Rc::clone(&mesh)).material(lambda, 1.0, damping);

        let momentum_potential = params
            .time_step
            .map(|dt| MomentumPotential::new(Rc::clone(&mesh), density, dt as f64));

        let problem = NonLinearProblem {
            tetmesh: Rc::clone(&mesh),
            prev_pos,
            energy_model: energy_model_builder.build(),
            gravity: Gravity::new(Rc::clone(&mesh), density, &gravity),
            momentum_potential,
            volume_constraint,
            interrupt_checker: Box::new(|| false),
            iterations: 0,
            params,
        };

        let mut ipopt = Ipopt::new(problem);

        ipopt.set_option("tol", params.tolerance as f64);
        ipopt.set_option("acceptable_tol", 10.0 * params.tolerance as f64);
        ipopt.set_option("max_iter", params.max_iterations as i32);
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

        Ok(Solver {
            solver: ipopt,
            step_count: 0,
        })
    }

    /// Set the interrupt checker to the given function.
    pub fn set_interrupter(&mut self, checker: Box<FnMut() -> bool>) {
        self.solver.solver_data_mut().problem.interrupt_checker = checker;
    }

    /// Get an immutable borrow for the underlying `TetMesh`.
    pub fn borrow_mesh(&self) -> Ref<'_, TetMesh> {
        self.solver.solver_data_ref().problem.tetmesh.borrow()
    }

    /// Get a mutable borrow for the underlying `TetMesh`.
    pub fn borrow_mut_mesh(&mut self) -> RefMut<'_, TetMesh> {
        self.solver.solver_data_ref().problem.tetmesh.borrow_mut()
    }

    /// Get solver parameters.
    pub fn params(&mut self) -> SimParams {
        self.solver.solver_data_mut().problem.params
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
        let fixed_iter = tetmesh.attrib_iter::<i8, VertexIndex>(FIXED_ATTRIB)?;
        prev_pos
            .iter_mut()
            .zip(pts.vertex_iter())
            .zip(fixed_iter)
            .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
            .for_each(|(pos, new_pos)| *pos = Vector3::from(*new_pos));
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
                mesh.attrib_iter::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
            )
            .zip(
                mesh.attrib_iter::<Matrix3<f64>, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
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
                mesh.attrib_iter::<Matrix3<f64>, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
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
            let (lambda, mu) = self.params().material.elasticity.lame_parameters();

            let SolverDataMut {
                ref mut problem,
                ref primal_variables,
                ..
            } = self.solver.solver_data_mut();

            let displacement: &[Vector3<f64>] = reinterpret_slice(primal_variables);

            // Get the tetmesh and prev_pos so we can update fixed vertices only.
            let mesh = &mut problem.tetmesh.borrow_mut();
            let mut prev_pos = problem.prev_pos.borrow_mut();

            Self::update_state(mesh, prev_pos.as_mut_slice(), displacement);

            // Write back elastic strain energy for visualization.
            Self::compute_strain_energy_attrib(mesh, lambda, mu);

            // Write back elastic forces on each node.
            Self::compute_elastic_forces_attrib(mesh, lambda, mu);
        }

        step_result
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
        material: MaterialProperties {
            elasticity: ElasticityProperties {
                bulk_modulus: 1750e6,
                shear_modulus: 10e6,
            },
            density: 1000.0,
            damping: 0.0,
        },
        gravity: [0.0f32, -9.81, 0.0],
        time_step: None,
        tolerance: 1e-9,
        max_iterations: 800,
        volume_constraint: false,
    };

    const DYNAMIC_PARAMS: SimParams = SimParams {
        material: MaterialProperties {
            elasticity: ElasticityProperties {
                bulk_modulus: 1e6,
                shear_modulus: 1e5,
            },
            ..STATIC_PARAMS.material
        },
        gravity: [0.0f32, 0.0, 0.0],
        time_step: Some(0.01),
        ..STATIC_PARAMS
    };

    fn make_one_tet_mesh() -> TetMesh {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ];
        let indices = vec![0, 2, 1, 3];
        let mut mesh = TetMesh::new(verts, indices);
        mesh.add_attrib_data::<i8, VertexIndex>(FIXED_ATTRIB, vec![1, 0, 0, 0])
            .unwrap();

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, ref_verts)
            .unwrap();
        mesh
    }

    fn make_three_tet_mesh_with_verts(verts: Vec<[f64; 3]>) -> TetMesh {
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

    #[test]
    fn one_tet_test() {
        let mesh = make_one_tet_mesh();

        assert!(Solver::new(mesh, STATIC_PARAMS).unwrap().step().is_ok());
    }

    #[test]
    fn one_tet_volume_constraint_test() {
        let mesh = make_one_tet_mesh();

        let params = SimParams {
            volume_constraint: true,
            ..STATIC_PARAMS
        };

        assert!(Solver::new(mesh, params).unwrap().step().is_ok());
    }

    /*
     * Three tet tests
     */

    #[test]
    fn three_tets_static_test() {
        let mesh = make_three_tet_mesh();
        let mut solver = Solver::new(mesh, STATIC_PARAMS).unwrap();
        assert!(solver.step().is_ok());
        let solution = solver.borrow_mesh();
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_static_expected.vtk")).unwrap();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn three_tets_dynamic_test() {
        let mesh = make_three_tet_mesh();
        let mut solver = Solver::new(mesh, DYNAMIC_PARAMS).unwrap();
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
        let params = SimParams {
            volume_constraint: true,
            ..STATIC_PARAMS
        };
        let mut solver = Solver::new(mesh, params).unwrap();
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
        let params = SimParams {
            volume_constraint: true,
            ..DYNAMIC_PARAMS
        };
        let mut solver = Solver::new(mesh, params).unwrap();
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

        let mut solver = Solver::new(mesh, DYNAMIC_PARAMS).unwrap();

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

        let params = SimParams {
            volume_constraint: true,
            ..DYNAMIC_PARAMS
        };

        let mut solver = Solver::new(mesh, params).unwrap();

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
        material: MaterialProperties {
            elasticity: ElasticityProperties {
                bulk_modulus: 300e6,
                shear_modulus: 100e6,
            },
            ..STATIC_PARAMS.material
        },
        gravity: [0.0f32, 0.0, 0.0],
        ..STATIC_PARAMS
    };

    #[test]
    fn box_stretch_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk")).unwrap();
        let mut solver = Solver::new(mesh, STRETCH_PARAMS).unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn box_stretch_volume_constraint_test() {
        let params = SimParams {
            volume_constraint: true,
            ..STRETCH_PARAMS
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk")).unwrap();
        let mut solver = Solver::new(mesh, params).unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_const_volume.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn box_twist_test() {
        let params = SimParams {
            material: MaterialProperties {
                elasticity: ElasticityProperties::from_young_poisson(1000.0, 0.0),
                ..STRETCH_PARAMS.material
            },
            ..STRETCH_PARAMS
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = Solver::new(mesh, params).unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    #[test]
    fn box_twist_volume_constraint_test() {
        let params = SimParams {
            material: MaterialProperties {
                elasticity: ElasticityProperties::from_young_poisson(1000.0, 0.0),
                ..STRETCH_PARAMS.material
            },
            volume_constraint: true,
            ..STRETCH_PARAMS
        };
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
        let mut solver = Solver::new(mesh, params).unwrap();
        assert!(solver.step().is_ok());
        let expected: TetMesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk")).unwrap();
        let solution = solver.borrow_mesh();
        compare_meshes(&solution, &expected);
    }

    /*
     * More complex tests
     */

    #[test]
    fn torus_medium_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        assert!(Solver::new(mesh, DYNAMIC_PARAMS).unwrap().step().is_ok());
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn torus_long_test() {
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();

        let mut engine = Solver::new(mesh, DYNAMIC_PARAMS).unwrap();
        for _i in 0..10 {
            //save_tetmesh_ascii(
            //    engine.borrow_mesh(),
            //    &PathBuf::from(format!("./out/mesh_{}.vtk", i)),
            //);
            let res = engine.step();
            assert!(res.is_ok());
        }
    }
}
