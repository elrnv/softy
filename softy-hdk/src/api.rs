use crate::{EL_SoftyMaterialProperties, EL_SoftySimParams};
use geo::mesh::{PointCloud, PolyMesh, TetMesh};
use geo::NumVertices;
use hdkrs::interop::CookResult;
use softy::{self, fem};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

mod solver;

pub use self::solver::*;

/// A registry for solvers to be used as a global instance. We use a `HashMap` instead of `Vec` to
/// avoid dealing with fragmentation.
struct SolverRegistry {
    /// Solver key counter. Having this is not strictly necessary, but it helps identify an
    /// unoccupied key in the registry quickly. We essentially increment this key indefinitely to
    /// generate new keys.
    pub key_counter: u32,

    pub solver_table: HashMap<u32, Arc<Mutex<dyn Solver>>>,
}

lazy_static! {
    static ref SOLVER_REGISTRY: RwLock<SolverRegistry> = {
        RwLock::new(SolverRegistry {
            key_counter: 0,
            solver_table: HashMap::new(),
        })
    };
}

#[derive(Debug)]
pub(crate) enum Error {
    RegistryFull,
    MissingSolverAndMesh,
    SolverCreate(softy::Error),
}

pub(crate) fn get_solver(
    solver_id: Option<u32>,
    tetmesh: Option<Box<TetMesh<f64>>>,
    polymesh: Option<Box<PolyMesh<f64>>>,
    params: EL_SoftySimParams,
) -> Result<(u32, Arc<Mutex<dyn Solver>>), Error> {
    // Verify that the given id points to a valid solver.
    let reg_solver = solver_id.and_then(|id| {
        SOLVER_REGISTRY
            .read()
            .unwrap()
            .solver_table
            .get(&id)
            .map(|x| (id, Arc::clone(x)))
    });

    if let Some((id, solver)) = reg_solver {
        Ok((id, solver))
    } else {
        // Given solver id is invalid, need to create a new solver.
        match tetmesh {
            Some(tetmesh) => register_new_solver(*tetmesh, polymesh, params),
            None => Err(Error::MissingSolverAndMesh),
        }
    }
}

impl Into<softy::SimParams> for EL_SoftySimParams {
    fn into(self) -> softy::SimParams {
        let EL_SoftySimParams {
            time_step,
            gravity,
            tolerance,
            max_iterations,
            outer_tolerance,
            max_outer_iterations,
            friction_iterations,
            print_level,
            derivative_test,
            mu_strategy,
            max_gradient_scaling,
            log_file,
            ..
        } = self;
        softy::SimParams {
            time_step: if time_step > 0.0 {
                Some(time_step)
            } else {
                None
            },
            gravity: [0.0, -gravity, 0.0],
            tolerance,
            max_iterations,
            outer_tolerance,
            max_outer_iterations,
            friction_iterations,
            print_level,
            derivative_test,
            mu_strategy: match mu_strategy {
                0 => softy::MuStrategy::Monotone,
                _ => softy::MuStrategy::Adaptive,
            },
            max_gradient_scaling,
            log_file: unsafe {
                std::ffi::CStr::from_ptr(log_file)
                    .to_str()
                    .ok()
                    .map(|path| std::path::PathBuf::from(path.to_string()))
            },
        }
    }
}

impl Into<softy::Material> for EL_SoftySimParams {
    fn into(self) -> softy::Material {
        let EL_SoftySimParams {
            material:
                EL_SoftyMaterialProperties {
                    bulk_modulus,
                    shear_modulus,
                    density,
                    damping,
                },
            volume_constraint,
            ..
        } = self;
        softy::Material {
            elasticity: softy::ElasticityParameters {
                bulk_modulus,
                shear_modulus,
            },
            incompressibility: volume_constraint,
            density,
            damping,
        }
    }
}

impl Into<softy::SmoothContactParams> for EL_SoftySimParams {
    fn into(self) -> softy::SmoothContactParams {
        let EL_SoftySimParams {
            contact_kernel,
            contact_type,
            contact_radius_multiplier,
            smoothness_tolerance,
            dynamic_friction,
            friction_inner_iterations,
            friction_tolerance,
            ..
        } = self;
        let radius_multiplier = f64::from(contact_radius_multiplier);
        let tolerance = f64::from(smoothness_tolerance);
        softy::SmoothContactParams {
            kernel: match contact_kernel {
                0 => softy::KernelType::Interpolating { radius_multiplier },
                1 => softy::KernelType::Approximate {
                    tolerance,
                    radius_multiplier,
                },
                2 => softy::KernelType::Cubic { radius_multiplier },
                _ => softy::KernelType::Global { tolerance },
            },
            contact_type: match contact_type {
                0 => softy::ContactType::Implicit,
                _ => softy::ContactType::Point,
            },
            friction_params: Some(softy::FrictionParams {
                dynamic_friction: f64::from(dynamic_friction),
                inner_iterations: friction_inner_iterations as usize,
                tolerance: f64::from(friction_tolerance),
                print_level: 5,
            }),
        }
    }
}

/// Register a new solver in the registry. (Rust side)
//#[allow(clippy::needless_pass_by_value)]
#[inline]
pub(crate) fn register_new_solver(
    tetmesh: TetMesh<f64>,
    shell: Option<Box<PolyMesh<f64>>>,
    params: EL_SoftySimParams,
) -> Result<(u32, Arc<Mutex<dyn Solver>>), Error> {
    // Build a basic solver with a solid material.
    let mut solver_builder = fem::SolverBuilder::new(params.into());

    solver_builder
        .add_solid(tetmesh)
        .solid_material(params.into());

    // Add a shell if one was given.
    if let Some(polymesh) = shell {
        solver_builder
            .add_shell((*polymesh).reversed())
            .smooth_contact_params(params.into());
    }

    let solver = match solver_builder.build() {
        Ok(solver) => solver,
        Err(err) => return Err(Error::SolverCreate(err)),
    };

    // Get a mutable reference to the solver registry.
    let SolverRegistry {
        ref mut key_counter,
        ref mut solver_table,
    } = *SOLVER_REGISTRY.write().unwrap();

    // Find a place in the registry to insert the new solver
    let mut counter = 0_u32; // use this to catch the case when registry is full.
    let vacant_id = loop {
        *key_counter += 1;
        counter += 1;
        if !solver_table.contains_key(key_counter) {
            break Some(*key_counter);
        } else if counter == 0 {
            break None;
        }
    };

    if let Some(id) = vacant_id {
        // Insert a new solver at the location we determined is vacant.
        let new_solver = solver_table
            .entry(id)
            .or_insert_with(|| Arc::new(Mutex::new(solver)));
        Ok((id, Arc::clone(new_solver)))
    } else {
        Err(Error::RegistryFull)
    }
}

/// Perform one solve step given a solver.
#[inline]
pub(crate) fn step<F>(
    solver: &mut Solver,
    tetmesh_points: Option<Box<PointCloud<f64>>>,
    polymesh_points: Option<Box<PointCloud<f64>>>,
    check_interrupt: F,
) -> (Option<TetMesh<f64>>, Option<PolyMesh<f64>>, CookResult)
where
    F: Fn() -> bool + Sync + Send + 'static,
{
    solver.set_interrupter(Box::new(check_interrupt));

    // Update mesh points
    if let Some(pts) = tetmesh_points {
        match solver.update_solid_vertices(&pts) {
            Err(softy::Error::SizeMismatch) =>
                return (None, None, CookResult::Error(
                        format!("Input points ({}) don't coincide with solver TetMesh ({}).",
                        (*pts).num_vertices(), solver.borrow_mesh().num_vertices()))),
            Err(softy::Error::AttribError(err)) =>
                return (None, None, CookResult::Warning(
                        format!("Failed to find 8-bit integer attribute \"fixed\", which marks animated vertices. ({:?})", err))),
            Err(err) =>
                return (None, None, CookResult::Error(
                        format!("Error updating tetmesh vertices. ({:?})", err))),
            _ => {}
        }
    }

    if let Some(pts) = polymesh_points {
        match solver.update_shell_vertices(&pts) {
            Err(softy::Error::SizeMismatch) => {
                return (
                    None,
                    None,
                    CookResult::Error(format!(
                        "Input points ({}) don't coincide with solver PolyMesh ({}).",
                        (*pts).num_vertices(),
                        solver
                            .try_borrow_kinematic_mesh()
                            .map(|x| x.num_vertices() as isize)
                            .unwrap_or(-1)
                    )),
                )
            }
            Err(softy::Error::NoKinematicMesh) => {
                return (
                    None,
                    None,
                    CookResult::Warning("Missing kinematic mesh.".to_string()),
                )
            }
            Err(err) => {
                return (
                    None,
                    None,
                    CookResult::Error(format!("Error updating polymesh vertices. ({:?})", err)),
                )
            }
            _ => {}
        }
    }

    let cook_result = convert_to_cookresult(solver.solve().into());
    let solver_trimesh = solver
        .try_borrow_kinematic_mesh()
        .map(|x| PolyMesh::from(x.clone()));
    let solver_tetmesh = solver.borrow_mesh().clone();
    (Some(solver_tetmesh), solver_trimesh, cook_result)
}

/// Clear all solvers from the registry and reset the counter.
#[inline]
pub(crate) fn clear_solver_registry() {
    let SolverRegistry {
        ref mut key_counter,
        ref mut solver_table,
    } = *SOLVER_REGISTRY.write().unwrap();
    *key_counter = 0;
    solver_table.clear();
}

/// Get a solver given the parameters and compute one step.
#[inline]
pub(crate) fn cook<F>(
    solver_id: Option<u32>,
    tetmesh: Option<Box<TetMesh<f64>>>,
    polymesh: Option<Box<PolyMesh<f64>>>,
    params: EL_SoftySimParams,
    check_interrupt: F,
) -> (Option<(u32, TetMesh<f64>)>, CookResult)
where
    F: Fn() -> bool + Sync + Send + 'static,
{
    match get_solver(solver_id, tetmesh, polymesh, params) {
        Ok((solver_id, solver)) => {
            let mut solver = solver.lock().unwrap();
            solver.set_interrupter(Box::new(check_interrupt));
            let cook_result = convert_to_cookresult(solver.solve().into());
            let solver_tetmesh = solver.borrow_mesh().clone();
            (Some((solver_id, solver_tetmesh)), cook_result)
        }
        Err(err) => (
            None,
            CookResult::Error(format!("Couldn't find or create a valid solver: {:?}", err)),
        ),
    }
}

fn convert_to_cookresult(res: softy::SimResult) -> CookResult {
    match res {
        softy::SimResult::Success(msg) => CookResult::Success(msg),
        softy::SimResult::Warning(msg) => CookResult::Warning(msg),
        softy::SimResult::Error(msg) => CookResult::Error(msg),
    }
}