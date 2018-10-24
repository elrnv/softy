use geo::mesh::{PolyMesh, TetMesh, PointCloud};
use geo::NumVertices;
use hdkrs::interop::CookResult;
use softy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use MaterialProperties;
use SimParams;

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
    _polymesh: Option<Box<PolyMesh<f64>>>,
    params: SimParams,
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
            Some(tetmesh) => register_new_solver(tetmesh, params),
            None => Err(Error::MissingSolverAndMesh),
        }
    }
}

impl Into<softy::SimParams> for SimParams {
    fn into(self) -> softy::SimParams {
        let SimParams {
            material:
                MaterialProperties {
                    bulk_modulus,
                    shear_modulus,
                    density,
                    damping,
                },
            time_step,
            gravity,
            tolerance,
        } = self;
        softy::SimParams {
            material: softy::MaterialProperties {
                bulk_modulus,
                shear_modulus,
                density,
                damping,
            },
            time_step: if time_step > 0.0 {
                Some(time_step)
            } else {
                None
            },
            gravity: [0.0, -gravity, 0.0],
            tolerance,
        }
    }
}

/// Register a new solver in the registry. (Rust side)
#[inline]
pub(crate) fn register_new_solver(
    tetmesh: Box<TetMesh<f64>>,
    params: SimParams,
) -> Result<(u32, Arc<Mutex<dyn Solver>>), Error> {
    let solver = match softy::FemEngine::new({ *tetmesh }, params.into()) {
        Ok(solver) => solver,
        Err(err) => return Err(Error::SolverCreate(err)),
    };

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
            .or_insert(Arc::new(Mutex::new(solver)));
        Ok((id, Arc::clone(new_solver)))
    } else {
        Err(Error::RegistryFull)
    }
}

/// Perform one solve step given a solver.
#[inline]
pub(crate) fn step<F>(
    solver: Arc<Mutex<dyn Solver>>,
    tetmesh_points: Option<Box<PointCloud<f64>>>,
    check_interrupt: F,
) -> (Option<TetMesh<f64>>, CookResult)
where
    F: Fn() -> bool + Sync + Send + 'static,
{
    let solver = &mut *solver.lock().unwrap();
    solver.set_interrupter(Box::new(check_interrupt));

    // Update mesh points
    if let Some(pts) = tetmesh_points {
        if !solver.update_mesh_vertices(&pts) {
            return (None, CookResult::Error(
                    format!("Input points ({}) don't coincide with solver mesh ({}).", (*pts).num_vertices(), solver.mesh_ref().num_vertices())))
        }
    }

    let cook_result = convert_to_cookresult(solver.solve().into());
    let solver_tetmesh = solver.mesh_ref().clone();
    (Some(solver_tetmesh), cook_result)
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
    params: SimParams,
    check_interrupt: F,
) -> (Option<(u32, TetMesh<f64>)>, CookResult)
where
    F: Fn() -> bool + Sync + Send + 'static,
{
    match get_solver(solver_id, tetmesh, polymesh, params.into()) {
        Ok((solver_id, solver)) => {
            let mut solver = solver.lock().unwrap();
            solver.set_interrupter(Box::new(check_interrupt));
            let cook_result = convert_to_cookresult(solver.solve().into());
            let solver_tetmesh = solver.mesh_ref().clone();
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
