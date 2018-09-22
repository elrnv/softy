use softy;
use geo::mesh::{PolyMesh, TetMesh};
use hdkrs::interop::CookResult;
use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap};
use SimParams;
use MaterialProperties;

mod solver;

use self::solver::Solver;

/// A registry for solvers to be used as a global instance. We use a `HashMap` instead of `Vec` to
/// avoid dealing with fragmentation.
struct SolverRegistry {
    /// Solver key counter. Having this is not strictly necessary, but it helps identify an
    /// unoccupied key in the registry quickly. We essentially increment this key indefinitely to
    /// generate new keys.
    pub key_counter: u32,

    pub solver_table: HashMap<u32, Arc<Mutex<dyn Solver + Send>>>,
}

lazy_static! {
    static ref SOLVER_REGISTRY: RwLock<SolverRegistry> = {
        RwLock::new(SolverRegistry {
            key_counter: 0,
            solver_table: HashMap::new(),
        })
    };
}

//fn get_solver_from_registry(solver_id: Option<u32>, registry: &MutexGuard<SolverRegistry>)
//    -> Option<(u32, &'static mut (dyn Solver + Send + 'static))>
//{
//}

#[derive(Debug)]
enum Error {
    RegistryFull,
    MissingSolverAndMesh,
    SolverCreate(softy::Error),
}

fn get_solver(
    solver_id: Option<u32>,
    tetmesh: Option<Box<TetMesh<f64>>>,
    _polymesh: Option<Box<PolyMesh<f64>>>,
    params: SimParams,
) -> Result<(u32, Arc<Mutex<dyn Solver + Send>>), Error>
{
    //let reg_solver = get_solver_from_registry(solver_id, &registry);
    // Verify that the given id points to a valid solver.
    let reg_solver = solver_id.and_then(|id| {
        SOLVER_REGISTRY.read().unwrap().solver_table.get(&id).map(|x| (id, Arc::clone(x)))
    });

    if let Some((id, solver)) = reg_solver {
        Ok((id, solver))
    } else {
        // Given solver id is invalid, need to create a new solver.
        if let Some(tetmesh) = tetmesh {
            let solver = match softy::FemEngine::new({*tetmesh}, params.into()) {
                Ok(solver) => solver,
                Err(err) => return Err(Error::SolverCreate(err)),
            };

            {
                let SolverRegistry { ref mut key_counter, ref mut solver_table } = *SOLVER_REGISTRY.write().unwrap();

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
                    let new_solver = solver_table.entry(id).or_insert(Arc::new(Mutex::new(solver)));
                    Ok((id, Arc::clone(new_solver)))
                } else {
                    Err(Error::RegistryFull)
                }
            }
        } else {
            Err(Error::MissingSolverAndMesh)
        }
    }
}

impl Into<softy::SimParams> for SimParams {
    fn into(self) -> softy::SimParams {
        let SimParams {
            material: MaterialProperties {
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
            time_step: if time_step > 0.0 { Some(time_step) } else { None },
            gravity: [0.0, -gravity, 0.0],
            tolerance,
        }
    }
}

/// Main entry point for C FFI.
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
        Err(err)=> (None, CookResult::Error(format!("Couldn't find or create a valid solver: {:?}", err))),
    }
}

fn convert_to_cookresult(res: softy::SimResult) -> CookResult {
    match res {
        softy::SimResult::Success(msg) => CookResult::Success(msg),
        softy::SimResult::Warning(msg) => CookResult::Warning(msg),
        softy::SimResult::Error(msg) => CookResult::Error(msg),
    }
}

