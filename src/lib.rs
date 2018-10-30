extern crate geometry as geo;
extern crate hdkrs;
extern crate softy;

#[macro_use]
extern crate lazy_static;

mod api;

pub use hdkrs::{cffi, interop};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialProperties {
    pub bulk_modulus: f32,
    pub shear_modulus: f32,
    pub density: f32,
    pub damping: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimParams {
    pub material: MaterialProperties,
    pub time_step: f32,
    pub gravity: f32,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub volume_constraint: bool,
}

/// Result reported from `register_new_solver` function.
/// In case of failure, solver_id is set to a negative number.
#[repr(C)]
pub struct RegistryResult {
    solver_id: i64,
}

/// Register a new solver in the registry. (C side)
/// This function consumes `tetmesh` and `polymesh`.
#[no_mangle]
pub unsafe extern "C" fn register_new_solver(
    tetmesh: *mut cffi::TetMesh,
    _polymesh: *mut cffi::PolyMesh,
    sim_params: SimParams,
) -> RegistryResult {
    if let Some(tetmesh) = interop::into_box(tetmesh) {
        match api::register_new_solver(
            tetmesh,
            sim_params.into(),
        ) {
            Ok((id, _)) => RegistryResult { solver_id: id as i64 },
            Err(_) => RegistryResult { solver_id: -1 },
        }
    } else {
        RegistryResult {
            solver_id: -1,
        }
    }
}

/// Validate the given id from C side by converting it to a Rust option type.
fn validate_id(id: i64) -> Option<u32> {
    if id == (id as u32) as i64 {
        Some(id as u32)
    } else {
        None
    }
}

/// Validate the given solver pointer from C side by converting it to an Arc if it still exists in
/// the registry.
unsafe fn validate_solver_ptr(solver: *mut SolverPtr) -> Option<Arc<Mutex<dyn api::Solver>>> {
    interop::into_box(solver as *mut Arc<Mutex<dyn api::Solver>>).map(|x| *x)
}

/// Move the `Arc` to the heap an return its raw pointer.
fn solver_ptr(solver: Arc<Mutex<dyn api::Solver>>) -> *mut SolverPtr {
    Box::into_raw(Box::new(solver)) as *mut SolverPtr
}

/// Opaque struct to represent `Arc<Mutex<api::Solver>>` on the C side.
pub struct SolverPtr;

#[repr(C)]
pub struct SolverResult {
    /// ID of the solver in the registry.
    id: i64,

    /// A non-owning pointer to a Solver struct from the registry.
    solver: *mut SolverPtr,
}

/// Register a new solver in the registry. (C side)
/// This function consumes `tetmesh` and `polymesh`.
#[no_mangle]
pub unsafe extern "C" fn get_solver(
    solver_id: i64,
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    sim_params: SimParams,
) -> SolverResult {
    match api::get_solver(
        validate_id(solver_id),
        interop::into_box(tetmesh),
        interop::into_box(polymesh),
        sim_params.into(),
    ) {
        Ok((id, solver)) => {
            assert!(Arc::strong_count(&solver) != 1);
            SolverResult {
                id: id as i64,
                solver: solver_ptr(solver),
            }
        },
        Err(_) => SolverResult {
            id: -1,
            solver: ::std::ptr::null_mut(),
        },
    }
}

/// Delete all solvers in the registry.
#[no_mangle]
pub unsafe extern "C" fn clear_solver_registry(
) {
    api::clear_solver_registry()
}

#[repr(C)]
pub struct StepResult {
    tetmesh: *mut cffi::TetMesh,
    cook_result: cffi::CookResult,
}

/// Perform one step of the solve given a solver.
#[no_mangle]
pub unsafe extern "C" fn step(
    solver: *mut SolverPtr,
    tetmesh_points: *mut cffi::PointCloud,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> StepResult {
    if let Some(solver) = validate_solver_ptr(solver) {
        let (tetmesh_mb, cook_result) = api::step(
            solver,
            interop::into_box(tetmesh_points),
            interop::interrupt_callback(interrupt_checker, check_interrupt),
        );
        if let Some(solver_tetmesh) = tetmesh_mb {
            StepResult {
                tetmesh: Box::into_raw(Box::new(solver_tetmesh)),
                cook_result: cook_result.into(),
            }
        } else {
            StepResult {
                tetmesh: ::std::ptr::null_mut(),
                cook_result: cook_result.into(),
            }
        }
    } else {
        StepResult {
            tetmesh: ::std::ptr::null_mut(),
            cook_result: hdkrs::interop::CookResult::Error("Invalid solver".to_string()).into(),
        }
    }
}

#[repr(C)]
pub struct SolveResult {
    solver_id: i64,
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    cook_result: cffi::CookResult,
}

/// Gets a valid solver and performs one step of the solve.
#[no_mangle]
pub unsafe extern "C" fn solve(
    solver_id: i64,
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    sim_params: SimParams,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> SolveResult {
    let (data, cook_result) = api::cook(
        validate_id(solver_id),
        interop::into_box(tetmesh),
        interop::into_box(polymesh),
        sim_params.into(),
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    );
    if let Some((new_solver_id, solver_tetmesh)) = data {
        SolveResult {
            solver_id: new_solver_id as i64,
            tetmesh: Box::into_raw(Box::new(solver_tetmesh)),
            polymesh: ::std::ptr::null_mut(),
            cook_result: cook_result.into(),
        }
    } else {
        SolveResult {
            solver_id: -1,
            tetmesh: ::std::ptr::null_mut(),
            polymesh: ::std::ptr::null_mut(),
            cook_result: cook_result.into(),
        }
    }
}
