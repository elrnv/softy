extern crate geometry as geo;
extern crate hdkrs;
extern crate softy;

mod api;

pub use hdkrs::{cffi, interop};

/// Opaque struct that represents the FEM Solver.
pub struct FemEngine;

/// Custom cook result that encapsulates the solver struct.
#[repr(C)]
pub struct SolverResult {
    solver: *mut FemEngine,
    result: cffi::CookResult,
}

/// Create a new solver.
#[no_mangle]
pub unsafe extern "C" fn new_solver(
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    sim_params: api::SimParams,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> SolverResult {
    match api::new_solver(
        interop::mesh(tetmesh),
        interop::mesh(polymesh),
        sim_params.into(),
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    ) {
        Ok(solver) => SolverResult {
            solver: Box::into_raw(solver) as *mut FemEngine,
            result: interop::CookResult::Success(String::new()).into()
        },
        Err(error) => SolverResult {
            solver: ::std::ptr::null_mut(),
            result: interop::CookResult::Error(format!("Error creating a new solver: {:?}", error)).into()
        },
    }
}

/// This is a helper function used to hide generics from C FFI.
fn free_solver_impl<F: FnMut() -> bool + Sync>(solver: *mut FemEngine) {
    if !solver.is_null() {
        let _ = Box::from_raw(solver as *mut softy::FemEngine<F>);
    }
}

/// What we create, we must destroy. Free memory allocated by the solver created by `new_solver`.
pub unsafe extern "C" fn free_solver(solver: *mut FemEngine) {
    free_solver_impl(solver);
}

/// Run a step of the simulation with the given solver.
#[no_mangle]
pub unsafe extern "C" fn step(
    solver: *mut FemEngine,
) -> cffi::CookResult {
    if solver.is_null() {
        return interop::CookResult::Error("No solver provided".to_string());
    }

    api::step(&mut *(solver as *mut softy::FemEngine)).into()
}
