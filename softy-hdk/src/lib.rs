#[macro_use]
extern crate lazy_static;

mod api;

use hdkrs::{cffi, interop};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_SoftyMaterialProperties {
    pub bulk_modulus: f32,
    pub shear_modulus: f32,
    pub density: f32,
    pub damping: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_SoftySimParams {
    pub material: EL_SoftyMaterialProperties,
    pub time_step: f32,
    pub gravity: f32,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub outer_tolerance: f32,
    pub max_outer_iterations: u32,
    pub volume_constraint: bool,
    pub contact_kernel: i32,
    pub contact_type: i32,
    pub contact_radius_multiplier: f32,
    pub smoothness_tolerance: f32,
    pub dynamic_friction: f32,
    pub friction_iterations: u32,
    pub print_level: u32,
    pub derivative_test: u32,
    pub mu_strategy: i32,
    pub max_gradient_scaling: f32,
    pub log_file: *const std::os::raw::c_char,
}

/// Result reported from `register_new_solver` function.
/// In case of failure, solver_id is set to a negative number.
#[repr(C)]
pub struct EL_SoftyRegistryResult {
    solver_id: i64,
    cook_result: cffi::HR_CookResult,
}

/// Register a new solver in the registry. (C side)
/// This function consumes `tetmesh` and `polymesh`.
#[no_mangle]
pub unsafe extern "C" fn el_softy_register_new_solver(
    tetmesh: *mut cffi::HR_TetMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    sim_params: EL_SoftySimParams,
) -> EL_SoftyRegistryResult {
    if let Some(tetmesh) = interop::into_box(tetmesh) {
        // Get an optional shell object (cloth or animated static object).
        let shell = interop::into_box(polymesh);

        match api::register_new_solver(*tetmesh, shell, sim_params) {
            Ok((id, _)) => EL_SoftyRegistryResult {
                solver_id: i64::from(id),
                cook_result: hdkrs::interop::CookResult::Success(String::new()).into(),
            },
            Err(err) => EL_SoftyRegistryResult {
                solver_id: -1,
                cook_result: hdkrs::interop::CookResult::Error(format!("Error: {:?}", err)).into(),
            },
        }
    } else {
        EL_SoftyRegistryResult {
            solver_id: -1,
            cook_result: hdkrs::interop::CookResult::Error("Given TetMesh is null.".to_owned())
                .into(),
        }
    }
}

/// Validate the given id from C side by converting it to a Rust option type.
fn validate_id(id: i64) -> Option<u32> {
    if id == i64::from(id as u32) {
        Some(id as u32)
    } else {
        None
    }
}

/// Validate the given solver pointer from C side by converting it to an `Arc` if it still exists
/// in the registry.
unsafe fn validate_solver_ptr(solver: *mut EL_SoftySolverPtr) -> Option<EL_SoftySolverPtr> {
    interop::into_box(solver).map(|x| *x)
}

/// Move the `Arc` to the heap and return its raw pointer.
fn solver_ptr(solver: EL_SoftySolverPtr) -> *mut EL_SoftySolverPtr {
    Box::into_raw(Box::new(solver)) as *mut EL_SoftySolverPtr
}

/// Opaque struct to represent `Arc<Mutex<api::Solver>>` on the C side.
#[allow(non_camel_case_types)]
pub type EL_SoftySolverPtr = Arc<Mutex<dyn api::Solver>>;

#[repr(C)]
pub struct EL_SoftySolverResult {
    /// ID of the solver in the registry.
    id: i64,

    /// A non-owning pointer to a Solver struct from the registry.
    solver: *mut EL_SoftySolverPtr,

    /// A Cook result indicating the overall status of the result (success/failure) along with a
    /// descriptive message reported back to the caller on the C side.
    cook_result: cffi::HR_CookResult,
}

/// Register a new solver in the registry. (C side)
/// This function consumes `tetmesh` and `polymesh`.
#[no_mangle]
pub unsafe extern "C" fn el_softy_get_solver(
    solver_id: i64,
    tetmesh: *mut cffi::HR_TetMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    sim_params: EL_SoftySimParams,
) -> EL_SoftySolverResult {
    match api::get_solver(
        validate_id(solver_id),
        interop::into_box(tetmesh),
        interop::into_box(polymesh),
        sim_params,
    ) {
        Ok((id, solver)) => {
            assert!(Arc::strong_count(&solver) != 1);
            EL_SoftySolverResult {
                id: i64::from(id),
                solver: solver_ptr(solver),
                cook_result: hdkrs::interop::CookResult::Success(String::new()).into(),
            }
        }
        Err(err) => EL_SoftySolverResult {
            id: -1,
            solver: ::std::ptr::null_mut(),
            cook_result: hdkrs::interop::CookResult::Error(format!("Error: {:?}", err)).into(),
        },
    }
}

/// Delete all solvers in the registry.
#[no_mangle]
pub unsafe extern "C" fn el_softy_clear_solver_registry() {
    api::clear_solver_registry()
}

#[repr(C)]
pub struct EL_SoftyStepResult {
    tetmesh: *mut cffi::HR_TetMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    cook_result: cffi::HR_CookResult,
}

/// Perform one step of the solve given a solver.
#[no_mangle]
pub unsafe extern "C" fn el_softy_step(
    solver: *mut EL_SoftySolverPtr,
    tetmesh_points: *mut cffi::HR_PointCloud,
    polymesh_points: *mut cffi::HR_PointCloud,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> EL_SoftyStepResult {
    if let Some(solver) = validate_solver_ptr(solver) {
        let (tetmesh_mb, polymesh_mb, cook_result) = api::step(
            &mut *solver.lock().unwrap(),
            interop::into_box(tetmesh_points),
            interop::into_box(polymesh_points),
            interop::interrupt_callback(interrupt_checker, check_interrupt),
        );
        if let Some(solver_tetmesh) = tetmesh_mb {
            let tetmesh = Box::into_raw(Box::new(solver_tetmesh.into()));
            if let Some(solver_polymesh) = polymesh_mb {
                EL_SoftyStepResult {
                    tetmesh,
                    polymesh: Box::into_raw(Box::new(solver_polymesh.into())),
                    cook_result: cook_result.into(),
                }
            } else {
                EL_SoftyStepResult {
                    tetmesh: tetmesh.into(),
                    polymesh: ::std::ptr::null_mut(),
                    cook_result: cook_result.into(),
                }
            }
        } else {
            EL_SoftyStepResult {
                tetmesh: ::std::ptr::null_mut(),
                polymesh: ::std::ptr::null_mut(),
                cook_result: cook_result.into(),
            }
        }
    } else {
        EL_SoftyStepResult {
            tetmesh: ::std::ptr::null_mut(),
            polymesh: ::std::ptr::null_mut(),
            cook_result: hdkrs::interop::CookResult::Error("Invalid solver".to_string()).into(),
        }
    }
}

#[repr(C)]
pub struct EL_SoftySolveResult {
    solver_id: i64,
    tetmesh: *mut cffi::HR_TetMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    cook_result: cffi::HR_CookResult,
}

/// Gets a valid solver and performs one step of the solve.
#[no_mangle]
pub unsafe extern "C" fn el_softy_solve(
    solver_id: i64,
    tetmesh: *mut cffi::HR_TetMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    sim_params: EL_SoftySimParams,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> EL_SoftySolveResult {
    let (data, cook_result) = api::cook(
        validate_id(solver_id),
        interop::into_box(tetmesh),
        interop::into_box(polymesh),
        sim_params,
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    );
    if let Some((new_solver_id, solver_tetmesh)) = data {
        EL_SoftySolveResult {
            solver_id: i64::from(new_solver_id),
            tetmesh: Box::into_raw(Box::new(solver_tetmesh.into())),
            polymesh: ::std::ptr::null_mut(),
            cook_result: cook_result.into(),
        }
    } else {
        EL_SoftySolveResult {
            solver_id: -1,
            tetmesh: ::std::ptr::null_mut(),
            polymesh: ::std::ptr::null_mut(),
            cook_result: cook_result.into(),
        }
    }
}
