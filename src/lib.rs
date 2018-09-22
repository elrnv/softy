extern crate geometry as geo;
extern crate hdkrs;
extern crate softy;

#[macro_use]
extern crate lazy_static;

mod api;

pub use hdkrs::{cffi, interop};

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
}

#[repr(C)]
pub struct SolveResult {
    solver_id: i64,
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    cook_result: cffi::CookResult,
}

// Main entry point from the HDK.
#[no_mangle]
pub unsafe extern "C" fn cook(
    solver_id: i64,
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    sim_params: SimParams,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> SolveResult {
    let (data, cook_result) = api::cook(
        if solver_id < 0 { None } else { Some(solver_id as u32) },
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

