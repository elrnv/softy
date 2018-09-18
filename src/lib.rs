extern crate geometry as geo;
extern crate hdkrs;
extern crate softy;

pub use hdkrs::{cffi, interop};

mod api;

/// Main entry point from Houdini SOP.
#[no_mangle]
pub unsafe extern "C" fn cook(
    tetmesh: *mut cffi::TetMesh,
    polymesh: *mut cffi::PolyMesh,
    sim_params: api::SimParams,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> cffi::CookResult {
    api::cook(
        interop::mesh(tetmesh),
        interop::mesh(polymesh),
        sim_params.into(),
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    ).into()
}
