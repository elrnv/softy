extern crate implicits;
extern crate geometry as geo;
extern crate hdkrs;

mod api;

pub use hdkrs::{cffi, interop};

/// Main entry point from Houdini SOP.
/// The purpose of this function is to cleanup the inputs for use in Rust code.
#[no_mangle]
pub unsafe extern "C" fn cook(
    samplemesh: *mut cffi::PolyMesh,
    polymesh: *mut cffi::PolyMesh,
    params: api::Params,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> cffi::CookResult {
    api::cook(
        interop::mesh(samplemesh),
        interop::mesh(polymesh),
        params.into(),
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    ).into()
}
