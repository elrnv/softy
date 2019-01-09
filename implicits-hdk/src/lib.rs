extern crate geometry as geo;
extern crate hdkrs;
extern crate implicits;

mod api;

pub use hdkrs::{cffi, interop};
pub use std::ptr::NonNull;

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
        interop::as_mut(samplemesh),
        interop::as_mut(polymesh),
        params.into(),
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    )
    .into()
}
