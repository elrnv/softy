mod api;

use hdkrs::{cffi, interop};

/// Main entry point from Houdini SOP.
/// The purpose of this function is to cleanup the inputs for use in Rust code.
#[no_mangle]
pub unsafe extern "C" fn el_iso_cook(
    samplemesh: *mut cffi::HR_PolyMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    params: api::Params,
    interrupt_checker: *mut cffi::c_void,
    check_interrupt: Option<extern "C" fn(*const cffi::c_void) -> bool>,
) -> cffi::HR_CookResult {
    api::cook(
        interop::as_mut(samplemesh),
        interop::as_mut(polymesh),
        params.into(),
        interop::interrupt_callback(interrupt_checker, check_interrupt),
    )
    .into()
}
