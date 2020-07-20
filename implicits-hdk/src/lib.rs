mod api;

use hdkrs::{cffi, interop};

pub use cimplicits::iso_default_params;
pub use cimplicits::ISO_Params;

/// The particular iso-surface related action to be taken.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum HISO_Action {
    /// Compute the implicit field at a given set of query points.
    ComputePotential,
    /// Project the given set of points to one side of the iso-surface.
    Project,
}

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HISO_Params {
    pub action: HISO_Action,
    pub iso_value: f32,      // Only used for projection
    pub project_below: bool, // Only used for projection
    pub debug: bool,         // Only used for potential computation
    pub iso_params: ISO_Params,
}

impl Default for HISO_Params {
    fn default() -> Self {
        HISO_Params {
            action: HISO_Action::ComputePotential,
            iso_value: 0.0,
            project_below: false,
            debug: false,
            iso_params: ISO_Params::default(),
        }
    }
}

/// Construct a default set of input parameters.
///
/// It is safer to use this function than populating the fields manually since it guarantees that
/// every field is properly initialized.
#[no_mangle]
pub unsafe extern "C" fn hiso_default_params() -> HISO_Params {
    Default::default()
}

impl Into<implicits::Params> for HISO_Params {
    fn into(self) -> implicits::Params {
        self.iso_params.into()
    }
}

/// Main entry point from Houdini SOP.
///
/// The purpose of this function is to cleanup the inputs for use in Rust code.
#[no_mangle]
pub unsafe extern "C" fn hiso_cook(
    samplemesh: *mut cffi::HR_PolyMesh,
    polymesh: *mut cffi::HR_PolyMesh,
    params: HISO_Params,
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
