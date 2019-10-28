/*!
 * # `softy-hdk`
 *
 * ## Material Properties
 *
 * Materials can be defined in a two main ways: via global material settings defined in the
 * `Material` tab of the `Softy SOP` or via variable material properties defined as primitive
 * attributes on the input meshes themselves. In either case the solver expects the input meshes to
 * have a corresponding material id starting with `1`. Material id `0` is reserved for a fixed
 * static object like the ground plane.
 *
 * If an input mesh contains a variable material property attribute, then its corresponding
 * property defined in the `Material` tab (if any) is ignored. This applies on a per material
 * property basis, which means that if `mu` is variable, `lambda` can be taken from the `Material`
 * tab.
 *
 * Currently there are 3 material properties required for simulation all expected to be in SI
 * units:
 *
 *  - `lambda` --- the first Lame paramter, which is similar to the bulk modulus.
 *  - `mu` --- the second Lame parameter, which is also called shear modulus.
 *  - `density` --- the density of the material.
 *
 * If both variable and global material properties are absent, an error is thrown.
 *
 */

// Future Plans:
// ### Friction and Contact
//
// Variable friction parameters identified by "cof\_#" float attributes where the # indicates the material
// id of the contacting material. These will be vertex attributes because they are more widely
// supported than tet face attributes (since there are surface only paramters).
// If such an attribute exists, then obviously a contact constraint is introduced between the current
// material and the referenced material.
//
// If no friction attributes exist, then contact is frictionless.

#[macro_use]
extern crate lazy_static;

mod api;

use hdkrs::{cffi, interop};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EL_SoftyObjectType {
    Solid,
    Shell,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EL_SoftyContactType {
    LinearizedPoint,
    Point,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EL_SoftyKernel {
    Interpolating,
    Approximate,
    Cubic,
    Global,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EL_SoftyMuStrategy {
    Monotone,
    Adaptive,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EL_SoftyMaterialProperties {
    pub object_type: EL_SoftyObjectType,
    pub density: f32,
    pub damping: f32,
    pub bulk_modulus: f32,
    pub shear_modulus: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EL_SoftyFrictionalContactParams {
    pub object_material_id: u32,
    pub collider_material_ids: EL_SoftyColliderMaterialIds,
    pub kernel: EL_SoftyKernel,
    pub contact_type: EL_SoftyContactType,
    pub radius_multiplier: f32,
    pub smoothness_tolerance: f32,
    pub dynamic_cof: f32,
    pub friction_tolerance: f32,
    pub friction_inner_iterations: u32,
}

/// A Helper trait to access C arrays.
pub(crate) trait AsSlice {
    type T;
    fn ptr(&self) -> *const Self::T;
    fn size(&self) -> usize;
    fn as_slice(&self) -> &[Self::T] {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.size()) }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EL_SoftyColliderMaterialIds {
    pub ptr: *const u32,
    pub size: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_SoftyMaterials {
    pub ptr: *const EL_SoftyMaterialProperties,
    pub size: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_SoftyFrictionalContacts {
    pub ptr: *const EL_SoftyFrictionalContactParams,
    pub size: usize,
}

impl AsSlice for EL_SoftyColliderMaterialIds {
    type T = u32;
    fn ptr(&self) -> *const Self::T {
        self.ptr
    }
    fn size(&self) -> usize {
        self.size
    }
}

impl AsSlice for EL_SoftyMaterials {
    type T = EL_SoftyMaterialProperties;
    fn ptr(&self) -> *const Self::T {
        self.ptr
    }
    fn size(&self) -> usize {
        self.size
    }
}

impl AsSlice for EL_SoftyFrictionalContacts {
    type T = EL_SoftyFrictionalContactParams;
    fn ptr(&self) -> *const Self::T {
        self.ptr
    }
    fn size(&self) -> usize {
        self.size
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_SoftySimParams {
    pub time_step: f32,
    pub gravity: f32,
    pub log_file: *const std::os::raw::c_char,

    // Materials
    pub materials: EL_SoftyMaterials,

    // Constraints
    pub volume_constraint: bool,
    pub friction_iterations: u32,
    pub frictional_contacts: EL_SoftyFrictionalContacts,

    // Optimization
    pub clear_velocity: bool,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub outer_tolerance: f32,
    pub max_outer_iterations: u32,

    // Ipopt
    pub mu_strategy: EL_SoftyMuStrategy,
    pub max_gradient_scaling: f32,
    pub print_level: u32,
    pub derivative_test: u32,
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
                    polymesh: Box::into_raw(Box::new(solver_polymesh.reversed().into())),
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
