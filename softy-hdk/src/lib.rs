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

use hdkrs::{PointCloud, PolyMesh, TetMesh, UnstructuredMesh};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

#[cxx::bridge(namespace = "softy")]
mod ffi {

    #[namespace = ""]
    extern "C++" {
        include!("hdkrs/src/lib.rs.h");
        type GU_Detail = hdkrs::ffi::GU_Detail;
    }

    #[derive(Debug)]
    pub enum SolverType {
        Ipopt,
        Newton,
        NewtonBacktracking,
        NewtonAssistedBacktracking,
        NewtonContactAssistedBacktracking,
        AdaptiveNewtonBacktracking,
        AdaptiveNewtonAssistedBacktracking,
        AdaptiveNewtonContactAssistedBacktracking,
        TrustRegion,
    }

    #[derive(Debug)]
    pub enum ObjectType {
        Solid,
        Shell,
        Rigid,
    }

    #[derive(Debug)]
    pub enum ElasticityModel {
        StableNeoHookean,
        NeoHookean,
    }

    #[derive(Debug)]
    pub enum ContactType {
        LinearizedPoint,
        Point,
    }

    #[derive(Debug)]
    pub enum Kernel {
        Compact,
        Approximate,
        Cubic,
        Global,
    }

    #[derive(Debug)]
    pub enum MuStrategy {
        Monotone,
        Adaptive,
    }

    #[derive(Debug)]
    pub enum TimeIntegration {
        BE,
        TR,
        BDF2,
        TRBDF2,
        TRBDF2U,
        SDIRK2,
    }

    #[derive(Debug)]
    pub enum Preconditioner {
        None,
        IncompleteJacobi,
        ApproximateJacobi,
    }

    #[derive(Debug)]
    pub enum FrictionProfile {
        Stabilized,
        Quadratic,
    }

    #[derive(Debug)]
    pub struct MaterialProperties {
        pub object_type: ObjectType,
        pub elasticity_model: ElasticityModel,
        pub density: f32,
        pub damping: f32,
        pub bending_stiffness: f32,
        pub bulk_modulus: f32,
        pub shear_modulus: f32,
    }

    #[derive(Debug)]
    pub struct FrictionalContactParams {
        pub object_material_id: u32,
        pub collider_material_ids: Vec<u32>,
        pub kernel: Kernel,
        pub contact_type: ContactType,
        pub radius_multiplier: f32,
        pub smoothness_tolerance: f32,
        pub contact_offset: f32,
        pub use_fixed: bool,
        pub smoothing_weight: f32,
        pub friction_forwarding: f32,
        pub dynamic_cof: f32,
        pub friction_profile: FrictionProfile,
        pub lagged_friction: bool,
        pub friction_tolerance: f32,
        pub friction_inner_iterations: u32,
    }

    #[derive(Debug)]
    pub struct SimParams {
        pub time_step: f32,
        pub gravity: f32,
        pub log_file: String,

        // Materials
        pub materials: Vec<MaterialProperties>,

        // Constraints
        pub volume_constraint: bool,
        pub friction_iterations: u32,
        pub frictional_contacts: Vec<FrictionalContactParams>,
        pub zone_pressurizations: Vec<f32>,
        pub compression_coefficients: Vec<f32>,
        pub hessian_approximation: Vec<u8>,

        // TODO: move these into FrictionalContactParams.
        pub friction_tolerance: f32, // epsilon
        pub contact_tolerance: f32,  // delta
        pub contact_iterations: u32,

        // Solver params
        pub solver_type: SolverType,
        pub backtracking_coeff: f32,
        pub velocity_clear_frequency: f32,
        pub tolerance: f32,
        pub max_iterations: u32,
        pub outer_tolerance: f32,
        pub residual_criterion: bool,
        pub residual_tolerance: f32,
        pub velocity_criterion: bool,
        pub velocity_tolerance: f32,
        pub acceleration_criterion: bool,
        pub acceleration_tolerance: f32,
        pub max_outer_iterations: u32,
        pub time_integration: TimeIntegration,
        pub preconditioner: Preconditioner,
        pub project_element_hessians: bool,

        // Ipopt
        pub mu_strategy: MuStrategy,
        pub max_gradient_scaling: f32,
        pub print_level: u32,
        pub derivative_test: u32,
    }

    /// Result reported from `register_new_solver` function.
    /// In case of failure, solver_id is set to a negative number.
    #[derive(Debug)]
    pub struct RegistryResult {
        solver_id: i64,
        cook_result: CookResult,
    }

    pub struct SolverResult {
        /// ID of the solver in the registry.
        id: i64,

        /// A reference to a Solver struct from the registry.
        solver: Box<SoftySolver>,

        /// A Cook result indicating the overall status of the result (success/failure) along with a
        /// descriptive message reported back to the caller on the C side.
        cook_result: CookResult,
    }

    pub struct StepResult {
        mesh: Box<Mesh>,
        cook_result: CookResult,
    }

    pub struct SolveResult {
        solver_id: i64,
        mesh: Box<Mesh>,
        cook_result: CookResult,
    }
    extern "Rust" {
        type Points;
        fn set(self: Pin<&mut Points>, detail: &GU_Detail);
        fn new_point_cloud() -> Box<Points>;
    }
    extern "Rust" {
        type Mesh;
        fn set(self: Pin<&mut Mesh>, detail: &GU_Detail);
        fn new_mesh() -> Box<Mesh>;
    }
    extern "Rust" {
        type MeshPoints;
        fn set_tetmesh_points(self: Pin<&mut MeshPoints>, input: &GU_Detail);
        fn set_polymesh_points(self: Pin<&mut MeshPoints>, input: &GU_Detail);
        fn new_mesh_points() -> Box<MeshPoints>;
    }
    extern "Rust" {
        type Meshes;
        fn set_tetmesh(self: Pin<&mut Meshes>, detail: &GU_Detail);
        fn set_polymesh(self: Pin<&mut Meshes>, detail: &GU_Detail);
        fn new_meshes() -> Box<Meshes>;
    }

    extern "Rust" {
        type SoftySolver;
        fn init_env_logger();
        fn register_new_solver(mesh: Box<Mesh>, sim_params: SimParams) -> RegistryResult;
        unsafe fn step<'a>(
            solver: Box<SoftySolver>,
            points: Box<Points>,
            interrupt_checker: UniquePtr<InterruptChecker>,
        ) -> StepResult;
        fn get_solver(solver_id: i64, mesh: Box<Mesh>, sim_params: SimParams) -> SolverResult;
        fn clear_solver_registry();

        fn add_mesh(detail: Pin<&mut GU_Detail>, mesh: Box<Mesh>);
    }

    #[namespace = "hdkrs"]
    extern "C++" {
        type InterruptChecker = hdkrs::ffi::InterruptChecker;
        type CookResult = hdkrs::ffi::CookResult;
    }
}

use ffi::*;

pub struct Mesh {
    mesh: Option<UnstructuredMesh>,
}

fn new_mesh() -> Box<Mesh> {
    Box::new(Mesh { mesh: None })
}

impl Mesh {
    fn set(mut self: Pin<&mut Mesh>, detail: &GU_Detail) {
        self.mesh = hdkrs::ffi::build_unstructured_mesh(detail).ok().map(|m| *m);
    }
}

pub struct Meshes {
    tetmesh: Option<TetMesh>,
    polymesh: Option<PolyMesh>,
}

fn new_meshes() -> Box<Meshes> {
    Box::new(Meshes {
        tetmesh: None,
        polymesh: None,
    })
}

pub struct Points {
    points: Option<PointCloud>,
}

fn new_point_cloud() -> Box<Points> {
    Box::new(Points { points: None })
}

impl Points {
    fn set(mut self: Pin<&mut Points>, detail: &GU_Detail) {
        self.points = hdkrs::ffi::build_pointcloud(detail).ok().map(|m| *m);
    }
}

pub struct MeshPoints {
    tetmesh_points: Option<PointCloud>,
    polymesh_points: Option<PointCloud>,
}

fn new_mesh_points() -> Box<MeshPoints> {
    Box::new(MeshPoints {
        tetmesh_points: None,
        polymesh_points: None,
    })
}

/// This function initializes env_logger. It will panic if called more than once.
pub fn init_env_logger() {
    env_logger::Builder::from_env("SOFTY_LOG")
        .format_timestamp(None)
        .init();
}

/// Register a new solver in the registry.
pub fn register_new_solver(mesh: Box<Mesh>, sim_params: SimParams) -> RegistryResult {
    match api::register_new_solver(mesh.mesh.map(|m| m.0), sim_params) {
        Ok((id, _)) => RegistryResult {
            solver_id: i64::from(id),
            cook_result: hdkrs::interop::CookResult::Success(String::new()).into(),
        },
        Err(err) => RegistryResult {
            solver_id: -1,
            cook_result: hdkrs::interop::CookResult::Error(format!("{}", err)).into(),
        },
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

fn add_mesh(mut detail: Pin<&mut GU_Detail>, mesh: Box<Mesh>) {
    if let Some(mut mesh) = mesh.mesh {
        mesh.0
            .reverse_if(|_, cell_type| matches!(cell_type, geo::mesh::CellType::Triangle));
        hdkrs::ffi::add_unstructured_mesh(detail.as_mut(), &mesh);
    }
}

impl Meshes {
    fn set_tetmesh(mut self: Pin<&mut Meshes>, detail: &GU_Detail) {
        self.tetmesh = hdkrs::ffi::build_tetmesh(detail).ok().map(|m| *m);
    }
    fn set_polymesh(mut self: Pin<&mut Meshes>, detail: &GU_Detail) {
        self.polymesh = hdkrs::ffi::build_polymesh(detail).ok().map(|m| *m);
    }
}

impl MeshPoints {
    fn set_tetmesh_points(mut self: Pin<&mut MeshPoints>, detail: &GU_Detail) {
        self.tetmesh_points = hdkrs::ffi::build_pointcloud(detail).ok().map(|m| *m);
    }
    fn set_polymesh_points(mut self: Pin<&mut MeshPoints>, detail: &GU_Detail) {
        self.polymesh_points = hdkrs::ffi::build_pointcloud(detail).ok().map(|m| *m);
    }
}

/// Opaque struct to represent `Arc<Mutex<api::Solver>>` on the C side.
pub enum SoftySolver {
    Some(Arc<Mutex<dyn api::Solver>>),
    None,
}

impl Into<Option<Arc<Mutex<dyn api::Solver>>>> for SoftySolver {
    fn into(self) -> Option<Arc<Mutex<dyn api::Solver>>> {
        match self {
            SoftySolver::Some(s) => Some(s),
            SoftySolver::None => None,
        }
    }
}

/// Register a new solver in the registry. (C side)
/// This function consumes `tetmesh` and `polymesh`.
pub fn get_solver(solver_id: i64, mesh: Box<Mesh>, sim_params: SimParams) -> SolverResult {
    match api::get_solver(validate_id(solver_id), mesh.mesh.map(|m| m.0), sim_params) {
        Ok((id, solver)) => {
            assert!(Arc::strong_count(&solver) != 1);
            SolverResult {
                id: i64::from(id),
                solver: Box::new(SoftySolver::Some(solver)),
                cook_result: hdkrs::interop::CookResult::Success(String::new()).into(),
            }
        }
        Err(err) => SolverResult {
            id: -1,
            solver: Box::new(SoftySolver::None),
            cook_result: hdkrs::interop::CookResult::Error(format!("{}", err)).into(),
        },
    }
}

/// Delete all solvers in the registry.
pub fn clear_solver_registry() {
    api::clear_solver_registry()
}

/// Perform one step of the solve given a solver.
pub fn step<'a>(
    solver: Box<SoftySolver>,
    points: Box<Points>,
    mut interrupt_checker: cxx::UniquePtr<InterruptChecker>,
) -> StepResult {
    let (mesh_mb, cook_result) = if let SoftySolver::Some(solver) = *solver {
        match solver.try_lock() {
            Ok(mut solver) => api::step(&mut *solver, points.points.map(|m| m.0), move || {
                interrupt_checker.pin_mut().check_interrupt()
            }),
            Err(err) => (
                None,
                hdkrs::interop::CookResult::Error(format!("Global solver lock: {}", err)),
            ),
        }
    } else {
        (
            None,
            hdkrs::interop::CookResult::Error("Invalid solver".to_string()),
        )
    };

    if let Some(solver_mesh) = mesh_mb {
        StepResult {
            mesh: Box::new(Mesh {
                mesh: Some(solver_mesh.into()),
            }),
            cook_result: cook_result.into(),
        }
    } else {
        StepResult {
            mesh: new_mesh(),
            cook_result: cook_result.into(),
        }
    }
}

/// Gets a valid solver and performs one step of the solve.
pub fn solve<'a>(
    solver_id: i64,
    mesh: Box<Mesh>,
    sim_params: SimParams,
    mut interrupt_checker: cxx::UniquePtr<InterruptChecker>,
) -> SolveResult {
    let (data, cook_result) = api::cook(
        validate_id(solver_id),
        mesh.mesh.map(|m| m.0),
        sim_params,
        move || interrupt_checker.pin_mut().check_interrupt(),
    );
    if let Some((new_solver_id, solver_mesh)) = data {
        SolveResult {
            solver_id: i64::from(new_solver_id),
            mesh: Box::new(Mesh {
                mesh: Some(solver_mesh.into()),
            }),
            cook_result: cook_result.into(),
        }
    } else {
        SolveResult {
            solver_id: -1,
            mesh: new_mesh(),
            cook_result: cook_result.into(),
        }
    }
}
