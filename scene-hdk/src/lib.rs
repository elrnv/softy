mod api;

use hdkrs::{PointCloud, UnstructuredMesh};
use softy::scene::SceneConfig;
use std::pin::Pin;

#[cxx::bridge(namespace = "softy")]
mod ffi {

    #[namespace = ""]
    extern "C++" {
        include!("hdkrs/src/lib.rs.h");
        type GU_Detail = hdkrs::ffi::GU_Detail;
    }

    #[derive(Debug)]
    pub enum SolverType {
        Newton,
        NewtonBacktracking,
        NewtonAssistedBacktracking,
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
    pub enum Kernel {
        Interpolating,
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
        pub radius_multiplier: f32,
        pub smoothness_tolerance: f32,
        pub contact_offset: f32,
        pub use_fixed: bool,
        // pub smoothing_weight: f32,
        // pub friction_forwarding: f32,
        pub dynamic_cof: f32,
        // pub friction_tolerance: f32,
        // pub friction_inner_iterations: u32,
    }

    #[derive(Debug)]
    pub struct SimParams {
        pub time_step: f32,
        pub gravity: f32,
        pub scene_file: String,

        // Materials
        pub materials: Vec<MaterialProperties>,

        // Constraints
        pub volume_constraint: bool,
        pub frictional_contacts: Vec<FrictionalContactParams>,

        // TODO: move these into FrictionalContactParams.
        pub friction_tolerance: f32, // epsilon
        pub contact_tolerance: f32,  // delta
        pub contact_iterations: u32,

        // Solver params
        pub solver_type: SolverType,
        pub velocity_clear_frequency: f32,
        pub tolerance: f32,
        pub max_iterations: u32,
        pub residual_criterion: bool,
        pub residual_tolerance: f32,
        pub velocity_criterion: bool,
        pub velocity_tolerance: f32,
        pub acceleration_criterion: bool,
        pub acceleration_tolerance: f32,
        pub max_outer_iterations: u32,
        pub time_integration: TimeIntegration,

        pub derivative_test: u32,
    }

    /// Result reported from `new_scene` function.
    #[derive(Debug)]
    pub struct SceneResult {
        scene: Box<SoftyScene>,
        cook_result: CookResult,
    }

    #[derive(Debug)]
    pub struct SaveResult {
        cook_result: CookResult,
    }

    #[derive(Debug)]
    pub struct AddKeyframeResult {
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
        type SoftyScene;
        fn init_env_logger();
        fn new_scene(mesh: Box<Mesh>, sim_params: SimParams) -> SceneResult;
        unsafe fn add_keyframe<'a>(
            self: Pin<&mut SoftyScene>,
            frame: u64,
            points: Box<Points>,
        ) -> AddKeyframeResult;
        fn save(self: Pin<&SoftyScene>, path: &str) -> SaveResult;
    }

    #[namespace = "hdkrs"]
    extern "C++" {
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

/// This function initializes env_logger. It will panic if called more than once.
pub fn init_env_logger() {
    env_logger::Builder::from_env("SOFTY_LOG")
        .format_timestamp(None)
        .init();
}

/// Create a new scene.
pub fn new_scene(mesh: Box<Mesh>, sim_params: SimParams) -> SceneResult {
    match api::new_scene(mesh.mesh.map(|m| m.0), sim_params) {
        Ok(scene) => SceneResult {
            scene: Box::new(SoftyScene { scene: Some(scene) }),
            cook_result: hdkrs::interop::CookResult::Success(String::new()).into(),
        },
        Err(err) => SceneResult {
            scene: Box::new(SoftyScene { scene: None }),
            cook_result: hdkrs::interop::CookResult::Error(format!("{}", err)).into(),
        },
    }
}

/// Opaque struct to represent a scene on the C side.
#[derive(Debug)]
pub struct SoftyScene {
    scene: Option<SceneConfig>,
}

impl Into<Option<SceneConfig>> for SoftyScene {
    fn into(self) -> Option<SceneConfig> {
        self.scene
    }
}


/// Gets a valid solver and performs one step of the solve.
impl SoftyScene {
    fn add_keyframe<'a>(
        mut self: Pin<&mut SoftyScene>,
        frame: u64,
        points: Box<Points>,
    ) -> AddKeyframeResult {
        let cook_result = if let Some(scene) = self.scene.as_mut() {
            if let Some(points) = points.points {
                api::add_keyframe(scene, frame, points.0)
            } else {
                hdkrs::interop::CookResult::Error("Missing points".to_string())
            }
        } else {
            hdkrs::interop::CookResult::Error("Missing scene".to_string())
        }.into();
        AddKeyframeResult {
            cook_result
        }
    }

    /// Perform one step of the solve given a solver.
    pub fn save(
        self: Pin<&SoftyScene>,
        path: &str,
    ) -> SaveResult {
        let cook_result = if let Some(scene) = self.scene.as_ref() {
            api::save(scene, path)
        } else {
            hdkrs::interop::CookResult::Error("Missing scene".to_string())
        }.into();
        SaveResult {
            cook_result
        }
    }
}
