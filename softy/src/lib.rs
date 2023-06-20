#![cfg_attr(feature = "unstable", feature(test))]
#![type_length_limit = "10000000"]

//#[global_allocator]
//static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

macro_rules! add_time {
    ($t:expr; $expr:expr) => {{
        let _t_begin = std::time::Instant::now();
        let result = $expr;
        $t += std::time::Instant::now() - _t_begin;
        result
    }};
}

mod attrib_defines;
mod constraint;
pub mod constraints;
mod contact;
mod energy;
pub mod energy_models;
pub mod fem;
mod friction;
pub mod io;
pub mod mask_iter;
mod matrix;
pub mod objects;
pub mod scene;

// TODO: This should be feature gated. Unfortunately this makes it tedious to
// run tests without passing the feature explicitly via the `--features` flag.
// Doing this automatically in Cargo.toml is blocked on this issue:
// https://github.com/rust-lang/cargo/issues/2911, for which there is a
// potential solution drafted in https://github.com/rust-lang/rfcs/pull/1956,
// which is also blocked at the time of this writing.
pub mod test_utils;

pub type PointCloud = geo::mesh::PointCloud<f64>;
pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;
pub type TriMesh = geo::mesh::TriMesh<f64>;
pub type Mesh = geo::mesh::Mesh<f64>;

pub type TetMeshExt = geo::mesh::TetMeshExt<f64>;
pub type TriMeshExt = geo::mesh::TriMeshExt<f64>;

pub use self::constraints::FrictionProfile;
pub use self::contact::*;
pub use self::fem::nl as nl_fem;
pub use self::friction::*;
pub use self::objects::init_mesh_source_index_attribute;
pub use self::objects::material::*;
use geo::attrib;
pub use io::*;
pub use utils::index::{CheckedIndex, Index};

pub use attrib_defines::*;

pub use implicits::KernelType;

use crate::scene::SceneError;
use thiserror::Error;

pub trait Real: tensr::Real + na::RealField + num_traits::FloatConst {}
impl<T> Real for T where T: tensr::Real + na::RealField + num_traits::FloatConst {}

/// An extension to the real trait that allows ops with f64 floats.
pub trait Real64: Real + tensr::Real64 {}
impl<T> Real64 for T where T: Real + tensr::Real64 {}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Size mismatch error")]
    SizeMismatch,
    #[error("Attribute error: {source}")]
    AttribError {
        #[from]
        source: attrib::Error,
    },
    #[error("Degenerate reference element detected: {:?}", .degens[0])]
    DegenerateReferenceElement {
        // Degens from Upcountry.
        // Look, keep it at the end of the laneway. No degens on the property.
        degens: Vec<usize>,
    },
    #[error("Inverted reference element detected: {:?}", .inverted[0])]
    InvertedReferenceElement { inverted: Vec<usize> },
    #[error("Error during main non-linear solve step: {result:?}")]
    NLSolveError { result: nl_fem::SolveResult },
    #[error("Invalid parameter: {name}")]
    InvalidParameter { name: String },
    #[error("Missing source index")]
    MissingSourceIndex,
    #[error("Missing elasticity parameters")]
    MissingElasticityParams,
    #[error("Missing density")]
    MissingDensity,
    #[error("Missing contact parameters")]
    MissingContactParams,
    #[error("Missing contact constraint")]
    MissingContactConstraint,
    #[error("No finite mass data found. Check that not all vertices are fixed.")]
    MissingMassData,
    #[error("No simulation mesh found")]
    NoSimulationMesh,
    #[error("No kinematic mesh found")]
    NoKinematicMesh,
    /// This may be an internal error.
    #[error("Incorrect object is used for the given material")]
    ObjectMaterialMismatch,
    /// Typically happens during debugging
    #[error("Error during mesh IO: {source}")]
    MeshIOError {
        #[from]
        source: geo::io::Error,
    },
    #[error("File I/O Error: {source}")]
    FileIOError {
        #[from]
        source: std::io::Error,
    },
    #[error("File I/O Error")]
    InvalidImplicitSurface,
    #[error("Error generating the implicit field: {source}")]
    ImplicitsError {
        #[from]
        source: implicits::Error,
    },
    #[error("Unimplemented feature: {description:?}")]
    UnimplementedFeature { description: String },
    #[error("Invalid solver configuration: {source}")]
    InvalidSolverConfig {
        #[from]
        source: fem::nl::Error,
    },
    #[error("Specified ID ({id}) for a contact constraint does not match any object ID in the input mesh")]
    ContactObjectIdError { id: usize },
    #[error("Derivative check failed")]
    DerivativeCheckFailure,
    #[error("Nothing to solve: no mesh or all vertices are fixed")]
    NothingToSolve,
    #[error("Failed to load configuration")]
    LoadConfig(#[from] LoadConfigError),
    #[error("Detected an orphaned vertex: {:?}", .orphaned[0])]
    OrphanedVertices { orphaned: Vec<usize> },
    #[error("Scene error: {:?}", .0)]
    SceneError(#[from] SceneError),
}

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<Error> for SimResult {
    fn from(err: Error) -> SimResult {
        match err {
            Error::SizeMismatch => SimResult::Error(format!("{}", err)),
            Error::AttribError { source } => SimResult::Error(format!("{}", source)),
            Error::MissingSourceIndex => {
                SimResult::Error("Missing source index vertex attribute".to_string())
            }
            Error::MissingElasticityParams => SimResult::Error(
                "Missing elasticity parameters or per-element elasticity attributes".to_string(),
            ),
            Error::MissingContactParams => {
                SimResult::Error("Missing smooth contact parameters".to_string())
            }
            Error::MissingContactConstraint => {
                SimResult::Error("Missing smooth contact constraint".to_string())
            }
            Error::NoSimulationMesh => SimResult::Error("Missing simulation mesh".to_string()),
            Error::NoKinematicMesh => SimResult::Error("Missing kinematic mesh".to_string()),
            Error::InvalidParameter { name } => {
                SimResult::Error(format!("Invalid parameter: {:?}", name))
            }
            Error::MeshIOError { source } => {
                SimResult::Error(format!("Error during mesh I/O: {:?}", source))
            }
            Error::FileIOError { source } => {
                SimResult::Error(format!("File I/O error: {:?}", source.kind()))
            }
            Error::InvalidImplicitSurface => {
                SimResult::Error("Error creating an implicit surface".to_string())
            }
            Error::ImplicitsError { source } => {
                SimResult::Error(format!("Error computing implicit surface: {:?}", source))
            }
            Error::UnimplementedFeature { description } => {
                SimResult::Error(format!("Unimplemented feature: {:?}", description))
            }
            _ => SimResult::Error(err.to_string()),
        }
    }
}

impl From<Result<nl_fem::StepResult, Error>> for SimResult {
    fn from(res: Result<nl_fem::StepResult, Error>) -> SimResult {
        match res {
            Ok(solve_result) => SimResult::Success(format!("{}", solve_result)),
            Err(err) => err.into(),
        }
    }
}

pub(crate) fn inf_norm<I, T: tensr::Real>(iter: I) -> T
where
    I: IntoIterator<Item = T>,
{
    iter.into_iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .unwrap_or_else(T::zero)
}
