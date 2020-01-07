#![cfg_attr(feature = "unstable", feature(test))]
#![type_length_limit = "10000000"]

//#[global_allocator]
//static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod attrib_defines;
mod constraint;
mod constraints;
mod contact;
mod energy;
pub mod energy_models;
pub mod fem;
mod friction;
pub mod mask_iter;
mod matrix;
mod objects;

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

pub type TetMeshExt = geo::mesh::TetMeshExt<f64>;
pub type TriMeshExt = geo::mesh::TriMeshExt<f64>;

pub use self::contact::*;
pub use self::contact::{ContactType, FrictionalContactParams};
pub use self::fem::{InnerSolveResult, MuStrategy, SimParams, SolveResult, Solver, SolverBuilder};
pub use self::friction::*;
pub use self::objects::init_mesh_source_index_attribute;
pub use self::objects::material::*;
use geo::mesh::attrib;
pub use utils::index::Index;

pub use attrib_defines::*;

pub use implicits::KernelType;

use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum Error {
    /// Size mismatch error
    SizeMismatch,
    #[snafu(display("Attribute error: {:?}", source))]
    AttribError {
        source: attrib::Error,
    },
    #[snafu(display("Degenerate reference element detected: {:?}", degens[0]))]
    DegenerateReferenceElement {
        // Degens from Upcountry.
        // Look, keep it at the end of the laneway. No degens on the property.
        degens: Vec<usize>,
    },
    /// Inverted reference element detected
    InvertedReferenceElement {
        inverted: Vec<usize>,
    },
    /// Error during main solve step. This reports iterations, objective value and max inner
    /// iterations.
    SolveError {
        status: ipopt::SolveStatus,
        result: SolveResult,
    },
    /// Error during an inner solve step. This reports iterations and objective value.
    InnerSolveError {
        status: ipopt::SolveStatus,
        objective_value: f64,
        iterations: u32,
    },
    #[snafu(display("Friction solve error: {:?}", status))]
    FrictionSolveError {
        status: ipopt::SolveStatus,
    },
    ContactSolveError {
        status: ipopt::SolveStatus,
    },
    SolverCreateError {
        source: ipopt::CreateError,
    },
    InvalidParameter {
        name: String,
    },
    MissingSourceIndex,
    MissingElasticityParams,
    MissingContactParams,
    MissingContactConstraint,
    NoSimulationMesh,
    NoKinematicMesh,
    /// Incorrect object is used for the given material. This may be an internal error.
    ObjectMaterialMismatch,
    /// Error during mesh IO. Typically during debugging.
    MeshIOError {
        source: geo::io::Error,
    },
    FileIOError {
        source: std::io::Error,
    },
    InvalidImplicitSurface,
    ImplicitsError {
        source: implicits::Error,
    },
    UnimplementedFeature {
        description: String,
    },
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::FileIOError { source: err }
    }
}

impl From<geo::io::Error> for Error {
    fn from(err: geo::io::Error) -> Error {
        Error::MeshIOError { source: err }
    }
}

impl From<ipopt::CreateError> for Error {
    fn from(err: ipopt::CreateError) -> Error {
        Error::SolverCreateError { source: err }
    }
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError { source: err }
    }
}

impl From<implicits::Error> for Error {
    fn from(err: implicits::Error) -> Error {
        Error::ImplicitsError { source: err }
    }
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
            Error::SolveError { status, result } => match status {
                ipopt::SolveStatus::MaximumIterationsExceeded => {
                    SimResult::Warning(format!("Maximum iterations exceeded \n{}", result))
                }
                status => SimResult::Error(format!("Solve failed: {:?}\n{}", status, result)),
            },
            Error::InnerSolveError {
                status,
                objective_value,
                iterations,
            } => SimResult::Error(format!(
                "Inner Solve failed: {:?}\nobjective value: {:?}\niterations: {:?}",
                status, objective_value, iterations
            )),
            Error::FrictionSolveError { status } => {
                SimResult::Error(format!("Friction Solve failed: {:?}", status))
            }
            Error::ContactSolveError { status } => {
                SimResult::Error(format!("Contact Solve failed: {:?}", status))
            }
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
            Error::SolverCreateError { source } => {
                SimResult::Error(format!("Failed to create a solver: {:?}", source))
            }
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

impl Into<SimResult> for Result<SolveResult, Error> {
    fn into(self) -> SimResult {
        match self {
            Ok(solve_result) => SimResult::Success(format!("{}", solve_result)),
            Err(err) => err.into(),
        }
    }
}

pub fn sim(
    tetmesh: Option<TetMesh>,
    material: SolidMaterial,
    polymesh: Option<PolyMesh>,
    sim_params: SimParams,
    interrupter: Option<Box<dyn FnMut() -> bool>>,
) -> SimResult {
    if let Some(mesh) = tetmesh {
        let mut builder = fem::SolverBuilder::new(sim_params);
        builder.add_solid(mesh, material);
        if let Some(shell_mesh) = polymesh {
            // The fixed shell is a distinct material
            let material_id = material.id + 1;
            builder.add_fixed(shell_mesh, material_id);
        }
        match builder.build() {
            Ok(mut engine) => {
                if let Some(interrupter) = interrupter {
                    engine.set_interrupter(interrupter);
                }
                engine.step().into()
            }
            Err(e) => e.into(),
        }
    } else {
        SimResult::Error("Tetmesh not found".into())
    }
}

pub(crate) fn inf_norm<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    iter.into_iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .unwrap_or(0.0)
}
