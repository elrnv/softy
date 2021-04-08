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
pub mod objects;

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
pub use self::fem::nl as nl_fem;
pub use self::fem::opt as opt_fem;
pub use self::friction::*;
pub use self::objects::init_mesh_source_index_attribute;
pub use self::objects::material::*;
use geo::mesh::attrib;
pub use utils::index::{CheckedIndex, Index};

pub use attrib_defines::*;

pub use implicits::KernelType;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Size mismatch error")]
    SizeMismatch,
    #[error("Attribute error: {source:?}")]
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
    #[error("Inverted reference element detected")]
    InvertedReferenceElement { inverted: Vec<usize> },
    #[error("Error during main non-linear solve step")]
    NLSolveError { result: nl_fem::SolveResult },
    #[error("Error during main optimization solve step")]
    /// This reports iterations, objective value and max inner iterations.
    OptSolveError {
        status: ipopt::SolveStatus,
        result: opt_fem::SolveResult,
    },
    #[error("Error during an inner solve step")]
    /// This reports iterations and objective value.
    InnerOptSolveError {
        status: ipopt::SolveStatus,
        objective_value: f64,
        iterations: u32,
    },
    #[error("Friction solve error: {:?}", .status)]
    FrictionSolveError {
        status: ipopt::SolveStatus,
        result: FrictionSolveResult,
    },
    #[error("Contact solve error: {:?}", .status)]
    ContactSolveError { status: ipopt::SolveStatus },
    #[error("Solver create error")]
    SolverCreateError {
        #[from]
        source: ipopt::CreateError,
    },
    #[error("Invalid parameter: {name:?}")]
    InvalidParameter { name: String },
    #[error("Missing source index")]
    MissingSourceIndex,
    #[error("Missing elasticity parameters")]
    MissingElasticityParams,
    #[error("Missing contact parameters")]
    MissingContactParams,
    #[error("Missing contact constraint")]
    MissingContactConstraint,
    #[error("No simulation mesh found")]
    NoSimulationMesh,
    #[error("No kinematic mesh found")]
    NoKinematicMesh,
    #[error("Incorrect object is used for the given material")]
    /// This may be an internal error.
    ObjectMaterialMismatch,
    #[error("Error during mesh IO")]
    /// Typically happens during debugging
    MeshIOError {
        #[from]
        source: geo::io::Error,
    },
    #[error("File I/O Error")]
    FileIOError {
        #[from]
        source: std::io::Error,
    },
    #[error("File I/O Error")]
    InvalidImplicitSurface,
    #[error("Error generating the implicit field")]
    ImplicitsError {
        #[from]
        source: implicits::Error,
    },
    #[error("Unimplemented feature: {description:?}")]
    UnimplementedFeature { description: String },
    #[error("Invalid solver configuration")]
    InvalidSolverConfig {
        #[from]
        source: fem::nl::newton::Error,
    },
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
            Error::OptSolveError { status, result } => match status {
                ipopt::SolveStatus::MaximumIterationsExceeded => {
                    SimResult::Warning(format!("Maximum iterations exceeded \n{}", result))
                }
                status => SimResult::Error(format!("Solve failed: {:?}\n{}", status, result)),
            },
            Error::InnerOptSolveError {
                status,
                objective_value,
                iterations,
            } => SimResult::Error(format!(
                "Inner Solve failed: {:?}\nobjective value: {:?}\niterations: {:?}",
                status, objective_value, iterations
            )),
            Error::FrictionSolveError { status, .. } => {
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

impl Into<SimResult> for Result<nl_fem::SolveResult, Error> {
    fn into(self) -> SimResult {
        match self {
            Ok(solve_result) => SimResult::Success(format!("{}", solve_result)),
            Err(err) => err.into(),
        }
    }
}

impl Into<SimResult> for Result<opt_fem::SolveResult, Error> {
    fn into(self) -> SimResult {
        match self {
            Ok(solve_result) => SimResult::Success(format!("{}", solve_result)),
            Err(err) => err.into(),
        }
    }
}

/// Simulate one step.
///
/// This function also serves as an example of how one may construct a solver using a solver builder and run one step.
pub fn sim(
    tetmesh: Option<TetMesh>,
    material: SolidMaterial,
    polymesh: Option<PolyMesh>,
    sim_params: crate::fem::opt::SimParams,
    interrupter: Option<Box<dyn FnMut() -> bool>>,
) -> SimResult {
    if let Some(mesh) = tetmesh {
        let mut builder = fem::opt::SolverBuilder::new(sim_params);
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

pub(crate) fn inf_norm<I, T: tensr::Real>(iter: I) -> T
where
    I: IntoIterator<Item = T>,
{
    iter.into_iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .unwrap_or(T::zero())
}
