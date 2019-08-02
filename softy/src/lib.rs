#![cfg_attr(feature = "unstable", feature(test))]

//#[global_allocator]
//static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod attrib_defines;
mod bench;
mod constraint;
mod constraints;
mod contact;
mod energy;
mod energy_models;
pub mod fem;
mod friction;
mod index;
pub mod mask_iter;
mod matrix;
mod objects;

#[cfg(test)]
pub(crate) mod test_utils;

pub type PointCloud = geo::mesh::PointCloud<f64>;
pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;
pub type TriMesh = geo::mesh::TriMesh<f64>;

pub use self::contact::*;
pub use self::contact::{ContactType, FrictionalContactParams};
pub use self::fem::{InnerSolveResult, MuStrategy, SimParams, SolveResult};
pub use self::friction::*;
pub use self::objects::material::*;
use geo::mesh::attrib;
pub use index::Index;

pub use attrib_defines::{SourceIndexType, SOURCE_INDEX_ATTRIB};

pub use implicits::KernelType;

#[derive(Debug)]
pub enum Error {
    SizeMismatch,
    AttribError(attrib::Error),
    InvertedReferenceElement,
    /// Error during main solve step. This reports iterations, objective value and max inner
    /// iterations.
    SolveError(ipopt::SolveStatus, SolveResult),
    /// Error during an inner solve step. This reports iterations and objective value.
    InnerSolveError {
        status: ipopt::SolveStatus,
        objective_value: f64,
        iterations: u32,
    },
    FrictionSolveError(ipopt::SolveStatus),
    SolverCreateError(ipopt::CreateError),
    InvalidParameter(String),
    MissingSourceIndex,
    MissingDensityParam,
    MissingElasticityParams,
    MissingContactParams,
    MissingContactConstraint,
    NoSimulationMesh,
    NoKinematicMesh,
    /// Error during mesh IO. Typically during debugging.
    MeshIOError(geo::io::Error),
    FileIOError(std::io::ErrorKind),
    InvalidImplicitSurface,
    ImplicitsError(implicits::Error),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::FileIOError(err.kind())
    }
}

impl From<geo::io::Error> for Error {
    fn from(err: geo::io::Error) -> Error {
        Error::MeshIOError(err)
    }
}

impl From<ipopt::CreateError> for Error {
    fn from(err: ipopt::CreateError) -> Error {
        Error::SolverCreateError(err)
    }
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError(err)
    }
}

impl From<implicits::Error> for Error {
    fn from(err: implicits::Error) -> Error {
        Error::ImplicitsError(err)
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
            Error::SizeMismatch => SimResult::Error("Size mismatch error".to_string()),
            Error::AttribError(e) => SimResult::Error(format!("Attribute error: {:?}", e)),
            Error::InvertedReferenceElement => {
                SimResult::Error("Inverted reference element detected".to_string())
            }
            Error::SolveError(e, solve_result) => match e {
                ipopt::SolveStatus::MaximumIterationsExceeded => {
                    SimResult::Warning(format!("Maximum iterations exceeded \n{}", solve_result))
                }
                e => SimResult::Error(format!("Solve failed: {:?}\n{}", e, solve_result)),
            },
            Error::InnerSolveError {
                status,
                objective_value,
                iterations,
            } => SimResult::Error(format!(
                "Inner Solve failed: {:?}\nobjective value: {:?}\niterations: {:?}",
                status, objective_value, iterations
            )),
            Error::FrictionSolveError(e) => {
                SimResult::Error(format!("Friction Solve failed: {:?}", e))
            }
            Error::MissingSourceIndex => {
                SimResult::Error("Missing source index vertex attribute".to_string())
            }
            Error::MissingDensityParam => SimResult::Error(
                "Missing density parameter or per-element density attribute".to_string(),
            ),
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
            Error::SolverCreateError(err) => {
                SimResult::Error(format!("Failed to create a solver: {:?}", err))
            }
            Error::InvalidParameter(err) => {
                SimResult::Error(format!("Invalid parameter: {:?}", err))
            }
            Error::MeshIOError(err) => {
                SimResult::Error(format!("Error during mesh I/O: {:?}", err))
            }
            Error::FileIOError(err) => SimResult::Error(format!("File I/O error: {:?}", err)),
            Error::InvalidImplicitSurface => {
                SimResult::Error("Error creating an implicit surface".to_string())
            }
            Error::ImplicitsError(err) => {
                SimResult::Error(format!("Error computing implicit surface: {:?}", err))
            }
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
        .max_by(|a, b| a.partial_cmp(b).expect("Detected NaNs"))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::mesh::{topology::*, Attrib, TetMesh};
    use test_utils::*;

    fn material() -> SolidMaterial {
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(1750e6, 10e6))
    }

    #[test]
    fn sim_test() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ];
        let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
        let mut mesh = TetMesh::new(verts, indices);
        mesh.add_attrib_data::<i8, VertexIndex>("fixed", vec![0, 0, 1, 1, 0, 0])
            .unwrap();

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
            .unwrap();

        assert!(
            match sim(Some(mesh), material(), None, STATIC_PARAMS, None) {
                SimResult::Success(_) => true,
                _ => false,
            }
        );
    }
}
