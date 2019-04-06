#![cfg_attr(feature = "unstable", feature(test))]

//#[global_allocator]
//static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

mod attrib_defines;
mod bench;
mod constraint;
mod constraints;
mod energy;
mod energy_models;
pub mod fem;
mod index;
pub mod mask_iter;
mod matrix;

#[cfg(test)]
pub(crate) mod test_utils;

pub type PointCloud = geo::mesh::PointCloud<f64>;
pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;
pub type TriMesh = geo::mesh::TriMesh<f64>;

pub use self::constraints::{ContactType, SmoothContactParams};
pub use self::fem::{
    ElasticityParameters, InnerSolveResult, Material, MuStrategy, SimParams, SolveResult,
};
use geo::mesh::attrib;
pub use index::Index;

#[derive(Debug)]
pub enum Error {
    SizeMismatch,
    AttribError(attrib::Error),
    InvertedReferenceElement,
    /// Error during main solve step. This reports iterations, objective value and max inner
    /// iterations.
    SolveError(ipopt::SolveStatus, SolveResult),
    /// Error during an inner solve step. This reports iterations and objective value.
    InnerSolveError(ipopt::SolveStatus, InnerSolveResult),
    SolverCreateError(ipopt::CreateError),
    InvalidParameter(String),
    MissingContactParams,
    NoSimulationMesh,
    NoKinematicMesh,
    /// Error during mesh IO. Typically during debugging.
    MeshIOError(geo::io::Error),
    InvalidImplicitSurface,
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

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<Error> for SimResult {
    fn from(err: Error) -> SimResult {
        match err {
            Error::SizeMismatch => SimResult::Error("Size mismatch error.".to_string()),
            Error::AttribError(e) => SimResult::Error(format!("Attribute error: {:?}", e)),
            Error::InvertedReferenceElement => {
                SimResult::Error("Inverted reference element detected.".to_string())
            }
            Error::SolveError(e, solve_result) => match e {
                ipopt::SolveStatus::MaximumIterationsExceeded => {
                    SimResult::Warning(format!("Maximum iterations exceeded. \n{}", solve_result))
                }
                e => SimResult::Error(format!("Solve failed: {:?}\n{}", e, solve_result)),
            },
            Error::InnerSolveError(e, solve_result) => {
                SimResult::Error(format!("Inner Solve failed: {:?}\n{:?}", e, solve_result))
            }
            Error::MissingContactParams => {
                SimResult::Error("Missing smooth contact parameters.".to_string())
            }
            Error::NoSimulationMesh => SimResult::Error("Missing simulation mesh.".to_string()),
            Error::NoKinematicMesh => SimResult::Error("Missing kinematic mesh.".to_string()),
            Error::SolverCreateError(err) => {
                SimResult::Error(format!("Failed to create a solver: {:?}", err))
            }
            Error::InvalidParameter(err) => {
                SimResult::Error(format!("Invalid parameter: {:?}", err))
            }
            Error::MeshIOError(err) => {
                SimResult::Error(format!("Error during mesh I/O: {:?}", err))
            }
            Error::InvalidImplicitSurface => {
                SimResult::Error("Error creating an implicit surface".to_string())
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
    material: Material,
    polymesh: Option<PolyMesh>,
    sim_params: SimParams,
    interrupter: Option<Box<FnMut() -> bool>>,
) -> SimResult {
    if let Some(mesh) = tetmesh {
        let mut builder = fem::SolverBuilder::new(sim_params);
        builder.add_solid(mesh).solid_material(material);
        if let Some(shell_mesh) = polymesh {
            builder.add_shell(shell_mesh);
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

pub(crate) fn inf_norm(vec: &[f64]) -> f64 {
    vec.iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).expect("Detected NaNs"))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::mesh::{topology::*, Attrib, TetMesh};
    use test_utils::*;

    const MATERIAL: Material = Material {
        elasticity: ElasticityParameters {
            bulk_modulus: 1750e6,
            shear_modulus: 10e6,
        },
        ..SOLID_MATERIAL
    };

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

        assert!(match sim(Some(mesh), MATERIAL, None, STATIC_PARAMS, None) {
            SimResult::Success(_) => true,
            _ => false,
        });
    }
}
