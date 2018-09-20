#![cfg_attr(feature = "unstable", feature(test))]

extern crate geometry as geo;
extern crate ipopt;
extern crate nalgebra as na;
extern crate rayon;
extern crate reinterpret;

mod bench;
mod energy;
mod energy_model;
mod fem;

pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;

// reexport params structs for interfacing.
pub use fem::{FemEngine, MaterialProperties, SimParams, Error};

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<fem::Error> for SimResult {
    fn from(err: fem::Error) -> SimResult {
        match err {
            fem::Error::AttribError(e) => {
                SimResult::Error(format!("Attribute error: {:?}", e).into())
            }
            fem::Error::InvertedReferenceElement => {
                SimResult::Error(format!("Inverted reference element detected.").into())
            }
            fem::Error::SolveError(e, result) => SimResult::Error(
                format!(
                    "Solve failed: {:?}\nIterations: {}\nObjective: {}",
                    e, result.iterations, result.objective_value
                ).into(),
            ),
        }
    }
}

pub fn sim<F>(
    tetmesh: Option<&mut TetMesh>,
    _polymesh: Option<&mut PolyMesh>,
    sim_params: SimParams,
    check_interrupt: F,
) -> SimResult
where
    F: FnMut() -> bool + Sync,
{
    if let Some(mesh) = tetmesh {
        match FemEngine::new(mesh, sim_params, check_interrupt) {
            Ok(mut engine) => match engine.step() {
                Err(e) => e.into(),
                Ok(fem::SolveResult {
                    iterations,
                    objective_value,
                }) => SimResult::Success(
                    format!("Iterations: {}\nObjective: {}", iterations, objective_value).into(),
                ),
            },
            Err(e) => e.into(),
        }
    } else {
        SimResult::Error("Tetmesh not found".into())
    }
}
