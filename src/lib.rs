#![cfg_attr(feature = "unstable", feature(test))]

extern crate alga;
extern crate geometry as geo;
extern crate ipopt;
extern crate nalgebra as na;

mod util;
mod bench;
mod energy;
mod energy_model;
mod fem;

pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

pub fn sim<F>(
    tetmesh: Option<&mut TetMesh>,
    _polymesh: Option<&mut PolyMesh>,
    check_interrupt: F,
) -> SimResult
where
    F: Fn() -> bool,
{
    if let Some(mesh) = tetmesh {
        match fem::run(mesh, check_interrupt) {
            Err(fem::Error::AttribError(e)) => {
                SimResult::Error(format!("Attribute error: {:?}", e).into())
            }
            Err(fem::Error::InvertedReferenceElement) => {
                SimResult::Error(format!("Inverted reference element detected.").into())
            }
            Err(fem::Error::SolveError(e)) => {
                SimResult::Error(format!("Solve failed: {:?}", e).into())
            }
            Ok(()) => SimResult::Success("".into()),
        }
    } else {
        SimResult::Error("Tetmesh not found".into())
    }
}
