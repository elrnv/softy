extern crate alga;
extern crate nalgebra as na;
extern crate ipopt;
extern crate geometry as geo;

mod energy;
mod nodal_fem_nlp;
mod fem;

pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

pub fn sim(tetmesh: Option<&mut TetMesh>, _polymesh: Option<&mut PolyMesh>) -> SimResult {
    if let Some(mesh) = tetmesh {
        match fem::run(mesh) {
            Err(fem::Error::AttribError(e)) =>
                SimResult::Error(format!("{:?}", e).into()),
            Err(fem::Error::InvertedReferenceElement) =>
                SimResult::Error(format!("Inverted reference element detected.").into()),
            Ok(()) => SimResult::Success("".into()),
        }
    } else {
      SimResult::Error("Tetmesh not found".into())
    }
}
