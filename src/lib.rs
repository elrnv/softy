extern crate geometry as geo;

pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

pub fn sim(_tetmesh: Option<&mut TetMesh>, _polymesh: Option<&mut PolyMesh>) -> SimResult {
    SimResult::Success(String::from(""))
}
