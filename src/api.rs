/**
 * Application specific code goes here.
 * C Parameter interface and the Rust cook entry point are defined here.
 */

use geo;

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Params {
    tolerance: f32,
}

/// Main entry point to Rust code.
pub fn cook<F>(
    _tetmesh: Option<&mut geo::mesh::TetMesh<f64>>,
    _polymesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    _params: Params,
    _check_interrupt: F,
) -> CookResult
where
    F: FnMut() -> bool + Sync,
{
    CookResult::Success("It Works!".to_string())
}

/// The Rust version of the cook result enum.
pub enum CookResult {
    Success(String),
    Warning(String),
    Error(String),
}

