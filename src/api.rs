/**
 * Application specific code goes here.
 * C Parameter interface and the Rust cook entry point are defined here.
 */
use geo;
use mls;

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub tolerance: f32,
}

/// Main entry point to Rust code.
pub fn cook<F>(
    samplemesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    polymesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    params: Params,
    check_interrupt: F,
) -> CookResult
where
    F: FnMut() -> bool + Sync,
{
    if let Some(samples) = samplemesh {
        if let Some(surface) = polymesh {
            mls::compute_mls(samples, surface, params, check_interrupt).into()
        } else {
            CookResult::Error("Missing Polygonal Surface".to_string())
        }
    } else {
        CookResult::Error("Missing Sample Mesh".to_string())
    }
}

/// The Rust version of the cook result enum.
pub enum CookResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<Result<(), mls::Error>> for CookResult {
    fn from(res: Result<(), mls::Error>) -> Self {
        match res {
            Ok(()) => CookResult::Success("".to_string()),
            Err(mls::Error::MissingNormals) => {
                CookResult::Error("Vertex normals are missing or have the wrong type.".to_string())
            }
            Err(mls::Error::Failure) => CookResult::Error("Internal Error.".to_string()),
        }
    }
}
