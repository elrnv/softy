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
    pub radius: f32,
    pub kernel: i32,
}

impl Into<mls::Params> for Params {
    fn into(self) -> mls::Params {
        let Params { tolerance, radius, kernel } = self;
        mls::Params {
            kernel: match kernel {
                0 => mls::Kernel::Interpolating {
                    radius: radius as f64,
                    tolerance: tolerance as f64,
                },
                1 => mls::Kernel::Cubic { radius: radius as f64 },
                2 => mls::Kernel::Global { tolerance: tolerance as f64 },
                _ => mls::Kernel::Hrbf { radius: radius as f64 },
            }
        }
    }
}

/// Main entry point to Rust code.
pub fn cook<F>(
    samplemesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    polymesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    params: Params,
    check_interrupt: F,
) -> CookResult
where
    F: Fn() -> bool + Sync + Send,
{
    if let Some(samples) = samplemesh {
        if let Some(surface) = polymesh {
            mls::compute_mls(samples, surface, params.into(), check_interrupt).into()
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
            Err(mls::Error::Interrupted) => {
                CookResult::Error("Execution was interrupted.".to_string())
            }
            Err(mls::Error::MissingNormals) => {
                CookResult::Error("Vertex normals are missing or have the wrong type.".to_string())
            }
            Err(mls::Error::Failure) => CookResult::Error("Internal Error.".to_string()),
        }
    }
}
