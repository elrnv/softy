/**
 * Application specific code goes here.
 * C Parameter interface and the Rust cook entry point are defined here.
 */
use geo;
use implicits;

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub tolerance: f32,
    pub radius: f32,
    pub kernel: i32,
}

impl Into<implicits::Params> for Params {
    fn into(self) -> implicits::Params {
        let Params { tolerance, radius, kernel } = self;
        implicits::Params {
            kernel: match kernel {
                0 => implicits::Kernel::Interpolating {
                    radius: radius as f64,
                },
                1 => implicits::Kernel::Approximate {
                    radius: radius as f64,
                    tolerance: tolerance as f64,
                },
                2 => implicits::Kernel::Cubic { radius: radius as f64 },
                3 => implicits::Kernel::Global { tolerance: tolerance as f64 },
                _ => implicits::Kernel::Hrbf,
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
            implicits::compute_potential(samples, surface, params.into(), check_interrupt).into()
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

impl From<Result<(), implicits::Error>> for CookResult {
    fn from(res: Result<(), implicits::Error>) -> Self {
        match res {
            Ok(()) => CookResult::Success("".to_string()),
            Err(implicits::Error::Interrupted) => {
                CookResult::Error("Execution was interrupted.".to_string())
            }
            Err(implicits::Error::MissingNormals) => {
                CookResult::Error("Vertex normals are missing or have the wrong type.".to_string())
            }
            Err(implicits::Error::Failure) => CookResult::Error("Internal Error.".to_string()),
        }
    }
}
