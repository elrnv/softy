/**
 * Application specific code goes here.
 * The Rust cook entry point is defined here.
 * This file is intended to be completely free from C FFI except for POD types, which must be
 * designated as `#[repr(C)]`.
 */

use geo;
use implicits;
use hdkrs::interop::CookResult;

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub tolerance: f32,
    pub radius: f32,
    pub kernel: i32,
    pub background_potential: i32,
    pub sample_type: i32,
}

impl Into<implicits::Params> for Params {
    fn into(self) -> implicits::Params {
        let Params { tolerance, radius, kernel, background_potential, sample_type } = self;
        implicits::Params {
            kernel: match kernel {
                0 => implicits::KernelType::Interpolating {
                    radius: radius as f64,
                },
                1 => implicits::KernelType::Approximate {
                    radius: radius as f64,
                    tolerance: tolerance as f64,
                },
                2 => implicits::KernelType::Cubic { radius: radius as f64 },
                3 => implicits::KernelType::Global { tolerance: tolerance as f64 },
                _ => implicits::KernelType::Hrbf,
            },
            background_potential: match background_potential {
                0 => implicits::BackgroundPotentialType::None,
                1 => implicits::BackgroundPotentialType::Zero,
                2 => implicits::BackgroundPotentialType::FromInput,
                3 => implicits::BackgroundPotentialType::DistanceBased,
                _ => implicits::BackgroundPotentialType::NormalBased,
            },
            sample_type: match sample_type {
                0 => implicits::SampleType::Vertex,
                _ => implicits::SampleType::Face,
            },
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
            let res = implicits::compute_potential(samples, surface, params.into(), check_interrupt);
            convert_to_cookresult(res)
        } else {
            CookResult::Error("Missing Polygonal Surface".to_string())
        }
    } else {
        CookResult::Error("Missing Sample Mesh".to_string())
    }
}

fn convert_to_cookresult(res: Result<(), implicits::Error>) -> CookResult {
    match res {
        Ok(()) => CookResult::Success("".to_string()),
        Err(implicits::Error::Interrupted) =>
            CookResult::Error("Execution was interrupted.".to_string()),
        Err(implicits::Error::MissingNormals) =>
            CookResult::Error("Vertex normals are missing or have the wrong type.".to_string()),
        Err(implicits::Error::Failure) =>
            CookResult::Error("Internal Error.".to_string()),
        Err(implicits::Error::UnsupportedKernel) =>
            CookResult::Error("Given kernel is not supported yet.".to_string()),
        Err(implicits::Error::IO(err)) =>
            CookResult::Error(format!("IO Error: {:?}", err)),
    }
}
