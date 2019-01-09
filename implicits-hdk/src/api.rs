/**
 * Application specific code goes here.
 * The Rust cook entry point is defined here.
 * This file is intended to be completely free from C FFI except for POD types, which must be
 * designated as `#[repr(C)]`.
 */
use geo;
use hdkrs::interop::CookResult;
use implicits;

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub action: i32,
    pub tolerance: f32,
    pub radius: f32,
    pub kernel: i32,
    pub background_potential: i32,
    pub sample_type: i32,
}

impl Into<implicits::Params> for Params {
    fn into(self) -> implicits::Params {
        let Params {
            tolerance,
            radius,
            kernel,
            background_potential,
            sample_type,
            ..
        } = self;
        implicits::Params {
            kernel: match kernel {
                0 => implicits::KernelType::Interpolating {
                    radius: radius as f64,
                },
                1 => implicits::KernelType::Approximate {
                    radius: radius as f64,
                    tolerance: tolerance as f64,
                },
                2 => implicits::KernelType::Cubic {
                    radius: radius as f64,
                },
                3 => implicits::KernelType::Global {
                    tolerance: tolerance as f64,
                },
                _ => implicits::KernelType::Hrbf,
            },
            background_field: match background_potential {
                0 => implicits::BackgroundFieldType::None,
                1 => implicits::BackgroundFieldType::Zero,
                2 => implicits::BackgroundFieldType::FromInput,
                3 => implicits::BackgroundFieldType::DistanceBased,
                _ => implicits::BackgroundFieldType::NormalBased,
            },
            sample_type: match sample_type {
                0 => implicits::SampleType::Vertex,
                _ => implicits::SampleType::Face,
            },
            ..Default::default()
        }
    }
}

fn project_vertices(
    samplemesh: &mut geo::mesh::PolyMesh<f64>,
    surface: &mut geo::mesh::PolyMesh<f64>,
    params: Params) -> Result<(), implicits::Error>
{
    use geo::mesh::VertexPositions;

    let surf = implicits::surface_from_polymesh(surface, params.into())?;

    let pos = samplemesh.vertex_positions_mut();
    surf.project_to_above(0.0, 1e-4, pos)?;

    Ok(())
}

/// Main entry point to Rust code.
pub fn cook<F>(
    samplemesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    polymesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    params: Params,
    check_interrupt: F,
) -> CookResult
where
    F: Fn() -> bool + Sync + Send + Clone,
{
    if let Some(samples) = samplemesh {
        if let Some(surface) = polymesh {
            match params.action {
                0 => { // Compute potential
                    let res = implicits::compute_potential_debug(
                        samples,
                        surface,
                        params.into(),
                        check_interrupt,
                    );
                    convert_to_cookresult(res)
                },
                _ => { // Project vertices
                    let res = project_vertices(samples, surface, params);
                    convert_to_cookresult(res)
                }
            }
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
        Err(implicits::Error::Interrupted) => {
            CookResult::Error("Execution was interrupted.".to_string())
        }
        Err(implicits::Error::MissingNormals) => {
            CookResult::Error("Vertex normals are missing or have the wrong type.".to_string())
        }
        Err(implicits::Error::Failure) => CookResult::Error("Internal Error.".to_string()),
        Err(implicits::Error::UnsupportedKernel) => {
            CookResult::Error("Given kernel is not supported yet.".to_string())
        }
        Err(implicits::Error::IO(err)) => CookResult::Error(format!("IO Error: {:?}", err)),
    }
}
