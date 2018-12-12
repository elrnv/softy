use reinterpret::*;
use std::os::raw::{c_int, c_double};

pub use implicits::{
    KernelType,
    BackgroundPotentialType,
    SampleType,
};

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub tolerance: f32,
    pub radius: f32,
    pub kernel: i32,
    pub background_potential: i32,
    pub sample_type: i32,
    pub max_step: f32,
}

impl Into<implicits::Params> for Params {
    fn into(self) -> implicits::Params {
        let Params { tolerance, radius, kernel, background_potential, sample_type, max_step } = self;
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
            max_step: max_step as f64
        }
    }
}

/// Opaque TriMesh type
pub struct TriMesh;

/// Opaque ImplicitSurface type
pub struct ImplicitSurface;

/// Create a triangle mesh from the given arrays of data.
#[no_mangle]
pub unsafe extern "C" fn create_trimesh(
    num_coords: c_int,
    coords: *const c_double,
    num_indices: c_int,
    indices: *const c_int,
) -> *mut TriMesh {
    assert_eq!(
        num_coords % 3,
        0,
        "Given coordinate array size is not a multiple of 3."
    );
    let coords = std::slice::from_raw_parts(coords, num_coords as usize);
    let positions: Vec<[f64;3]> = reinterpret_vec(coords.iter().map(|&x| f64::from(x)).collect());
    let indices = std::slice::from_raw_parts(indices, num_indices as usize).iter().map(|&x| x as usize).collect();

    let mesh = Box::new(geometry::mesh::TriMesh::new(positions, indices));
    Box::into_raw(mesh) as *mut TriMesh
}

/// Create a new implicit surface. If creation fails, a null pointer is returned.
#[no_mangle]
pub unsafe extern "C" fn create_implicit_surface(
    trimesh: *const TriMesh,
    params: Params,
) -> *mut ImplicitSurface {
    match implicits::surface_from_trimesh(
        &*(trimesh as *const geometry::mesh::TriMesh<f64>),
	params.into()
    ) {
        Ok(surf) => Box::into_raw(Box::new(surf)) as *mut ImplicitSurface,
        Err(_) => std::ptr::null_mut()
    }
}
