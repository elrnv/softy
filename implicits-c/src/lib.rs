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

/// Free memory allocated for the trimesh.
#[no_mangle]
pub unsafe extern "C" fn free_trimesh(trimesh: *mut TriMesh) {
    if !trimesh.is_null() {
        let _ = Box::from_raw(trimesh);
    }
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

/// Free memory allocated for the implicit surface.
#[no_mangle]
pub unsafe extern "C" fn free_implicit_surface(implicit_surface: *mut ImplicitSurface) {
    if !implicit_surface.is_null() {
        let _ = Box::from_raw(implicit_surface);
    }
}

/// Compute potential. Return 0 on success.
#[no_mangle]
pub unsafe extern "C" fn compute_potential(
    implicit_surface: *const ImplicitSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
    out_potential: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize*3);
    let query_points: &[[f64;3]] = reinterpret_slice(coords);
    let out_potential = std::slice::from_raw_parts_mut(out_potential, num_query_points as usize);

    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.potential(query_points, out_potential) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}

/// Get the number of non zeros in the Jacobian of the implicit function.
#[no_mangle]
pub unsafe extern "C" fn num_jacobian_non_zeros(
    implicit_surface: *const ImplicitSurface,
) -> c_int {
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);
    surf.num_surface_jacobian_entries() as c_int
}

/// Compute the index structure of the Jacobian for the given implicit function.
#[no_mangle]
pub unsafe extern "C" fn compute_jacobian_indices(
    implicit_surface: *const ImplicitSurface,
    num_non_zeros: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    let n = num_non_zeros as usize;
    let rows = std::slice::from_raw_parts_mut(rows, n);
    let cols = std::slice::from_raw_parts_mut(cols, n);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.surface_jacobian_indices_iter() {
        Ok(iter) => {
            for ((out_row, out_col), (r,c)) in rows.iter_mut().zip(cols.iter_mut()).zip(iter) {
                *out_row = r as c_int;
                *out_col = c as c_int;
            }
            0
        }
        Err(_) => 1,
    }
}

/// Compute Jacobian values for the given implicit function. The values correspond to the indices
/// provided by `compute_jacobian_indices`.
#[no_mangle]
pub unsafe extern "C" fn compute_jacobian_values(
    implicit_surface: *const ImplicitSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
    num_non_zeros: c_int,
    values: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize*3);
    let query_points: &[[f64;3]] = reinterpret_slice(coords);
    let vals = std::slice::from_raw_parts_mut(values, num_non_zeros as usize);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.surface_jacobian_values(query_points, vals) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}
