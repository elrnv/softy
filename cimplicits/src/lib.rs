use reinterpret::*;
use std::os::raw::{c_double, c_int};

pub use implicits::{BackgroundFieldType, KernelType, SampleType};

/// A C interface for passing parameters from SOP parameters to the Rust code.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_IsoParams {
    pub tolerance: f32,
    pub radius: f32,
    pub kernel: i32,
    pub background_potential: i32,
    pub weighted: i32,
    pub sample_type: i32,
    pub max_step: f32,
}

impl Into<implicits::Params> for EL_IsoParams {
    fn into(self) -> implicits::Params {
        let EL_IsoParams {
            tolerance,
            radius,
            kernel,
            background_potential,
            weighted,
            sample_type,
            max_step,
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
            background_field: implicits::BackgroundFieldParams {
                field_type: match background_potential {
                    0 => implicits::BackgroundFieldType::Zero,
                    1 => implicits::BackgroundFieldType::FromInput,
                    _ => implicits::BackgroundFieldType::DistanceBased,
                },
                weighted: weighted != 0,
            },
            sample_type: match sample_type {
                0 => implicits::SampleType::Vertex,
                _ => implicits::SampleType::Face,
            },
            max_step: max_step as f64,
        }
    }
}

/// Opaque TriMesh type
#[allow(non_camel_case_types)]
pub struct EL_IsoTriMesh;

/// Opaque iso-surface type
#[allow(non_camel_case_types)]
pub struct EL_IsoSurface;

/// Create a triangle mesh from the given arrays of data.
#[no_mangle]
pub unsafe extern "C" fn el_iso_create_trimesh(
    num_coords: c_int,
    coords: *const c_double,
    num_indices: c_int,
    indices: *const c_int,
) -> *mut EL_IsoTriMesh {
    assert_eq!(
        num_coords % 3,
        0,
        "Given coordinate array size is not a multiple of 3."
    );
    let coords = std::slice::from_raw_parts(coords, num_coords as usize);
    let positions: Vec<[f64; 3]> = reinterpret_vec(coords.iter().map(|&x| f64::from(x)).collect());
    let indices = std::slice::from_raw_parts(indices, num_indices as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let mesh = Box::new(geometry::mesh::TriMesh::new(positions, indices));
    Box::into_raw(mesh) as *mut EL_IsoTriMesh
}

/// Free memory allocated for the trimesh.
#[no_mangle]
pub unsafe extern "C" fn el_iso_free_trimesh(trimesh: *mut EL_IsoTriMesh) {
    if !trimesh.is_null() {
        let _ = Box::from_raw(trimesh);
    }
}

/// Create a new implicit surface. If creation fails, a null pointer is returned.
#[no_mangle]
pub unsafe extern "C" fn el_iso_create_implicit_surface(
    trimesh: *const EL_IsoTriMesh,
    params: EL_IsoParams,
) -> *mut EL_IsoSurface {
    match implicits::surface_from_trimesh::<f64>(
        &*(trimesh as *const geometry::mesh::TriMesh<f64>),
        params.into(),
    ) {
        Ok(surf) => Box::into_raw(Box::new(surf)) as *mut EL_IsoSurface,
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free memory allocated for the implicit surface.
#[no_mangle]
pub unsafe extern "C" fn el_iso_free_implicit_surface(implicit_surface: *mut EL_IsoSurface) {
    if !implicit_surface.is_null() {
        let _ = Box::from_raw(implicit_surface);
    }
}

/// Compute potential. Return 0 on success.
#[no_mangle]
pub unsafe extern "C" fn el_iso_compute_potential(
    implicit_surface: *const EL_IsoSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
    out_potential: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let out_potential = std::slice::from_raw_parts_mut(out_potential, num_query_points as usize);

    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.potential(query_points, out_potential) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}

/// Project the given positions to the given iso value of the potential field represented by this
/// implicit surface.
#[no_mangle]
pub unsafe extern "C" fn el_iso_project_to_above(
    implicit_surface: *const EL_IsoSurface,
    iso_value: f64,
    tolerance: f64,
    num_query_points: c_int,
    query_point_coords: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts_mut(query_point_coords, num_query_points as usize * 3);
    let query_points: &mut [[f64; 3]] = reinterpret_mut_slice(coords);

    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.project_to_above(iso_value, tolerance, query_points) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}

/// Get the number of entries in the sparse Jacobian of the implicit function with respect to surface
/// vertices.
///
/// These typically corresond to non-zero entries of the Jacobian, however they can
/// actually be zero and multiple entries can correspond to the same position in the matrix,
/// in which case the corresponding values will be summed.
///
/// This function determines the sizes of the mutable arrays expected in
/// `el_iso_surface_jacobian_indices` and `el_iso_surface_jacobian_values` functions.
#[no_mangle]
pub unsafe extern "C" fn el_iso_num_surface_jacobian_entries(implicit_surface: *const EL_IsoSurface) -> c_int {
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);
    surf.num_surface_jacobian_entries().unwrap_or(0) as c_int
}

/// Compute the index structure of the sparse surface Jacobian for the given implicit function.
/// Each computed row-column index pair corresponds to an entry in the sparse Jacobian.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_jacobian_indices(
    implicit_surface: *const EL_IsoSurface,
    num_entries: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    let n = num_entries as usize;
    let rows = std::slice::from_raw_parts_mut(rows, n);
    let cols = std::slice::from_raw_parts_mut(cols, n);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.surface_jacobian_indices_iter() {
        Ok(iter) => {
            for ((out_row, out_col), (r, c)) in rows.iter_mut().zip(cols.iter_mut()).zip(iter) {
                *out_row = r as c_int;
                *out_col = c as c_int;
            }
            0
        }
        Err(_) => 1,
    }
}

/// Compute surface Jacobian values for the given implicit function. The values correspond to the
/// indices provided by `el_iso_surface_jacobian_indices`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_jacobian_values(
    implicit_surface: *const EL_IsoSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
    num_entries: c_int,
    values: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let vals = std::slice::from_raw_parts_mut(values, num_entries as usize);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.surface_jacobian_values(query_points, vals) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Compute query Jacobian for the given implicit function. Each triplet of coordinates corresponds
/// to the Jacobian at each query point with respect to the query position. This means that
/// `values` has size `num_query_points*3`, just like `query_point_coords`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_query_jacobian(
    implicit_surface: *const EL_IsoSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
    values: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let vals = std::slice::from_raw_parts_mut(values, num_query_points as usize * 3);
    let value_vecs: &mut [[f64; 3]] = reinterpret_mut_slice(vals);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.query_jacobian_full(query_points, value_vecs) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Get the number of entries in the sparse Hessian product of the implicit function with respect
/// to surface vertices.
/// These typically corresond to non-zero entries of the Hessian product, however they can
/// actually be zero and multiple entries can correspond to the same position in the matrix,
/// in which case the corresponding values will be summed.
///
/// The Hessian product refers to the product of the full Hessian of the implicit
/// function contracted against a vector multipliers in the number of query points. Thus this
/// Hessian product is an 3n-by-3n matrix, where n is the number of surface vertices.
/// This function will return 0 if no query points were previously cached.
#[no_mangle]
pub unsafe extern "C" fn el_iso_num_surface_hessian_product_entries(implicit_surface: *const EL_IsoSurface) -> c_int {
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);
    surf.num_surface_hessian_product_entries().unwrap_or(0) as c_int
}

/// Compute the index structure of the surface Hessian product for the given implicit function.
///
/// The Hessian product refers to the product of the full Hessian of the implicit
/// function contracted against a vector multipliers in the number of query points. Thus this
/// Hessian product is an 3n-by-3n matrix, where n is the number of surface vertices.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_hessian_product_indices(
    implicit_surface: *const EL_IsoSurface,
    num_entries: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    let n = num_entries as usize;
    let rows = std::slice::from_raw_parts_mut(rows, n);
    let cols = std::slice::from_raw_parts_mut(cols, n);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.surface_hessian_product_indices_iter() {
        Ok(iter) => {
            for ((out_row, out_col), (r, c)) in rows.iter_mut().zip(cols.iter_mut()).zip(iter) {
                *out_row = r as c_int;
                *out_col = c as c_int;
            }
            0
        }
        Err(_) => 1,
    }
}

/// Compute surface Hessian product values for the given implicit function.
///
/// The Hessian product refers to the product of the full Hessian of the implicit
/// function contracted against a vector multipliers in the number of query points. Thus this
/// Hessian product is an 3n-by-3n matrix, where n is the number of surface vertices.
///
/// `num_query_points`   - number of triplets provided in `query_point_coords`.
/// `query_point_coords` - triplets of coordinates (x,y,z) of the query points.
/// `multipliers` - vector of multiplers of size equal to `num_query_points` that is to be
///                 multiplied by the full Hessian to get the Hessian product.
/// `num_entries` - size of the `values` array, which should be equal to
///                 `el_iso_num_surface_hessian_product_entries`.
/// `values` - matrix entries corresponding to the indices provided by
///            `el_iso_surface_hessian_product_indices`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_hessian_product_values(
    implicit_surface: *const EL_IsoSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
    multipliers: *const f64,
    num_entries: c_int,
    values: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let multipliers = std::slice::from_raw_parts(multipliers, num_query_points as usize);
    let vals = std::slice::from_raw_parts_mut(values, num_entries as usize);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    match surf.surface_hessian_product_values(query_points, multipliers, vals) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Invalidate neighbour cache. This must be done explicitly after we have completed a simulation
/// step. This implies that the implicit surface may change.
#[no_mangle]
pub unsafe extern "C" fn el_iso_invalidate_neighbour_cache(
    implicit_surface: *const EL_IsoSurface,
) {
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);
    surf.invalidate_query_neighbourhood();
}

/// Recompute the neighbour cache if invalidated. This is done automatically when computing
/// potential or jacobian. But it needs to be done explicitly to generate the correct Jacobian
/// sparsity pattern because query points are not typically passed when requesting Jacobian
/// indices.
#[no_mangle]
pub unsafe extern "C" fn el_iso_cache_neighbours(
    implicit_surface: *const EL_IsoSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
) {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let surf = &*(implicit_surface as *const implicits::ImplicitSurface);

    surf.cache_neighbours(query_points);
}

/// Update the implicit surface with new vertex positions for the underlying mesh.
/// The `position_coords` array is expected to have size of 3 times `num_positions`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_update(
    implicit_surface: *mut EL_IsoSurface,
    num_positions: c_int,
    position_coords: *const f64,
) -> c_int {
    let coords= std::slice::from_raw_parts(position_coords, num_positions as usize * 3);
    let pos: &[[f64; 3]] = reinterpret_slice(coords);
    let surf = &mut *(implicit_surface as *mut implicits::ImplicitSurface);
    surf.update(pos.iter().cloned()) as c_int
}
