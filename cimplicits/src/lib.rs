#![type_length_limit = "10000000"]

use reinterpret::*;
use std::os::raw::{c_double, c_int};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum SampleType {
    /// Samples are located at mesh vertices and normals are computed using an area weighted
    /// normals of adjacent triangles.
    Vertex,
    /// Samples are located at triangle centroids. This type of implicit surface is typically
    /// closer to the triangle surface than Vertex based implicits, especially when the triangle
    /// mesh is close to being uniform.
    Face,
}

/// The style of background potential to be used alongside a local potential field.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum BackgroundFieldType {
    /// Background field is set to be zero.
    Zero,
    /// When computing the potential field, the input values will be used as the background
    /// potential.
    FromInput,
    /// A signed distance to the closest sample point is used for the background field.
    DistanceBased,
}

/// The type of kernel to be used when interpolating potential values from a neighbourhood of
/// samples.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum KernelType {
    /// Local Interpolating kernel. This kernel will generate an implicit surface that passes
    /// through each sample point exactly, however it can produce singularities for large radii
    /// and is generally not suitable for simulation.
    LocalInterpolating,
    /// Local Approximately Interpolating kernel. This kernel approximately interpolates the
    /// sample points defined by the input triangle mesh. The tradeoff between interpolating
    /// quality and smoothness is determined by the given tolerance. (See [Most and Bucher 2005]
    /// for details).
    LocalApproximate,
    /// Local Cubic kernel. This is the simplest and smoothest kernel without any
    /// interpolation control. It approximates the Gaussian kernel while being fast to compute.
    /// Not unlike the Gaussian kernel, this kernel has poor interpolation properties with large
    /// radii (See [Most and Bucher 2005] for details).
    LocalCubic,
    /// Global inverse squared distance kernel. A global kernel traditionally used for surface
    /// reconstruction. It is unsuitable for simulation since the potential at each query point
    /// will depend on every single mesh sample. (See [Shen et al. 2004] for details).
    GlobalInvDist,
    /// Hermite Radial Basis Functions. This is not strictly a kernel, but this option can be
    /// interpreted as a more expensive global kernel. This option produces the smoothest potential
    /// fields while also being interpolating. However, evaluating the potential field at a set
    /// of query points requires a solve of a dense linear system. This option is not recommended
    /// for simulation.
    GlobalHrbf,
}

/// A C interface for passing parameters that determine the construction of the implicit potential
/// field.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EL_IsoParams {
    /// Used by a selection of kernels like the Approximate and Global, this value determines how
    /// cloesely the surface approximates a given triangle mesh. Lower values result in more
    /// accurate surfaces, while larger values produce smoother surfaces.
    pub tolerance: f32,
    /// The value by which the internal kernel radius is multiplied. The internal radius is
    /// determined to be the maximum of the smallest radius around each triangle centroid which
    /// contains all of the triangles vertices.
    pub radius_multiplier: f32,
    /// The type of interpolation kernel to use for interpolating local potentials around each
    /// triangle.
    pub kernel: KernelType,
    /// Option for a type of background potential to use outside the radius of influence of all
    /// samples of the triangle mesh.
    pub background_field: BackgroundFieldType,
    /// `false` -> Do NOT mix the background potential with the local potential field.
    /// `true` -> Mix the background potential with the local potential field.
    /// Note: when using the Distance based potential field, it is recommended to leave this at 0
    /// because the Distance based field contains discontinuities which will pollute the local
    /// field if mixed in.
    pub weighted: bool,
    /// The positions of sample point used to define the implicit surface.
    pub sample_type: SampleType,
    /// The max_step parameter determines the additional distance (beyond the radius of influence)
    /// to consider when computing the set of neighbourhood samples for a given set of query
    /// points. This is typically used for simulation, where the set of neighbourhoods is expected
    /// to stay constant when the positions of the samples for the implicit surface or the
    /// positions of the query points may be perturbed. If positions are unperturbed between calls
    /// to `el_iso_cache_neighbours`, then this can be set to 0.0.
    pub max_step: f32,
}

impl Into<implicits::Params> for EL_IsoParams {
    fn into(self) -> implicits::Params {
        let EL_IsoParams {
            tolerance,
            radius_multiplier,
            kernel,
            background_field,
            weighted,
            sample_type,
            max_step,
        } = self;
        implicits::Params {
            kernel: match kernel {
                KernelType::LocalInterpolating => implicits::KernelType::Interpolating {
                    radius_multiplier: radius_multiplier as f64,
                },
                KernelType::LocalApproximate => implicits::KernelType::Approximate {
                    radius_multiplier: radius_multiplier as f64,
                    tolerance: tolerance as f64,
                },
                KernelType::LocalCubic => implicits::KernelType::Cubic {
                    radius_multiplier: radius_multiplier as f64,
                },
                KernelType::GlobalInvDist => implicits::KernelType::Global {
                    tolerance: tolerance as f64,
                },
                KernelType::GlobalHrbf => implicits::KernelType::Hrbf,
            },
            background_field: implicits::BackgroundFieldParams {
                field_type: match background_field {
                    BackgroundFieldType::Zero => implicits::BackgroundFieldType::Zero,
                    BackgroundFieldType::FromInput => implicits::BackgroundFieldType::FromInput,
                    BackgroundFieldType::DistanceBased => {
                        implicits::BackgroundFieldType::DistanceBased
                    }
                },
                weighted,
            },
            sample_type: match sample_type {
                SampleType::Vertex => implicits::SampleType::Vertex,
                SampleType::Face => implicits::SampleType::Face,
            },
            max_step: max_step as f64,
        }
    }
}

/// Opaque TriMesh type.
#[allow(non_camel_case_types)]
pub struct EL_IsoTriMesh;

/// Opaque iso-surface type.
#[allow(non_camel_case_types)]
pub struct EL_IsoSurface;

/// Opaque query topology for an iso-surface.
#[allow(non_camel_case_types)]
pub struct EL_IsoQueryTopo;

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
        let _ = Box::from_raw(trimesh as *mut geometry::mesh::TriMesh<f64>);
    }
}

/// Create a new implicit surface. If creation fails, a null pointer is returned.
#[no_mangle]
pub unsafe extern "C" fn el_iso_create_implicit_surface(
    trimesh: *const EL_IsoTriMesh,
    params: EL_IsoParams,
) -> *mut EL_IsoSurface {
    match implicits::mls_from_trimesh::<f64>(
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
        let _ = Box::from_raw(implicit_surface as *mut implicits::MLS<f64>);
    }
}

/// Create a topology data structure connecting the given query points with the implicit surface.
/// If creation fails, a null pointer is returned.
/// This function consumes the given `implicit_surface`, so it is an error to call
/// `el_iso_free_implicit_surface` on it after calling this function.
#[no_mangle]
pub unsafe extern "C" fn el_iso_query_topology(
    implicit_surface: *mut EL_IsoSurface,
    num_query_points: c_int,
    query_point_coords: *const f64,
) -> *mut EL_IsoQueryTopo {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    if !implicit_surface.is_null() {
        let surf = Box::from_raw(implicit_surface as *mut implicits::MLS<f64>);
        Box::into_raw(Box::new((*surf).query_topo(query_points))) as *mut EL_IsoQueryTopo
    } else {
        std::ptr::null_mut()
    }
}

/// Free memory allocated for the query topology.
#[no_mangle]
pub unsafe extern "C" fn el_iso_free_query_topology(query_topo: *mut EL_IsoQueryTopo) {
    if !query_topo.is_null() {
        let _ = Box::from_raw(query_topo as *mut implicits::QueryTopo<f64>);
    }
}

/// Compute potential. Return 0 on success.
#[no_mangle]
pub unsafe extern "C" fn el_iso_compute_potential(
    query_topo: *const EL_IsoQueryTopo,
    num_query_points: c_int,
    query_point_coords: *const f64,
    out_potential: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let out_potential = std::slice::from_raw_parts_mut(out_potential, num_query_points as usize);

    let surf = &*(query_topo as *const implicits::QueryTopo);

    surf.potential(query_points, out_potential);
    0
}

/// Project the given positions to below the given iso value of the potential field represented by
/// this implicit surface. Only vertices with potentials below the given `iso_value` are actually
/// modified.  The resulting projected `query_point_coords` are guaranteed to be
/// below the given `iso_value` given that this function returns 0. Only query points within
/// this surface's radius are considered for projection. Those projected will be within a
/// `tolerance` of the given `iso_value` but strictly below the `iso_value`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_project_to_below(
    query_topo: *const EL_IsoQueryTopo,
    iso_value: f64,
    tolerance: f64,
    num_query_points: c_int,
    query_point_coords: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts_mut(query_point_coords, num_query_points as usize * 3);
    let query_points: &mut [[f64; 3]] = reinterpret_mut_slice(coords);

    let surf = &*(query_topo as *const implicits::QueryTopo);

    if surf.project_to_below(iso_value, tolerance, query_points) {
        0
    } else {
        1
    }
}

/// Project the given positions to above the given iso value of the potential field represented by
/// this implicit surface. Only vertices with potentials above the given `iso_value` are actually
/// modified.  The resulting projected `query_point_coords` are guaranteed to be
/// above the given `iso_value` given that this function returns 0. Only query points within
/// this surface's radius are considered for projection. Those projected will be within a
/// `tolerance` of the given `iso_value` but strictly above the `iso_value`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_project_to_above(
    query_topo: *const EL_IsoQueryTopo,
    iso_value: f64,
    tolerance: f64,
    num_query_points: c_int,
    query_point_coords: *mut f64,
) -> c_int {
    let coords = std::slice::from_raw_parts_mut(query_point_coords, num_query_points as usize * 3);
    let query_points: &mut [[f64; 3]] = reinterpret_mut_slice(coords);

    let surf = &*(query_topo as *const implicits::QueryTopo);

    if surf.project_to_above(iso_value, tolerance, query_points) {
        0
    } else {
        1
    }
}

// The following macros provide a generic implementation of the C functions computing jacobians
// below.
macro_rules! impl_num_jac_entries {
    ($iso:ident.$fn:ident ()) => {
        (&*($iso as *const implicits::QueryTopo)).$fn() as c_int
    };
}

macro_rules! impl_jac_indices {
    (($rows:ident, $cols:ident)[$num:ident] <- $iso:ident.$fn:ident ()) => {
        let n = $num as usize;
        let rows = std::slice::from_raw_parts_mut($rows, n);
        let cols = std::slice::from_raw_parts_mut($cols, n);
        let surf = &*($iso as *const implicits::QueryTopo);

        for ((out_row, out_col), (r, c)) in rows.iter_mut().zip(cols.iter_mut()).zip(surf.$fn()) {
            *out_row = r as c_int;
            *out_col = c as c_int;
        }
        0
    };
}

macro_rules! impl_jac_values {
    ($vals:ident[$num_vals:ident] <- $iso:ident.$fn:ident ($coords:ident[$num_coords:expr])) => {
        let coords = std::slice::from_raw_parts($coords, $num_coords as usize);
        let query_points: &[[f64; 3]] = reinterpret_slice(coords);
        let vals = std::slice::from_raw_parts_mut($vals, $num_vals as usize);
        let surf = &*($iso as *const implicits::QueryTopo);

        surf.$fn(query_points, vals);
        0
    };
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
pub unsafe extern "C" fn el_iso_num_surface_jacobian_entries(
    query_topo: *const EL_IsoQueryTopo,
) -> c_int {
    impl_num_jac_entries! {
        query_topo.num_surface_jacobian_entries()
    }
}

/// Compute the index structure of the sparse surface Jacobian for the given implicit function.
/// Each computed row-column index pair corresponds to an entry in the sparse Jacobian.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_jacobian_indices(
    query_topo: *const EL_IsoQueryTopo,
    num_entries: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    impl_jac_indices! {
        (rows, cols)[num_entries] <- query_topo.surface_jacobian_indices_iter()
    }
}

/// Compute surface Jacobian values for the given implicit function. The values correspond to the
/// indices provided by `el_iso_surface_jacobian_indices`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_jacobian_values(
    query_topo: *const EL_IsoQueryTopo,
    num_query_points: c_int,
    query_point_coords: *const f64,
    num_entries: c_int,
    values: *mut f64,
) -> c_int {
    impl_jac_values! {
        values[num_entries] <- query_topo.surface_jacobian_values(query_point_coords[num_query_points*3])
    }
}

/// Get the number of entries in the sparse Jacobian of the implicit function with respect to
/// query points.
///
/// This function determines the sizes of the mutable arrays expected in
/// `el_iso_query_jacobian_indices` and `el_iso_query_jacobian_values` functions.
#[no_mangle]
pub unsafe extern "C" fn el_iso_num_query_jacobian_entries(
    query_topo: *const EL_IsoQueryTopo,
) -> c_int {
    impl_num_jac_entries! {
        query_topo.num_query_jacobian_entries()
    }
}

/// Compute the index structure of the sparse query Jacobian for the given implicit function.
/// Each computed row-column index pair corresponds to an entry in the sparse Jacobian.
#[no_mangle]
pub unsafe extern "C" fn el_iso_query_jacobian_indices(
    query_topo: *const EL_IsoQueryTopo,
    num_entries: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    impl_jac_indices! {
        (rows, cols)[num_entries] <- query_topo.query_jacobian_indices_iter()
    }
}

/// Compute query Jacobian for the given implicit function.
#[no_mangle]
pub unsafe extern "C" fn el_iso_query_jacobian_values(
    query_topo: *const EL_IsoQueryTopo,
    num_query_points: c_int,
    query_point_coords: *const f64,
    num_entries: c_int,
    values: *mut f64,
) -> c_int {
    impl_jac_values! {
        values[num_entries] <- query_topo.query_jacobian_values(query_point_coords[num_query_points*3])
    }
}

/// Get the number of entries in the sparse Jacobian of the query positions with respect to
/// sample points.
///
/// This function determines the sizes of the mutable arrays expected in
/// `el_iso_contact_jacobian_indices` and `el_iso_contact_jacobian_values` functions.
#[no_mangle]
pub unsafe extern "C" fn el_iso_num_contact_jacobian_entries(
    query_topo: *const EL_IsoQueryTopo,
) -> c_int {
    impl_num_jac_entries! {
        query_topo.num_contact_jacobian_entries()
    }
}

/// Compute the index structure of the sparse contact Jacobian for the given implicit function.
/// Each computed row-column index pair corresponds to an entry in the sparse Jacobian.
#[no_mangle]
pub unsafe extern "C" fn el_iso_contact_jacobian_indices(
    query_topo: *const EL_IsoQueryTopo,
    num_entries: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    impl_jac_indices! {
        (rows, cols)[num_entries] <- query_topo.contact_jacobian_indices_iter()
    }
}

/// Compute contact Jacobian for the given implicit function.
#[no_mangle]
pub unsafe extern "C" fn el_iso_contact_jacobian_values(
    query_topo: *const EL_IsoQueryTopo,
    num_query_points: c_int,
    query_point_coords: *const f64,
    num_entries: c_int,
    values: *mut f64,
) -> c_int {
    impl_jac_values! {
        values[num_entries] <- query_topo.contact_jacobian_values(query_point_coords[num_query_points*3])
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
#[no_mangle]
pub unsafe extern "C" fn el_iso_num_surface_hessian_product_entries(
    query_topo: *const EL_IsoQueryTopo,
) -> c_int {
    let surf = &*(query_topo as *const implicits::QueryTopo);
    surf.num_surface_hessian_product_entries().unwrap_or(0) as c_int
}

/// Compute the index structure of the surface Hessian product for the given implicit function.
///
/// The Hessian product refers to the product of the full Hessian of the implicit
/// function contracted against a vector multipliers in the number of query points. Thus this
/// Hessian product is an 3n-by-3n matrix, where n is the number of surface vertices.
#[no_mangle]
pub unsafe extern "C" fn el_iso_surface_hessian_product_indices(
    query_topo: *const EL_IsoQueryTopo,
    num_entries: c_int,
    rows: *mut c_int,
    cols: *mut c_int,
) -> c_int {
    let n = num_entries as usize;
    let rows = std::slice::from_raw_parts_mut(rows, n);
    let cols = std::slice::from_raw_parts_mut(cols, n);
    let surf = &*(query_topo as *const implicits::QueryTopo);

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
    query_topo: *const EL_IsoQueryTopo,
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
    let surf = &*(query_topo as *const implicits::QueryTopo);

    match surf.surface_hessian_product_values(query_points, multipliers, vals) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Return the number of query points.
#[no_mangle]
pub unsafe extern "C" fn el_iso_num_query_points(query_topo: *const EL_IsoQueryTopo) -> c_int {
    let surf = &*(query_topo as *const implicits::QueryTopo);
    surf.num_query_points() as c_int
}

/// Populate an array of indices enumerating all query points with non-empty neighbourhoods. The
/// indices for query points with empty neighbourhoods are set to -1.
/// This function is convenient for mapping from query point indices to constraint indices.
/// The `neighbourhood_indices` array is expected be the same size as the number of query points.
#[no_mangle]
pub unsafe extern "C" fn el_iso_neighbourhood_indices(
    query_topo: *const EL_IsoQueryTopo,
    num_query_points: c_int,
    neighbourhood_indices: *mut c_int,
) {
    let surf = &*(query_topo as *const implicits::QueryTopo);
    let neighbourhood_indices =
        std::slice::from_raw_parts_mut(neighbourhood_indices, num_query_points as usize);

    let sizes = surf.neighbourhood_sizes();
    assert_eq!(num_query_points as usize, sizes.len());

    // Initialize indices to -1.
    for idx in neighbourhood_indices.iter_mut() {
        *idx = -1;
    }

    // Index nonempty neighbourhoods
    for (i, (idx, _)) in neighbourhood_indices
        .iter_mut()
        .zip(sizes.iter())
        .filter(|&(_, &s)| s != 0)
        .enumerate()
    {
        *idx = i as c_int;
    }
}

/// Recompute the neighbour cache if invalidated. This is done automatically when computing
/// potential or jacobian. But it needs to be done explicitly to generate the correct Jacobian
/// sparsity pattern because query points are not typically passed when requesting Jacobian
/// indices.
#[no_mangle]
pub unsafe extern "C" fn el_iso_reset(
    query_topo: *mut EL_IsoQueryTopo,
    num_query_points: c_int,
    query_point_coords: *const f64,
) {
    let coords = std::slice::from_raw_parts(query_point_coords, num_query_points as usize * 3);
    let query_points: &[[f64; 3]] = reinterpret_slice(coords);
    let topo = &mut *(query_topo as *mut implicits::QueryTopo);

    topo.reset(query_points);
}

/// Update the implicit surface with new vertex positions for the underlying mesh.
/// The `position_coords` array is expected to have size of 3 times `num_positions`.
#[no_mangle]
pub unsafe extern "C" fn el_iso_update_surface(
    query_topo: *mut EL_IsoQueryTopo,
    num_positions: c_int,
    position_coords: *const f64,
) -> c_int {
    let coords = std::slice::from_raw_parts(position_coords, num_positions as usize * 3);
    let pos: &[[f64; 3]] = reinterpret_slice(coords);
    let topo = &mut *(query_topo as *mut implicits::QueryTopo);
    topo.update_surface(pos.iter().cloned()) as c_int
}

/// Update the implicit surface with an updated maximum allowed additional step size on top of what
/// the radius already allows. In other words, the predetermined radius (determined at
/// initialization) plus this given step is the distance that the underlying implicit surface
/// expects the surface mesh and query points to move, while still maintaining validity of all
/// Jacobians.
#[no_mangle]
pub unsafe extern "C" fn el_iso_update_max_step(query_topo: *mut EL_IsoQueryTopo, max_step: f32) {
    let surf = &mut *(query_topo as *mut implicits::QueryTopo);
    surf.update_max_step(f64::from(max_step));
}

/// Get the current MLS radius.
#[no_mangle]
pub unsafe extern "C" fn el_iso_get_radius(query_topo: *mut EL_IsoQueryTopo) -> f32 {
    let surf = &mut *(query_topo as *mut implicits::QueryTopo);
    surf.radius() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the interface works by constructing a simple potential from a single triangle.
    #[test]
    fn triangle_test() {
        let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let t = vec![0, 1, 2];

        let trimesh = unsafe {
            el_iso_create_trimesh(v.len() as i32, v.as_ptr(), t.len() as i32, t.as_ptr())
        };

        let params = EL_IsoParams {
            tolerance: 1e-5,
            radius_multiplier: 2.0,
            kernel: KernelType::LocalApproximate,
            background_field: BackgroundFieldType::DistanceBased,
            weighted: false,
            sample_type: SampleType::Face,
            max_step: 0.0,
        };

        let surf = unsafe { el_iso_create_implicit_surface(trimesh, params) };

        let q = vec![0.5; 3];

        let mut p = vec![0.0];

        let query_topo = unsafe { el_iso_query_topology(surf, 1, q.as_ptr()) };

        let error = unsafe { el_iso_compute_potential(query_topo, 1, q.as_ptr(), p.as_mut_ptr()) };

        unsafe {
            el_iso_free_trimesh(trimesh);
            el_iso_free_query_topology(query_topo);
        }

        assert_eq!(error, 0);
        assert_eq!(p[0], 0.5);
    }
}
