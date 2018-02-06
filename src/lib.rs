extern crate alga;
extern crate geometry as geo;
extern crate libc;
extern crate nalgebra as na;

pub mod fem;

/// C API

use geo::mesh::Attrib;
use geo::topology::*;
use std::slice;
use std::ffi::CStr;
use libc::{c_char, c_double, c_float, c_int, c_longlong, c_schar, c_ulonglong, size_t};

use na::{DefaultAllocator, DimName, Real, U3, VectorN};

/// Wrapper around a rust polygon mesh struct.
#[derive(Clone, Debug)]
pub struct PolyMesh(geo::mesh::PolyMesh<f64>);

/// Wrapper around a rust tetmesh struct.
#[derive(Clone, Debug)]
pub struct TetMesh(geo::mesh::TetMesh<f64>);

#[repr(C)]
pub struct PointArray {
    capacity: size_t,
    size: size_t,
    array: *mut [f64; 3],
}

#[repr(C)]
pub struct IndexArray {
    capacity: size_t,
    size: size_t,
    array: *mut size_t,
}

#[no_mangle]
pub unsafe extern "C" fn get_tetmesh_points(mesh: *const TetMesh) -> PointArray {
    assert!(!mesh.is_null());
    let mut pts: Vec<[f64;3]> = Vec::new();

    for &pt in (*mesh).0.vertex_iter() {
        pts.push(pt)
    }

    let arr = PointArray {
        capacity: pts.capacity(),
        size: pts.len(),
        array: pts.as_mut_slice().as_mut_ptr(),
    };

    ::std::mem::forget(pts);

    arr
}

#[no_mangle]
pub unsafe extern "C" fn get_polymesh_points(mesh: *const PolyMesh) -> PointArray {
    assert!(!mesh.is_null());
    let mut pts: Vec<[f64;3]> = Vec::new();

    for &pt in (*mesh).0.vertex_iter() {
        pts.push(pt)
    }

    let arr = PointArray {
        capacity: pts.capacity(),
        size: pts.len(),
        array: pts.as_mut_slice().as_mut_ptr(),
    };

    ::std::mem::forget(pts);

    arr
}

#[no_mangle]
pub unsafe extern "C" fn get_tetmesh_indices(mesh: *const TetMesh) -> IndexArray {
    assert!(!mesh.is_null());
    let mut indices = Vec::new();

    for cell in (*mesh).0.cell_iter() {
        for &idx in cell.iter() {
            indices.push(idx);
        }
    }

    let arr = IndexArray {
        capacity: indices.capacity(),
        size: indices.len(),
        array: indices.as_mut_slice().as_mut_ptr(),
    };

    ::std::mem::forget(indices);

    arr
}

#[no_mangle]
pub unsafe extern "C" fn get_polymesh_indices(mesh: *const PolyMesh) -> IndexArray {
    assert!(!mesh.is_null());
    let mut indices = Vec::new();

    for poly in (*mesh).0.face_iter() {
        indices.push(poly.len());
        for &idx in poly.iter() {
            indices.push(idx);
        }
    }

    let arr = IndexArray {
        capacity: indices.capacity(),
        size: indices.len(),
        array: indices.as_mut_slice().as_mut_ptr(),
    };

    ::std::mem::forget(indices);

    arr
}

#[no_mangle]
pub unsafe extern "C" fn free_point_array(arr: PointArray) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_index_array(arr: IndexArray) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[repr(C)]
pub enum AttribLocation {
    Vertex,
    Face,
    Cell,
    FaceVertex,
    CellVertex,
}

#[repr(C)]
pub enum SimResult {
    Success,
    Failure,
}

#[no_mangle]
pub unsafe extern "C" fn sim(_tetmesh: *mut TetMesh, _polymesh: *mut PolyMesh) -> SimResult {
    println!("Running simulation!");
    SimResult::Success
}

#[no_mangle]
pub unsafe extern "C" fn make_polymesh(
    ncoords: size_t,
    coords: *const c_double,
    nindices: size_t,
    indices: *const size_t,
) -> *mut PolyMesh {
    // check invariants
    assert!(
        ncoords % 3 == 0,
        "Given coordinate array size is not a multiple of 3."
    );

    let indices = slice::from_raw_parts(indices, nindices);
    let verts = ptr_to_vec_of_vectors::<_, U3>((ncoords / 3) as usize, coords)
        .into_iter().map(|x| x.into()).collect();

    let polymesh = Box::new(PolyMesh(geo::mesh::PolyMesh::new(verts, indices)));

    Box::into_raw(polymesh)
}

#[no_mangle]
pub unsafe extern "C" fn make_tetmesh(
    ncoords: size_t,
    coords: *const c_double,
    nindices: size_t,
    indices: *const size_t,
) -> *mut TetMesh {
    // check invariants
    assert!(
        ncoords % 3 == 0,
        "Given coordinate array size is not a multiple of 3."
    );

    let indices = slice::from_raw_parts(indices, nindices).to_vec();
    let verts = ptr_to_vec_of_vectors::<_, U3>((ncoords / 3) as usize, coords)
        .into_iter().map(|x| x.into()).collect();

    let tetmesh = Box::new(TetMesh(geo::mesh::TetMesh::new(verts, indices)));

    Box::into_raw(tetmesh)
}

/// Helper macro for converting C-style data to a `Vec<[T;$tuple_size]>`
/// `data` is the const pointer of to the data to be copied.
/// `size` is the number of elements returned in the vector.
macro_rules! ptr_to_vec_of_arrays {
    ($size:ident, $data:ident, $ty:ty, 1) => {
        {
            slice::from_raw_parts($data, $size).to_vec()
        }
    };
    ($size:ident, $data:ident, $ty:ty, $tuple_size:expr) => {
        {
            assert!($size % $tuple_size == 0, "Wrong tuple size for array.");
            let nelem = $size/$tuple_size;
            let mut data = Vec::with_capacity(nelem);
            for i in 0..nelem as isize {
                let mut s: [$ty; $tuple_size] = ::std::mem::uninitialized();
                for k in 0..$tuple_size {
                    s[k] = (*$data.offset($tuple_size * i + k as isize)).clone();
                }

                data.push( s );
            }
            data
        }
    }
}

macro_rules! impl_add_attrib {
    (_impl PolyMesh, $data_type:ty, $tuple_size:expr, $mesh:ident,
     $len:ident, $data:ident, $name:ident, $loc:ident) => {
        let vec = ptr_to_vec_of_arrays!($len, $data, $data_type, $tuple_size);
        impl_add_attrib!(_impl_surface $mesh, $loc, $name, vec);
    };
    (_impl TetMesh, $data_type:ty, $tuple_size:expr, $mesh:ident,
     $len:ident, $data:ident, $name:ident, $loc:ident) => {
        let vec = ptr_to_vec_of_arrays!($len, $data, $data_type, $tuple_size);
        impl_add_attrib!(_impl_volume $mesh, $loc, $name, vec);
    };
    // Surface type meshes like tri- or quad-meshes typically have face attributes but no
    // cell attributes.
    (_impl_surface $mesh:ident, $loc:ident, $name:ident, $vec:ident) => {
        {
            match $loc {
                AttribLocation::Vertex => {
                    (*$mesh).0.add_attrib_data::<_,VertexIndex>($name, $vec).ok();
                },
                AttribLocation::Face => {
                    (*$mesh).0.add_attrib_data::<_,FaceIndex>($name, $vec).ok();
                },
                AttribLocation::FaceVertex => {
                    (*$mesh).0.add_attrib_data::<_,FaceVertexIndex>($name, $vec).ok();
                },
                _ => (),
            };
        }
    };
    // Volume type meshes like tet and hex meshes have cell attributes.
    (_impl_volume $mesh:ident, $loc:ident, $name:ident, $vec:ident) => {
        {
            match $loc {
                AttribLocation::Vertex => {
                    (*$mesh).0.add_attrib_data::<_,VertexIndex>($name, $vec).ok();
                },
                AttribLocation::Cell=> {
                    (*$mesh).0.add_attrib_data::<_,CellIndex>($name, $vec).ok();
                },
                AttribLocation::CellVertex => {
                    (*$mesh).0.add_attrib_data::<_,CellVertexIndex>($name, $vec).ok();
                },
                _ => (),
            };
        }
    };
    // Main implemnetation of the add attribute function.
    ($mtype:ident, $mesh:ident,
     $loc:ident, $name:ident, $tuple_size:ident,
     $n:ident, $data:ident: $dty:ty) => {
        assert!(!$mesh.is_null(), "Can't add attributes to a null pointer.");

        if let Ok(name_str) = CStr::from_ptr($name).to_str() {
            match $tuple_size {
                1 => {impl_add_attrib!(_impl $mtype, $dty, 1, $mesh, $n, $data, name_str, $loc);},
                2 => {impl_add_attrib!(_impl $mtype, $dty, 2, $mesh, $n, $data, name_str, $loc);},
                3 => {impl_add_attrib!(_impl $mtype, $dty, 3, $mesh, $n, $data, name_str, $loc);},
                4 => {impl_add_attrib!(_impl $mtype, $dty, 4, $mesh, $n, $data, name_str, $loc);},
                5 => {impl_add_attrib!(_impl $mtype, $dty, 5, $mesh, $n, $data, name_str, $loc);},
                6 => {impl_add_attrib!(_impl $mtype, $dty, 6, $mesh, $n, $data, name_str, $loc);},
                7 => {impl_add_attrib!(_impl $mtype, $dty, 7, $mesh, $n, $data, name_str, $loc);},
                8 => {impl_add_attrib!(_impl $mtype, $dty, 8, $mesh, $n, $data, name_str, $loc);},
                9 => {impl_add_attrib!(_impl $mtype, $dty, 9, $mesh, $n, $data, name_str, $loc);},
                10 => {impl_add_attrib!(_impl $mtype, $dty, 10, $mesh, $n, $data, name_str, $loc);},
                11 => {impl_add_attrib!(_impl $mtype, $dty, 11, $mesh, $n, $data, name_str, $loc);},
                12 => {impl_add_attrib!(_impl $mtype, $dty, 12, $mesh, $n, $data, name_str, $loc);},
                13 => {impl_add_attrib!(_impl $mtype, $dty, 13, $mesh, $n, $data, name_str, $loc);},
                14 => {impl_add_attrib!(_impl $mtype, $dty, 14, $mesh, $n, $data, name_str, $loc);},
                15 => {impl_add_attrib!(_impl $mtype, $dty, 15, $mesh, $n, $data, name_str, $loc);},
                16 => {impl_add_attrib!(_impl $mtype, $dty, 16, $mesh, $n, $data, name_str, $loc);},
                _ => (),
            }
        }
    };
    // Helpers for the implementation for string attributes below.
    (_impl_str PolyMesh, $mesh:ident, $loc:ident, $name:ident, $vec:ident) => {
        impl_add_attrib!(_impl_surface $mesh, $loc, $name, $vec);
    };
    (_impl_str TetMesh, $mesh:ident, $loc:ident, $name:ident, $vec:ident) => {
        impl_add_attrib!(_impl_volume $mesh, $loc, $name, $vec);
    };
    // Implementation for string attributes
    ($mesh_type:ident, $mesh:ident,
     $loc:ident, $name:ident, $tuple_size:ident,
     $nstrings:ident, $strings:ident, $len:ident, $data:ident) => {
        assert!(!$mesh.is_null(), "Can't add attributes to a null pointer.");
        assert!($tuple_size == 1, "Only 1 dimensional string attributes currently supported.");

        if let Ok(name_str) = CStr::from_ptr($name).to_str() {
            let indices = slice::from_raw_parts($data, $len);

            // TODO: Storing owned srings is expensive for large meshes when strings are shared. This
            // should be refactored to store shared strings, which may need a refactor of the attribute
            // system.

            let mut vec = Vec::new();
            for &i in indices {
                assert!(i < $nstrings as u64);
                let cstr = *$strings.offset(i as isize);
                if let Ok(s) = CStr::from_ptr(cstr).to_str() {
                    vec.push(String::from(s))
                }
            }

            impl_add_attrib!(_impl_str $mesh_type, $mesh, $loc, name_str, vec);
        }
    }
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_polymesh_attrib_f32(
    mesh: *mut PolyMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_float,
) {
    impl_add_attrib!(PolyMesh, mesh, loc, name, tuple_size, len, data: c_float);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_polymesh_attrib_f64(
    mesh: *mut PolyMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_double,
) {
    impl_add_attrib!(PolyMesh, mesh, loc, name, tuple_size, len, data: c_double);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_polymesh_attrib_i8(
    mesh: *mut PolyMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_schar,
) {
    impl_add_attrib!(PolyMesh, mesh, loc, name, tuple_size, len, data: c_schar);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_polymesh_attrib_i32(
    mesh: *mut PolyMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_int,
) {
    impl_add_attrib!(PolyMesh, mesh, loc, name, tuple_size, len, data: c_int);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_polymesh_attrib_i64(
    mesh: *mut PolyMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_longlong,
) {
    impl_add_attrib!(PolyMesh, mesh, loc, name, tuple_size, len, data: c_longlong);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_tetmesh_attrib_f32(
    mesh: *mut TetMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_float,
) {
    impl_add_attrib!(TetMesh, mesh, loc, name, tuple_size, len, data: c_float);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_tetmesh_attrib_f64(
    mesh: *mut TetMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_double,
) {
    impl_add_attrib!(TetMesh, mesh, loc, name, tuple_size, len, data: c_double);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_tetmesh_attrib_i8(
    mesh: *mut TetMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_schar,
) {
    impl_add_attrib!(TetMesh, mesh, loc, name, tuple_size, len, data: c_schar);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_tetmesh_attrib_i32(
    mesh: *mut TetMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_int,
) {
    impl_add_attrib!(TetMesh, mesh, loc, name, tuple_size, len, data: c_int);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_tetmesh_attrib_i64(
    mesh: *mut TetMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    len: size_t,
    data: *const c_longlong,
) {
    impl_add_attrib!(TetMesh, mesh, loc, name, tuple_size, len, data: c_longlong);
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_polymesh_attrib_str(
    mesh: *mut PolyMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    nstrings: size_t,
    strings: *const *const c_char,
    len: size_t,
    data: *const c_ulonglong,
) {
    impl_add_attrib!(
        PolyMesh,
        mesh,
        loc,
        name,
        tuple_size,
        nstrings,
        strings,
        len,
        data
    );
}

/// If the given mesh is null, this function will panic.
#[no_mangle]
pub unsafe extern "C" fn add_tetmesh_attrib_str(
    mesh: *mut TetMesh,
    loc: AttribLocation,
    name: *const c_char,
    tuple_size: size_t,
    nstrings: size_t,
    strings: *const *const c_char,
    len: size_t,
    data: *const c_ulonglong,
) {
    impl_add_attrib!(
        TetMesh,
        mesh,
        loc,
        name,
        tuple_size,
        nstrings,
        strings,
        len,
        data
    );
}

/// Helper routine for converting C-style data to `nalgebra` `VectorN`s.
/// `npts` is the number of vectors to output, which means that `data_ptr` must point to an array of
/// `n*D::dim()` doubles.
unsafe fn ptr_to_vec_of_vectors<T: Real, D: DimName>(
    npts: usize,
    data_ptr: *const T,
) -> Vec<VectorN<T, D>>
where
    DefaultAllocator: na::allocator::Allocator<T, D>,
{
    let mut data = Vec::with_capacity(npts);
    for i in 0..npts as isize {
        data.push(VectorN::from_fn(|r, _| {
            *data_ptr.offset(D::dim() as isize * i + r as isize)
        }));
    }
    data
}

///// Helper routine for converting C-style data to `nalgebra` `Point`s.
///// This function is similar to the one above but produces points instead of vectors.
//unsafe fn ptr_to_vec_of_points<T: Real, D: DimName>(
//    npts: usize,
//    data_ptr: *const T,
//) -> Vec<Point<T, D>>
//where
//    DefaultAllocator: na::allocator::Allocator<T, D>,
//{
//    ptr_to_vec_of_vectors(npts, data_ptr)
//        .into_iter()
//        .map(|v| Point::from_coordinates(v))
//        .collect()
//}
