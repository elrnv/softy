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
use libc::{c_char, c_double, c_longlong, size_t};

use na::{DefaultAllocator, DimName, Point, Real, U3, VectorN};

/// Wrapper around a rust polygon mesh struct.
#[derive(Clone, Debug)]
pub struct PolyMesh(geo::mesh::PolyMesh<f64>);

/// Wrapper around a rust tetmesh struct.
#[derive(Clone, Debug)]
pub struct TetMesh(geo::mesh::TetMesh<f64>);

#[repr(C)]
pub enum AttribLocation {
    Vertex,
    Face,
    Cell,
    FaceVertex,
    CellVertex,
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
    let verts = ptr_to_vec_of_points::<_, U3>((ncoords / 3) as usize, coords);

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
    let verts = ptr_to_vec_of_points::<_, U3>((ncoords / 3) as usize, coords);

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
    ($mesh_type:ident, $mesh:ident,
     $loc:ident, $name:ident, $tuple_size:ident,
     $len:ident, $data:ident: $data_type:ty) => {
        assert!(!$mesh.is_null(), "Can't add attributes to a null pointer.");

        if let Ok(name_str) = CStr::from_ptr($name).to_str() {
            if $tuple_size == 1 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 1,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 2 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 2,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 3 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 3,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 4 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 4,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 5 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 5,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 6 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 6,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 7 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 7,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 8 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 8,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 9 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 9,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 10 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 10,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 11 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 11,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 12 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 12,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 13 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 13,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 14 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 14,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 15 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 15,
                                 $mesh, $len, $data, name_str, $loc);
            } else if $tuple_size == 16 {
                impl_add_attrib!(_impl $mesh_type, $data_type, 16,
                                 $mesh, $len, $data, name_str, $loc);
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
                assert!(i >= 0 && i < $nstrings as i64);
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
    data: *const c_longlong,
) {
    impl_add_attrib!(PolyMesh, mesh, loc, name, tuple_size, nstrings, strings, len, data);
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
    data: *const c_longlong,
) {
    impl_add_attrib!(TetMesh, mesh, loc, name, tuple_size, nstrings, strings, len, data);
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

/// Helper routine for converting C-style data to `nalgebra` `Point`s.
/// This function is similar to the one above but produces points instead of vectors.
unsafe fn ptr_to_vec_of_points<T: Real, D: DimName>(
    npts: usize,
    data_ptr: *const T,
) -> Vec<Point<T, D>>
where
    DefaultAllocator: na::allocator::Allocator<T, D>,
{
    ptr_to_vec_of_vectors(npts, data_ptr)
        .into_iter()
        .map(|v| Point::from_coordinates(v))
        .collect()
}
