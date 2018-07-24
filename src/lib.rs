extern crate geometry as geo;
extern crate libc;
extern crate nalgebra as na;

mod api;
mod mls;

use geo::mesh::attrib;
use geo::mesh::topology as topo;
use geo::mesh::Attrib;
use libc::{c_char, c_double, c_float, c_int, c_longlong, c_schar, c_void, size_t};
use std::any::TypeId;
use std::collections::hash_map::Iter;
use std::ffi::{CStr, CString};
use std::slice;

/// Wrapper around a rust polygon mesh struct.
#[derive(Clone, Debug)]
pub struct PolyMesh {
    pub mesh: geo::mesh::PolyMesh<f64>,
}

/// Wrapper around a rust tetmesh struct.
#[derive(Clone, Debug)]
pub struct TetMesh {
    pub mesh: geo::mesh::TetMesh<f64>,
}

impl From<api::CookResult> for CookResult {
    fn from(res: api::CookResult) -> CookResult {
        match res {
            api::CookResult::Success(msg) => CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: CookResultTag::Success,
            },
            api::CookResult::Warning(msg) => CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: CookResultTag::Warning,
            },
            api::CookResult::Error(msg) => CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: CookResultTag::Error,
            },
        }
    }
}

/// Main entry point from Houdini SOP.
/// The purpose of this function is to cleanup the inputs for use in Rust code.
#[no_mangle]
pub unsafe extern "C" fn cook(
    samplemesh: *mut PolyMesh,
    polymesh: *mut PolyMesh,
    params: api::Params,
    interrupt_checker: *mut c_void,
    check_interrupt: Option<extern "C" fn(*mut c_void) -> bool>,
) -> CookResult {
    let interrupt_ref = &mut *interrupt_checker; // conversion needed sicne *mut c_void is not Send
    let interrupt_callback = || match check_interrupt {
        Some(cb) => cb(interrupt_ref as *mut c_void),
        None => true,
    };
    api::cook(
        if samplemesh.is_null() {
            None
        } else {
            Some(&mut (*samplemesh).mesh)
        },
        if polymesh.is_null() {
            None
        } else {
            Some(&mut (*polymesh).mesh)
        },
        params.into(),
        interrupt_callback,
    ).into()
}

#[repr(C)]
pub enum CookResultTag {
    Success,
    Warning,
    Error,
}

/// Result for C interop.
#[repr(C)]
pub struct CookResult {
    message: *mut c_char,
    tag: CookResultTag,
}

#[no_mangle]
pub unsafe extern "C" fn free_result(res: CookResult) {
    let _ = CString::from_raw(res.message);
}

pub enum VertexIndex {}
pub enum FaceIndex {}
pub enum CellIndex {}
pub enum FaceVertexIndex {}
pub enum CellVertexIndex {}

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
    let mut pts: Vec<[f64; 3]> = Vec::new();

    for &pt in (*mesh).mesh.vertex_positions().iter() {
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
    let mut pts: Vec<[f64; 3]> = Vec::new();

    for &pt in (*mesh).mesh.vertex_iter() {
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

    for cell in (*mesh).mesh.cell_iter() {
        for &idx in cell.get().iter() {
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

/// Polygon mesh indices is a contiguous set of polygon indices, each in the form:
/// `n, i_1, i_2, ..., i_n` where `n` is the number of sides on a polygon.
#[no_mangle]
pub unsafe extern "C" fn get_polymesh_indices(mesh: *const PolyMesh) -> IndexArray {
    assert!(!mesh.is_null());
    let mut indices = Vec::new();

    for poly in (*mesh).mesh.face_iter() {
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

// Required for cbindgen to produce these opaque structs.
mod missing_structs {
    #[allow(dead_code)]
    struct AttribIter;

    #[allow(dead_code)]
    struct CString;
}

pub enum AttribIter<'a> {
    Vertex(Iter<'a, String, attrib::Attribute<topo::VertexIndex>>),
    Face(Iter<'a, String, attrib::Attribute<topo::FaceIndex>>),
    Cell(Iter<'a, String, attrib::Attribute<topo::CellIndex>>),
    FaceVertex(Iter<'a, String, attrib::Attribute<topo::FaceVertexIndex>>),
    CellVertex(Iter<'a, String, attrib::Attribute<topo::CellVertexIndex>>),
    None,
}

// Another workaround for ffi. To ensure the lifetime can be elided in AttribIter in the return
// results for the two functions below, we need to pass a parameter with a lifetime parameter.
// Otherwise cbindgen doesn't know how to ignore lifetime parameters on returned types for some
// reason.
pub struct Dummy<'a> {
    p: ::std::marker::PhantomData<&'a u32>,
}

#[no_mangle]
pub unsafe extern "C" fn tetmesh_attrib_iter(
    mesh_ptr: *mut TetMesh,
    loc: AttribLocation,
    _d: *const Dummy,
) -> *mut AttribIter {
    assert!(!mesh_ptr.is_null());

    let mesh = &mut (*mesh_ptr).mesh;

    let iter = Box::new(match loc {
        AttribLocation::Vertex => {
            AttribIter::Vertex(mesh.attrib_dict::<topo::VertexIndex>().iter())
        }
        AttribLocation::Cell => AttribIter::Cell(mesh.attrib_dict::<topo::CellIndex>().iter()),
        AttribLocation::CellVertex => {
            AttribIter::CellVertex(mesh.attrib_dict::<topo::CellVertexIndex>().iter())
        }
        _ => return ::std::ptr::null::<AttribIter>() as *mut AttribIter,
    });

    Box::into_raw(iter)
}

#[no_mangle]
pub unsafe extern "C" fn polymesh_attrib_iter(
    mesh_ptr: *mut PolyMesh,
    loc: AttribLocation,
    _d: *const Dummy,
) -> *mut AttribIter {
    assert!(!mesh_ptr.is_null());

    let mesh = &mut (*mesh_ptr).mesh;

    let iter = Box::new(match loc {
        AttribLocation::Vertex => {
            AttribIter::Vertex(mesh.attrib_dict::<topo::VertexIndex>().iter())
        }
        AttribLocation::Face => AttribIter::Face(mesh.attrib_dict::<topo::FaceIndex>().iter()),
        AttribLocation::FaceVertex => {
            AttribIter::FaceVertex(mesh.attrib_dict::<topo::FaceVertexIndex>().iter())
        }
        _ => return ::std::ptr::null::<AttribIter>() as *mut AttribIter,
    });

    Box::into_raw(iter)
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_iter(data: *mut AttribIter) {
    let _ = Box::from_raw(data);
}

/// Wrapper around the `attrib::Attribute` struct to eliminate generics for ffi.
#[derive(Debug)]
#[repr(C)]
pub enum AttribData<'a> {
    Vertex(&'a attrib::Attribute<topo::VertexIndex>),
    Face(&'a attrib::Attribute<topo::FaceIndex>),
    Cell(&'a attrib::Attribute<topo::CellIndex>),
    FaceVertex(&'a attrib::Attribute<topo::FaceVertexIndex>),
    CellVertex(&'a attrib::Attribute<topo::CellVertexIndex>),
    None,
}

/// Opaque type to store data about a particular attribute. This struct owns the string it
/// contains thus it must be freed when done.
#[derive(Debug)]
pub struct Attribute<'a> {
    name: CString,
    data: AttribData<'a>,
}

/// Produces an `Attribute` struct that references the next available attribute data.
#[no_mangle]
pub unsafe extern "C" fn attrib_iter_next(iter_ptr: *mut AttribIter) -> *mut Attribute {
    assert!(!iter_ptr.is_null());

    let null = ::std::ptr::null::<Attribute>() as *mut Attribute;

    match *iter_ptr {
        AttribIter::Vertex(ref mut iter) => iter.next().map_or(null, |(k, v)| {
            Box::into_raw(Box::new(Attribute {
                name: CString::new(k.as_str()).unwrap(),
                data: AttribData::Vertex(v),
            }))
        }),
        AttribIter::Face(ref mut iter) => iter.next().map_or(null, |(k, v)| {
            Box::into_raw(Box::new(Attribute {
                name: CString::new(k.as_str()).unwrap(),
                data: AttribData::Face(v),
            }))
        }),
        AttribIter::Cell(ref mut iter) => iter.next().map_or(null, |(k, v)| {
            Box::into_raw(Box::new(Attribute {
                name: CString::new(k.as_str()).unwrap(),
                data: AttribData::Cell(v),
            }))
        }),
        AttribIter::FaceVertex(ref mut iter) => iter.next().map_or(null, |(k, v)| {
            Box::into_raw(Box::new(Attribute {
                name: CString::new(k.as_str()).unwrap(),
                data: AttribData::FaceVertex(v),
            }))
        }),
        AttribIter::CellVertex(ref mut iter) => iter.next().map_or(null, |(k, v)| {
            Box::into_raw(Box::new(Attribute {
                name: CString::new(k.as_str()).unwrap(),
                data: AttribData::CellVertex(v),
            }))
        }),
        AttribIter::None => null,
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_attribute(attrib: *mut Attribute) {
    let _ = Box::from_raw(attrib);
}

#[no_mangle]
pub unsafe extern "C" fn attrib_name(data: *const Attribute) -> *const c_char {
    if data.is_null() {
        CString::new("").unwrap().as_ptr()
    } else {
        (*data).name.as_ptr()
    }
}

#[repr(C)]
pub enum DataType {
    I8,
    I32,
    I64,
    F32,
    F64,
    Str,
    Unsupported,
}

macro_rules! impl_supported_types {
    ($var:ident, $type:ty, $($array_sizes:expr),*) => {
        $var == TypeId::of::<$type>() ||
            $($var == TypeId::of::<[$type;$array_sizes]>())||*
    }
}

macro_rules! impl_supported_sizes {
    ($var:ident, $($types:ty),*) => {
        $($var == TypeId::of::<$types>())||*
    };
    ($var:ident, $array_size:expr, $($types:ty),*) => {
        $($var == TypeId::of::<[$types;$array_size]>())||*
    }
}

macro_rules! cast_to_vec {
    ($type:ident, $data:ident) => {
        (*$data).into_vec::<$type>().unwrap_or(Vec::new())
    };
    ($type:ident, $data:ident, $tuple_size:expr) => {
        (*$data)
            .into_vec::<[$type; $tuple_size]>()
            .unwrap_or(Vec::new())
            .iter()
            .flat_map(|x| x.iter().cloned())
            .collect()
    };
}

pub fn attrib_type_id<I>(attrib: &attrib::Attribute<I>) -> DataType {
    match attrib.type_id() {
        x if impl_supported_types!(
            x, i8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) =>
        {
            DataType::I8
        }
        x if impl_supported_types!(
            x, i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) =>
        {
            DataType::I32
        }
        x if impl_supported_types!(
            x, i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) =>
        {
            DataType::I64
        }
        x if impl_supported_types!(
            x, f32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) =>
        {
            DataType::F32
        }
        x if impl_supported_types!(
            x, f64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) =>
        {
            DataType::F64
        }
        x if impl_supported_types!(
            x, String, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ) =>
        {
            DataType::Str
        }
        _ => DataType::Unsupported,
    }
}

pub fn attrib_flat_array<I, T: 'static + Clone>(attrib: &attrib::Attribute<I>) -> (Vec<T>, usize) {
    let tuple_size = match attrib.type_id() {
        x if impl_supported_sizes!(x, i8, i32, i64, f32, f64, String) => 1,
        x if impl_supported_sizes!(x, 1, i8, i32, i64, f32, f64, String) => 1,
        x if impl_supported_sizes!(x, 2, i8, i32, i64, f32, f64, String) => 2,
        x if impl_supported_sizes!(x, 3, i8, i32, i64, f32, f64, String) => 3,
        x if impl_supported_sizes!(x, 4, i8, i32, i64, f32, f64, String) => 4,
        x if impl_supported_sizes!(x, 5, i8, i32, i64, f32, f64, String) => 5,
        x if impl_supported_sizes!(x, 6, i8, i32, i64, f32, f64, String) => 6,
        x if impl_supported_sizes!(x, 7, i8, i32, i64, f32, f64, String) => 7,
        x if impl_supported_sizes!(x, 8, i8, i32, i64, f32, f64, String) => 8,
        x if impl_supported_sizes!(x, 9, i8, i32, i64, f32, f64, String) => 9,
        x if impl_supported_sizes!(x, 10, i8, i32, i64, f32, f64, String) => 10,
        x if impl_supported_sizes!(x, 11, i8, i32, i64, f32, f64, String) => 11,
        x if impl_supported_sizes!(x, 12, i8, i32, i64, f32, f64, String) => 12,
        x if impl_supported_sizes!(x, 13, i8, i32, i64, f32, f64, String) => 13,
        x if impl_supported_sizes!(x, 14, i8, i32, i64, f32, f64, String) => 14,
        x if impl_supported_sizes!(x, 15, i8, i32, i64, f32, f64, String) => 15,
        x if impl_supported_sizes!(x, 16, i8, i32, i64, f32, f64, String) => 16,
        _ => 0,
    };
    let flat_vec = match tuple_size {
        1 => cast_to_vec!(T, attrib),
        2 => cast_to_vec!(T, attrib, 2),
        3 => cast_to_vec!(T, attrib, 3),
        4 => cast_to_vec!(T, attrib, 4),
        5 => cast_to_vec!(T, attrib, 5),
        6 => cast_to_vec!(T, attrib, 6),
        7 => cast_to_vec!(T, attrib, 7),
        8 => cast_to_vec!(T, attrib, 8),
        9 => cast_to_vec!(T, attrib, 9),
        10 => cast_to_vec!(T, attrib, 10),
        11 => cast_to_vec!(T, attrib, 11),
        12 => cast_to_vec!(T, attrib, 12),
        13 => cast_to_vec!(T, attrib, 13),
        14 => cast_to_vec!(T, attrib, 14),
        15 => cast_to_vec!(T, attrib, 15),
        16 => cast_to_vec!(T, attrib, 16),
        _ => Vec::new(),
    };

    (flat_vec, tuple_size)
}

#[no_mangle]
pub unsafe extern "C" fn attrib_data_type(attrib: *const Attribute) -> DataType {
    if attrib.is_null() {
        DataType::Unsupported
    } else {
        match (*attrib).data {
            AttribData::Vertex(a) => attrib_type_id(a),
            AttribData::Face(a) => attrib_type_id(a),
            AttribData::Cell(a) => attrib_type_id(a),
            AttribData::FaceVertex(a) => attrib_type_id(a),
            AttribData::CellVertex(a) => attrib_type_id(a),
            AttribData::None => DataType::Unsupported,
        }
    }
}

#[repr(C)]
pub struct AttribArrayI8 {
    capacity: size_t,
    size: size_t,
    tuple_size: size_t,
    array: *mut i8,
}
#[repr(C)]
pub struct AttribArrayI32 {
    capacity: size_t,
    size: size_t,
    tuple_size: size_t,
    array: *mut i32,
}
#[repr(C)]
pub struct AttribArrayI64 {
    capacity: size_t,
    size: size_t,
    tuple_size: size_t,
    array: *mut i64,
}
#[repr(C)]
pub struct AttribArrayF32 {
    capacity: size_t,
    size: size_t,
    tuple_size: size_t,
    array: *mut f32,
}
#[repr(C)]
pub struct AttribArrayF64 {
    capacity: size_t,
    size: size_t,
    tuple_size: size_t,
    array: *mut f64,
}

#[repr(C)]
pub struct AttribArrayStr {
    capacity: size_t,
    size: size_t,
    tuple_size: size_t,
    array: *mut *mut c_char,
}
macro_rules! impl_get_attrib_data {
    (_impl_make_array AttribArrayStr, $vec:ident, $tuple_size:ident) => {{
        let mut vec: Vec<*mut c_char> = $vec
            .iter()
            .map(|x: &String| CString::new(x.as_str()).unwrap().into_raw())
            .collect();

        let arr = AttribArrayStr {
            capacity: vec.capacity(),
            size: vec.len(),
            tuple_size: $tuple_size,
            array: vec.as_mut_ptr(),
        };

        ::std::mem::forget(vec);

        arr
    }};
    (_impl_make_array $array_name:ident, $vec:ident, $tuple_size:ident) => {{
        let arr = $array_name {
            capacity: $vec.capacity(),
            size: $vec.len(),
            tuple_size: $tuple_size,
            array: $vec.as_mut_ptr(),
        };

        ::std::mem::forget($vec);

        arr
    }};
    ($array_name:ident, $attrib_data:ident) => {{
        #[allow(unused_mut)] // compiler gets confused here, suppress the warning.
        let (mut vec, tuple_size) = if $attrib_data.is_null() {
            (Vec::new(), 0)
        } else {
            match (*$attrib_data).data {
                AttribData::Vertex(data) => attrib_flat_array(data),
                AttribData::Face(data) => attrib_flat_array(data),
                AttribData::Cell(data) => attrib_flat_array(data),
                AttribData::FaceVertex(data) => attrib_flat_array(data),
                AttribData::CellVertex(data) => attrib_flat_array(data),
                AttribData::None => (Vec::new(), 0),
            }
        };

        impl_get_attrib_data!(_impl_make_array $array_name, vec, tuple_size)
    }};
}

#[no_mangle]
pub unsafe extern "C" fn attrib_data_i8(attrib: *mut Attribute) -> AttribArrayI8 {
    impl_get_attrib_data!(AttribArrayI8, attrib)
}
#[no_mangle]
pub unsafe extern "C" fn attrib_data_i32(attrib: *mut Attribute) -> AttribArrayI32 {
    impl_get_attrib_data!(AttribArrayI32, attrib)
}
#[no_mangle]
pub unsafe extern "C" fn attrib_data_i64(attrib: *mut Attribute) -> AttribArrayI64 {
    impl_get_attrib_data!(AttribArrayI64, attrib)
}
#[no_mangle]
pub unsafe extern "C" fn attrib_data_f32(attrib: *mut Attribute) -> AttribArrayF32 {
    impl_get_attrib_data!(AttribArrayF32, attrib)
}
#[no_mangle]
pub unsafe extern "C" fn attrib_data_f64(attrib: *mut Attribute) -> AttribArrayF64 {
    impl_get_attrib_data!(AttribArrayF64, attrib)
}
#[no_mangle]
pub unsafe extern "C" fn attrib_data_str(attrib: *mut Attribute) -> AttribArrayStr {
    impl_get_attrib_data!(AttribArrayStr, attrib)
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_data_i8(arr: AttribArrayI8) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_data_i32(arr: AttribArrayI32) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_data_i64(arr: AttribArrayI64) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_data_f32(arr: AttribArrayF32) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_data_f64(arr: AttribArrayF64) {
    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_attrib_data_str(arr: AttribArrayStr) {
    for i in 0..arr.size as isize {
        let _ = CString::from_raw(*arr.array.offset(i));
    }

    let _ = Vec::from_raw_parts(arr.array, arr.size, arr.capacity);
}

#[no_mangle]
pub unsafe extern "C" fn free_iter(iter: *mut AttribIter) {
    if !iter.is_null() {
        let _ = Box::from_raw(iter);
    }
}

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
    let verts = ptr_to_vec_of_triples((ncoords / 3) as usize, coords);

    let polymesh = Box::new(PolyMesh {
        mesh: geo::mesh::PolyMesh::new(verts, indices),
    });

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
    let verts = ptr_to_vec_of_triples((ncoords / 3) as usize, coords);

    let tetmesh = Box::new(TetMesh {
        mesh: geo::mesh::TetMesh::new(verts, indices),
    });

    Box::into_raw(tetmesh)
}

#[no_mangle]
pub unsafe extern "C" fn free_tetmesh(mesh: *mut TetMesh) {
    if !mesh.is_null() {
        let _ = Box::from_raw(mesh);
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_polymesh(mesh: *mut PolyMesh) {
    if !mesh.is_null() {
        let _ = Box::from_raw(mesh);
    }
}

/// Helper macro for converting C-style data to a `Vec<[T;$tuple_size]>`
/// `data` is the const pointer of to the data to be copied.
/// `size` is the number of elements returned in the vector.
macro_rules! ptr_to_vec_of_arrays {
    ($size:ident, $data:ident, $ty:ty) => {{
        slice::from_raw_parts($data, $size).to_vec()
    }};
    ($size:ident, $data:ident, $ty:ty, $tuple_size:expr) => {{
        assert!($size % $tuple_size == 0, "Wrong tuple size for array.");
        let nelem = $size / $tuple_size;
        let mut data = Vec::with_capacity(nelem);
        for i in 0..nelem as isize {
            let mut s: [$ty; $tuple_size] = ::std::mem::uninitialized();
            for k in 0..$tuple_size {
                s[k] = (*$data.offset($tuple_size * i + k as isize)).clone();
            }

            data.push(s);
        }
        data
    }};
}

macro_rules! impl_add_attrib {
    (_impl PolyMesh, $data_type:ty, $mesh:ident,
     $len:ident, $data:ident, $name:ident, $loc:ident) => {
        let vec = ptr_to_vec_of_arrays!($len, $data, $data_type);
        impl_add_attrib!(_impl_surface $mesh, $loc, $name, vec);
    };
    (_impl TetMesh, $data_type:ty, $mesh:ident,
     $len:ident, $data:ident, $name:ident, $loc:ident) => {
        let vec = ptr_to_vec_of_arrays!($len, $data, $data_type);
        impl_add_attrib!(_impl_volume $mesh, $loc, $name, vec);
    };
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
                    (*$mesh).mesh.add_attrib_data::<_,topo::VertexIndex>($name, $vec).ok();
                },
                AttribLocation::Face => {
                    (*$mesh).mesh.add_attrib_data::<_,topo::FaceIndex>($name, $vec).ok();
                },
                AttribLocation::FaceVertex => {
                    (*$mesh).mesh.add_attrib_data::<_,topo::FaceVertexIndex>($name, $vec).ok();
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
                    (*$mesh).mesh.add_attrib_data::<_,topo::VertexIndex>($name, $vec).ok();
                },
                AttribLocation::Cell=> {
                    (*$mesh).mesh.add_attrib_data::<_,topo::CellIndex>($name, $vec).ok();
                },
                AttribLocation::CellVertex => {
                    (*$mesh).mesh.add_attrib_data::<_,topo::CellVertexIndex>($name, $vec).ok();
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
                1 => {impl_add_attrib!(_impl $mtype, $dty, $mesh, $n, $data, name_str, $loc);},
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
        if $tuple_size != 1 {
            return;
        }

        if let Ok(name_str) = CStr::from_ptr($name).to_str() {
            let indices = slice::from_raw_parts($data, $len);

            // TODO: Storing owned srings is expensive for large meshes when strings are shared.
            // This should be refactored to store shared strings, which may need a refactor of the
            // attribute system.

            let mut vec = Vec::new();
            for &i in indices {
                if i >= 0 {
                    assert!(i < $nstrings as i64);
                    let cstr = *$strings.offset(i as isize);
                    if let Ok(s) = CStr::from_ptr(cstr).to_str() {
                        vec.push(String::from(s))
                    }
                } else {
                    vec.push(String::from(""))
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

/// Helper routine for converting C-style data to `[T;3]`s.
/// `num` is the number of arrays to output, which means that `data_ptr` must point to an array of
/// `n*3` elements.
unsafe fn ptr_to_vec_of_triples<T: Copy>(num_elem: usize, data_ptr: *const T) -> Vec<[T; 3]> {
    let mut data = Vec::with_capacity(num_elem);
    for i in 0..num_elem as isize {
        data.push([
            *data_ptr.offset(3 * i),
            *data_ptr.offset(3 * i + 1),
            *data_ptr.offset(3 * i + 2),
        ]);
    }
    data
}
