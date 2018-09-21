//!
//! This module defines utility functions for translating C types to rust types.
//!

pub use libc::c_void;
use std::ffi::CString;
use std::ptr::NonNull;
use cffi;

//
// Translate interrupt callback
//

/// Utility to cast the void pointer to the interrupt checker function a valid Rust type.
pub unsafe fn interrupt_callback(checker: *mut c_void,
                                 check_interrupt: Option<extern "C" fn(*const c_void) -> bool>
                                 ) -> impl Fn() -> bool {
    let interrupt_ref = &*checker; // conversion needed since *mut c_void is not Send
    move || match check_interrupt {
        Some(cb) => cb(interrupt_ref as *const c_void),
        None => true,
    }
}

//
// Translate result type
//

/// The Rust version of the cook result enum.
pub enum CookResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<CookResult> for cffi::CookResult {
    fn from(res: CookResult) -> cffi::CookResult {
        match res {
            CookResult::Success(msg) => cffi::CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: cffi::CookResultTag::Success,
            },
            CookResult::Warning(msg) => cffi::CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: cffi::CookResultTag::Warning,
            },
            CookResult::Error(msg) => cffi::CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: cffi::CookResultTag::Error,
            },
        }
    }
}

//
// Translate mesh data structures
//

//pub trait Mesh {
//    type MeshType;
//
//    fn mesh_mut(&mut self) -> &mut Self::MeshType;
//    fn mesh_ref(&self) -> &Self::MeshType;
//    fn mesh(self) -> Self::MeshType;
//}
//
////TODO: the following can be implemented automatically with a procedural derive macro.
//impl Mesh for cffi::TetMesh {
//    type MeshType = geo::mesh::TetMesh<f64>;
//    fn mesh_mut(&mut self) -> &mut Self::MeshType { &mut self.mesh }
//    fn mesh_ref(&self) -> &Self::MeshType { &self.mesh }
//    fn mesh(self) -> Self::MeshType { self.mesh }
//}
//
//impl Mesh for cffi::PolyMesh {
//    type MeshType = geo::mesh::PolyMesh<f64>;
//    fn mesh_mut(&mut self) -> &mut Self::MeshType { &mut self.mesh }
//    fn mesh_ref(&self) -> &Self::MeshType { &self.mesh }
//    fn mesh(self) -> Self::MeshType { self.mesh }
//}
//
///// Utility to convert a mesh pointer to an `Option<&mut Mesh>` type.
///// Although this function does a `nullptr` check, it is still unsafe since the pointer can be
///// invalid.
//pub unsafe fn mesh_mut<'a, M: Mesh + 'a>(mesh_ptr: *mut M) -> Option<&'a mut <M as Mesh>::MeshType> {
//    if mesh_ptr.is_null() {
//        None
//    } else {
//        Some((*mesh_ptr).mesh_mut())
//    }
//}
//
//pub unsafe fn mesh_box<M: Mesh>(mesh_ptr: *mut M) -> Option<Box<<M as Mesh>::MeshType>> {
//    if mesh_ptr.is_null() {
//        None
//    } else {
//        let mesh_box = Box::from_raw(mesh_ptr);
//        Some(Box::new((*mesh_box).mesh()))
//    }
//}
//

/// A convenience utility to convert a mutable pointer to an optional mutable reference.
pub unsafe fn as_mut<'a, T: 'a>(ptr: *mut T) -> Option<&'a mut T> {
    NonNull::new(ptr).map(|x| &mut *x.as_ptr())
}
