//!
//! This module defines utility functions and types for interfacing with hdkrs api from Rust.
//!

use geo;
pub use libc::c_void;

pub trait Mesh {
    type MeshType;

    fn mesh(&mut self) -> &mut Self::MeshType; // consuming getter
}

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

//TODO: the following can be implemented automatically with a procedural derive macro.
impl Mesh for TetMesh {
    type MeshType = geo::mesh::TetMesh<f64>;
    fn mesh(&mut self) -> &mut Self::MeshType  { &mut self.mesh }
}

impl Mesh for PolyMesh {
    type MeshType = geo::mesh::PolyMesh<f64>;
    fn mesh(&mut self) -> &mut Self::MeshType { &mut self.mesh }
}

/// Utility to convert a mesh pointer to an Rust `Option<&mut Mesh>` type.
/// Although this function does a `nullptr` check, it is still unsafe since the pointer can be
/// invalid.
pub unsafe fn mesh<'a, M: Mesh + 'a>(mesh_ptr: *mut M) -> Option<&'a mut <M as Mesh>::MeshType> {
    if mesh_ptr.is_null() {
        None
    } else {
        Some((*mesh_ptr).mesh())
    }
}

/// Utility to cast the void pointer to the interrupt checker function a valid Rust type.
pub unsafe fn interrupt_callback(checker: *mut c_void,
                                 check_interrupt: Option<extern "C" fn(*const c_void) -> bool>
                                 ) -> impl Fn() -> bool {
    let interrupt_ref = &*checker; // conversion needed sicne *mut c_void is not Send
    move || match check_interrupt {
        Some(cb) => cb(interrupt_ref as *const c_void),
        None => true,
    }
}

/// The Rust version of the cook result enum.
pub enum CookResult {
    Success(String),
    Warning(String),
    Error(String),
}
