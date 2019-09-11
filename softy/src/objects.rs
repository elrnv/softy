pub mod material;
pub mod shell;
pub mod solid;

pub use material::*;
pub use shell::*;
pub use solid::*;

use geo::mesh::{
    attrib::{AttribIndex, VertexAttrib},
    Attrib,
};

/// A helper trait used to abstract over different types of objects.
pub trait Object {
    type Mesh: Attrib + VertexAttrib;
    type Material: Deformable;
    type ElementIndex: AttribIndex<Self::Mesh>;
    fn num_elements(&self) -> usize;
    fn mesh(&self) -> &Self::Mesh;
    fn material(&self) -> &Self::Material;
    fn mesh_mut(&mut self) -> &mut Self::Mesh;
    fn material_mut(&mut self) -> &mut Self::Material;
}
