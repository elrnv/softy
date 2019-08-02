use crate::energy_models::gravity::*;
use crate::objects::*;
use crate::TriMesh;
use geo::mesh::topology::*;

/// A soft shell represented by a trimesh. It is effectively a triangle mesh decorated by
/// physical material properties that govern how it behaves.
pub struct TriMeshShell {
    pub trimesh: TriMesh,
    pub material: ShellMaterial,
}

// TODO: This impl can be automated with a derive macro
impl Object for TriMeshShell {
    type Mesh = TriMesh;
    type Material = ShellMaterial;
    type ElementIndex = FaceIndex;
    fn num_elements(&self) -> usize {
        self.trimesh.num_faces()
    }
    fn mesh(&self) -> &TriMesh {
        &self.trimesh
    }
    fn material(&self) -> &ShellMaterial {
        &self.material
    }
    fn mesh_mut(&mut self) -> &mut TriMesh {
        &mut self.trimesh
    }
    fn material_mut(&mut self) -> &mut ShellMaterial {
        &mut self.material
    }
}

impl TriMeshShell {
    pub fn new(trimesh: TriMesh, material: ShellMaterial) -> TriMeshShell {
        TriMeshShell { trimesh, material }
    }
}

//impl<'a> Elasticity<TriMeshNeoHookean<'a>> for TriMeshShell {
//    fn elasticity(&'a self) -> TriMeshNeoHookean<'a> {
//        match self.material.properties {
//            ShellMaterial::Deformable => unimplemented!(),// TriMeshNeoHookean(self),
//            ShellMaterial::Rigid => unimplemented!(),
//            ShellMaterial::Static => unimplemented!(),
//        }
//    }
//}
//
//impl<'a> Inertia<TriMeshInertia<'a>> for TriMeshShell {
//    fn inertia(&'a self) -> TriMeshInertia<'a> {
//        match self.material.properties {
//            ShellMaterial::Deformable => unimplemented!(), //TriMeshInertia(self),
//            ShellMaterial::Rigid => unimplemented!(),
//            ShellMaterial::Static => unimplemented!(),
//        }
//    }
//}

impl<'a> Gravity<'a, TriMeshGravity<'a>> for TriMeshShell {
    fn gravity(&'a self, g: [f64; 3]) -> TriMeshGravity<'a> {
        match self.material.properties {
            ShellProperties::Deformable { .. } => TriMeshGravity::new(self, g),
            ShellProperties::Rigid { .. } => TriMeshGravity::new(self, g),
            ShellProperties::Fixed => unimplemented!(),
        }
    }
}
