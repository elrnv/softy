use crate::energy_models::gravity::*;
use crate::objects::*;
use crate::TriMesh;
use geo::mesh::topology::*;

/// A rigid object represented by a trimesh.
pub struct TriMeshRigidBody {
    pub trimesh: TriMesh,
    pub material: RigidMaterial,
}

// TODO: This impl can be automated with a derive macro
impl Object for TriMeshRigidBody {
    type Mesh = TriMesh;
    type Material = RigidMaterial;
    type ElementIndex = FaceIndex;
    fn num_elements(&self) -> usize {
        self.trimesh.num_faces()
    }
    fn mesh(&self) -> &TriMesh {
        &self.trimesh
    }
    fn material(&self) -> &RigidMaterial {
        &self.material
    }
    fn mesh_mut(&mut self) -> &mut TriMesh {
        &mut self.trimesh
    }
    fn material_mut(&mut self) -> &mut RigidMaterial {
        &mut self.material
    }
}

impl TriMeshRigidBody {
    pub fn new(trimesh: TriMesh, material: RigidMaterial) -> TriMeshRigidBody {
        TriMeshRigidBody { trimesh, material }
    }
}

//impl<'a> Elasticity<TriMeshNeoHookean<'a>> for TriMeshShell {
//    fn elasticity(&'a self) -> TriMeshNeoHookean<'a> {
//        unimplemented!(),// TriMeshNeoHookean(self),
//    }
//}
//
//impl<'a> Inertia<TriMeshInertia<'a>> for TriMeshShell {
//    fn inertia(&'a self) -> TriMeshInertia<'a> {
//        unimplemented!(), //TriMeshInertia(self),
//    }
//}

impl<'a> Gravity<'a, TriMeshGravity<'a>> for TriMeshRigidBody {
    fn gravity(&'a self, g: [f64; 3]) -> TriMeshGravity<'a> {
        TriMeshGravity::new(self, g)
    }
}
