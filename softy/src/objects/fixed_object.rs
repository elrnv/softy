use crate::objects::*;
use crate::TriMesh;
use geo::mesh::topology::*;

pub struct TriMeshFixedObject {
    pub trimesh: TriMesh,
    pub material: FixedMaterial,
}

// TODO: This impl can be automated with a derive macro
impl Object for TriMeshFixedObject {
    type Mesh = TriMesh;
    type Material = FixedMaterial;
    type ElementIndex = FaceIndex;
    fn num_elements(&self) -> usize {
        self.trimesh.num_faces()
    }
    fn mesh(&self) -> &TriMesh {
        &self.trimesh
    }
    fn material(&self) -> &FixedMaterial {
        &self.material
    }
    fn mesh_mut(&mut self) -> &mut TriMesh {
        &mut self.trimesh
    }
    fn material_mut(&mut self) -> &mut FixedMaterial {
        &mut self.material
    }
}

impl TriMeshFixedObject {
    pub fn new(trimesh: TriMesh, material: FixedMaterial) -> TriMeshFixedObject {
        TriMeshFixedObject { trimesh, material }
    }
}
