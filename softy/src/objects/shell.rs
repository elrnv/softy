use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::energy_models::elasticity::*;
use crate::fem::problem::Var;
use crate::objects::*;
use crate::TriMesh;
use geo::mesh::topology::*;

/// A soft shell represented by a trimesh. It is effectively a triangle mesh decorated by
/// physical material properties that govern how it behaves.
#[derive(Clone, Debug)]
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

    pub fn tagged_mesh(&self) -> Var<&TriMesh> {
        match self.material.properties {
            ShellProperties::Fixed => Var::Fixed(&self.trimesh),
            _ => Var::Variable(&self.trimesh),
        }
    }
}

impl<'a> Elasticity<'a, Option<TriMeshNeoHookean<'a, f64>>> for TriMeshShell {
    fn elasticity(&'a self) -> Option<TriMeshNeoHookean<'a, f64>> {
        match self.material.properties {
            ShellProperties::Deformable { .. } => Some(TriMeshNeoHookean::new(self)),
            _ => None,
        }
    }
}

impl<'a> Inertia<'a, Option<TriMeshInertia<'a>>> for TriMeshShell {
    fn inertia(&'a self) -> Option<TriMeshInertia<'a>> {
        match self.material.properties {
            ShellProperties::Fixed => None,
            _ => Some(TriMeshInertia(self)),
        }
    }
}

impl<'a> Gravity<'a, Option<TriMeshGravity<'a>>> for TriMeshShell {
    fn gravity(&'a self, g: [f64; 3]) -> Option<TriMeshGravity<'a>> {
        match self.material.properties {
            ShellProperties::Fixed => None,
            _ => Some(TriMeshGravity::new(self, g)),
        }
    }
}
