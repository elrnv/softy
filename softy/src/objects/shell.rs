use crate::TriMesh;
use crate::objects::Material;

/// A soft shell represented by a trimesh. It is effectively a triangle mesh decorated by
/// physical material properties that govern how it behaves.
pub struct TriMeshShell {
    pub trimesh: TriMesh,
    pub material: ShellMaterial,
}

impl TriMeshShell {
    pub fn new(trimesh: TriMesh, material: ShellMaterial) -> TriMeshShell {
        TriMeshShell {
            trimesh,
            material
        }
    }
}

impl<'a> Elasticity<TriMeshNeoHookean<'a>> for TriMeshShell {
    fn elasticity(&'a self) -> TriMeshNeoHookean<'a> {
        match self.material.properties {
            ShellMaterial::Deformable => unimplemented!(),// TriMeshNeoHookean(self),
            ShellMaterial::Rigid => unimplemented!(),
            ShellMaterial::Static => unimplemented!(),
        }
    }
}

impl<'a> Inertia<TriMeshInertia<'a>> for TriMeshShell {
    fn inertia(&'a self) -> TriMeshInertia<'a> {
        match self.material.properties {
            ShellMaterial::Deformable => unimplemented!(), //TriMeshInertia(self),
            ShellMaterial::Rigid => unimplemented!(),
            ShellMaterial::Static => unimplemented!(),
        }
    }
}

impl<'a> Gravity<TriMeshGravity<'a>> for TriMeshShell {
    fn gravity(&'a self, g: [f64; 3]) -> TriMeshGravity<'a> {
        match self.material.properties {
            ShellMaterial::Deformable => TriMeshGravity { shell: self, g: g.into() },
            ShellMaterial::Rigid => TriMeshGravity { shell: self, g: g.into() },
            ShellMaterial::Static => unimplemented!(),
        }
    }
}

