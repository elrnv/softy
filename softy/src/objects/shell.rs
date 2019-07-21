use crate::TriMesh;
use crate::objects::Material;

/// A soft solid represented by a trimesh. It is effectively a tetrahedral mesh decorated by
/// physical material properties that govern how it behaves.
pub struct TriMeshShell {
    pub trimesh: TriMesh,
    pub material: Material,
}

impl TriMeshShell {
    pub fn new(trimesh: TriMesh, material: Material) -> TriMeshShell {
        TriMeshShell {
            trimesh,
            material
        }
    }
}

impl<'a> Elasticity<TriMeshNeoHookean<'a>> for TriMeshShell {
    fn elasticity(&'a self) -> TriMeshNeoHookean<'a> {
        TriMeshNeoHookean(self)
    }
}

impl<'a> Inertia<TriMeshInertia<'a>> for TriMeshShell {
    fn inertia(&'a self) -> TriMeshInertia<'a> {
        TriMeshInertia(self)
    }
}

impl<'a> Gravity<TriMeshGravity<'a>> for TriMeshShell {
    fn gravity(&'a self, g: [f64; 3]) -> TriMeshGravity<'a> {
        TriMeshGravity { solid: self, g: g.into() }
    }
}

