use crate::TetMesh;
use crate::objects::Material;

/// A soft solid represented by a tetmesh. It is effectively a tetrahedral mesh decorated by
/// physical material properties that govern how it behaves.
pub struct TetMeshSolid {
    pub tetmesh: TetMesh,
    pub material: Material,
}

impl TetMeshSolid {
    pub fn new(tetmesh: TetMesh, material: Material) -> TetMeshSolid {
        TetMeshSolid {
            tetmesh,
            material
        }
    }
}

impl<'a> Elasticity<TetMeshNeoHookean<'a>> for TetMeshSolid {
    fn elasticity(&'a self) -> TetMeshNeoHookean<'a> {
        TetMeshNeoHookean(self)
    }
}

impl<'a> Inertia<TetMeshInertia<'a>> for TetMeshSolid {
    fn inertia(&'a self) -> TetMeshInertia<'a> {
        TetMeshInertia(self)
    }
}

impl<'a> Gravity<TetMeshGravity<'a>> for TetMeshSolid {
    fn gravity(&'a self, g: [f64; 3]) -> TetMeshGravity<'a> {
        TetMeshGravity { solid: self, g: g.into() }
    }
}


/// A subset of vertices of a `TetMesh` (or `TetMeshSolid`) on the surface of the mesh.
pub struct TetMeshSurface {
    /// Vertex indices into the original mesh.
    indices: Vec<usize>,
}
