use crate::energy_models::elasticity::TetMeshNeoHookean;
use crate::objects::Material;
use crate::{TetMesh, TriMesh};

/// A soft solid represented by a tetmesh. It is effectively a tetrahedral mesh decorated by
/// physical material properties that govern how it behaves.
/// Additionally this struct may precompute its own surface topology on demand, which is
/// useful in contact problems.
pub struct TetMeshSolid {
    pub tetmesh: TetMesh,
    pub material: SolidMaterial,
    pub surface: RefCell<Option<TetMeshSurface>>,
}

impl TetMeshSolid {
    pub fn new(tetmesh: TetMesh, material: SolidMaterial) -> TetMeshSolid {
        TetMeshSolid {
            tetmesh,
            material,
            surface: None,
        }
    }

    pub fn surface(&self) -> &TetMeshSurface {
        let mut surface = self.surface.borrow_mut();
        surface.as_ref().unwrap_or_else(move || {
            *surface = Some(TetMeshSurface::from(&self.tetmesh));
            surface.as_ref().unwrap()
        })
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
        TetMeshGravity {
            solid: self,
            g: g.into(),
        }
    }
}

pub(crate) struct TetMeshSurface {
    pub indices: Vec<usize>,
    pub trimesh: TriMesh,
}

impl From<&TetMesh> for TetMeshSurface {
    /// Extract the triangle surface of this tetmesh. The returned trimesh
    /// maintains a link to the original tetmesh via the.
    fn from(solid: &TetMesh) -> TetMeshSurface {
        let mut trimesh = solid.tetmesh.surface_trimesh_with_mapping(
            TETMESH_VERTEX_INDEX_ATTRIB,
            None,
            None,
            None,
        );
        let indices = trimesh
            .remove_attrib::<VertexIndex>(TETMESH_VERTEX_INDEX_ATTRIB)
            .expect("Failed to map indices.")
            .into_buffer()
            .into_vec::<usize>()
            .expect("Incorrect index type: not usize");

        TetMeshSurface { indices, trimesh }
    }
}
