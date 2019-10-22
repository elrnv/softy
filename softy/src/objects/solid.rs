use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::objects::{material::*, Object};
use crate::{TetMesh, TriMesh};
use geo::mesh::{topology::*, Attrib};
use lazycell::LazyCell;

/// A soft solid represented by a tetmesh. It is effectively a tetrahedral mesh decorated by
/// physical material properties that govern how it behaves.
/// Additionally this struct may precompute its own surface topology on demand, which is
/// useful in contact problems.
#[derive(Clone, Debug)]
pub struct TetMeshSolid {
    pub tetmesh: TetMesh,
    pub material: SolidMaterial,
    pub(crate) surface: LazyCell<TetMeshSurface>,
}

// TODO: This impl can be automated with a derive macro
impl Object for TetMeshSolid {
    type Mesh = TetMesh;
    type Material = SolidMaterial;
    type ElementIndex = CellIndex;
    fn num_elements(&self) -> usize {
        self.tetmesh.num_cells()
    }
    fn mesh(&self) -> &TetMesh {
        &self.tetmesh
    }
    fn material(&self) -> &SolidMaterial {
        &self.material
    }
    fn mesh_mut(&mut self) -> &mut TetMesh {
        &mut self.tetmesh
    }
    fn material_mut(&mut self) -> &mut SolidMaterial {
        &mut self.material
    }
}

impl TetMeshSolid {
    pub fn new(tetmesh: TetMesh, material: SolidMaterial) -> TetMeshSolid {
        TetMeshSolid {
            tetmesh,
            material,
            surface: LazyCell::new(),
        }
    }

    pub(crate) fn surface(&self) -> &TetMeshSurface {
        self.surface.borrow_with(|| {
            TetMeshSurface::from(&self.tetmesh)
        })
    }
}

impl<'a> Elasticity<'a, TetMeshNeoHookean<'a>> for TetMeshSolid {
    fn elasticity(&'a self) -> TetMeshNeoHookean<'a> {
        TetMeshNeoHookean(self)
    }
}

impl<'a> Inertia<'a, TetMeshInertia<'a>> for TetMeshSolid {
    fn inertia(&'a self) -> TetMeshInertia<'a> {
        TetMeshInertia(self)
    }
}

impl<'a> Gravity<'a, TetMeshGravity<'a>> for TetMeshSolid {
    fn gravity(&'a self, g: [f64; 3]) -> TetMeshGravity<'a> {
        TetMeshGravity::new(self, g)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TetMeshSurface {
    pub indices: Vec<usize>,
    pub trimesh: TriMesh,
}

impl From<&TetMesh> for TetMeshSurface {
    /// Extract the triangle surface of this tetmesh. The returned trimesh
    /// maintains a link to the original tetmesh via the.
    fn from(tetmesh: &TetMesh) -> TetMeshSurface {
        let mut trimesh = tetmesh.surface_trimesh_with_mapping(
            Some(TETMESH_VERTEX_INDEX_ATTRIB),
            None,
            None,
            None,
        );

        let mut indices = trimesh
            .remove_attrib::<VertexIndex>(TETMESH_VERTEX_INDEX_ATTRIB)
            .expect("Failed to map indices.")
            .into_buffer()
            .into_vec::<TetMeshVertexIndexType>()
            .expect("Incorrect index type: not usize");

        trimesh.sort_vertices_by_key(|k| indices[k]);

        indices.sort_unstable();

        TetMeshSurface { indices, trimesh }
    }
}
