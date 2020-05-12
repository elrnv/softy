use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::energy_models::Either;
use crate::objects::{material::*, *};
use crate::{TetMesh, TriMesh};
use geo::mesh::Attrib;
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
    type ElementIndex = CellIndex;
    fn num_elements(&self) -> usize {
        self.tetmesh.num_cells()
    }
    fn mesh(&self) -> &TetMesh {
        &self.tetmesh
    }
    fn mesh_mut(&mut self) -> &mut TetMesh {
        &mut self.tetmesh
    }
    fn material_scale(&self) -> f32 {
        self.material.scale()
    }
    fn material_id(&self) -> usize {
        self.material.id
    }
}

impl TetMeshSolid {
    /// A helper function to flag all elements with all vertices fixed as fixed.
    pub(crate) fn init_fixed_element_attribute(&mut self) -> Result<(), Error> {
        let fixed_verts = self
            .mesh()
            .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;

        let fixed_elements: Vec<_> = self
            .mesh()
            .cell_iter()
            .map(|cell| cell.iter().map(|&vi| fixed_verts[vi]).sum::<i8>() / 4)
            .collect();

        self.mesh_mut()
            .set_attrib_data::<FixedIntType, CellIndex>(FIXED_ATTRIB, &fixed_elements)?;

        Ok(())
    }
}

impl DynamicObject for TetMeshSolid {
    fn scaled_density(&self) -> Option<f32> {
        self.material.scaled_density()
    }
}

impl DeformableObject for TetMeshSolid {}
impl ElasticObject for TetMeshSolid {
    fn scaled_elasticity(&self) -> Option<ElasticityParameters> {
        self.material.scaled_elasticity()
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
        self.surface
            .borrow_with(|| TetMeshSurface::from(&self.tetmesh))
    }
}

impl<'a> Elasticity<'a, Either<TetMeshNeoHookean<'a, f64>, TetMeshStableNeoHookean<'a, f64>>>
    for TetMeshSolid
{
    fn elasticity(
        &'a self,
    ) -> Either<TetMeshNeoHookean<'a, f64>, TetMeshStableNeoHookean<'a, f64>> {
        match self.material.model() {
            ElasticityModel::NeoHookean => Either::Left(TetMeshNeoHookean::new(self)),
            ElasticityModel::StableNeoHookean => Either::Right(TetMeshStableNeoHookean::new(self)),
        }
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
    /// maintains a link to the original tetmesh via the `indices` vector.
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
            .into_data()
            .into_vec::<TetMeshVertexIndexType>()
            .expect("Incorrect index type: not usize");

        trimesh.sort_vertices_by_key(|k| indices[k]);

        indices.sort_unstable();

        TetMeshSurface { indices, trimesh }
    }
}
