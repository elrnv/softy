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
    pub(crate) deforming_surface: LazyCell<TetMeshSurface>,
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
    fn density(&self) -> Option<f32> {
        self.material.density()
    }
}

impl DeformableObject for TetMeshSolid {}
impl ElasticObject for TetMeshSolid {
    fn elasticity_parameters(&self) -> Option<ElasticityParameters> {
        self.material.elasticity()
    }
}

impl TetMeshSolid {
    pub fn new(tetmesh: TetMesh, material: SolidMaterial) -> TetMeshSolid {
        TetMeshSolid {
            tetmesh,
            material,
            surface: LazyCell::new(),
            deforming_surface: LazyCell::new(),
        }
    }

    /// Build the surface of the underlying `TetMesh`.
    pub(crate) fn surface(&self, use_fixed: bool) -> &TetMeshSurface {
        if use_fixed {
            self.surface
                .borrow_with(|| TetMeshSurface::from(&self.tetmesh))
        } else {
            self.deforming_surface()
        }
    }

    /// Build the deforming part of the surface of the underlying `TetMesh`.
    pub(crate) fn deforming_surface(&self) -> &TetMeshSurface {
        self.deforming_surface
            .borrow_with(|| TetMeshSurface::new(&self.tetmesh, false))
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
    /// Vertex indices into the original tetmesh.
    pub indices: Vec<usize>,
    /// The triangle mesh representing the entire surface othe original tetmesh.
    pub trimesh: TriMesh,
}

impl TetMeshSurface {
    /// Extract the triangle surface vertices of this tetmesh.
    ///
    /// The returned trimesh maintains a link to the original tetmesh via the `indices` vector.
    fn new(tetmesh: &TetMesh, use_fixed: bool) -> TetMeshSurface {
        let mut trimesh = tetmesh.surface_trimesh_with_mapping(
            Some(TETMESH_VERTEX_INDEX_ATTRIB),
            None,
            None,
            None,
        );

        // Get the vertex indices into the original tetmesh.
        let mut indices = trimesh
            .remove_attrib::<VertexIndex>(TETMESH_VERTEX_INDEX_ATTRIB)
            .expect("Failed to map indices.")
            .into_data()
            .into_vec::<TetMeshVertexIndexType>()
            .expect("Incorrect index type: not usize");

        if !use_fixed {
            use geo::algo::Split;

            // Extract only the deforming part of the trimesh
            let partition: Vec<_> = trimesh
                .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
                .expect("Missing fixed attribute")
                .map(|&i| i as usize)
                .collect();

            let meshes = trimesh.split(&partition, 2);
            trimesh = if meshes.len() > 1 {
                meshes.into_iter().next().unwrap()
            } else {
                // This will be the case when all vertices are fixed.
                TriMesh::new(vec![], vec![])
            };
        }

        // Sort trimesh vertices according to their order in the original tetmesh.
        trimesh.sort_vertices_by_key(|k| indices[k]);

        // Also sort the tetmesh indices to make sure they correspond to the trimesh indices.
        indices.sort();

        // Note: The reason why it is critical to sort the indices is to allow them to be used
        // inside flatk Subsets, which expect unique sorted indices.
        // This enables parallel mutable iteration.

        TetMeshSurface { indices, trimesh }
    }
}

impl From<&TetMesh> for TetMeshSurface {
    /// Extract the triangle surface of this tetmesh.
    ///
    /// The returned trimesh maintains a link to the original tetmesh via the `indices` vector.
    fn from(tetmesh: &TetMesh) -> TetMeshSurface {
        TetMeshSurface::new(tetmesh, true)
    }
}
