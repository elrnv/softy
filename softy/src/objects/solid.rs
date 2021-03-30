use lazycell::LazyCell;

use flatk::{zip, Chunked3};
use geo::mesh::Attrib;
use tensr::{Matrix3, Real, Vector3};

use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::energy_models::Either;
use crate::objects::{material::*, *};
use crate::{TetMesh, TriMesh};

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
    #[inline]
    pub(crate) fn surface(&self, use_fixed: bool) -> &TetMeshSurface {
        if use_fixed {
            self.entire_surface()
        } else {
            self.deforming_surface()
        }
    }

    /// Build the entire surface of the underlying `TetMesh`.
    #[inline]
    pub(crate) fn entire_surface(&self) -> &TetMeshSurface {
        self.surface
            .borrow_with(|| TetMeshSurface::from(&self.tetmesh))
    }

    /// Build the deforming part of the surface of the underlying `TetMesh`.
    #[inline]
    pub(crate) fn deforming_surface(&self) -> &TetMeshSurface {
        self.deforming_surface
            .borrow_with(|| TetMeshSurface::new(&self.tetmesh, false))
    }

    /// Given a tetmesh, compute the strain energy per tetrahedron.
    fn compute_strain_energy_attrib(&mut self) {
        use geo::ops::ShapeMatrix;
        // Overwrite the "strain_energy" attribute.
        let mut strain = self
            .tetmesh
            .remove_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB)
            .unwrap();
        strain
            .direct_iter_mut::<f64>()
            .unwrap()
            .zip(zip!(
                self.tetmesh
                    .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                    .unwrap()
                    .map(|&x| f64::from(x)),
                self.tetmesh
                    .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                    .unwrap()
                    .map(|&x| f64::from(x)),
                self.tetmesh
                    .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                    .unwrap(),
                self.tetmesh
                    .attrib_iter::<RefTetShapeMtxInvType, CellIndex>(
                        REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                    )
                    .unwrap(),
                self.tetmesh.tet_iter()
            ))
            .for_each(|(strain, (lambda, mu, &vol, &ref_shape_mtx_inv, tet))| {
                let shape_mtx = Matrix3::new(tet.shape_matrix());
                *strain =
                    NeoHookeanTetEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu).energy()
            });

        self.tetmesh
            .insert_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB, strain)
            .unwrap();
    }

    /// Computes the elastic forces per vertex, and saves it at a vertex attribute.
    fn compute_elastic_forces_attrib(&mut self) {
        use geo::ops::ShapeMatrix;
        let mut forces_attrib = self
            .tetmesh
            .remove_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB)
            .unwrap();

        let mut forces =
            Chunked3::from_array_slice_mut(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

        // Reset forces
        for f in forces.iter_mut() {
            *f = [0.0; 3];
        }

        let grad_iter = zip!(
            self.tetmesh
                .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            self.tetmesh
                .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            self.tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            self.tetmesh
                .attrib_iter::<RefTetShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB,)
                .unwrap(),
            self.tetmesh.tet_iter()
        )
        .map(|(lambda, mu, &vol, &ref_shape_mtx_inv, tet)| {
            let shape_mtx = Matrix3::new(tet.shape_matrix());
            NeoHookeanTetEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu)
                .energy_gradient()
        });

        for (grad, cell) in grad_iter.zip(self.tetmesh.cells().iter()) {
            for j in 0..4 {
                let f = Vector3::new(forces[cell[j]]);
                forces[cell[j]] = (f - grad[j]).into();
            }
        }

        // Reinsert forces back into the attrib map
        self.tetmesh
            .insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
    }
}

impl TetMeshSolid {
    #[inline]
    pub fn elasticity<'a, T: Real>(
        &'a self,
    ) -> Either<TetMeshNeoHookean<'a, T>, TetMeshStableNeoHookean<'a, T>> {
        match self.material.model() {
            ElasticityModel::NeoHookean => Either::Left(TetMeshNeoHookean::new(self)),
            ElasticityModel::StableNeoHookean => Either::Right(TetMeshStableNeoHookean::new(self)),
        }
    }
}

impl<'a> Inertia<'a, TetMeshInertia<'a>> for TetMeshSolid {
    #[inline]
    fn inertia(&'a self) -> TetMeshInertia<'a> {
        TetMeshInertia(self)
    }
}

impl<'a> Gravity<'a, TetMeshGravity<'a>> for TetMeshSolid {
    #[inline]
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
            if meshes.len() > 1 {
                trimesh = meshes.into_iter().next().unwrap();
                // Filter out indices that correspond to fixed vertices.
                indices = indices
                    .into_iter()
                    .zip(partition.iter())
                    .filter_map(|(idx, &part)| if part < 1 { Some(idx) } else { None })
                    .collect();
            } else {
                // This will be the case when all vertices are fixed.
                trimesh = TriMesh::new(vec![], vec![]);
                indices = vec![];
            }
        }

        debug_assert_eq!(trimesh.num_vertices(), indices.len());

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
    #[inline]
    fn from(tetmesh: &TetMesh) -> TetMeshSurface {
        TetMeshSurface::new(tetmesh, true)
    }
}
