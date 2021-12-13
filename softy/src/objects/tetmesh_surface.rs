#[cfg(feature = "optsolver")]
use crate::attrib_defines::*;
#[cfg(feature = "optsolver")]
use crate::{TetMesh, TriMesh};
#[cfg(feature = "optsolver")]
use geo::{attrib::Attrib, mesh::topology::*};

#[derive(Clone, Debug)]
#[cfg(feature = "optsolver")]
pub(crate) struct TetMeshSurface {
    /// Vertex indices into the original tetmesh.
    pub indices: Vec<usize>,
    /// The triangle mesh representing the entire surface of the original TetMesh.
    pub trimesh: TriMesh,
}

#[cfg(feature = "optsolver")]
impl TetMeshSurface {
    /// Extract the triangle surface vertices of this tetmesh.
    ///
    /// The returned trimesh maintains a link to the original tetmesh via the `indices` vector.
    pub fn new(
        tetmesh: &TetMesh,
        use_fixed: bool,
        filter: impl FnMut(&geo::mesh::TetFace) -> bool,
    ) -> TetMeshSurface {
        let mut trimesh = tetmesh.surface_trimesh_with_mapping_and_filter(
            Some(TETMESH_VERTEX_INDEX_ATTRIB),
            None,
            None,
            None,
            filter,
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

#[cfg(feature = "optsolver")]
impl From<&TetMesh> for TetMeshSurface {
    /// Extract the triangle surface of this tetmesh.
    ///
    /// The returned trimesh maintains a link to the original tetmesh via the `indices` vector.
    #[inline]
    fn from(tetmesh: &TetMesh) -> TetMeshSurface {
        TetMeshSurface::new(tetmesh, true, |_| true)
    }
}
