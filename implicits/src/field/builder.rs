use super::*;
use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::*, PointCloud, TriMesh, VertexMesh};
use crate::kernel::KernelType;
use std::cell::RefCell;

/// A mesh type to represent the samples for the implicit surface. This enum is used solely for
/// building the implicit surface.
#[derive(Clone, Debug, PartialEq)]
enum SamplesMesh<'a> {
    TriMesh(&'a TriMesh<f64>),
    PointCloud(PointCloud<f64>),
    None,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitSurfaceBuilder<'mesh> {
    kernel: KernelType,
    bg_field: BackgroundFieldType,
    mesh: SamplesMesh<'mesh>,
    max_step: f64,
    sample_type: SampleType,
}

impl Default for ImplicitSurfaceBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'mesh> ImplicitSurfaceBuilder<'mesh> {
    pub fn new() -> Self {
        ImplicitSurfaceBuilder {
            kernel: KernelType::Approximate {
                radius: 1.0,
                tolerance: 1e-5,
            },
            bg_field: BackgroundFieldType::Zero,
            mesh: SamplesMesh::None,
            max_step: 0.0, // This is a sane default for static implicit surfaces.
            sample_type: SampleType::Vertex,
        }
    }

    pub fn kernel(&mut self, kernel: KernelType) -> &mut Self {
        self.kernel = kernel;
        self
    }

    /// Initialize fit data using a vertex mesh which can include positions, normals and offsets
    /// as attributes on the mesh struct.  The normals attribute is expected to be named "N" and
    /// have type `[f32;3]`.  The offsets attribute is expected to be named "offsets" and have type
    /// `f32`.  This function initializes the `vertices`, `vertex_normals`, `sample_offsets`
    /// and sets `sample_type` to `SampleType::Vertex`.
    ///
    /// Note that this initializer can create only a static implicit surface because the topology
    /// of the mesh is unknown so the normals cannot be recomputed.
    pub fn vertex_mesh<M: VertexMesh<f64>>(&mut self, mesh: &M) -> &mut Self {
        self.mesh = SamplesMesh::PointCloud(PointCloud::from(mesh));
        self.sample_type = SampleType::Vertex; // No topology info is available, so this must be vertex based.
        self
    }

    /// Initialize fit data using a triangle mesh which can include positions, normals and offsets
    /// as attributes on the mesh struct.
    ///
    /// ## Normals
    ///
    /// The normals attribute is expected to be named "N" and
    /// have type `[f32;3]`.  Note that the normals are recomputed when an implicit surface is
    /// updated, so the normals given here are only used until the first call to
    /// `ImplicitSurface::update`.
    ///
    /// ## Offsets
    ///
    /// The offsets attribute is expected to be named "offsets" and have type
    /// `f32`.
    pub fn trimesh(&mut self, mesh: &'mesh TriMesh<f64>) -> &mut Self {
        self.mesh = SamplesMesh::TriMesh(mesh);
        self
    }

    pub fn background_field(
        &mut self,
        bg_field: BackgroundFieldType,
    ) -> &mut Self {
        self.bg_field = bg_field;
        self
    }

    pub fn max_step(&mut self, max_step: f64) -> &mut Self {
        self.max_step = max_step;
        self
    }

    pub fn sample_type(&mut self, ty: SampleType) -> &mut Self {
        self.sample_type = ty;
        self
    }

    /// A helper function to extract a `Vec` of vertex positions from a `VertexMesh`.
    fn vertex_positions_from_mesh<M: VertexMesh<f64>>(mesh: &M) -> Vec<Vector3<f64>> {
        mesh.vertex_position_iter()
            .map(|&x| {
                Vector3(x)
                    .cast::<f64>()
                    .expect("Failed to convert positions to f64")
            })
        .collect()
    }
    /// A helper function to extract an offset vector from the mesh.
    fn vertex_offsets_from_mesh<M: VertexMesh<f64>>(mesh: &M) -> Vec<f64> {
        mesh.attrib_iter::<f32, VertexIndex>("offset")
            .map(|iter| iter.map(|&x| f64::from(x)).collect())
            .unwrap_or_else(|_| vec![0.0f64; mesh.num_vertices()])
    }


    /// A helper function to extract a normals vector from the mesh.
    /// The resulting vector is empty if no normals are found.
    fn vertex_normals_from_mesh<M: VertexMesh<f64>>(mesh: &M) -> Vec<Vector3<f64>> {
        mesh
            .attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| {
                iter.map(|nml| Vector3(*nml).cast::<f64>().unwrap())
                    .collect()
            })
        .unwrap_or_default()
    }

    /// Builds the implicit surface. This function returns `None` when theres is not enough data to
    /// make a valid implict surface. For example if kernel radius is 0.0 or points is empty, this
    /// function will return `None`.
    pub fn build(&self) -> Option<ImplicitSurface> {
        let ImplicitSurfaceBuilder {
            kernel,
            bg_field,
            mesh,
            max_step,
            sample_type,
        } = self.clone();
        // Cannot build an implicit surface when the radius is 0.0.
        match kernel {
            KernelType::Interpolating { radius }
            | KernelType::Approximate { radius, .. }
            | KernelType::Cubic { radius } => {
                if radius == 0.0 {
                    return None;
                }
            }
            _ => {} // Nothing to be done for global support kernels.
        }

        let (samples, vertices, triangles) = match mesh {
            SamplesMesh::PointCloud(ptcloud) => {
                let vertices = Self::vertex_positions_from_mesh(&ptcloud);

                if sample_type == SampleType::Face {
                    return None; // Given an incompatible sample type.
                }

                let sample_offsets = Self::vertex_offsets_from_mesh(&ptcloud);
                assert_eq!(vertices.len(), sample_offsets.len());

                let vertex_normals = Self::vertex_normals_from_mesh(&ptcloud);
                if vertex_normals.is_empty() {
                    return None; // Must have normals on vertices for point clouds.
                }

                assert_eq!(vertices.len(), vertex_normals.len());
                let samples = Samples {
                    points: vertices.clone(),
                    normals: vertex_normals,
                    offsets: sample_offsets,
                };

                (samples, vertices, Vec::new())
            }
            SamplesMesh::TriMesh(mesh) => {
                let vertices = Self::vertex_positions_from_mesh(mesh);
                let triangles = reinterpret::reinterpret_slice(mesh.faces()).to_vec();

                // Build the samples.
                let samples = match sample_type {
                    SampleType::Vertex => {
                        let sample_offsets = Self::vertex_offsets_from_mesh(mesh);
                        assert_eq!(vertices.len(), sample_offsets.len());

                        let mut vertex_normals = Self::vertex_normals_from_mesh(mesh);
                        if vertex_normals.is_empty() {
                            // If empty, just recompute from the given topology.
                            vertex_normals = vec![Vector3::zeros(); vertices.len()];
                            ImplicitSurface::compute_vertex_area_normals(
                                &triangles,
                                &vertices,
                                &mut vertex_normals,
                                );
                        }

                        assert_eq!(vertices.len(), vertex_normals.len());
                        Samples {
                            points: vertices.clone(),
                            normals: vertex_normals,
                            offsets: sample_offsets,
                        }
                    }
                    SampleType::Face => {
                        // Can't create a face centered potential if there are no faces.
                        if triangles.is_empty() {
                            return None;
                        }

                        // One sample per triangle.
                        let sample_offsets = mesh
                            .attrib_iter::<f32, FaceIndex>("offset")
                            .map(|iter| iter.map(|&x| f64::from(x)).collect())
                            .unwrap_or_else(|_| vec![0.0f64; mesh.num_faces()]);
                        assert_eq!(triangles.len(), sample_offsets.len());

                        Samples::new_triangle_samples(&triangles, &vertices, sample_offsets)
                    }
                };

                (samples, vertices, triangles)
            }

            // Cannot build an implicit surface without sample points. This is an error.
            SamplesMesh::None => return None,
        };

        let mut dual_topo = Vec::new();
        // Build the dual topology. Only needed for vertex centric implicit surfaces.
        if let SampleType::Vertex = sample_type {
            // Construct dual topology for a vertex centered implicit surface which we may
            // need to differentiate.
            if !triangles.is_empty() {
                // Compute the one ring and store it in the dual topo vectors.
                dual_topo.resize(vertices.len(), Vec::new());
                for (tri_idx, tri) in triangles.iter().enumerate() {
                    for &vidx in tri {
                        dual_topo[vidx].push(tri_idx);
                    }
                }
            }
        }

        // Build the rtree.
        let rtree = build_rtree_from_samples(&samples);

        Some(ImplicitSurface {
            kernel,
            bg_field_type: bg_field,
            spatial_tree: rtree,
            surface_topo: triangles,
            surface_vertex_positions: vertices,
            samples,
            max_step,
            neighbour_cache: RefCell::new(NeighbourCache::new()),
            dual_topo,
            sample_type,
        })
    }
}
