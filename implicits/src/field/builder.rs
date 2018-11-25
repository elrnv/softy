use super::*;
use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::*, TriMesh, VertexMesh, VertexPositions};
use crate::kernel::KernelType;
use std::cell::RefCell;

#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitSurfaceBuilder {
    kernel: KernelType,
    background_potential: BackgroundPotentialType,
    triangles: Vec<[usize; 3]>,
    vertices: Vec<Vector3<f64>>,
    vertex_normals: Vec<Vector3<f64>>,
    sample_offsets: Vec<f64>,
    max_step: f64,
    sample_type: SampleType,
}

impl ImplicitSurfaceBuilder {
    pub fn new() -> Self {
        ImplicitSurfaceBuilder {
            kernel: KernelType::Approximate {
                radius: 1.0,
                tolerance: 1e-5,
            },
            background_potential: BackgroundPotentialType::Zero,
            triangles: Vec::new(),
            vertices: Vec::new(),
            vertex_normals: Vec::new(),
            sample_offsets: Vec::new(),
            max_step: 0.0, // This is a sane default for static implicit surfaces.
            sample_type: SampleType::Vertex,
        }
    }

    pub fn triangles(&mut self, triangles: Vec<[usize; 3]>) -> &mut Self {
        self.triangles = triangles;
        self
    }

    pub fn vertices(&mut self, points: Vec<[f64; 3]>) -> &mut Self {
        self.vertices = reinterpret::reinterpret_vec(points);
        self
    }

    pub fn offsets(&mut self, offsets: Vec<f64>) -> &mut Self {
        self.sample_offsets = offsets;
        self
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
        self.vertices = mesh
            .vertex_positions()
            .iter()
            .map(|&x| {
                Vector3(x)
                    .cast::<f64>()
                    .expect("Failed to convert positions to f64")
            })
            .collect();
        self.vertex_normals = mesh
            .attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| {
                iter.map(|nml| Vector3(*nml).cast::<f64>().unwrap())
                    .collect()
            })
            .unwrap_or(vec![Vector3::zeros(); mesh.num_vertices()]);
        self.sample_offsets = mesh
            .attrib_iter::<f32, VertexIndex>("offset")
            .map(|iter| iter.map(|&x| x as f64).collect())
            .unwrap_or(vec![0.0f64; mesh.num_vertices()]);
        self.sample_type = SampleType::Vertex;
        self
    }

    /// Initialize fit data using a triangle mesh which can include positions, normals and offsets
    /// as attributes on the mesh struct.  The normals attribute is expected to be named "N" and
    /// have type `[f32;3]`.  The offsets attribute is expected to be named "offsets" and have type
    /// `f32`.  This function initializes the `vertices`, `vertex_normals`, `sample_offsets`,
    /// `triangles` and sets `sample_type` to `SampleType::Vertex`.
    pub fn vertex_samples_from_mesh(&mut self, mesh: &TriMesh<f64>) -> &mut Self {
        self.vertices = mesh
            .vertex_positions()
            .iter()
            .map(|&x| {
                Vector3(x)
                    .cast::<f64>()
                    .expect("Failed to convert positions to f64")
            })
            .collect();
        self.vertex_normals = mesh
            .attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| {
                iter.map(|nml| Vector3(*nml).cast::<f64>().unwrap())
                    .collect()
            })
            .unwrap_or(Vec::new());
        self.sample_offsets = mesh
            .attrib_iter::<f32, VertexIndex>("offset")
            .map(|iter| iter.map(|&x| x as f64).collect())
            .unwrap_or(vec![0.0f64; mesh.num_vertices()]);
        self.triangles = reinterpret::reinterpret_slice(mesh.faces()).to_vec();
        self.sample_type = SampleType::Vertex;
        self
    }

    /// Initialize fit data using a triangle mesh.
    /// No normal attributes are expected, the triangle normals are computed automatically.
    /// The offsets attribute on faces is expected to be named "offsets" and have type `f32`.
    /// This function initializes the `vertices`, `sample_offsets` and sets `sample_type` to
    /// `SampleType::Face`.
    pub fn face_samples_from_mesh(&mut self, mesh: &TriMesh<f64>) -> &mut Self {
        self.vertices = mesh
            .vertex_positions()
            .iter()
            .map(|&x| {
                Vector3(x)
                    .cast::<f64>()
                    .expect("Failed to convert positions to f64")
            })
            .collect();
        self.sample_offsets = mesh
            .attrib_iter::<f32, FaceIndex>("offset")
            .map(|iter| iter.map(|&x| x as f64).collect())
            .unwrap_or(vec![0.0f64; mesh.num_faces()]);
        self.triangles = reinterpret::reinterpret_slice(mesh.faces()).to_vec();
        self.sample_type = SampleType::Face;
        self
    }

    pub fn background_potential(
        &mut self,
        background_potential: BackgroundPotentialType,
    ) -> &mut Self {
        self.background_potential = background_potential;
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

    /// Builds the implicit surface. This function returns `None` when theres is not enough data to
    /// make a valid implict surface. For example if kernel radius is 0.0 or points is empty, this
    /// function will return `None`.
    pub fn build(&self) -> Option<ImplicitSurface> {
        let ImplicitSurfaceBuilder {
            kernel,
            background_potential,
            triangles,
            vertices,
            mut vertex_normals,
            mut sample_offsets,
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

        // Cannot build an implicit surface without sample points. This is an error.
        if vertices.is_empty() {
            return None;
        }

        let mut dual_topo = Vec::new();
        // Build the dual topology. Only needed for vertex centric implicit surfaces.
        match sample_type {
            SampleType::Vertex => {
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
            _ => {}
        }

        // Build the samples.
        let samples = match sample_type {
            SampleType::Vertex => {
                if sample_offsets.is_empty() {
                    sample_offsets = vec![0.0; vertices.len()];
                }

                assert_eq!(vertices.len(), sample_offsets.len());
                if vertex_normals.is_empty() {
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
                if sample_offsets.is_empty() {
                    sample_offsets = vec![0.0; triangles.len()];
                }
                assert_eq!(triangles.len(), sample_offsets.len());
                Samples::new_triangle_samples(&triangles, &vertices, sample_offsets)
            }
        };

        // Build the rtree.
        let rtree = build_rtree_from_samples(&samples);

        Some(ImplicitSurface {
            kernel,
            bg_potential_type: background_potential,
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
