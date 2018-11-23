use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::VertexIndex, VertexMesh};
use crate::kernel::KernelType;
use std::cell::RefCell;
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitSurfaceBuilder {
    kernel: KernelType,
    background_potential: BackgroundPotentialType,
    triangles: Vec<[usize; 3]>,
    vertices: Vec<Vector3<f64>>,
    vertex_normals: Vec<Vector3<f64>>,
    sample_offsets: Vec<f64>,
    max_step: f64,
    surface_type: ImplicitSurfaceType,
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
            surface_type: ImplicitSurfaceType::Face,
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

    /// Initialize fit data using a mesh type which can include positions, normals and offsets as
    /// attributes on the mesh struct.
    /// The normals attribute is expected to be named "N" and have type `[f32;3]`.
    /// The offsets attribute is expected to be named "offsets" and have type `f32`.
    pub fn mesh<M: VertexMesh<f64>>(&mut self, mesh: &M) -> &mut Self {
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
        self
    }

    pub fn background_potential(&mut self, background_potential: BackgroundPotentialType) -> &mut Self {
        self.background_potential = background_potential;
        self
    }

    pub fn max_step(&mut self, max_step: f64) -> &mut Self {
        self.max_step = max_step;
        self
    }

    pub fn surface_type(&mut self, ty: ImplicitSurfaceType) -> &mut Self {
        self.surface_type = ty;
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
            surface_type,
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
            _ => { } // Nothing to be done for global support kernels.
        }

        // Cannot build an implicit surface without sample points. This is an error.
        if vertices.is_empty() {
            return None;
        }

        if vertex_normals.is_empty() {
            vertex_normals = vec![Vector3::zeros(); vertices.len()];
        }

        assert_eq!(vertices.len(), vertex_normals.len());

        let mut dual_topo = Vec::new();
        // Surface type dependent setup:
        match surface_type {
            ImplicitSurfaceType::Vertex => {
                if sample_offsets.is_empty() {
                    sample_offsets = vec![0.0; vertices.len()];
                }

                assert_eq!(vertices.len(), sample_offsets.len());

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
            ImplicitSurfaceType::Face => {
                // Can't create a face centered potential if there are no faces.
                if triangles.is_empty() {
                    return None;
                }

                // One sample per triangle.
                if sample_offsets.is_empty() {
                    sample_offsets = vec![0.0; triangles.len()];
                }

                assert_eq!(triangles.len(), sample_offsets.len());
            }
        }

        // Build the samples.
        let samples = match surface_type {
            ImplicitSurfaceType::Vertex =>
                Samples { points: vertices, normals: vertex_normals, offsets: sample_offsets },
            ImplicitSurfaceType::Face => {
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
            surface_type,
        })
    }
}

