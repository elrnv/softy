use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::VertexIndex, VertexMesh};
use crate::kernel::KernelType;
use std::cell::RefCell;
use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ImplicitSurfaceType {
    Vertex,
    Face,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitSurfaceBuilder {
    kernel: KernelType,
    background_potential: bool,
    triangles: Vec<[usize; 3]>,
    points: Vec<Vector3<f64>>,
    normals: Vec<Vector3<f64>>,
    offsets: Vec<f64>,
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
            background_potential: false,
            triangles: Vec::new(),
            points: Vec::new(),
            normals: Vec::new(),
            offsets: Vec::new(),
            max_step: 0.0, // This is a sane default for static implicit surfaces.
            surface_type: ImplicitSurfaceType::Face,
        }
    }

    pub fn with_triangles(&mut self, triangles: Vec<[usize; 3]>) -> &mut Self {
        self.triangles = triangles;
        self
    }

    pub fn with_points(&mut self, points: Vec<[f64; 3]>) -> &mut Self {
        self.points = reinterpret::reinterpret_vec(points);
        self
    }

    pub fn with_offsets(&mut self, offsets: Vec<f64>) -> &mut Self {
        self.offsets = offsets;
        self
    }

    pub fn with_kernel(&mut self, kernel: KernelType) -> &mut Self {
        self.kernel = kernel;
        self
    }

    /// Initialize fit data using a mesh type which can include positions, normals and offsets as
    /// attributes on the mesh struct.
    /// The normals attribute is expected to be named "N" and have type `[f32;3]`.
    /// The offsets attribute is expected to be named "offsets" and have type `f32`.
    pub fn with_mesh<M: VertexMesh<f64>>(&mut self, mesh: &M) -> &mut Self {
        self.points = mesh
            .vertex_positions()
            .iter()
            .map(|&x| {
                Vector3(x)
                    .cast::<f64>()
                    .expect("Failed to convert positions to f64")
            })
            .collect();
        self.normals = mesh
            .attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| {
                iter.map(|nml| Vector3(*nml).cast::<f64>().unwrap())
                    .collect()
            })
            .unwrap_or(vec![Vector3::zeros(); mesh.num_vertices()]);
        self.offsets = mesh
            .attrib_iter::<f32, VertexIndex>("offset")
            .map(|iter| iter.map(|&x| x as f64).collect())
            .unwrap_or(vec![0.0f64; mesh.num_vertices()]);
        self
    }

    pub fn with_background_potential(&mut self, background_potential: bool) -> &mut Self {
        self.background_potential = background_potential;
        self
    }

    pub fn with_max_step(&mut self, max_step: f64) -> &mut Self {
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
            points,
            mut normals,
            mut offsets,
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
        if points.is_empty() {
            return None;
        }

        if normals.is_empty() {
            normals = vec![Vector3::zeros(); points.len()];
        }

        if offsets.is_empty() {
            offsets = vec![0.0; points.len()];
        }

        assert_eq!(points.len(), normals.len());
        assert_eq!(points.len(), offsets.len());

        let mut dual_topo = Vec::new();
        if !triangles.is_empty() {
            // Compute the one ring and store it in the dual topo vectors.
            dual_topo.resize(points.len(), Vec::new());
            for (tri_idx, tri) in triangles.iter().enumerate() {
                for &vidx in tri {
                    dual_topo[vidx].push(tri_idx);
                }
            }
        }

        let oriented_points_iter = oriented_points_iter(&points, &normals);
        Some(ImplicitSurface {
            kernel,
            background_potential,
            spatial_tree: build_rtree(oriented_points_iter),
            surface_topo: triangles,
            samples: Samples {
                points,
                normals,
                offsets,
            },
            max_step,
            neighbour_cache: RefCell::new(NeighbourCache::new()),
            dual_topo,
        })
    }
}

