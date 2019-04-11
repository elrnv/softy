use super::*;
use crate::kernel::KernelType;
use geo::math::Vector3;
use geo::mesh::{topology::*, PointCloud, TriMesh, VertexMesh};
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
    bg_field: BackgroundFieldParams,
    mesh: SamplesMesh<'mesh>,
    max_step: f64,
    base_radius: Option<f64>,
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
                radius_multiplier: 1.0,
                tolerance: 1e-5,
            },
            bg_field: BackgroundFieldParams {
                field_type: BackgroundFieldType::Zero,
                weighted: false,
            },
            mesh: SamplesMesh::None,
            max_step: 0.0, // This is a sane default for static implicit surfaces.
            base_radius: None,
            sample_type: SampleType::Vertex,
        }
    }

    pub fn kernel(&mut self, kernel: KernelType) -> &mut Self {
        self.kernel = kernel;
        self
    }

    pub fn base_radius(&mut self, base_radius: f64) -> &mut Self {
        self.base_radius = Some(base_radius);
        self
    }

    /// Initialize fit data using a vertex mesh which can include positions, normals and offsets
    /// as attributes on the mesh struct.  The normals attribute is expected to be named "N" and
    /// have type `[f32;3]`.  The offsets attribute is expected to be named "offsets" and have type
    /// `f32`.  This function initializes the `vertices`, `vertex_normals`, `sample_values`
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

    pub fn background_field(&mut self, bg_field: BackgroundFieldParams) -> &mut Self {
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
    fn vertex_positions_from_mesh<T: Real, M: VertexMesh<f64>>(mesh: &M) -> Vec<Vector3<T>> {
        mesh.vertex_position_iter()
            .map(|&x| {
                Vector3(x)
                    .cast::<T>()
                    .expect("Failed to convert positions float type")
            })
            .collect()
    }
    /// A helper function to extract an offset vector from the mesh.
    fn vertex_offsets_from_mesh<T: Real, M: VertexMesh<f64>>(mesh: &M) -> Vec<T> {
        mesh.attrib_iter::<f32, VertexIndex>("offset")
            .map(|iter| iter.map(|&x| T::from(x).unwrap()).collect())
            .unwrap_or_else(|_| vec![T::zero(); mesh.num_vertices()])
    }

    /// A helper function to extract a normals vector from the mesh.
    /// The resulting vector is empty if no normals are found.
    fn vertex_normals_from_mesh<T: Real, M: VertexMesh<f64>>(mesh: &M) -> Vec<Vector3<T>> {
        mesh.attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| iter.map(|nml| Vector3(*nml).cast::<T>().unwrap()).collect())
            .unwrap_or_default()
    }

    /// A helper function to extract a satcked velocity vector from the mesh.
    /// The resulting vector is empty if no velocities are found.
    fn vertex_velocities_from_mesh<T: Real, M: VertexMesh<f64>>(mesh: &M) -> Vec<Vector3<T>> {
        mesh.attrib_iter::<[f32; 3], VertexIndex>("V")
            .map(|iter| iter.map(|vel| Vector3(*vel).cast::<T>().unwrap()).collect())
            .unwrap_or_else(|_| vec![Vector3::zeros(); mesh.num_vertices()])
    }

    /// A helper function to compute the dual topology of a triangle mesh. The dual topology
    /// corresponds to a list of indices to adjacent triangles on each vertex of the triangle mesh.
    /// If triangles is empty, then an empty vector is returned.
    pub(crate) fn compute_dual_topo(num_verts: usize, triangles: &[[usize; 3]]) -> Vec<Vec<usize>> {
        let mut dual_topo = Vec::new();

        if !triangles.is_empty() {
            // Compute the one ring and store it in the dual topo vectors.
            dual_topo.resize(num_verts, Vec::new());
            for (tri_idx, tri) in triangles.iter().enumerate() {
                for &vidx in tri {
                    dual_topo[vidx].push(tri_idx);
                }
            }
        }

        dual_topo
    }

    /// Given a triangle mesh, determine the smallest radius that will contain each triangle in the
    /// mesh. More precisely, find `r` such that `|x_t - c_t| <= r` for all vertices `x_t` and all triangles
    /// `t`, where `c_t` is the centroid of triangle `t`.
    pub(crate) fn compute_base_radius(trimesh: &TriMesh<f64>) -> f64 {
        use geo::ops::Centroid;
        use geo::prim::Triangle;
        use geo::mesh::VertexPositions;
        let pos = trimesh.vertex_positions();
        trimesh.face_iter().map(|f| {
            let tri = Triangle::from_indexed_slice(f.get(), pos);
            let c = tri.centroid();
            let verts = [tri.0, tri.1, tri.2];
            verts.into_iter().map(|&x| (x - c).norm_squared())
                .max_by(|a,b| a.partial_cmp(b).expect("Detected NaN. Please report this bug."))
                .unwrap() // we know there are 3 vertices.
        }).max_by(|a,b| a.partial_cmp(b).expect("Detected NaN. Please report this bug."))
        .expect("Empty triangle mesh.")
        .sqrt()
    }

    /// Builds the implicit surface. This function returns `None` when theres is not enough data to
    /// make a valid implict surface. For example if kernel radius is 0.0 or points is empty, this
    /// function will return `None`.
    pub fn build<T: Real + Send + Sync>(&self) -> Option<ImplicitSurface<T>>
    where
        Sample<T>: spade::SpatialObject,
    {
        let ImplicitSurfaceBuilder {
            kernel,
            bg_field,
            mesh,
            max_step,
            base_radius,
            sample_type,
        } = self.clone();
        // Cannot build an implicit surface when the radius is 0.0.
        match kernel {
            KernelType::Interpolating { radius_multiplier }
            | KernelType::Approximate { radius_multiplier, .. }
            | KernelType::Cubic { radius_multiplier } => {
                if radius_multiplier == 0.0 {
                    return None;
                }
            }
            _ => {} // Radius is not used in global support kernels
        }

        let (base_radius, samples, vertices, triangles) = match mesh {
            SamplesMesh::PointCloud(ptcloud) => {
                let vertices = Self::vertex_positions_from_mesh(&ptcloud);

                let base_radius = if base_radius.is_none() {
                    return None; // Can't automatically determine the base radius.
                } else {
                    base_radius.unwrap()
                };

                if sample_type == SampleType::Face {
                    return None; // Given an incompatible sample type.
                }

                let sample_values = Self::vertex_offsets_from_mesh(&ptcloud);
                assert_eq!(vertices.len(), sample_values.len());

                let vertex_normals = Self::vertex_normals_from_mesh(&ptcloud);
                if vertex_normals.is_empty() {
                    return None; // Must have normals on vertices for point clouds.
                }

                let velocities = Self::vertex_velocities_from_mesh(&ptcloud);

                assert_eq!(vertices.len(), vertex_normals.len());
                let samples = Samples {
                    points: vertices.clone(),
                    normals: vertex_normals,
                    velocities,
                    values: sample_values,
                };

                (base_radius, samples, vertices, Vec::new())
            }
            SamplesMesh::TriMesh(mesh) => {
                let base_radius = base_radius.unwrap_or_else(|| Self::compute_base_radius(mesh));

                let vertices = Self::vertex_positions_from_mesh(mesh);
                let triangles = reinterpret::reinterpret_slice(mesh.faces()).to_vec();

                // Build the samples.
                let samples = match sample_type {
                    SampleType::Vertex => {
                        let sample_values = Self::vertex_offsets_from_mesh(mesh);
                        assert_eq!(vertices.len(), sample_values.len());

                        let vertex_normals = Self::vertex_normals_from_mesh(mesh);
                        let mut samples = Samples::new_vertex_samples(
                            &triangles,
                            &vertices,
                            if vertex_normals.is_empty() {
                                None
                            } else {
                                Some(&vertex_normals)
                            },
                            sample_values,
                        );

                        samples.velocities = Self::vertex_velocities_from_mesh(mesh);
                        assert_eq!(vertices.len(), samples.velocities.len());
                        samples
                    }
                    SampleType::Face => {
                        // Can't create a face centered potential if there are no faces.
                        if triangles.is_empty() {
                            return None;
                        }

                        // One sample per triangle.
                        let sample_values = mesh
                            .attrib_iter::<f32, FaceIndex>("offset")
                            .map(|iter| iter.map(|&x| T::from(x).unwrap()).collect())
                            .unwrap_or_else(|_| vec![T::zero(); mesh.num_faces()]);
                        assert_eq!(triangles.len(), sample_values.len());

                        let mut samples =
                            Samples::new_triangle_samples(&triangles, &vertices, sample_values);

                        let vertex_velocities = Self::vertex_velocities_from_mesh(mesh);

                        let mut face_velocities = vec![Vector3::zeros(); triangles.len()];

                        for (&tri, fvel) in triangles.iter().zip(face_velocities.iter_mut()) {
                            for i in 0..3 {
                                *fvel += vertex_velocities[tri[i]];
                            }
                            *fvel /= T::from(3.0).unwrap();
                        }

                        samples.velocities = face_velocities;
                        samples
                    }
                };

                (base_radius, samples, vertices, triangles)
            }

            // Cannot build an implicit surface without sample points. This is an error.
            SamplesMesh::None => return None,
        };

        // Build the dual topology.
        let dual_topo = ImplicitSurfaceBuilder::compute_dual_topo(vertices.len(), &triangles);

        // Build the rtree.
        let rtree = build_rtree_from_samples(&samples);

        Some(ImplicitSurface {
            kernel,
            base_radius,
            bg_field_params: bg_field,
            spatial_tree: rtree,
            surface_topo: triangles,
            surface_vertex_positions: vertices,
            samples,
            max_step: T::from(max_step).unwrap(),
            query_neighbourhood: RefCell::new(Neighbourhood::new()),
            dual_topo,
            sample_type,
        })
    }
}
