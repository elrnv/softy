use super::*;
use geo::mesh::{topology::*, PointCloud, TriMesh, VertexMesh};
use num_traits::Zero;
use tensr::{Scalar, Vector3};

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

    /// Set the kernel type for this implicit surface.
    pub fn kernel(&mut self, kernel: KernelType) -> &mut Self {
        self.kernel = kernel;
        self
    }

    /// Set the base kernel radius.
    ///
    /// The true radius will be this value multiplied by the radius multiplier.
    pub fn base_radius(&mut self, base_radius: f64) -> &mut Self {
        self.base_radius = Some(base_radius);
        self
    }

    /// Initialize fit data using a vertex mesh which can include positions, normals and offsets
    /// as attributes on the mesh struct.
    ///
    /// The normals attribute is expected to be named "N" and have type `[f32;3]`.  The offsets
    /// attribute is expected to be named "offsets" and have type `f32`.  This function initializes
    /// the `vertices`, `vertex_normals`, `sample_values` and sets `sample_type` to
    /// `SampleType::Vertex`.
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
    fn vertex_positions_from_mesh<T: Scalar, M: VertexMesh<f64>>(mesh: &M) -> Vec<[T; 3]> {
        mesh.vertex_position_iter()
            .map(|&x| Vector3::new(x).cast::<T>().into())
            .collect()
    }
    /// A helper function to extract an offset vector from the mesh.
    fn vertex_offsets_from_mesh<T: Scalar, M: VertexMesh<f64>>(mesh: &M) -> Vec<T> {
        mesh.attrib_iter::<f32, VertexIndex>("offset")
            .map(|iter| iter.map(|&x| T::from(x).unwrap()).collect())
            .unwrap_or_else(|_| vec![T::zero(); mesh.num_vertices()])
    }

    /// A helper function to extract a normals vector from the mesh.
    /// The resulting vector is empty if no normals are found.
    fn vertex_normals_from_mesh<T: Scalar, M: VertexMesh<f64>>(mesh: &M) -> Vec<[T; 3]> {
        mesh.attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| {
                iter.map(|nml| Vector3::new(*nml).cast::<T>().into())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// A helper function to extract a satcked velocity vector from the mesh.
    /// The resulting vector is empty if no velocities are found.
    fn vertex_velocities_from_mesh<T: Scalar, M: VertexMesh<f64>>(mesh: &M) -> Vec<Vector3<T>> {
        mesh.attrib_iter::<[f32; 3], VertexIndex>("V")
            .map(|iter| iter.map(|vel| Vector3::new(*vel).cast::<T>()).collect())
            .unwrap_or_else(|_| vec![Vector3::zero(); mesh.num_vertices()])
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
        use geo::mesh::VertexPositions;
        use geo::ops::Centroid;
        let pos = trimesh.vertex_positions();
        trimesh
            .face_iter()
            .map(|f| {
                let tri = Triangle::from_indexed_slice(f, pos);
                let c = Vector3::new(tri.centroid());
                let verts: [[f64; 3]; 3] = [tri.0.into(), tri.1.into(), tri.2.into()];
                verts
                    .iter()
                    .map(|&x| (Vector3::new(x) - c).norm_squared())
                    .max_by(|a, b| {
                        a.partial_cmp(b)
                            .expect("Detected NaN. Please report this bug.")
                    })
                    .unwrap() // we know there are 3 vertices.
            })
            .max_by(|a, b| {
                a.partial_cmp(b)
                    .expect("Detected NaN. Please report this bug.")
            })
            .expect("Empty triangle mesh.")
            .sqrt()
    }

    /// Base radius can be determined automatically from the mesh with topology data.
    ///
    /// Point clouds do not, and hence require an explicit one to be specified.  This function
    /// computes the radius if needed, or otherwise reproduces the given one.  If no mesh is given,
    /// no radius is valid so we return `None`.
    fn build_base_radius(&self) -> Option<f64> {
        match &self.mesh {
            SamplesMesh::PointCloud(_) => {
                if self.base_radius.is_none() {
                    None // Can't automatically determine the base radius.
                } else {
                    Some(self.base_radius.unwrap())
                }
            }
            SamplesMesh::TriMesh(mesh) => Some(
                self.base_radius
                    .unwrap_or_else(|| Self::compute_base_radius(mesh)),
            ),
            SamplesMesh::None => None,
        }
    }

    /// Builds the base for any implicit surface.
    ///
    /// This function returns `None` when there is not enough data to make a valid implict surface.
    /// For example if base radius is 0.0 or points is empty, this function will return `None`.
    fn build_base<T: Real>(&self) -> Option<ImplicitSurfaceBase<T>> {
        let ImplicitSurfaceBuilder {
            bg_field,
            mesh,
            sample_type,
            ..
        } = self.clone();

        let (samples, vertices, triangles) = match mesh {
            SamplesMesh::PointCloud(ptcloud) => {
                let vertices = Self::vertex_positions_from_mesh(&ptcloud);

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
                    positions: vertices.clone(),
                    normals: vertex_normals,
                    velocities,
                    values: sample_values,
                };

                (samples, vertices, Vec::new())
            }
            SamplesMesh::TriMesh(mesh) => {
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

                        let mut face_velocities = vec![Vector3::zero(); triangles.len()];

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

                (samples, vertices, triangles)
            }

            // Cannot build an implicit surface without sample points. This is an error.
            SamplesMesh::None => return None,
        };

        // Build the dual topology.
        let dual_topo = ImplicitSurfaceBuilder::compute_dual_topo(vertices.len(), &triangles);

        let spatial_tree = build_rtree_from_samples(&samples);

        Some(ImplicitSurfaceBase {
            bg_field_params: bg_field,
            surface_topo: triangles,
            surface_vertex_positions: vertices,
            samples,
            dual_topo,
            sample_type,
            spatial_tree,
        })
    }

    /// Builds the implicit surface. This function returns `None` when there is not enough data to
    /// make a valid implict surface. For example if kernel radius is 0.0 or points is empty, this
    /// function will return `None`.
    pub fn build_generic<T: Real>(&self) -> Option<ImplicitSurface<T>>
    where
        Sample<T>: rstar::RTreeObject,
    {
        if let KernelType::Hrbf = self.kernel {
            let surf_base = Box::new(self.build_base()?);
            return Some(ImplicitSurface::Hrbf(HrbfSurface { surf_base }));
        }

        Some(ImplicitSurface::MLS(self.build_mls()?))
    }

    /// Builds the a local mls implicit surface. This function returns `None` when there is not enough data to
    /// make a valid implict surface. For example if kernel radius is 0.0 or points is empty, this
    /// function will return `None`.
    pub fn build_local_mls<T: Real>(&self) -> Option<LocalMLS<T>>
    where
        Sample<T>: rstar::RTreeObject,
    {
        // Cannot build a local implicit surface when the radius is 0.0 or infinite.
        match self.kernel {
            KernelType::Interpolating { radius_multiplier }
            | KernelType::Approximate {
                radius_multiplier, ..
            }
            | KernelType::Cubic { radius_multiplier } => {
                if radius_multiplier == 0.0 {
                    return None;
                }
            }
            KernelType::Global { .. } | KernelType::Hrbf => {
                return None;
            }
        }

        let surf_base = Box::new(self.build_base()?);
        let base_radius = self.build_base_radius()?;

        match self.kernel {
            KernelType::Interpolating { .. }
            | KernelType::Cubic { .. }
            | KernelType::Approximate { .. } => Some(LocalMLS {
                kernel: self.kernel.into(),
                base_radius,
                max_step: T::from(self.max_step).unwrap(),
                surf_base,
            }),
            _ => None,
        }
    }

    /// Builds an mls based implicit surface. This function returns `None` when there is not enough data to
    /// make a valid implict surface. For example if kernel radius is 0.0 or points is empty, this
    /// function will return `None`.
    pub fn build_mls<T: Real>(&self) -> Option<MLS<T>>
    where
        Sample<T>: rstar::RTreeObject,
    {
        if let KernelType::Hrbf = self.kernel {
            return None;
        }

        if let KernelType::Global { .. } = self.kernel {
            let surf_base = Box::new(self.build_base()?);
            return Some(MLS::Global(GlobalMLS {
                kernel: self.kernel.into(),
                surf_base,
            }));
        }

        Some(MLS::Local(self.build_local_mls()?))
    }
}
