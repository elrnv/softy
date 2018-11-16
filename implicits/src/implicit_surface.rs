//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::geo::math::{Matrix3, Vector3};
use crate::geo::mesh::{topology::VertexIndex, VertexMesh};
use crate::geo::prim::Triangle;
use crate::geo::Real;
use crate::kernel::{self, KernelType, SphericalKernel};
use rayon::prelude::*;
use spade::{rtree::RTree, BoundingRect, SpatialObject};
use std::cell::{Ref, RefCell};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrientedPoint {
    pub index: i32,
    pub pos: Vector3<f64>,
    pub nml: Vector3<f64>,
}

impl SpatialObject for OrientedPoint {
    type Point = [f64; 3];

    fn mbr(&self) -> BoundingRect<Self::Point> {
        BoundingRect::from_point(self.pos.into())
    }

    fn distance2(&self, point: &Self::Point) -> f64 {
        (Vector3(*point) - self.pos).norm_squared()
    }
}

pub fn oriented_points_iter<'a, V3>(
    points: &'a [V3],
    normals: &'a [V3],
) -> impl Iterator<Item = OrientedPoint> + Clone + 'a
where
    V3: Into<Vector3<f64>> + Copy,
{
    normals
        .iter()
        .zip(points.iter())
        .enumerate()
        .map(|(i, (&nml, &pos))| OrientedPoint {
            index: i as i32,
            pos: pos.into(),
            nml: nml.into(),
        })
}

pub fn points_and_normals_from_mesh<'a, M: VertexMesh<f64>>(
    mesh: &M,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let points: Vec<Vector3<f64>> = mesh
        .vertex_position_iter()
        .map(|&x| -> Vector3<f64> {
            Vector3(x)
                .cast::<f64>()
                .expect("Failed to convert positions to f64")
        })
        .collect();

    let normals = mesh.attrib_iter::<[f32; 3], VertexIndex>("N").ok().map_or(
        vec![Vector3::zeros(); points.len()],
        |iter| {
            iter.map(|&nml| Vector3(nml).cast::<f64>().unwrap())
                .collect()
        },
    );

    (points, normals)
}

pub fn build_rtree(
    oriented_points_iter: impl Iterator<Item = OrientedPoint>,
) -> RTree<OrientedPoint> {
    let mut rtree = RTree::new();
    for pt in oriented_points_iter {
        rtree.insert(pt);
    }
    rtree
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
            ..
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
            max_step: self.max_step,
            neighbour_cache: RefCell::new(NeighbourCache::new()),
            dual_topo,
        })
    }
}

/// Cache neighbouring sample points for each query point.
/// Note that this determines the entire sparsity structure of the query point neighbourhoods.
#[derive(Clone, Debug, PartialEq)]
struct NeighbourCache {
    /// For each query point with a non-trivial neighbourhood of sample points, record the
    /// neighbours indices in this vector. Along with the vector of neighbours, store the index of
    /// the original query point, because this vector is a sparse subset of all the query points.
    pub points: Vec<(usize, Vec<usize>)>,

    /// Marks the cache valid or not. If this flag is false, the cache needs to be recomputed, but
    /// we can reuse the already allocated memory.
    pub valid: bool,
}

impl NeighbourCache {
    fn new() -> Self {
        NeighbourCache {
            points: Vec::new(),
            valid: false,
        }
    }
}

/// A set of data stored on each sample for the implicit surface.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sample<T: Real> {
    /// Index of the sample in the original vector or array.
    pub index: usize,
    /// Position of the sample in 3D space.
    pub pos: Vector3<T>,
    /// Normal of the sample.
    pub nml: Vector3<T>,
    /// Offset stored at the sample point.
    pub off: f64,
}

/// Sample points that define the implicit surface including the point positions, normals and
/// offsets.
#[derive(Clone, Debug, PartialEq)]
pub struct Samples<T: Real> {
    /// Sample point positions defining the implicit surface.
    pub points: Vec<Vector3<T>>,
    /// Normals that define the potential field gradient at every sample point.
    pub normals: Vec<Vector3<T>>,

    /// Potential values at the interpolating points. These offsets indicate the values to
    /// match by interpolating implicit surfaces. This means that the zero iso-surface will not
    /// necessarily pass through the given points.
    pub offsets: Vec<f64>,
}

/// A view into to the positions, normals and offsets of the sample points. This view need not be
/// contiguous as it often isnt.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplesView<'i, 'd: 'i, T: Real> {
    /// Indices into the sample points.
    indices: &'i [usize],
    /// Sample point positions defining the implicit surface.
    points: &'d [Vector3<T>],
    /// Normals that define the potential field gradient at every sample point.
    normals: &'d [Vector3<T>],
    /// Potential values at the interpolating points. These offsets indicate the values to
    /// match by interpolating implicit surfaces. This means that the zero iso-surface will not
    /// necessarily pass through the given points.
    offsets: &'d [f64],
}

impl<'i, 'd: 'i, T: Real> SamplesView<'i, 'd, T> {
    /// Create a view of samples with a given indices slice into the provided samples.
    #[inline]
    pub fn new(indices: &'i [usize], samples: &'d Samples<T>) -> Self {
        SamplesView {
            indices,
            points: samples.points.as_slice(),
            normals: samples.normals.as_slice(),
            offsets: samples.offsets.as_slice(),
        }
    }

    #[inline]
    pub fn from_view(indices: &'i [usize], samples: SamplesView<'i, 'd, T>) -> Self {
        SamplesView {
            indices,
            points: samples.points.clone(),
            normals: samples.normals.clone(),
            offsets: samples.offsets.clone(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub fn iter(&'i self) -> impl Iterator<Item = Sample<T>> + 'i {
        let SamplesView {
            ref indices,
            ref points,
            ref normals,
            ref offsets,
        } = self;
        indices.iter().map(move |&i| Sample {
            index: i,
            pos: points[i],
            nml: normals[i],
            off: offsets[i],
        })
    }

    /// Consuming iterator.
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = Sample<T>> + 'i {
        let SamplesView {
            indices,
            points,
            normals,
            offsets,
        } = self;
        indices.into_iter().map(move |&i| Sample {
            index: i,
            pos: points[i],
            nml: normals[i],
            off: offsets[i],
        })
    }

    #[inline]
    pub fn points(&'d self) -> &'d [Vector3<T>] {
        self.points
    }

    #[inline]
    pub fn normals(&'d self) -> &'d [Vector3<T>] {
        self.normals
    }
}

/// Precomputed data used for background potential computation.
#[derive(Copy, Clone, Debug)]
pub(crate) enum BackgroundPotentialValue<T: Real> {
    /// The value of the background potential at the query point.
    Constant(T),
    /// A Dynamic potential is computed based on the distance to the closest sample point.
    ClosestSampleDistance,
}

/// This struct represents the data needed to compute a background potential at a local query
/// point. This struct also conviently computes useful information about the neighbourhood (like
/// closest distance to a sample point) that can be reused elsewhere.
#[derive(Clone, Debug)]
pub(crate) struct BackgroundPotential<'a, T: Real, K: SphericalKernel<T> + Clone + std::fmt::Debug>
{
    /// Position of the point at which we should evaluate the potential field.
    pub query_pos: Vector3<T>,
    /// Samples that influence the potential field.
    pub samples: SamplesView<'a, 'a, T>,
    /// Data needed to compute the background potential value and its derivative.
    pub bg_potential_value: BackgroundPotentialValue<T>,
    /// The reciprocal of the sum of all the weights in the neighbourhood of the query point.
    pub weight_sum_inv: T,
    /// The spherical kernel used to compute the weights.
    pub kernel: K,
    /// The distance to the closest sample.
    pub closest_sample_dist: T,
    /// Displacement vector to the query point from the closest sample point.
    pub closest_sample_disp: Vector3<T>,
    /// The index of the closest sample point.
    pub closest_sample_index: usize,
    /// Radius of the neighbourhood of the query point.
    pub radius: T,
}

impl<'a, T: Real, K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a>
    BackgroundPotential<'a, T, K>
{
    /// Pass in the unnormalized weight sum excluding the weight for the background potential.
    fn new(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        radius: T,
        kernel: K,
        dynamic_bg: bool,
    ) -> Self {
        // Precompute data about the closest sample point (displacement, index and distance).
        let min_sample = samples
            .iter()
            .map(|Sample { index, pos, .. }| {
                let disp = q - pos;
                (index, disp, disp.norm_squared())
            })
            .min_by(|(_, _, d0), (_, _, d1)| {
                d0.partial_cmp(d1)
                    .expect("Detected NaN. Please report this bug.")
            });

        if min_sample.is_none() {
            panic!("No surface samples found. Please report this bug.");
        }

        let (closest_sample_index, closest_sample_disp, mut closest_sample_dist) =
            min_sample.unwrap();
        closest_sample_dist = closest_sample_dist.sqrt();

        // Compute the weight sum here. This will be available to the usesr of bg data.
        let mut weight_sum = T::zero();
        for Sample { pos, .. } in samples.iter() {
            let w = kernel.with_closest_dist(closest_sample_dist).eval(q, pos);
            weight_sum += w;
        }

        // Initialize the background potential struct.
        let mut bg = BackgroundPotential {
            query_pos: q,
            samples,
            bg_potential_value: if dynamic_bg {
                BackgroundPotentialValue::ClosestSampleDistance
            } else {
                BackgroundPotentialValue::Constant(T::zero()) // TODO: Support non-zero constant potentials
            },
            weight_sum_inv: T::zero(), // temporarily set the weight_sum_inv
            kernel,
            closest_sample_dist,
            closest_sample_disp,
            closest_sample_index,
            radius,
        };

        // Finalize the weight sum reciprocal.
        bg.weight_sum_inv = T::one() / (weight_sum + bg.background_weight());

        bg
    }

    fn closest_sample_dist(&self) -> T {
        self.closest_sample_dist
    }

    fn weight_sum_inv(&self) -> T {
        self.weight_sum_inv
    }

    fn background_weight(&self) -> T {
        self.kernel.f(self.radius - self.closest_sample_dist)
    }

    fn background_weight_gradient(&self, index: usize) -> Vector3<T> {
        if index == self.closest_sample_index {
            self.closest_sample_disp
                * (self.kernel.df(self.radius - self.closest_sample_dist)
                    / self.closest_sample_dist)
        } else {
            Vector3::zeros()
        }
    }

    /// Compute the unnormalized weighted background potential value. This is typically very
    /// simple, but the caller must remember to multiply it by the `weight_sum_inv` to get the true
    /// background potential contribution.
    fn compute_unnormalized_weighted_potential(&self) -> T {
        // Unpack background data.
        let BackgroundPotential {
            bg_potential_value,
            closest_sample_dist: dist,
            ..
        } = *self;

        self.background_weight()
            * match bg_potential_value {
                BackgroundPotentialValue::Constant(potential) => potential,
                BackgroundPotentialValue::ClosestSampleDistance => dist,
            }
    }

    /// Compute background potential derivative contribution.
    /// Compute derivative if the closest point is in the neighbourhood. Otherwise we
    /// assume the background potential is constant.
    pub(crate) fn compute_jacobian(&self) -> impl Iterator<Item = Vector3<T>> + 'a {
        // Unpack background data.
        let BackgroundPotential {
            query_pos: q,
            samples,
            bg_potential_value,
            weight_sum_inv,
            kernel,
            closest_sample_dist: dist,
            closest_sample_disp: disp,
            closest_sample_index,
            radius,
        } = *self;

        // The unnormalized weight evaluated at the distance to the boundary of the
        // neighbourhood.
        let wb = kernel.f(radius - dist);

        // Gradient of the unnormalized weight evaluated at the distance to the
        // boundary of the neighbourhood.
        let dwbdp = self.background_weight_gradient(closest_sample_index);

        samples.into_iter().map(move |Sample { index, pos, .. }| {
            // Gradient of the unnormalized weight for the current sample point.
            let dwdp = -kernel.with_closest_dist(dist).grad(q, pos);

            // This term is valid for constant or dynamic background potentials.
            let constant_term =
                |potential: T| dwdp * (-potential * wb * weight_sum_inv * weight_sum_inv);

            match bg_potential_value {
                BackgroundPotentialValue::Constant(potential) => constant_term(potential),
                BackgroundPotentialValue::ClosestSampleDistance => {
                    let mut grad = constant_term(dist);

                    if index == closest_sample_index {
                        //grad += dwbdp * dist + disp * (wb / dist)
                        grad += dwbdp * (dist * weight_sum_inv * (T::one() - weight_sum_inv * wb))
                            - disp * (wb * weight_sum_inv / dist)
                    }

                    grad
                }
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct ImplicitSurface {
    /// The type of kernel to use for fitting the data.
    kernel: KernelType,

    /// Toggle for computing a simple background potential that will be mixed in with the local
    /// potentials. If `true`, this background potential will be automatically computed, if
    /// `false`, the values in the input will be used as the background potential to be mixed in.
    background_potential: bool,

    /// Local search tree for fast proximity queries.
    spatial_tree: RTree<OrientedPoint>,

    /// Surface triangles representing the surface discretization to be approximated.
    /// This topology also defines the normals to the surface.
    surface_topo: Vec<[usize; 3]>,

    /// Sample points defining the entire implicit surface.
    samples: Samples<f64>,

    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative may be denser
    /// than then needed, which typically results in slower performance.  If it is set too low,
    /// there may be errors in the derivative. It is the callers responsibility to set this step
    /// accurately using `update_max_step`. If the implicit surface is not changing, leave this at
    /// 0.0.
    max_step: f64,

    /// Cache the neighbouring sample points for each query point we see. This cache can be
    /// invalidated explicitly when the sparsity pattern is expected to change. This is wrapped in
    /// a `RefCell` because it may be updated in non mutable functions since it's a cache.
    neighbour_cache: RefCell<NeighbourCache>,

    /// Vertex neighbourhood topology. For each vertex, this vector stores all the indices to
    /// adjacent triangles.
    dual_topo: Vec<Vec<usize>>,
}

impl ImplicitSurface {
    const PARALLEL_CHUNK_SIZE: usize = 5000;

    /// Number of sample triangles.
    pub fn num_triangles(&self) -> usize {
        return self.surface_topo.len();
    }

    /// Number of sample points used to represent the implicit surface.
    pub fn num_points(&self) -> usize {
        return self.samples.points.len();
    }

    /// Compute unnormalized area weighted vertex normals given a triangle topology.
    /// Having this function be static and generic helps us test the normal derivatives.
    pub(crate) fn compute_vertex_area_normals<T: Real>(
        surf_topo: &[[usize; 3]],
        points: &[Vector3<T>],
        normals: &mut [Vector3<T>],
    ) {
        // Clear the normals.
        for nml in normals.iter_mut() {
            *nml = Vector3::zeros();
        }

        for tri_indices in surf_topo.iter() {
            let tri = Triangle::from_indexed_slice(tri_indices, &points);
            let area_nml = tri.area_normal();
            normals[tri_indices[0]] += area_nml;
            normals[tri_indices[1]] += area_nml;
            normals[tri_indices[2]] += area_nml;
        }
    }

    /// Compute the gradient vector product of the `compute_vertex_unit_normals` function with respect to
    /// vertices given in the sample view.
    ///
    /// This function returns an iterator with the same size as `samples.len()`.
    ///
    /// Note that the product vector is given by a closure `dx` which must give a valid vector
    /// value for any vertex index, however not all indices will be used since only the
    /// neighbourhood of vertex at `index` will have non-zero gradients.
    pub(crate) fn compute_vertex_unit_normals_gradient_products<'a, T: Real, F>(
        samples: SamplesView<'a, 'a, T>,
        surface_topo: &'a [[usize; 3]],
        dual_topo: &'a [Vec<usize>],
        mut dx: F,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        F: FnMut(Sample<T>) -> Vector3<T> + 'a,
    {
        samples.clone().into_iter().map(move |sample| {
            let Sample { index, nml, .. } = sample;
            let norm_inv = T::one() / nml.norm();
            // Compute the normal component of the derivative
            let nml_proj = Matrix3::identity() - nml * (nml.transpose() * (norm_inv * norm_inv));
            let mut nml_deriv = Vector3::zeros();
            // Look at the ring of triangles around the vertex with respect to which we are
            // taking the derivative.
            for &tri_idx in dual_topo[index].iter() {
                let tri_indices = &surface_topo[tri_idx];
                // Pull contributions from all neighbours on the surface, not just ones part of the
                // neighbourhood,
                let tri = Triangle::from_indexed_slice(tri_indices, samples.points());
                let nml_grad = tri.area_normal_gradient(
                    tri_indices
                        .iter()
                        .position(|&j| j == index)
                        .expect("Triangle mesh topology corruption."),
                );
                let mut tri_grad = nml_proj * (dx(sample) * norm_inv);
                for sample in SamplesView::from_view(tri_indices, samples.clone()).into_iter() {
                    if sample.index != index {
                        let normk_inv = T::one() / sample.nml.norm();
                        let nmlk_proj = Matrix3::identity()
                            - sample.nml * (sample.nml.transpose() * (normk_inv * normk_inv));
                        tri_grad += nmlk_proj * (dx(sample) * normk_inv);
                    }
                }
                nml_deriv += nml_grad * tri_grad;
            }
            nml_deriv
        })
    }

    /// Update the normals stored on the `ImplicitSurface`. This is usually called when the points
    /// have changed. Note that these are unnormalized to make it eaiser to compute derivatives.
    pub fn recompute_normals(&mut self) {
        let ImplicitSurface {
            samples:
                Samples {
                    ref points,
                    ref mut normals,
                    ..
                },
            ref surface_topo,
            ..
        } = self;

        Self::compute_vertex_area_normals(surface_topo, points, normals);
    }

    /// Update points and normals (oriented points) using an iterator.
    pub fn update_points<I>(&mut self, points_iter: I)
    where
        I: Iterator<Item = [f64; 3]>,
    {
        for (p, new_p) in self.samples.points.iter_mut().zip(points_iter) {
            *p = new_p.into();
        }

        self.recompute_normals();

        let ImplicitSurface {
            samples:
                Samples {
                    ref mut points,
                    ref mut normals,
                    ..
                },
            ref mut spatial_tree,
            ..
        } = self;

        *spatial_tree = build_rtree(oriented_points_iter(points, normals));
    }

    /// Compute neighbour cache if it has been invalidated
    pub fn cache_neighbours(&self, query_points: &[[f64; 3]]) {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            samples:
                Samples {
                    ref points,
                    ref normals,
                    ..
                },
            max_step,
            ..
        } = self;
        match *kernel {
            KernelType::Interpolating { radius }
            | KernelType::Approximate { radius, .. }
            | KernelType::Cubic { radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                self.cached_neighbours_borrow(query_points, neigh);
            }
            KernelType::Global { .. } | KernelType::Hrbf => {
                // Global kernel, all points are neighbours
                let neigh = |_| oriented_points_iter(points, normals);
                self.cached_neighbours_borrow(query_points, neigh);
            }
        }
    }

    /// Compute neighbour cache if it hasn't been computed yet. Return the neighbours of the given
    /// query points. Note that the cache must be invalidated explicitly, there is no real way to
    /// automatically cache results because both: query points and sample points may change
    /// slightly, but we expect the neighbourhood information to remain the same.
    fn cached_neighbours_borrow<'a, I, N>(
        &'a self,
        query_points: &[[f64; 3]],
        neigh: N,
    ) -> Ref<'a, NeighbourCache>
    where
        I: Iterator<Item = OrientedPoint> + 'a,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        if !self.neighbour_cache.borrow().valid {
            let mut cache = self.neighbour_cache.borrow_mut();
            cache.points.clear();
            cache.points.reserve(query_points.len());

            for (qi, q) in query_points.iter().enumerate() {
                let neighbours_iter = neigh(*q);
                // Below we try to reuse the allocated memory by previously cached members for
                // points.

                // Cache points
                for (iter_count, ni) in neighbours_iter.map(|op| op.index).enumerate() {
                    if iter_count == 0 {
                        // Initialize entry if there are any neighbours.
                        cache.points.push((qi, Vec::new()));
                    }

                    debug_assert!(!cache.points.is_empty());
                    let last_mut = cache.points.last_mut().unwrap();

                    last_mut.1.push(ni as usize);
                }
            }

            cache.valid = true;
        }

        self.neighbour_cache.borrow()
    }

    /// Set `neighbour_cache` to None. This triggers recomputation of the neighbour cache next time
    /// the potential or its derivatives are requested.
    pub fn invalidate_neighbour_cache(&self) {
        let mut cache = self.neighbour_cache.borrow_mut();
        cache.valid = false;
    }

    /// The number of query points currently in the cache.
    pub fn num_cached_query_points(&self) -> usize {
        let cache = self.neighbour_cache.borrow();
        cache.points.len()
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: f64) {
        self.max_step = max_step;
    }

    /// Compute the implicit surface potential.
    pub fn potential(
        &self,
        query_points: &[[f64; 3]],
        out_potential: &mut [f64],
    ) -> Result<(), super::Error> {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            samples:
                Samples {
                    ref points,
                    ref normals,
                    ref offsets,
                },
            max_step,
            ..
        } = *self;

        match *kernel {
            KernelType::Interpolating { radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kern = kernel::LocalInterpolating::new(radius);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Approximate { tolerance, radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Cubic { radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kern = kernel::LocalCubic::new(radius);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Global { tolerance } => {
                // Global kernel, all points are neighbours
                let neigh = |_| oriented_points_iter(points, normals);
                let radius = 1.0;
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Hrbf => {
                // Global kernel, all points are neighbours.
                Self::compute_hrbf(query_points, points, normals, offsets, out_potential)
            }
        }
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls<'a, I, K, N>(
        &self,
        query_points: &[[f64; 3]],
        radius: f64,
        kernel: K,
        neigh: N,
        out_potential: &mut [f64],
    ) -> Result<(), super::Error>
    where
        I: Iterator<Item = OrientedPoint> + 'a,
        K: SphericalKernel<f64> + Copy + std::fmt::Debug + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        let neigh = self.cached_neighbours_borrow(query_points, neigh);

        let ImplicitSurface {
            ref samples,
            background_potential,
            ..
        } = *self;

        zip!(
            neigh
                .points
                .par_iter()
                .map(|(qi, neigh)| (query_points[*qi], neigh)),
            out_potential.par_iter_mut()
        )
        .for_each(|((q, neighbours), potential)| {
            Self::compute_local_potential_at(
                Vector3(q),
                SamplesView::new(neighbours, samples),
                radius,
                kernel,
                background_potential,
                potential,
            );
        });

        Ok(())
    }

    /// Compute the potential at a given query point. If the potential is invalid or nieghbourhood
    /// is empty, `potential` is not modified, otherwise it's updated.
    /// Note: passing the output parameter potential as a mut reference allows us to optionally mix
    /// a preinitialized custom global potential field with the local potential.
    pub(crate) fn compute_local_potential_at<T: Real, K>(
        q: Vector3<T>,
        samples: SamplesView<T>,
        radius: f64,
        kernel: K,
        dynamic_bg: bool,
        potential: &mut T,
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        if samples.is_empty() {
            return;
        }

        let radius = T::from(radius).unwrap();
        let bg = BackgroundPotential::new(q, samples, radius, kernel, dynamic_bg);
        let closest_d = bg.closest_sample_dist();

        // Generate a background potential field for every query point. This will be mixed
        // in with the computed potentials for local methods.
        *potential = bg.compute_unnormalized_weighted_potential();

        let mut numerator = T::zero();
        for Sample { pos, nml, off, .. } in samples.iter() {
            let w = kernel.with_closest_dist(closest_d).eval(q, pos);
            let p = T::from(off).unwrap() + nml.dot(q - pos) / nml.norm();

            numerator += w * p;
        }

        *potential = (*potential + numerator) * bg.weight_sum_inv();
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn num_surface_jacobian_entries(&self) -> usize {
        let cache = self.neighbour_cache.borrow();
        cache.points.iter().map(|(_, pts)| pts.len() * 3).sum()
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)>, super::Error> {
        match self.kernel {
            KernelType::Approximate { .. } => Ok(self.mls_surface_jacobian_indices_iter()),
            _ => Err(super::Error::UnsupportedKernel),
        }
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices(
        &self,
        rows: &mut [usize],
        cols: &mut [usize],
    ) -> Result<(), super::Error> {
        match self.kernel {
            KernelType::Approximate { .. } => Ok(self.mls_surface_jacobian_indices(rows, cols)),
            _ => Err(super::Error::UnsupportedKernel),
        }
    }

    pub fn surface_jacobian_values(
        &self,
        query_points: &[[f64; 3]],
        values: &mut [f64],
    ) -> Result<(), super::Error> {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            max_step,
            ..
        } = *self;

        match *kernel {
            KernelType::Approximate { tolerance, radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kernel = kernel::LocalApproximate::new(radius, tolerance);
                self.mls_surface_jacobian_values(query_points, radius, kernel, neigh, values);
                Ok(())
            }
            _ => Err(super::Error::UnsupportedKernel),
        }
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    fn mls_surface_jacobian_indices_iter(&self) -> impl Iterator<Item = (usize, usize)> {
        let ImplicitSurface {
            ref neighbour_cache,
            ..
        } = *self;
        let cache = neighbour_cache.borrow();
        cache
            .points
            .clone()
            .into_iter()
            .enumerate()
            .flat_map(move |(row, (_, nbr_points))| {
                nbr_points
                    .into_iter()
                    .flat_map(move |col| (0..3).map(move |i| (row, 3 * col + i)))
            })
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    fn mls_surface_jacobian_indices(&self, rows: &mut [usize], cols: &mut [usize]) {
        // For each row
        let cache = self.neighbour_cache.borrow();
        let row_col_iter =
            cache
                .points
                .iter()
                .enumerate()
                .flat_map(move |(row, (_, nbr_points))| {
                    nbr_points
                        .iter()
                        .cloned()
                        .flat_map(move |col| (0..3).map(move |i| (row, 3 * col + i)))
                });

        for ((row, col), out_row, out_col) in zip!(row_col_iter, rows.iter_mut(), cols.iter_mut()) {
            *out_row = row;
            *out_col = col;
        }
    }

    fn mls_surface_jacobian_values<'a, I, K, N>(
        &self,
        query_points: &[[f64; 3]],
        radius: f64,
        kernel: K,
        neigh: N,
        values: &mut [f64],
    ) where
        I: Iterator<Item = OrientedPoint> + 'a,
        K: SphericalKernel<f64> + std::fmt::Debug + Copy + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        let ImplicitSurface {
            ref samples,
            ref dual_topo,
            ref surface_topo,
            background_potential,
            ..
        } = *self;

        let neigh = self.cached_neighbours_borrow(query_points, neigh);

        // For each row
        let jac_iter = neigh
            .points
            .iter()
            .map(|(qi, neigh)| (query_points[*qi], neigh))
            .flat_map(move |(q, nbr_points)| {
                Self::compute_jacobian_at(
                    Vector3(q),
                    SamplesView::new(nbr_points, samples),
                    radius,
                    kernel,
                    background_potential,
                    surface_topo,
                    dual_topo,
                )
            });
        let value_vecs: &mut [[f64; 3]] = reinterpret::reinterpret_mut_slice(values);

        jac_iter
            .zip(value_vecs.iter_mut())
            .for_each(|(new_vec, vec)| {
                *vec = new_vec.into();
            });
    }

    /// Compute the jacobian for the implicit surface potential given by the samples with the
    /// specified kernel.
    pub(crate) fn compute_jacobian_at<'a, T: Real, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        radius: f64,
        kernel: K,
        dynamic_bg: bool,
        surface_topo: &'a [[usize; 3]],
        dual_topo: &'a [Vec<usize>],
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        // Find the closest vertex for background potential derivative.
        let radius = T::from(radius).unwrap();

        // Compute background potential derivative contribution.
        // Compute derivative if the closest point in the neighbourhood. Otherwise we
        // assume the background potential is constant.
        let bg = BackgroundPotential::new(q, samples, radius, kernel, dynamic_bg);

        let closest_d = bg.closest_sample_dist();

        // Background potential adds to the total weight sum, so we should get the updated weight
        // sum from there.
        let weight_sum_inv = bg.weight_sum_inv();
        let weight_sum_inv2 = weight_sum_inv * weight_sum_inv;

        // Background potential Jacobian.
        let bg_deriv_iter = bg.compute_jacobian();

        // Main function Jacobian.
        let main_deriv_iter = samples.clone().into_iter().map(
            move |Sample {
                      index,
                      pos,
                      nml,
                      off,
                  }| {
                let diff = q - pos;

                let norm_inv = T::one() / nml.norm();
                let unit_nml = nml * norm_inv;

                let mut dw_neigh = T::zero();

                for Sample {
                    pos: posk,
                    nml: nmlk,
                    off: offk,
                    ..
                } in samples.iter()
                {
                    let wk = kernel.with_closest_dist(closest_d).eval(q, posk);
                    let diffk = q - posk;
                    let pk = T::from(offk).unwrap() + (nmlk.dot(diffk) / nmlk.norm());
                    dw_neigh -= wk * pk;
                }

                let dw = -kernel.with_closest_dist(closest_d).grad(q, pos);
                let mut dw_p = dw * (dw_neigh * weight_sum_inv2);

                // Contribution from the background potential
                let dwb = bg.background_weight_gradient(index);
                dw_p += dwb * (dw_neigh * weight_sum_inv2);

                dw_p += dw * (weight_sum_inv * (T::from(off).unwrap() + unit_nml.dot(diff)));

                // Compute the normal component of the derivative
                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                let nml_deriv = unit_nml * (w * weight_sum_inv);
                dw_p - nml_deriv
            },
        );

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_deriv_iter = ImplicitSurface::compute_vertex_unit_normals_gradient_products(
            samples.clone(),
            &surface_topo,
            &dual_topo,
            move |Sample { pos, .. }| {
                let wk = kernel.with_closest_dist(closest_d).eval(q, pos);
                (q - pos) * (wk * weight_sum_inv)
            },
        );

        zip!(main_deriv_iter, nml_deriv_iter, bg_deriv_iter).map(|(m, n, b)| m + n + b)
    }

    fn compute_hrbf<V3>(
        query_points: &[[f64; 3]],
        points: &[V3],
        normals: &[V3],
        offsets: &[f64],
        out_potential: &mut [f64],
    ) -> Result<(), super::Error>
    where
        V3: Into<[f64; 3]> + Copy,
    {
        let pts: Vec<crate::na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = p.into();
                crate::na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = normals
            .iter()
            .map(|&n| crate::na::Vector3::from(n.into()))
            .collect();

        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());

        hrbf.fit_offset(&pts, offsets, &nmls);

        query_points
            .par_iter()
            .zip(out_potential.par_iter_mut())
            .for_each(|(q, potential)| {
                *potential = hrbf.eval(crate::na::Point3::from(*q));
            });

        Ok(())
    }

    /// Update points, normals and offsets from a given vertex mesh.
    pub fn update_oriented_points_with_mesh<M: VertexMesh<f64>>(&mut self, mesh: &M) {
        let ImplicitSurface {
            ref mut spatial_tree,
            samples:
                Samples {
                    ref mut points,
                    ref mut normals,
                    ref mut offsets,
                },
            ..
        } = self;

        let (pts, nmls) = points_and_normals_from_mesh(mesh);
        *points = pts;
        *normals = nmls;

        *spatial_tree = build_rtree(oriented_points_iter(points, normals));

        if mesh.num_vertices() == offsets.len() {
            // Update offsets if any.
            if let Ok(offset_iter) = mesh.attrib_iter::<f32, VertexIndex>("offset") {
                for (off, new_off) in offsets.iter_mut().zip(offset_iter.map(|&x| x as f64)) {
                    *off = new_off;
                }
            }
        } else {
            // Given point cloud has a different size than our internal represnetation. We need to
            // overwrite all internal data vectors.

            // Overwrite offsets or remove them.
            if let Ok(offset_iter) = mesh.attrib_iter::<f32, VertexIndex>("offset") {
                *offsets = offset_iter.map(|&x| x as f64).collect();
            } else {
                *offsets = vec![0.0; mesh.num_vertices()];
            }
        }

        // Check invariant.
        assert_eq!(points.len(), offsets.len());
        assert_eq!(normals.len(), offsets.len());
    }

    /*
     * The methods below are designed for debugging and visualization.
     */

    /// Compute the implicit surface potential on the given polygon mesh.
    pub fn compute_potential_on_mesh<F, M>(
        &self,
        mesh: &mut M,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<f64>,
    {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            samples:
                Samples {
                    ref points,
                    ref normals,
                    ref offsets,
                },
            background_potential: generate_bg_potential,
            ..
        } = *self;

        // Initialize potential with zeros.
        {
            let mut bg_potential = vec![0.0f32; mesh.num_vertices()];
            if generate_bg_potential {
                // Generate a background potential field for every sample point. This will be mixed
                // in with the computed potentials for local methods. Global methods like HRBF
                // ignore this field.
                for (pos, potential) in
                    zip!(mesh.vertex_positions().iter(), bg_potential.iter_mut())
                {
                    if let Some(nearest_neigh) = spatial_tree.nearest_neighbor(pos) {
                        let q = Vector3(*pos);
                        let p = nearest_neigh.pos;
                        let nml = nearest_neigh.nml;
                        *potential = (q - p).dot(nml) as f32;
                    } else {
                        return Err(super::Error::Failure);
                    }
                }
                mesh.set_attrib_data::<_, VertexIndex>("potential", &bg_potential)?;
            } else {
                mesh.attrib_or_add_data::<_, VertexIndex>("potential", &bg_potential)?;
            }
        }

        match *kernel {
            KernelType::Interpolating { radius } => {
                let radius2 = radius * radius;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kern = kernel::LocalInterpolating::new(radius);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Approximate { tolerance, radius } => {
                let radius2 = radius * radius;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Cubic { radius } => {
                let radius2 = radius * radius;
                let neigh = |q| {
                    spatial_tree
                        .lookup_in_circle(&q, &radius2)
                        .into_iter()
                        .cloned()
                };
                let kern = kernel::LocalCubic::new(radius);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Global { tolerance } => {
                let neigh = |_| oriented_points_iter(points, normals);
                let radius = 1.0;
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Hrbf => {
                Self::compute_hrbf_on_mesh(mesh, points, normals, offsets, interrupt)
            }
        }
    }
    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls_on_mesh<'a, I, K, N, F, M>(
        mesh: &mut M,
        radius: f64,
        kernel: K,
        neigh: N,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        I: Iterator<Item = OrientedPoint> + Clone + 'a,
        K: SphericalKernel<f64> + Copy + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<f64>,
    {
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut weight_attrib_data = vec![[0f32; 12]; mesh.num_vertices()];
        let sample_pos = mesh.vertex_positions().to_vec();

        for (q_chunk, num_neighs_chunk, neighs_chunk, weight_chunk, potential_chunk) in zip!(
            sample_pos.chunks(Self::PARALLEL_CHUNK_SIZE),
            num_neighs_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            neighs_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            weight_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            mesh.attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(Self::PARALLEL_CHUNK_SIZE)
        ) {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            zip!(
                q_chunk.par_iter(),
                num_neighs_chunk.par_iter_mut(),
                neighs_chunk.par_iter_mut(),
                weight_chunk.par_iter_mut(),
                potential_chunk.par_iter_mut()
            )
            .for_each(|(q, num_neighs, neighs, weight, potential)| {
                let neighbours_iter = neigh(*q);

                // Record number of neighbours in total.
                *num_neighs = -1 as i32;
                if let (lower, Some(upper)) = neighbours_iter.size_hint() {
                    if lower == upper {
                        *num_neighs = lower as i32;
                    }
                }

                // Record up to 11 neighbours
                for (k, neigh) in neighbours_iter.clone().enumerate() {
                    if k >= 11 {
                        break;
                    }
                    neighs[k] = neigh.index as i32;
                }

                if *num_neighs != 0 {
                    let mut closest_d = radius;
                    for nbr in neighbours_iter.clone() {
                        closest_d = closest_d.min((Vector3(*q) - nbr.pos).norm());
                    }
                    let mut weights = neighbours_iter
                        .clone()
                        .enumerate()
                        .map(|(i, nbr)| {
                            let w = kernel
                                .with_closest_dist(closest_d)
                                .eval(Vector3(*q), nbr.pos);
                            if i < 11 {
                                // Record the current weight
                                weight[i] = w as f32;
                            }
                            w
                        })
                        .collect::<Vec<f64>>();

                    // Add a background potential
                    let bg = [q[0] - radius + closest_d, q[1], q[2]];
                    let w = kernel
                        .with_closest_dist(closest_d)
                        .eval(Vector3(*q), bg.into());
                    // Record the last weight for the background potential
                    weight[11] = w as f32;
                    weights.push(w);

                    let mut potentials = neighbours_iter
                        .clone()
                        .map(|nbr| nbr.nml.dot(Vector3(*q) - nbr.pos))
                        .collect::<Vec<f64>>();

                    // Add background potential
                    potentials.push(*potential as f64);

                    let denominator: f64 = weights.iter().sum();
                    let numerator: f64 = weights
                        .iter()
                        .zip(potentials.iter())
                        .map(|(w, p)| w * p)
                        .sum();

                    if denominator != 0.0 {
                        *potential = (numerator / denominator) as f32;
                    }
                }
            });
        }

        {
            mesh.set_attrib_data::<_, VertexIndex>("num_neighbours", &num_neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("neighbours", &neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("weights", &weight_attrib_data)?;
        }

        Ok(())
    }

    fn compute_hrbf_on_mesh<F, M, V3>(
        mesh: &mut M,
        points: &[V3],
        normals: &[V3],
        offsets: &[f64],
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<f64>,
        V3: Into<[f64; 3]> + Copy,
    {
        let sample_pos = mesh.vertex_positions().to_vec();

        let pts: Vec<crate::na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = p.into();
                crate::na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = normals
            .iter()
            .map(|&n| {
                let nml: [f64; 3] = n.into();
                crate::na::Vector3::from(nml)
            })
            .collect();
        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());
        hrbf.fit_offset(&pts, offsets, &nmls);

        for (q_chunk, potential_chunk) in sample_pos.chunks(Self::PARALLEL_CHUNK_SIZE).zip(
            mesh.attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(Self::PARALLEL_CHUNK_SIZE),
        ) {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            q_chunk
                .par_iter()
                .zip(potential_chunk.par_iter_mut())
                .for_each(|(q, potential)| {
                    *potential = hrbf.eval(crate::na::Point3::from(*q)) as f32;
                });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::F;

    fn easy_potential_derivative(radius: f64, dynamic_bg_potential: bool) {
        // The set of samples is just one point. These are initialized using a forward
        // differentiator.
        let mut samples = Samples {
            points: vec![Vector3([0.2, 0.1, 0.0]).map(|x| F::cst(x))],
            normals: vec![Vector3([0.3, 1.0, 0.1]).map(|x| F::cst(x))],
            offsets: vec![0.0],
        };

        // The set of neighbours is the one sample given.
        let neighbours = vec![0];

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        // Initialize the query point.
        let q = Vector3([0.5, 0.3, 0.0]).map(|x| F::cst(x));

        // There is no surface for the set of samples. As a result, the normal derivative should be
        // skipped in this test.
        let surf_topo = vec![];
        let dual_topo = vec![vec![]];

        // Create a view of the samples for the Jacobian computation.
        let view = SamplesView::new(neighbours.as_ref(), &samples);

        // Compute the complete jacobian.
        let jac: Vec<Vector3<F>> = ImplicitSurface::compute_jacobian_at(
            q,
            view,
            radius,
            kernel,
            dynamic_bg_potential,
            &surf_topo,
            &dual_topo,
        )
        .collect();

        // Test the accuracy of each component of the jacobian against an autodiff version of the
        // derivative.
        for i in 0..3 {
            // Set a variable to take the derivative with respect to, using autodiff.
            samples.points[0][i] = F::var(samples.points[0][i]);

            // Create a view of the samples for the potential function.
            let view = SamplesView::new(neighbours.as_ref(), &samples);

            // Initialize background potential to zero.
            let mut p = F::cst(0.0);

            // Compute the local potential function. After calling this function, calling
            // `.deriv()` on the potential output will give us the derivative with resepct to the
            // preset variable.
            ImplicitSurface::compute_local_potential_at(
                q,
                view,
                radius,
                kernel,
                dynamic_bg_potential,
                &mut p,
            );

            // Check the derivative of the autodiff with our previously computed Jacobian.
            assert_relative_eq!(
                jac[0][i].value(),
                p.deriv(),
                max_relative = 1e-6,
                epsilon = 1e-12
            );

            // Reset the variable back to being a constant.
            samples.points[0][i] = F::cst(samples.points[0][i]);
        }
    }

    #[test]
    fn easy_potential_derivative_test() {
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            easy_potential_derivative(radius, true); // dynamic potential
            easy_potential_derivative(radius, false); // constant potential
        }
    }

    /// A more complex test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    fn hard_potential_derivative<P: FnMut() -> Vector3<f64>>(
        dynamic_bg_potential: bool,
        radius: f64,
        perturb: &mut P,
    ) {
        // This is a similar test to the one above, but has a non-trivial surface topology for the
        // surface.

        let h = 1.18032;
        let tri_verts = vec![
            Vector3([0.5, h, 0.0]) + perturb(),
            Vector3([-0.25, h, 0.433013]) + perturb(),
            Vector3([-0.25, h, -0.433013]) + perturb(),
        ];

        let tet_verts = vec![
            Vector3([0.0, 1.0, 0.0]) + perturb(),
            Vector3([-0.94281, -0.33333, 0.0]) + perturb(),
            Vector3([0.471405, -0.33333, 0.816498]) + perturb(),
            Vector3([0.471405, -0.33333, -0.816498]) + perturb(),
        ];

        let tet_faces = vec![[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]];

        // Compute normals. Make sure this is done the same way as everywhere else.
        let mut normals = vec![Vector3::zeros(); tet_verts.len()];
        ImplicitSurface::compute_vertex_area_normals(&tet_faces, &tet_verts, &mut normals);

        let dual_topo = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3]];

        // Initialize the samples with regular f64 for now to keep debug output clean.
        let samples = Samples {
            points: tet_verts.clone(),
            normals: normals.clone(),
            offsets: vec![0.0; 4],
        };

        let neighbours = vec![0, 1, 2, 3];

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert the samples to use autodiff constants.
        let mut ad_samples = Samples {
            points: samples
                .points
                .iter()
                .cloned()
                .map(|vec| vec.map(|x| F::cst(x)))
                .collect(),
            normals: normals
                .iter()
                .cloned()
                .map(|vec| vec.map(|x| F::cst(x)))
                .collect(),
            offsets: samples.offsets.clone(),
        };

        for &q in tri_verts.iter() {
            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);
            let jac: Vec<Vector3<f64>> = ImplicitSurface::compute_jacobian_at(
                q,
                view,
                radius,
                kernel,
                dynamic_bg_potential,
                &tet_faces,
                &dual_topo,
            )
            .collect();

            assert_eq!(jac.len(), neighbours.len());

            let q = q.map(|x| F::cst(x));

            for &vtx in neighbours.iter() {
                for i in 0..3 {
                    ad_samples.points[vtx][i] = F::var(ad_samples.points[vtx][i]);

                    // Compute normals. This is necessary to capture the normal derivatives.
                    ImplicitSurface::compute_vertex_area_normals(
                        &tet_faces,
                        &ad_samples.points,
                        &mut ad_samples.normals,
                    );

                    let view = SamplesView::new(neighbours.as_ref(), &ad_samples);
                    let mut p = F::cst(0.0);
                    ImplicitSurface::compute_local_potential_at(
                        q,
                        view,
                        radius,
                        kernel,
                        dynamic_bg_potential,
                        &mut p,
                    );

                    assert_relative_eq!(
                        jac[vtx][i],
                        p.deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );

                    ad_samples.points[vtx][i] = F::cst(ad_samples.points[vtx][i]);
                }
            }
        }
    }

    #[test]
    fn hard_potential_derivative_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-0.1, 0.1);

        let mut perturb = || Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]);

        // Run for some number of perturbations
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            hard_potential_derivative(true, radius, &mut perturb);
            hard_potential_derivative(false, radius, &mut perturb);
        }
    }

    /// Compute normalized area weighted vertex normals given a triangle topology.
    /// This is a helper function for the `normal_derivative_test`.
    /// Note that it is strictly more useful to precompute unnormalized vertex normals because they
    /// cary more information like area.
    pub(crate) fn compute_vertex_unit_normals<T: Real>(
        surf_topo: &[[usize; 3]],
        points: &[Vector3<T>],
        normals: &mut [Vector3<T>],
    ) {
        // Compute area normals.
        ImplicitSurface::compute_vertex_area_normals(surf_topo, points, normals);

        // Normalize.
        for nml in normals.iter_mut() {
            *nml = *nml / nml.norm();
        }
    }

    /// Test the derivatives of our normal computation method.
    #[test]
    fn normal_derivative_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
        let tet_verts = vec![
            Vector3([0.0, 1.0, 0.0]),
            Vector3([-0.94281, -0.33333, 0.0]),
            Vector3([0.471405, -0.33333, 0.816498]),
            Vector3([0.471405, -0.33333, -0.816498]),
        ];

        let tet_faces = vec![[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]];

        let mut normals = vec![Vector3::zeros(); tet_verts.len()];
        ImplicitSurface::compute_vertex_area_normals(
            tet_faces.as_slice(),
            tet_verts.as_slice(),
            &mut normals,
        );

        // Vertex to triangle map
        let dual_topo = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3]];

        // Initialize the samples with regular f64 for now to keep debug output clean.
        let samples = Samples {
            points: tet_verts.clone(),
            normals: normals.clone(),
            offsets: vec![0.0; 4], // This is not actually used in this test.
        };

        let indices = vec![0, 1, 2, 3]; // look at all the vertices

        // Convert the samples to use autodiff constants.
        let mut ad_samples = Samples {
            points: samples
                .points
                .iter()
                .cloned()
                .map(|vec| vec.map(|x| F::cst(x)))
                .collect(),
            normals: vec![Vector3::<F>::zeros(); normals.len()],
            offsets: samples.offsets.clone(),
        };

        // Set a random product vector.
        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-1.0, 1.0);
        let dxs: Vec<_> = (0..tet_verts.len())
            .map(move |_| Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]))
            .collect();
        let dx = move |Sample { index, .. }| dxs[index];

        // Compute the normal gradient product.
        let view = SamplesView::new(indices.as_ref(), &samples);
        let grad_iter = ImplicitSurface::compute_vertex_unit_normals_gradient_products(
            view,
            &tet_faces,
            &dual_topo,
            dx.clone(),
        );

        for (&vtx, g) in indices.iter().zip(grad_iter) {
            for i in 0..3 {
                ad_samples.points[vtx][i] = F::var(ad_samples.points[vtx][i]);

                // Compute normalized normals. This is necessary to capture the normal derivatives.
                compute_vertex_unit_normals(
                    &tet_faces,
                    &ad_samples.points,
                    &mut ad_samples.normals,
                );

                let mut exp = F::cst(0.0);
                for sample in view.clone().iter() {
                    exp += ad_samples.normals[sample.index].dot(dx(sample).map(|x| F::cst(x)));
                }

                assert_relative_eq!(g[i], exp.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                ad_samples.points[vtx][i] = F::cst(ad_samples.points[vtx][i]);
            }
        }
    }

    #[test]
    fn dynamic_background_potential_derivative_test() {
        // Prepare data
        let q = Vector3([0.1, 0.3, 0.2]);
        let points = vec![
            Vector3([0.3, 0.2, 0.1]),
            Vector3([0.4, 0.2, 0.1]),
            Vector3([0.2, 0.1, 0.3]),
        ];

        let samples = Samples {
            points: points.clone(),
            normals: vec![Vector3::zeros(); points.len()], // Not used
            offsets: vec![0.0; points.len()],              // Not used
        };

        let indices: Vec<usize> = (0..points.len()).collect();

        let radius = 2.0;

        // Initialize kernel.
        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Create a view to the data to be iterated.
        let view = SamplesView::new(indices.as_slice(), &samples);

        // Initialize a background potential. This function takes care of a lot of the setup.
        let bg = BackgroundPotential::new(q, view, radius, kernel, true);

        // Compute manual Jacobian. This is the function being tested for correctness.
        let jac: Vec<_> = bg.compute_jacobian().collect();

        // Prepare autodiff variables.
        let mut ad_samples = Samples {
            points: points.iter().map(|&pos| pos.map(|x| F::cst(x))).collect(),
            normals: vec![Vector3::zeros(); points.len()], // Not used
            offsets: vec![0.0; points.len()],              // Not used
        };

        let q = q.map(|x| F::cst(x));

        // Perform the derivative test on each of the variables.
        for i in 0..points.len() {
            for j in 0..3 {
                ad_samples.points[i][j] = F::var(ad_samples.points[i][j]);

                // Initialize an autodiff version of the potential.
                // This should be done outside the inner loop over samples, but here we make an
                // exception for simplicity.
                let view = SamplesView::new(indices.as_slice(), &ad_samples);
                let ad_bg = BackgroundPotential::new(q, view, F::cst(radius), kernel, true);

                let p = ad_bg.compute_unnormalized_weighted_potential() * ad_bg.weight_sum_inv();

                assert_relative_eq!(jac[i][j], p.deriv());
                ad_samples.points[i][j] = F::cst(ad_samples.points[i][j]);
            }
        }
    }
}
