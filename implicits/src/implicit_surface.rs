//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::geo::Real;
use crate::geo::math::{Vector3, Matrix3};
use crate::geo::mesh::{VertexMesh, topology::VertexIndex};
use crate::geo::prim::{Triangle};
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
where V3: Into<Vector3<f64>> + Copy
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
    mesh: &M
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let points: Vec<Vector3<f64>> = mesh
        .vertex_position_iter()
        .map(|&x| -> Vector3<f64> {
            Vector3(x)
                .cast::<f64>()
                .expect("Failed to convert positions to f64")
        }).collect();

    let normals = mesh.attrib_iter::<[f32; 3], VertexIndex>("N").ok()
        .map_or( vec![Vector3::zeros(); points.len()], |iter| {
            iter.map(|&nml| Vector3(nml).cast::<f64>().unwrap()).collect()
        });

    (points, normals)
}

pub fn build_rtree(oriented_points_iter: impl Iterator<Item=OrientedPoint>) -> RTree<OrientedPoint> {
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
    triangles: Vec<[usize;3]>,
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

    pub fn build(&self) -> ImplicitSurface {
        let ImplicitSurfaceBuilder {
            kernel,
            background_potential,
            triangles,
            points,
            mut normals,
            mut offsets,
            ..
        } = self.clone();

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
        let mut surf = ImplicitSurface {
            kernel,
            background_potential,
            spatial_tree: build_rtree(oriented_points_iter),
            surface_topo: triangles,
            samples: Samples { points, normals, offsets },
            max_step: self.max_step,
            neighbour_cache: RefCell::new(NeighbourCache::new()),
            dual_topo,
        };
        surf.recompute_normals();
        surf
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
#[derive(Clone, Debug, PartialEq)]
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
    pub fn new(indices: &'i [usize], samples: &'d Samples<T>) -> Self{
        SamplesView {
            indices,
            points: samples.points.as_slice(),
            normals: samples.normals.as_slice(),
            offsets: samples.offsets.as_slice(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn iter(&'i self) -> impl Iterator<Item = Sample<T>> + 'i {
        let SamplesView {
            ref indices,
            ref points,
            ref normals,
            ref offsets,
        } = self;
        indices.iter().map(move |&i| Sample { index: i, pos: points[i], nml: normals[i], off: offsets[i] })
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
        indices.into_iter().map(move |&i| Sample { index: i, pos: points[i], nml: normals[i], off: offsets[i] })
    }

    #[inline]
    pub fn index(&self, i: usize) -> usize {
        self.indices[i]
    }

    #[inline]
    pub fn point(&self, i: usize) -> Vector3<T> {
        self.points[self.indices[i]]
    }

    #[inline]
    pub fn point_iter(&'d self, i: usize) -> impl Iterator<Item = Vector3<T>> + 'd {
        let points = &self.points;
        self.indices.iter().map(move |&i| points[i])
    }

    #[inline]
    pub fn normal(&self, i: usize) -> Vector3<T> {
        self.normals[self.indices[i]]
    }

    #[inline]
    pub fn normal_iter(&'d self, i: usize) -> impl Iterator<Item = Vector3<T>> + 'd {
        let normals = &self.normals;
        self.indices.iter().map(move |&i| normals[i])
    }

    #[inline]
    pub fn offset(&self, i: usize) -> f64 {
        self.offsets[self.indices[i]]
    }

    #[inline]
    pub fn offset_iter(&'d self, i: usize) -> impl Iterator<Item = f64> + 'd {
        let offsets = &self.offsets;
        self.indices.iter().map(move |&i| offsets[i])
    }

    /// Get all sample data at the given index
    #[inline]
    pub fn at(&self, i: usize) -> (Vector3<T>, Vector3<T>, f64) {
        (self.point(i), self.normal(i), self.offset(i))
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
    surface_topo: Vec<[usize;3]>,

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
        return self.surface_topo.len()
    }

    /// Number of sample points used to represent the implicit surface.
    pub fn num_points(&self) -> usize {
        return self.samples.points.len()
    }

    /// Update the normals stored on the `ImplicitSurface`. This is usually called when the points
    /// have changed. Note that these are unnormalized to make it eaiser to compute derivatives.
    pub fn recompute_normals(&mut self) {
        let ImplicitSurface {
            samples: Samples { ref mut points, ref mut normals, .. },
            ref surface_topo,
            ..
        } = self;

        // Clear the normals.
        for nml in normals.iter_mut() {
            *nml = Vector3::zeros();
        }

        for tri_indices in surface_topo.iter() {
            let tri = Triangle::from_indexed_slice(tri_indices, &points);
            let area_nml = tri.area_normal();
            normals[tri_indices[0]] += area_nml;
            normals[tri_indices[1]] += area_nml;
            normals[tri_indices[2]] += area_nml;
        }
    }

    /// Update points and normals (oriented points) using an iterator.
    pub fn update_points<I>(&mut self, points_iter: I) 
        where I: Iterator<Item = [f64;3]>,
    {
        for (p, new_p) in self.samples.points.iter_mut().zip(points_iter) {
            *p = new_p.into();
        }

        //self.recompute_normals();

        let ImplicitSurface {
            samples: Samples { ref mut points, ref mut normals, .. },
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
            samples: Samples { ref points, ref normals, .. },
            max_step,
            ..
        } = self;
        match *kernel {
            KernelType::Interpolating { radius } |
            KernelType::Approximate { radius, .. } |
            KernelType::Cubic { radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
                self.cached_neighbours_borrow(query_points, neigh);
            }
            KernelType::Global { .. } |
            KernelType::Hrbf => {
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
    fn cached_neighbours_borrow<'a, I, N>(&'a self, query_points: &[[f64; 3]], neigh: N) -> Ref<'a, NeighbourCache>
    where
        I: Iterator<Item=OrientedPoint> + 'a,
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
    pub fn potential(&self,
                     query_points: &[[f64; 3]],
                     out_potential: &mut [f64]) -> Result<(), super::Error>
    {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            samples: Samples { ref points, ref normals, ref offsets },
            max_step,
            ..
        } = *self;

        match *kernel {
            KernelType::Interpolating { radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
                let kern = kernel::LocalInterpolating::new(radius);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Approximate { tolerance, radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Cubic { radius } => {
                let radius_ext = radius + max_step;
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
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
        I: Iterator<Item=OrientedPoint> + 'a,
        K: SphericalKernel<f64> + Copy + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        let neigh = self.cached_neighbours_borrow(query_points, neigh);

        let ImplicitSurface {
            ref samples,
            ref spatial_tree,
            background_potential,
            ..
        } = *self;

        if background_potential {
            let cached_points = &neigh.points;
            assert_eq!(out_potential.len(), cached_points.len());

            // Initialize potential with zeros.
            for p in out_potential.iter_mut() {
                *p = 0.0;
            }
            // Generate a background potential field for every query point. This will be mixed
            // in with the computed potentials for local methods. Global methods like HRBF
            // ignore this field.
            for (pos, potential) in
                zip!(cached_points.iter().map(|&(qi, _)| query_points[qi]), out_potential.iter_mut())
            {
                if let Some(nearest_neigh) = spatial_tree.nearest_neighbor(&pos) {
                    let q = Vector3(pos);
                    let p = nearest_neigh.pos;
                    let nml = nearest_neigh.nml;
                    *potential = (q - p).dot(nml) / nml.norm();
                } else {
                    return Err(super::Error::Failure);
                }
            }
        }

        zip!(
            neigh.points.par_iter().map(|(qi, neigh)| (query_points[*qi], neigh)),
            out_potential.par_iter_mut()
        ).for_each(|((q, neighbours), potential)| {
            if let Some(p) = Self::compute_potential_at(Vector3(q), SamplesView::new(neighbours, samples), radius, kernel) {
                *potential = p;
            }
        });

        Ok(())
    }

    /// Compute the potential at a given query point. If the potential is invalid or nieghbourhood
    /// is empty, return None.
    pub(crate) fn compute_potential_at<T: Real, K>(q: Vector3<T>, samples: SamplesView<T>, radius: f64, kernel: K) -> Option<T>
        where K: SphericalKernel<T> + Copy + Sync + Send,
    {
        if samples.is_empty() {
            return None;
        }

        let mut closest_d = radius;
        for sample in samples.iter() {
            closest_d = closest_d.min((q - sample.pos).norm().to_f64().unwrap());
        }

        let mut denominator = T::zero();
        let mut numerator = T::zero();
        for Sample { pos, nml, off, .. } in samples.iter() {
            let w = kernel.with_closest_dist(closest_d).eval(q, pos);
            let p = T::from(off).unwrap() + nml.dot(q - pos) / nml.norm();

            denominator += w;
            numerator += w * p;
        }

        // Background potential
        //let bg = [q[0] - radius + closest_d, q[1], q[2]];
        //let w = kernel.with_closest_dist(closest_d).eval(q, bg.into());
        //denominator += w;
        //numerator += w * (*potential as f64);

        if denominator != T::zero() {
            Some(numerator / denominator)
        } else {
            None
        }
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn num_surface_jacobian_entries(&self) -> usize {
        let cache = self.neighbour_cache.borrow();
        cache.points.iter().map(|(_, pts)| pts.len()*3).sum()
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_iter(&self) -> Result<impl Iterator<Item = (usize, usize)>, super::Error>
    {
        match self.kernel {
            KernelType::Approximate { .. } => {
                Ok(self.mls_surface_jacobian_indices_iter())
            }
            _ => Err(super::Error::UnsupportedKernel)
        }
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices(&self, rows: &mut [usize], cols: &mut [usize]) -> Result<(), super::Error>
    {
        match self.kernel {
            KernelType::Approximate { .. } => {
                Ok(self.mls_surface_jacobian_indices(rows, cols))
            }
            _ => Err(super::Error::UnsupportedKernel)
        }
    }

    pub fn surface_jacobian_values(&self,
                                   query_points: &[[f64; 3]],
                                   values: &mut [f64])
    -> Result<(), super::Error>
    {
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
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
                let kernel = kernel::LocalApproximate::new(radius, tolerance);
                self.mls_surface_jacobian_values(query_points, radius, kernel, neigh, values);
                Ok(())
            }
            _ => Err(super::Error::UnsupportedKernel)
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
        cache.points.clone()
            .into_iter()
            .enumerate()
            .flat_map(move |(row, (_, nbr_points))|
                nbr_points.into_iter().flat_map(move |col| (0..3).map(move |i| (row, 3*col+i))
            ))
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    fn mls_surface_jacobian_indices(&self, rows: &mut [usize], cols: &mut [usize]) {
        // For each row
        let cache = self.neighbour_cache.borrow();
        let row_col_iter = cache.points
            .iter()
            .enumerate()
            .flat_map(move |(row, (_, nbr_points))|
                nbr_points.iter().cloned().flat_map(move |col| (0..3).map(move |i| (row, 3*col+i))
            ));

        for ((row, col), out_row, out_col) in zip!(row_col_iter, rows.iter_mut(), cols.iter_mut())  {
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
    )
    where
        I: Iterator<Item=OrientedPoint> + 'a,
        K: SphericalKernel<f64> + Copy + Sync + Send,
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

        let jac_iter = 
            // For each row
            neigh.points.iter().map(|(qi, neigh)| (query_points[*qi], neigh))
        .flat_map(move |(q, nbr_points)| {
            Self::compute_jacobian_at(Vector3(q), SamplesView::new(nbr_points, samples), radius, kernel)
        });
        let value_vecs: &mut [[f64;3]] = reinterpret::reinterpret_mut_slice(values);

        jac_iter.zip(value_vecs.iter_mut()).for_each(|(new_vec, vec)| {
            *vec = new_vec.into();
        });
    }

    /// Compute the jacobian for the implicit surface potential given by the samples with the
    /// specified kernel.
    pub(crate) fn compute_jacobian_at<'a, T: Real, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        radius: f64,
        kernel: K
    ) -> impl Iterator<Item=Vector3<T>> + 'a
        where K: SphericalKernel<T> + Copy + Sync + Send,
    {
        // Find the closest vertex for background potential derivative.
        let mut closest_d = radius;
        let mut closest_i = -1isize;
        for Sample { index, pos, .. } in samples.iter() {
            let dist = (q - pos).norm().to_f64().unwrap();
            if dist < closest_d {
                closest_d = dist;
                closest_i = index as isize;
            }
        }

        let mut weight_sum = T::zero();
        for Sample { pos, off, .. } in samples.iter() {
            let w = kernel.with_closest_dist(closest_d).eval(q, pos);
            weight_sum += w;
        }

        // Background potential weight
        //let bg_pos = Vector3([q[0] - radius + closest_d, q[1], q[2]]);
        //let bg_w = kernel.with_closest_dist(closest_d).eval(Vector3(q), bg_pos);
        //weight_sum += bg_w;

        let weight_sum2 = weight_sum * weight_sum;

        // For each column
        samples.clone().into_iter().map(move |Sample { pos, nml, off, .. }| {
            let diff = q - pos;
            let norm_inv = T::one() / nml.norm();

            // Compute background potential derivative contribution.
            // Compute derivative if the closest point in the neighbourhood. Otherwise we
            // assume the background potential is constant.
            let bg_deriv = //if background_potential && i as isize == closest_i {
                //-diff / closest_d
            //} else {
                Vector3::zeros();
            //};

            if weight_sum == T::zero() { // bg_w
                return bg_deriv;
            }

            let w = kernel.with_closest_dist(closest_d).eval(q, pos);
            let dw = -kernel.with_closest_dist(closest_d).grad(q, pos);

            let mut dw_neigh = T::zero();

            for Sample { pos: posk, nml: nmlk, off: offk, .. } in samples.iter() {
                let wk = kernel.with_closest_dist(closest_d).eval(q, posk);
                let pk = T::from(offk).unwrap() + (nmlk.dot(q - posk) / nmlk.norm());
                dw_neigh += wk * pk;
            }

            let mut dw_p = dw * (dw_neigh/weight_sum2);

            dw_p += (dw / weight_sum) * (T::from(off).unwrap() + nml.dot(diff) * norm_inv);

            // Compute the normal component of the derivative
            //let nml_proj = Matrix3::identity() - nml*nml.transpose()/(norm*norm);
            let mut nml_deriv = nml * (-(w/weight_sum)*norm_inv);
            // Look at the ring of triangles around the vertex with respect to which we are
            // taking the derivative.
            /*
            for &tri_idx in dual_topo[i].iter() {
                let tri_indices = &surface_topo[tri_idx];
                let tri = Triangle::from_indexed_slice(tri_indices, &points);
                let nml_grad = tri.area_normal_gradient(
                    tri_indices.iter().position(|&x| x == i).expect("Triangle mesh topology corruption."));
                for &k in tri_indices.iter() {
                    if k == i {
                        let proj_diff = nml_proj * diff;
                        let normalized_w = w/weight_sum;
                        nml_deriv +=  (nml_grad * proj_diff) * (normalized_w/norm) //- nml * (normalized_w/norm)
                    } else {
                        let nmlk = normals[k];
                        let normk = nmlk.norm();
                        let nmlk_proj = Matrix3::identity() - nmlk*nmlk.transpose()/(normk*normk);
                        let posk = points[k];
                        let wk = kernel.eval(Vector3(q), posk) / weight_sum;
                        let proj_diff = nmlk_proj  * (Vector3(q) - posk);
                        nml_deriv += (wk * nml_grad * proj_diff) / normk;
                    }
                }
            }*/

            dw_p + bg_deriv + nml_deriv
        })
    }

    fn compute_hrbf<V3>(
        query_points: &[[f64; 3]],
        points: &[V3],
        normals: &[V3],
        offsets: &[f64],
        out_potential: &mut [f64],
    ) -> Result<(), super::Error>
    where
        V3: Into<[f64;3]> + Copy
    {
        let pts: Vec<crate::na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64;3] = p.into();
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
            samples: Samples { ref mut points, ref mut normals, ref mut offsets },
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
            samples: Samples { ref points, ref normals, ref offsets },
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
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
                let kern = kernel::LocalInterpolating::new(radius);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Approximate { tolerance, radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Cubic { radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2).into_iter().cloned();
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
        I: Iterator<Item=OrientedPoint> + Clone + 'a,
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
                    let mut weights = neighbours_iter.clone().enumerate()
                        .map(|(i, nbr)| {
                            let w = kernel.with_closest_dist(closest_d).eval(Vector3(*q), nbr.pos);
                            if i < 11 {
                                // Record the current weight
                                weight[i] = w as f32;
                            }
                            w
                        })
                        .collect::<Vec<f64>>();

                    // Add a background potential
                    let bg = [q[0] - radius + closest_d, q[1], q[2]];
                    let w = kernel.with_closest_dist(closest_d).eval(Vector3(*q), bg.into());
                    // Record the last weight for the background potential
                    weight[11] = w as f32;
                    weights.push(w);

                    let mut potentials = neighbours_iter.clone()
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
        V3: Into<[f64;3]> + Copy
    {
        let sample_pos = mesh.vertex_positions().to_vec();

        let pts: Vec<crate::na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64;3] = p.into();
                crate::na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = normals
            .iter()
            .map(|&n| {
                let nml: [f64;3] = n.into();
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

    #[test]
    fn test_potential_derivative() {
        let mut samples = Samples {
            points: vec![Vector3([0.0, 0.0, 0.0]).map(|x| F::cst(x))],
            normals: vec![Vector3([0.0, 1.0, 0.0]).map(|x| F::cst(x))],
            offsets: vec![0.0],
        };

        let neighbours = vec![0];

        let radius = 2.0;
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        let q = Vector3([0.0, 0.1, 0.0]).map(|x| F::cst(x));

        for i in 0..3 {
            samples.points[0][i] = F::var(samples.points[0][i]);

            let view = SamplesView::new(neighbours.as_ref(), &samples);

            if let Some(p) = ImplicitSurface::compute_potential_at(q, view.clone(), radius, kernel) {
                let jac: Vec<Vector3<F>> = ImplicitSurface::compute_jacobian_at(q, view, radius, kernel).collect();
                assert_relative_eq!(jac[0][i].value(), p.deriv(), max_relative=1e-8);
            }

            samples.points[0][i] = F::cst(samples.points[0][i]);
        }

    }
}
