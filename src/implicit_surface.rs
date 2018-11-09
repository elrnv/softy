//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::geo::math::Vector3;
use crate::geo::mesh::{VertexMesh, topology::VertexIndex};
use crate::kernel::{self, Kernel, KernelType};
use rayon::prelude::*;
use spade::{rtree::RTree, BoundingRect, SpatialObject};
use std::cell::RefCell;

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

pub fn oriented_points_iter<'a>(
    points: &'a [[f64; 3]],
    normals: &'a [[f64; 3]],
) -> impl Iterator<Item = OrientedPoint> + Clone + 'a {
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

pub fn oriented_points_from_mesh<'a, M: VertexMesh<f64>>(
    mesh: &M
) -> Vec<OrientedPoint> {
    let points_iter = mesh
        .vertex_position_iter()
        .map(|&x| -> [f64;3] {
            Vector3(x)
                .cast::<f64>()
                .expect("Failed to convert positions to f64")
                .into()
        });

    if let Ok(iter) = mesh.attrib_iter::<[f32; 3], VertexIndex>("N") {
        let normals_iter = iter.map(|&nml| Vector3(nml).cast::<f64>().unwrap().into());
        points_iter
            .zip(normals_iter)
            .enumerate()
            .map(|(i, (pos, nml)) : (usize, ([f64;3], [f64;3]))| OrientedPoint {
                index: i as i32,
                pos: pos.into(),
                nml: nml.into(),
            }).collect()
    } else {
        points_iter
            .enumerate()
            .map(|(i, pos)| OrientedPoint {
                index: i as i32,
                pos: pos.into(),
                nml: Vector3::zeros(),
            }).collect()
    }
}

pub fn build_rtree(oriented_points: &[OrientedPoint]) -> RTree<OrientedPoint> {
    let mut rtree = RTree::new();
    for pt in oriented_points.iter() {
        rtree.insert(*pt);
    }
    rtree
}

fn dist(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    (a - b).norm()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitSurfaceBuilder {
    kernel: KernelType,
    background_potential: bool,
    points: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
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
            points: Vec::new(),
            normals: Vec::new(),
            offsets: Vec::new(),
            max_step: 0.0, // This is a sane default for static implicit surfaces.
        }
    }

    pub fn with_points(&mut self, points: Vec<[f64; 3]>) -> &mut Self {
        self.points = points;
        self
    }

    pub fn with_normals(&mut self, normals: Vec<[f64; 3]>) -> &mut Self {
        self.normals = normals;
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
                    .into()
            })
            .collect();
        self.normals = mesh
            .attrib_iter::<[f32; 3], VertexIndex>("N")
            .map(|iter| {
                iter.map(|nml| Vector3(*nml).cast::<f64>().unwrap().into())
                    .collect()
            })
            .unwrap_or(vec![[0.0f64; 3]; mesh.num_vertices()]);
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
    }

    pub fn build(&self) -> ImplicitSurface {
        let ImplicitSurfaceBuilder {
            kernel,
            background_potential,
            points,
            mut normals,
            mut offsets,
        } = self.clone();

        if normals.is_empty() {
            normals = vec![[0.0; 3]; points.len()];
        }

        if offsets.is_empty() {
            offsets = vec![0.0; points.len()];
        }

        assert_eq!(points.len(), normals.len());
        assert_eq!(points.len(), offsets.len());

        let oriented_points: Vec<OrientedPoint> = oriented_points_iter(&points, &normals).collect();
        ImplicitSurface {
            kernel,
            background_potential,
            spatial_tree: build_rtree(&oriented_points),
            oriented_points,
            offsets,
            max_step: self.max_step,
            neighbour_cache: RefCell::new(None),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImplicitSurface {
    kernel: KernelType,
    background_potential: bool,
    spatial_tree: RTree<OrientedPoint>,
    oriented_points: Vec<OrientedPoint>,
    /// Potential values at the interpolating points. These offsets indicate the values to
    /// match by interpolating implicit surfaces. This means that the zero iso-surface will ot
    /// necessarily pass through the given points.
    offsets: Vec<f64>,

    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative may be denser
    /// than then needed, which typically results in slower performance.  If it is set too low,
    /// there may be errors in the derivative. It is the callers responsibility to set this step
    /// accurately using `update_max_step`. If the implicit surface is not changing, leave this at
    /// 0.0.
    max_step: f64,

    /// Cache the neighbouring indices of the sample points (from oriented_points vector) for each
    /// query point we see. This cache can be invalidated explicitly when the sparsity pattern is
    /// expected to change. This is wrapped in a `RefCell` because it may be updated in non mutable
    /// functions since it's a cache.
    neighbour_cache: RefCell<Option<Vec<Vec<usize>>>>,
}

impl ImplicitSurface {
    const PARALLEL_CHUNK_SIZE: usize = 5000;

    /// Number of sample points used to represent the implicit surface.
    pub fn num_points(&self) -> usize {
        return self.oriented_points.len()
    }

    /// Update points and normals (oriented points) using an iterator.
    pub fn update_points_and_normals<I,J>(&mut self, points_iter: I, mb_normals_iter: Option<J>) 
        where I: Iterator<Item = [f64;3]>,
              J: Iterator<Item = [f64;3]>,
    {
        let ImplicitSurface {
            ref mut spatial_tree,
            ref mut oriented_points,
            ..
        } = self;

        match mb_normals_iter {
            Some(normals_iter) => {
                for (op, (pos, nml)) in oriented_points.iter_mut()
                    .zip(points_iter.zip(normals_iter)) {
                    op.pos = pos.into();
                    op.nml = nml.into();
                }
            }
            None => {
                for (op, pos) in oriented_points.iter_mut().zip(points_iter) {
                    op.pos = pos.into();
                    op.nml = Vector3::zeros();
                }
            }
        }

        *spatial_tree = build_rtree(oriented_points);
    }

    /// Compute neighbour cache if it hasn't been computed yet. Return the neighbours of the given
    /// query points. Note that the cache must be invalidated explicitly, there is no real way to
    /// automatically cache results because both: query points and sample points may change
    /// slightly, but we expect the neighbourhood information to remain the same.
    pub fn cached_neighbours<N>(&self, query_points: &[[f64; 3]], neigh: N) -> &[Vec<usize>]
    where
        N: Fn([f64; 3]) -> Vec<&'a OrientedPoint> + Sync + Send,
    {
        if self.neighbour_cache.borrow().is_none() {
            let mut cache = self.neighbour_cache.borrow_mut();
            *cache = Some(vec![Vec::new(); query_points.len()]);

            zip!(
                cache.par_iter_mut(),
                query_points.par_iter()
            ).for_each(|(c, q)| {
                let neighbours: Vec<&OrientedPoint> = neigh(*q);
                *c = neighbours.map(|&&op| op.index).collect();
            });
        }
        self.neighbour_cache.borrow().unwrap().as_slice()
    }

    /// Set `neighbour_cache` to None. This triggers recomputation of the neighbour cache next time
    /// the potential or its derivatives are requested.
    pub fn invalidate_neighbour_cache(&self) {
        let cache = self.neighbour_cache.borrow_mut();
        *cache = None;
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
            background_potential: generate_bg_potential,
            ref spatial_tree,
            ref oriented_points,
            ref offsets,
            max_step,
        } = *self;

        assert_eq!(out_potential.len(), query_points.len());

        if generate_bg_potential {
            // Initialize potential with zeros.
            for p in out_potential.iter_mut() {
                *p = 0.0;
            }
            // Generate a background potential field for every sample point. This will be mixed
            // in with the computed potentials for local methods. Global methods like HRBF
            // ignore this field.
            for (pos, potential) in
                zip!(query_points.iter(), out_potential.iter_mut())
            {
                if let Some(nearest_neigh) = spatial_tree.nearest_neighbor(pos) {
                    let q = Vector3(*pos);
                    let p = nearest_neigh.pos;
                    let nml = nearest_neigh.nml;
                    *potential = (q - p).dot(nml);
                } else {
                    return Err(super::Error::Failure);
                }
            }
        }

        match *kernel {
            KernelType::Interpolating { radius } => {
                let radius2 = radius * radius + max_step;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, closest_dist| {
                    kernel::LocalInterpolating::new(radius)
                        .update_closest(closest_dist)
                        .f(dist(x, p))
                };
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Approximate { tolerance, radius } => {
                let radius2 = radius * radius + max_step;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| kernel::LocalApproximate::new(radius, tolerance).f(dist(x, p));
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Cubic { radius } => {
                let radius2 = radius * radius + max_step;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| kernel::LocalCubic::new(radius).f(dist(x, p));
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Global { tolerance } => {
                let neigh = |_| oriented_points.iter().collect(); // Global kernel, all points are neighbours
                let radius = 1.0;
                let kern = |x, p, _| kernel::GlobalInvDistance2::new(tolerance).f(dist(x, p));
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Hrbf => {
                // Global kernel, all points are neighbours.
                Self::compute_hrbf(query_points, oriented_points, offsets, out_potential)
            }
        }
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls<'a, K, N, F>(
        &self,
        query_points: &[[f64; 3]],
        radius: f64,
        kernel: K,
        neigh: N,
        out_potential: &mut [f64],
    ) -> Result<(), super::Error>
    where
        K: Fn(Vector3<f64>, Vector3<f64>, f64) -> f64 + Sync + Send,
        N: Fn([f64; 3]) -> Vec<&'a OrientedPoint> + Sync + Send,
    {
        let neigh = self.cached_neighbours(query_points, neigh);

        let ImplicitSurface {
            oriented_points,
            ..
        } = *self;

        zip!(
            query_points.par_iter(),
            neigh.par_iter(),
            out_potential.par_iter_mut()
        ).for_each(|(q, neighbours, potential)| {
            if !neighbours.is_empty() {
                let mut closest_d = radius;
                for nbr in neighbours.iter() {
                    closest_d = closest_d.min(dist(Vector3(*q), nbr.pos));
                }

                let mut denominator = 0.0;
                let mut numerator = 0.0;
                for npt in neighbours.iter().map(|i| oriented_points[i]) {
                    let w = kernel(Vector3(*q), npt.pos, closest_d);
                    let p = npt.nml.dot(Vector3(*q) - npt.pos);

                    denominator += w;
                    numerator += w * p;
                }

                // Background potential
                let bg = [q[0] - radius + closest_d, q[1], q[2]];
                let w = kernel(Vector3(*q), bg.into(), closest_d);
                denominator += w;
                numerator += w * (*potential as f64);

                if denominator != 0.0 {
                    *potential = numerator / denominator;
                }
            }
        });

        Ok(())
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_iter<'a>(&'a self) -> Result<impl Iterator<Item = (usize, usize)> + 'a, super::Error>
    {
        match self.kernel {
            KernelType::Approximate { .. } => {
                Ok(self.mls_surface_jacobian_indices_iter())
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
            ref oriented_points,
            ref offsets,
            background_potential: generate_bg_potential,
            max_step,
        } = *self;

        match *kernel {
            KernelType::Approximate { tolerance, radius } => {
                let radius2 = radius * radius + max_step;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| kernel::LocalApproximate::new(radius, tolerance).f(dist(x, p));
                let dkern = |x, p, _| kernel::LocalApproximate::new(radius, tolerance).df(dist(x, p));
                self.mls_surface_jacobian_values(query_points, radius, kern, neigh, values);
                Ok(())
            }
            _ => Err(super::Error::UnsupportedKernel)
        }
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    fn mls_surface_jacobian_indices_iter<'a>(&self) -> impl Iterator<Item = (usize, usize)> + 'a {
        // For each row
        self.neighbour_cache.borrow()
            .expect("Neighbour computation missing, can't initialize jacobian indices.")
            .iter().enumerate().flat_map(move |(row, neighbours)|
                // For each column
                neighbours.iter().map(move |col| (row, col))
            )
    }

    fn mls_surface_jacobian_values<'a, K, N>(
        &self,
        query_points: &[[f64; 3]],
        radius: f64,
        kernel: K,
        neigh: N,
        values: &mut [f64],
    )
    where
        K: Fn(Vector3<f64>, Vector3<f64>, f64) -> f64 + Sync + Send,
        N: Fn([f64; 3]) -> Vec<&'a OrientedPoint> + Sync + Send,
    {
        let ImplicitSurface {
            oriented_points,
            ..
        } = *self;

        let neigh = self.cached_neighbours(query_points, neigh);
        // For each row
        zip!(
            query_points.par_iter(),
            neigh.par_iter()
        ).flat_map(|(q, neighbours)| {
            if !neighbours.is_empty() {
                let mut denominator = 0.0;
                let mut numerator = 0.0;
                for npt in neighbours.iter().map(|i| oriented_points[i]) {
                    let w = kernel(Vector3(*q), npt.pos, 0.0);
                    let p = npt.nml.dot(Vector3(*q) - npt.pos);

                    denominator += w;
                    numerator += w * p;
                }

                // Background potential
                let mut closest_d = radius;
                for nbr in neighbours.iter() {
                    closest_d = closest_d.min(dist(Vector3(*q), nbr.pos));
                }
                let bg = [q[0] - radius + closest_d, q[1], q[2]];
                let w = kernel(Vector3(*q), bg.into(), 0.0);
                denominator += w;
                numerator += w * (*potential as f64);

                if denominator != 0.0 {
                    *potential = numerator / denominator;
                }
            }
        });
    }

    fn compute_hrbf<F>(
        query_points: &[[f64; 3]],
        oriented_points: &[OrientedPoint],
        offsets: &[f64],
        interrupt: F,
        out_potential: &mut [f64],
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
    {
        let pts: Vec<crate::na::Point3<f64>> = oriented_points
            .iter()
            .map(|op| crate::na::Point3::from(op.pos.into_inner()))
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = oriented_points
            .iter()
            .map(|op| crate::na::Vector3::from(op.nml.into_inner()))
            .collect();

        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());

        hrbf.fit_offset(&pts, offsets, &nmls);

        for (q_chunk, potential_chunk) in query_points.chunks(Self::PARALLEL_CHUNK_SIZE).zip(
            out_potential.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
        ) {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            q_chunk
                .par_iter()
                .zip(potential_chunk.par_iter_mut())
                .for_each(|(q, potential)| {
                    *potential = hrbf.eval(crate::na::Point3::from(*q));
                });
        }

        Ok(())
    }

    /// Jacobian with respect to sample points that define the implicit surface.
    pub fn surface_jacobian_indices<F>(&self,
                                       max_step: f64,
                                       indices: &mut [f64]) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
    {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            ref oriented_points,
            ref offsets,
            background_potential: generate_bg_potential,
        } = *self;

        match *kernel {
            KernelType::Approximate { tolerance, radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| kernel::LocalApproximate::new(radius, tolerance).f(dist(x, p));
                Self::compute_mls(query_points, radius, kern, neigh, interrupt, out_potential)
            }
            _ => { unimplemented!() }
        }
    }


    /// Update points, normals and offsets from a given vertex mesh.
    pub fn update_oriented_points_with_mesh<M: VertexMesh<f64>>(&mut self, mesh: &M) {
        let ImplicitSurface {
            ref mut spatial_tree,
            ref mut oriented_points,
            ref mut offsets,
            ..
        } = self;

        *oriented_points = oriented_points_from_mesh(mesh);

        *spatial_tree = build_rtree(oriented_points);

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
        assert_eq!(oriented_points.len(), offsets.len());
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
            ref oriented_points,
            ref offsets,
            background_potential: generate_bg_potential,
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
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, closest_dist| {
                    kernel::LocalInterpolating::new(radius)
                        .update_closest(closest_dist)
                        .f(dist(x, p))
                };
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Approximate { tolerance, radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| kernel::LocalApproximate::new(radius, tolerance).f(dist(x, p));
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Cubic { radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| kernel::LocalCubic::new(radius).f(dist(x, p));
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Global { tolerance } => {
                let neigh = |_| oriented_points.iter().collect();
                let radius = 1.0;
                let kern = |x, p, _| kernel::GlobalInvDistance2::new(tolerance).f(dist(x, p));
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Hrbf => {
                Self::compute_hrbf_on_mesh(mesh, oriented_points, offsets, interrupt)
            }
        }
    }
    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls_on_mesh<'a, K, N, F, M>(
        mesh: &mut M,
        radius: f64,
        kernel: K,
        neigh: N,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        K: Fn(Vector3<f64>, Vector3<f64>, f64) -> f64 + Sync + Send,
        N: Fn([f64; 3]) -> Vec<&'a OrientedPoint> + Sync + Send,
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
                let neighbours_vec: Vec<&OrientedPoint> = neigh(*q);
                let neighbours = &neighbours_vec;

                // Record number of neighbours in total.
                *num_neighs = neighbours.len() as i32;

                // Record up to 11 neighbours
                for (k, neigh) in neighbours_vec.iter().enumerate() {
                    if k >= 11 {
                        break;
                    }
                    neighs[k] = neigh.index as i32;
                }

                if !neighbours.is_empty() {
                    let n = neighbours.len() + 1;
                    let mut closest_d = radius;
                    for nbr in neighbours.iter() {
                        closest_d = closest_d.min(dist(Vector3(*q), nbr.pos));
                    }
                    let weights = (0..n)
                        .map(|i| {
                            if i == neighbours.len() {
                                let bg = [q[0] - radius + closest_d, q[1], q[2]];
                                let w = kernel(Vector3(*q), bg.into(), closest_d);
                                // Record the last weight for the background potential
                                weight[11] = w as f32;
                                w
                            } else {
                                let w = kernel(Vector3(*q), neighbours[i].pos, closest_d);
                                if i < 11 {
                                    // Record the current weight
                                    weight[i] = w as f32;
                                }
                                w
                            }
                        })
                        .collect::<Vec<f64>>();

                    let potentials = (0..n)
                        .map(|i| {
                            if i == neighbours.len() {
                                *potential as f64
                            } else {
                                neighbours[i].nml.dot(Vector3(*q) - neighbours[i].pos)
                            }
                        })
                        .collect::<Vec<f64>>();

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

    fn compute_hrbf_on_mesh<F, M>(
        mesh: &mut M,
        oriented_points: &[OrientedPoint],
        offsets: &[f64],
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<f64>,
    {
        let sample_pos = mesh.vertex_positions().to_vec();

        let pts: Vec<crate::na::Point3<f64>> = oriented_points
            .iter()
            .map(|op| crate::na::Point3::from(op.pos.into_inner()))
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = oriented_points
            .iter()
            .map(|op| crate::na::Vector3::from(op.nml.into_inner()))
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
