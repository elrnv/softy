//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::geo::math::{Matrix3, Vector3};
use crate::geo::mesh::{attrib::*, topology::VertexIndex, VertexMesh};
use crate::geo::prim::Triangle;
use crate::geo::Real;
use crate::kernel::{self, KernelType, SphericalKernel};
use rayon::prelude::*;
use spade::rtree::RTree;
use std::cell::{Ref, RefCell};

pub mod builder;
pub mod samples;
pub mod spatial_tree;
pub mod background_potential;
pub mod neighbour_cache;

pub use self::builder::*;
pub use self::samples::*;
pub use self::spatial_tree::*;

pub(crate) use self::neighbour_cache::NeighbourCache;
pub(crate) use self::background_potential::*;
pub use self::background_potential::BackgroundPotentialType;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SampleType {
    Vertex,
    Face,
}

#[derive(Clone, Debug)]
pub struct ImplicitSurface {
    /// The type of kernel to use for fitting the data.
    kernel: KernelType,

    /// Enum for choosing how to compute a background potential field that will be mixed in with
    /// the local potentials. If `true`, this background potential will be automatically computed,
    /// if `false`, the values in the input will be used as the background potential to be mixed
    /// in.
    bg_potential_type: BackgroundPotentialType,

    /// Local search tree for fast proximity queries.
    spatial_tree: RTree<Sample<f64>>,

    /// Surface triangles representing the surface discretization to be approximated.
    /// This topology also defines the normals to the surface.
    surface_topo: Vec<[usize; 3]>,

    /// Save the vertex positions of the mesh because the samples may not coincide (e.g. face
    /// centered samples).
    surface_vertex_positions: Vec<Vector3<f64>>,

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

    /// The type of implicit surface. For example should the samples be centered at vertices or
    /// face centroids.
    sample_type: SampleType,
}

impl ImplicitSurface {
    const PARALLEL_CHUNK_SIZE: usize = 5000;

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

    /// Compute the gradient vector product of the face normals with respect to
    /// surface vertices.
    ///
    /// This function returns an iterator with the same size as `surface_vertices.len()`.
    ///
    /// Note that the product vector is given by a closure `multiplier` which must give a valid
    /// vector value for any vertex index, however not all indices will be used since only the
    /// neighbourhood of vertex at `index` will have non-zero gradients.
    pub(crate) fn compute_face_unit_normals_gradient_products<'a, T: Real, F>(
        samples: SamplesView<'a, 'a, T>,
        surface_vertices: &'a [Vector3<T>],
        surface_topo: &'a [[usize; 3]],
        mut multiplier: F,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        F: FnMut(Sample<T>) -> Vector3<T> + 'a,
    {
        samples.clone().into_iter().zip(surface_topo.iter()).flat_map(move |(sample, tri_indices)| {
            let norm_inv = T::one() / sample.nml.norm(); 
            let nml = sample.nml * norm_inv;
            let nml_proj = Matrix3::identity() - nml * nml.transpose();
            let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
            let mult = multiplier(sample);
            (0..3).map(move |i| tri.area_normal_gradient(i) * (nml_proj * (mult * norm_inv)))
        })
    }

    /// Update the stored samples. This assumes that vertex positions have been updated.
    fn update_samples(&mut self) {
        let ImplicitSurface {
            ref mut samples,
            ref surface_topo,
            ref surface_vertex_positions,
            sample_type,
            ..
        } = self;

        match sample_type {
            SampleType::Vertex => {
                let Samples { 
                    ref mut points,
                    ref mut normals,
                    ..
                } = samples;

                for (vertex_pos, sample_pos) in surface_vertex_positions.iter().zip(points.iter_mut()) {
                    *sample_pos = (*vertex_pos).into();
                }

                Self::compute_vertex_area_normals(surface_topo, points, normals);
            }
            SampleType::Face => {
                samples.update_triangle_samples(surface_topo, &surface_vertex_positions);
            }
        }
    }

    /// Update vertex positions and samples using an iterator over mesh vertices.
    pub fn update<I>(&mut self, vertex_iter: I)
    where
        I: Iterator<Item = [f64; 3]>,
    {
        // First we update the surface vertex positions.
        for (p, new_p) in self.surface_vertex_positions.iter_mut().zip(vertex_iter) {
            *p = new_p.into();
        }

        // Then update the samples that determine the shape of the implicit surface.
        self.update_samples();

        let ImplicitSurface {
            ref samples,
            ref mut spatial_tree,
            ..
        } = self;

        // Funally update the rtree responsible for neighbour search.
        *spatial_tree = build_rtree_from_samples(samples);
    }

    /// Compute neighbour cache if it has been invalidated
    pub fn cache_neighbours(&self, query_points: &[[f64; 3]]) {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            ref samples,
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
                let neigh = |_| samples.iter();
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
    ) -> Ref<'a, [(usize, Vec<usize>)]>
    where
        I: Iterator<Item = Sample<f64>> + 'a,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        {
            let mut cache = self.neighbour_cache.borrow_mut();
            cache.neighbour_points(query_points, neigh);
        }

        // Note there is no RefMut -> Ref map as of this writing, so we have to retrieve
        // neighbour_points twice: once to recompute cache, and once to return a Ref.
        Ref::map(self.neighbour_cache.borrow(), |c| c.cached_neighbour_points())
    }

    /// Set `neighbour_cache` to None. This triggers recomputation of the neighbour cache next time
    /// the potential or its derivatives are requested.
    pub fn invalidate_neighbour_cache(&self) {
        let mut cache = self.neighbour_cache.borrow_mut();
        cache.invalidate();
    }

    /// The number of query points currently in the cache.
    pub fn num_cached_query_points(&self) -> usize {
        let cache = self.neighbour_cache.borrow();
        cache.cached_neighbour_points().len()
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
            ref samples,
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
                let neigh = |_| samples.iter();
                let radius = 1.0;
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                self.compute_mls(query_points, radius, kern, neigh, out_potential)
            }
            KernelType::Hrbf => {
                // Global kernel, all points are neighbours.
                Self::compute_hrbf(query_points, samples, out_potential)
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
        I: Iterator<Item = Sample<f64>> + 'a,
        K: SphericalKernel<f64> + Copy + std::fmt::Debug + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        let neigh_points = self.cached_neighbours_borrow(query_points, neigh);

        assert_eq!(neigh_points.len(), out_potential.len());

        let ImplicitSurface {
            ref samples,
            bg_potential_type,
            ..
        } = *self;

        zip!(
            neigh_points
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
                bg_potential_type,
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
        bg_potential: BackgroundPotentialType,
        potential: &mut T,
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        if samples.is_empty() {
            return;
        }

        let radius = T::from(radius).unwrap();
        let bg = BackgroundPotential::new(q, samples, radius, kernel, BackgroundPotentialValue::val(bg_potential, *potential));
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
        let num_pts_per_sample = match self.sample_type {
            SampleType::Vertex => 1,
            SampleType::Face => 3,
        };
        cache.cached_neighbour_points().iter().map(|(_, pts)| pts.len()).sum::<usize>() * 3 * num_pts_per_sample
    }

    /// Compute the indices for the implicit surface potential jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = (usize, usize)> + 'a>, super::Error> {
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
    fn mls_surface_jacobian_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (usize, usize)> + 'a> {
        let cached_pts = {
            let cache = self.neighbour_cache.borrow();
            cache.cached_neighbour_points().to_vec()
        };
        match self.sample_type {
            SampleType::Vertex => 
                Box::new(cached_pts
                .into_iter()
                .enumerate()
                .flat_map(move |(row, (_, nbr_points))| {
                    nbr_points
                        .into_iter()
                        .flat_map(move |col| {
                            (0..3).map(move |i| (row, 3 * col + i))
                        })
                })),
            SampleType::Face => {
                let ImplicitSurface { ref surface_topo, .. } = *self;
                Box::new(cached_pts
                .into_iter()
                .enumerate()
                .flat_map(move |(row, (_, nbr_points))| {
                    nbr_points
                        .into_iter()
                        .flat_map(move |pidx| {
                            surface_topo[pidx].iter().flat_map(move |&col| {
                                (0..3).map(move |i| (row, 3 * col + i))
                            })
                        })
                }))
            }
        }
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    fn mls_surface_jacobian_indices(&self, rows: &mut [usize], cols: &mut [usize]) {
        // For each row
        let cache = self.neighbour_cache.borrow();
        match self.sample_type {
            SampleType::Vertex => {
                let row_col_iter = cache.cached_neighbour_points()
                .iter()
                .enumerate()
                .flat_map(move |(row, (_, nbr_points))| {
                    nbr_points
                        .iter()
                        .flat_map(move |&col| (0..3).map(move |i| (row, 3 * col + i)))
                });
                for ((row, col), out_row, out_col) in zip!(row_col_iter, rows.iter_mut(), cols.iter_mut()) {
                    *out_row = row;
                    *out_col = col;
                }
            }
            SampleType::Face => {
                let row_col_iter = cache.cached_neighbour_points()
                .iter()
                .enumerate()
                .flat_map(move |(row, (_, nbr_points))| {
                    nbr_points
                        .iter()
                        .flat_map(move |&pidx| {
                            self.surface_topo[pidx].iter().flat_map(move |&col| {
                                (0..3).map(move |i| (row, 3 * col + i))
                            })
                        })
                });
                for ((row, col), out_row, out_col) in zip!(row_col_iter, rows.iter_mut(), cols.iter_mut()) {
                    *out_row = row;
                    *out_col = col;
                }
            }
        };

    }

    fn vertex_jacobian_at<'a, T: Real, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        radius: f64,
        kernel: K,
        surface_topo: &'a [[usize;3]],
        dual_topo: &'a [Vec<usize>],
        bg_potential_type: BackgroundPotentialType,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
        where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, view, radius, kernel, bg_potential_type);

        // Background potential Jacobian.
        let bg_jac = bg.compute_jacobian();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let main_jac = Self::compute_jacobian_at(q, view, kernel, bg);

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_jac = ImplicitSurface::compute_vertex_unit_normals_gradient_products(
            view.clone(),
            &surface_topo,
            &dual_topo,
            move |Sample { pos, .. }| {
                let wk = kernel.with_closest_dist(closest_d).eval(q, pos);
                (q - pos) * (wk * weight_sum_inv)
            },
            );

        zip!(bg_jac, main_jac, nml_jac).map(|(b, m, n)| b + m + n)
    }

    fn face_jacobian_at<'a, T: Real, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        radius: f64,
        kernel: K,
        surface_topo: &'a [[usize;3]],
        surface_vertex_positions: &'a [Vector3<T>],
        bg_potential_type: BackgroundPotentialType,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
        where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, view, radius, kernel, bg_potential_type);

        // Background potential Jacobian.
        let bg_jac = bg.compute_jacobian();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let main_jac = Self::compute_jacobian_at(q, view, kernel, bg);

        let third = T::from(1.0/3.0).unwrap();

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_jac = Self::compute_face_unit_normals_gradient_products(
            view.clone(),
            surface_vertex_positions,
            &surface_topo,
            move |Sample { pos, .. }| {
                let wk = kernel.with_closest_dist(closest_d).eval(q, pos);
                (q - pos) * (wk * weight_sum_inv)
            },
            );

        // There are 3 contributions from each sample to each vertex.
        zip!(bg_jac, main_jac).flat_map(move |(b, m)| std::iter::repeat(b + m).take(3)).zip(nml_jac).map(move |(m, n)| (m * third + n))
    }

    fn mls_surface_jacobian_values<'a, I, K, N>(
        &self,
        query_points: &[[f64; 3]],
        radius: f64,
        kernel: K,
        neigh: N,
        values: &mut [f64],
    ) where
        I: Iterator<Item = Sample<f64>> + 'a,
        K: SphericalKernel<f64> + std::fmt::Debug + Copy + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        let value_vecs: &mut [[f64; 3]] = reinterpret::reinterpret_mut_slice(values);

        let neigh_points = self.cached_neighbours_borrow(query_points, neigh);
        
        let ImplicitSurface {
            ref samples,
            ref surface_topo,
            ref dual_topo,
            ref surface_vertex_positions,
            bg_potential_type,
            sample_type,
            ..
        } = * self;

        match sample_type {
            SampleType::Vertex => {
                // For each row (query point)
                let vtx_jac = neigh_points
                    .iter()
                    .map(|(qi, neigh)| (Vector3(query_points[*qi]), neigh))
                    .flat_map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::vertex_jacobian_at(q, view, radius, kernel, surface_topo, dual_topo, bg_potential_type)
                    });

                value_vecs.iter_mut().zip(vtx_jac)
                    .for_each(|(vec, new_vec)| {
                        *vec = new_vec.into();
                    });
            }
            SampleType::Face => {
                let face_jac = neigh_points
                    .iter()
                    .map(|(qi, neigh)| (Vector3(query_points[*qi]), neigh))
                    .flat_map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);

                        Self::face_jacobian_at(q, view, radius, kernel, surface_topo, surface_vertex_positions, bg_potential_type)
                    });

                value_vecs.iter_mut().zip(face_jac)
                    .for_each(|(vec, new_vec)| {
                        *vec = new_vec.into();
                    });
            }
        }
    }
    
    /// Compute the background potential field. This function returns a struct that provides some
    /// useful quanitities for computing derivatives of the field.
    pub(crate) fn compute_background_potential<'a, T: Real, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        radius: f64,
        kernel: K,
        bg_type: BackgroundPotentialType,
    ) -> BackgroundPotential<'a, T, K>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        // Find the closest vertex for background potential derivative.
        let radius = T::from(radius).unwrap();

        // Compute background potential derivative contribution.
        // Compute derivative if the closest point in the neighbourhood. Otherwise we
        // assume the background potential is constant.
        BackgroundPotential::new(q, samples, radius, kernel, BackgroundPotentialValue::jac(bg_type))
    }

    /// Compute the Jacobian for the implicit surface potential given by the samples with the
    /// specified kernel assuming constant normals. This Jacobian is with respect to sample points.
    pub(crate) fn compute_jacobian_at<'a, T: Real, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        bg: BackgroundPotential<'a, T, K>,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let closest_d = bg.closest_sample_dist();

        // Background potential adds to the total weight sum, so we should get the updated weight
        // sum from there.
        let weight_sum_inv = bg.weight_sum_inv();
        let weight_sum_inv2 = weight_sum_inv * weight_sum_inv;

        samples.clone().into_iter().map(
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
        )
    }

    fn compute_hrbf(
        query_points: &[[f64; 3]],
        samples: &Samples<f64>,
        out_potential: &mut [f64],
    ) -> Result<(), super::Error>
    {
        let Samples {
            ref points,
            ref normals,
            ref offsets,
        } = samples;

        let pts: Vec<crate::na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = p.into();
                crate::na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = normals
            .iter()
            .map(|&n| crate::na::Vector3::from(Into::<[f64;3]>::into(n)))
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
            ref samples,
            ..
        } = *self;

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
                self.compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
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
                self.compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
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
                self.compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Global { tolerance } => {
                let neigh = |_| samples.iter();
                let radius = 1.0;
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                self.compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            KernelType::Hrbf => {
                Self::compute_hrbf_on_mesh(mesh, samples, interrupt)
            }
        }
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls_on_mesh<'a, I, K, N, F, M>(
        &self,
        mesh: &mut M,
        radius: f64,
        kernel: K,
        neigh: N,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        I: Iterator<Item = Sample<f64>> + Clone + 'a,
        K: SphericalKernel<f64> + std::fmt::Debug + Copy + Sync + Send,
        N: Fn([f64; 3]) -> I + Sync + Send,
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<f64>,
    {
        let ImplicitSurface {
            ref samples,
            bg_potential_type,
            ..
        } = *self;

        // Move the potential attrib out of the mesh. We will reinsert it after we are done.
        let potential_attrib = mesh.remove_attrib::<VertexIndex>("potential")
            .ok() // convert to option (None when it doesn't exist)
            .unwrap_or(Attribute::from_vec(vec![0.0f32; mesh.num_vertices()]));

        let mut potential = potential_attrib.into_buffer().cast_into_vec::<f32>();
        if potential.is_empty() {
            // Couldn't cast, which means potential is of some non-numeric type.
            // We overwrite it because we need that attribute spot.
            potential = vec![0.0f32; mesh.num_vertices()];
        }

        let query_points = mesh.vertex_positions();
        let neigh_points = self.cached_neighbours_borrow(&query_points, neigh);

        // Construct a vector of neighbours for all query points (not just the ones with
        // neighbours).
        let mut neigh_all_points = vec![Vec::new(); query_points.len()];
        for (qi, neigh) in neigh_points.iter().cloned() {
            neigh_all_points[qi] = neigh;
        }

        // Initialize extra debug info.
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut bg_weight_attrib_data = vec![0f32; mesh.num_vertices()];
        let mut weight_sum_attrib_data = vec![0f32; mesh.num_vertices()];

        for (q_chunk, neigh, num_neighs_chunk, neighs_chunk, bg_weight_chunk, weight_sum_chunk, potential_chunk) in zip!(
            query_points.chunks(Self::PARALLEL_CHUNK_SIZE),
            neigh_all_points.chunks(Self::PARALLEL_CHUNK_SIZE),
            num_neighs_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            neighs_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            bg_weight_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            weight_sum_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            potential.chunks_mut(Self::PARALLEL_CHUNK_SIZE)
        ) {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            zip!(
                q_chunk.par_iter().map(|&v| Vector3(v)),
                neigh.par_iter(),
                num_neighs_chunk.par_iter_mut(),
                neighs_chunk.par_iter_mut(),
                bg_weight_chunk.par_iter_mut(),
                weight_sum_chunk.par_iter_mut(),
                potential_chunk.par_iter_mut()
            )
            .for_each(|(q, neighs, num_neighs, out_neighs, bg_weight, weight_sum, potential)| {
                let view = SamplesView::new(neighs, &samples);

                // Record number of neighbours in total.
                *num_neighs = view.len() as i32;

                // Record up to 11 neighbours
                for (k, neigh) in view.iter().take(11).enumerate() {
                    out_neighs[k] = neigh.index as i32;
                }

                if !view.is_empty() {
                    let bg = BackgroundPotential::new(q, view, radius, kernel,
                          BackgroundPotentialValue::val(bg_potential_type, *potential as f64));
                    let closest_d = bg.closest_sample_dist();
                    *bg_weight = bg.background_weight() as f32;
                    *weight_sum = (1.0 / bg.weight_sum_inv()) as f32;

                    *potential = bg.compute_unnormalized_weighted_potential() as f32;

                    let mut numerator = 0.0;
                    for Sample { pos, nml, off, .. } in view.iter() {
                        let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                        let p = off + nml.dot(q - pos) / nml.norm();

                        numerator += w * p;
                    }

                    *potential = (*potential + numerator as f32) * bg.weight_sum_inv() as f32;
                }
            });
        }

        {
            mesh.set_attrib_data::<_, VertexIndex>("num_neighbours", &num_neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("neighbours", &neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("bg_weight", &bg_weight_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("weight_sum", &weight_sum_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("potential", &potential)?;
        }

        Ok(())
    }

    fn compute_hrbf_on_mesh<F, M>(
        mesh: &mut M,
        samples: &Samples<f64>,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<f64>,
    {
        let Samples {
            ref points,
            ref normals,
            ref offsets,
        } = samples;
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

    fn easy_potential_derivative(radius: f64, bg_potential_type: BackgroundPotentialType) {
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

        // Eliminate no neighbour test.
        if bg_potential_type == BackgroundPotentialType::None && (q - samples.points[0]).norm() >= F::cst(radius) {
            // Nothing to test, the potential is expected to be NaN in this case.
            return;
        }

        // There is no surface for the set of samples. As a result, the normal derivative should be
        // skipped in this test.
        let surf_topo = vec![];
        let dual_topo = vec![vec![]];

        // Create a view of the samples for the Jacobian computation.
        let view = SamplesView::new(neighbours.as_ref(), &samples);

        // Compute the complete jacobian.
        let jac: Vec<Vector3<F>> = ImplicitSurface::vertex_jacobian_at(
            q,
            view,
            radius,
            kernel,
            &surf_topo,
            &dual_topo,
            bg_potential_type,
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
                bg_potential_type,
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
            easy_potential_derivative(radius, BackgroundPotentialType::None);
            easy_potential_derivative(radius, BackgroundPotentialType::Zero);
            easy_potential_derivative(radius, BackgroundPotentialType::FromInput);
            easy_potential_derivative(radius, BackgroundPotentialType::DistanceBased);
            easy_potential_derivative(radius, BackgroundPotentialType::NormalBased);
        }
    }

    /// Convert samples to autodiff number type constants.
    fn samples_to_autodiff(samples: Samples<f64>) -> Samples<F> {
        let Samples {
            points,
            normals,
            offsets,
        } = samples;

        Samples {
            points: points.into_iter().map(|vec| vec.map(|x| F::cst(x))).collect(),
            normals: normals.into_iter().map(|vec| vec.map(|x| F::cst(x))).collect(),
            offsets,
        }
    }

    /// A more complex test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    /// This function tests vertex based implicit surfaaces.
    fn hard_vertex_potential_derivative<P: FnMut() -> Vector3<f64>>(
        bg_potential_type: BackgroundPotentialType,
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

        let (tet_verts, tet_faces) = make_tet();

        let dual_topo = vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3]];

        // Initialize the samples with regular f64 for now to keep debug output clean.
        // Compute normals. Make sure this is done the same way as everywhere else.
        let mut normals = vec![Vector3::zeros(); tet_verts.len()];
        ImplicitSurface::compute_vertex_area_normals(&tet_faces, &tet_verts, &mut normals);

        let samples = Samples {
            points: tet_verts.clone(),
            normals: normals.clone(),
            offsets: vec![0.0; 4],
        };

        let neighbours = vec![0, 1, 2, 3];

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert the samples to use autodiff constants.
        let mut ad_samples = samples_to_autodiff(samples.clone());

        for &q in tri_verts.iter() {
            // Eliminate no neighbour test.
            if bg_potential_type == BackgroundPotentialType::None {
                if samples.points.iter().all(|&p| (q - p).norm() >= radius) {
                    // Nothing to test, the potential is expected to be NaN in this case.
                    continue;
                }
            }

            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);
            let jac: Vec<Vector3<f64>> = ImplicitSurface::vertex_jacobian_at(
                q,
                view,
                radius,
                kernel,
                &tet_faces,
                &dual_topo,
                bg_potential_type,
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
                        bg_potential_type,
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

    /// A more complex test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    /// This function tests face centric implicit surfaces.
    fn hard_face_potential_derivative<P: FnMut() -> Vector3<f64>>(
        bg_potential_type: BackgroundPotentialType,
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

        let (tet_verts, tet_faces) = make_tet();

        let samples = Samples::new_triangle_samples(&tet_faces, &tet_verts, vec![0.0; 4]);

        let neighbours = vec![0, 1, 2, 3]; // All tet faces

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> =
            tet_verts.iter().cloned().map(|v| v.map(|x| F::cst(x))).collect();

        for &q in tri_verts.iter() {
            // Eliminate no neighbour test when there is no background potential.
            if bg_potential_type == BackgroundPotentialType::None {
                if samples.points.iter().all(|&p| (q - p).norm() >= radius) {
                    // Nothing to test, the potential is expected to be NaN in this case.
                    continue;
                }
            }

            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);
            let jac: Vec<Vector3<f64>> = ImplicitSurface::face_jacobian_at(
                q,
                view,
                radius,
                kernel,
                &tet_faces,
                &tet_verts,
                bg_potential_type,
            )
            .collect();

            assert_eq!(jac.len(), 3*neighbours.len());

            // Reduce the Jacobian from face vertices to vertices.
            let tet_indices: &[usize] = reinterpret::reinterpret_slice(&tet_faces);
            let mut vert_jac = vec![Vector3::zeros(); tet_verts.len()];
            for (&jac, &vtx_idx) in jac.iter().zip(tet_indices) {
                vert_jac[vtx_idx] += jac;
            }

            assert_eq!(vert_jac.len(), tet_verts.len());

            let q = q.map(|x| F::cst(x));

            for &vtx in neighbours.iter() {
                for i in 0..3 {
                    ad_tet_verts[vtx][i] = F::var(ad_tet_verts[vtx][i]);

                    let ad_samples = Samples::new_triangle_samples(&tet_faces, &ad_tet_verts, vec![0.0; 4]);

                    let view = SamplesView::new(neighbours.as_ref(), &ad_samples);
                    let mut p = F::cst(0.0);
                    ImplicitSurface::compute_local_potential_at(
                        q,
                        view,
                        radius,
                        kernel,
                        bg_potential_type,
                        &mut p,
                    );

                    assert_relative_eq!(
                        vert_jac[vtx][i],
                        p.deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );

                    ad_tet_verts[vtx][i] = F::cst(ad_tet_verts[vtx][i]);
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
            hard_vertex_potential_derivative(BackgroundPotentialType::None, radius, &mut perturb);
            hard_vertex_potential_derivative(BackgroundPotentialType::Zero, radius, &mut perturb);
            hard_vertex_potential_derivative(BackgroundPotentialType::FromInput, radius, &mut perturb);
            hard_vertex_potential_derivative(BackgroundPotentialType::DistanceBased, radius, &mut perturb);
            hard_vertex_potential_derivative(BackgroundPotentialType::NormalBased, radius, &mut perturb);

            hard_face_potential_derivative(BackgroundPotentialType::None, radius, &mut perturb);
            hard_face_potential_derivative(BackgroundPotentialType::Zero, radius, &mut perturb);
            hard_face_potential_derivative(BackgroundPotentialType::FromInput, radius, &mut perturb);
            hard_face_potential_derivative(BackgroundPotentialType::DistanceBased, radius, &mut perturb);
            hard_face_potential_derivative(BackgroundPotentialType::NormalBased, radius, &mut perturb);
        }
    }

    /// Compute normalized area weighted vertex normals given a triangle topology.
    /// This is a helper function for the `vertex_normal_derivative_test`.
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

    /// Generate a tetrahedron with vertex positions and indices for the triangle faces.
    fn make_tet() -> (Vec<Vector3<f64>>, Vec<[usize;3]>){
        let tet_verts = vec![
            Vector3([0.0, 1.0, 0.0]),
            Vector3([-0.94281, -0.33333, 0.0]),
            Vector3([0.471405, -0.33333, 0.816498]),
            Vector3([0.471405, -0.33333, -0.816498]),
        ];

        let tet_faces = vec![[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]];

        (tet_verts, tet_faces)
    }

    /// Test the derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_derivative_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

        let (tet_verts, tet_faces) = make_tet();

        // Initialize the samples with regular f64 for now to keep debug output clean.
        let samples = Samples::new_triangle_samples(&tet_faces, &tet_verts, vec![0.0; 4]);

        let indices = vec![0, 1, 2, 3]; // look at all the faces

        // Set a random product vector.
        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-1.0, 1.0);
        let multipliers: Vec<_> = (0..tet_faces.len())
            .map(move |_| Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]))
            .collect();
        let multiplier = move |Sample { index, .. }| multipliers[index];

        // Compute the normal gradient product.
        let view = SamplesView::new(indices.as_ref(), &samples);
        let grad_iter = ImplicitSurface::compute_face_unit_normals_gradient_products(
            view,
            &tet_verts,
            &tet_faces,
            multiplier.clone(),
        );

        let tet_indices: &[usize] = reinterpret::reinterpret_slice(&tet_faces);
        let mut vert_grad = vec![Vector3::zeros(); tet_verts.len()];
        for (g, &vtx_idx) in grad_iter.zip(tet_indices) {
            vert_grad[vtx_idx] += g;
        }

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> = tet_verts.iter().cloned().map(|v| v.map(|x| F::cst(x))).collect();

        for (vtx, g) in vert_grad.iter().enumerate() {
            for i in 0..3 {
                ad_tet_verts[vtx][i] = F::var(ad_tet_verts[vtx][i]);

                // Convert the samples to use autodiff constants.
                let mut ad_samples = Samples::new_triangle_samples(&tet_faces, &ad_tet_verts, vec![0.0; 4]);

                // Normalize face normals
                for nml in ad_samples.normals.iter_mut() {
                    *nml = *nml / nml.norm();
                }

                let mut exp = F::cst(0.0);
                for sample in view.clone().iter() {
                    exp += ad_samples.normals[sample.index].dot(multiplier(sample).map(|x| F::cst(x)));
                }

                assert_relative_eq!(g[i], exp.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                ad_tet_verts[vtx][i] = F::cst(ad_tet_verts[vtx][i]);
            }
        }
    }

    /// Test the derivatives of our normal computation method for vertex normals.
    #[test]
    fn vertex_normal_derivative_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

        let (tet_verts, tet_faces) = make_tet();

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
        let mut ad_samples = samples_to_autodiff(samples.clone());

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
        let bg = BackgroundPotential::new(q, view, radius, kernel,
                                          BackgroundPotentialValue::jac(BackgroundPotentialType::DistanceBased));

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
                let ad_bg = BackgroundPotential::new(q, view, F::cst(radius), kernel,
                BackgroundPotentialValue::val(BackgroundPotentialType::DistanceBased, F::cst(0.0)));

                let p = ad_bg.compute_unnormalized_weighted_potential() * ad_bg.weight_sum_inv();

                assert_relative_eq!(jac[i][j], p.deriv());
                ad_samples.points[i][j] = F::cst(ad_samples.points[i][j]);
            }
        }
    }
}
