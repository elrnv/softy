//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::kernel::{self, KernelType, SphericalKernel};
use geo::math::{Matrix3, Vector3};
use geo::mesh::{attrib::*, topology::VertexIndex, VertexMesh};
use geo::prim::Triangle;
use geo::Real;
use nalgebra as na;
use num_traits::cast;
use rayon::prelude::*;
use spade::rtree::RTree;
use std::cell::{Ref, RefCell};

pub mod background_field;
pub mod builder;
pub mod hessian;
pub mod jacobian;
pub mod neighbour_cache;
pub mod samples;
pub mod spatial_tree;

pub use self::builder::*;
pub use self::samples::*;
pub use self::spatial_tree::*;

pub use self::background_field::BackgroundFieldType;
pub(crate) use self::background_field::*;
pub(crate) use self::neighbour_cache::Neighbourhood;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SampleType {
    Vertex,
    Face,
}

/// Implicit surface type. `V` is the value type of the implicit function. Note that if `V` is a
/// vector, this type will fit a vector field.
#[derive(Clone, Debug)]
pub struct ImplicitSurface<T = f64>
where
    T: Real,
{
    /// The type of kernel to use for fitting the data.
    kernel: KernelType,

    /// Enum for choosing how to compute a background potential field that will be mixed in with
    /// the local potentials. If `true`, this background potential will be automatically computed,
    /// if `false`, the values in the input will be used as the background potential to be mixed
    /// in.
    bg_field_type: BackgroundFieldType,

    /// Local search tree for fast proximity queries.
    spatial_tree: RTree<Sample<T>>,

    /// Surface triangles representing the surface discretization to be approximated.
    /// This topology also defines the normals to the surface.
    surface_topo: Vec<[usize; 3]>,

    /// Save the vertex positions of the mesh because the samples may not coincide (e.g. face
    /// centered samples).
    surface_vertex_positions: Vec<Vector3<T>>,

    /// Sample points defining the entire implicit surface.
    samples: Samples<T>,

    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative may be denser
    /// than then needed, which typically results in slower performance.  If it is set too low,
    /// there may be errors in the derivative. It is the callers responsibility to set this step
    /// accurately using `update_max_step`. If the implicit surface is not changing, leave this at
    /// 0.0.
    max_step: T,

    /// Cache the neighbouring sample points for each query point we see. This cache can be
    /// invalidated explicitly when the sparsity pattern is expected to change. This is wrapped in
    /// a `RefCell` because it may be updated in non mutable functions since it's a cache.
    query_neighbourhood: RefCell<Neighbourhood>,

    /// Vertex neighbourhood topology. For each vertex, this vector stores all the indices to
    /// adjacent triangles.
    dual_topo: Vec<Vec<usize>>,

    /// The type of implicit surface. For example should the samples be centered at vertices or
    /// face centroids.
    sample_type: SampleType,
}

impl<T: Real + Send + Sync> ImplicitSurface<T> {
    const PARALLEL_CHUNK_SIZE: usize = 5000;

    /// Return the number of samples used by this implicit surface.
    pub fn samples(&self) -> &Samples<T> {
        &self.samples
    }

    /// Return the number of samples used by this implicit surface.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Compute unnormalized area weighted vertex normals given a triangle topology.
    /// Having this function be static and generic helps us test the normal derivatives.
    pub(crate) fn compute_vertex_area_normals<V3>(
        surf_topo: &[[usize; 3]],
        points: &[V3],
        normals: &mut [Vector3<T>],
    ) where
        V3: Into<Vector3<T>> + Clone,
    {
        // Clear the normals.
        for nml in normals.iter_mut() {
            *nml = Vector3::zeros();
        }

        for tri_indices in surf_topo.iter() {
            let tri = Triangle::from_indexed_slice(tri_indices, points);
            let area_nml = tri.area_normal();
            normals[tri_indices[0]] += area_nml;
            normals[tri_indices[1]] += area_nml;
            normals[tri_indices[2]] += area_nml;
        }
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
                    // ref mut tangents
                    ..
                } = samples;

                for (vertex_pos, sample_pos) in
                    surface_vertex_positions.iter().zip(points.iter_mut())
                {
                    *sample_pos = *vertex_pos;
                }

                Self::compute_vertex_area_normals(surface_topo, points, normals);
            }
            SampleType::Face => {
                samples.update_triangle_samples(surface_topo, &surface_vertex_positions);
            }
        }
    }

    /// Update vertex positions and samples using an iterator over mesh vertices. This is a very
    /// permissive `update` function, which will update as many positions as possible and recompute
    /// the implicit surface data (like samples and spatial tree if needed) whether or not enough
    /// positions were specified to cover all surface vertices. This function will return the
    /// number of vertices that were indeed updated.
    pub fn update<I>(&mut self, vertex_iter: I) -> usize
    where
        I: Iterator<Item = [T; 3]>,
    {
        // First we update the surface vertex positions.
        let mut num_updated = 0;
        for (p, new_p) in self.surface_vertex_positions.iter_mut().zip(vertex_iter) {
            *p = new_p.into();
            num_updated += 1;
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

        num_updated
    }

    /// Compute neighbour cache if it has been invalidated
    pub fn cache_neighbours(&self, query_points: &[[T; 3]]) {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            ref samples,
            ref surface_topo,
            ref dual_topo,
            max_step,
            sample_type,
            ..
        } = *self;

        match *kernel {
            KernelType::Interpolating { radius }
            | KernelType::Approximate { radius, .. }
            | KernelType::Cubic { radius } => {
                let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    let q_pos = Vector3(q).cast::<f64>().unwrap().into();
                    spatial_tree
                        .lookup_in_circle(&q_pos, &radius2)
                        .into_iter()
                        .cloned()
                };
                let mut cache = self.query_neighbourhood.borrow_mut();
                cache.compute_neighbourhoods(query_points, neigh, surface_topo, dual_topo, sample_type);
            }
            KernelType::Global { .. } | KernelType::Hrbf => {
                // Global kernel, all points are neighbours
                let neigh = |_| samples.iter();
                let mut cache = self.query_neighbourhood.borrow_mut();
                cache.compute_neighbourhoods(query_points, neigh, surface_topo, dual_topo, sample_type);
            }
        }
    }

    /// This function returns previously cached neighbours if the neighbour cache is
    /// valid, and `None` otherwise.
    fn trivial_neighbourhood_borrow<'a>(&'a self) -> Result<Ref<'a, [Vec<usize>]>, super::Error> {
        // Note there is no RefMut -> Ref map as of this writing, which is why recomputing the
        // cache and actually retrieving the neighbour points is done separately
        let valid = self.query_neighbourhood.borrow().trivial_set().is_some();
        if valid {
            Ok(Ref::map(self.query_neighbourhood.borrow(), |c| {
                c.trivial_set().unwrap()
            }))
        } else {
            Err(super::Error::MissingNeighbourData)
        }
    }

    /// This function returns previously cached neighbours if the neighbour cache is
    /// valid, and `None` otherwise.
    fn extended_neighbourhood_borrow<'a>(&'a self) -> Result<Ref<'a, [Vec<usize>]>, super::Error> {
        // Note there is no RefMut -> Ref map as of this writing, which is why recomputing the
        // cache and actually retrieving the neighbour points is done separately
        let valid = self.query_neighbourhood.borrow().extended_set().is_some();
        if valid {
            Ok(Ref::map(self.query_neighbourhood.borrow(), |c| {
                c.extended_set().unwrap()
            }))
        } else {
            Err(super::Error::MissingNeighbourData)
        }
    }

    /// Set `query_neighbourhood` to None. This triggers recomputation of the neighbour cache next time
    /// the potential or its derivatives are requested.
    pub fn invalidate_query_neighbourhood(&self) {
        let mut cache = self.query_neighbourhood.borrow_mut();
        cache.invalidate();
    }

    /// The number of query points currently in the cache.
    pub fn num_cached_query_points(&self) -> usize {
        let cache = self.query_neighbourhood.borrow();
        cache.trivial_set().map_or(0, |x| x.len())
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        self.max_step = max_step;
    }

    /// Project the given set of positions to be above the specified iso-value along the gradient.
    /// If the query point is already above the given iso-value, then it is not modified.
    /// The given `epsilon` determines how far above the iso-surface the point is allowed to be
    /// projected, essentially it is the thickness above the iso-surface of value projections.
    /// This function will return true if convergence is achieve and false if the projection needed
    /// more iterations.
    pub fn project_to_above(
        &self,
        iso_value: T,
        epsilon: T,
        query_points: &mut [[T; 3]],
    ) -> Result<bool, super::Error> {
        let mut candidate_points = query_points.to_vec();
        let mut potential = vec![T::zero(); query_points.len()];
        let mut candidate_potential = vec![T::zero(); query_points.len()];
        let mut steps = vec![[T::zero(); 3]; query_points.len()];

        let max_steps = 10;
        let max_binary_search_iters = 10;

        let mut convergence = true;

        for i in 0..max_steps {
            self.potential(query_points, &mut potential)?;

            // Count the number of points with values less than iso_value.
            let count_violations = potential.iter().filter(|&&x| x < iso_value).count();

            if count_violations == 0 {
                break;
            }

            // The transpose of the potential gradient at each of the query points.
            self.query_jacobian(query_points, &mut steps)?;

            // Compute initial step directions
            for (step, &value) in
                zip!(steps.iter_mut(), potential.iter()).filter(|(_, &pot)| pot < iso_value)
            {
                let nml = Vector3(*step);
                let offset = (epsilon * T::from(0.5).unwrap() + (iso_value - value)) / nml.norm();
                *step = (nml * offset).into();
            }

            for j in 0..max_binary_search_iters {
                // Try this step
                for (p, q, &step, _) in zip!(
                    candidate_points.iter_mut(),
                    query_points.iter(),
                    steps.iter(),
                    potential.iter()
                )
                .filter(|(_, _, _, &pot)| pot < iso_value)
                {
                    *p = (Vector3(*q) + Vector3(step)).into();
                }

                self.potential(&candidate_points, &mut candidate_potential)?;

                let mut count_overshoots = 0;
                for (step, _, _) in zip!(
                    steps.iter_mut(),
                    potential.iter(),
                    candidate_potential.iter()
                )
                .filter(|(_, &old, &new)| old < iso_value && new > iso_value + epsilon)
                {
                    *step = (Vector3(*step) * T::from(0.5).unwrap()).into();
                    count_overshoots += 1;
                }

                if count_overshoots == 0 {
                    break;
                }

                if j == max_binary_search_iters - 1 {
                    convergence = false;
                }
            }

            // Update query points
            query_points
                .iter_mut()
                .zip(candidate_points.iter())
                .for_each(|(q, p)| *q = *p);

            if i == max_steps - 1 {
                convergence = false;
            }
        }

        Ok(convergence)
    }

    /// Compute the implicit surface potential.
    pub fn potential(
        &self,
        query_points: &[[T; 3]],
        out_field: &mut [T],
    ) -> Result<(), super::Error> {
        let ImplicitSurface {
            ref kernel,
            ref samples,
            ..
        } = *self;

        match *kernel {
            KernelType::Interpolating { radius } => {
                let kern = kernel::LocalInterpolating::new(radius);
                self.compute_mls(query_points, kern, out_field)
            }
            KernelType::Approximate { tolerance, radius } => {
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                self.compute_mls(query_points, kern, out_field)
            }
            KernelType::Cubic { radius } => {
                let kern = kernel::LocalCubic::new(radius);
                self.compute_mls(query_points, kern, out_field)
            }
            KernelType::Global { tolerance } => {
                // Global kernel, all points are neighbours
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                self.compute_mls(query_points, kern, out_field)
            }
            KernelType::Hrbf => {
                // Global kernel, all points are neighbours.
                Self::compute_hrbf(query_points, samples, out_field)
            }
        }
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls<'a, K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        out_field: &'a mut [T],
    ) -> Result<(), super::Error>
    where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        self.cache_neighbours(query_points);
        let neigh_points = self.trivial_neighbourhood_borrow()?;

        assert_eq!(neigh_points.len(), out_field.len());

        let ImplicitSurface {
            ref samples,
            bg_field_type,
            ..
        } = *self;

        zip!(
            query_points.par_iter(),
            neigh_points.par_iter(),
            out_field.par_iter_mut()
        )
        .filter(|(_, nbrs, _)| !nbrs.is_empty())
        .for_each(move |(q, neighbours, field)| {
            Self::compute_potential_at(
                Vector3(*q),
                SamplesView::new(neighbours, samples),
                kernel,
                bg_field_type,
                field,
            );
        });

        Ok(())
    }

    /// Compute the potential at a given query point. If the potential is invalid or nieghbourhood
    /// is empty, `potential` is not modified, otherwise it's updated.
    /// Note: passing the output parameter potential as a mut reference allows us to optionally mix
    /// a preinitialized custom global potential field with the local potential.
    pub(crate) fn compute_potential_at<K>(
        q: Vector3<T>,
        samples: SamplesView<T>,
        kernel: K,
        bg_potential: BackgroundFieldType,
        potential: &mut T,
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        if samples.is_empty() {
            return;
        }

        let bg = BackgroundField::new(
            q,
            samples,
            kernel,
            BackgroundFieldValue::val(bg_potential, *potential),
        );

        let weight_sum_inv = bg.weight_sum_inv();

        // Generate a background potential field for every query point. This will be mixed
        // in with the computed potentials for local methods.
        *potential = bg.compute_unnormalized_weighted_scalar_field() * weight_sum_inv;

        let local_field = Self::compute_local_potential_at(q, samples, kernel, weight_sum_inv, bg.closest_sample_dist());

        *potential += local_field;
    }

    /// Compiute the potential field (excluding background field) at a given query point. If the
    #[inline]
    pub(crate) fn compute_local_potential_at<K>(
        q: Vector3<T>,
        samples: SamplesView<T>,
        kernel: K,
        weight_sum_inv: T,
        closest_d: T,
    ) -> T
    where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        samples.iter().map(|Sample { pos, nml, value, .. }| {
            let w = kernel.with_closest_dist(closest_d).eval(q, pos);
            let p = value + nml.dot(q - pos) / nml.norm();
            w * p
        }).sum::<T>() * weight_sum_inv
    }

    /*
     * The following functions interpolate vector fields instead of potentials
     */

    /// Compute vector field on the surface.
    pub fn vector_field(
        &self,
        query_points: &[[T; 3]],
        out_vectors: &mut [[T; 3]],
    ) -> Result<(), super::Error> {
        match self.kernel {
            KernelType::Interpolating { radius } => {
                let kern = kernel::LocalInterpolating::new(radius);
                self.compute_mls_vector_field(query_points, kern, out_vectors)
            }
            KernelType::Approximate { tolerance, radius } => {
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                self.compute_mls_vector_field(query_points, kern, out_vectors)
            }
            KernelType::Cubic { radius } => {
                let kern = kernel::LocalCubic::new(radius);
                self.compute_mls_vector_field(query_points, kern, out_vectors)
            }
            KernelType::Global { tolerance } => {
                // Global kernel, all points are neighbours
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                self.compute_mls_vector_field(query_points, kern, out_vectors)
            }
            _ => {
                // unimplemented ( do nothing )
                Err(super::Error::UnsupportedKernel)
            }
        }
    }

    /// Interpolate the given vector field at the given query points.
    fn compute_mls_vector_field<'a, K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        out_vectors: &'a mut [[T; 3]],
    ) -> Result<(), super::Error>
    where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        self.cache_neighbours(query_points);
        let neigh_points = self.trivial_neighbourhood_borrow()?;

        assert_eq!(neigh_points.len(), out_vectors.len());

        let ImplicitSurface {
            ref samples,
            bg_field_type,
            ..
        } = *self;

        zip!(
            query_points.par_iter(),
            neigh_points.par_iter(),
            out_vectors.par_iter_mut()
        )
        .filter(|(_, nbrs, _)| !nbrs.is_empty())
        .for_each(move |(q, neighbours, vector)| {
            Self::compute_local_vector_at(
                Vector3(*q),
                SamplesView::new(neighbours, samples),
                kernel,
                bg_field_type,
                vector,
            );
        });

        Ok(())
    }

    pub(crate) fn compute_local_vector_at<K>(
        q: Vector3<T>,
        samples: SamplesView<T>,
        kernel: K,
        bg_potential: BackgroundFieldType,
        vector: &mut [T; 3],
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        if samples.is_empty() {
            return;
        }

        let bg = BackgroundField::new(
            q,
            samples,
            kernel,
            BackgroundFieldValue::val(bg_potential, Vector3(*vector)),
        );
        let closest_dist = bg.closest_sample_dist();

        let weight_sum_inv = bg.weight_sum_inv();

        // Generate a background potential field for every query point. This will be mixed
        // in with the computed potentials for local methods.
        let mut out_field = bg.compute_unnormalized_weighted_vector_field();

        let grad_w_sum_normalized =
            Self::normalized_neighbour_weight_gradient(q, samples, kernel, bg);

        for Sample {
            pos, vel, value, ..
        } in samples.iter()
        {
            out_field += Self::sample_contact_jacobian_at(
                q,
                pos,
                value,
                kernel,
                vel,
                grad_w_sum_normalized,
                weight_sum_inv,
                closest_dist,
            );
        }

        *vector = out_field.into();
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
        M: VertexMesh<T>,
    {
        let ImplicitSurface {
            ref kernel,
            ref samples,
            ..
        } = *self;

        match *kernel {
            KernelType::Interpolating { radius } => {
                let kern = kernel::LocalInterpolating::new(radius);
                self.compute_mls_on_mesh(mesh, kern, interrupt)
            }
            KernelType::Approximate { tolerance, radius } => {
                let kern = kernel::LocalApproximate::new(radius, tolerance);
                self.compute_mls_on_mesh(mesh, kern, interrupt)
            }
            KernelType::Cubic { radius } => {
                let kern = kernel::LocalCubic::new(radius);
                self.compute_mls_on_mesh(mesh, kern, interrupt)
            }
            KernelType::Global { tolerance } => {
                let kern = kernel::GlobalInvDistance2::new(tolerance);
                self.compute_mls_on_mesh(mesh, kern, interrupt)
            }
            KernelType::Hrbf => Self::compute_hrbf_on_mesh(mesh, samples, interrupt),
        }
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls_on_mesh<'a, K, F, M>(
        &self,
        mesh: &mut M,
        kernel: K,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
    {
        let ImplicitSurface {
            ref samples,
            bg_field_type,
            ..
        } = *self;

        // Move the potential attrib out of the mesh. We will reinsert it after we are done.
        let potential_attrib = mesh
            .remove_attrib::<VertexIndex>("potential")
            .ok() // convert to option (None when it doesn't exist)
            .unwrap_or_else(|| Attribute::from_vec(vec![0.0f32; mesh.num_vertices()]));

        let mut potential = potential_attrib.into_buffer().cast_into_vec::<f32>();
        if potential.is_empty() {
            // Couldn't cast, which means potential is of some non-numeric type.
            // We overwrite it because we need that attribute spot.
            potential = vec![0.0f32; mesh.num_vertices()];
        }

        // Overwrite these attributes.
        mesh.remove_attrib::<VertexIndex>("normals").ok();
        let mut normals = vec![[0.0f32; 3]; mesh.num_vertices()];
        mesh.remove_attrib::<VertexIndex>("tangents").ok();
        let mut tangents = vec![[0.0f32; 3]; mesh.num_vertices()];

        let query_points = mesh.vertex_positions();
        self.cache_neighbours(&query_points);
        let neigh_points = self.trivial_neighbourhood_borrow()?;

        // Initialize extra debug info.
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut bg_weight_attrib_data = vec![0f32; mesh.num_vertices()];
        let mut weight_sum_attrib_data = vec![0f32; mesh.num_vertices()];

        for (
            q_chunk,
            neigh,
            num_neighs_chunk,
            neighs_chunk,
            bg_weight_chunk,
            weight_sum_chunk,
            potential_chunk,
            normals_chunk,
            tangents_chunk,
        ) in zip!(
            query_points.chunks(Self::PARALLEL_CHUNK_SIZE),
            neigh_points.chunks(Self::PARALLEL_CHUNK_SIZE),
            num_neighs_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            neighs_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            bg_weight_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            weight_sum_attrib_data.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            potential.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            normals.chunks_mut(Self::PARALLEL_CHUNK_SIZE),
            tangents.chunks_mut(Self::PARALLEL_CHUNK_SIZE)
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
                potential_chunk.par_iter_mut(),
                normals_chunk.par_iter_mut(),
                tangents_chunk.par_iter_mut()
            )
            .for_each(
                |(
                    q,
                    neighs,
                    num_neighs,
                    out_neighs,
                    bg_weight,
                    weight_sum,
                    potential,
                    normal,
                    tangent,
                )| {
                    let view = SamplesView::new(neighs, &samples);

                    // Record number of neighbours in total.
                    *num_neighs = view.len() as i32;

                    // Record up to 11 neighbours
                    for (k, neigh) in view.iter().take(11).enumerate() {
                        out_neighs[k] = neigh.index as i32;
                    }

                    if !view.is_empty() {
                        let bg = BackgroundField::new(
                            q,
                            view,
                            kernel,
                            BackgroundFieldValue::val(bg_field_type, T::from(*potential).unwrap()),
                        );
                        let closest_d = bg.closest_sample_dist();
                        *bg_weight = bg.background_weight().to_f32().unwrap();
                        *weight_sum = bg.weight_sum.to_f32().unwrap();
                        let weight_sum_inv = bg.weight_sum_inv();

                        *potential = bg
                            .compute_unnormalized_weighted_scalar_field()
                            .to_f32()
                            .unwrap();

                        let mut grad_w_sum_normalized = Vector3::zeros();
                        for grad in samples.iter().map(|Sample { pos, .. }| {
                            kernel.with_closest_dist(closest_d).grad(q, pos)
                        }) {
                            grad_w_sum_normalized += grad;
                        }
                        grad_w_sum_normalized *= weight_sum_inv;

                        let mut out_normal = Vector3::zeros();
                        let mut out_tangent = Vector3::zeros();

                        let mut numerator = T::zero();
                        for Sample {
                            pos,
                            nml,
                            vel,
                            value,
                            ..
                        } in view.iter()
                        {
                            let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                            let grad_w = kernel.with_closest_dist(closest_d).grad(q, pos);
                            let w_normalized = w * weight_sum_inv;
                            let grad_w_normalized =
                                grad_w * weight_sum_inv - grad_w_sum_normalized * w_normalized;

                            let p = value + nml.dot(q - pos) / nml.norm();

                            numerator += w * p;
                            out_normal +=
                                grad_w_normalized * (q - pos).dot(nml) + nml * w_normalized;
                            out_tangent +=
                                grad_w_normalized * (q - pos).dot(vel) + vel * w_normalized;
                        }

                        *potential = (*potential + numerator.to_f32().unwrap())
                            * bg.weight_sum_inv().to_f32().unwrap();
                        *normal = out_normal.map(|x| x.to_f32().unwrap()).into();
                        *tangent = out_tangent.map(|x| x.to_f32().unwrap()).into();
                    }
                },
            );
        }

        {
            mesh.set_attrib_data::<_, VertexIndex>("num_neighbours", &num_neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("neighbours", &neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("bg_weight", &bg_weight_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("weight_sum", &weight_sum_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("potential", &potential)?;
            mesh.set_attrib_data::<_, VertexIndex>("normals", &normals)?;
            mesh.set_attrib_data::<_, VertexIndex>("tangents", &tangents)?;
        }

        Ok(())
    }

    /// Compute the gradient vector product of the `compute_vertex_unit_normals` function with respect to
    /// vertices given in the sample view.
    ///
    /// This function returns an iterator with the same size as `samples.len()`.
    ///
    /// Note that the product vector is given by a closure `dx` which must give a valid vector
    /// value for any vertex index, however not all indices will be used since only the
    /// neighbourhood of vertex at `index` will have non-zero gradients.
    pub(crate) fn compute_vertex_unit_normals_gradient_products<'a, F>(
        samples: SamplesView<'a, 'a, T>,
        surface_topo: &'a [[usize; 3]],
        dual_topo: &'a [Vec<usize>],
        mut dx: F,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        F: FnMut(Sample<T>) -> Vector3<T> + 'a,
    {
        samples.into_iter().map(move |sample| {
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
                for sample in samples.from_view(tri_indices).into_iter() {
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
    pub(crate) fn compute_face_unit_normals_gradient_products<'a, F>(
        samples: SamplesView<'a, 'a, T>,
        surface_vertices: &'a [Vector3<T>],
        surface_topo: &'a [[usize; 3]],
        mut multiplier: F,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        F: FnMut(Sample<T>) -> Vector3<T> + 'a,
    {
        samples.into_iter().flat_map(move |sample| {
            let mult = multiplier(sample);
            let grad = Self::face_unit_normal_gradient_iter(sample, surface_vertices, surface_topo);
            grad.map(move |g| g * mult)
        })
    }

    /// Compute the gradient of the face normal at the given sample with respect to
    /// its vertices. The returned triple of `Matrix3`s corresonds to the block column vector of
    /// three matrices corresponding to each triangle vertex, which together construct the actual
    /// `9x3` component-wise gradient.
    pub(crate) fn face_unit_normal_gradient_iter(
        sample: Sample<T>,
        surface_vertices: &[Vector3<T>],
        surface_topo: &[[usize; 3]],
    ) -> impl Iterator<Item = Matrix3<T>> {
        let nml_proj = Self::scaled_tangent_projection(sample);
        let tri_indices = &surface_topo[sample.index];
        let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
        (0..3).map(move |i| tri.area_normal_gradient(i) * nml_proj)
    }

    /// Compute the gradient of the face normal at the given sample with respect to
    /// the given vertex.
    pub(crate) fn face_unit_normal_gradient(
        sample: Sample<T>,
        vtx_idx: usize,
        surface_vertices: &[Vector3<T>],
        surface_topo: &[[usize; 3]],
    ) -> Matrix3<T> {
        let nml_proj = Self::scaled_tangent_projection(sample);
        let tri_indices = &surface_topo[sample.index];
        let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
        tri.area_normal_gradient(vtx_idx) * nml_proj
    }

    /// Compute the matrix for projecting on the tangent plane of the given sample inversely scaled
    /// by the local area (normal norm reciprocal).
    pub(crate) fn scaled_tangent_projection(
        sample: Sample<T>,
    ) -> Matrix3<T> {
        let nml_norm_inv = T::one() / sample.nml.norm();
        let nml = sample.nml * nml_norm_inv;
        Matrix3::diag([nml_norm_inv;3]) - (nml * nml_norm_inv) * nml.transpose()
    }

    /// Compute the symmetric jacobian of the face normals with respect to
    /// surface vertices. This is the Jacobian plus its transpose.
    /// This function is needed to compute the Hessian, which means we are only interested in the
    /// lower triangular part.
    pub(crate) fn face_unit_normals_symmetric_jacobian<'a, F>(
        samples: SamplesView<'a, 'a, T>,
        surface_vertices: &'a [Vector3<T>],
        surface_topo: &'a [[usize; 3]],
        mut multiplier: F,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
    where
        F: FnMut(Sample<T>) -> T + 'a,
    {
        let third = T::one() / T::from(3.0).unwrap();
        samples.into_iter().flat_map(move |sample| {
            let tri_indices = &surface_topo[sample.index];
            let nml_proj = Self::scaled_tangent_projection(sample); // symmetric
            let lambda = multiplier(sample);
            (0..3).flat_map(move |k| {
                let vtx_row = tri_indices[k];
                let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
                (0..3)
                    .filter(move |&l| tri_indices[l] <= vtx_row)
                    .map(move |l| {
                        let vtx_col = tri_indices[l];
                        // TODO: The following matrix probably has a simpler form (possible
                        // diagonal?) Rewrite in terms of this form.
                        let mtx = tri.area_normal_gradient(k) * nml_proj - nml_proj * tri.area_normal_gradient(l);
                        (vtx_row, vtx_col, mtx * (lambda * third))
                    })
            })
        })
    }

    /// Compute the background potential field. This function returns a struct that provides some
    /// useful quanitities for computing derivatives of the field.
    pub(crate) fn compute_background_potential<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        bg_type: BackgroundFieldType,
    ) -> BackgroundField<'a, T, T, K>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        // Construct a background field for computing derivative contribution.
        BackgroundField::new(
            q,
            samples,
            kernel,
            BackgroundFieldValue::jac(bg_type),
        )
    }

    fn compute_hrbf(
        query_points: &[[T; 3]],
        samples: &Samples<T>,
        out_potential: &mut [T],
    ) -> Result<(), super::Error> {
        let Samples {
            ref points,
            ref normals,
            ref values,
            ..
        } = samples;

        let pts: Vec<na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = p.cast::<f64>().unwrap().into();
                na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<na::Vector3<f64>> = normals
            .iter()
            .map(|&n| {
                let nml: [f64; 3] = n.cast::<f64>().unwrap().into();
                na::Vector3::from(nml).normalize()
            })
            .collect();

        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());

        let hrbf_values: Vec<f64> = values.iter().map(|&x| x.to_f64().unwrap()).collect();

        hrbf.fit_offset(&pts, &hrbf_values, &nmls);

        query_points
            .par_iter()
            .zip(out_potential.par_iter_mut())
            .for_each(|(q, potential)| {
                let pos: [f64; 3] = Vector3(*q).cast::<f64>().unwrap().into();
                *potential = T::from(hrbf.eval(na::Point3::from(pos))).unwrap();
            });

        Ok(())
    }

    fn compute_hrbf_on_mesh<F, M>(
        mesh: &mut M,
        samples: &Samples<T>,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
    {
        // Move the potential attrib out of the mesh. We will reinsert it after we are done.
        let potential_attrib = mesh
            .remove_attrib::<VertexIndex>("potential")
            .ok() // convert to option (None when it doesn't exist)
            .unwrap_or_else(|| Attribute::from_vec(vec![0.0f32; mesh.num_vertices()]));

        let mut potential = potential_attrib.into_buffer().cast_into_vec::<f32>();
        if potential.is_empty() {
            // Couldn't cast, which means potential is of some non-numeric type.
            // We overwrite it because we need that attribute spot.
            potential = vec![0.0f32; mesh.num_vertices()];
        }

        let Samples {
            ref points,
            ref normals,
            ref values,
            ..
        } = samples;
        let sample_pos = mesh.vertex_positions().to_vec();

        let pts: Vec<na::Point3<f64>> = points
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = p.cast::<f64>().unwrap().into();
                na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<na::Vector3<f64>> = normals
            .iter()
            .map(|&n| {
                let nml: [f64; 3] = n.cast::<f64>().unwrap().into();
                na::Vector3::from(nml).normalize()
            })
            .collect();
        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());

        let hrbf_values: Vec<f64> = values.iter().map(|&x| x.to_f64().unwrap()).collect();
        hrbf.fit_offset(&pts, &hrbf_values, &nmls);

        for (q_chunk, potential_chunk) in sample_pos
            .chunks(Self::PARALLEL_CHUNK_SIZE)
            .zip(potential.chunks_mut(Self::PARALLEL_CHUNK_SIZE))
        {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            q_chunk
                .par_iter()
                .zip(potential_chunk.par_iter_mut())
                .for_each(|(q, potential)| {
                    let pos: [f64; 3] = Vector3(*q).cast::<f64>().unwrap().into();
                    *potential = hrbf.eval(na::Point3::from(pos)) as f32;
                });
        }

        mesh.set_attrib_data::<_, VertexIndex>("potential", &potential)?;

        Ok(())
    }
}

/// Generate a tetrahedron with vertex positions and indices for the triangle faces.
#[cfg(test)]
pub(crate) fn make_tet() -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    use geo::mesh::TriMesh;
    use utils::*;
    let tet = make_regular_tet();
    let TriMesh {
        vertex_positions,
        indices,
        ..
    } = TriMesh::from(tet);
    let tet_verts = vertex_positions.into_iter().map(|x| Vector3(x)).collect();
    let tet_faces = reinterpret::reinterpret_vec(indices.into_vec());

    (tet_verts, tet_faces)
}

/// Generate a random vector of `Vector3` multipliers.
#[cfg(test)]
pub(crate) fn random_vectors(n: usize) -> Vec<Vector3<f64>> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n)
        .map(move |_| Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]))
        .collect()
}

#[cfg(test)]
mod tests {
    use geo::mesh::*;

    /// Test the projection.
    #[test]
    fn projection_test() -> Result<(), crate::Error> {
        use crate::*;

        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);

        // Translate the mesh slightly in z.
        utils::translate(&mut sphere, [0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = surface_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.5,
                },
                background_field: BackgroundFieldType::DistanceBased,
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
        )?;

        // Make a mesh to be projected.
        let mut grid = make_grid(22, 22);

        // Get grid node positions to be projected.
        let pos = grid.vertex_positions_mut();

        // Compute potential before projection.
        let mut init_potential = vec![0.0; pos.len()];
        surface.potential(pos, &mut init_potential)?;

        // Project grid outside the implicit surface.
        assert!(surface.project_to_above(0.0, 1e-4, pos)?);

        // Compute potential after projection.
        let mut final_potential = vec![0.0; pos.len()];
        surface.potential(pos, &mut final_potential)?;

        for (&old, &new) in init_potential.iter().zip(final_potential.iter()) {
            // Check that all vertices are outside the implicit solid.
            assert!(new >= 0.0, "old = {}", old);
            if old < 0.0 {
                // Check that the projected vertices are now within the narrow band of valid
                // projections (between 0 and 1e-4).
                assert!(new <= 1e-4, "new = {}", new);
            }
        }

        Ok(())
    }

}
