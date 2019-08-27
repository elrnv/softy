//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::kernel::{KernelType, SphericalKernel};
use geo::math::{Matrix3, Vector3};
use geo::mesh::{attrib::*, topology::VertexIndex, VertexMesh};
use geo::prim::Triangle;
use geo::Real;
use num_traits::cast;
use rayon::prelude::*;
use rstar::RTree;
use serde::{Deserialize, Serialize};
use std::cell::{Ref, RefCell};
use utils::zip;

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

pub(crate) use self::background_field::*;
pub use self::background_field::{BackgroundFieldParams, BackgroundFieldType};
pub(crate) use self::neighbour_cache::Neighbourhood;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SampleType {
    Vertex,
    Face,
}

/// Side of the implicit field. This is used to indicate a side of the implicit field with respect
/// to some iso-value, where `Above` refers to the potential above the iso-value and `Below` refers
/// to the potential below a certain iso-value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side {
    Above,
    Below,
}

/// Implicit surface type. `V` is the value type of the implicit function. Note that if `V` is a
/// vector, this type will fit a vector field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplicitSurface<T = f64>
where
    T: Real,
{
    /// The type of kernel to use for fitting the data.
    kernel: KernelType,

    /// The base radius or local feature size used to
    base_radius: f64,

    /// Enum for choosing how to compute a background potential field that may be mixed in with
    /// the local potentials.
    bg_field_params: BackgroundFieldParams,

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

    /// Radius of influence ( kernel radius ) for this implicit surface.
    pub fn radius(&self) -> f64 {
        self.base_radius
            * match self.kernel {
                KernelType::Interpolating { radius_multiplier }
                | KernelType::Approximate {
                    radius_multiplier, ..
                }
                | KernelType::Cubic { radius_multiplier } => radius_multiplier,
                KernelType::Global { .. } | KernelType::Hrbf => std::f64::INFINITY,
            }
    }

    /// Return the surface vertex positions used by this implicit surface.
    pub fn surface_vertex_positions(&self) -> &[Vector3<T>] {
        &self.surface_vertex_positions
    }

    /// Return the surface topology used by this implicit surface.
    pub fn surface_topology(&self) -> &[[usize; 3]] {
        &self.surface_topo
    }

    /// Return the number of samples used by this implicit surface.
    pub fn samples(&self) -> &Samples<T> {
        &self.samples
    }

    /// Return the number of samples used by this implicit surface.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
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

                // Compute unnormalized area weighted vertex normals given a triangle topology.
                geo::algo::compute_vertex_area_weighted_normals(points, surface_topo, normals);
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

    pub fn nearest_neighbour_lookup(&self, q: [T; 3]) -> Option<&Sample<T>> {
        let q_pos = Vector3(q).cast::<f64>().unwrap().into();
        self.spatial_tree.nearest_neighbor(&q_pos)
    }

    /// Compute neighbour cache if it has been invalidated. Return true if neighbour cache has
    /// changed.
    pub fn cache_neighbours(&self, query_points: &[[T; 3]]) -> bool {
        let ImplicitSurface {
            ref kernel,
            base_radius,
            ref spatial_tree,
            ref samples,
            ref surface_topo,
            ref dual_topo,
            max_step,
            sample_type,
            ..
        } = *self;

        match *kernel {
            KernelType::Interpolating { radius_multiplier }
            | KernelType::Approximate {
                radius_multiplier, ..
            }
            | KernelType::Cubic { radius_multiplier } => {
                let radius = base_radius * radius_multiplier;
                let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    let q_pos = Vector3(q).cast::<f64>().unwrap().into();
                    spatial_tree
                        .locate_within_distance(q_pos, radius2)
                        .into_iter()
                        .cloned()
                };
                let mut cache = self.query_neighbourhood.borrow_mut();
                cache.compute_neighbourhoods(
                    query_points,
                    neigh,
                    |q| {
                        spatial_tree
                            .nearest_neighbor(&Vector3(q).cast::<f64>().unwrap().into())
                            .expect("Empty spatial tree")
                    },
                    surface_topo,
                    dual_topo,
                    sample_type,
                )
            }
            KernelType::Global { .. } | KernelType::Hrbf => {
                // Global kernel, all points are neighbours
                let neigh = |_| samples.iter();
                let mut cache = self.query_neighbourhood.borrow_mut();
                cache.compute_neighbourhoods(
                    query_points,
                    neigh,
                    |q| {
                        spatial_tree
                            .nearest_neighbor(&Vector3(q).cast::<f64>().unwrap().into())
                            .expect("Empty spatial tree")
                    },
                    surface_topo,
                    dual_topo,
                    sample_type,
                )
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

    /// This function returns previously cached closest samples if the neighbour cache is
    /// valid, and `None` otherwise.
    fn closest_samples_borrow(&self) -> Result<Ref<'_, [usize]>, super::Error> {
        // Note there is no RefMut -> Ref map as of this writing, which is why recomputing the
        // cache and actually retrieving the neighbour points is done separately
        let valid = self.query_neighbourhood.borrow().closest_set().is_some();
        if valid {
            Ok(Ref::map(self.query_neighbourhood.borrow(), |c| {
                c.closest_set().unwrap()
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

    /// The number of query points in the cache (regardless if their neighbourhood is empty).
    /// This function returns `None` if the cache is invalid.
    pub fn num_cached_query_points(&self) -> Result<usize, super::Error> {
        self.trivial_neighbourhood_borrow()
            .map(|neighbourhoods| neighbourhoods.len())
    }

    /// The number of query points with non-empty neighbourhoods in the cache.
    /// This function returns `None` if the cache is invalid.
    pub fn num_cached_neighbourhoods(&self) -> Result<usize, super::Error> {
        self.trivial_neighbourhood_borrow()
            .map(|neighbourhoods| neighbourhoods.iter().filter(|x| !x.is_empty()).count())
    }

    /// Return a vector of indices for query points with non-empty neighbourhoods.
    pub fn nonempty_neighbourhood_indices(&self) -> Result<Vec<usize>, super::Error> {
        let set = match self.sample_type {
            SampleType::Vertex => self.extended_neighbourhood_borrow(),
            SampleType::Face => self.trivial_neighbourhood_borrow(),
        };

        set.map(|neighbourhoods| {
            neighbourhoods
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_empty())
                .map(|(i, _)| i)
                .collect()
        })
    }

    /// Return a vector over query points, giving the sizes of each cached neighbourhood.
    pub fn cached_neighbourhood_sizes(&self) -> Result<Vec<usize>, super::Error> {
        let set = match self.sample_type {
            SampleType::Vertex => self.extended_neighbourhood_borrow(),
            SampleType::Face => self.trivial_neighbourhood_borrow(),
        };

        set.map(|neighbourhoods| neighbourhoods.iter().map(std::vec::Vec::len).collect())
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        self.max_step = max_step;
    }

    pub fn update_radius_multiplier(&mut self, new_radius_multiplier: f64) {
        match self.kernel {
            KernelType::Interpolating {
                ref mut radius_multiplier,
            }
            | KernelType::Approximate {
                ref mut radius_multiplier,
                ..
            }
            | KernelType::Cubic {
                ref mut radius_multiplier,
            } => {
                *radius_multiplier = new_radius_multiplier;
            }
            _ => {}
        }
    }

    /// Project the given set of positions to be below the specified iso-value along the gradient.
    /// If the query point is already below the given iso-value, then it is not modified.
    /// The given `epsilon` determines how far below the iso-surface the point is allowed to be
    /// projected, essentially it is the thickness below the iso-surface of value projections.
    /// This function will return true if convergence is achieved and false if the projection needed
    /// more iterations.
    pub fn project_to_below(
        &self,
        iso_value: T,
        epsilon: T,
        query_points: &mut [[T; 3]],
    ) -> Result<bool, super::Error> {
        self.project(Side::Below, iso_value, epsilon, query_points)
    }

    /// Project the given set of positions to be above the specified iso-value along the gradient.
    /// If the query point is already above the given iso-value, then it is not modified.
    /// The given `epsilon` determines how far above the iso-surface the point is allowed to be
    /// projected, essentially it is the thickness above the iso-surface of value projections.
    /// This function will return true if convergence is achieved and false if the projection needed
    /// more iterations.
    pub fn project_to_above(
        &self,
        iso_value: T,
        epsilon: T,
        query_points: &mut [[T; 3]],
    ) -> Result<bool, super::Error> {
        self.project(Side::Above, iso_value, epsilon, query_points)
    }

    /// Project the given set of positions to be above (below) the specified iso-value along the
    /// gradient.  If the query point is already above (below) the given iso-value, then it is not
    /// modified.  The given `epsilon` determines how far above (below) the iso-surface the point
    /// is allowed to be projected, essentially it is the thickness above (below) the iso-surface
    /// of value projections.  This function will return true if convergence is achieved and false
    /// if the projection needed more iterations.
    pub fn project(
        &self,
        side: Side,
        iso_value: T,
        epsilon: T,
        query_points: &mut [[T; 3]],
    ) -> Result<bool, super::Error> {
        let multiplier = match side {
            Side::Above => T::one(),
            Side::Below => -T::one(),
        };
        let iso_value = iso_value * multiplier;

        let mut candidate_points = query_points.to_vec();
        let mut potential = vec![T::zero(); query_points.len()];
        let mut candidate_potential = vec![T::zero(); query_points.len()];
        let mut steps = vec![[T::zero(); 3]; query_points.len()];
        let mut nml_sizes = vec![T::zero(); query_points.len()];

        let max_steps = 20;
        let max_binary_search_iters = 10;

        let mut convergence = true;

        for i in 0..max_steps {
            self.potential(query_points, &mut potential)?;
            potential.iter_mut().for_each(|x| *x *= multiplier);

            // The transpose of the potential gradient at each of the query points.
            self.query_jacobian_full(query_points, &mut steps)?;

            for (norm, step) in nml_sizes.iter_mut().zip(steps.iter()) {
                *norm = Vector3(*step).norm();
            }

            // Count the number of points with values less than iso_value.
            let count_violations = potential
                .iter()
                .zip(nml_sizes.iter())
                .filter(|&(&x, &norm)| x < iso_value && norm != T::zero())
                .count();

            if count_violations == 0 {
                break;
            }

            // Compute initial step directions
            for (step, &norm, &value) in zip!(steps.iter_mut(), nml_sizes.iter(), potential.iter())
                .filter(|(_, &norm, &pot)| pot < iso_value && norm != T::zero())
            {
                let nml = Vector3(*step);
                let offset = (epsilon * T::from(0.5).unwrap() + (iso_value - value)) / norm;
                *step = (nml * (multiplier * offset)).into();
            }

            for j in 0..max_binary_search_iters {
                // Try this step
                for (p, q, &step, _, _) in zip!(
                    candidate_points.iter_mut(),
                    query_points.iter(),
                    steps.iter(),
                    nml_sizes.iter(),
                    potential.iter()
                )
                .filter(|(_, _, _, &norm, &pot)| pot < iso_value && norm != T::zero())
                {
                    *p = (Vector3(*q) + Vector3(step)).into();
                }

                self.potential(&candidate_points, &mut candidate_potential)?;
                candidate_potential
                    .iter_mut()
                    .for_each(|x| *x *= multiplier);

                let mut count_overshoots = 0;
                for (step, _, _, _) in zip!(
                    steps.iter_mut(),
                    nml_sizes.iter(),
                    potential.iter(),
                    candidate_potential.iter()
                )
                .filter(|(_, &norm, &old, &new)| {
                    old < iso_value && new > iso_value + epsilon && norm != T::zero()
                }) {
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

            // Since the query points have changed position, we invalidate the cache to force a
            // recomputation of neighbourhoods.
            self.invalidate_query_neighbourhood();

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
        debug_assert!(
            query_points.iter().all(|&q| q.iter().all(|&x| !x.is_nan())),
            "Detected NaNs in query points. Please report this bug."
        );

        let ImplicitSurface {
            kernel,
            base_radius,
            ref samples,
            ..
        } = *self;

        match_kernel_as_spherical!(
            kernel,
            base_radius,
            |kern| self.compute_mls(query_points, kern, out_field),
            || Self::compute_hrbf(query_points, samples, out_field)
        )
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
            bg_field_params,
            ..
        } = *self;

        zip!(
            query_points.par_iter(),
            neigh_points.par_iter(),
            out_field.par_iter_mut()
        )
        //.filter(|(_, nbrs, _)| !nbrs.is_empty())
        .for_each(move |(q, neighbours, field)| {
            Self::compute_potential_at(
                Vector3(*q),
                SamplesView::new(neighbours, samples),
                kernel,
                bg_field_params,
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
        bg_potential: BackgroundFieldParams,
        potential: &mut T,
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        if samples.is_empty() {
            return;
        }

        let bg =
            BackgroundField::local(q, samples, kernel, bg_potential, Some(*potential)).unwrap();

        let weight_sum_inv = bg.weight_sum_inv();

        // Generate a background potential field for every query point. This will be mixed
        // in with the computed potentials for local methods.
        *potential = bg.compute_unnormalized_weighted_scalar_field() * weight_sum_inv;

        let local_field = Self::compute_local_potential_at(
            q,
            samples,
            kernel,
            weight_sum_inv,
            bg.closest_sample_dist(),
        );

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
        samples
            .iter()
            .map(
                |Sample {
                     pos, nml, value, ..
                 }| {
                    let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                    let p = value + nml.dot(q - pos) / nml.norm();
                    w * p
                },
            )
            .sum::<T>()
            * weight_sum_inv
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
        match_kernel_as_spherical!(
            self.kernel,
            self.base_radius,
            |kern| self.compute_mls_vector_field(query_points, kern, out_vectors),
            || Err(super::Error::UnsupportedKernel)
        )
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
            bg_field_params,
            ..
        } = *self;

        zip!(
            query_points.par_iter(),
            neigh_points.par_iter(),
            out_vectors.par_iter_mut()
        )
        //.filter(|(_, nbrs, _)| !nbrs.is_empty())
        .for_each(move |(q, neighbours, vector)| {
            Self::compute_local_vector_at(
                Vector3(*q),
                SamplesView::new(neighbours, samples),
                kernel,
                bg_field_params,
                vector,
            );
        });

        Ok(())
    }

    pub(crate) fn compute_local_vector_at<K>(
        q: Vector3<T>,
        samples: SamplesView<T>,
        kernel: K,
        bg_potential: BackgroundFieldParams,
        vector: &mut [T; 3],
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        if samples.is_empty() {
            return;
        }

        let bg = BackgroundField::local(q, samples, kernel, bg_potential, Some(Vector3(*vector)))
            .unwrap();

        // Generate a background potential field for every query point. This will be mixed
        // in with the computed potentials for local methods.
        let mut out_field = bg.compute_unnormalized_weighted_vector_field();

        let closest_dist = bg.closest_sample_dist();

        let weight_sum_inv = bg.weight_sum_inv();

        let grad_phi = Self::query_jacobian_at(q, samples, None, kernel, bg_potential);

        for Sample { pos, vel, nml, .. } in samples.iter() {
            out_field += Self::sample_contact_jacobian_product_at(
                q,
                pos,
                nml,
                kernel,
                grad_phi,
                weight_sum_inv,
                closest_dist,
                vel,
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
            kernel,
            ref samples,
            ..
        } = *self;

        match_kernel_as_spherical!(
            kernel,
            self.base_radius,
            |kern| self.compute_mls_on_mesh(mesh, kern, interrupt),
            || Self::compute_hrbf_on_mesh(mesh, samples, interrupt)
        )
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls_on_mesh<K, F, M>(
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
            bg_field_params,
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
        let closest_points = self.closest_samples_borrow()?;

        // Initialize extra debug info.
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut bg_weight_attrib_data = vec![0f32; mesh.num_vertices()];
        let mut weight_sum_attrib_data = vec![0f32; mesh.num_vertices()];

        for (
            q_chunk,
            neigh,
            closest,
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
            closest_points.chunks(Self::PARALLEL_CHUNK_SIZE),
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
                closest.par_iter(),
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
                    closest,
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

                    let bg = BackgroundField::global(
                        q,
                        view,
                        *closest,
                        kernel,
                        bg_field_params,
                        Some(T::from(*potential).unwrap()),
                    );

                    let closest_d = bg.closest_sample_dist();
                    *bg_weight = bg.background_weight().to_f32().unwrap();
                    *weight_sum = bg.weight_sum.to_f32().unwrap();
                    let weight_sum_inv = bg.weight_sum_inv();

                    *potential = bg
                        .compute_unnormalized_weighted_scalar_field()
                        .to_f32()
                        .unwrap();

                    if !view.is_empty() {
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

                            // Compute vector interpolation
                            let grad_phi = Self::query_jacobian_at(
                                q,
                                view,
                                Some(*closest),
                                kernel,
                                bg_field_params,
                            );

                            let nml_dot_grad = nml.dot(grad_phi);
                            // Handle degenerate case when nml and grad are exactly opposing. In
                            // this case the solution is not unique, so we pick one.
                            let rot = if nml_dot_grad != -T::one() {
                                let u = nml.cross(grad_phi);
                                let ux = u.skew();
                                Matrix3::identity() + ux + (ux * ux) / (T::one() + nml_dot_grad)
                            } else {
                                // TODO: take a convenient unit vector u that is
                                // orthogonal to nml and compute the rotation as
                                //let ux = u.skew();
                                //Matrix3::identity() + (ux*ux) * 2
                                Matrix3::identity()
                            };

                            out_tangent += (rot * vel) * w_normalized;
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
                let tri = Triangle::from_indexed_slice(tri_indices, samples.all_points());
                let nml_grad = tri.area_normal_gradient(
                    tri_indices
                        .iter()
                        .position(|&j| j == index)
                        .expect("Triangle mesh topology corruption."),
                );
                let mut tri_grad = nml_proj * (dx(sample) * norm_inv);
                for sample in SamplesView::from_view(tri_indices, samples).into_iter() {
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
    //pub(crate) fn face_unit_normal_gradient(
    //    sample: Sample<T>,
    //    vtx_idx: usize,
    //    surface_vertices: &[Vector3<T>],
    //    surface_topo: &[[usize; 3]],
    //) -> Matrix3<T> {
    //    let nml_proj = Self::scaled_tangent_projection(sample);
    //    let tri_indices = &surface_topo[sample.index];
    //    let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
    //    tri.area_normal_gradient(vtx_idx) * nml_proj
    //}

    /// Compute the matrix for projecting on the tangent plane of the given sample inversely scaled
    /// by the local area (normal norm reciprocal).
    pub(crate) fn scaled_tangent_projection(sample: Sample<T>) -> Matrix3<T> {
        let nml_norm_inv = T::one() / sample.nml.norm();
        let nml = sample.nml * nml_norm_inv;
        Matrix3::diag([nml_norm_inv; 3]) - (nml * nml_norm_inv) * nml.transpose()
    }

    ///// Compute the background potential field. This function returns a struct that provides some
    ///// useful quanitities for computing derivatives of the field.
    //pub(crate) fn compute_background_potential<'a, K: 'a>(
    //    q: Vector3<T>,
    //    samples: SamplesView<'a, 'a, T>,
    //    closest: usize,
    //    kernel: K,
    //    bg_type: BackgroundFieldType,
    //) -> BackgroundField<'a, T, T, K>
    //where
    //    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    //{
    //    // Construct a background field for computing derivative contribution.
    //    BackgroundField::new(q, samples, closest, kernel, BackgroundFieldValue::jac(bg_type))
    //}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use geo::mesh::*;

    // Helper function for testing. This is an implicit surface and grid mesh pair where each
    // vertex of the grid mesh has a non-empty local neighbpourhood of the implicit surface.
    // The `reverse` option reverses each triangle in the sphere to create an inverted implicit
    // surface.
    fn make_octahedron_and_grid_local(
        reverse: bool,
    ) -> Result<(ImplicitSurface, PolyMesh<f64>), crate::Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);
        if reverse {
            sphere.reverse();
        }

        // Translate the mesh slightly in z.
        utils::translate(&mut sphere, [0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = surface_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier: 2.45,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    // Distance based background is discontinuous, this is bad for projection, so
                    // we always opt for an unweighted background to make sure that local
                    // potentials are always high quality
                    weighted: false,
                },
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
        )?;

        // Make a mesh to be projected.
        let grid = make_grid(23, 23);

        Ok((surface, grid))
    }

    // Helper function for testing. This is an implicit surface and grid mesh pair where each
    // vertex of the grid mesh has a non-empty local neighbpourhood of the implicit surface.
    // The `reverse` option reverses each triangle in the sphere to create an inverted implicit
    // surface.
    fn make_octahedron_and_grid(
        reverse: bool,
        radius_multiplier: f64,
    ) -> Result<(ImplicitSurface, PolyMesh<f64>), crate::Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);
        if reverse {
            sphere.reverse();
        }

        // Translate the mesh slightly in z.
        utils::translate(&mut sphere, [0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = surface_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: false,
                },
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
        )?;

        // Make a mesh to be projected.
        let mut grid = make_grid(22, 22);

        utils::uniform_scale(&mut grid, 2.0);

        Ok((surface, grid))
    }

    fn projection_tester(
        surface: &ImplicitSurface,
        mut grid: PolyMesh<f64>,
        side: Side,
    ) -> Result<(), crate::Error> {
        let epsilon = 1e-4;
        let init_potential = {
            // Get grid node positions to be projected.
            let pos = grid.vertex_positions_mut();

            // Compute potential before projection.
            let mut init_potential = vec![0.0; pos.len()];
            surface.potential(pos, &mut init_potential)?;

            // Project grid outside the implicit surface.
            assert!(surface.project(side, 0.0, epsilon, pos)?);
            init_potential
        };

        // Compute potential after projection.
        let mut final_potential = vec![0.0; init_potential.len()];
        surface.potential(grid.vertex_positions(), &mut final_potential)?;

        //use geo::mesh::topology::VertexIndex;
        //grid.set_attrib_data::<_, VertexIndex>("init_potential", &init_potential);
        //grid.set_attrib_data::<_, VertexIndex>("final_potential", &final_potential);
        //geo::io::save_polymesh(&grid, &std::path::PathBuf::from("out/mesh.vtk"))?;

        for (&old, &new) in init_potential.iter().zip(final_potential.iter()) {
            // Check that all vertices are outside the implicit solid.
            match side {
                Side::Above => {
                    assert!(new >= 0.0, "new = {}, old = {}", new, old);
                    if old < 0.0 {
                        // Check that the projected vertices are now within the narrow band of valid
                        // projections (between 0 and epsilon).
                        assert!(new <= epsilon, "new = {}", new);
                    }
                }
                Side::Below => {
                    assert!(new <= 0.0, "new = {}, old = {}", new, old);
                    if old > 0.0 {
                        // Check that the projected vertices are now within the narrow band of valid
                        // projections (between 0 and epsilon).
                        assert!(new >= -epsilon, "new = {}", new);
                    }
                }
            }
        }

        Ok(())
    }

    /// Test projection where each projected vertex has a non-empty local neighbourhood of the
    /// implicit surface.
    #[test]
    fn local_projection_test() -> Result<(), crate::Error> {
        let (surface, grid) = make_octahedron_and_grid_local(false)?;
        projection_tester(&surface, grid, Side::Above)?;
        let (surface, grid) = make_octahedron_and_grid_local(true)?;
        projection_tester(&surface, grid, Side::Below)
    }

    /// Test projection where some projected vertices may not have a local neighbourhood at all.
    /// This is a more complex test than the local_projection_test
    #[test]
    fn global_projection_test() -> Result<(), crate::Error> {
        let (surface, grid) = make_octahedron_and_grid(false, 2.45)?;
        projection_tester(&surface, grid, Side::Above)?;
        let (surface, grid) = make_octahedron_and_grid(true, 2.45)?;
        projection_tester(&surface, grid, Side::Below)
    }

    /// Test with a radius multiplier less than 1.0. Although not strictly useful, this should not
    /// crash.
    #[test]
    fn narrow_projection_test() -> Result<(), crate::Error> {
        // Make a mesh to be projected.
        use geo::mesh::attrib::*;
        use geo::mesh::topology::*;

        let mut grid = utils::make_grid(utils::Grid {
            rows: 18,
            cols: 19,
            orientation: utils::AxisPlaneOrientation::ZX,
        });
        grid.add_attrib::<_, VertexIndex>("potential", 0.0f32)
            .unwrap();

        grid.reverse();

        utils::translate(&mut grid, [0.0, 0.12639757990837097, 0.0]);

        let torus = geo::io::load_polymesh("assets/projection_torus.vtk")?;

        // Construct the implicit surface.
        let surface = surface_from_polymesh(
            &torus,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.000009999999747378752,
                    radius_multiplier: 0.7599999904632568,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: false,
                },
                sample_type: SampleType::Face,
                ..Default::default()
            },
        )?;

        projection_tester(&surface, grid, Side::Above)?;

        Ok(())
    }

    /// This struct helps deserialize testing assets without having to store an rtree.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct ImplicitSurfaceNoTree {
        kernel: KernelType,
        base_radius: f64,
        bg_field_params: BackgroundFieldParams,
        surface_topo: Vec<[usize; 3]>,
        surface_vertex_positions: Vec<Vector3<f64>>,
        samples: Samples<f64>,
        max_step: f64,
        query_neighbourhood: RefCell<Neighbourhood>,
        dual_topo: Vec<Vec<usize>>,
        sample_type: SampleType,
    }

    /// Test a specific case where the projection direction can be zero, which could result in
    /// NaNs. This case must not crash.
    #[test]
    fn zero_step_projection_test() -> Result<(), crate::Error> {
        use std::io::Read;
        let iso_value = 0.0;
        let epsilon = 0.0001;
        let mut query_points: Vec<[f64; 3]> = {
            let mut file = std::fs::File::open("assets/grid_points.json")
                .expect("Failed to open query points file");
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .expect("Failed to read grid points json.");
            serde_json::from_str(&contents).expect("Failed to deserialize grid points.")
        };

        let surface: ImplicitSurface<f64> = {
            let mut file = std::fs::File::open("assets/torus_surf_no_tree.json")
                .expect("Failed to open torus surface file");
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .expect("Failed to read torus surface json.");
            let ImplicitSurfaceNoTree {
                kernel,
                base_radius,
                bg_field_params,
                surface_topo,
                surface_vertex_positions,
                samples,
                max_step,
                query_neighbourhood,
                dual_topo,
                sample_type,
            } = serde_json::from_str(&contents).expect("Failed to deserialize torus surface.");
            ImplicitSurface {
                kernel,
                base_radius,
                bg_field_params,
                spatial_tree: build_rtree_from_samples(&samples),
                surface_topo,
                surface_vertex_positions,
                samples,
                max_step,
                query_neighbourhood,
                dual_topo,
                sample_type,
            }
        };

        let init_potential = {
            // Compute potential before projection.
            let mut init_potential = vec![0.0; query_points.len()];
            surface.potential(&query_points, &mut init_potential)?;
            init_potential
        };

        // Project grid outside the implicit surface.
        assert!(surface.project_to_above(iso_value, epsilon, &mut query_points)?);

        // Compute potential after projection.
        let mut final_potential = vec![0.0; init_potential.len()];
        surface.potential(&query_points, &mut final_potential)?;

        for (i, (&old, &new)) in init_potential
            .iter()
            .zip(final_potential.iter())
            .enumerate()
        {
            // Check that all vertices are outside the implicit solid.
            assert!(new >= 0.0, "new = {}, old = {}, i = {}", new, old, i);
            if old < 0.0 {
                // Check that the projected vertices are now within the narrow band of valid
                // projections (between 0 and epsilon).
                assert!(new <= epsilon, "new = {}", new);
            }
        }

        Ok(())
    }

    #[test]
    fn cached_neighbourhoods() -> Result<(), crate::Error> {
        // Local test
        let (surface, grid) = make_octahedron_and_grid_local(false)?;
        surface.cache_neighbours(grid.vertex_positions());
        assert_eq!(
            surface.num_cached_neighbourhoods()?,
            surface.nonempty_neighbourhood_indices()?.len()
        );

        // Non-local test
        let (surface, grid) = make_octahedron_and_grid(false, 2.45)?;
        surface.cache_neighbours(grid.vertex_positions());
        assert_eq!(
            surface.num_cached_neighbourhoods()?,
            surface.nonempty_neighbourhood_indices()?.len()
        );
        Ok(())
    }

}
