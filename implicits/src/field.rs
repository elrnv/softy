//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::kernel::*;
use geo::math::{Matrix3, Vector3};
use geo::mesh::{attrib::*, topology::VertexIndex, VertexMesh};
use geo::prim::Triangle;
pub use geo::Real;
use num_traits::cast;
use rayon::{iter::Either, prelude::*};
use rstar::RTree;
use serde::{Deserialize, Serialize};
use utils::zip;

macro_rules! apply_kernel_query_fn {
    ($surf:expr, $f:expr) => {
        match *$surf {
            QueryTopo::Local {
                surf:
                    LocalMLS {
                        kernel,
                        base_radius,
                        ..
                    },
                ..
            } => apply_as_spherical!(kernel, base_radius, $f),
            QueryTopo::Global {
                surf: GlobalMLS { kernel, .. },
                ..
            } => apply_as_spherical!(kernel, $f),
        }
    };
}

macro_rules! apply_kernel_fn {
    ($surf:expr, $f:expr) => {
        match *$surf {
            MLS::Local(LocalMLS {
                kernel,
                base_radius,
                ..
            }) => apply_as_spherical!(kernel, base_radius, $f),
            MLS::Global(GlobalMLS { kernel, .. }) => apply_as_spherical!(kernel, $f),
        }
    };
}

pub mod background_field;
pub mod builder;
pub mod hessian;
pub mod jacobian;
pub mod neighbour_cache;
pub mod query;
pub mod samples;
pub mod spatial_tree;

pub use self::builder::*;
pub use self::query::*;
pub use self::samples::*;
pub use self::spatial_tree::*;

pub(crate) use self::background_field::*;
pub use self::background_field::{BackgroundFieldParams, BackgroundFieldType};
pub(crate) use self::neighbour_cache::Neighbourhood;

const PARALLEL_CHUNK_SIZE: usize = 5000;

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

/// Implicit surface type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplicitSurfaceBase<T = f64>
where
    T: Real,
{
    /// Enum for choosing how to compute a background potential field that may be mixed in with
    /// the local potentials.
    bg_field_params: BackgroundFieldParams,

    /// Surface triangles representing the surface discretization to be approximated.
    /// This topology also defines the normals to the surface.
    surface_topo: Vec<[usize; 3]>,

    /// Save the vertex positions of the mesh because the samples may not coincide (e.g. face
    /// centered samples).
    surface_vertex_positions: Vec<Vector3<T>>,

    /// Sample points defining the entire implicit surface.
    samples: Samples<T>,

    /// Vertex neighbourhood topology. For each vertex, this vector stores all the indices to
    /// adjacent triangles.
    dual_topo: Vec<Vec<usize>>,

    /// The type of implicit surface. For example should the samples be centered at vertices or
    /// face centroids.
    sample_type: SampleType,

    /// Local search tree for fast proximity queries.
    spatial_tree: RTree<Sample<T>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MLS<T = f64>
where
    T: Real,
{
    Local(LocalMLS<T>),
    Global(GlobalMLS<T>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalMLS<T = f64>
where
    T: Real,
{
    kernel: LocalKernel,

    base_radius: f64,

    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern. If this is set too large, the derivative may be denser
    /// than then needed, which typically results in slower performance.  If it is set too low,
    /// there may be errors in the derivative. It is the callers responsibility to set this step
    /// accurately using `update_max_step`. If the implicit surface is not changing, leave this at
    /// 0.0.
    max_step: T,

    /// Store the neighbouring sample points for each query point we see.
    query_neighbourhood: Neighbourhood,

    surf_base: ImplicitSurfaceBase<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalMLS<T = f64>
where
    T: Real,
{
    kernel: GlobalKernel,
    /// An array of sample indices `0..#samples`. This is here to make neighbour api compatible
    /// with local MLS.
    sample_indices: Vec<usize>,
    /// Closest sample indices.
    closest_samples: Vec<usize>,
    surf_base: ImplicitSurfaceBase<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HrbfSurface<T = f64>
where
    T: Real,
{
    surf_base: ImplicitSurfaceBase<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImplicitSurface<T = f64>
where
    T: Real,
{
    Hrbf(HrbfSurface<T>),
    MLS(MLS<T>),
}

impl<T: Real + Send + Sync> ImplicitSurfaceBase<T> {
    /// Update the stored samples. This assumes that vertex positions have been updated.
    fn update_samples(&mut self) {
        let ImplicitSurfaceBase {
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

        // Finally update the rtree responsible for neighbour search.
        self.spatial_tree = build_rtree_from_samples(&self.samples);

        num_updated
    }
}

impl<T: Real + Send + Sync> MLS<T> {
    fn base(&self) -> &ImplicitSurfaceBase<T> {
        match self {
            MLS::Local(LocalMLS { surf_base, .. }) | MLS::Global(GlobalMLS { surf_base, .. }) => {
                surf_base
            }
        }
    }

    fn base_mut(&mut self) -> &mut ImplicitSurfaceBase<T> {
        match self {
            MLS::Local(LocalMLS { surf_base, .. }) | MLS::Global(GlobalMLS { surf_base, .. }) => {
                surf_base
            }
        }
    }

    /// Radius of influence ( kernel radius ) for this implicit surface.
    pub fn radius(&self) -> f64 {
        match self {
            MLS::Local(local) => local.kernel.radius_multiplier() * local.base_radius,
            MLS::Global(_) => std::f64::INFINITY,
        }
    }

    /// Return the surface vertex positions used by this implicit surface.
    pub fn surface_vertex_positions(&self) -> &[Vector3<T>] {
        &self.base().surface_vertex_positions
    }

    /// Return the surface topology used by this implicit surface.
    pub fn surface_topology(&self) -> &[[usize; 3]] {
        &self.base().surface_topo
    }

    /// Return the number of samples used by this implicit surface.
    pub fn samples(&self) -> &Samples<T> {
        &self.base().samples
    }

    /// Return the number of samples used by this implicit surface.
    pub fn num_samples(&self) -> usize {
        self.base().samples.len()
    }

    /// Build a query topology type to be able to ask questions about the surface potential and
    /// derivatives at a set of query points.
    ///
    /// This type contains the necessary neighbourhood information to make queries fast.
    pub fn query_topo<Q>(self, query_points: Q) -> QueryTopo<T>
    where
        Q: AsRef<[[T; 3]]>,
    {
        QueryTopo::new(query_points.as_ref(), self)
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
        self.base_mut().update(vertex_iter)
    }

    /// The number of query points in the cache (regardless if their neighbourhood is empty).
    /// This function returns `None` if the cache is invalid.
    pub fn num_query_points(&self) -> Result<usize, super::Error> {
        self.trivial_neighbourhood_seq().map(|neigh| neigh.len())
    }

    /// The number of query points with non-empty neighbourhoods.
    /// This function returns `None` if the cache is invalid.
    pub fn num_neighbourhoods(&self) -> Result<usize, super::Error> {
        self.trivial_neighbourhood_seq()
            .map(|neigh| neigh.filter(|x| !x.is_empty()).count())
    }

    /// Return a vector of indices for query points with non-empty neighbourhoods.
    pub fn nonempty_neighbourhood_indices(&self) -> Result<Vec<usize>, super::Error> {
        match self.base().sample_type {
            SampleType::Vertex => self
                .extended_neighbourhood_seq()
                .map(Self::nonempty_neighbourhood_indices_impl),
            SampleType::Face => self
                .trivial_neighbourhood_seq()
                .map(Self::nonempty_neighbourhood_indices_impl),
        }
    }

    /// Return a vector over query points, giving the sizes of each neighbourhood.
    pub fn neighbourhood_sizes(&self) -> Result<Vec<usize>, super::Error> {
        match self.base().sample_type {
            SampleType::Vertex => self
                .extended_neighbourhood_seq()
                .map(|neighbourhoods| neighbourhoods.map(|x| x.len()).collect()),
            SampleType::Face => self
                .trivial_neighbourhood_seq()
                .map(|neighbourhoods| neighbourhoods.map(|x| x.len()).collect()),
        }
    }

    pub fn num_neighbours_within_distance<Q: Into<[T; 3]>>(&self, q: Q, radius: f64) -> usize {
        let q_pos = Vector3(q.into()).cast::<f64>().unwrap().into();
        self.base()
            .spatial_tree
            .locate_within_distance(q_pos, radius * radius)
            .count()
    }

    pub fn nearest_neighbour_lookup(&self, q: [T; 3]) -> Option<&Sample<T>> {
        let q_pos = Vector3(q).cast::<f64>().unwrap().into();
        self.base()
            .spatial_tree
            .nearest_neighbor_iter(&q_pos)
            .next()
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        if let MLS::Local(local) = self {
            local.max_step = max_step;
        }
    }

    pub fn update_radius_multiplier(&mut self, new_radius_multiplier: f64) {
        match self {
            MLS::Local(LocalMLS { kernel, .. }) => {
                *kernel = kernel.with_radius_multiplier(new_radius_multiplier);
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

            if i == max_steps - 1 {
                convergence = false;
            }
        }

        Ok(convergence)
    }

    /// This function returns precomputed neighbours.
    pub fn trivial_neighbourhood_par<'a>(
        &'a self,
    ) -> Result<
        impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a,
        super::Error,
    > {
        Ok(match self {
            MLS::Local(local) => Either::Left(
                local
                    .query_neighbourhood
                    .trivial_set()
                    .par_iter()
                    .map(|x| x.as_slice()),
            ),
            MLS::Global(global) => Either::Right(global.trivial_neighbourhood_par()),
        })
    }

    /// This function returns precomputed closest samples.
    pub fn closest_samples_par<'a>(
        &'a self,
    ) -> Result<impl ParallelIterator<Item = usize> + IndexedParallelIterator + 'a, super::Error>
    {
        Ok(match self {
            MLS::Local(local) => {
                Either::Left(local.query_neighbourhood.closest_set().par_iter().cloned())
            }
            MLS::Global(global) => Either::Right(global.closest_samples()?.par_iter().cloned()),
        })
    }

    /// This function returns precomputed neighbours.
    pub fn extended_neighbourhood_par<'a>(
        &'a self,
    ) -> Result<
        impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a,
        super::Error,
    > {
        Ok(match self {
            MLS::Local(local) => Either::Left(
                local
                    .query_neighbourhood
                    .extended_set()
                    .par_iter()
                    .map(|x| x.as_slice()),
            ),
            MLS::Global(global) => Either::Right(global.extended_neighbourhood_par()),
        })
    }

    /// This function returns precomputed neighbours.
    pub fn trivial_neighbourhood_par_chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> Result<
        impl ParallelIterator<Item = Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a>>
            + IndexedParallelIterator
            + 'a,
        super::Error,
    > {
        Ok(match self {
            MLS::Local(local) => Either::Left(
                local
                    .query_neighbourhood
                    .trivial_set()
                    .as_parallel_slice()
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let out: Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a> =
                            Box::new(chunk.iter().map(|nbrs| nbrs.as_slice()));
                        out
                    }),
            ),
            MLS::Global(global) => {
                Either::Right(global.trivial_neighbourhood_par_chunks(chunk_size))
            }
        })
    }

    /// This function returns precomputed closest samples.
    pub fn closest_samples_par_chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> Result<
        impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a,
        super::Error,
    > {
        Ok(match self {
            MLS::Local(local) => Either::Left(
                local
                    .query_neighbourhood
                    .closest_set()
                    .as_parallel_slice()
                    .par_chunks(chunk_size),
            ),
            MLS::Global(global) => Either::Right(
                global
                    .closest_samples()?
                    .as_parallel_slice()
                    .par_chunks(chunk_size),
            ),
        })
    }

    /// This function returns precomputed neighbours.
    pub fn trivial_neighbourhood_seq<'a>(
        &'a self,
    ) -> Result<impl Iterator<Item = &'a [usize]> + ExactSizeIterator + Clone + 'a, super::Error>
    {
        Ok(match self {
            MLS::Local(local) => Either::Left(
                local
                    .query_neighbourhood
                    .trivial_set()
                    .iter()
                    .map(|x| x.as_slice()),
            ),
            MLS::Global(global) => Either::Right(global.trivial_neighbourhood_seq()),
        })
    }

    /// This function returns precomputed closest samples.
    pub fn closest_samples_seq<'a>(
        &'a self,
    ) -> Result<impl Iterator<Item = usize> + 'a, super::Error> {
        Ok(match self {
            MLS::Local(local) => {
                Either::Left(local.query_neighbourhood.closest_set().iter().cloned())
            }
            MLS::Global(global) => Either::Right(global.closest_samples()?.iter().cloned()),
        })
    }

    /// This function returns precomputed neighbours.
    pub fn extended_neighbourhood_seq<'a>(
        &'a self,
    ) -> Result<impl Iterator<Item = &'a [usize]> + 'a, super::Error> {
        Ok(match self {
            MLS::Local(local) => Either::Left(
                local
                    .query_neighbourhood
                    .extended_set()
                    .iter()
                    .map(|x| x.as_slice()),
            ),
            MLS::Global(global) => Either::Right(global.extended_neighbourhood_seq()),
        })
    }
    /// Compute neighbour storage. Return true if neighbours have changed.
    pub fn compute_neighbours(&mut self, query_points: &[[T; 3]]) -> bool {
        match *self {
            MLS::Local(LocalMLS {
                base_radius,
                kernel,
                max_step,
                ref mut query_neighbourhood,
                surf_base:
                    ImplicitSurfaceBase {
                        ref spatial_tree,
                        ref surface_topo,
                        ref dual_topo,
                        sample_type,
                        ..
                    },
                ..
            }) => {
                let radius = base_radius * kernel.radius_multiplier(); // TODO: refactor with self.radius() fn
                let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    let q_pos = Vector3(q).cast::<f64>().unwrap().into();
                    spatial_tree.locate_within_distance(q_pos, radius2).cloned()
                };
                query_neighbourhood.compute_neighbourhoods(
                    query_points,
                    neigh,
                    |q| {
                        spatial_tree
                            .nearest_neighbor_iter(&Vector3(q).cast::<f64>().unwrap().into())
                            .next()
                            .expect("Empty spatial tree")
                    },
                    surface_topo,
                    dual_topo,
                    sample_type,
                )
            }
            MLS::Global(GlobalMLS {
                ref mut closest_samples,
                surf_base:
                    ImplicitSurfaceBase {
                        ref spatial_tree, ..
                    },
                ..
            }) => {
                // Recompute closest samples using the spatial tree
                neighbour_cache::compute_closest_set(
                    query_points,
                    |q| {
                        spatial_tree
                            .nearest_neighbor(&Vector3(q).cast::<f64>().unwrap().into())
                            .expect("Empty spatial tree")
                    },
                    closest_samples,
                );
                true
            }
        }
    }

    /// Given a set of neighbourhoods, return the indices of the non-empty ones.
    pub fn nonempty_neighbourhood_indices_impl<'a>(
        neighbourhoods: impl Iterator<Item = &'a [usize]>,
    ) -> Vec<usize> {
        neighbourhoods
            .enumerate()
            .filter(|(_, x)| !x.is_empty())
            .map(|(i, _)| i)
            .collect()
    }

    /*
     * Main potential computation
     */

    /// Compute the ml potential.
    pub fn potential(
        &self,
        query_points: &[[T; 3]],
        out_field: &mut [T],
    ) -> Result<(), super::Error> {
        debug_assert!(
            query_points.iter().all(|&q| q.iter().all(|&x| !x.is_nan())),
            "Detected NaNs in query points. Please report this bug."
        );

        apply_kernel_fn!(self, |kernel| self.compute_potential(
            query_points,
            kernel,
            out_field
        ))
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_potential<'a, K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        out_field: &'a mut [T],
    ) -> Result<(), super::Error>
    where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq()?;

        assert_eq!(neigh_points.len(), out_field.len());

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *self.base();

        zip!(query_points.iter(), neigh_points, out_field.iter_mut())
            //.filter(|(_, nbrs, _)| !nbrs.is_empty())
            .for_each(move |(q, neighbours, field)| {
                compute_potential_at(
                    Vector3(*q),
                    SamplesView::new(neighbours, samples),
                    kernel,
                    bg_field_params,
                    field,
                );
            });

        Ok(())
    }

    /*
     * Vector field computation
     */

    /// Interpolate the given vector field at the given query points.
    fn compute_vector_field<'a, K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        out_vectors: &'a mut [[T; 3]],
    ) -> Result<(), super::Error>
    where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_par()?;

        assert_eq!(neigh_points.len(), out_vectors.len());

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *self.base();

        zip!(
            query_points.par_iter(),
            neigh_points,
            out_vectors.par_iter_mut()
        )
        //.filter(|(_, nbrs, _)| !nbrs.is_empty())
        .for_each(move |(q, neighbours, vector)| {
            compute_local_vector_at(
                Vector3(*q),
                SamplesView::new(neighbours, samples),
                kernel,
                bg_field_params,
                vector,
            );
        });

        Ok(())
    }

    /*
     * Compute MLS potential on mesh
     */

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_on_mesh<K, F, M>(
        self,
        mesh: &mut M,
        kernel: K,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
        T: na::RealField,
    {
        let query_surf = QueryTopo::new(mesh.vertex_positions(), self);

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *query_surf.base();

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

        // Alternative potential for prototyping
        let alt_potential_attrib = mesh
            .remove_attrib::<VertexIndex>("alt_potential")
            .ok() // convert to option (None when it doesn't exist)
            .unwrap_or_else(|| Attribute::from_vec(vec![0.0f32; mesh.num_vertices()]));

        let mut alt_potential = alt_potential_attrib.into_buffer().cast_into_vec::<f32>();
        if alt_potential.is_empty() {
            // Couldn't cast, which means potential is of some non-numeric type.
            // We overwrite it because we need that attribute spot.
            alt_potential = vec![0.0f32; mesh.num_vertices()];
        }

        // Overwrite these attributes.
        mesh.remove_attrib::<VertexIndex>("normals").ok();
        let mut normals = vec![[0.0f32; 3]; mesh.num_vertices()];
        mesh.remove_attrib::<VertexIndex>("tangents").ok();
        let mut tangents = vec![[0.0f32; 3]; mesh.num_vertices()];

        let query_points = mesh.vertex_positions();
        let neigh_points = query_surf.trivial_neighbourhood_par_chunks(PARALLEL_CHUNK_SIZE);
        let closest_points = query_surf.closest_samples_par_chunks(PARALLEL_CHUNK_SIZE);

        // Initialize extra debug info.
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut bg_weight_attrib_data = vec![0f32; mesh.num_vertices()];
        let mut weight_sum_attrib_data = vec![0f32; mesh.num_vertices()];

        let result = zip!(
            query_points
                .as_parallel_slice()
                .par_chunks(PARALLEL_CHUNK_SIZE),
            neigh_points,
            closest_points,
            num_neighs_attrib_data
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            neighs_attrib_data
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            bg_weight_attrib_data
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            weight_sum_attrib_data
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            potential
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            alt_potential
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            normals
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE),
            tangents
                .as_parallel_slice_mut()
                .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        )
        .map(
            |(
                q_chunk,
                neigh,
                closest,
                num_neighs_chunk,
                neighs_chunk,
                bg_weight_chunk,
                weight_sum_chunk,
                potential_chunk,
                alt_potential_chunk,
                normals_chunk,
                tangents_chunk,
            )| {
                if interrupt() {
                    return Err(super::Error::Interrupted);
                }

                zip!(
                    q_chunk.iter().map(|&v| Vector3(v)),
                    neigh,
                    closest.iter(),
                    num_neighs_chunk.iter_mut(),
                    neighs_chunk.iter_mut(),
                    bg_weight_chunk.iter_mut(),
                    weight_sum_chunk.iter_mut(),
                    potential_chunk.iter_mut(),
                    alt_potential_chunk.iter_mut(),
                    normals_chunk.iter_mut(),
                    tangents_chunk.iter_mut()
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
                        alt_potential,
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

                        *potential = (weight_sum_inv
                            * bg.compute_unnormalized_weighted_scalar_field())
                        .to_f32()
                        .unwrap();

                        *alt_potential = (weight_sum_inv
                            * bg.compute_unnormalized_weighted_scalar_field())
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

                            let p = compute_local_potential_at(
                                q,
                                view,
                                kernel,
                                weight_sum_inv,
                                closest_d,
                            );

                            let alt_p = alt_compute_local_potential_at(
                                q,
                                view,
                                kernel,
                                weight_sum_inv,
                                closest_d,
                            );

                            for Sample { pos, nml, vel, .. } in view.iter() {
                                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                                let grad_w = kernel.with_closest_dist(closest_d).grad(q, pos);
                                let w_normalized = w * weight_sum_inv;
                                let grad_w_normalized =
                                    grad_w * weight_sum_inv - grad_w_sum_normalized * w_normalized;

                                out_normal +=
                                    grad_w_normalized * (q - pos).dot(nml) + nml * w_normalized;

                                // Compute vector interpolation
                                let grad_phi = jacobian::query_jacobian_at(
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

                            *potential += p.to_f32().unwrap();
                            *alt_potential += alt_p.to_f32().unwrap();
                            *normal = out_normal.map(|x| x.to_f32().unwrap()).into();
                            *tangent = out_tangent.map(|x| x.to_f32().unwrap()).into();
                        }
                    },
                );
                Ok(())
            },
        )
        .reduce(|| Ok(()), |acc, result| acc.and(result));

        {
            mesh.set_attrib_data::<_, VertexIndex>("num_neighbours", &num_neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("neighbours", &neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("bg_weight", &bg_weight_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("weight_sum", &weight_sum_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("potential", &potential)?;
            mesh.set_attrib_data::<_, VertexIndex>("alt_potential", &alt_potential)?;
            mesh.set_attrib_data::<_, VertexIndex>("normals", &normals)?;
            mesh.set_attrib_data::<_, VertexIndex>("tangents", &tangents)?;
        }

        result
    }
}

impl<T: Real + Send + Sync> ImplicitSurface<T> {
    fn base(&self) -> &ImplicitSurfaceBase<T> {
        match self {
            ImplicitSurface::MLS(mls) => mls.base(),
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => surf_base,
        }
    }

    fn base_mut(&mut self) -> &mut ImplicitSurfaceBase<T> {
        match self {
            ImplicitSurface::MLS(mls) => mls.base_mut(),
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => surf_base,
        }
    }

    /// Radius of influence ( kernel radius ) for this implicit surface.
    pub fn radius(&self) -> f64 {
        match self {
            ImplicitSurface::MLS(mls) => mls.radius(),
            ImplicitSurface::Hrbf(_) => std::f64::INFINITY,
        }
    }

    /// Return the surface vertex positions used by this implicit surface.
    pub fn surface_vertex_positions(&self) -> &[Vector3<T>] {
        &self.base().surface_vertex_positions
    }

    /// Return the surface topology used by this implicit surface.
    pub fn surface_topology(&self) -> &[[usize; 3]] {
        &self.base().surface_topo
    }

    /// Return the number of samples used by this implicit surface.
    pub fn samples(&self) -> &Samples<T> {
        &self.base().samples
    }

    /// Return the number of samples used by this implicit surface.
    pub fn num_samples(&self) -> usize {
        self.base().samples.len()
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
        self.base_mut().update(vertex_iter)
    }

    /*
    /// This function produces an iterator with an item for each query point which itself is an
    /// iterator over all neighbours (samples) of the corresponding query point.
    pub fn nearest_neighbour_iter<'a>(&'a self, query_points: &'a [[T; 3]]) -> impl Iterator<Item = impl Iterator<Item = Sample<T>> + 'a> + 'a {
        let ImplicitSurface {
            ref kernel,
            base_radius,
            ref spatial_tree,
            ref samples,
            max_step,
            ..
        } = *self;

        let radius = base_radius * radius_multiplier;
        let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
        let radius2 = radius_ext * radius_ext;
        query_points.iter().map(move |&q| {
            let q_pos = Vector3(q).cast::<f64>().unwrap().into();
            self.spatial_tree.locate_within_distance(q_pos, radius2).cloned()
        })

        // Global kernel, all points are neighbours
        (None, Some(query_points.iter().map(move |_| {
            self.samples.iter()
        })))
    }
    */

    /// Compute neighbour storage. Return true if neighbours have been changed.
    pub fn compute_neighbours(&mut self, query_points: &[[T; 3]]) -> bool {
        match self {
            ImplicitSurface::MLS(mls) => mls.compute_neighbours(query_points),
            _ => {
                true
                // Nothing to be done here, each query point neighbours all samples.
                //// Global kernel, all points are neighbours
                //let neigh = |_| samples.iter();
                //let mut cache = self.query_neighbourhood.borrow_mut();
                //cache.compute_neighbourhoods(
                //    query_points,
                //    neigh,
                //    |q| {
                //        spatial_tree
                //            .nearest_neighbor(&Vector3(q).cast::<f64>().unwrap().into())
                //            .expect("Empty spatial tree")
                //    },
                //    surface_topo,
                //    dual_topo,
                //    sample_type,
                //)
            }
        }
    }

    /// The number of query points in the cache (regardless if their neighbourhood is empty).
    /// This function returns `None` if the cache is invalid.
    pub fn num_query_points(&self) -> Result<usize, super::Error> {
        if let ImplicitSurface::MLS(mls) = self {
            mls.num_query_points()
        } else {
            Err(super::Error::UnsupportedKernel)
        }
    }

    /// The number of query points with non-empty neighbourhoods in the cache.
    /// This function returns `None` if the cache is invalid.
    pub fn num_neighbourhoods(&self) -> Result<usize, super::Error> {
        if let ImplicitSurface::MLS(mls) = self {
            mls.num_neighbourhoods()
        } else {
            Err(super::Error::UnsupportedKernel)
        }
    }

    /// Return a vector of indices for query points with non-empty neighbourhoods.
    pub fn nonempty_neighbourhood_indices(&self) -> Result<Vec<usize>, super::Error> {
        if let ImplicitSurface::MLS(mls) = self {
            mls.nonempty_neighbourhood_indices()
        } else {
            Err(super::Error::UnsupportedKernel)
        }
    }

    /// Return a vector over query points, giving the sizes of each neighbourhood.
    pub fn neighbourhood_sizes(&self) -> Result<Vec<usize>, super::Error> {
        if let ImplicitSurface::MLS(mls) = self {
            mls.neighbourhood_sizes()
        } else {
            Err(super::Error::UnsupportedKernel)
        }
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        if let ImplicitSurface::MLS(mls) = self {
            mls.update_max_step(max_step);
        }
    }

    pub fn update_radius_multiplier(&mut self, new_radius_multiplier: f64) {
        if let ImplicitSurface::MLS(mls) = self {
            mls.update_radius_multiplier(new_radius_multiplier);
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
        match self {
            ImplicitSurface::MLS(mls) => mls.potential(query_points, out_field),
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => {
                Self::compute_hrbf(query_points, &surf_base.samples, out_field)
            }
        }
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
        match self {
            ImplicitSurface::MLS(mls) => apply_kernel_fn!(mls, |kernel| mls.compute_vector_field(
                query_points,
                kernel,
                out_vectors
            )),
            ImplicitSurface::Hrbf(HrbfSurface { .. }) => Err(super::Error::UnsupportedKernel),
        }
    }

    /*
     * The methods below are designed for debugging and visualization.
     */

    /// Compute the implicit surface potential on the given polygon mesh.
    pub fn compute_potential_on_mesh<F, M>(
        self,
        mesh: &mut M,
        interrupt: F,
    ) -> Result<(), super::Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
        T: na::RealField,
    {
        match self {
            ImplicitSurface::MLS(mls) => {
                apply_kernel_fn!(&mls, |kernel| mls.compute_on_mesh(mesh, kernel, interrupt))
            }
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => {
                Self::compute_hrbf_on_mesh(mesh, &surf_base.samples, interrupt)
            }
        }
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
        debug_assert!(
            query_points.iter().all(|&q| q.iter().all(|&x| !x.is_nan())),
            "Detected NaNs in query points. Please report this bug."
        );

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
            .chunks(PARALLEL_CHUNK_SIZE)
            .zip(potential.chunks_mut(PARALLEL_CHUNK_SIZE))
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

/*
 * Potential function compoenents
 *
 * The following routines compute parts of various potential functions defining implicit surfaces.
 */

/// Compute the potential at a given query point. If the potential is invalid or nieghbourhood
/// is empty, `potential` is not modified, otherwise it's updated.
/// Note: passing the output parameter potential as a mut reference allows us to optionally mix
/// a pre-initialized custom global potential field with the local potential.
pub(crate) fn compute_potential_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    bg_potential: BackgroundFieldParams,
    potential: &mut T,
) where
    T: Real + Send + Sync,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
{
    if samples.is_empty() {
        return;
    }

    let bg = BackgroundField::local(q, samples, kernel, bg_potential, Some(*potential)).unwrap();

    let weight_sum_inv = bg.weight_sum_inv();

    // Generate a background potential field for every query point. This will be mixed
    // in with the computed potentials for local methods.
    *potential = bg.compute_unnormalized_weighted_scalar_field() * weight_sum_inv;

    let local_field =
        compute_local_potential_at(q, samples, kernel, weight_sum_inv, bg.closest_sample_dist());

    *potential += local_field;
}

/// Compute the potential field (excluding background field) at a given query point.
pub(crate) fn compute_local_potential_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    weight_sum_inv: T,
    closest_d: T,
) -> T
where
    T: Real + Send + Sync,
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

/// Compute the potential field (excluding background field) at a given query point.
pub(crate) fn alt_compute_local_potential_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    weight_sum_inv: T,
    closest_d: T,
) -> T
where
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    T: na::RealField + Real + Send + Sync,
{
    use na::{DMatrix, DVector};
    let basis = DMatrix::from_iterator(
        4,
        samples.len(),
        samples
            .iter()
            .flat_map(|s| vec![T::one(), s.pos[0], s.pos[1], s.pos[2]].into_iter()),
    );

    let diag_weights: Vec<T> = samples
        .iter()
        .map(|s| kernel.with_closest_dist(closest_d).eval(q, s.pos) * weight_sum_inv)
        .collect();

    let weights = DMatrix::from_diagonal(&DVector::from_vec(diag_weights));

    let basis_view = &basis;
    let h = basis_view * &weights * basis_view.transpose();

    let sample_data: Vec<T> = samples
        .iter()
        .map(|s| s.value + s.nml.dot(q - s.pos) / s.nml.norm())
        .collect();

    let rhs = basis * weights * DVector::from_vec(sample_data);

    h.svd(true, true)
        .solve(&rhs, T::from(1e-9).unwrap())
        .map(|c| c[0] + q[0] * c[1] + q[1] * c[2] + q[2] * c[3])
        .unwrap_or(T::from(std::f64::NAN).unwrap())
}

pub(crate) fn compute_local_vector_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    bg_potential: BackgroundFieldParams,
    vector: &mut [T; 3],
) where
    T: Real + Send + Sync,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
{
    if samples.is_empty() {
        return;
    }

    let bg =
        BackgroundField::local(q, samples, kernel, bg_potential, Some(Vector3(*vector))).unwrap();

    // Generate a background potential field for every query point. This will be mixed
    // in with the computed potentials for local methods.
    let mut out_field = bg.compute_unnormalized_weighted_vector_field();

    let closest_dist = bg.closest_sample_dist();

    let weight_sum_inv = bg.weight_sum_inv();

    let grad_phi = jacobian::query_jacobian_at(q, samples, None, kernel, bg_potential);

    for Sample { pos, vel, nml, .. } in samples.iter() {
        out_field += jacobian::sample_contact_jacobian_product_at(
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
 * Derivative and normal computation utils
 */

/// Compute the gradient vector product of the face normals with respect to
/// surface vertices.
///
/// This function returns an iterator with the same size as `surface_vertices.len()`.
///
/// Note that the product vector is given by a closure `multiplier` which must give a valid
/// vector value for any vertex index, however not all indices will be used since only the
/// neighbourhood of vertex at `index` will have non-zero gradients.
pub(crate) fn compute_face_unit_normals_gradient_products<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_vertices: &'a [Vector3<T>],
    surface_topo: &'a [[usize; 3]],
    mut multiplier: F,
) -> impl Iterator<Item = Vector3<T>> + 'a
where
    T: Real + Send + Sync,
    F: FnMut(Sample<T>) -> Vector3<T> + 'a,
{
    samples.into_iter().flat_map(move |sample| {
        let mult = multiplier(sample);
        let grad = face_unit_normal_gradient_iter(sample, surface_vertices, surface_topo);
        grad.map(move |g| g * mult)
    })
}

/// Compute the gradient of the face normal at the given sample with respect to
/// its vertices. The returned triple of `Matrix3`s corresonds to the block column vector of
/// three matrices corresponding to each triangle vertex, which together construct the actual
/// `9x3` component-wise gradient.
pub(crate) fn face_unit_normal_gradient_iter<T>(
    sample: Sample<T>,
    surface_vertices: &[Vector3<T>],
    surface_topo: &[[usize; 3]],
) -> impl Iterator<Item = Matrix3<T>>
where
    T: Real + Send + Sync,
{
    let nml_proj = scaled_tangent_projection(sample);
    let tri_indices = &surface_topo[sample.index];
    let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
    (0..3).map(move |i| tri.area_normal_gradient(i) * nml_proj)
}

/// Compute the matrix for projecting on the tangent plane of the given sample inversely scaled
/// by the local area (normal norm reciprocal).
pub(crate) fn scaled_tangent_projection<T>(sample: Sample<T>) -> Matrix3<T>
where
    T: Real + Send + Sync,
{
    let nml_norm_inv = T::one() / sample.nml.norm();
    let nml = sample.nml * nml_norm_inv;
    Matrix3::diag([nml_norm_inv; 3]) - (nml * nml_norm_inv) * nml.transpose()
}

/// Compute the gradient vector product of the `compute_vertex_unit_normals` function with respect to
/// vertices given in the sample view.
///
/// This function returns an iterator with the same size as `samples.len()`.
///
/// Note that the product vector is given by a closure `dx` which must give a valid vector
/// value for any vertex index, however not all indices will be used since only the
/// neighbourhood of vertex at `index` will have non-zero gradients.
pub(crate) fn compute_vertex_unit_normals_gradient_products<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_topo: &'a [[usize; 3]],
    dual_topo: &'a [Vec<usize>],
    mut dx: F,
) -> impl Iterator<Item = Vector3<T>> + 'a
where
    T: Real + Send + Sync,
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

/// Utility function to subdivide a range into a `Vec` of smaller ranges with a given size.
fn split_range_into_chunks(
    rng: std::ops::Range<usize>,
    chunk_size: usize,
) -> Vec<std::ops::Range<usize>> {
    let len = rng.end - rng.start;
    let num_uniform_chunks = len / chunk_size;
    let mut ranges = Vec::with_capacity(num_uniform_chunks + 1);
    for i in 0..num_uniform_chunks {
        let begin = i * chunk_size + rng.start;
        let end = (i + 1) * chunk_size + rng.start;
        ranges.push(begin..end);
    }

    // Add remainder chunk
    if len > ranges.len() * chunk_size {
        ranges.push(num_uniform_chunks * chunk_size + rng.start..rng.end);
    }
    ranges
}

impl<T: Real + Send + Sync> GlobalMLS<T> {
    pub fn trivial_neighbourhood_seq<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a [usize]> + ExactSizeIterator + Clone + 'a {
        (0..self.closest_samples.len())
            .into_iter()
            .map(move |_| self.sample_indices.as_slice())
    }

    pub fn closest_samples(&self) -> Result<&[usize], super::Error> {
        if !self.closest_samples.is_empty() {
            Ok(self.closest_samples.as_slice())
        } else {
            Err(super::Error::MissingNeighbourData)
        }
    }

    pub fn extended_neighbourhood_seq<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a [usize]> + ExactSizeIterator + 'a {
        self.trivial_neighbourhood_seq()
    }
    pub fn trivial_neighbourhood_par<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a {
        (0..self.closest_samples.len())
            .into_par_iter()
            .map(move |_| self.sample_indices.as_slice())
    }
    pub fn extended_neighbourhood_par<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a {
        self.trivial_neighbourhood_par()
    }
    pub fn trivial_neighbourhood_par_chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a>>
                 + IndexedParallelIterator
                 + 'a {
        let ranges = split_range_into_chunks(0..self.closest_samples.len(), chunk_size);
        ranges.into_par_iter().map(move |rng| {
            let out: Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a> =
                Box::new(rng.map(move |_| self.sample_indices.as_slice()));
            out
        })
    }
    pub fn extended_neighbourhood_par_chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = impl Iterator<Item = &'a [usize]> + ExactSizeIterator + 'a>
                 + IndexedParallelIterator
                 + 'a {
        let ranges = split_range_into_chunks(0..self.closest_samples.len(), chunk_size);
        ranges
            .into_par_iter()
            .map(move |rng| rng.map(move |_| self.sample_indices.as_slice()))
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

    #[test]
    fn split_range_into_chunks() {
        let r = 0..4;
        assert_eq!(super::split_range_into_chunks(r, 2), vec![0..2, 2..4]);
        let r = 0..17;
        assert_eq!(
            super::split_range_into_chunks(r, 5),
            vec![0..5, 5..10, 10..15, 15..17]
        );
        let r = 0..17;
        assert_eq!(
            super::split_range_into_chunks(r, 3),
            vec![0..3, 3..6, 6..9, 9..12, 12..15, 15..17]
        );
    }

    // Helper function for testing. This is an implicit surface and grid mesh pair where each
    // vertex of the grid mesh has a non-empty local neighbpourhood of the implicit surface.
    // The `reverse` option reverses each triangle in the sphere to create an inverted implicit
    // surface.
    fn make_octahedron_and_grid_local(reverse: bool) -> Result<(MLS, PolyMesh<f64>), crate::Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);
        if reverse {
            sphere.reverse();
        }

        // Translate the mesh slightly in z.
        utils::translate(&mut sphere, [0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = mls_from_polymesh(
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
    ) -> Result<(MLS, PolyMesh<f64>), crate::Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);
        if reverse {
            sphere.reverse();
        }

        // Translate the mesh slightly in z.
        utils::translate(&mut sphere, [0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = mls_from_polymesh(
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
        query_surf: &QueryTopo,
        mut grid: PolyMesh<f64>,
        side: Side,
    ) -> Result<(), crate::Error> {
        let epsilon = 1e-4;
        let init_potential = {
            // Get grid node positions to be projected.
            let pos = grid.vertex_positions_mut();

            // Compute potential before projection.
            let mut init_potential = vec![0.0; pos.len()];
            query_surf.potential(pos, &mut init_potential);

            // Project grid outside the implicit surface.
            assert!(query_surf.project(side, 0.0, epsilon, pos));
            init_potential
        };

        // Compute potential after projection.
        let mut final_potential = vec![0.0; init_potential.len()];
        query_surf.potential(grid.vertex_positions(), &mut final_potential);

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
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Above)?;

        let (surface, grid) = make_octahedron_and_grid_local(true)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Below)
    }

    /// Test projection where some projected vertices may not have a local neighbourhood at all.
    /// This is a more complex test than the local_projection_test
    #[test]
    fn global_projection_test() -> Result<(), crate::Error> {
        let (surface, grid) = make_octahedron_and_grid(false, 2.45)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Above)?;

        let (surface, grid) = make_octahedron_and_grid(true, 2.45)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Below)
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
        let surface = mls_from_polymesh(
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

        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Above)?;

        Ok(())
    }

    mod test_structs {
        use serde::{Deserialize, Serialize};
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct NeighbourCache<T> {
            pub points: Vec<T>,
            pub valid: bool,
        }

        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct Neighbourhood {
            closest_set: NeighbourCache<usize>,
            trivial_set: NeighbourCache<Vec<usize>>,
            extended_set: NeighbourCache<Vec<usize>>,
        }

        /// This struct helps deserialize testing assets without having to store an rtree.
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub struct ImplicitSurfaceNoTree {
            pub kernel: super::KernelType,
            pub base_radius: f64,
            pub bg_field_params: super::BackgroundFieldParams,
            pub surface_topo: Vec<[usize; 3]>,
            pub surface_vertex_positions: Vec<super::Vector3<f64>>,
            pub samples: super::Samples<f64>,
            pub max_step: f64,
            pub query_neighbourhood: std::cell::RefCell<Neighbourhood>,
            pub dual_topo: Vec<Vec<usize>>,
            pub sample_type: super::SampleType,
        }
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

        let surface: MLS<f64> = {
            let mut file = std::fs::File::open("assets/torus_surf_no_tree.json")
                .expect("Failed to open torus surface file");
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .expect("Failed to read torus surface json.");
            let test_structs::ImplicitSurfaceNoTree {
                kernel,
                base_radius,
                bg_field_params,
                surface_topo,
                surface_vertex_positions,
                samples,
                max_step,
                dual_topo,
                sample_type,
                ..
            } = serde_json::from_str(&contents).expect("Failed to deserialize torus surface.");
            let spatial_tree = build_rtree_from_samples(&samples);
            MLS::Local(LocalMLS {
                kernel: kernel.into(),
                base_radius,
                max_step,
                query_neighbourhood: Neighbourhood::new(),
                surf_base: ImplicitSurfaceBase {
                    bg_field_params,
                    surface_topo,
                    surface_vertex_positions,
                    samples,
                    dual_topo,
                    sample_type,
                    spatial_tree,
                },
            })
        };

        let query_surf = QueryTopo::new(&query_points, surface);

        let init_potential = {
            // Compute potential before projection.
            let mut init_potential = vec![0.0; query_points.len()];
            query_surf.potential(&query_points, &mut init_potential);
            init_potential
        };

        // Project grid outside the implicit surface.
        assert!(query_surf.project_to_above(iso_value, epsilon, &mut query_points));

        // Compute potential after projection.
        let mut final_potential = vec![0.0; init_potential.len()];
        query_surf.potential(&query_points, &mut final_potential);

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
    fn neighbourhoods() -> Result<(), crate::Error> {
        // Local test
        let (surface, grid) = make_octahedron_and_grid_local(false)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        assert_eq!(
            query_surf.num_neighbourhoods(),
            query_surf.nonempty_neighbourhood_indices().len()
        );

        // Non-local test
        let (surface, grid) = make_octahedron_and_grid(false, 2.45)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        assert_eq!(
            query_surf.num_neighbourhoods(),
            query_surf.nonempty_neighbourhood_indices().len()
        );
        Ok(())
    }
}
