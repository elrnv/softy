use super::*;
use rayon::iter::Either;
use serde::{Deserialize, Serialize};

/// A data structure to store precomputed query information about an implicit surface. For example
/// precomputed sample neighbours to each query point are stored here for local potential fields.
/// Note that this data structure doesn't store the actual query positions, just the topologies,
/// since different query positions can be used with the same topology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QueryTopo<T = f64>
where
    T: Real,
{
    Local {
        /// Local mls implicit surface.
        surf: LocalMLS<T>,
        /// Store the neighbouring sample points for each query point we see.
        neighbourhood: Neighbourhood,
    },
    Global {
        /// Global mls implicit surface.
        surf: GlobalMLS<T>,
        /// Closest sample indices.
        closest_samples: Vec<usize>,
        /// An array of sample indices `0..#samples`. This is here to make neighbour api compatible
        /// with local MLS.
        sample_indices: Vec<usize>,
    },
}

impl<T: Real + Send + Sync> QueryTopo<T> {
    /*
     * Constructors
     */
    pub fn new(query_points: &[[T; 3]], mls: MLS<T>) -> Self {
        let mut query_topo = match mls {
            MLS::Local(surf) => QueryTopo::Local {
                surf,
                neighbourhood: Neighbourhood::new(),
            },
            MLS::Global(surf) => {
                let sample_indices: Vec<_> = (0..surf.surf_base.samples.len()).collect();
                QueryTopo::Global {
                    surf,
                    closest_samples: Vec::new(),
                    sample_indices,
                }
            }
        };
        query_topo.reset(query_points);
        query_topo
    }

    /// Compute neighbour topology. Return true if topology has changed.
    pub fn reset(&mut self, query_points: &[[T; 3]]) -> bool {
        match *self {
            QueryTopo::Local {
                ref mut neighbourhood,
                surf:
                    LocalMLS {
                        base_radius,
                        kernel,
                        max_step,
                        surf_base:
                            ImplicitSurfaceBase {
                                ref spatial_tree,
                                ref surface_topo,
                                ref dual_topo,
                                sample_type,
                                ..
                            },
                        ..
                    },
            } => {
                let radius = base_radius * kernel.radius_multiplier(); // TODO: refactor with self.radius() fn
                let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
                let radius2 = radius_ext * radius_ext;
                let neighbourhood_query = |q| {
                    let q_pos = Vector3(q).cast::<f64>().unwrap().into();
                    spatial_tree.locate_within_distance(q_pos, radius2).cloned()
                };
                let closest_sample_query = |q| {
                    spatial_tree
                        .nearest_neighbor(&Vector3(q).cast::<f64>().unwrap().into())
                        .expect("Empty spatial tree")
                };
                neighbourhood.compute_neighbourhoods(
                    query_points,
                    neighbourhood_query,
                    closest_sample_query,
                    surface_topo,
                    dual_topo,
                    sample_type,
                )
            }
            QueryTopo::Global {
                ref mut closest_samples,
                surf:
                    GlobalMLS {
                        surf_base:
                            ImplicitSurfaceBase {
                                ref spatial_tree, ..
                            },
                        ..
                    },
                ..
            } => neighbour_cache::compute_closest_set(
                query_points,
                |q| {
                    spatial_tree
                        .nearest_neighbor(&Vector3(q).cast::<f64>().unwrap().into())
                        .expect("Empty spatial tree")
                },
                closest_samples,
            ),
        }
    }

    /*
     * Conversions
     */
    pub fn into_surf(self) -> MLS<T> {
        match self {
            QueryTopo::Local { surf, .. } => MLS::Local(surf),
            QueryTopo::Global { surf, .. } => MLS::Global(surf),
        }
    }

    /*
     * Base accessors
     */

    pub fn base(&self) -> &ImplicitSurfaceBase<T> {
        match self {
            QueryTopo::Local { surf, .. } => &surf.surf_base,
            QueryTopo::Global { surf, .. } => &surf.surf_base,
        }
    }

    pub fn base_mut(&mut self) -> &mut ImplicitSurfaceBase<T> {
        match self {
            QueryTopo::Local { surf, .. } => &mut surf.surf_base,
            QueryTopo::Global { surf, .. } => &mut surf.surf_base,
        }
    }

    /// Radius of influence ( kernel radius ) for this implicit surface.
    pub fn radius(&self) -> f64 {
        match self {
            QueryTopo::Local { surf, .. } => surf.kernel.radius_multiplier() * surf.base_radius,
            QueryTopo::Global { .. } => std::f64::INFINITY,
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

    /// The number of query points in the cache (regardless if their neighbourhood is empty).
    /// This function returns `None` if the cache is invalid.
    pub fn num_query_points(&self) -> usize {
        self.trivial_neighbourhood_seq().len()
    }

    /// The number of query points with non-empty neighbourhoods.
    /// This function returns `None` if the cache is invalid.
    pub fn num_neighbourhoods(&self) -> usize {
        self.trivial_neighbourhood_seq()
            .filter(|x| !x.is_empty())
            .count()
    }

    /// Return a vector of indices for query points with non-empty neighbourhoods.
    pub fn nonempty_neighbourhood_indices(&self) -> Vec<usize> {
        match self.base().sample_type {
            SampleType::Vertex => {
                Self::nonempty_neighbourhood_indices_impl(self.extended_neighbourhood_seq())
            }
            SampleType::Face => {
                Self::nonempty_neighbourhood_indices_impl(self.trivial_neighbourhood_seq())
            }
        }
    }

    /// Return a vector over query points, giving the sizes of each neighbourhood.
    pub fn neighbourhood_sizes(&self) -> Vec<usize> {
        match self.base().sample_type {
            SampleType::Vertex => self.extended_neighbourhood_seq().map(|x| x.len()).collect(),
            SampleType::Face => self.trivial_neighbourhood_seq().map(|x| x.len()).collect(),
        }
    }

    /*
     * Update functions
     */

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        if let QueryTopo::Local { surf, .. } = self {
            surf.max_step = max_step;
        }
    }

    pub fn update_radius_multiplier(&mut self, new_radius_multiplier: f64) {
        match self {
            QueryTopo::Local {
                surf: LocalMLS { kernel, .. },
                ..
            } => {
                *kernel = kernel.with_radius_multiplier(new_radius_multiplier);
            }
            _ => {}
        }
    }

    /// Update vertex positions and samples using an iterator over mesh vertices. This is a very
    /// permissive `update` function, which will update as many positions as possible and recompute
    /// the implicit surface data (like samples and spatial tree if needed) whether or not enough
    /// positions were specified to cover all surface vertices. This function will return the
    /// number of vertices that were indeed updated.
    ///
    /// Note that this function does not update the query topology, however the topology may still
    /// be valid if the surface has not moved very far.
    pub fn update_surface<I>(&mut self, vertex_iter: I) -> usize
    where
        I: Iterator<Item = [T; 3]>,
    {
        self.base_mut().update(vertex_iter)
    }

    /*
     * Neighbourhood accessors
     */

    /// This function returns precomputed neighbours.
    pub fn trivial_neighbourhood_par<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => {
                Either::Left(neighbourhood.trivial_set().par_iter().map(|x| x.as_slice()))
            }
            QueryTopo::Global {
                closest_samples,
                sample_indices,
                ..
            } => Either::Right(
                (0..closest_samples.len())
                    .into_par_iter()
                    .map(move |_| sample_indices.as_slice()),
            ),
        }
    }

    /// This function returns precomputed closest samples.
    pub fn closest_samples_par<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = usize> + IndexedParallelIterator + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => {
                Either::Left(neighbourhood.closest_set().par_iter().cloned())
            }
            QueryTopo::Global {
                closest_samples, ..
            } => Either::Right(closest_samples.as_parallel_slice().par_iter().cloned()),
        }
    }

    /// This function returns precomputed neighbours.
    pub fn extended_neighbourhood_par<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => Either::Left(
                neighbourhood
                    .extended_set()
                    .par_iter()
                    .map(|x| x.as_slice()),
            ),
            QueryTopo::Global { .. } => Either::Right(self.trivial_neighbourhood_par()),
        }
    }

    /// This function returns precomputed neighbours.
    pub fn trivial_neighbourhood_par_chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a>>
                 + IndexedParallelIterator
                 + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => Either::Left(
                neighbourhood
                    .trivial_set()
                    .as_parallel_slice()
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let out: Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a> =
                            Box::new(chunk.iter().map(|nbrs| nbrs.as_slice()));
                        out
                    }),
            ),
            QueryTopo::Global {
                closest_samples,
                sample_indices,
                ..
            } => {
                let ranges = split_range_into_chunks(0..closest_samples.len(), chunk_size);
                Either::Right(ranges.into_par_iter().map(move |rng| {
                    let out: Box<dyn Iterator<Item = &'a [usize]> + Send + Sync + 'a> =
                        Box::new(rng.map(move |_| sample_indices.as_slice()));
                    out
                }))
            }
        }
    }

    /// This function returns precomputed closest samples.
    pub fn closest_samples_par_chunks<'a>(
        &'a self,
        chunk_size: usize,
    ) -> impl ParallelIterator<Item = &'a [usize]> + IndexedParallelIterator + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => Either::Left(
                neighbourhood
                    .closest_set()
                    .as_parallel_slice()
                    .par_chunks(chunk_size),
            ),
            QueryTopo::Global {
                closest_samples, ..
            } => Either::Right(closest_samples.as_parallel_slice().par_chunks(chunk_size)),
        }
    }

    /// This function returns precomputed neighbours.
    pub fn trivial_neighbourhood_seq<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a [usize]> + ExactSizeIterator + Clone + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => {
                Either::Left(neighbourhood.trivial_set().iter().map(|x| x.as_slice()))
            }
            QueryTopo::Global {
                closest_samples,
                sample_indices,
                ..
            } => Either::Right(
                (0..closest_samples.len())
                    .into_iter()
                    .map(move |_| sample_indices.as_slice()),
            ),
        }
    }

    /// This function returns precomputed closest samples.
    pub fn closest_samples_seq<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => {
                Either::Left(neighbourhood.closest_set().iter().cloned())
            }
            QueryTopo::Global {
                closest_samples, ..
            } => Either::Right(closest_samples.iter().cloned()),
        }
    }

    /// This function returns precomputed neighbours.
    pub fn extended_neighbourhood_seq<'a>(&'a self) -> impl Iterator<Item = &'a [usize]> + 'a {
        match self {
            QueryTopo::Local { neighbourhood, .. } => {
                Either::Left(neighbourhood.extended_set().iter().map(|x| x.as_slice()))
            }
            QueryTopo::Global { .. } => Either::Right(self.trivial_neighbourhood_seq()),
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
     * Queries and computations
     */

    pub fn num_neighbours_within_distance<Q: Into<[T; 3]>>(&self, q: Q, radius: f64) -> usize {
        let q_pos = Vector3(q.into()).cast::<f64>().unwrap().into();
        self.base()
            .spatial_tree
            .locate_within_distance(q_pos, radius * radius)
            .count()
    }

    pub fn nearest_neighbour_lookup<Q: Into<[T; 3]>>(&self, q: Q) -> Option<&Sample<T>> {
        let q_pos = Vector3(q.into()).cast::<f64>().unwrap().into();
        self.base().spatial_tree.nearest_neighbor(&q_pos)
    }

    /// Project the given set of positions to be below the specified iso-value along the gradient.
    /// If the query point is already below the given iso-value, then it is not modified.
    /// The given `epsilon` determines how far below the iso-surface the point is allowed to be
    /// projected, essentially it is the thickness below the iso-surface of value projections.
    /// This function will return true if convergence is achieved and false if the projection needed
    /// more iterations.
    pub fn project_to_below(&self, iso_value: T, epsilon: T, query_points: &mut [[T; 3]]) -> bool {
        self.project(Side::Below, iso_value, epsilon, query_points)
    }

    /// Project the given set of positions to be above the specified iso-value along the gradient.
    /// If the query point is already above the given iso-value, then it is not modified.
    /// The given `epsilon` determines how far above the iso-surface the point is allowed to be
    /// projected, essentially it is the thickness above the iso-surface of value projections.
    /// This function will return true if convergence is achieved and false if the projection needed
    /// more iterations.
    pub fn project_to_above(&self, iso_value: T, epsilon: T, query_points: &mut [[T; 3]]) -> bool {
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
    ) -> bool {
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
            self.potential(query_points, &mut potential);
            potential.iter_mut().for_each(|x| *x *= multiplier);

            // The transpose of the potential gradient at each of the query points.
            self.query_jacobian_full(query_points, &mut steps);

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

                self.potential(&candidate_points, &mut candidate_potential);
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

        convergence
    }

    /*
     * Main potential computation
     */

    /// Compute the mls potential.
    pub fn potential(&self, query_points: &[[T; 3]], out_field: &mut [T]) {
        debug_assert!(
            query_points.iter().all(|&q| q.iter().all(|&x| !x.is_nan())),
            "Detected NaNs in query points. Please report this bug."
        );

        apply_kernel_query_fn!(self, |kernel| self.compute_potential(
            query_points,
            kernel,
            out_field
        ))
    }

    /// Compute the mls potential on query points with non-empty neighbourhoods.
    pub fn local_potential(&self, query_points: &[[T; 3]], out_field: &mut [T]) {
        debug_assert!(
            query_points.iter().all(|&q| q.iter().all(|&x| !x.is_nan())),
            "Detected NaNs in query points. Please report this bug."
        );

        apply_kernel_query_fn!(self, |kernel| self.compute_local_potential(
            query_points,
            kernel,
            out_field
        ))
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_potential<'a, K>(&self, query_points: &[[T; 3]], kernel: K, out_field: &'a mut [T])
    where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq();

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
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface on
    /// query points with non-empty neighbourhoods.
    fn compute_local_potential<'a, K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        out_field: &'a mut [T],
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *self.base();

        zip!(query_points.iter(), neigh_points)
            .filter(|(_, nbrs)| !nbrs.is_empty())
            .zip(out_field.iter_mut())
            .for_each(move |((q, neighbours), field)| {
                compute_potential_at(
                    Vector3(*q),
                    SamplesView::new(neighbours, samples),
                    kernel,
                    bg_field_params,
                    field,
                );
            });
    }

    /*
     * Vector field computation
     */

    /// Compute vector field on the surface.
    pub fn vector_field(&self, query_points: &[[T; 3]], out_vectors: &mut [[T; 3]]) {
        apply_kernel_query_fn!(self, |kernel| self.compute_vector_field(
            query_points,
            kernel,
            out_vectors
        ))
    }

    /// Interpolate the given vector field at the given query points.
    fn compute_vector_field<'a, K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        out_vectors: &'a mut [[T; 3]],
    ) where
        K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_par();

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
    }
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