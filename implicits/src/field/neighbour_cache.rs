use super::{samples::Sample, SampleType};
use geo::Real;
use rayon::prelude::*;
use std::collections::BTreeSet;

/// Cache data structure that stores information about the neighbourhood of a query point.
/// For radial neighbourhood sets, this determines the entire sparsity structure for each query
/// point.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighbourCache<T> {
    /// For each query point, record the neighbour data in this vector.
    pub points: Vec<T>,

    /// Marks the cache valid or not. If this flag is false, the cache needs to be recomputed, but
    /// we can reuse the already allocated memory.
    pub valid: bool,
}

impl<T> NeighbourCache<T> {
    fn new() -> Self {
        NeighbourCache {
            points: Vec::new(),
            valid: false,
        }
    }

    /// Check if this cache is valid.
    fn is_valid(&self) -> bool {
        self.valid
    }
}

/// There are three types of neighbourhoods for each query point:
///   1. closest samples to each query point, which we call the *closest set*,
///   1. the set of samples within a distance of the query point, which we call the *trivial
///      set* and
///   2. an extended set of samples reachable via triangles adjacent to the trivial set, which we
///      dub the *extended set*.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Neighbourhood {
    /// The closest sample to each query point.
    closest_set: NeighbourCache<usize>,
    /// The trivial neighbourhood of a set of query points. These are simply the set of samples that
    /// are within a certain distance from each query point.
    trivial_set: NeighbourCache<Vec<usize>>,
    /// The extended neighbourhood of a set of query points. This is a superset of the trivial
    /// neighbourhood that includes samples that are topologically adjacent to the trivial set.
    extended_set: NeighbourCache<Vec<usize>>,
}

impl Neighbourhood {
    pub(crate) fn new() -> Self {
        Neighbourhood {
            closest_set: NeighbourCache::new(),
            trivial_set: NeighbourCache::new(),
            extended_set: NeighbourCache::new(),
        }
    }

    /// Compute closest point set if it is empty (e.g. hasn't been created yet). Return the 
    /// closest sample to each query point.
    ///
    /// Note that this set must be invalidated explicitly, there is no real way to automatically
    /// cache results because both: query points and sample points may change slightly, but we
    /// expect the neighbourhood information to remain the same.
    pub(crate) fn compute_closest_set<'a, T, C>(
        &mut self,
        query_points: &[[T; 3]],
        closest: C,
    ) -> &[usize]
    where
        T: Real + Send + Sync + 'a,
        C: Fn([T; 3]) -> &'a Sample<T> + Send + Sync,
    {
        let set = &mut self.closest_set;

        if !set.is_valid() {
            // Allocate additional neighbourhoods to match the size of query_points.
            set.points.resize(query_points.len(), 0);

            query_points
                .par_iter()
                .zip(set.points.par_iter_mut())
                .for_each(|(q, sample_idx)| {
                    *sample_idx = closest(*q).index;
                });

            set.valid = true;
        }

        &set.points
    }

    /// Getter for the closest set of samples.
    pub(crate) fn closest_set(&self) -> Option<&[usize]> {
        if self.closest_set.valid {
            Some(&self.closest_set.points)
        } else {
            None
        }
    }

    /// Compute neighbour cache if it is invalid (or hasn't been created yet). Return the *trivial
    /// set* of neighbours of the given query points.
    ///
    /// Note that the cache must be invalidated explicitly, there is no real way to automatically
    /// cache results because both: query points and sample points may change slightly, but we
    /// expect the neighbourhood information to remain the same.
    pub(crate) fn compute_trivial_set<'a, T, I, N>(
        &mut self,
        query_points: &[[T; 3]],
        neigh: N,
    ) -> &[Vec<usize>]
    where
        T: Real + Send + Sync,
        I: Iterator<Item = Sample<T>> + 'a,
        N: Fn([T; 3]) -> I + Sync + Send,
    {
        let cache = &mut self.trivial_set;

        if !cache.is_valid() {
            // Allocate additional neighbourhoods to match the size of query_points.
            cache.points.resize(query_points.len(), Vec::new());

            query_points
                .par_iter()
                .zip(cache.points.par_iter_mut())
                .for_each(|(q, pts)| {
                    pts.clear();
                    pts.extend(neigh(*q).map(|op| op.index));
                });

            cache.valid = true;
        }

        &cache.points
    }

    pub(crate) fn trivial_set(&self) -> Option<&[Vec<usize>]> {
        if self.trivial_set.valid {
            Some(&self.trivial_set.points)
        } else {
            None
        }
    }

    /// Compute neighbour cache if it is invalid (or hasn't been created yet). Return the *extended
    /// set* of neighbours of the given query points. Note that this function requires the trival
    /// set to already be computed. If trivial set has not been previously computed, use the
    /// `compute_neighbourhoods` function to compute both. This fuction will return `None` if the
    /// trivial set is invalid.
    ///
    /// Note that the cache must be invalidated explicitly, there is no real way to automatically
    /// cache results because both: query points and sample points may change slightly, but we
    /// expect the neighbourhood information to remain the same.
    pub(crate) fn compute_extended_set<'a, T>(
        &mut self,
        query_points: &[[T; 3]],
        tri_topo: &[[usize; 3]],
        dual_topo: &[Vec<usize>],
        sample_type: SampleType,
    ) -> Option<&[Vec<usize>]>
    where
        T: Real + Send + Sync,
    {
        let Neighbourhood {
            trivial_set,
            extended_set,
            ..
        } = self;

        if !trivial_set.is_valid() {
            return None;
        }

        if !extended_set.is_valid() {
            // Allocate additional neighbourhoods to match the size of query_points.
            extended_set.points.resize(query_points.len(), Vec::new());

            trivial_set
                .points
                .par_iter()
                .zip(extended_set.points.par_iter_mut())
                .for_each(|(triv_pts, ext_pts)| {
                    ext_pts.clear();
                    let mut set: BTreeSet<_> = triv_pts.into_iter().collect();
                    if sample_type == SampleType::Vertex && !dual_topo.is_empty() {
                        set.extend(triv_pts.iter().flat_map(|&pt| {
                            dual_topo[pt].iter().flat_map(|&tri| tri_topo[tri].iter())
                        }));
                    }
                    ext_pts.extend(set.into_iter());
                });

            extended_set.valid = true;
        }

        Some(&extended_set.points)
    }

    pub(crate) fn compute_neighbourhoods<'a, T, I, N, C>(
        &mut self,
        query_points: &[[T; 3]],
        neigh: N,
        closest: C,
        tri_topo: &[[usize; 3]],
        dual_topo: &[Vec<usize>],
        sample_type: SampleType,
    ) where
        T: Real + Send + Sync + 'a,
        I: Iterator<Item = Sample<T>> + 'a,
        N: Fn([T; 3]) -> I + Sync + Send,
        C: Fn([T; 3]) -> &'a Sample<T> + Send + Sync,
    {
        self.compute_closest_set(query_points, closest);
        self.compute_trivial_set(query_points, neigh);
        self.compute_extended_set(query_points, tri_topo, dual_topo, sample_type)
            .unwrap();
    }

    pub(crate) fn extended_set(&self) -> Option<&[Vec<usize>]> {
        if self.extended_set.valid {
            Some(&self.extended_set.points)
        } else {
            None
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.closest_set.valid = false;
        self.trivial_set.valid = false;
        self.extended_set.valid = false;
    }
}
