use super::samples::Sample;
use geo::Real;
use rayon::prelude::*;
use std::collections::BTreeSet;

/// Cache neighbouring sample points for each query point.
/// Note that this determines the entire sparsity structure of the query point neighbourhoods.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighbourCache {
    /// For each query point, record the neighbours' indices in this vector.
    pub points: Vec<Vec<usize>>,

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

    /// Check if this cache is valid.
    fn is_valid(&self) -> bool {
        self.valid
    }
}

/// There are two types of neighbourhoods for each query point:
///   1. the set of samples within a distance of the query point, which we call the *trivial
///      set* and
///   2. an extended set of samples reachable via triangles adjacent to the trivial set, which we
///      dub the *extended set*.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Neighbourhood {
    /// The trivial neighbourhood of a set of query points. These are simply the set of samples that
    /// are within a certain distance from each query point.
    trivial_set: NeighbourCache,
    /// The extended neighbourhood of a set of query points. This is a superset of the trivial
    /// neighbourhood that includes samples that are topologically adjacent to the trivial set.
    extended_set: NeighbourCache,
}

impl Neighbourhood {
    pub(crate) fn new() -> Self {
        Neighbourhood {
            trivial_set: NeighbourCache::new(),
            extended_set: NeighbourCache::new(),
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
    ) -> Option<&[Vec<usize>]>
    where
        T: Real + Send + Sync,
    {
        let Neighbourhood {
            trivial_set,
            extended_set,
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
                    if !dual_topo.is_empty() {
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

    pub(crate) fn compute_neighbourhoods<'a, T, I, N>(
        &mut self,
        query_points: &[[T; 3]],
        neigh: N,
        tri_topo: &[[usize; 3]],
        dual_topo: &[Vec<usize>],
    ) where
        T: Real + Send + Sync,
        I: Iterator<Item = Sample<T>> + 'a,
        N: Fn([T; 3]) -> I + Sync + Send,
    {
        self.compute_trivial_set(query_points, neigh);
        self.compute_extended_set(query_points, tri_topo, dual_topo)
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
        self.trivial_set.valid = false;
        self.extended_set.valid = false;
    }
}
