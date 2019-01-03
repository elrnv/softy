use super::samples::Sample;
use geo::Real;
use rayon::prelude::*;

/// Cache neighbouring sample points for each query point.
/// Note that this determines the entire sparsity structure of the query point neighbourhoods.
/// This structure is meant to live inside a `RefCell`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NeighbourCache {
    /// For each query point, record the neighbours indices in this vector.
    points: Vec<Vec<usize>>,

    /// Marks the cache valid or not. If this flag is false, the cache needs to be recomputed, but
    /// we can reuse the already allocated memory.
    valid: bool,
}

impl NeighbourCache {
    pub(crate) fn new() -> Self {
        NeighbourCache {
            points: Vec::new(),
            valid: false,
        }
    }

    /// Get the neighbour points for each query point. This call just returns the currently cached
    /// points and doesn't trigger recomputation. Sometimes there is not enough information
    /// available for recomputation, so the user is responsible for ensuring points are upto date.
    pub(crate) fn cached_neighbour_points(&self) -> &[Vec<usize>] {
        &self.points
    }

    /// Check if this cache is valid.
    pub(crate) fn is_valid(&self) -> bool {
        self.valid
    }

    /// Invalidate this cache. Subsequent retrievals will trigger recomputation of this neighbour
    /// points.
    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Compute neighbour cache if it hasn't been computed or it is invalid. Return the neighbours
    /// of the given query points. Note that the cache must be invalidated explicitly, there is no
    /// real way to automatically cache results because both: query points and sample points may
    /// change slightly, but we expect the neighbourhood information to remain the same.
    pub(crate) fn neighbour_points<'a, T, I, N>(
        &mut self,
        query_points: &[[T; 3]],
        neigh: N,
    ) -> &[Vec<usize>]
    where
        T: Real + Send + Sync,
        I: Iterator<Item = Sample<T>> + 'a,
        N: Fn([T; 3]) -> I + Sync + Send,
    {
        if !self.is_valid() {
            // Allocate additional neighbourhoods to match the size of query_points.
            self.points.resize(query_points.len(), Vec::new());

            query_points
                .par_iter()
                .zip(self.points.par_iter_mut())
                .for_each(|(q, pts)| {
                    pts.clear();
                    pts.extend(neigh(*q).map(|op| op.index));
                });

            self.valid = true;
        }

        self.cached_neighbour_points()
    }
}
