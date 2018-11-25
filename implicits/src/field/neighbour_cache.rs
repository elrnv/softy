use super::samples::Sample;

/// Cache neighbouring sample points for each query point.
/// Note that this determines the entire sparsity structure of the query point neighbourhoods.
/// This structure is meant to live inside a `RefCell`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NeighbourCache {
    /// For each query point with a non-trivial neighbourhood of sample points, record the
    /// neighbours indices in this vector. Along with the vector of neighbours, store the index of
    /// the original query point, because this vector is a sparse subset of all the query points.
    points: Vec<(usize, Vec<usize>)>,

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
    pub(crate) fn cached_neighbour_points(&self) -> &[(usize, Vec<usize>)] {
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

    /// Compute neighbour cache if it hasn't been computed yet it is invalid. Return the neighbours
    /// of the given query points. Note that the cache must be invalidated explicitly, there is no
    /// real way to automatically cache results because both: query points and sample points may
    /// change slightly, but we expect the neighbourhood information to remain the same.
    pub(crate) fn neighbour_points<'a, I, N>(
        &mut self,
        query_points: &[[f64; 3]],
        neigh: N,
    ) -> &[(usize, Vec<usize>)]
    where
        I: Iterator<Item = Sample<f64>> + 'a,
        N: Fn([f64; 3]) -> I + Sync + Send,
    {
        if !self.is_valid() {
            self.points.clear();
            self.points.reserve(query_points.len());

            for (qi, q) in query_points.iter().enumerate() {
                let neighbours_iter = neigh(*q);
                // Below we try to reuse the allocated memory by previously cached members for
                // points.

                // Cache points
                for (iter_count, ni) in neighbours_iter.map(|op| op.index).enumerate() {
                    if iter_count == 0 {
                        // Initialize entry if there are any neighbours.
                        self.points.push((qi, Vec::new()));
                    }

                    debug_assert!(!self.points.is_empty());
                    let last_mut = self.points.last_mut().unwrap();

                    last_mut.1.push(ni as usize);
                }
            }

            self.valid = true;
        }

        self.cached_neighbour_points()
    }
}
