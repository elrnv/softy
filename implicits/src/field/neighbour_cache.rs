use super::{samples::Sample, SampleType};
use geo::Real;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub(crate) fn compute_closest_set<'a, T, C>(
    query_points: &[[T; 3]],
    closest: C,
    set: &mut Vec<usize>,
) -> bool
where
    T: Real + Send + Sync + 'a,
    C: Fn([T; 3]) -> &'a Sample<T> + Send + Sync,
{
    let mut changed = false;

    // Allocate additional neighbourhoods to match the size of query_points.
    changed |= query_points.len() != set.len();
    set.resize(query_points.len(), 0);

    changed |= query_points
        .par_iter()
        .zip(set.par_iter_mut())
        .map(|(q, sample_idx)| {
            let closest_index = closest(*q).index;
            let changed = *sample_idx != closest_index;
            *sample_idx = closest_index;
            changed
        })
        .reduce(|| false, |a, b| a || b);

    changed
}

/// There are three types of neighbourhoods for each query point:
///   1. closest samples to each query point, which we call the *closest set*,
///   1. the set of samples within a distance of the query point, which we call the *trivial
///      set* and
///   2. an extended set of samples reachable via triangles adjacent to the trivial set, which we
///      dub the *extended set*.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Neighbourhood {
    /// The closest sample to each query point.
    closest_set: Vec<usize>,
    /// The trivial neighbourhood of a set of query points. These are simply the set of samples that
    /// are within a certain distance from each query point.
    trivial_set: Vec<Vec<usize>>,
    /// The extended neighbourhood of a set of query points. This is a superset of the trivial
    /// neighbourhood that includes samples that are topologically adjacent to the trivial set.
    extended_set: Vec<Vec<usize>>,
}

impl Neighbourhood {
    pub(crate) fn new() -> Self {
        Neighbourhood {
            closest_set: Vec::new(),
            trivial_set: Vec::new(),
            extended_set: Vec::new(),
        }
    }

    /// Compute closest point set if it is empty (e.g. hasn't been created yet). Return `true` if
    /// the closest set has changed and `false` otherwise.
    pub(crate) fn compute_closest_set<'a, T, C>(
        &mut self,
        query_points: &[[T; 3]],
        closest: C,
    ) -> bool
    where
        T: Real + Send + Sync + 'a,
        C: Fn([T; 3]) -> &'a Sample<T> + Send + Sync,
    {
        compute_closest_set(query_points, closest, &mut self.closest_set)
    }

    /// Getter for the closest set of samples.
    pub(crate) fn closest_set(&self) -> &[usize] {
        &self.closest_set
    }

    /// Simple hasher for slices of unsigned integers. This is useful for doing approximate but
    /// efficient equality comparisons.
    fn compute_hash(v: &[usize]) -> u64 {
        // For efficient comparisons:
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut s = DefaultHasher::new();
        v.hash(&mut s);
        s.finish()
    }

    /// Compute neighbour cache if it is invalid (or hasn't been created yet). Return `true` if the
    /// trivial set of neighbours has changed and `false` otherwise.
    pub(crate) fn compute_trivial_set<'a, T, I, N>(
        &mut self,
        query_points: &[[T; 3]],
        neigh: N,
    ) -> bool
    where
        T: Real + Send + Sync,
        I: Iterator<Item = Sample<T>> + 'a,
        N: Fn([T; 3]) -> I + Sync + Send,
    {
        let cache = &mut self.trivial_set;
        let mut changed = false;

        // Allocate additional neighbourhoods to match the size of query_points.
        changed |= query_points.len() != cache.len();
        cache.resize(query_points.len(), Vec::new());

        changed |= query_points
            .par_iter()
            .zip(cache.par_iter_mut())
            .map(|(q, pts)| {
                let init_hash = Self::compute_hash(&pts);
                pts.clear();
                pts.extend(neigh(*q).map(|op| op.index));

                // This way of detecting changes has false positives, but for the purposes of
                // simulation, this should affect a negligible number of frames.
                // TODO: find percentage of false positives
                init_hash != Self::compute_hash(&pts)
            })
            .reduce(|| false, |a, b| a || b);

        changed
    }

    pub(crate) fn trivial_set(&self) -> &[Vec<usize>] {
        &self.trivial_set
    }

    /// Compute neighbour cache. Return `true` if the
    /// extended set of neighbours of the given query points has changed and `false` otherwise.
    /// Note that this function requires the trival
    /// set to already be computed. If trivial set has not been previously computed, use the
    /// `compute_neighbourhoods` function to compute both. This fuction will return `None` if the
    /// trivial set is invalid.
    pub(crate) fn compute_extended_set<T>(
        &mut self,
        query_points: &[[T; 3]],
        tri_topo: &[[usize; 3]],
        dual_topo: &[Vec<usize>],
        sample_type: SampleType,
    ) -> Option<bool>
    where
        T: Real + Send + Sync,
    {
        let Neighbourhood {
            trivial_set,
            extended_set,
            ..
        } = self;

        if trivial_set.len() != query_points.len() {
            return None;
        }

        let mut changed = false;

        // Allocate additional neighbourhoods to match the size of query_points.
        changed |= query_points.len() != extended_set.len();
        extended_set.resize(query_points.len(), Vec::new());

        changed |= trivial_set
            .par_iter()
            .zip(extended_set.par_iter_mut())
            .map(|(triv_pts, ext_pts)| {
                let init_hash = Self::compute_hash(&ext_pts);
                ext_pts.clear();
                let mut set: BTreeSet<_> = triv_pts.iter().collect();
                if sample_type == SampleType::Vertex && !dual_topo.is_empty() {
                    set.extend(triv_pts.iter().flat_map(|&pt| {
                        dual_topo[pt].iter().flat_map(|&tri| tri_topo[tri].iter())
                    }));
                }
                ext_pts.extend(set.into_iter());

                // TODO: check for rate of false positives.
                init_hash != Self::compute_hash(&ext_pts)
            })
            .reduce(|| false, |a, b| a || b);

        Some(changed)
    }

    /// Compute neighbouroods. Return `true` if neighbourhood sparsity has changed and `false`
    /// otherwise.
    pub(crate) fn compute_neighbourhoods<'a, T, I, N, C>(
        &mut self,
        query_points: &[[T; 3]],
        neigh: N,
        closest: C,
        tri_topo: &[[usize; 3]],
        dual_topo: &[Vec<usize>],
        sample_type: SampleType,
    ) -> bool
    where
        T: Real + Send + Sync + 'a,
        I: Iterator<Item = Sample<T>> + 'a,
        N: Fn([T; 3]) -> I + Sync + Send,
        C: Fn([T; 3]) -> &'a Sample<T> + Send + Sync,
    {
        self.compute_closest_set(query_points, closest)
            | self.compute_trivial_set(query_points, neigh)
            | self
                .compute_extended_set(query_points, tri_topo, dual_topo, sample_type)
                .unwrap()
    }

    pub(crate) fn extended_set(&self) -> &[Vec<usize>] {
        &self.extended_set
    }
}
