use super::samples::*;
use geo::math::Vector3;
use geo::Real;
use rstar::{RTree, AABB, RTreeObject, PointDistance};

/// Implement the trait required for `Sample` to live inside an `RTree`.
impl<T: Real> RTreeObject for Sample<T> {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.pos.cast::<f64>().unwrap().into())
    }
}

impl<T: Real> PointDistance for Sample<T> {
    fn distance_2(&self, point: &[f64; 3]) -> f64 {
        (Vector3(*point) - self.pos.cast::<f64>().unwrap()).norm_squared()
    }

    fn contains_point(&self, point: &[f64; 3]) -> bool {
        *point == self.pos.cast::<f64>().unwrap().into_inner()
    }
    fn distance_2_if_less_or_equal(&self, point: &[f64; 3], max_distance_2: f64) -> Option<f64> {
        let distance_2 = self.distance_2(point);
        if distance_2 <= max_distance_2 {
            Some(distance_2)
        } else {
            None
        }
    }
}

/// Build an rtree from a set of samples.
pub fn build_rtree_from_samples<T: Real + Send + Sync>(samples: &Samples<T>) -> RTree<Sample<T>>
where
    Sample<T>: RTreeObject,
{
    let mut rtree = RTree::new();
    for pt in samples.iter() {
        rtree.insert(pt);
    }
    rtree
}
