use super::samples::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use utils::soap::{IntoData, Scalar, Vector3};

/// Implement the trait required for `Sample` to live inside an `RTree`.
impl<T: Scalar> RTreeObject for Sample<T> {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.pos.cast::<f64>().into())
    }
}

impl<T: Scalar> PointDistance for Sample<T> {
    fn distance_2(&self, point: &[f64; 3]) -> f64 {
        (Vector3::from(*point) - self.pos.cast::<f64>()).norm_squared()
    }

    fn contains_point(&self, point: &[f64; 3]) -> bool {
        *point == self.pos.cast::<f64>().into_data()
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
pub fn build_rtree_from_samples<T: Real>(samples: &Samples<T>) -> RTree<Sample<T>>
where
    Sample<T>: RTreeObject,
{
    RTree::bulk_load(samples.iter().collect::<Vec<_>>())
}
