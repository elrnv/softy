use super::samples::*;
use geo::math::Vector3;
use geo::Real;
use spade::{rtree::RTree, BoundingRect, SpatialObject};

/// Implement the trait required for `Sample` to live inside an `RTree`.
impl<T: Real> SpatialObject for Sample<T> {
    type Point = [f64; 3];

    fn mbr(&self) -> BoundingRect<Self::Point> {
        BoundingRect::from_point(self.pos.cast::<f64>().unwrap().into())
    }

    fn distance2(&self, point: &Self::Point) -> f64 {
        (Vector3(*point) - self.pos.cast::<f64>().unwrap()).norm_squared()
    }
}

/// Build an rtree from a set of samples.
pub fn build_rtree_from_samples<T: Real + Send + Sync>(samples: &Samples<T>) -> RTree<Sample<T>>
    where Sample<T>: SpatialObject
{
    let mut rtree = RTree::new();
    for pt in samples.iter() {
        rtree.insert(pt);
    }
    rtree
}
