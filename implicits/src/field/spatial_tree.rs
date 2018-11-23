use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::VertexIndex, VertexMesh};
use spade::{SpadeNum, rtree::RTree, BoundingRect, SpatialObject};
use super::samples::*;

/// Implement the trait required for `Sample` to live inside an `RTree`.
impl<T: SpadeNum> SpatialObject for Sample<T> {
    type Point = [T; 3];

    fn mbr(&self) -> BoundingRect<Self::Point> {
        BoundingRect::from_point(self.pos.into())
    }

    fn distance2(&self, point: &Self::Point) -> T {
        (Vector3(*point) - self.pos).norm_squared()
    }
}

/// Given a vertex based mesh, extract the vertex positions (points) and vertex normals and return
/// them in as two vectors in that order.
pub fn points_and_normals_from_mesh<'a, M: VertexMesh<f64>>(
    mesh: &M,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let points: Vec<Vector3<f64>> = mesh
        .vertex_position_iter()
        .map(|&x| -> Vector3<f64> {
            Vector3(x)
                .cast::<f64>()
                .expect("Failed to convert positions to f64")
        })
        .collect();

    let normals = mesh.attrib_iter::<[f32; 3], VertexIndex>("N").ok().map_or(
        vec![Vector3::zeros(); points.len()],
        |iter| {
            iter.map(|&nml| Vector3(nml).cast::<f64>().unwrap())
                .collect()
        },
    );

    (points, normals)
}

/// Build an rtree from a set of samples.
pub fn build_rtree_from_samples<T: SpadeNum>(
    samples: &Samples<T>,
) -> RTree<Sample<T>> {
    let mut rtree = RTree::new();
    for pt in samples.iter() {
        rtree.insert(pt);
    }
    rtree
}
