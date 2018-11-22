use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::VertexIndex, VertexMesh};
use spade::{rtree::RTree, BoundingRect, SpatialObject};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrientedPoint {
    /// Index within the original mesh where this point came from.
    pub index: i32,
    /// Positions of the point in space.
    pub pos: Vector3<f64>,
    /// Orientation of the point.
    pub nml: Vector3<f64>,
}

/// Implement the trait required for `OrientedPoint` to live inside an `RTree`.
impl SpatialObject for OrientedPoint {
    type Point = [f64; 3];

    fn mbr(&self) -> BoundingRect<Self::Point> {
        BoundingRect::from_point(self.pos.into())
    }

    fn distance2(&self, point: &Self::Point) -> f64 {
        (Vector3(*point) - self.pos).norm_squared()
    }
}

/// Create an iterator of oriented points given a slice of `points` and a slice of `normals`. The
/// index in the resulting oriented points refer to their position inside these slices, which are
/// expected to be of the same size.
pub fn oriented_points_iter<'a, V3>(
    points: &'a [V3],
    normals: &'a [V3],
) -> impl Iterator<Item = OrientedPoint> + Clone + 'a
where
    V3: Into<Vector3<f64>> + Copy,
{
    debug_assert_eq!(points.len(), normals.len());
    normals
        .iter()
        .zip(points.iter())
        .enumerate()
        .map(|(i, (&nml, &pos))| OrientedPoint {
            index: i as i32,
            pos: pos.into(),
            nml: nml.into(),
        })
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

/// Given an iterator over `OrientedPoint`s, produce an `RTree` populated with these points.
/// The iterator can be generated using the `oriented_points_iter` function.
pub fn build_rtree(
    oriented_points_iter: impl Iterator<Item = OrientedPoint>,
) -> RTree<OrientedPoint> {
    let mut rtree = RTree::new();
    for pt in oriented_points_iter {
        rtree.insert(pt);
    }
    rtree
}
