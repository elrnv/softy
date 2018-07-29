extern crate spade;
extern crate rayon;
extern crate nalgebra as na;
extern crate geometry as geo;

use rayon::prelude::*;
use geo::mesh::{attrib, topology::*, Attrib, PolyMesh};
use na::{dot, DMatrix, DVector, Dynamic, Matrix, MatrixVec, U1, Vector3};
use spade::{SpatialObject, BoundingRect, rtree::RTree};

type BasisMatrix = Matrix<f64, Dynamic, U1, MatrixVec<f64, Dynamic, U1>>;

#[derive(Copy, Clone, Debug)]
pub enum Kernel {
    Interpolating { radius: f64, tolerance: f64 },
    Cubic { radius: f64 },
    Global { tolerance: f64 },
}

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub kernel: Kernel,
}

//pub fn build_triangle_rtree(polymesh: &PolyMesh<f64>) -> Result<RTree<SimpleTriangle>, Error> {
//
//    let pos = triangles.vertices();
//
//    let triangles = polymesh.face_iter().map(|face| {
//        if face.len() != 3 {
//            return Err(Error::InvalidTriangleMesh);
//        }
//
//        SimpleTriangle::new(pos[face[0]], pos[face[1]], pos[face[2]])
//    });
//
//    Ok(RTree::bulk_load(triangles))
//}

macro_rules! mls {
    ($samples:ident, $kernel:ident, $compute_neighbours:ident, $interrupt:ident) => {
        let sample_pos = $samples.vertices();

        let chunk_size = 5000;

        for (q_chunk, potential_chunk) in sample_pos.chunks(chunk_size).zip(
            $samples
                .attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(chunk_size)) {

            if $interrupt() {
                break;
            }

            q_chunk.par_iter().zip(potential_chunk.par_iter_mut()).for_each(|(q, potential)| {
                let neighbours_vec: Vec<&OrientedPoint> = $compute_neighbours(*q);
                let neighbours = &neighbours_vec;

                let n = neighbours.len();

                let B = BasisMatrix::from_element(n, 1.0);
                let W = DMatrix::from_fn(n, n, |i, j| {
                    if i == j { $kernel(*q, neighbours[i].pos, neighbours) } else { 0.0 }
                });
                let S = DVector::from_fn(n, |i, _| {
                    dot(
                        &Vector3::from(neighbours[i].nml),
                        &(Vector3::from(*q) - Vector3::from(neighbours[i].pos)),
                    )
                });

                let BTW = B.transpose() * W;
                let A = &BTW * &B;
                let b = &BTW * &S;

                if let Some(c) = A.lu().solve(&b) {
                    *potential = c[0] as f32;
                }
            });
        }
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrientedPoint {
    pub pos: [f64; 3],
    pub nml: [f64; 3],
}

impl SpatialObject for OrientedPoint {
    type Point = [f64; 3];

    fn mbr(&self) -> BoundingRect<Self::Point> {
        BoundingRect::from_point(self.pos)
    }

    fn distance2(&self, point: &[f64; 3]) -> f64 {
        let x = point[0] - self.pos[0];
        let y = point[1] - self.pos[1];
        let z = point[2] - self.pos[2];
        x*x + y*y + z*z
    }
}

fn global_inv_dist2_kernel(r: f64, epsilon: f64) -> f64 {
    let w = 1.0 / (r * r + epsilon * epsilon);
    w * w
}

fn local_cubic_kernel(r: f64, radius: f64) -> f64 {
    if r > radius {
        return 0.0;
    }

    1.0 - 3.0*r*r/(radius*radius) + 2.0*r*r*r/(radius*radius*radius)
}

fn local_interpolating_kernel<I>(r: f64, radius: f64, distances: I, tolerance: f64) -> f64
where I: Iterator<Item=f64>,
{
    if r > radius {
        return 0.0;
    }

    let eps = tolerance;

    let w = |d| {
        let ddeps = 1.0/(d*d + eps);
        let epsp1 = 1.0/(1.0 + eps);
        (ddeps*ddeps - epsp1*epsp1)/(1.0/(eps*eps) - epsp1*epsp1)
    };

    let mut denom = 0.0;
    for d in distances {
        denom += w(d/radius);
    }

    w(r/radius)/denom
}

fn dist(a: [f64;3], b: [f64;3]) -> f64 {
    (Vector3::from(a) - Vector3::from(b)).norm()
}

#[allow(non_snake_case)]
pub fn compute_mls<F>(
    samples: &mut PolyMesh<f64>,
    surface: &mut PolyMesh<f64>,
    params: Params,
    interrupt: F,
) -> Result<(), Error> 
    where F: Fn() -> bool + Sync + Send,
{
    // Initialize potential with zeros.
    {
        samples.attrib_or_add::<_, VertexIndex>("potential", 0.0f32)?;
    }

    // Check that we have normals
    {
        surface.attrib_check::<[f32; 3], VertexIndex>("N")?;
    }

    let oriented_points: Vec<OrientedPoint> = surface
        .attrib_iter::<[f32; 3], VertexIndex>("N")?
        .map(|x| [x[0] as f64, x[1] as f64, x[2] as f64])
        .zip(surface.vertex_iter())
        .map(|(nml, &pos)| OrientedPoint { pos, nml }).collect();

    let rtree = {
        let mut rtree = RTree::new();
        for pt in oriented_points.iter() {
            rtree.insert(*pt);
        }
        rtree
    };

    match params.kernel {
        Kernel::Interpolating { tolerance, radius } => {
            let neigh = |q| rtree.lookup_in_circle(&q, &radius);
            let kern = |x, p, neighbours: &[&OrientedPoint]| {
                let distances = neighbours.iter().map(|&neigh| dist(neigh.pos, x));
                let d = dist(x, p);
                local_interpolating_kernel(d, radius, distances, tolerance)
            };
            mls!(samples, kern, neigh, interrupt);
        }
        Kernel::Cubic { radius } => {
            let neigh = |q| rtree.lookup_in_circle(&q, &radius);
            let kern = |x, p, _| {
                local_cubic_kernel(dist(x, p), radius)
            };
            mls!(samples, kern, neigh, interrupt);
        }
        Kernel::Global { tolerance } => {
            let neigh = |_| oriented_points.iter().collect();
            let kern = |x, p, _| {
                global_inv_dist2_kernel(dist(x, p), tolerance)
            };
            mls!(samples, kern, neigh, interrupt);
        }
    }

    if interrupt() {
        Err(Error::Interrupted)
    } else {
        Ok(())
    }
}

pub enum Error {
    Interrupted,
    MissingNormals,
    Failure,
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Self {
        match err {
            attrib::Error::TypeMismatch => Error::MissingNormals,
            _ => Error::Failure,
        }
    }
}
