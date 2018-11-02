extern crate geometry as geo;
extern crate hrbf;
extern crate nalgebra as na;
extern crate rayon;
extern crate spade;

#[cfg(test)]
#[macro_use]
extern crate approx;

use geo::mesh::{attrib, topology::*, Attrib, PolyMesh};
use na::{dot, Vector3};
use rayon::prelude::*;
use spade::{rtree::RTree, BoundingRect, SpatialObject};

#[macro_use]
pub mod zip;

#[derive(Copy, Clone, Debug)]
pub enum Kernel {
    Interpolating { radius: f64 },
    Approximate { radius: f64, tolerance: f64 },
    Cubic { radius: f64 },
    Global { tolerance: f64 },
    Hrbf,
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

macro_rules! hrbf {
    ($samples:ident, $oriented_points:ident, $offsets:ident, $interrupt:ident) => {
        let sample_pos = $samples.vertices();

        let chunk_size = 5000;

        let pts: Vec<na::Point3<f64>> = $oriented_points
            .iter()
            .map(|op| na::Point3::from(op.pos))
            .collect();
        let nmls: Vec<na::Vector3<f64>> = $oriented_points
            .iter()
            .map(|op| na::Vector3::from(op.nml))
            .collect();
        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());
        hrbf.fit_offset(&pts, &$offsets, &nmls);

        for (q_chunk, potential_chunk) in sample_pos.chunks(chunk_size).zip(
            $samples
                .attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(chunk_size),
        ) {
            if $interrupt() {
                break;
            }

            q_chunk
                .par_iter()
                .zip(potential_chunk.par_iter_mut())
                .for_each(|(q, potential)| {
                    *potential = hrbf.eval(na::Point3::from(*q)) as f32;
                });
        }
    };
}

macro_rules! mls {
    ($samples:ident, $kernel:ident, $radius:ident, $compute_neighbours:ident, $interrupt:ident,
     $num_neighs_attrib_data:ident, $neighs_attrib_data:ident, $weight_attrib_data:ident) => {
        let sample_pos = $samples.vertices();

        let chunk_size = 5000;

        for (q_chunk, num_neighs_chunk, neighs_chunk, weight_chunk, potential_chunk) in zip!(
            sample_pos.chunks(chunk_size),
            $num_neighs_attrib_data.chunks_mut(chunk_size),
            $neighs_attrib_data.chunks_mut(chunk_size),
            $weight_attrib_data.chunks_mut(chunk_size),
            $samples
                .attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(chunk_size)
        ) {
            if $interrupt() {
                break;
            }

            zip!(
                q_chunk.par_iter(),
                num_neighs_chunk.par_iter_mut(),
                neighs_chunk.par_iter_mut(),
                weight_chunk.par_iter_mut(),
                potential_chunk.par_iter_mut()
            )
            .for_each(|(q, num_neighs, neighs, weight, potential)| {
                let neighbours_vec: Vec<&OrientedPoint> = $compute_neighbours(*q);
                let neighbours = &neighbours_vec;

                *num_neighs = neighbours.len() as i32;

                for (k, neigh) in neighbours_vec.iter().enumerate() {
                    if k >= 11 {
                        break;
                    }
                    neighs[k] = neigh.index as i32;
                }

                if !neighbours.is_empty() {
                    let n = neighbours.len() + 1;
                    let mut closest_d = $radius as f64;
                    for nbr in neighbours.iter() {
                        closest_d = closest_d.min(dist(*q, nbr.pos));
                    }
                    let weights = (0..n)
                        .map(|i| {
                            if i == neighbours.len() {
                                let bg = [q[0] - $radius + closest_d, q[1], q[2]];
                                let w = $kernel(*q, bg, closest_d);
                                weight[11] = w as f32;
                                w
                            } else {
                                let w = $kernel(*q, neighbours[i].pos, closest_d);
                                if i < 11 {
                                    weight[i] = w as f32;
                                }
                                w
                            }
                        })
                        .collect::<Vec<f64>>();

                    let potentials = (0..n)
                        .map(|i| {
                            if i == neighbours.len() {
                                *potential as f64
                            } else {
                                dot(
                                    &Vector3::from(neighbours[i].nml),
                                    &(Vector3::from(*q) - Vector3::from(neighbours[i].pos)),
                                )
                            }
                        })
                        .collect::<Vec<f64>>();

                    let denominator: f64 = weights.iter().sum();
                    let numerator: f64 = weights
                        .iter()
                        .zip(potentials.iter())
                        .map(|(w, p)| w * p)
                        .sum();

                    if denominator != 0.0 {
                        *potential = (numerator / denominator) as f32;
                    }
                }
            });
        }
    };
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrientedPoint {
    pub index: i32,
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
        x * x + y * y + z * z
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

    1.0 - 3.0 * r * r / (radius * radius) + 2.0 * r * r * r / (radius * radius * radius)
}

fn local_interpolating_kernel(x: [f64; 3], p: [f64; 3], radius: f64, closest_d: f64) -> f64 {
    let r = dist(x, p);
    if r > radius {
        return 0.0;
    }

    let envelope = local_cubic_kernel(r, radius);

    let s = r / radius;
    let sc = closest_d / radius;
    envelope * sc * sc * (1.0 / (s * s) - 1.0)
}

fn local_approximate_kernel(r: f64, radius: f64, tolerance: f64) -> f64 {
    if r > radius {
        return 0.0;
    }

    let eps = tolerance;

    let w = |d| {
        let ddeps = 1.0 / (d * d + eps);
        let epsp1 = 1.0 / (1.0 + eps);
        (ddeps * ddeps - epsp1 * epsp1) / (1.0 / (eps * eps) - epsp1 * epsp1)
    };

    w(r / radius) // /denom
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (Vector3::from(a) - Vector3::from(b)).norm()
}

pub fn compute_potential<F>(
    samples: &mut PolyMesh<f64>,
    surface: &mut PolyMesh<f64>,
    params: Params,
    interrupt: F,
) -> Result<(), Error>
where
    F: Fn() -> bool + Sync + Send,
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
        .enumerate()
        .map(|(i, (nml, &pos))| OrientedPoint {
            index: i as i32,
            pos,
            nml,
        })
        .collect();

    let offsets = surface
        .attrib_iter::<f32, VertexIndex>("offset")
        .map(|iter| iter.map(|&x| x as f64).collect())
        .unwrap_or(vec![0.0f64; surface.num_vertices()]);

    let rtree = {
        let mut rtree = RTree::new();
        for pt in oriented_points.iter() {
            rtree.insert(*pt);
        }
        rtree
    };

    let mut num_neighs_attrib_data = vec![0i32; samples.num_vertices()];
    let mut neighs_attrib_data = vec![[-1i32; 11]; samples.num_vertices()];
    let mut weight_attrib_data = vec![[0f32; 12]; samples.num_vertices()];

    match params.kernel {
        Kernel::Interpolating { radius } => {
            let radius2 = radius * radius;
            let neigh = |q| rtree.lookup_in_circle(&q, &radius2);
            let kern = |x, p, closest_dist| local_interpolating_kernel(x, p, radius, closest_dist);
            mls!(
                samples,
                kern,
                radius,
                neigh,
                interrupt,
                num_neighs_attrib_data,
                neighs_attrib_data,
                weight_attrib_data
            );
        }
        Kernel::Approximate { tolerance, radius } => {
            let radius2 = radius * radius;
            let neigh = |q| rtree.lookup_in_circle(&q, &radius2);
            let kern = |x, p, _| local_approximate_kernel(dist(x, p), radius, tolerance);
            mls!(
                samples,
                kern,
                radius,
                neigh,
                interrupt,
                num_neighs_attrib_data,
                neighs_attrib_data,
                weight_attrib_data
            );
        }
        Kernel::Cubic { radius } => {
            let radius2 = radius * radius;
            let neigh = |q| rtree.lookup_in_circle(&q, &radius2);
            let kern = |x, p, _| local_cubic_kernel(dist(x, p), radius);
            mls!(
                samples,
                kern,
                radius,
                neigh,
                interrupt,
                num_neighs_attrib_data,
                neighs_attrib_data,
                weight_attrib_data
            );
        }
        Kernel::Global { tolerance } => {
            let neigh = |_| oriented_points.iter().collect();
            let radius = 1.0;
            let kern = |x, p, _| global_inv_dist2_kernel(dist(x, p), tolerance);
            mls!(
                samples,
                kern,
                radius,
                neigh,
                interrupt,
                num_neighs_attrib_data,
                neighs_attrib_data,
                weight_attrib_data
            );
        }
        Kernel::Hrbf => {
            hrbf!(samples, oriented_points, offsets, interrupt);
        }
    }

    {
        samples.set_attrib_data::<_, VertexIndex>("num_neighbours", &num_neighs_attrib_data)?;
        samples.set_attrib_data::<_, VertexIndex>("neighbours", &neighs_attrib_data)?;
        samples.set_attrib_data::<_, VertexIndex>("weights", &weight_attrib_data)?;
    }

    if interrupt() {
        Err(Error::Interrupted)
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum Error {
    Interrupted,
    MissingNormals,
    Failure,
    IO(geo::io::Error),
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Self {
        match err {
            attrib::Error::TypeMismatch => Error::MissingNormals,
            _ => Error::Failure,
        }
    }
}

impl From<geo::io::Error> for Error {
    fn from(err: geo::io::Error) -> Self {
        Error::IO(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::io::load_polymesh;
    use std::path::PathBuf;

    /// Generate a [-1,1]x[-1,1] mesh grid with the given cell resolution.
    fn make_grid(nx: usize, ny: usize) -> PolyMesh<f64> {
        let mut positions = Vec::new();

        // iterate over vertices
        for i in 0..nx + 1 {
            for j in 0..ny + 1 {
                positions.push([
                    -1.0 + 2.0 * (i as f64) / nx as f64,
                    -1.0 + 2.0 * (j as f64) / ny as f64,
                    0.0,
                ]);
            }
        }

        let mut indices = Vec::new();

        // iterate over faces
        for i in 0..nx {
            for j in 0..ny {
                indices.push(4);
                indices.push((nx + 1) * j + i);
                indices.push((nx + 1) * j + i + 1);
                indices.push((nx + 1) * (j + 1) + i + 1);
                indices.push((nx + 1) * (j + 1) + i);
            }
        }

        let mut mesh = PolyMesh::new(positions, &indices);
        mesh.add_attrib::<_, VertexIndex>("potential", 0.0f32)
            .unwrap();
        mesh
    }

    #[test]
    fn approximate_kernel_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let points = vec![
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
        ];

        let normals = vec![
            [-1.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ];

        let mut sphere = PolyMesh::new(points, &vec![]);
        sphere.add_attrib_data::<_, VertexIndex>("N", normals)?;
        compute_potential(
            &mut grid,
            &mut sphere,
            Params {
                kernel: Kernel::Approximate {
                    tolerance: 0.00001,
                    radius: 1.5,
                },
            },
            || false,
        )?;
        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> = load_polymesh(&PathBuf::from(
            "assets/approximate_sphere_test_grid_expected.vtk",
        ))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }
}
