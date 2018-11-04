//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use rayon::prelude::*;
use spade::{rtree::RTree, BoundingRect, SpatialObject};
use geo::mesh::{topology::*, Attrib, PolyMesh};
use geo::math::{Vector3};
use kernels::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrientedPoint {
    pub index: i32,
    pub pos: Vector3<f64>,
    pub nml: Vector3<f64>,
}

impl SpatialObject for OrientedPoint {
    type Point = [f64; 3];

    fn mbr(&self) -> BoundingRect<Self::Point> {
        BoundingRect::from_point(self.pos.into())
    }

    fn distance2(&self, point: &Self::Point) -> f64 {
        (Vector3(*point) - self.pos).norm_squared()
    }
}

pub fn oriented_points_iter<'a>(points: &'a [[f64;3]], normals: &'a [[f64;3]])
    -> impl Iterator<Item=OrientedPoint> + Clone + 'a
{
    normals.iter().zip(points.iter())
        .enumerate()
        .map(|(i, (&nml, &pos))| OrientedPoint {
            index: i as i32,
            pos: pos.into(),
            nml: nml.into(),
        })
}

pub fn build_rtree(oriented_points: &[OrientedPoint]) -> RTree<OrientedPoint> {
    let mut rtree = RTree::new();
    for pt in oriented_points.iter() {
        rtree.insert(*pt);
    }
    rtree
}

fn dist(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    (a - b).norm()
}

pub struct ImplicitSurface {
    kernel: Kernel,
    spatial_tree: RTree<OrientedPoint>,
    oriented_points: Vec<OrientedPoint>,
    /// Potential values at the interpolating points. These offsets indicate the values to
    /// match by interpolating implicit surfaces. This means that the zero iso-surface will ot
    /// necessarily pass through the given points.
    offsets: Vec<f64>,
}

impl ImplicitSurface {
    /// Create a new implicit surface struct.
    pub fn new(kernel: Kernel, points: &[[f64;3]], normals: &[[f64;3]]) -> ImplicitSurface {
        assert_eq!(points.len(), normals.len());
        let oriented_points: Vec<OrientedPoint> = oriented_points_iter(points, normals).collect();
        ImplicitSurface {
            kernel,
            spatial_tree: build_rtree(&oriented_points),
            oriented_points,
            offsets: vec![0.0f64; points.len()]
        }
    }

    pub fn with_offsets(kernel: Kernel, points: &[[f64;3]], normals: &[[f64;3]], offsets: &[f64]) -> ImplicitSurface {
        ImplicitSurface {
            offsets: offsets.to_vec(),
            ..ImplicitSurface::new(kernel, points, normals)
        }
    }

    /// Compute the implicit surface potential on the given polygon mesh.
    pub fn compute_potential_on_mesh<F>(&self, mesh: &mut PolyMesh<f64>, interrupt: F) -> Result<(), super::Error>
        where F: Fn() -> bool + Sync + Send
    {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            ref oriented_points,
            ref offsets,
        } = *self;

        // Initialize potential with zeros.
        {
            mesh.attrib_or_add::<_, VertexIndex>("potential", 0.0f32)?;
        }

        match *kernel {
            Kernel::Interpolating { radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, closest_dist| local_interpolating_kernel(dist(x, p), radius, closest_dist);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            Kernel::Approximate { tolerance, radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| local_approximate_kernel(dist(x, p), radius, tolerance);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            Kernel::Cubic { radius } => {
                let radius2 = radius * radius;
                let neigh = |q| spatial_tree.lookup_in_circle(&q, &radius2);
                let kern = |x, p, _| local_cubic_kernel(dist(x, p), radius);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            Kernel::Global { tolerance } => {
                let neigh = |_| oriented_points.iter().collect();
                let radius = 1.0;
                let kern = |x, p, _| global_inv_dist2_kernel(dist(x, p), tolerance);
                Self::compute_mls_on_mesh(mesh, radius, kern, neigh, interrupt)
            }
            Kernel::Hrbf => {
                Self::compute_hrbf_on_mesh(mesh, oriented_points, offsets, interrupt)
            }
        }
    }

    /// Given a slice of query points, compute the potential at each point.
    pub fn compute_potential(&self, _query_points: &[[f64;3]], _out_potential: &mut [f64]) {
    }

    /// Compute the implicit surface potential. The `interrupt` callback will be called
    /// intermittently and it returns true, the computation will halt.
    pub fn compute_potential_interruptable<F>(&self, _query_points: &[[f64;3]], _interrupt: F, _out_potential: &mut [f64])
        where F: Fn() -> bool + Sync + Send
    {
    }

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_mls_on_mesh<'a, K, N, F>(mesh: &mut PolyMesh<f64>, radius: f64, kernel: K, neigh: N, interrupt: F)
    -> Result<(), super::Error>
        where K: Fn(Vector3<f64>, Vector3<f64>, f64) -> f64 + Sync + Send,
              N: Fn([f64;3]) -> Vec<&'a OrientedPoint> + Sync + Send,
              F: Fn() -> bool + Sync + Send,
    {
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut weight_attrib_data = vec![[0f32; 12]; mesh.num_vertices()];
        let sample_pos = mesh.vertex_positions().to_vec();

        let chunk_size = 5000;

        for (q_chunk, num_neighs_chunk, neighs_chunk, weight_chunk, potential_chunk) in zip!(
            sample_pos.chunks(chunk_size),
            num_neighs_attrib_data.chunks_mut(chunk_size),
            neighs_attrib_data.chunks_mut(chunk_size),
            weight_attrib_data.chunks_mut(chunk_size),
            mesh.attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(chunk_size)
        ) {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            zip!(
                q_chunk.par_iter(),
                num_neighs_chunk.par_iter_mut(),
                neighs_chunk.par_iter_mut(),
                weight_chunk.par_iter_mut(),
                potential_chunk.par_iter_mut()
            )
            .for_each(|(q, num_neighs, neighs, weight, potential)| {
                let neighbours_vec: Vec<&OrientedPoint> = neigh(*q);
                let neighbours = &neighbours_vec;

                // Record number of neighbours in total.
                *num_neighs = neighbours.len() as i32;

                // Record up to 11 neighbours
                for (k, neigh) in neighbours_vec.iter().enumerate() {
                    if k >= 11 {
                        break;
                    }
                    neighs[k] = neigh.index as i32;
                }

                if !neighbours.is_empty() {
                    let n = neighbours.len() + 1;
                    let mut closest_d = radius;
                    for nbr in neighbours.iter() {
                        closest_d = closest_d.min(dist(Vector3(*q), nbr.pos));
                    }
                    let weights = (0..n)
                        .map(|i| {
                            if i == neighbours.len() {
                                let bg = [q[0] - radius + closest_d, q[1], q[2]];
                                let w = kernel(Vector3(*q), bg.into(), closest_d);
                                // Record the last weight for the background potential
                                weight[11] = w as f32;
                                w
                            } else {
                                let w = kernel(Vector3(*q), neighbours[i].pos, closest_d);
                                if i < 11 {
                                    // Record the current weight
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
                                neighbours[i].nml.dot(Vector3(*q) - neighbours[i].pos)
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

        {
            mesh.set_attrib_data::<_, VertexIndex>("num_neighbours", &num_neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("neighbours", &neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("weights", &weight_attrib_data)?;
        }

        Ok(())
    }

    fn compute_hrbf_on_mesh<F>(mesh: &mut PolyMesh<f64>, oriented_points: &[OrientedPoint], offsets: &[f64], interrupt: F)
        -> Result<(), super::Error>
    where F: Fn() -> bool + Sync + Send
    {
        let sample_pos = mesh.vertex_positions().to_vec();

        let chunk_size = 5000;

        let pts: Vec<crate::na::Point3<f64>> = oriented_points.iter()
            .map(|op| crate::na::Point3::from(op.pos.into_inner()))
            .collect();
        let nmls: Vec<crate::na::Vector3<f64>> = oriented_points.iter()
            .map(|op| crate::na::Vector3::from(op.nml.into_inner()))
            .collect();
        let mut hrbf = hrbf::HRBF::<f64, hrbf::Pow3<f64>>::new(pts.clone());
        hrbf.fit_offset(&pts, offsets, &nmls);

        for (q_chunk, potential_chunk) in sample_pos.chunks(chunk_size).zip(
            mesh
                .attrib_as_mut_slice::<f32, VertexIndex>("potential")
                .unwrap()
                .chunks_mut(chunk_size),
        ) {
            if interrupt() {
                return Err(super::Error::Interrupted);
            }

            q_chunk
                .par_iter()
                .zip(potential_chunk.par_iter_mut())
                .for_each(|(q, potential)| {
                    *potential = hrbf.eval(crate::na::Point3::from(*q)) as f32;
                });
        }

        Ok(())
    }
}
