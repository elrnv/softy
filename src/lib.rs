extern crate geometry as geo;
extern crate hrbf;
extern crate nalgebra as na;
extern crate rayon;
extern crate spade;

#[cfg(test)]
#[macro_use]
extern crate approx;

use geo::mesh::{attrib, topology::*, Attrib, PolyMesh};
use geo::math::{Vector3};

#[macro_use]
pub mod zip;

pub mod implicit_surface;
mod kernels;

pub use implicit_surface::*;
pub use kernels::Kernel;

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub kernel: Kernel,
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
    // Check that we have normals
    let points = surface.vertex_positions();
    let normals: Vec<[f64;3]> = surface.attrib_iter::<[f32; 3], VertexIndex>("N")?
        .map(|nml| Vector3(*nml).cast::<f64>().unwrap().into()).collect();
    let offsets = surface
        .attrib_iter::<f32, VertexIndex>("offset")
        .map(|iter| iter.map(|&x| x as f64).collect())
        .unwrap_or(vec![0.0f64; samples.num_vertices()]);

    let implicit_surface = ImplicitSurface::with_offsets(params.kernel, points, &normals, &offsets);

    implicit_surface.compute_potential_on_mesh(samples, interrupt)?;
    Ok(())
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
        use geo::math::Vector3;

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

        let mut sphere = PolyMesh::new(points.clone(), &vec![]);
        sphere.add_attrib_data::<_, VertexIndex>("N", normals.clone())?;

        {
            let grid_points = grid.vertex_positions().to_vec();
            let potential_iter_mut = grid.attrib_iter_mut::<f32, VertexIndex>("potential").unwrap();
            for (q, pot) in zip!(grid_points.iter().map(|&x| Vector3(x)), potential_iter_mut) {
                let (p, nml) = points.iter().zip(normals.iter()).map(|(&p,&n)| (Vector3(p), Vector3(n)))
                    .min_by(|&(a,_), &(b,_)| (a-q).norm().partial_cmp(&(b-q).norm()).unwrap()).unwrap();
                *pot = Vector3([nml[0] as f64, nml[1] as f64, nml[2] as f64]).dot(q - p) as f32;
            }
        }

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
