#![cfg_attr(feature = "unstable", feature(test))]
extern crate geometry as geo;
extern crate hrbf;
extern crate nalgebra as na;
extern crate rayon;
extern crate reinterpret;
extern crate spade;

#[cfg(test)]
#[macro_use]
extern crate approx;

use crate::geo::mesh::{attrib, PolyMesh, TriMesh};

#[macro_use]
pub mod zip;

pub mod field;
mod kernel;

pub use crate::field::*;
pub use crate::kernel::KernelType;

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub kernel: KernelType,
    pub background_potential: BackgroundPotentialType,
    pub sample_type: SampleType,
}

pub fn compute_potential<F>(
    query_points: &mut PolyMesh<f64>,
    surface: &mut PolyMesh<f64>,
    params: Params,
    interrupt: F,
) -> Result<(), Error>
where
    F: Fn() -> bool + Sync + Send,
{
    let mut builder = ImplicitSurfaceBuilder::new();

    builder.kernel(params.kernel);
    builder.background_potential(params.background_potential);

    match params.sample_type {
        SampleType::Vertex => {
            builder.vertex_samples_from_mesh(&TriMesh::from(surface.clone()));
        }
        SampleType::Face => {
            builder.face_samples_from_mesh(&TriMesh::from(surface.clone()));
        }
    }

    if let Some(implicit_surface) = builder.build() {
        implicit_surface.compute_potential_on_mesh(query_points, interrupt)?;
        Ok(())
    } else {
        Err(Error::Failure)
    }
}

#[derive(Debug)]
pub enum Error {
    Interrupted,
    MissingNormals,
    UnsupportedKernel,
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
    use crate::geo::io::load_polymesh;
    use crate::geo::mesh::topology::*;
    use crate::geo::mesh::*;
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

    fn make_sample_octahedron() -> TriMesh<f64> {
        let points = vec![
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
        ];

        let indices = vec![
            0, 5, 3, 4, 0, 3, 1, 4, 3, 5, 1, 3, 5, 0, 2, 0, 4, 2, 4, 1, 2, 1, 5, 2,
        ];

        TriMesh::new(points, indices)
    }

    /// Test the non-dynamic API.
    #[test]
    fn vertex_samples_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let trimesh = make_sample_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.5,
                },
                background_potential: BackgroundPotentialType::DistanceBased,
                sample_type: SampleType::Vertex,
            },
            || false,
        )?;

        //geo::io::save_polymesh(&grid, &PathBuf::from("mesh.vtk")).unwrap();

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> = load_polymesh(&PathBuf::from(
            "assets/octahedron_vertex_grid_expected.vtk",
        ))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }

    /// Face centered implicit surface test.
    #[test]
    fn face_samples_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let trimesh = make_sample_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.5,
                },
                background_potential: BackgroundPotentialType::DistanceBased,
                sample_type: SampleType::Face,
            },
            || false,
        )?;

        geo::io::save_polymesh(&grid, &PathBuf::from("mesh.vtk")).unwrap();

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> = load_polymesh(&PathBuf::from(
            "assets/octahedron_face_grid_expected.vtk",
        ))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }

    // TODO: Make this work. Need to generalize potential computation to type T: Real
    /*
    /// Test the derivatives of the dynamic API.
    #[test]
    fn meshless_approximate_kernel_derivatives_test() -> Result<(), Error> {
        use autodiff::F;
    
        let mut grid = make_grid(22, 22);
    
        let trimesh = make_sample_octahedron();
        let triangles: Vec<[usize;3]> = trimesh.face_iter().cloned().map(|x| x.into_inner()).collect();
        for cur_pt_idx in 0..trimesh.num_vertices() { // for each vertex
            for i in 0..3 { // for each component
    
                // Initialize autodiff variable to differentiate with respect to.
                let points: Vec<[F;3]> = trimesh.vertex_position_iter().enumerate().map(|(vtx_idx, pos)| {
                    let mut pos = [F::cst(pos[0]), F::cst(pos[1]), F::cst(pos[2])];
                    if vtx_idx == cur_pt_idx {
                        pos[i] = F::var(pos[i]);
                    }
                    pos
                }).collect();
    
                let implicit_surface = ImplicitSurfaceBuilder::new()
                    .kernel(
                        KernelType::Approximate {
                            tolerance: 0.00001,
                            radius: 1.5,
                        }
                    )
                    .background_potential(true)
                    .triangles(triangles)
                    .points(points)
                    .build();
    
                let potential: Vec<F> = vec![F::cst(0); grid.num_vertices()];
    
                implicit_surface.potential(grid.vertex_positions(), &potential)?;
    
                println!("{:?}", potential);
    
                //let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
                //let expected_grid: PolyMesh<f64> = load_polymesh(&PathBuf::from(
                //    "assets/approximate_sphere_test_grid_expected.vtk",
                //))?;
                //let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;
    
                //for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
                //    assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
                //}
            }
        }
    
        Ok(())
    }
    */
}
