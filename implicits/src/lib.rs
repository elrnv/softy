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
    pub max_step: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            // Note that this is not a good default and should always be set explicitly.
            kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.0,
            },
            background_potential: BackgroundPotentialType::None,
            sample_type: SampleType::Face,
            max_step: 0.0,
        }
    }
}

/// Compute potential with debug information on the given mesh.
pub fn compute_potential_debug<F>(
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

pub fn dynamic_surface<F>(
    query_points: &mut PolyMesh<f64>,
    surface: &mut PolyMesh<f64>,
    params: Params,
    interrupt: F,
) -> Result<ImplicitSurface, Error>
where
    F: Fn() -> bool + Sync + Send,
{
    let mut builder = ImplicitSurfaceBuilder::new();

    builder.kernel(params.kernel);
    builder.max_step(params.max_step);
    builder.background_potential(params.background_potential);
    builder.sample_type(params.sample_type);
    builder.vertices()

    if let Some(implicit_surface) = builder.build() {
        implicit_surface.potential(query_points, interrupt)?;
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
    fn make_grid(rows: usize, cols: usize) -> PolyMesh<f64> {
        use utils::*;
        let mut mesh = make_grid(Grid { rows, cols, orientation: AxisPlaneOrientation::XY });
        mesh.add_attrib::<_, VertexIndex>("potential", 0.0f32)
            .unwrap();
        mesh
    }

    /// Test the non-dynamic API.
    #[test]
    fn vertex_samples_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let trimesh = utils::make_sample_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential_debug(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.5,
                },
                background_potential: BackgroundPotentialType::DistanceBased,
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
            || false,
        )?;

        //geo::io::save_polymesh(&grid, &PathBuf::from("mesh.vtk")).unwrap();

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> =
            load_polymesh(&PathBuf::from("assets/octahedron_vertex_grid_expected.vtk"))?;
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

        let trimesh = utils::make_sample_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential_debug(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.5,
                },
                background_potential: BackgroundPotentialType::DistanceBased,
                sample_type: SampleType::Face,
                ..Default::default()
            },
            || false,
        )?;

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> =
            load_polymesh(&PathBuf::from("assets/octahedron_face_grid_expected.vtk"))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }

    /// Vertex centered HRBF surface test.
    #[test]
    fn hrbf_vertex_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let trimesh = utils::make_sample_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential_debug(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Hrbf,
                background_potential: BackgroundPotentialType::DistanceBased,
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
            || false,
        )?;

        //geo::io::save_polymesh(&grid, &PathBuf::from("mesh.vtk")).unwrap();

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> =
            load_polymesh(&PathBuf::from("assets/hrbf_octahedron_vertex_grid_expected.vtk"))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }

    /// Test the dynamic API (I.e. one which provides derivatives). We don't test the derivatives
    /// themselves here.
    #[test]
    fn vertex_samples_dynamic_test() -> Result<(), Error> {
        use super::geo::mesh::VertexPositions;

        let grid = make_grid(22, 22);

        let trimesh = utils::make_sample_octahedron();

        let mut builder = ImplicitSurfaceBuilder::new();
        builder.kernel(KernelType::Approximate { tolerance: 1e-5, radius: 1.5 })
            .background_potential(BackgroundPotentialType::DistanceBased)
            .sample_type(SampleType::Vertex)
            .triangles(reinterpret::reinterpret_slice::<_, [usize;3]>(trimesh.faces()).to_vec())
            .vertices(trimesh.vertex_positions().to_vec());

        let surf = builder.build().expect("Failed to create implicit surface.");

        let mut potential = vec![0.0f64; grid.num_vertices()];
        surf.potential(grid.vertex_positions(), &mut potential).expect("Failed to compute potential.");

        //geo::io::save_polymesh(&grid, &PathBuf::from("mesh.vtk")).unwrap();

        let expected_grid: PolyMesh<f64> =
            load_polymesh(&PathBuf::from("assets/octahedron_vertex_grid_expected.vtk"))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, &exp_pot) in potential.into_iter().zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot as f64, max_relative = 1e-6);
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
    
        let trimesh = utils::make_sample_octahedron();
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
