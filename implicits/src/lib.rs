#![cfg_attr(feature = "unstable", feature(test))]
#![type_length_limit = "9650144"]
#![allow(clippy::just_underscores_and_digits)]

#[cfg(test)]
#[macro_use]
extern crate approx;

use geo::mesh::{attrib, PointCloud, PolyMesh, TriMesh};

#[macro_use]
mod kernel;
pub mod field;

pub use crate::field::*;
pub use crate::kernel::KernelType;
use tensr::Real;
use thiserror::Error;

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub kernel: KernelType,
    pub background_field: BackgroundFieldParams,
    pub sample_type: SampleType,
    pub max_step: f64,
    pub base_radius: Option<f64>,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            // Note that this is not a good default and should always be set explicitly.
            kernel: KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier: 1.1,
            },
            background_field: BackgroundFieldParams {
                field_type: BackgroundFieldType::Zero,
                weighted: false,
            },
            sample_type: SampleType::Face,
            max_step: 0.0,
            base_radius: None,
        }
    }
}

/// Compute potential with debug information on the given mesh.
/// This function builds an implicit surface and computes values on the given query points. For a
/// reusable implicit surface use the `surface_from_*` function.
pub fn compute_potential_debug<F>(
    query_points: &mut PolyMesh<f64>,
    surface: &mut PolyMesh<f64>,
    params: Params,
    interrupt: F,
) -> Result<(), Error>
where
    F: Fn() -> bool + Sync + Send,
{
    surface_from_polymesh(surface, params)
        .and_then(|surf| surf.compute_potential_on_mesh(query_points, interrupt))
}

/// A convenience routine for building an implicit surface from a given set of parameters and a
/// given `TriMesh`.
pub fn surface_from_trimesh<T: Real>(
    surface: &TriMesh<f64>,
    params: Params,
) -> Result<ImplicitSurface<T>, Error> {
    let mut surf_builder = ImplicitSurfaceBuilder::new();
    surf_builder
        .kernel(params.kernel)
        .max_step(params.max_step)
        .background_field(params.background_field)
        .sample_type(params.sample_type)
        .trimesh(surface);

    if let Some(base_radius) = params.base_radius {
        surf_builder.base_radius(base_radius);
    }

    let surf = surf_builder.build_generic();

    match surf {
        Some(s) => Ok(s),
        None => Err(Error::Failure),
    }
}

/// A convenience routine for building an implicit surface from a given set of parameters and a
/// given `PolyMesh`.
pub fn surface_from_polymesh<T: Real>(
    surface: &PolyMesh<f64>,
    params: Params,
) -> Result<ImplicitSurface<T>, Error> {
    let surf_trimesh = TriMesh::from(surface.clone());
    surface_from_trimesh::<T>(&surf_trimesh, params)
}

/// A convenience routine for building an implicit surface from a given set of parameters and a
/// given `TriMesh`.
pub fn mls_from_trimesh<T: Real>(surface: &TriMesh<f64>, params: Params) -> Result<MLS<T>, Error> {
    let mut surf_builder = ImplicitSurfaceBuilder::new();
    surf_builder
        .kernel(params.kernel)
        .max_step(params.max_step)
        .background_field(params.background_field)
        .sample_type(params.sample_type)
        .trimesh(surface);

    if let Some(base_radius) = params.base_radius {
        surf_builder.base_radius(base_radius);
    }

    let surf = surf_builder.build_mls();

    match surf {
        Some(s) => Ok(s),
        None => Err(Error::Failure),
    }
}

/// A convenience routine for building an implicit surface from a given set of parameters and a
/// given `PolyMesh`.
pub fn mls_from_polymesh<T: Real>(
    surface: &PolyMesh<f64>,
    params: Params,
) -> Result<MLS<T>, Error> {
    let surf_trimesh = TriMesh::from(surface.clone());
    mls_from_trimesh::<T>(&surf_trimesh, params)
}

/// A convenience routine for building an implicit surface from a given set of parameters and a
/// given `PointCloud`.
pub fn mls_from_pointcloud<T: Real>(
    ptcloud: &PointCloud<f64>,
    params: Params,
) -> Result<MLS<T>, Error> {
    let mut surf_builder = ImplicitSurfaceBuilder::new();
    surf_builder
        .kernel(params.kernel)
        .max_step(params.max_step)
        .background_field(params.background_field)
        .sample_type(params.sample_type)
        .vertex_mesh(ptcloud);

    if let Some(base_radius) = params.base_radius {
        surf_builder.base_radius(base_radius);
    }

    let surf = surf_builder.build_mls();

    match surf {
        Some(s) => Ok(s),
        None => Err(Error::Failure),
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("computation was interrupted")]
    Interrupted,
    #[error("normals are either missing or have the wrong type")]
    MissingNormals,
    #[error("unsupported kernel is specified")]
    UnsupportedKernel,
    #[error("unsupported sample type is specified")]
    UnsupportedSampleType,
    #[error("missing neighbour data")]
    MissingNeighbourData,
    #[error("invalid background field construction")]
    InvalidBackgroundConstruction,
    #[error("failed to compute implicit surface")]
    Failure,
    #[error("IO error")]
    IO {
        #[from]
        source: geo::io::Error,
    },
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Self {
        match err {
            attrib::Error::TypeMismatch { .. } => Error::MissingNormals,
            _ => Error::Failure,
        }
    }
}

/// Generate a [-1,1]x[-1,1] mesh grid with the given cell resolution and a potential field
/// attribute.
#[cfg(test)]
pub(crate) fn make_grid(rows: usize, cols: usize) -> PolyMesh<f64> {
    use geo::mesh::attrib::*;
    use geo::mesh::builder::*;
    use geo::mesh::topology::*;

    let mut mesh = GridBuilder {
        rows,
        cols,
        orientation: AxisPlaneOrientation::XY,
    }
    .build();
    mesh.add_attrib::<_, VertexIndex>("potential", 0.0f32)
        .unwrap();
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::io::load_polymesh;
    use geo::mesh::builder::*;
    use geo::mesh::topology::*;
    use geo::mesh::*;
    use std::path::PathBuf;

    /// Test the non-dynamic API.
    #[test]
    fn vertex_samples_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential_debug(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier: 3.6742346141747673,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: true,
                },
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
            || false,
        )?;

        //geo::io::save_polymesh(&grid, &PathBuf::from("out/octahedron_vertex_grid_expected.vtk")).unwrap();

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

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential_debug(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier: 3.6742346141747673,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: true,
                },
                sample_type: SampleType::Face,
                ..Default::default()
            },
            || false,
        )?;

        //geo::io::save_polymesh(
        //    &grid,
        //    &PathBuf::from("out/octahedron_face_grid_expected.vtk"),
        //)
        //.unwrap();

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> =
            load_polymesh(&PathBuf::from("assets/octahedron_face_grid_expected.vtk"))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }

    //    #[test]
    //    fn ephemeral_test() -> Result<(), Error> {
    //        let mut grid = make_grid(22, 22);
    //
    //        let trimesh = utils::make_sample_octahedron();
    //
    //        let mut sphere = PolyMesh::from(trimesh);
    //
    //        compute_potential_debug(
    //            &mut grid,
    //            &mut sphere,
    //            Params {
    //                kernel: KernelType::Approximate {
    //                    tolerance: 0.00001,
    //                    radius_multiplier: 3.6742346141747673,
    //                },
    //                background_field: BackgroundFieldParams {
    //                    field_type: BackgroundFieldType::DistanceBased,
    //                    weighted: false,
    //                },
    //                sample_type: SampleType::Face,
    //                ..Default::default()
    //            },
    //            || false,
    //        )?;
    //
    //        //geo::io::save_polymesh(
    //        //    &grid,
    //        //    &PathBuf::from("out/octahedron_face_grid_expected.vtk"),
    //        //)
    //        //.unwrap();
    //
    //        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
    //        let alt_potential_iter = grid.attrib_iter::<f32, VertexIndex>("alt_potential")?;
    //
    //        for (sol_pot, alt_pot) in solution_potential_iter.zip(alt_potential_iter) {
    //            assert_relative_eq!(sol_pot, alt_pot, max_relative = 1e-6);
    //        }
    //
    //        Ok(())
    //    }

    /// Vertex centered HRBF surface test.
    #[test]
    fn hrbf_vertex_test() -> Result<(), Error> {
        let mut grid = make_grid(22, 22);

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let mut sphere = PolyMesh::from(trimesh);

        compute_potential_debug(
            &mut grid,
            &mut sphere,
            Params {
                kernel: KernelType::Hrbf,
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: true,
                },
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
            || false,
        )?;

        //geo::io::save_polymesh(&grid, &PathBuf::from("mesh.vtk")).unwrap();

        let solution_potential_iter = grid.attrib_iter::<f32, VertexIndex>("potential")?;
        let expected_grid: PolyMesh<f64> = load_polymesh(&PathBuf::from(
            "assets/hrbf_octahedron_vertex_grid_expected.vtk",
        ))?;
        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, exp_pot) in solution_potential_iter.zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot, max_relative = 1e-6);
        }

        Ok(())
    }

    /// Test the dynamic API (I.e. one which provides derivatives). We don't test the derivatives
    /// themselves here, but we do verify that both APIs produce the same result.
    #[test]
    fn vertex_samples_dynamic_test() -> Result<(), Error> {
        use geo::mesh::VertexPositions;

        let grid = make_grid(22, 22);

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let mut builder = ImplicitSurfaceBuilder::new();
        builder
            .kernel(KernelType::Approximate {
                tolerance: 1e-5,
                radius_multiplier: 3.6742346141747673,
            })
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: true,
            })
            .sample_type(SampleType::Vertex)
            .trimesh(&trimesh);

        let surf = builder
            .build_mls()
            .expect("Failed to create implicit surface.");

        let query_surf = QueryTopo::new(grid.vertex_positions(), surf);

        let mut potential = vec![0.0f64; grid.num_vertices()];
        query_surf.potential(grid.vertex_positions(), &mut potential);

        let expected_grid: PolyMesh<f64> =
            load_polymesh(&PathBuf::from("assets/octahedron_vertex_grid_expected.vtk"))?;

        let expected_potential_iter = expected_grid.attrib_iter::<f32, VertexIndex>("potential")?;

        for (sol_pot, &exp_pot) in potential.into_iter().zip(expected_potential_iter) {
            assert_relative_eq!(sol_pot, exp_pot as f64, max_relative = 1e-6);
        }

        Ok(())
    }

    /// Test the surface jacobian of the implicit surface.
    #[test]
    fn surface_jacobian_test() {
        use autodiff::F1;
        use tensr::Vector3;

        let grid = make_grid(22, 22);
        let grid_pos: Vec<_> = grid
            .vertex_position_iter()
            .map(|&p| Vector3::new(p).map(|x| F1::cst(x)).into())
            .collect();

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let implicit_surface = ImplicitSurfaceBuilder::new()
            .kernel(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier: 2.45,
            })
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: true,
            })
            .sample_type(SampleType::Face)
            .trimesh(&trimesh)
            .build_mls::<F1>()
            .expect("Failed to create implicit surface.");

        let mut query_surf = implicit_surface.query_topo(&grid_pos);
        let nnz = query_surf.num_surface_jacobian_entries();
        let mut vals = vec![F1::cst(0.0); nnz];
        query_surf.surface_jacobian_values(&grid_pos, &mut vals);

        // This is the full jacobian (including all zeros).
        let mut jac =
            vec![vec![0.0; grid_pos.len()]; query_surf.surface_vertex_positions().len() * 3];

        for (idx, &val) in query_surf.surface_jacobian_indices_iter().zip(vals.iter()) {
            jac[idx.1][idx.0] += val.value();
        }

        for cur_pt_idx in 0..trimesh.num_vertices() {
            // for each vertex
            for i in 0..3 {
                // for each component
                let verts: Vec<_> = query_surf
                    .surface_vertex_positions()
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(j, mut v)| {
                        v = Vector3::new(v).mapd(|x| F1::cst(x)).into(); // reset variables to constants
                        if j == cur_pt_idx {
                            v[i] = F1::var(v[i]); // set the current tested variable
                        }
                        v.into()
                    })
                    .collect();

                query_surf.update_surface(verts.into_iter());

                let mut potential: Vec<F1> = vec![F1::cst(0); grid.num_vertices()];

                query_surf.potential(&grid_pos, &mut potential);

                let col = 3 * cur_pt_idx + i;
                for row in 0..grid_pos.len() {
                    if !relative_eq!(
                        jac[col][row],
                        potential[row].deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10,
                    ) {
                        panic!(
                            "({}, {}) jac = {} vs ad_jac = {}",
                            row,
                            col,
                            jac[col][row],
                            potential[row].deriv()
                        );
                    }
                }
            }
        }
    }

    /// Test the query jacobian of the implicit surface.
    #[test]
    fn complex_query_jacobian_test() {
        use autodiff::F1;
        use rayon::prelude::*;
        use tensr::Vector3;

        let grid = make_grid(23, 23);
        let mut grid_pos: Vec<_> = grid
            .vertex_position_iter()
            .map(|&p| Vector3::new(p).map(|x| F1::cst(x)).into())
            .collect();

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let implicit_surface = ImplicitSurfaceBuilder::new()
            .kernel(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier: 2.45,
            })
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: true,
            })
            .sample_type(SampleType::Face)
            .trimesh(&trimesh)
            .build_mls::<F1>()
            .expect("Failed to create implicit surface.");

        let query_surf = implicit_surface.query_topo(&grid_pos);
        // Testing the parallel version:
        let vals: Vec<_> = query_surf
            .query_jacobian_block_par_iter(&grid_pos)
            .filter_map(|x| x)
            .collect();

        // This is the full jacobian (including all zeros).
        let mut jac = vec![vec![0.0; grid_pos.len()]; grid_pos.len() * 3];
        //let mut flat_jac = vec![0.0; grid_pos.len() * 3];

        for (idx, &val) in query_surf
            .query_jacobian_block_indices_iter()
            .zip(vals.iter())
        {
            for i in 0..3 {
                jac[3 * idx.1 + i][idx.0] += val[i].value();
            }
            //flat_jac[idx.1] = val.value();
        }

        //let mut exp_flat_jac = vec![0.0; grid_pos.len() * 3];

        for cur_pt_idx in 0..grid_pos.len() {
            // for each vertex
            for i in 0..3 {
                // for each component
                grid_pos[cur_pt_idx][i] = F1::var(grid_pos[cur_pt_idx][i]);

                let mut potential: Vec<F1> = vec![F1::cst(0); grid.num_vertices()];
                query_surf.potential(&grid_pos, &mut potential);

                let col = 3 * cur_pt_idx + i;
                for row in 0..grid_pos.len() {
                    //if !relative_eq!(
                    //    jac[col][row],
                    //    potential[row].deriv(),
                    //    max_relative = 1e-5,
                    //    epsilon = 1e-10
                    //) {
                    //    println!(
                    //        "({:?}, {:?}) => {:?} vs {:?}",
                    //        row,
                    //        col,
                    //        jac[col][row],
                    //        potential[row].deriv()
                    //    );
                    //}
                    assert_relative_eq!(
                        jac[col][row],
                        potential[row].deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );
                }
                //exp_flat_jac[col] = potential[cur_pt_idx].deriv();
                grid_pos[cur_pt_idx][i] = F1::cst(grid_pos[cur_pt_idx][i]);
            }
        }

        //geo::io::save_polymesh(&grid, &PathBuf::from("out/gradient.vtk"))?;
    }

    /// Test the query Hessian of the implicit surface.
    #[test]
    fn complex_query_hessian_test() {
        use autodiff::F1;
        use tensr::Vector3;

        let grid = make_grid(11, 11);
        let mut grid_pos: Vec<_> = grid
            .vertex_position_iter()
            .map(|&p| Vector3::new(p).map(|x| F1::cst(x)).into())
            .collect();

        let trimesh = PlatonicSolidBuilder::build_octahedron();

        let implicit_surface = ImplicitSurfaceBuilder::new()
            .kernel(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier: 2.45,
            })
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: true,
            })
            .sample_type(SampleType::Face)
            .trimesh(&trimesh)
            .build_mls::<F1>()
            .expect("Failed to create implicit surface.");

        let query_surf = implicit_surface.query_topo(&grid_pos);
        let nnz = query_surf.num_query_hessian_product_entries();
        let mut vals = vec![F1::cst(0.0); nnz];
        let multipliers = vec![F1::cst(1.0); nnz]; // all at once
        query_surf.query_hessian_product_values(&grid_pos, &multipliers, &mut vals);

        // This is the full jacobian (including all zeros).
        let mut hess = vec![vec![0.0; grid_pos.len() * 3]; grid_pos.len() * 3];

        for (idx, &val) in query_surf
            .query_hessian_product_indices_iter()
            .zip(vals.iter())
        {
            hess[idx.1][idx.0] += val.value();
            if idx.1 != idx.0 {
                hess[idx.0][idx.1] += val.value();
            }
        }

        let mut success = true;
        for cur_pt_idx in 0..grid_pos.len() {
            // for each vertex
            for i in 0..3 {
                // for each component
                grid_pos[cur_pt_idx][i] = F1::var(grid_pos[cur_pt_idx][i]);

                let nnz = query_surf.num_query_jacobian_entries();
                let mut vals = vec![F1::cst(0.0); nnz];
                query_surf.query_jacobian_values(&grid_pos, &mut vals);

                // This is the full jacobian (including all zeros).
                let mut flat_jac_deriv = vec![0.0; grid_pos.len() * 3];

                for (idx, &val) in query_surf.query_jacobian_indices_iter().zip(vals.iter()) {
                    flat_jac_deriv[idx.1] += val.deriv();
                }

                let row = 3 * cur_pt_idx + i;
                for col in 0..grid_pos.len() * 3 {
                    if !relative_eq!(
                        hess[col][row],
                        flat_jac_deriv[col],
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    ) {
                        success = false;
                        eprintln!(
                            "({:?}, {:?}) => {:?} vs {:?}",
                            row, col, hess[col][row], flat_jac_deriv[col]
                        );
                    }
                }
                grid_pos[cur_pt_idx][i] = F1::cst(grid_pos[cur_pt_idx][i]);
            }
            assert!(success);
        }
    }
}
