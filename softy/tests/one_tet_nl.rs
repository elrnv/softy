mod test_utils;

use approx::*;
use geo::topology::{CellIndex, FaceIndex};
use softy::nl_fem::*;
use softy::Error;
use softy::*;
pub use test_utils::*;

/// Helper function to generate a simple solver for one initially deformed tet under gravity.
fn one_tet_solver(config_idx: u32) -> Solver<impl NLSolver<NLProblem<f64>, f64>, f64> {
    let mesh = make_one_deformed_tet_mesh();

    SolverBuilder::new(static_nl_params(config_idx))
        .set_mesh(mesh)
        .set_material(default_solid())
        .build()
        .expect("Failed to build a solver for a one tet test.")
}

/// Test that the solver produces no change for an equilibrium configuration.
#[test]
fn equilibrium() {
    for config_idx in 0..num_static_configs() {
        let params = SimParams {
            gravity: [0.0f32, 0.0, 0.0],
            ..static_nl_params(config_idx)
        };

        let mesh = make_one_tet_mesh();

        let mut solver = SolverBuilder::new(params)
            .set_mesh(mesh.clone())
            .set_material(default_solid())
            .build()
            .unwrap();
        assert!(solver.step().is_ok());

        // Expect the tet to remain in original configuration
        let solution_verts = PointCloud::new(solver.vertex_positions());
        compare_meshes(&solution_verts, &mesh, 1e-6);
    }
}

/// Test one deformed tet under gravity fixed at two vertices. This is not an easy test because
/// the initial condition is far from the solution and this is a fully static solve.
/// This test has a unique solution.
#[test]
fn simple_deformed() {
    init_logger();
    for config_idx in 0..num_static_configs() {
        let mut solver = one_tet_solver(config_idx);

        geo::io::save_mesh(&solver.mesh(), "./out/before_one_tet.vtk").unwrap();

        assert!(solver.step().is_ok());
        let verts = solver.vertex_positions();

        geo::io::save_mesh(&solver.mesh(), "./out/one_tet.vtk").unwrap();

        // Check that the free verts are below the horizontal.
        assert!(verts[2][1] < 0.0 && verts[3][1] < 0.0);

        // Check that they are approximately at the same altitude.
        assert_relative_eq!(verts[2][1], verts[3][1], max_relative = 1e-3);
    }
}

/// Test that subsequent outer iterations don't change the solution when the solver has converged.
/// This is not the case with linearized constraints.
#[test]
fn consistent_outer_iterations() -> Result<(), Error> {
    init_logger();
    for config_idx in 0..num_static_configs() {
        let params = SimParams {
            ..static_nl_params(config_idx)
        };

        let mesh = make_one_deformed_tet_mesh();

        let mut solver = SolverBuilder::new(params)
            .set_mesh(mesh.clone())
            .set_material(default_solid())
            .build()?;
        solver.step()?;

        let solution = PointCloud::new(solver.vertex_positions());
        let mut expected_solver = one_tet_solver(config_idx);
        expected_solver.step()?;
        let expected = PointCloud::new(expected_solver.vertex_positions());
        compare_meshes(&solution, &expected, 1e-6);
    }
    Ok(())
}

#[test]
fn volume_penalty() -> Result<(), Error> {
    use geo::{attrib::Attrib, topology::NumCells};
    init_logger();
    let mut mesh = make_one_deformed_tet_mesh();
    mesh.insert_attrib_data::<VolumeZoneIdType, CellIndex>(
        VOLUME_ZONE_ID_ATTRIB,
        vec![1; mesh.num_cells()],
    )?;

    let material = default_solid();

    for config_idx in 0..num_static_configs() {
        let mut solver = SolverBuilder::new(static_nl_params(config_idx))
            .set_mesh(mesh.clone())
            .set_material(material)
            .set_volume_penalty_params(vec![1.0], vec![0.0001], vec![false])
            .build::<f64>()?;
        solver.step()?;
    }
    Ok(())
}

/// Tests that volume penalty works on triangle meshes.
#[test]
fn volume_penalty_triangles() -> Result<(), Error> {
    use geo::{attrib::Attrib, topology::NumFaces};
    init_logger();
    let mesh = make_one_deformed_tet_mesh();
    let mut mesh = mesh.surface_trimesh();
    mesh.insert_attrib_data::<VolumeZoneIdType, FaceIndex>(
        VOLUME_ZONE_ID_ATTRIB,
        vec![1; mesh.num_faces()],
    )?;

    let material = default_shell();

    for config_idx in 0..num_static_configs() {
        let mut solver = SolverBuilder::new(static_nl_params(config_idx))
            .set_mesh(mesh.clone())
            .set_material(material)
            .set_volume_penalty_params(vec![1.0], vec![0.0001], vec![false])
            .build::<f64>()?;
        solver.step()?;
    }
    Ok(())
}
