mod test_utils;

use approx::*;
use geo::mesh::VertexPositions;
use softy::opt_fem::*;
use softy::*;
pub use test_utils::*;

/// Helper function to generate a simple solver for one initially deformed tet under gravity.
fn one_tet_solver() -> Solver {
    let mesh = make_one_deformed_tet_mesh();

    SolverBuilder::new(SimParams {
        print_level: 0,
        derivative_test: 0,
        ..STATIC_OPT_PARAMS
    })
    .add_solid(mesh, default_solid())
    .build()
    .expect("Failed to build a solver for a one tet test.")
}

/// Test that the solver produces no change for an equilibrium configuration.
#[test]
fn equilibrium() {
    let params = SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..STATIC_OPT_PARAMS
    };

    let mesh = make_one_tet_mesh();

    let mut solver = SolverBuilder::new(params)
        .add_solid(mesh.clone(), default_solid())
        .build()
        .unwrap();
    assert!(solver.step().is_ok());

    // Expect the tet to remain in original configuration
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &mesh, 1e-6);
}

/// Test one deformed tet under gravity fixed at two vertices. This is not an easy test because
/// the initial condition is far from the solution and this is a fully static solve.
/// This test has a unique solution.
#[test]
fn simple_deformed() {
    let mut solver = one_tet_solver();
    assert!(solver.step().is_ok());
    let solution = &solver.solid(0).tetmesh;

    let verts = solution.vertex_positions();

    // Check that the free verts are below the horizontal.
    assert!(verts[2][1] < 0.0 && verts[3][1] < 0.0);

    // Check that they are approximately at the same altitude.
    assert_relative_eq!(verts[2][1], verts[3][1], max_relative = 1e-3);
}

/// Test that subsequent outer iterations don't change the solution when Ipopt has converged.
/// This is not the case with linearized constraints.
#[test]
fn consistent_outer_iterations() -> Result<(), Error> {
    let params = SimParams {
        outer_tolerance: 1e-5, // This is a fairly strict tolerance.
        ..STATIC_OPT_PARAMS
    };

    let mesh = make_one_deformed_tet_mesh();

    let mut solver = SolverBuilder::new(params)
        .add_solid(mesh.clone(), default_solid())
        .build()?;
    solver.step()?;

    let solution = &solver.solid(0).tetmesh;
    let mut expected_solver = one_tet_solver();
    expected_solver.step()?;
    let expected = &expected_solver.solid(0).tetmesh;
    compare_meshes(solution, &expected, 1e-6);
    Ok(())
}

#[test]
fn volume_constraint() -> Result<(), Error> {
    let mesh = make_one_deformed_tet_mesh();

    let material = default_solid().with_volume_preservation(true);

    let mut solver = SolverBuilder::new(STATIC_OPT_PARAMS)
        .add_solid(mesh, material)
        .build()?;
    solver.step()?;
    Ok(())
}
