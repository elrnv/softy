mod test_utils;

use approx::*;
use geo::mesh::VertexPositions;
use softy::*;
pub use test_utils::*;

/// Helper function to generate a simple solver for two initially deformed tets under gravity.
fn two_tet_solver() -> Solver {
    let tet1 = make_one_deformed_tet_mesh();
    let mut tet2 = make_one_deformed_tet_mesh();

    utils::translate(&mut tet2, [0.0, 0.5, 0.0]);

    SolverBuilder::new(SimParams {
        print_level: 0,
        derivative_test: 0,
        ..STATIC_PARAMS
    })
    .add_solid(tet1, SOLID_MATERIAL)
    .add_solid(tet2, SOLID_MATERIAL)
    .build()
    .expect("Failed to build a solver for a two tet test.")
}

/// Ball with constant volume bouncing on an implicit surface.
#[test]
fn two_deformed_tets_test() {
    let mut solver = two_tet_solver();
    assert!(solver.step().is_ok());
    let solution0 = &solver.solid(0).tetmesh;
    let solution1 = &solver.solid(1).tetmesh;

    let verts = solution0.vertex_positions();

    // Check that the free verts are below the horizontal.
    assert!(verts[2][1] < 0.0 && verts[3][1] < 0.0);

    // Check that they are approximately at the same altitude.
    assert_relative_eq!(verts[2][1], verts[3][1], max_relative = 1e-3);

    let verts = solution1.vertex_positions();

    // Check that the free verts are below the horizontal.
    assert!(verts[2][1] < 0.5 && verts[3][1] < 0.5);

    // Check that they are approximately at the same altitude.
    assert_relative_eq!(verts[2][1], verts[3][1], max_relative = 1e-3);
}
