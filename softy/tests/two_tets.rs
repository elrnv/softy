mod test_utils;

#[cfg(feature = "optsolver")]
use approx::*;
#[cfg(feature = "optsolver")]
use geo::mesh::VertexPositions;
#[cfg(feature = "optsolver")]
use geo::ops::transform::*;
#[cfg(feature = "optsolver")]
use softy::fem::opt::*;
#[cfg(feature = "optsolver")]
use softy::*;
pub use test_utils::*;

/// Helper function to generate a simple solver for two initially deformed tets under gravity.
#[cfg(feature = "optsolver")]
fn two_tet_solver() -> Solver {
    let tet1 = make_one_deformed_tet_mesh();
    let mut tet2 = make_one_deformed_tet_mesh();

    tet2.translate([0.0, 0.5, 0.0]);

    SolverBuilder::new(STATIC_OPT_PARAMS)
        .add_solid(tet1, default_solid())
        .add_solid(tet2, default_solid())
        .build()
        .expect("Failed to build a solver for a two tet test.")
}

#[test]
#[cfg(feature = "optsolver")]
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

#[test]
#[cfg(feature = "optsolver")]
fn volume_constraint() -> Result<(), Error> {
    let tet1 = make_one_deformed_tet_mesh();
    let mut tet2 = make_one_deformed_tet_mesh();

    tet2.translate([0.0, 0.5, 0.0]);

    let material = default_solid().with_volume_preservation(true);

    let mut solver = SolverBuilder::new(STATIC_OPT_PARAMS)
        .add_solid(tet1, material)
        .add_solid(tet2, material)
        .build()?;
    solver.step()?;
    Ok(())
}
