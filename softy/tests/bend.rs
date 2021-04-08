use approx::*;

use geo::mesh::topology::VertexIndex;
use geo::mesh::{Attrib, VertexPositions};
use softy::opt_fem::*;
use softy::*;
pub use test_utils::*;

mod test_utils;

fn soft_shell_material() -> SoftShellMaterial {
    SoftShellMaterial::new(0)
        .with_elasticity(ElasticityParameters::from_bulk_shear(1000e3, 100e3))
        .with_density(1000.0)
        .with_bending_stiffness(1000.0)
}

/// Test that the solver produces no change for an equilibrium configuration.
#[test]
fn equilibrium() {
    init_logger();
    let params = SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..STATIC_OPT_PARAMS
    };

    let mut mesh = make_four_tri_mesh();

    // Unbend last vertex triangle:
    mesh.vertex_positions_mut()[5][1] = 0.0;

    // Fix another vertex to remove remaining null space.
    // (Though this should not affect most solvers).
    mesh.attrib_as_mut_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
        .unwrap()[2] = 1;

    let mut solver = SolverBuilder::new(params)
        .add_soft_shell(PolyMesh::from(mesh.clone()), soft_shell_material())
        .build()
        .expect("Failed to build a solver for a three triangle test.");
    assert!(solver.step().is_ok());

    // Expect the tet to remain in original configuration
    let solution = &solver.shell(0).trimesh;
    compare_meshes(solution, &mesh, 1e-6);
}

#[test]
fn simple_static_undeformed() {
    init_logger();
    let mut mesh = make_two_tri_mesh();

    mesh.vertex_positions_mut()[0][2] = 0.0; // Unbend

    let mut solver = SolverBuilder::new(SimParams {
        ..STATIC_OPT_PARAMS
    })
    .add_soft_shell(PolyMesh::from(mesh), soft_shell_material())
    .build()
    .expect("Failed to build a solver for a three triangle test.");

    assert!(solver.step().is_ok());
    let solution = &solver.shell(0).trimesh;

    let verts = solution.vertex_positions();

    // Check that all the vertices are in x-y plane
    for i in 0..3 {
        assert_relative_eq!(verts[i][2], 0.0, max_relative = 1e-5, epsilon = 1e-5);
    }

    // Check that that bottom verts are slightly below the xz plane (due to stretching).
    assert!(verts[0][1] < 0.0);
    assert!(verts[2][1] < 0.0);
}

/// Test the shell under gravity.
#[test]
fn static_deformed() {
    init_logger();
    let mesh = make_three_tri_mesh();
    let mut solver = SolverBuilder::new(SimParams {
        ..STATIC_OPT_PARAMS
    })
    .add_soft_shell(PolyMesh::from(mesh), soft_shell_material())
    .build()
    .expect("Failed to build a solver for a three triangle test.");

    assert!(solver.step().is_ok());
    let solution = &solver.shell(0).trimesh;

    let verts = solution.vertex_positions();

    dbg!(verts);
    // Check that the bottom vert is somewhat close to 0.0 on z axis.
    assert!(verts[3][2] < 0.4 && verts[3][2] > -0.4);

    // Check that the bottom vertex is below its original altitude.
    assert!(verts[3][1] < -1.0);

    // Check that the bottom vertex is still somewhat aligned in x.
    assert_relative_eq!(verts[3][0], 0.5, max_relative = 0.02, epsilon = 0.02);

    // Check that the middle unfixed vertices are displaced into the positive z.
    assert!(verts[0][2] > 0.0);
    assert!(verts[1][2] > 0.0);
}

/// Test the shell under gravity for one dynamic time step.
#[test]
fn dynamic_deformed() {
    init_logger();
    let mesh = make_three_tri_mesh();
    let mut solver = SolverBuilder::new(SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        ..DYNAMIC_OPT_PARAMS
    })
    .add_soft_shell(PolyMesh::from(mesh), soft_shell_material())
    .build()
    .expect("Failed to build a solver for a three triangle test.");

    assert!(solver.step().is_ok());
    let solution = &solver.shell(0).trimesh;

    let verts = solution.vertex_positions();

    // Check that the bottom vert is closer to 0.0 on z axis.
    assert!(verts[3][2] < 0.5);

    // Check that the bottom vertex is below its original altitude.
    assert!(verts[3][1] < -1.0);

    // Check that the bottom vertex is still somewhat aligned in x.
    assert_relative_eq!(verts[3][0], 0.5, max_relative = 0.02, epsilon = 0.02);

    // Check that the middle unfixed vertices are displaced into the positive z.
    assert!(verts[0][2] > 0.0);
    assert!(verts[1][2] > 0.0);
}
