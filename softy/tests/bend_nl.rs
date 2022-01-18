use approx::*;

use geo::attrib::Attrib;
use geo::mesh::topology::VertexIndex;
use geo::mesh::VertexPositions;
use softy::nl_fem::*;
use softy::{Error, *};
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
        max_iterations: 1,
        gravity: [0.0f32; 3],
        time_step: Some(0.01),
        ..static_nl_params()
    };

    let mut mesh = make_four_tri_mesh();

    // Unbend last vertex triangle:
    mesh.vertex_positions_mut()[5][1] = 0.0;

    // Fix another vertex to remove remaining null space.
    // (Though this should not affect most solvers).
    mesh.attrib_as_mut_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
        .unwrap()[2] = 1;

    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh.clone()))
        .set_materials(vec![soft_shell_material().into(); 1])
        .build::<f64>()
        .expect("Failed to build a solver for a three triangle test.");
    let result = solver.step();
    assert!(result.is_ok());

    // Expect the triangles to remain in original configuration
    let solution_verts  = PointCloud::new(
        solver.vertex_positions(),
    );
    compare_meshes(&solution_verts, &mesh, 1e-6);
}

#[test]
fn simple_quasi_static_undeformed() -> Result<(), Error> {
    init_logger();
    let mut mesh = make_two_tri_mesh();

    mesh.vertex_positions_mut()[0][2] = 0.0; // Unbend

    let mut solver = SolverBuilder::new(SimParams {
        velocity_clear_frequency: 100.0,
        time_step: Some(0.01),
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh.clone()))
    .set_materials(vec![soft_shell_material().into(); 1])
    .build::<f64>()
    .expect("Failed to build a solver for a three triangle test.");

    // Simulate for 1 second quasi-static simulation
    for _ in 0..100 {
        solver.step()?;
    }

    let verts = solver.vertex_positions();

    // Check that all the vertices are in x-y plane
    for i in 0..3 {
        assert_relative_eq!(verts[i][2], 0.0, max_relative = 1e-5, epsilon = 1e-5);
    }

    // Check that that bottom verts are slightly below the xz plane (due to stretching).
    assert!(verts[0][1] < 0.0);
    assert!(verts[2][1] < 0.0);
    Ok(())
}

/// Test the shell under gravity.
#[test]
fn quasi_static_deformed() -> Result<(), Error> {
    init_logger();
    let mesh = make_three_tri_mesh();
    dbg!(mesh.vertex_positions());
    let mut solver = SolverBuilder::new(SimParams {
        velocity_clear_frequency: 20.0,
        time_step: Some(0.1),
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh.clone()))
    .set_materials(vec![soft_shell_material().into()])
    .build::<f64>()
    .expect("Failed to build a solver for a three triangle test.");

    // Simulate for 1 second.
    for _ in 0..40 {
        solver.step()?;
    }

    let verts = solver.vertex_positions();

    // Check that the bottom vert is somewhat close to 0.0 on z axis.
    assert!(verts[3][2] < 0.4 && verts[3][2] > -0.4);

    // Check that the bottom vertex is below its original altitude.
    assert!(verts[3][1] < -1.0);

    // Check that the bottom vertex is still somewhat aligned in x.
    assert_relative_eq!(verts[3][0], 0.5, max_relative = 0.02, epsilon = 0.02);

    // Check that the middle unfixed vertices are displaced into the positive z.
    assert!(verts[0][2] > 0.0);
    assert!(verts[1][2] > 0.0);
    Ok(())
}

/// Test the shell under gravity for one dynamic time step.
#[test]
fn dynamic_deformed() -> Result<(), Error> {
    init_logger();
    let mesh = make_three_tri_mesh();
    let mut solver = SolverBuilder::new(SimParams {
        time_step: Some(0.05),
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh))
    .set_materials(vec![soft_shell_material().into()])
    .build::<f64>()
    .expect("Failed to build a solver for a three triangle test.");

    solver.step()?;
    let shell = &solver.shell();
    let solution = TriMesh::new(
        solver.vertex_positions(),
        shell.triangle_elements.triangles.clone(),
    );

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
    Ok(())
}
