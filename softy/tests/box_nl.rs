mod test_utils;

use geo::attrib::Attrib;
use geo::topology::{CellIndex, FaceIndex, NumCells, NumFaces};
use softy::fem::nl::SimParams as NLParams;
use softy::fem::nl::*;
use softy::{
    load_material, Elasticity, Error, Material, Mesh, PointCloud, SolidMaterial, TetMesh,
    VolumeZoneIdType, VOLUME_ZONE_ID_ATTRIB,
};
use std::path::PathBuf;
pub use test_utils::*;

/// Test that the solver produces no change for an equilibrium configuration for a
/// tetrahedralized box. This example also uses a softer material and a momentum term
/// (dynamics enabled), which is more sensitive to perturbations.
#[test]
fn equilibrium() -> Result<(), Error> {
    init_logger();
    let params = SimParams {
        max_iterations: 1,
        gravity: [0.0f32; 3],
        ..static_nl_params()
    };

    let soft_material = SolidMaterial::new(0)
        .with_elasticity(Elasticity::from_young_poisson(1000.0, 0.49))
        .with_volume_preservation(false)
        .with_density(1000.0);

    // Box in equilibrium configuration should stay in equilibrium configuration
    let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk")).unwrap();

    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh.clone()))
        .set_materials(vec![soft_material.into()])
        .build::<f64>()?;
    solver.step()?;

    // Expect the box to remain in original configuration
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &mesh, 1e-6);
    Ok(())
}

#[test]
fn stretch_plain() -> Result<(), Error> {
    init_logger();
    let mesh = make_stretched_box(4);
    let material: Material = load_material("assets/medium_solid_material.ron")?;
    let mut solver = SolverBuilder::new(NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh))
    .set_materials(vec![material])
    .build::<f64>()?;
    solver.step()?;
    let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched.vtk"))?;
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &expected, 1e-2);
    Ok(())
}

#[test]
#[ignore]
fn stretch_plain_large() -> Result<(), Error> {
    init_logger();
    let mesh = make_stretched_box(10);
    let material: Material = load_material("assets/medium_solid_material.ron")?;
    let mut solver = SolverBuilder::new(NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh))
    .set_materials(vec![material])
    .build::<f64>()?;
    solver.step()?;
    Ok(())
}

#[test]
fn stretch_volume_penalty() -> Result<(), Error> {
    init_logger();
    let material: Material = load_material("assets/medium_solid_material.ron")?;
    let mut mesh = make_stretched_box(4);
    mesh.insert_attrib_data::<VolumeZoneIdType, CellIndex>(
        VOLUME_ZONE_ID_ATTRIB,
        vec![1; mesh.num_cells()],
    )?;
    // geo::io::save_tetmesh(&mesh, "./out/stretched_tetmesh.vtk");
    let mut solver = SolverBuilder::new(NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        // Skip derivative test since volume penalty does not use full hessian for high compression coefficients.
        // derivative_test: 0,
        ..static_nl_params()
    })
        .set_mesh(Mesh::from(mesh))
        .set_materials(vec![material])
        .set_volume_penalty_params(vec![1.0], vec![0.1], vec![false])
        .build()?;
    solver.step()?;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_const_volume.vtk"))?;
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &expected, 1e-3);
    Ok(())
}

#[test]
fn stretch_triangles() -> Result<(), Error> {
    init_logger();
    let material: Material = load_material("assets/soft_shell_material.ron")?;
    let mesh = make_stretched_box(2);
    let mesh = mesh.surface_trimesh();

    let mut solver = SolverBuilder::new(NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh))
    .set_materials(vec![material])
    .build()?;
    solver.step()?;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_triangles.vtk"))?;
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &expected, 1e-3);
    Ok(())
}

#[test]
fn stretch_volume_penalty_triangles() -> Result<(), Error> {
    use geo::attrib::Attrib;
    init_logger();
    let material: Material = load_material("assets/soft_shell_material.ron")?;
    let mesh = make_stretched_box(3);
    let mut mesh = mesh.surface_trimesh();
    mesh.insert_attrib_data::<VolumeZoneIdType, FaceIndex>(
        VOLUME_ZONE_ID_ATTRIB,
        vec![1; mesh.num_faces()],
    )?;
    // geo::io::save_polymesh(&geo::PolyMesh::from(mesh.clone()), "./out/stretched_trimesh.vtk");
    let mut solver = SolverBuilder::new(NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..static_nl_params()
    })
    .set_mesh(Mesh::from(mesh))
    .set_materials(vec![material])
    .set_volume_penalty_params(vec![1.0], vec![0.01], vec![false])
    .build()?;
    solver.step()?;
    let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from(
        "assets/box_stretched_const_volume_triangles.vtk",
    ))?;
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &expected, 1e-3);
    Ok(())
}

#[test]
fn twist_plain() -> Result<(), Error> {
    init_logger();
    let material: Material = load_material("assets/no_poisson_soft_solid_material.ron")?;
    let mesh = geo::io::load_tetmesh("assets/box_twist.vtk")?;
    let params = NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..static_nl_params()
    };
    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh))
        .set_materials(vec![material.into()])
        .build::<f64>()?;
    solver.step()?;
    let expected: TetMesh = geo::io::load_tetmesh("assets/box_twisted.vtk")?;
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &expected, 1e-3);
    Ok(())
}

#[test]
fn twist_damped_dynamic() -> Result<(), Error> {
    init_logger();
    let material: Material = load_material("assets/no_poisson_damped_soft_solid_material.ron")?;
    let mesh = geo::io::load_tetmesh("assets/box_twist.vtk")?;
    let params = NLParams {
        gravity: [0.0f32, 0.0, 0.0],
        time_step: Some(0.5),
        time_integration: TimeIntegration::TRBDF2(0.5), // Exotic integrator to make sure damping derivative is right
        ..static_nl_params()
    };
    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh))
        .set_materials(vec![material.into()])
        .build::<f64>()?;
    for _ in 0..50 {
        solver.step()?;
    }
    let expected: TetMesh = geo::io::load_tetmesh("assets/box_twisted.vtk")?;
    let solution_verts = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution_verts, &expected, 1e-3);
    Ok(())
}

#[cfg(not(debug_assertions))]
#[test]
fn twist_dynamic_volume_penalty() -> Result<(), Error> {
    init_logger();
    let material: Material = load_material("assets/no_poisson_damped_soft_solid_material.ron")?;

    // We use a large time step to get the simulation to settle to the static sim with less
    // iterations.
    let params = NLParams {
        gravity: [0.0; 3],
        time_step: Some(0.1),
        // Skip derivative test since volume penalty does not use full hessian for high compression coefficients.
        derivative_test: 0,
        ..static_nl_params()
    };

    let mut mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
    mesh.insert_attrib_data::<VolumeZoneIdType, CellIndex>(
        VOLUME_ZONE_ID_ATTRIB,
        vec![1; mesh.num_cells()],
    )?;
    let mut solver = SolverBuilder::new(params.clone())
        .set_mesh(mesh)
        .set_material(material)
        .set_volume_penalty_params(vec![1.0], vec![0.1], vec![false])
        .build()?;

    // The dynamic sim needs to settle
    for i in 1u32..150 {
        let result = solver.step()?;
        geo::io::save_mesh(
            &solver.mesh(),
            &format!("out/box_twisted/box_twisted_const_volume_{i}.vtk"),
        )?;
        assert!(
            result.iterations <= params.max_iterations,
            "Unconstrained solver ran out of outer iterations."
        );
    }

    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
    let solution = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution, &expected, 1e-2);
    Ok(())
}

#[test]
fn twist_volume_penalty() -> Result<(), Error> {
    init_logger();
    let material: Material = load_material("assets/no_poisson_soft_solid_material.ron")?;
    let mut mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
    mesh.insert_attrib_data::<VolumeZoneIdType, CellIndex>(
        VOLUME_ZONE_ID_ATTRIB,
        vec![1; mesh.num_cells()],
    )?;
    let params = NLParams {
        gravity: [0.0; 3],
        // // Skip derivative test since volume penalty does not use full hessian for high compression coefficients.
        // derivative_test: 0,
        ..static_nl_params()
    };
    let mut solver = SolverBuilder::new(params)
        .set_mesh(mesh)
        .set_material(material)
        .set_volume_penalty_params(vec![1.0], vec![0.1], vec![false])
        .build()?;
    solver.step()?;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
    let solution = PointCloud::new(solver.vertex_positions());
    compare_meshes(&solution, &expected, 1e-6);
    Ok(())
}
