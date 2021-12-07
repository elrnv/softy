mod test_utils;

use geo;
use softy::opt_fem::*;
use softy::*;
use std::path::PathBuf;
pub use test_utils::*;

#[test]
fn static_sim() -> Result<(), Error> {
    let mesh = make_three_tet_mesh();
    let mut solver = SolverBuilder::new(STATIC_OPT_PARAMS)
        .add_solid(mesh, default_solid())
        .build()?;
    solver.step()?;
    let solution = &solver.solid(0).tetmesh;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_static_expected.vtk"))?;
    compare_meshes(solution, &expected, 1e-3);
    Ok(())
}

#[test]
fn dynamic_sim() -> Result<(), Error> {
    let mesh = make_three_tet_mesh();
    let mut solver = SolverBuilder::new(DYNAMIC_OPT_PARAMS)
        .add_solid(mesh, default_solid())
        .build()?;
    solver.step()?;
    let solution = &solver.solid(0).tetmesh;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/three_tets_dynamic_expected.vtk"))?;
    compare_meshes(solution, &expected, 1e-2);
    Ok(())
}

#[test]
fn static_volume_constraint() -> Result<(), Error> {
    let mesh = make_three_tet_mesh();
    let material = default_solid().with_volume_preservation(true);
    let mut solver = SolverBuilder::new(STATIC_OPT_PARAMS)
        .add_solid(mesh, material)
        .build()?;
    solver.step()?;
    let solution = &solver.solid(0).tetmesh;
    let exptected = geo::io::load_tetmesh(&PathBuf::from(
        "assets/three_tets_static_volume_constraint_expected.vtk",
    ))?;
    compare_meshes(solution, &exptected, 1e-3);
    Ok(())
}

#[test]
fn dynamic_volume_constraint() -> Result<(), Error> {
    let mesh = make_three_tet_mesh();
    let material = default_solid().with_volume_preservation(true);
    let mut solver = SolverBuilder::new(DYNAMIC_OPT_PARAMS)
        .add_solid(mesh, material)
        .build()?;
    solver.step()?;
    let solution = &solver.solid(0).tetmesh;
    let expected = geo::io::load_tetmesh(&PathBuf::from(
        "assets/three_tets_dynamic_volume_constraint_expected.vtk",
    ))?;
    compare_meshes(solution, &expected, 1e-2);
    Ok(())
}

#[test]
fn animation() -> Result<(), Error> {
    let mut verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let mesh = make_three_tet_mesh_with_verts(verts.clone());

    let mut solver = SolverBuilder::new(DYNAMIC_OPT_PARAMS)
        .add_solid(mesh, default_solid())
        .build()?;

    for frame in 1u32..100 {
        let offset = 0.01 * (if frame < 50 { frame } else { 0 } as f64);
        verts.iter_mut().for_each(|x| (*x)[1] += offset);
        let pts = PointCloud::new(verts.clone());
        solver.update_solid_vertices(&pts)?;
        solver.step()?;
    }
    Ok(())
}

#[test]
fn animation_volume_constraint() -> Result<(), Error> {
    let mut verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let mesh = make_three_tet_mesh_with_verts(verts.clone());

    let incompressible_material = default_solid().with_volume_preservation(true);

    let mut solver = SolverBuilder::new(DYNAMIC_OPT_PARAMS)
        .add_solid(mesh, incompressible_material)
        .build()?;

    for frame in 1u32..100 {
        let offset = 0.01 * (if frame < 50 { frame } else { 0 } as f64);
        verts.iter_mut().for_each(|x| (*x)[1] += offset);
        let pts = PointCloud::new(verts.clone());
        //save_tetmesh_ascii(
        //    solver.borrow_mesh(),
        //    &PathBuf::from(format!("./out/mesh_{}.vtk", frame)),
        //);
        solver.update_solid_vertices(&pts)?;
        solver.step()?;
    }
    Ok(())
}

/// Check that an inverted reference element with all vertices fixed doesn't break the simulation.
#[test]
fn inverted_fixed_reference_element_test() -> Result<(), Error> {
    use geo::attrib::*;
    use geo::mesh::topology::*;
    let mut mesh = make_three_tet_mesh();

    // Invert the first tet.
    mesh.indices[0].swap(0, 1);

    // Fix all verts of the first tet.
    let mut fixed = vec![0i8; mesh.num_vertices()];
    fixed[mesh.cell(0)[0]] = 1;
    fixed[mesh.cell(0)[1]] = 1;
    fixed[mesh.cell(0)[2]] = 1;
    fixed[mesh.cell(0)[3]] = 1;
    mesh.set_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed)?;

    let mut solver = SolverBuilder::new(DYNAMIC_OPT_PARAMS)
        .add_solid(mesh, default_solid())
        .build()?;
    solver.step()?;

    Ok(())
}

/// Check that an inverted element with all vertices fixed doesn't break the simulation.
#[test]
fn inverted_fixed_element_test() -> Result<(), Error> {
    use geo::attrib::*;
    use geo::mesh::topology::*;
    use geo::mesh::VertexPositions;
    let mut mesh = make_three_tet_mesh();

    // Invert the first tet.
    mesh.vertex_positions_mut()[4] = mesh.vertex_positions_mut()[3];

    // Fix all verts of the first tet.
    let mut fixed = vec![0i8; mesh.num_vertices()];
    fixed[mesh.cell(0)[0]] = 1;
    fixed[mesh.cell(0)[1]] = 1;
    fixed[mesh.cell(0)[2]] = 1;
    fixed[mesh.cell(0)[3]] = 1;
    mesh.set_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed)?;

    let mut solver = SolverBuilder::new(DYNAMIC_OPT_PARAMS)
        .add_solid(mesh, default_solid())
        .build()?;
    solver.step()?;

    Ok(())
}
