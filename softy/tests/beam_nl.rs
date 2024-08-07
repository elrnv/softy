mod test_utils;

use geo::attrib::Attrib;
use geo::mesh::builder::BoxBuilder;
use geo::mesh::VertexPositions;
use geo::ops::{Scale, Translate};
use geo::topology::{CellIndex, NumCells, NumVertices, VertexIndex};
use softy::fem::nl::*;
use softy::{
    load_material, Error, FixedIntType, Material, MaterialIdType, Mesh, RefPosType, TetMesh,
    FIXED_ATTRIB, MATERIAL_ID_ATTRIB, REFERENCE_VERTEX_POS_ATTRIB,
};
pub use test_utils::*;

pub fn make_beam(i: u32) -> TetMesh {
    let mut mesh = BoxBuilder {
        divisions: [5 * i - 1, i - 1, i - 1],
    }
    .build_tetmesh();
    mesh.scale([2.0, 0.1, 0.1]);
    mesh.translate([1.0, 0.0, 0.0]);
    let ref_verts = mesh
        .vertex_position_iter()
        .map(|&[a, b, c]| [a as f32, b as f32, c as f32])
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, ref_verts)
        .expect("Failed to add reference positions to beam tetmesh");
    mesh.insert_attrib_data::<MaterialIdType, CellIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_cells()],
    )
    .unwrap();

    let mut fixed = vec![0; mesh.num_vertices()];
    fixed
        .iter_mut()
        .zip(mesh.vertex_positions())
        .for_each(|(f, p)| {
            if p[0] <= 0.0 {
                *f = 1;
            }
        });
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed)
        .expect("Failed to add fixed attribute to beam tetmesh");
    mesh
}

#[test]
fn beam() -> Result<(), Error> {
    init_logger();
    let params = softy::io::load_nl_params("assets/beam_nl_params.ron").unwrap();
    let material: Material = load_material("assets/solid_beam_material.ron")?;
    #[cfg(not(debug_assertions))]
    let mesh = make_beam(5);
    #[cfg(debug_assertions)]
    let mesh = make_beam(1);
    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh))
        .set_materials(vec![material.into()])
        .build::<f64>()?;
    // geo::io::save_mesh(&solver.mesh(), &format!("out/beam/beam{}.vtk", 0))?;
    let mut iterations = Vec::new();
    for _i in 0..200 {
        let step_result = solver.step()?;
        let result = step_result.first_solve_result();
        iterations.push(result.iterations);
        assert_eq!(result.status, Status::Success);
        // geo::io::save_mesh(&solver.mesh(), &format!("out/beam/beam{}.vtk", _i + 1))?;
    }
    eprintln!("iterations: {:?}", iterations);
    Ok(())
}

#[cfg(not(debug_assertions))]
#[test]
fn large_beam() -> Result<(), Error> {
    init_logger();
    let params = softy::io::load_nl_params("assets/beam_nl_params.ron").unwrap();
    let material: Material = load_material("assets/solid_beam_material.ron")?;
    let mesh = make_beam(5);
    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh))
        .set_materials(vec![material.into()])
        .build::<f64>()?;
    // geo::io::save_mesh(&solver.mesh(), &format!("out/beam/beam{}.vtk", 0))?;
    let mut iterations = Vec::new();
    for _i in 0..11 {
        let step_result = solver.step()?;
        let result = step_result.first_solve_result();
        iterations.push(result.iterations);
        assert_eq!(result.status, Status::Success);
        geo::io::save_mesh(&solver.mesh(), &format!("out/beam/beam{}.vtk", _i + 1))?;
    }
    eprintln!("iterations: {:?}", iterations);
    Ok(())
}
