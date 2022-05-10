use geo::algo::Merge;
use geo::attrib::Attrib;
use geo::mesh::VertexPositions;
use geo::topology::{CellIndex, FaceIndex, NumCells, NumFaces, NumVertices, VertexIndex};
use geo::{Mesh, TriMesh};
use softy::nl_fem::{SimParams, SolverBuilder};
pub use softy::test_utils::*;
use softy::{
    constraints::FrictionalContactParams, Error, FixedIntType, MaterialIdType, ObjectIdType,
    PolyMesh, SolidMaterial, TetMesh, FIXED_ATTRIB, MATERIAL_ID_ATTRIB, OBJECT_ID_ATTRIB,
};

/// Utility function to compare positions of two meshes.
#[allow(dead_code)]
pub fn compare_meshes<M1, M2>(solution: &M1, expected: &M2, tol: f64)
where
    M1: VertexPositions<Element = [f64; 3]>,
    M2: VertexPositions<Element = [f64; 3]>,
{
    use approx::*;
    for (pos, expected_pos) in solution
        .vertex_positions()
        .iter()
        .zip(expected.vertex_positions().iter())
    {
        assert_relative_eq!(pos[0], expected_pos[0], max_relative = tol, epsilon = 5e-6);
        assert_relative_eq!(pos[1], expected_pos[1], max_relative = tol, epsilon = 5e-6);
        assert_relative_eq!(pos[2], expected_pos[2], max_relative = tol, epsilon = 5e-6);
    }
}

pub fn contact_tester(
    material: SolidMaterial,
    fc_params: FrictionalContactParams,
    tetmesh: TetMesh,
    surface: PolyMesh,
    implicit_tetmesh: bool,
    num_steps: u32,
) -> Result<(), Error> {
    init_logger();
    for config_idx in 0..num_static_configs() {
        let params = SimParams {
            time_step: Some(0.01),
            ..static_nl_params(config_idx)
        };
        let mut tetmesh = tetmesh.clone();
        let mut surface = surface.clone();

        tetmesh.insert_attrib_data::<MaterialIdType, CellIndex>(
            MATERIAL_ID_ATTRIB,
            vec![1; tetmesh.num_cells()],
        )?;
        tetmesh.insert_attrib_data::<ObjectIdType, CellIndex>(
            OBJECT_ID_ATTRIB,
            vec![1; tetmesh.num_cells()],
        )?;
        surface.insert_attrib_data::<ObjectIdType, FaceIndex>(
            OBJECT_ID_ATTRIB,
            vec![0; surface.num_faces()],
        )?;
        surface.insert_attrib_data::<FixedIntType, VertexIndex>(
            FIXED_ATTRIB,
            vec![1; surface.num_vertices()],
        )?;

        let coupling = if implicit_tetmesh { (1, 0) } else { (0, 1) };

        let mut mesh = Mesh::from(tetmesh);
        mesh.merge(Mesh::from(TriMesh::from(surface)));

        let mut solver = SolverBuilder::new(params.clone())
            .set_mesh(mesh)
            .set_materials(vec![material.with_id(1).into()])
            .add_frictional_contact(fc_params, coupling)
            .build::<f64>()?;

        let mut iterations = vec![];
        for i in 0..num_steps {
            log::debug!("step: {:?}", i);
            let step_result = solver.step()?;
            let res = step_result.first_solve_result();
            log::debug!("status = {:?}", res.status);
            log::debug!("iterations = {:?}", res.iterations);
            iterations.push(res.iterations);

            geo::io::save_mesh(&solver.mesh(), &format!("./out/solid_mesh_{}.vtk", i))?;
            assert!(
                res.iterations <= params.max_iterations,
                "Exceeded max outer iterations."
            );
        }
        eprintln!("{:?}", iterations);
    }

    Ok(())
}

pub fn init_logger() {
    let _ = env_logger::Builder::from_env("SOFTY_LOG")
        .is_test(true)
        .try_init();
}
