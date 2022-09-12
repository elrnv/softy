use tensr::{IntoData, Vector3};

use geo::attrib::Attrib;
use geo::builder::AxisPlaneOrientation;
use geo::mesh::VertexPositions;
use geo::topology::{CellVertexIndex, FaceIndex, FaceVertexIndex, NumFaces};
use softy::nl_fem::*;
use softy::*;
pub use test_utils::*;

mod test_utils;

fn soft_shell_material() -> Result<Material, LoadConfigError> {
    softy::io::load_material("assets/soft_shell_material.ron")
}

/// Test that cell vertex coordiantes remain unchanged after the solve.
#[test]
fn cell_vertex_attributes() {
    init_logger();
    let params = SimParams {
        max_iterations: 1,
        gravity: [0.0f32; 3],
        time_step: Some(0.01),
        ..static_nl_params(2)
    };

    let mut mesh = TriMesh::from(
        geo::builder::GridBuilder {
            rows: 3,
            cols: 3,
            orientation: AxisPlaneOrientation::XY,
        }
        .build(),
    );

    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    let verts = mesh.vertex_positions();

    let verts_f32: Vec<[f32; 3]> = mesh
        .face_iter()
        .flat_map(|face| {
            face.iter()
                .map(|&x| Vector3::new(verts[x]).cast::<f32>().into_data())
        })
        .collect();
    mesh.insert_attrib_data::<RefPosType, FaceVertexIndex>(
        REFERENCE_CELL_VERTEX_POS_ATTRIB,
        verts_f32.clone(),
    )
    .unwrap();

    let mut solver = SolverBuilder::new(params)
        .set_mesh(Mesh::from(mesh.clone()))
        .set_materials(vec![soft_shell_material().unwrap(); 1])
        .build::<f64>()
        .expect("Failed to build a solver for a three triangle test.");

    let result = solver.step();
    assert!(result.is_ok());

    // Expect the triangles to remain in original configuration
    let soln_mesh = solver.mesh();
    let ref_pos = soln_mesh
        .attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)
        .unwrap();

    // Make sure the cell vertex attributes remain the same.
    assert_eq!(&verts_f32, &ref_pos);
}
