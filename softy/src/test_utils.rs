use crate::attrib_defines::*;
use crate::TetMesh;
use geo::mesh::attrib::*;
use geo::mesh::topology::VertexIndex;
use geo::mesh::VertexPositions;

pub fn make_one_tet_mesh() -> TetMesh {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![0, 2, 1, 3];
    let mut mesh = TetMesh::new(verts.clone(), indices);
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0, 0])
        .unwrap();

    mesh.add_attrib_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts)
        .unwrap();
    mesh
}

pub fn make_one_deformed_tet_mesh() -> TetMesh {
    let mut mesh = make_one_tet_mesh();
    mesh.vertex_positions_mut()[3][2] = 2.0;
    mesh
}

pub fn make_three_tet_mesh_with_verts(verts: Vec<[f64; 3]>) -> TetMesh {
    let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
    let mut mesh = TetMesh::new(verts, indices);
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 0, 1, 1, 0, 0])
        .unwrap();

    let ref_verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];

    mesh.add_attrib_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, ref_verts)
        .unwrap();
    mesh
}

pub fn make_three_tet_mesh() -> TetMesh {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
    ];
    make_three_tet_mesh_with_verts(verts)
}
