mod test_utils;

use geo::mesh::attrib::*;
use geo::mesh::topology::*;
use softy::*;
pub use test_utils::*;

fn material() -> SolidMaterial {
    SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(1750e6, 10e6))
}

#[test]
fn sim_test() {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
    ];
    let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
    let mut mesh = TetMesh::new(verts, indices);
    mesh.add_attrib_data::<i8, VertexIndex>("fixed", vec![0, 0, 1, 1, 0, 0])
        .unwrap();

    let ref_verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];

    mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
        .unwrap();

    assert!(
        match sim(Some(mesh), material(), None, STATIC_PARAMS, None) {
            SimResult::Success(_) => true,
            _ => false,
        }
    );
}
