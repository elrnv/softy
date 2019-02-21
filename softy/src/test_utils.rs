use crate::TetMesh;
use geo::mesh::attrib::*;
use geo::mesh::topology::VertexIndex;
use geo::mesh::VertexPositions;
use crate::{Material, ElasticityParameters, SimParams, MuStrategy};
use crate::attrib_defines::*;

/*
 * Setup code
 */

pub(crate) const STATIC_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, -9.81, 0.0],
    time_step: None,
    tolerance: 1e-9,
    max_iterations: 300,
    max_outer_iterations: 1,
    outer_tolerance: 0.001,
    print_level: 0,
    derivative_test: 0,
    mu_strategy: MuStrategy::Adaptive,
    max_gradient_scaling: 1e-5,
};

pub(crate) const DYNAMIC_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, 0.0, 0.0],
    time_step: Some(0.01),
    ..STATIC_PARAMS
};

// Note: The key to getting reliable simulations here is to keep bulk_modulus, shear_modulus
// (mu) and density in the same range of magnitude. Higher stiffnesses compared to denisty will
// produce highly oscillatory configurations and keep the solver from converging fast.
// As an example if we increase the moduli below by 1000, the solver can't converge, even in
// 300 steps.
pub(crate) const SOLID_MATERIAL: Material = Material {
    elasticity: ElasticityParameters {
        bulk_modulus: 100e3,
        shear_modulus: 10e3,
    },
    incompressibility: false,
    density: 1000.0,
    damping: 0.0,
};

pub(crate) fn make_one_tet_mesh() -> TetMesh {
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

pub(crate) fn make_one_deformed_tet_mesh() -> TetMesh {
    let mut mesh = make_one_tet_mesh();
    mesh.vertex_positions_mut()[3][2] = 2.0;
    mesh
}

pub(crate) fn make_three_tet_mesh_with_verts(verts: Vec<[f64; 3]>) -> TetMesh {
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

pub(crate) fn make_three_tet_mesh() -> TetMesh {
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
