use crate::attrib_defines::*;
use crate::fem::*;
use crate::objects::*;
use crate::{TriMesh, TetMesh, PolyMesh};
use geo::mesh::attrib::*;
use geo::mesh::topology::VertexIndex;
use geo::mesh::VertexPositions;
use geo::mesh::builder::*;
use geo::ops::*;
use utils::soap::{Vector3, IntoData};

pub const STATIC_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, -9.81, 0.0],
    time_step: None,
    clear_velocity: false,
    tolerance: 1e-10,
    max_iterations: 300,
    max_outer_iterations: 1,
    friction_iterations: 0,
    outer_tolerance: 0.001,
    print_level: 0,
    derivative_test: 0,
    mu_strategy: MuStrategy::Adaptive,
    max_gradient_scaling: 5e-6,
    log_file: None,
};

//pub(crate) const QUASI_STATIC_PARAMS: SimParams = SimParams {
//    gravity: [0.0f32, 0.0, 0.0],
//    time_step: Some(0.01),
//    clear_velocity: true,
//    ..STATIC_PARAMS
//};

pub const DYNAMIC_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, 0.0, 0.0],
    time_step: Some(0.01),
    ..STATIC_PARAMS
};

// Note: The key to getting reliable simulations here is to keep bulk_modulus, shear_modulus
// (mu) and density in the same range of magnitude. Higher stiffnesses compared to denisty will
// produce highly oscillatory configurations and keep the solver from converging fast.
// As an example if we increase the moduli below by 1000, the solver can't converge, even in
// 300 steps.
pub fn default_solid() -> SolidMaterial {
    SolidMaterial::new(0)
        .with_elasticity(ElasticityParameters {
            lambda: 93333.33,
            mu: 10e3,
            model: ElasticityModel::NeoHookean,
        })
        .with_density(1000.0)
}

/// A flat triangle in the xz plane with vertices at origin, in positive x and positive z
/// directions with the first two vertices fixed such that there is a unique solution when the
/// triangle is simulated under gravity.
pub fn make_one_tri_mesh() -> TriMesh {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![0, 2, 1];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0])
        .unwrap();

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>().into_data()).collect();
    mesh.add_attrib_data::<_, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

/// The triangle from `make_one_tri_mesh` sample deformed at the unfixed vertex
pub fn make_one_deformed_tri_mesh() -> TriMesh {
    let mut mesh = make_one_tri_mesh();
    mesh.vertex_positions_mut()[2][2] = 2.0;
    mesh
}

/// Two triangles forming a quad unconstrained undeformed.
pub fn make_two_tri_mesh() -> TriMesh {
    let mut verts = vec![
        [1.0, 0.0, 0.25],  // slight bend
        [0.0, 1.0, 0.0],
        [0.0; 3],
        [1.0, 1.0, 0.0],
    ];

    let indices = vec![2, 0, 1, 0, 3, 1];
    let mut mesh = TriMesh::new(verts.clone(), indices);

    // Top two vertices are fixed to remove the nullspace in shell sims.
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 1, 0, 1]).unwrap();

    // Reference configuration is flat.
    verts[0][2] = 0.0;

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>()
        .into_data()).collect();
    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

/// Three triangle strip.
pub fn make_three_tri_mesh() -> TriMesh {
    let mut verts = vec![
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, -1.0, 0.5],
        [1.0, 1.0, 0.0],
    ];

    let indices = vec![0, 1, 2, 0, 3, 1, 1, 4, 2];
    let mut mesh = TriMesh::new(verts.clone(), indices);

    // Top two vertices are fixed to remove the nullspace in shell sims.
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 0, 1, 0, 1])
        .unwrap();

    // Reference configuration is bent the other way.
    verts[3][2] = -0.5;

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>().into_data()).collect();
    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

/// A strip of two quads in the xz plane fixed at two vertices.
pub fn make_four_tri_mesh() -> TriMesh {
    let mut verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.1, 2.0], // slight bend
    ];
    let indices = vec![0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0, 0, 0, 0])
        .unwrap();

    // Reference configuration is flat.
    verts[5][1] = 0.0;

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>().into_data()).collect();
    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

/// A strip of two quads in the xz plane fixed at two vertices.
/// First two quads are oriented in the opposite direction.
pub fn make_four_tri_mesh_unoriented() -> TriMesh {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
    ];
    let indices = vec![0, 2, 1, 2, 3, 1, 2, 3, 4, 4, 3, 5];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0, 0, 0, 0])
        .unwrap();

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>().into_data()).collect();
    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

pub fn make_one_tet_trimesh() -> TriMesh {
    let verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    mesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0, 0])
        .unwrap();

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>().into_data()).collect();
    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

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

    let verts_f32: Vec<_> = verts.iter().map(|&x| Vector3::new(x).cast::<f32>().into_data()).collect();
    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, verts_f32)
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

    mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, ref_verts)
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

/// Create a box of unit size centered at the origin with `i` cells in each dimension.
pub fn make_box(i: usize) -> TetMesh {
    let mut box_mesh = SolidBoxBuilder { res: [i, i, i] }.build();
    box_mesh.uniform_scale(0.5);
    let ref_verts = box_mesh.vertex_position_iter().map(|&[a,b,c]| [a as f32, b as f32, c as f32]).collect();
    box_mesh.add_attrib_data::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB, ref_verts)
        .expect("Failed to add reference positions to box tetmesh");
    box_mesh
}

/// Create a stretched box centered at the origin with `i` cells in each dimension.
/// The box is 3x1x1 with the vertices at the min and max x coordinates displaced to -1.5 and 1.5
/// respectively. The displace vertices are marked as fixed.
pub fn make_stretched_box(i: usize) -> TetMesh {
    let mut box_mesh = make_box(i);
    let mut fixed = vec![0i8; box_mesh.vertex_positions().len()];

    // Stretch along the x axis
    for (v, f) in box_mesh.vertex_position_iter_mut().zip(fixed.iter_mut()) {
        if v[0] == 0.5 {
            *f = 1;
            v[0] = 1.5;
        }
        if v[0] == -0.5 {
            *f = 1;
            v[0] = -1.5;
        }
    }

    box_mesh
        .add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed)
        .unwrap();

    box_mesh
}

pub fn make_grid(i: usize) -> PolyMesh {
    GridBuilder {
        rows: i,
        cols: i,
        orientation: AxisPlaneOrientation::ZX,
    }.build()
}
