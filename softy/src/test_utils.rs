use tensr::{IntoData, Vector3};

use crate::attrib_defines::*;
use crate::fem::nl::{state::VertexType, SimParams as NLParams};
use crate::objects::*;
use crate::{PolyMesh, TetMesh, TriMesh};
use geo::attrib::*;
use geo::mesh::builder::*;
use geo::mesh::topology::{CellIndex, FaceIndex, NumCells, NumFaces, VertexIndex};
use geo::mesh::VertexPositions;
use geo::ops::*;
use geo::topology::{CellVertex, CellVertexIndex};

/// Returns the total number of configurations availble.
pub fn num_static_configs() -> u32 {
    4
}

/// Returns an iterator of supported configs.
pub fn static_configs() -> impl Iterator<Item = u32> {
    static_config_slice().iter().cloned()
}

#[cfg(not(target_os = "macos"))]
#[cfg(not(feature = "mkl"))]
pub fn static_config_slice() -> &'static [u32] {
    // Exclude direct solver configs since these are not supported on non macOS platforms without MKL
    &[2, 3]
}

#[cfg(not(target_os = "macos"))]
#[cfg(feature = "mkl")]
pub fn static_config_slice() -> &'static [u32] {
    &[0, 1, 2, 3]
}

#[cfg(target_os = "macos")]
pub fn static_config_slice() -> &'static [u32] {
    &[0, 1, 2, 3]
}

/// Gets the name of the config which points to a ron file in the assets directory.
pub fn config_name(config: u32) -> &'static str {
    assert!(config < num_static_configs());
    match config {
        0 => "direct_static_nl_params",
        1 => "direct_assisted_static_nl_params",
        2 => "iterative_static_nl_params",
        _ => "iterative_assisted_static_nl_params",
    }
}

/// Get a sample configuration. These should all be tested to make sure all work.
pub fn static_nl_params(config: u32) -> NLParams {
    crate::io::load_nl_params(&format!("assets/{}.ron", config_name(config))).unwrap()
}

pub fn vertex_types_from_fixed(fixed: &[FixedIntType]) -> Vec<VertexType> {
    fixed
        .iter()
        .map(|&x| {
            if x == 1 {
                VertexType::Fixed
            } else {
                VertexType::Free
            }
        })
        .collect()
}

// Note: The key to getting reliable simulations here is to keep bulk_modulus, shear_modulus
// (mu) and density in the same range of magnitude. Higher stiffnesses compared to density will
// produce highly oscillatory configurations and keep the solver from converging fast.
// As an example if we increase the moduli below by 1000, the solver can't converge, even in
// 300 steps.
pub fn default_solid() -> SolidMaterial {
    SolidMaterial::new(0)
        .with_elasticity(Elasticity::from_lame(
            93333.33,
            10e3,
            ElasticityModel::StableNeoHookean,
        ))
        .with_density(1000.0)
}

pub fn default_shell() -> SoftShellMaterial {
    SoftShellMaterial::new(0)
        .with_elasticity(Elasticity::from_lame(
            93333.33,
            10e3,
            ElasticityModel::NeoHookean,
        ))
        .with_density(1000.0)
}

/// A flat triangle in the xz plane with vertices at origin, in positive x and positive z
/// directions with the first two vertices fixed such that there is a unique solution when the
/// triangle is simulated under gravity.
pub fn make_one_tri_mesh() -> TriMesh {
    let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let indices = vec![[0, 2, 1]];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    let fixed = vec![1, 1, 0];
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();
    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<_, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
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
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.25], // slight bend
        [1.0, 1.0, 0.0],
    ];

    let indices = vec![[2, 0, 1], [0, 3, 1]];
    let mut mesh = TriMesh::new(verts.clone(), indices);

    let fixed = vec![0, 1, 0, 1];
    // Top two vertices are fixed to remove the nullspace in shell sims.
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();

    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    // Reference configuration is flat.
    verts[2][2] = 0.0;

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
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

    let indices = vec![[0, 1, 2], [0, 3, 1], [1, 4, 2]];
    let mut mesh = TriMesh::new(verts.clone(), indices);

    let fixed = vec![0, 0, 1, 0, 1];

    // Top two vertices are fixed to remove the nullspace in shell sims.
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();

    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    // Reference configuration is bent the other way.
    verts[3][2] = -0.5;

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
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
    let indices = vec![[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    let fixed = vec![1, 1, 0, 0, 0, 0];
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();

    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    // Reference configuration is flat.
    verts[5][1] = 0.0;

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
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
    let indices = vec![[0, 2, 1], [2, 3, 1], [2, 3, 4], [4, 3, 5]];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    let fixed = vec![1, 1, 0, 0, 0, 0];
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();
    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
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
    let indices = vec![[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]];
    let mut mesh = TriMesh::new(verts.clone(), indices);
    let fixed = vec![1, 1, 0, 0];
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();
    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_faces()],
    )
    .unwrap();

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
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
    let indices = vec![[0, 2, 1, 3]];
    let mut mesh = TetMesh::new(verts.clone(), indices);
    let fixed = vec![1, 1, 0, 0];
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();
    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, CellIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_cells()],
    )
    .unwrap();

    let verts_f32: Vec<_> = verts
        .iter()
        .map(|&x| Vector3::new(x).cast::<f32>().into_data())
        .collect();
    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, verts_f32)
        .unwrap();
    mesh
}

pub fn make_one_deformed_tet_mesh() -> TetMesh {
    let mut mesh = make_one_tet_mesh();
    mesh.vertex_positions_mut()[3][2] = 2.0;
    mesh.vertex_positions_mut()[1][0] = -1.0;
    mesh.vertex_positions_mut()[2][1] = -1.0;
    mesh
}

pub fn make_three_tet_mesh_with_verts(verts: Vec<[f64; 3]>) -> TetMesh {
    let indices = vec![[5, 2, 4, 0], [3, 2, 5, 0], [1, 0, 3, 5]];
    let mut mesh = TetMesh::new(verts, indices);
    let fixed = vec![0, 0, 1, 1, 0, 0];
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();
    mesh.insert_attrib_data::<VertexType, VertexIndex>(
        VERTEX_TYPE_ATTRIB,
        vertex_types_from_fixed(&fixed),
    )
    .unwrap();
    mesh.insert_attrib_data::<MaterialIdType, CellIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; mesh.num_cells()],
    )
    .unwrap();

    let ref_verts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];

    mesh.insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, ref_verts)
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
    let mut box_mesh: TetMesh = BoxBuilder {
        divisions: [i as u32 - 1; 3],
    }
    .build();
    box_mesh.uniform_scale(0.5);
    let ref_verts = box_mesh
        .vertex_position_iter()
        .map(|&[a, b, c]| [a as f32, b as f32, c as f32])
        .collect();
    box_mesh
        .insert_attrib_data::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB, ref_verts)
        .expect("Failed to add reference positions to box tetmesh");
    let ref_pos_attrib = box_mesh
        .attrib::<VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
        .unwrap()
        .promote_with_len(box_mesh.num_cell_vertices(), |output, input| {
            for (cvi, mut out) in output.into_iter().enumerate() {
                let ci = cvi / 4;
                let vi = cvi % 4;
                out.clone_from_other(
                    input.get(box_mesh.cell_to_vertex(ci, vi).unwrap().into_inner()),
                )
                .unwrap();
            }
        });
    box_mesh
        .attrib_dict_mut::<CellVertexIndex>()
        .insert(REFERENCE_CELL_VERTEX_POS_ATTRIB.to_string(), ref_pos_attrib);
    box_mesh
        .insert_attrib_data::<MaterialIdType, CellIndex>(
            MATERIAL_ID_ATTRIB,
            vec![0; box_mesh.num_cells()],
        )
        .unwrap();
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
        .insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed.clone())
        .unwrap();
    box_mesh
        .insert_attrib_data::<VertexType, VertexIndex>(
            VERTEX_TYPE_ATTRIB,
            vertex_types_from_fixed(&fixed),
        )
        .unwrap();

    box_mesh
}

pub fn make_grid(i: usize) -> PolyMesh {
    GridBuilder {
        rows: i,
        cols: i,
        orientation: AxisPlaneOrientation::ZX,
    }
    .build()
}

pub fn init_logger() {
    #[cfg(test)]
    #[cfg(debug_assertions)]
    let _ = env_logger::Builder::from_env("SOFTY_LOG")
        .is_test(true)
        .try_init();
}
