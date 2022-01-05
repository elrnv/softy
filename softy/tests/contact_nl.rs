mod test_utils;

use approx::*;
use geo::algo::{Merge, TypedMesh};
use geo::attrib::Attrib;
use geo::mesh::builder::PlatonicSolidBuilder;
use geo::mesh::topology::*;
use geo::mesh::VertexPositions;
use softy::fem::nl::{SimParams, SolverBuilder};
use softy::nl_fem::Status::Success;
use softy::*;
use std::path::PathBuf;
use test_utils::*;

pub fn medium_solid_material() -> SolidMaterial {
    default_solid().with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
}
pub fn medium_shell_material() -> SoftShellMaterial {
    default_shell().with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
}

fn compute_distance_potential_tetmesh(
    sample_mesh: &TriMesh,
    tetmesh: &TetMesh,
    kernel: KernelType,
) -> Vec<f32> {
    compute_distance_potential(sample_mesh, &tetmesh.surface_trimesh(), kernel, true)
}

fn compute_distance_potential(
    sample_mesh: &TriMesh,
    surface_trimesh: &TriMesh,
    kernel: KernelType,
    test_f64: bool,
) -> Vec<f32> {
    use implicits::*;

    // There are currently two different ways to compute the implicit function representing the
    // contact constraint. Since this is a test we do it both ways and make sure the result is
    // the same. This doubles as a test for the implicits package.

    let mut trimesh_copy = PolyMesh::from(sample_mesh.clone());

    let params = implicits::Params {
        kernel,
        background_field: BackgroundFieldParams {
            field_type: BackgroundFieldType::DistanceBased,
            weighted: false,
        },
        sample_type: SampleType::Face,
        ..Default::default()
    };

    let mut surface_polymesh = PolyMesh::from(surface_trimesh.clone());
    compute_potential_debug(&mut trimesh_copy, &mut surface_polymesh, params, || false)
        .expect("Failed to compute constraint value");

    let pot_attrib = trimesh_copy
        .direct_attrib_clone_into_vec::<f32, VertexIndex>("potential")
        .expect("Potential attribute doesn't exist");

    if test_f64 {
        let surf =
            mls_from_trimesh(&surface_trimesh, params).expect("Failed to build implicit surface.");

        let query_surf = surf.query_topo(sample_mesh.vertex_positions());

        let mut pot_attrib64 = vec![0.0f64; sample_mesh.num_vertices()];
        query_surf.potential(sample_mesh.vertex_positions(), &mut pot_attrib64);

        // Make sure the two potentials are identical.
        log::trace!("potential = {:?}", pot_attrib);
        log::trace!("potential64 = {:?}", pot_attrib64);
        for (&x, &y) in pot_attrib.iter().zip(pot_attrib64.iter()) {
            assert_relative_eq!(x, y as f32, max_relative = 1e-5);
        }
    }

    pot_attrib.into_iter().map(|x| x as f32).collect()
}

//fn make_grids(i: usize) -> (PolyMesh, PolyMesh) {
//    let mut bottom_grid = make_grid(i);
//    bottom_grid.translate([0.0, -0.39, 0.0]);
//
//    let mut top_grid = make_grid(i);
//    top_grid.reverse();
//    top_grid.translate([0.0, 0.39, 0.0]);
//    (top_grid, bottom_grid)
//}

//#[test]
//fn box_squish_full() -> Result<(), Error> {
//    init_logger();
//    let contact_box = make_box(4).scaled([1.0, 0.8, 1.0]);
//
//    let (top_grid, bottom_grid) = make_grids(10);
//
//    let params = SimParams {
//        time_step: Some(0.02),
//        ..DYNAMIC_NL_PARAMS
//    };
//
//    let fc_point = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 1.1,
//            tolerance: 0.001,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let mut solver = SolverBuilder::new(params.clone())
//        .add_solid(contact_box.clone(), medium_solid_material())
//        .add_fixed(top_grid.clone(), 1)
//        .add_fixed(bottom_grid.clone(), 1)
//        .add_frictional_contact(fc_point, (1, 0))
//        .build::<f64>()?;
//
//    for _ in 0..10 {
//        solver.step()?;
//    }
//    let expected = geo::io::load_tetmesh("assets/box_squished.vtk")?;
//    let solution = &solver.solid(0).tetmesh;
//    compare_meshes(solution, &expected, 1e-4);
//
//    Ok(())
//}
//
//#[test]
//fn box_squish_linearized() -> Result<(), Error> {
//    init_logger();
//    let contact_box = make_box(4).scaled([1.0, 0.8, 1.0]);
//
//    let (top_grid, bottom_grid) = make_grids(10);
//
//    let params = SimParams {
//        time_step: Some(0.02),
//        ..DYNAMIC_NL_PARAMS
//    };
//
//    let fc_params = FrictionalContactParams {
//        contact_type: ContactType::LinearizedPoint,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 1.1,
//            tolerance: 0.001,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let mut solver = SolverBuilder::new(params)
//        .add_solid(contact_box.clone(), medium_solid_material())
//        .add_fixed(top_grid, 1)
//        .add_fixed(bottom_grid, 1)
//        .add_frictional_contact(fc_params, (1, 0))
//        .build::<f64>()?;
//
//    for _ in 0..10 {
//        solver.step()?;
//    }
//    let expected = geo::io::load_tetmesh("assets/box_squished.vtk")?;
//    let solution = &solver.solid(0).tetmesh;
//    compare_meshes(solution, &expected, 1e-4);
//    Ok(())
//}
//

#[test]
fn tri_push() -> Result<(), Error> {
    init_logger();
    // A triangle is being pushed on top of a tet.
    let tri_verts = vec![
        [0.0, -0.1, 0.0],
        [0.0866026, 0.05, 0.0],
        [-0.0866026, 0.05, 0.0],
        [0.0, 0.0, 0.001],
        [0.0866026, 0.0, 100.0], // far away to avoid being part of the solve
        [-0.0866026, 0.0, 100.0],
    ];
    let tri = vec![[0, 1, 2], [3, 4, 5]];
    let mut trimesh = TriMesh::new(tri_verts.clone(), tri);
    trimesh
        .insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 1, 1, 1, 1, 1])?;
    trimesh.insert_attrib_data::<ObjectIdType, FaceIndex>(OBJECT_ID_ATTRIB, vec![1, 0])?;

    let meshes = trimesh.clone().split_by_face_partition(&vec![0, 1], 2);

    compute_distance_potential(
        &meshes.0[1],
        &meshes.0[0],
        KernelType::Approximate {
            radius_multiplier: 10000.0,
            tolerance: 0.01,
        },
        true,
    );

    // Set contact parameters
    let kernel = KernelType::Approximate {
        radius_multiplier: 1.59,
        tolerance: 0.001,
    };

    let params = SimParams {
        jacobian_test: true,
        time_step: Some(0.2),
        ..static_nl_params()
    };

    let mut solver = SolverBuilder::new(params.clone())
        .set_mesh(Mesh::from(trimesh.clone()))
        .set_materials(vec![medium_shell_material().with_id(1).into()])
        .add_frictional_contact(
            FrictionalContactParams {
                contact_type: ContactType::Point,
                kernel,
                contact_offset: 0.0,
                use_fixed: true,
                friction_params: None,
            },
            (1, 0),
        )
        .build::<f64>()?;

    let solve_result = solver.step()?;
    assert_eq!(solve_result.status, Success);

    let mesh = solver.mesh();

    // Expect no push since the triangle is outside the surface.
    assert_relative_eq!(
        mesh.vertex_positions[0][2],
        tri_verts[0][2],
        max_relative = 1e-5,
        epsilon = 1e-6
    );

    // The triangle should be a bit lower due to gravity
    assert!(mesh.vertex_positions[0][1] < -0.1);

    // Push the static triangle vertex into the hanging triangle.
    let offset = 0.0015;
    let mut all_verts: Vec<_> = tri_verts.clone();
    all_verts[3][2] -= offset;

    let pts = PointCloud::new(all_verts);
    assert!(solver.update_vertices(&pts).is_ok());
    let solve_result = solver.step()?;
    assert_eq!(solve_result.status, Success);
    assert!(solve_result.iterations < params.max_iterations);

    let split_into_parts = |result_mesh: Mesh| {
        let typed_meshes = result_mesh.split_into_typed_meshes();
        typed_meshes
            .iter()
            .find_map(|mesh| {
                if let TypedMesh::Tri(mesh) = mesh {
                    let mut partition = mesh
                        .clone()
                        .split_by_face_partition(&vec![0, 1], 2)
                        .0
                        .into_iter();
                    Some((partition.next().unwrap(), partition.next().unwrap()))
                } else {
                    None
                }
            })
            .unwrap()
    };
    let (obj, coll) = split_into_parts(solver.mesh());

    let constraint = compute_distance_potential(&coll, &obj, kernel, false);
    assert!(
        constraint.iter().all(|&x| x >= -params.contact_tolerance),
        "Distance potential still negative after push: {:?} ",
        &constraint
    );

    // Check that the free vertex moved away
    assert!(obj.vertex_positions[0][2] < -offset);

    Ok(())
}

#[test]
fn tet_push() -> Result<(), Error> {
    init_logger();
    // A triangle is being pushed on top of a tet.
    let height = 1.18032;
    let tri_verts = vec![
        [0.1, height, 0.0],
        [-0.05, height, 0.0866026],
        [-0.05, height, -0.0866026],
    ];
    let tri = vec![[0, 2, 1]];
    let orig_trimesh = TriMesh::new(tri_verts.clone(), tri);
    let mut trimesh = Mesh::from(orig_trimesh.clone());
    trimesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 1])?;
    trimesh.insert_attrib_data::<ObjectIdType, CellIndex>(OBJECT_ID_ATTRIB, vec![0])?;

    let orig_tetmesh = PlatonicSolidBuilder::build_tetrahedron();
    let mut mesh = Mesh::from(orig_tetmesh.clone());
    mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 1, 1, 1])?;
    mesh.insert_attrib_data::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB, vec![1])?;
    mesh.insert_attrib_data::<ObjectIdType, CellIndex>(OBJECT_ID_ATTRIB, vec![1])?;

    mesh.merge(trimesh);

    // Set contact parameters
    let kernel = KernelType::Approximate {
        radius_multiplier: 1.59,
        tolerance: 0.001,
    };

    compute_distance_potential_tetmesh(&orig_trimesh, &orig_tetmesh, kernel);

    let params = SimParams {
        gravity: [0.0f32; 3],
        jacobian_test: true,
        time_step: Some(1.2),
        ..static_nl_params()
    };

    let mut solver = SolverBuilder::new(params.clone())
        .set_mesh(mesh.clone())
        .set_materials(vec![medium_solid_material().with_id(1).into()])
        .add_frictional_contact(
            FrictionalContactParams {
                contact_type: ContactType::Point,
                kernel,
                contact_offset: 0.0,
                use_fixed: true,
                friction_params: None,
            },
            (1, 0),
        )
        .build::<f64>()?;

    let solve_result = solver.step()?;
    assert_eq!(solve_result.iterations, 1); // should be no more than one outer iteration

    let split_into_parts = |result_mesh: Mesh| {
        let typed_meshes = result_mesh.split_into_typed_meshes();
        let tetmesh = typed_meshes
            .iter()
            .find_map(|mesh| {
                if let TypedMesh::Tet(mesh) = mesh {
                    Some(mesh.clone())
                } else {
                    None
                }
            })
            .unwrap();
        let trimesh = typed_meshes
            .iter()
            .find_map(|mesh| {
                if let TypedMesh::Tri(mesh) = mesh {
                    Some(mesh.clone())
                } else {
                    None
                }
            })
            .unwrap();
        (tetmesh, trimesh)
    };

    let (tetmesh, trimesh) = split_into_parts(solver.mesh());

    // Expect no push since the triangle is outside the surface.
    for (pos, exp_pos) in solver
        .mesh()
        .vertex_position_iter()
        .zip(mesh.vertex_positions().iter())
    {
        for i in 0..3 {
            assert_relative_eq!(pos[i], exp_pos[i], max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    // Verify constraint, should be positive before push
    let constraint = compute_distance_potential_tetmesh(&trimesh, &tetmesh, kernel);
    assert!(constraint.iter().all(|&x| x >= 0.0f32));

    // Simulate push
    let offset = 0.34;
    let all_verts: Vec<_> = orig_tetmesh
        .vertex_position_iter()
        .cloned()
        .chain(tri_verts.iter().map(|&[x, y, z]| [x, y - offset, z]))
        .collect();
    let pts = PointCloud::new(all_verts);
    assert!(solver.update_vertices(&pts).is_ok());
    let solve_result = solver.step()?;
    assert!(solve_result.iterations <= params.max_iterations);

    let mesh = solver.mesh();
    let (tetmesh, trimesh) = split_into_parts(mesh.clone());

    // Verify constraint, should be positive after push
    let constraint = compute_distance_potential_tetmesh(&trimesh, &tetmesh, kernel);
    assert!(
        constraint.iter().all(|&x| x >= -params.contact_tolerance),
        "Distance potential still negative after push: {:?} ",
        &constraint
    );

    // Expect only the top vertex to be pushed down.
    let pos = mesh.vertex_position(0);
    let exp_pos = [0.0, 0.629, 0.0];
    for i in 0..3 {
        assert_relative_eq!(pos[i], exp_pos[i], epsilon = 1e-3);
    }

    Ok(())
}

fn ball_tri_push_tester(
    material: SolidMaterial,
    fc_params: FrictionalContactParams,
) -> Result<(), Error> {
    init_logger();
    let mut tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball_fixed.vtk")).unwrap();
    tetmesh.insert_attrib_data::<MaterialIdType, CellIndex>(
        MATERIAL_ID_ATTRIB,
        vec![1; tetmesh.num_cells()],
    )?;
    tetmesh.insert_attrib_data::<ObjectIdType, CellIndex>(
        OBJECT_ID_ATTRIB,
        vec![1; tetmesh.num_cells()],
    )?;

    geo::io::save_tetmesh(&tetmesh, "./out/tetmesh_before.vtk");
    let params = static_nl_params();

    // If material is omitted it is assumed to be material 0 which is a completely fixed/animated mesh.
    let mut polymesh = geo::io::load_polymesh(&PathBuf::from("assets/tri.vtk"))?;
    polymesh.insert_attrib_data::<MaterialIdType, FaceIndex>(
        MATERIAL_ID_ATTRIB,
        vec![0; polymesh.num_faces()],
    )?;
    polymesh.insert_attrib_data::<ObjectIdType, FaceIndex>(
        OBJECT_ID_ATTRIB,
        vec![0; polymesh.num_faces()],
    )?;
    polymesh.insert_attrib_data::<FixedIntType, VertexIndex>(
        FIXED_ATTRIB,
        vec![1; polymesh.num_vertices()],
    )?;
    let mut mesh = Mesh::from(TriMesh::from(polymesh));
    mesh.merge(Mesh::from(tetmesh));

    let mut solver = SolverBuilder::new(params.clone())
        .set_mesh(mesh)
        .set_materials(vec![
            FixedMaterial::new(0).into(),
            material.with_id(1).into(),
        ])
        .add_frictional_contact(fc_params, (0, 1))
        .build::<f64>()?;

    let res = solver.step()?;

    geo::io::save_mesh(&solver.mesh(), "./out/mesh_after.vtk");

    //println!("res = {:?}", res);
    assert!(
        res.iterations <= params.max_iterations,
        "Exceeded max outer iterations."
    );
    Ok(())
}

#[test]
fn ball_tri_push_plain() -> Result<(), Error> {
    let material =
        default_solid().with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4));
    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 19.812,
            tolerance: 0.07,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: None,
    };

    ball_tri_push_tester(material, fc_params)
}

//#[test]
//fn ball_tri_push_volume_constraint() -> Result<(), Error> {
//    let material = default_solid()
//        .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
//        .with_volume_preservation(true);
//    let fc_params = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 1.812,
//            tolerance: 0.07,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    ball_tri_push_tester(material, fc_params)
//}
//
//fn ball_bounce_tester(
//    material: SolidMaterial,
//    fc_params: FrictionalContactParams,
//    tetmesh: TetMesh,
//    implicit_index: usize,
//    points_index: usize,
//) -> Result<(), Error> {
//    init_logger();
//    let params = SimParams {
//        max_linsolve_iterations: 200,
//        tolerance: 0.1,
//        max_iterations: 20,
//        gravity: [0.0f32, -9.81, 0.0],
//        time_step: Some(0.0208333),
//        ..DYNAMIC_NL_PARAMS
//    };
//
//    let mut grid = GridBuilder {
//        rows: 4,
//        cols: 4,
//        orientation: AxisPlaneOrientation::ZX,
//    }
//    .build();
//
//    grid.scale([3.0, 1.0, 3.0]);
//    grid.translate([0.0, -3.0, 0.0]);
//
//    let mut solver = SolverBuilder::new(params.clone())
//        .add_solid(tetmesh, material.with_id(0))
//        .add_fixed(grid, 1)
//        .add_frictional_contact(fc_params, (implicit_index, points_index))
//        .build::<f64>()?;
//
//    for _ in 0..50 {
//        let res = solver.step()?;
//        //println!("res = {:?}", res);
//        //geo::io::save_tetmesh(
//        //    &solver.borrow_mesh(),
//        //    &PathBuf::from(format!("./out/mesh_{}.vtk", i)),
//        //    ).unwrap();
//        assert!(
//            res.iterations <= params.max_iterations,
//            "Exceeded max iterations."
//        );
//    }
//
//    Ok(())
//}
//
//#[test]
//fn ball_bounce_on_points_plain() -> Result<(), Error> {
//    let material =
//        default_solid().with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4));
//
//    let sc_params = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 1.1,
//            tolerance: 0.01,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;
//
//    ball_bounce_tester(material, sc_params, tetmesh, 0, 1)
//}
//
//#[test]
//fn ball_bounce_on_points_volume_constraint() -> Result<(), Error> {
//    let material = default_solid()
//        .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
//        .with_volume_preservation(true);
//
//    let sc_params = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 1.1,
//            tolerance: 0.01,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;
//
//    ball_bounce_tester(material, sc_params, tetmesh, 0, 1)
//}
//
///// Tet bouncing on an implicit surface. This is an easy test where the tet sees the entire
///// local implicit surface.
//#[test]
//fn tet_bounce_on_implicit() -> Result<(), Error> {
//    let material =
//        default_solid().with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));
//
//    let sc_params = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 20.0, // deliberately large radius
//            tolerance: 0.0001,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let tetmesh = PlatonicSolidBuilder::build_tetrahedron();
//
//    ball_bounce_tester(material, sc_params, tetmesh, 1, 0)
//}
//
///// Ball bouncing on an implicit surface.
//#[test]
//fn ball_bounce_on_implicit_plain() -> Result<(), Error> {
//    let material =
//        default_solid().with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));
//
//    let sc_params = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 2.0,
//            tolerance: 0.0001,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;
//
//    ball_bounce_tester(material, sc_params, tetmesh, 1, 0)
//}
//
///// Ball with constant volume bouncing on an implicit surface.
//#[test]
//fn ball_bounce_on_implicit_volume_constraint() -> Result<(), Error> {
//    let material = default_solid()
//        .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
//        .with_volume_preservation(true);
//
//    let sc_params = FrictionalContactParams {
//        contact_type: ContactType::Point,
//        kernel: KernelType::Approximate {
//            radius_multiplier: 2.0,
//            tolerance: 0.0001,
//        },
//        contact_offset: 0.0,
//        use_fixed: true,
//        friction_params: None,
//    };
//
//    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;
//
//    ball_bounce_tester(material, sc_params, tetmesh, 1, 0)
//}
//
///// Two tets in contact with each-other. This test verifies that the contact is resolved.
//#[test]
//fn two_tets_in_contact() -> Result<(), Error> {
//    init_logger();
//    let mut tet_bottom = PlatonicSolidBuilder::build_tetrahedron();
//    let mut tet_top = PlatonicSolidBuilder::build_tetrahedron();
//    tet_top.translate([0.0, 0.9, 0.0]); // translate up by 0.3
//
//    // Fix bottom tet at the base.
//    tet_bottom.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 1, 1, 1])?;
//
//    // Fix top tet at the peak vertex and one other vertex to simplify the problem.
//    tet_top.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![1, 1, 0, 0])?;
//
//    // Set contact parameters
//    let kernel = KernelType::Approximate {
//        radius_multiplier: 2.0,
//        tolerance: 0.001,
//    };
//
//    let params = SimParams {
//        max_iterations: 20,
//        gravity: [0.0f32, 0.0, 0.0],
//        //print_level: 5,
//        //derivative_test: 2,
//        ..STATIC_NL_PARAMS
//    };
//
//    let mut solver = SolverBuilder::new(params.clone())
//        .add_solid(tet_top.clone(), medium_solid_material().with_id(0))
//        .add_solid(tet_bottom.clone(), medium_solid_material().with_id(1))
//        .add_frictional_contact(
//            FrictionalContactParams {
//                contact_type: ContactType::Point,
//                kernel,
//                contact_offset: 0.0,
//                use_fixed: true,
//                friction_params: None,
//            },
//            (0, 1),
//        )
//        .build::<f64>()?;
//
//    let solve_result = solver.step()?;
//
//    //use std::path::PathBuf;
//
//    //let resmesh1 = &solver.solid(0).tetmesh;
//    //let resmesh2 = &solver.solid(1).tetmesh;
//    //geo::io::save_tetmesh(resmesh1, &PathBuf::from("./out/mesh1.vtk"));
//    //geo::io::save_tetmesh(resmesh2, &PathBuf::from("./out/mesh2.vtk"));
//    assert_eq!(solve_result.iterations, 1); // should be no more than one outer iteration
//    Ok(())
//}
