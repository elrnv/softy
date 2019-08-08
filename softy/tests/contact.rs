mod test_utils;

use approx::*;
use geo::mesh::attrib::Attrib;
use geo::mesh::topology::*;
use geo::mesh::VertexPositions;
use softy::*;
use std::path::PathBuf;
pub use test_utils::*;
use utils::*;

pub fn medium_solid_material() -> SolidMaterial {
    SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
}

fn compute_contact_constraint(
    sample_mesh: &PolyMesh,
    tetmesh: &TetMesh,
    kernel: KernelType,
) -> Vec<f32> {
    use implicits::*;

    // There are currently two different ways to compute the implicit function representing the
    // contact constraint. Since this is a test we do it both ways and make sure the result is
    // the same. This doubles as a test for the implicits package.

    let mut trimesh_copy = sample_mesh.clone();
    let surface_trimesh = tetmesh.surface_trimesh();

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
        .attrib_clone_into_vec::<f32, VertexIndex>("potential")
        .expect("Potential attribute doesn't exist");

    {
        let surf = surface_from_trimesh(&surface_trimesh, params)
            .expect("Failed to build implicit surface.");

        let mut pot_attrib64 = vec![0.0f64; sample_mesh.num_vertices()];
        surf.potential(sample_mesh.vertex_positions(), &mut pot_attrib64)
            .expect("Failed to compute contact constraint potential.");

        // Make sure the two potentials are identical.
        println!("potential = {:?}", pot_attrib);
        println!("potential64 = {:?}", pot_attrib64);
        for (&x, &y) in pot_attrib.iter().zip(pot_attrib64.iter()) {
            assert_relative_eq!(x, y as f32, max_relative = 1e-5);
        }
    }

    pot_attrib.into_iter().map(|x| x as f32).collect()
}

#[test]
fn tet_push_test() -> Result<(), Error> {
    // A triangle is being pushed on top of a tet.
    let height = 1.18032;
    let mut tri_verts = vec![
        [0.1, height, 0.0],
        [-0.05, height, 0.0866026],
        [-0.05, height, -0.0866026],
    ];

    let tri = vec![3, 0, 2, 1];

    let mut tetmesh = make_regular_tet();

    // Set fixed vertices
    tetmesh.add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, vec![0, 1, 1, 1])?;

    let trimesh = PolyMesh::new(tri_verts.clone(), &tri);

    // Set contact parameters
    let kernel = KernelType::Approximate {
        radius_multiplier: 1.59,
        tolerance: 0.001,
    };

    //compute_contact_constraint(&trimesh, &tetmesh, radius, tolerance);

    let params = SimParams {
        outer_tolerance: 1e-5, // This is a fairly strict tolerance.
        max_outer_iterations: 20,
        gravity: [0.0f32, 0.0, 0.0],
        ..STATIC_PARAMS
    };

    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(tetmesh.clone(), medium_solid_material().with_id(0))
        .add_fixed(trimesh.clone(), 1)
        .add_frictional_contact(
            FrictionalContactParams {
                contact_type: ContactType::Point,
                kernel,
                friction_params: None,
            },
            (0, 1),
        )
        .build()?;

    let solve_result = solver.step()?;
    assert_eq!(solve_result.iterations, 1); // should be no more than one outer iteration

    // Expect no push since the triangle is outside the surface.
    for (pos, exp_pos) in solver
        .solid(0)
        .tetmesh
        .vertex_position_iter()
        .zip(tetmesh.vertex_positions().iter())
    {
        for i in 0..3 {
            assert_relative_eq!(pos[i], exp_pos[i], max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    // Verify constraint, should be positive before push
    let constraint = compute_contact_constraint(&trimesh, &solver.solid(0).tetmesh, kernel);
    assert!(constraint.iter().all(|&x| x >= 0.0f32));

    // Simulate push
    let offset = 0.34;
    tri_verts.iter_mut().for_each(|x| (*x)[1] -= offset);
    let pts = PointCloud::new(tri_verts.clone());
    assert!(solver.update_shell_vertices(&pts).is_ok());
    let solve_result = solver.step()?;
    assert!(solve_result.iterations <= params.max_outer_iterations);

    // Verify constraint, should be positive after push
    let constraint = compute_contact_constraint(&trimesh, &solver.solid(0).tetmesh, kernel);
    assert!(constraint.iter().all(|&x| x >= -params.outer_tolerance));

    // Expect only the top vertex to be pushed down.
    let offset_verts = vec![
        [0.0, 0.629, 0.0],
        tetmesh.vertex_position(1),
        tetmesh.vertex_position(2),
        tetmesh.vertex_position(3),
    ];

    for (pos, exp_pos) in solver
        .solid(0)
        .tetmesh
        .vertex_position_iter()
        .zip(offset_verts.iter())
    {
        for i in 0..3 {
            assert_relative_eq!(pos[i], exp_pos[i], epsilon = 1e-3);
        }
    }

    Ok(())
}

fn ball_tri_push_tester(
    material: SolidMaterial,
    fc_params: FrictionalContactParams,
) -> Result<(), Error> {
    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball_fixed.vtk")).unwrap();

    let params = SimParams {
        max_iterations: 100,
        outer_tolerance: 1e-5, // This is a fairly strict tolerance.
        max_outer_iterations: 200,
        ..DYNAMIC_PARAMS
    };

    let polymesh = geo::io::load_polymesh(&PathBuf::from("assets/tri.vtk"))?;
    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(tetmesh, material.with_id(0))
        .add_fixed(polymesh, 1)
        .add_frictional_contact(fc_params, (0, 1))
        .build()?;

    let res = solver.step()?;
    //println!("res = {:?}", res);
    assert!(
        res.iterations <= params.max_outer_iterations,
        "Exceeded max outer iterations."
    );
    Ok(())
}

#[test]
fn ball_tri_push_test() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4));
    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.812,
            tolerance: 0.07,
        },
        friction_params: None,
    };

    ball_tri_push_tester(material, fc_params)
}

#[test]
fn ball_tri_push_volume_constraint_test() -> Result<(), Error> {
    let material = SOLID_MATERIAL
        .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
        .with_volume_preservation(true);
    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.812,
            tolerance: 0.07,
        },
        friction_params: None,
    };

    ball_tri_push_tester(material, fc_params)
}

fn ball_bounce_tester(
    material: SolidMaterial,
    fc_params: FrictionalContactParams,
    tetmesh: TetMesh,
) -> Result<(), Error> {
    let friction_iterations = if fc_params.friction_params.is_some() {
        1
    } else {
        0
    };
    let params = SimParams {
        max_iterations: 200,
        outer_tolerance: 0.1,
        max_outer_iterations: 20,
        gravity: [0.0f32, -9.81, 0.0],
        time_step: Some(0.0208333),
        friction_iterations,
        ..DYNAMIC_PARAMS
    };

    let mut grid = make_grid(Grid {
        rows: 4,
        cols: 4,
        orientation: AxisPlaneOrientation::ZX,
    });

    scale(&mut grid, [3.0, 1.0, 3.0].into());
    translate(&mut grid, [0.0, -3.0, 0.0].into());

    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(tetmesh, material.with_id(0))
        .add_fixed(grid, 1)
        .add_frictional_contact(fc_params, (0, 1))
        .build()?;

    for _ in 0..50 {
        let res = solver.step()?;
        //println!("res = {:?}", res);
        //geo::io::save_tetmesh(
        //    &solver.borrow_mesh(),
        //    &PathBuf::from(format!("./out/mesh_{}.vtk", i)),
        //    ).unwrap();
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );
    }

    Ok(())
}

#[test]
fn ball_bounce_on_points_test() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4));

    let sc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.1,
            tolerance: 0.01,
        },
        friction_params: None,
    };

    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

    ball_bounce_tester(material, sc_params, tetmesh)
}

#[test]
fn ball_bounce_on_points_volume_constraint_test() -> Result<(), Error> {
    let material = SOLID_MATERIAL
        .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
        .with_volume_preservation(true);

    let sc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.1,
            tolerance: 0.01,
        },
        friction_params: None,
    };

    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

    ball_bounce_tester(material, sc_params, tetmesh)
}

/// Tet bouncing on an implicit surface. This is an easy test where the tet sees the entire
/// local implicit surface.
#[test]
fn tet_bounce_on_implicit_test() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));

    let sc_params = FrictionalContactParams {
        contact_type: ContactType::Implicit,
        kernel: KernelType::Approximate {
            radius_multiplier: 20.0, // deliberately large radius
            tolerance: 0.0001,
        },
        friction_params: None,
    };

    let tetmesh = make_regular_tet();

    ball_bounce_tester(material, sc_params, tetmesh)
}

/// Ball bouncing on an implicit surface.
#[test]
fn ball_bounce_on_implicit_test() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));

    let sc_params = FrictionalContactParams {
        contact_type: ContactType::Implicit,
        kernel: KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 0.0001,
        },
        friction_params: None,
    };

    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

    ball_bounce_tester(material, sc_params, tetmesh)
}

/// Ball with constant volume bouncing on an implicit surface.
#[test]
fn ball_bounce_on_implicit_volume_constraint_test() -> Result<(), Error> {
    let material = SOLID_MATERIAL
        .with_elasticity(ElasticityParameters::from_young_poisson(10e6, 0.4))
        .with_volume_preservation(true);

    let sc_params = FrictionalContactParams {
        contact_type: ContactType::Implicit,
        kernel: KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 0.0001,
        },
        friction_params: None,
    };

    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

    ball_bounce_tester(material, sc_params, tetmesh)
}

/// Ball bouncing on an implicit surface with staggered projections friction.
//#[test]
#[allow(dead_code)]
fn ball_bounce_on_sp_implicit_test() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(10e5, 0.4));

    let sc_params = FrictionalContactParams {
        contact_type: ContactType::SPImplicit,
        kernel: KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 0.0001,
        },
        friction_params: Some(FrictionParams {
            dynamic_friction: 0.2,
            inner_iterations: 100,
            tolerance: 1e-5,
            print_level: 0,
        }),
    };

    let tetmesh = geo::io::load_tetmesh(&PathBuf::from("assets/ball.vtk"))?;

    ball_bounce_tester(material, sc_params, tetmesh)
}
