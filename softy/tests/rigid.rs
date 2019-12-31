mod test_utils;

use approx::*;
use geo::mesh::attrib::*;
use geo::mesh::topology::*;
use geo::ops::*;
use softy::*;
pub use test_utils::*;
use utils::soap::*;
use geo::mesh::VertexPositions;

#[test]
fn rigid_box_under_gravity_one_step() {
    init_logger();

    use geo::mesh::VertexPositions;
    let params = SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..DYNAMIC_PARAMS
    };

    let rigid_material = RigidMaterial::new(0, 1000.0);

    let mesh = PolyMesh::from(make_box(4).surface_trimesh());

    let mut solver = SolverBuilder::new(params)
        .add_shell(mesh.clone(), rigid_material)
        .build()
        .unwrap();
    assert!(solver.step().is_ok());

    // Expect the box to remain in original configuration
    let solution = &solver.shell(0).trimesh;
    for (new, old) in solution
        .vertex_position_iter()
        .zip(mesh.vertex_position_iter())
    {
        // Check that the box is falling.
        dbg!(new, old);
        assert!(new[1] < old[1]);
    }
}

#[test]
fn rigid_box_rotate() {
    init_logger();

    use geo::mesh::VertexPositions;
    let params = SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..DYNAMIC_PARAMS
    };

    let rigid_material = RigidMaterial::new(0, 1.0);

    let mut mesh = PolyMesh::from(make_box(4).surface_trimesh());
    let orig_pos = mesh.vertex_positions().to_vec();

    // Initialize the box to rotate about the y-axis
    let init_vel: Vec<_> = mesh
        .vertex_position_iter()
        .map(|&p| {
            Vector3::new([0.0, 1.0, 0.0])
                .cross(p.into_tensor())
                .into_data()
        })
        .collect();
    mesh.add_attrib_data::<VelType, VertexIndex>(VELOCITY_ATTRIB, init_vel)
        .unwrap();

    let mut solver = SolverBuilder::new(params)
        .add_shell(mesh.clone(), rigid_material)
        .build()
        .unwrap();

    for _ in 0..30 {
        assert!(solver.step().is_ok());
        // Ensure all vertices are stationary in the y direction.
        for (p, orig_p) in solver
            .shell(0)
            .trimesh
            .vertex_position_iter()
            .zip(orig_pos.iter())
        {
            // equal
            assert_relative_eq!(p[1], orig_p[1]);

            // not equal if their original coordinates were not zero.
            if orig_p[0] != 0.0 {
                assert_relative_ne!(p[0], orig_p[0]);
            }

            if orig_p[2] != 0.0 {
                assert_relative_ne!(p[2], orig_p[2]);
            }
        }
    }
}

/// Same as rotate by the box is perturbed from its original configuration.
#[test]
fn rigid_box_perturbed_rotate() {
    init_logger();

    let dt = 0.1_f32;

    use geo::mesh::VertexPositions;
    let params = SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        time_step: Some(dt),
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..DYNAMIC_PARAMS
    };

    let rigid_material = RigidMaterial::new(0, 1.0);

    let mut mesh = PolyMesh::from(make_box(4).surface_trimesh());
    let mut exp_mesh = mesh.clone();
    let ref_pos = mesh.vertex_positions().to_vec();
    // Non-trivial translation and rotation
    let pi = std::f64::consts::PI;
    mesh.rotate_by_vector([pi / 4.0, 0.0, 0.0]);
    mesh.translate([0.1, 2.2, 0.3]);

    // Initialize the box to rotate about the y-axis
    let init_vel: Vec<_> = ref_pos
        .iter()
        .map(|&p| {
            (rotate(
                [0.0, 1.0, 0.0].into_tensor().cross(p.into_tensor()),
                [pi / 4.0, 0.0, 0.0],
            ) + Vector3::new([0.1, 0.2, 0.3]))
            .into_data()
        })
        .collect();
    mesh.add_attrib_data::<VelType, VertexIndex>(VELOCITY_ATTRIB, init_vel)
        .unwrap();

    let mut solver = SolverBuilder::new(params)
        .add_shell(mesh, rigid_material)
        .build()
        .unwrap();

    exp_mesh.rotate_by_vector([pi / 4.0, 0.0, 0.0]);
    let axis = [0.0, 1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()].into_tensor();
    exp_mesh.rotate_by_vector((axis * dt as f64 * 30.0).into());
    exp_mesh.translate([0.1, 2.2, 0.3]);
    let dir = [0.1, 0.2, 0.3].into_tensor();
    exp_mesh.translate((dir * dt as f64 * 30.0).into());

    for _ in 0..30 {
        assert!(solver.step().is_ok());
    }

    for (p, exp_p) in solver
        .shell(0)
        .trimesh
        .vertex_position_iter()
        .zip(exp_mesh.vertex_position_iter())
    {
        assert_relative_eq!(p.into_tensor(), exp_p.into_tensor(), max_relative = 1e-7);
    }
}

/// Test that a rigid object will fall and rest on a flat plane.
#[test]
fn rigid_simple_contact() {
    init_logger();

    use geo::mesh::VertexPositions;
    let params = SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..DYNAMIC_PARAMS
    };

    let rigid_material = RigidMaterial::new(1, 1.0);

    let mut grid = make_grid(4);
    grid.scale([2.0, 1.0, 2.0]);
    grid.translate([0.0, -0.7, 0.0]);

    let mut mesh = PolyMesh::from(make_box(4).surface_trimesh());

    let fc = FrictionalContactParams {
        contact_type: ContactType::LinearizedPoint,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.1,
            tolerance: 0.001,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: None,
    };

    let mut solver = SolverBuilder::new(params)
        .add_shell(mesh.clone(), rigid_material)
        .add_shell(grid, FixedMaterial::new(0))
        .add_frictional_contact(fc, (0, 1))
        .build()
        .unwrap();

    mesh.translate([0.0, -0.2, 0.0]);
    let exp_pos = mesh.vertex_positions();

    // The box should hit the grid at frame 19.
    // Here we check that nothing happens afterwards.
    for i in 0..30 {
        assert!(solver.step().is_ok());
        // Check that nothing fell through the grid
        for p in solver.shell(0).trimesh.vertex_position_iter() {
            assert!(p[1] >= -0.7);
            // Check that the points didn't slide in the xz plane.
            assert!(p[0] <= 0.5);
            assert!(p[0] >= -0.5);
            assert!(p[2] <= 0.5);
            assert!(p[2] >= -0.5);
        }

        if i > 19 {
            // After the collision, check that all vertices are basically translated down by -0.2:
            for (p, exp_p) in solver
                .shell(0)
                .trimesh
                .vertex_position_iter()
                .zip(exp_pos.iter())
            {
                assert_relative_eq!(p[0], exp_p[0]);
                assert_relative_eq!(p[2], exp_p[2]);
                assert_relative_eq!(p[1], exp_p[1], max_relative = 1e-3, epsilon = 1e-7);
            }
        }
    }
}

/// Test that a rigid box will start rotating when hit at a corner.
#[test]
fn rigid_torque_contact() {
    init_logger();

    let params = SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..DYNAMIC_PARAMS
    };

    let rigid_material = RigidMaterial::new(1, 1.0);

    let mesh = PolyMesh::from(make_box(2).surface_trimesh());
    let mut collider = PolyMesh::from(make_box(5).surface_trimesh());
    collider.scale([1.0, 1.0, 1.2]);
    collider.translate([0.7, -1.1, 0.0]);

    let fc = FrictionalContactParams {
        contact_type: ContactType::LinearizedPoint,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.1,
            tolerance: 0.001,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: None,
    };

    let mut solver = SolverBuilder::new(params)
        .add_shell(mesh.clone(), rigid_material)
        .add_shell(collider, FixedMaterial::new(0))
        .add_frictional_contact(fc, (0, 1))
        .build()
        .unwrap();

    for _ in 0..30 {
        assert!(solver.step().is_ok());
        // Check that points are still within the original z coorinate bounds since the problem is
        // symmetric.
        for p in solver.shell(0).trimesh.vertex_position_iter() {
            assert!(p[2] >= -0.51);
            assert!(p[2] <= 0.51);
        }
    }
}

/// Same test as `rigid_torque_contact` but with a perturbed configuration.
#[test]
fn rigid_torque_complex_contact() {
    init_logger();

    let params = SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        outer_tolerance: 1e-10, // This is a fairly strict tolerance.
        ..DYNAMIC_PARAMS
    };

    let rigid_material = RigidMaterial::new(1, 1.0);

    let mut mesh = PolyMesh::from(make_box(2).surface_trimesh());
    mesh.translate([0.01, 0.02, 0.03]);
    mesh.rotate_by_vector([0.01, 0.02, 0.03]);
    let mut collider = PolyMesh::from(make_box(2).surface_trimesh());
    collider.scale([1.0, 1.0, 1.2]);
    collider.translate([0.7, -1.1, 0.0]);

    let fc = FrictionalContactParams {
        contact_type: ContactType::LinearizedPoint,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.1,
            tolerance: 0.001,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: None,
    };

    let mut solver = SolverBuilder::new(params)
        .add_shell(mesh.clone(), rigid_material)
        .add_shell(collider, FixedMaterial::new(0))
        .add_frictional_contact(fc, (0, 1))
        .build()
        .unwrap();

    for _ in 0..60 {
        assert!(solver.step().is_ok());
    }

    // Check expected box bounds.
    for p in solver.shell(0).trimesh.vertex_position_iter() {
        assert!(p[0] >= -0.9);
        assert!(p[0] <= 0.2);
        assert!(p[1] >= -1.6);
        assert!(p[1] <= -0.5);
        assert!(p[2] >= -0.5);
        assert!(p[2] <= 0.7);
    }
}
