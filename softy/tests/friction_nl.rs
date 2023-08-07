mod test_utils;

use geo::algo::{Merge, TypedMesh};
use geo::attrib::Attrib;
use geo::mesh::builder::*;
use geo::mesh::{VertexMesh, VertexPositions};
use geo::ops::transform::*;
use geo::topology::{CellIndex, FaceIndex, NumCells, NumFaces, NumVertices, VertexIndex};
use num_traits::Zero;
use softy::constraints::{FrictionParams, FrictionalContactParams};
use softy::nl_fem::*;
use softy::scene::Scene;
use softy::Error;
use softy::*;
use std::io::Write;
use tensr::Vector3;
pub use test_utils::*;

fn fc_params(mu: f64, lagged: bool, incomplete_jacobian: bool) -> FrictionParams {
    FrictionParams {
        dynamic_friction: mu,
        static_friction: mu,
        viscous_friction: 0.0,
        stribeck_velocity: 0.0,
        epsilon: 0.001,
        friction_profile: FrictionProfile::default(),
        lagged,
        incomplete_jacobian,
    }
}

fn isolate_tetmesh(mesh: &Mesh) -> TetMesh {
    mesh.split_into_typed_meshes()
        .into_iter()
        .find_map(|m| {
            if let TypedMesh::Tet(m) = m {
                Some(m)
            } else {
                None
            }
        })
        .unwrap()
}

fn compute_centroid<M: VertexPositions<Element = [f64; 3]>>(mesh: &M) -> Vector3<f64> {
    let mut centroid = Vector3::zero();
    let n = mesh.vertex_positions().len();
    for &x in mesh.vertex_position_iter() {
        centroid += Vector3::from(x) / n as f64;
    }
    centroid
}

fn compute_avg_vel<M: VertexMesh<f64>>(mesh: &M) -> Vector3<f64> {
    let mut avg = Vector3::zero();
    let n = mesh.vertex_positions().len();
    let vel = mesh
        .attrib_as_slice::<VelType, VertexIndex>(VELOCITY_ATTRIB)
        .unwrap();
    for &x in vel.iter() {
        avg += Vector3::from(x) / n as f64;
    }
    avg
}

fn friction_experiment(
    material: SolidMaterial,
    fc_params: FrictionalContactParams,
    params: SimParams,
    mut tetmesh: TetMesh,
    mut surface: PolyMesh,
    implicit_tetmesh: bool,
    steps: u32,
    experiment_name: impl AsRef<str>,
) -> Result<(), Error> {
    init_logger();

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

    let mesh = solver.mesh();

    geo::io::save_mesh(
        &mesh,
        &format!(
            "./out/box_slide_{}/slide_mesh_{}.vtk",
            params.time_step.unwrap(),
            0
        ),
    )?;

    let tetmesh = isolate_tetmesh(&mesh);

    let mut distances = vec![];
    let mut vels = vec![];
    let centroid = compute_centroid(&tetmesh);
    let mut avg_vel = compute_avg_vel(&tetmesh).norm();

    let mut actual_steps = 0;
    for i in 1..(steps * 1000) {
        solver.step().ok();
        let tetmesh = isolate_tetmesh(&solver.mesh());
        let centroid_next = compute_centroid(&tetmesh);
        distances.push((centroid_next - centroid).norm());
        let avg_vel_next = compute_avg_vel(&tetmesh).norm();
        vels.push(avg_vel_next);
        if (avg_vel_next - avg_vel).abs() < 1e-12 {
            break;
        }
        avg_vel = avg_vel_next;
        geo::io::save_mesh(
            &solver.mesh(),
            &format!(
                "./out/box_slide_{}/slide_mesh_{}.vtk",
                params.time_step.unwrap(),
                i
            ),
        )?;
        //assert!(
        //    res.iterations <= params.max_iterations,
        //    "Exceeded max outer iterations."
        //);
        actual_steps += 1;
    }

    {
        let mut f =
            std::fs::File::create(&format!("./out/{}.txt", experiment_name.as_ref())).unwrap();
        writeln!(f, "expected_steps = {};", steps)?;
        writeln!(f, "actual_steps = {};", actual_steps)?;
        writeln!(f, "d = [")?;
        for &d in distances.iter() {
            write!(f, "{}, ", d)?;
        }
        writeln!(f, "];")?;
        writeln!(f, "v = [")?;
        for &v in vels.iter() {
            write!(f, "{}, ", v)?;
        }
        writeln!(f, "];")?;
    }

    Ok(())
}

// TODO: tweak this test to have a specific outcome.
// /// This test ensures that a violation in the max step doesn't break the derivatives.
// #[test]
// fn max_step_violation() -> Result<(), Error> {
//     let material = default_solid().with_elasticity(Elasticity::from_young_poisson(1e5, 0.4));
//     let fc_params = FrictionalContactParams {
//         kernel: KernelType::Approximate {
//             radius_multiplier: 1.5,
//             tolerance: 0.001,
//         },
//         contact_offset: 0.0,
//         use_fixed: true,
//         friction_params: Some(fc_params(0.2)),
//     };
//
//     let mut tetmesh = PlatonicSolidBuilder::new().build_tetrahedron();
//     let init_vel = vec![[0.0, -2.0, 0.0]; tetmesh.num_vertices()];
//     tetmesh.insert_attrib_data::<VelType, VertexIndex>(VELOCITY_ATTRIB, init_vel)?;
//
//     let mut surface = GridBuilder {
//         rows: 5,
//         cols: 5,
//         orientation: AxisPlaneOrientation::ZX,
//     }
//     .build();
//     surface.scale([2.0, 1.0, 2.0]);
//     surface.rotate([1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
//     surface.translate([0.0, -0.4, 0.0]);
//
//     // geo::io::save_polymesh(&surface, "./out/ramp.vtk");
//     // geo::io::save_tetmesh(&tetmesh, "./out/mesh.vtk");
//
//     contact_tester(material, fc_params, tetmesh, surface, true, 1)
// }

/// A regular tetrahedron sliding on a flat surface.
#[test]
#[cfg(not(debug_assertions))]
fn sliding_tet_on_points_lagged() -> Result<(), Error> {
    let material = default_solid().with_elasticity(Elasticity::from_young_poisson_with_model(
        1e5,
        0.4,
        ElasticityModel::StableNeoHookean,
    ));
    let fc_params = FrictionalContactParams {
        kernel: KernelType::Approximate {
            radius_multiplier: 2.5,
            tolerance: 0.001,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: Some(fc_params(0.2, true)),
    };

    let tetmesh = PlatonicSolidBuilder::new().build_tetrahedron();

    let mut surface = GridBuilder {
        rows: 10,
        cols: 10,
        orientation: AxisPlaneOrientation::ZX,
    }
    .build();
    surface.scale([2.0, 1.0, 2.0]);
    surface.rotate([1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
    surface.translate([0.0, -0.7, 0.0]);
    dbg!(&material);

    // geo::io::save_polymesh(&surface, "./out/ramp.vtk");
    // geo::io::save_tetmesh(&tetmesh, "./out/mesh.vtk");

    contact_tester(material, fc_params, tetmesh, surface, true, 10)
}

/// A regular tetrahedron sliding on a flat surface.
#[test]
#[cfg(not(debug_assertions))]
fn sliding_tet_on_points() -> Result<(), Error> {
    let material = default_solid().with_elasticity(Elasticity::from_young_poisson(1e5, 0.4));
    let fc_params = FrictionalContactParams {
        kernel: KernelType::Approximate {
            radius_multiplier: 2.5,
            tolerance: 0.001,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: Some(fc_params(0.2, false)),
    };

    let tetmesh = PlatonicSolidBuilder::new().build_tetrahedron();

    let mut surface = GridBuilder {
        rows: 5,
        cols: 5,
        orientation: AxisPlaneOrientation::ZX,
    }
    .build();
    surface.scale([2.0, 1.0, 2.0]);
    surface.rotate([1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
    surface.translate([0.0, -0.6, 0.0]);

    // geo::io::save_polymesh(&surface, "./out/ramp.vtk");
    // geo::io::save_tetmesh(&tetmesh, "./out/mesh.vtk");

    contact_tester(material, fc_params, tetmesh, surface, true, 10)
}

/// A regular tetrahedron sliding on a flat surface.
#[test]
#[cfg(not(debug_assertions))]
fn sliding_tet_on_implicit() -> Result<(), Error> {
    let material = default_solid().with_elasticity(Elasticity::from_young_poisson(1e5, 0.4));

    let fc_params = FrictionalContactParams {
        kernel: KernelType::Approximate {
            radius_multiplier: 1.5,
            tolerance: 0.001,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: Some(fc_params(0.5, false)),
    };

    let tetmesh = PlatonicSolidBuilder::new().build_tetrahedron();
    let mut surface = GridBuilder {
        rows: 1,
        cols: 1,
        orientation: AxisPlaneOrientation::ZX,
    }
    .build();
    surface.rotate([1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
    surface.translate([0.0, -0.7, 0.0]);

    // geo::io::save_polymesh(&surface, "./out/polymesh.vtk")?;

    contact_tester(material, fc_params, tetmesh, surface, false, 50)
}

/// A regular tetrahedron sliding on a flat surface.
// TODO: Investigate why this is failing
#[test]
#[ignore]
fn sliding_tet_on_implicit_lagged() -> Result<(), Error> {
    let material = default_solid().with_elasticity(Elasticity::from_young_poisson_with_model(
        1e5,
        0.4,
        ElasticityModel::StableNeoHookean,
    ));

    let contact_tolerance = 0.001;

    let fc_params = FrictionalContactParams {
        kernel: KernelType::Approximate {
            radius_multiplier: 20.5,
            tolerance: 0.001,
        },
        friction_params: fc_params(0.5, true, false),
        tolerance: contact_tolerance,
        ..Default::default()
    };

    let tetmesh = PlatonicSolidBuilder::new().build_tetrahedron();
    let mut surface = GridBuilder {
        rows: 1,
        cols: 1,
        orientation: AxisPlaneOrientation::ZX,
    }
    .build();
    surface.rotate([1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
    surface.translate([0.0, -0.7, 0.0]);

    // geo::io::save_polymesh(&surface, "./out/polymesh.vtk")?;

    contact_tester(material, fc_params, tetmesh, surface, false, 50)
}

// A test with both surfaces deformable
#[test]
fn fully_deformable_friction_coupling() -> Result<(), Error> {
    init_logger();
    let scene = Scene::load_from_ron("./assets/fully_deformable_friction_coupling_config.ron")?;
    scene.run(8)?;
    Ok(())
}

// A test with both surfaces deformable
#[test]
fn fully_deformable_friction_coupling_iterative() -> Result<(), Error> {
    init_logger();
    let mut scene = Scene::load_from_ron("./assets/fully_deformable_friction_coupling_config.ron")?;
    // Set to a very low tolerance and large number of iterations so we can compare
    // The result with the direct solve.
    scene.config.sim_params.linsolve = LinearSolver::Iterative {
        tolerance: 0.00001,
        max_iterations: 10000,
    };
    scene.run(8)?;
    Ok(())
}

// A test with both surfaces deformable
#[test]
fn fully_deformable_lagged_friction_coupling() -> Result<(), Error> {
    init_logger();
    let scene =
        Scene::load_from_ron("./assets/fully_deformable_lagged_friction_coupling_config.ron")?;
    scene.run(8)?;
    Ok(())
}

/// A regular box sliding on a flat surface.
#[test]
#[ignore]
fn box_slide_experiment() -> Result<(), Error> {
    let material = default_solid().with_elasticity(Elasticity::from_young_poisson(1e9, 0.0));

    let contact_tolerance = 0.001;

    let fc_params = FrictionalContactParams {
        kernel: KernelType::Approximate {
            radius_multiplier: 1.5,
            tolerance: 0.001,
        },
        friction_params: fc_params(0.177, false, false),
        tolerance: contact_tolerance,
        ..Default::default()
    };

    for config_idx in 0..num_static_configs() {
        let mut params = static_nl_params(config_idx);
        let pi = std::f64::consts::PI;

        let mut tetmesh = BoxBuilder::default().build_tetmesh();
        tetmesh.uniform_scale(0.5);
        tetmesh.rotate([0.0, 0.0, 1.0], pi / 18.0);

        let mut surface = GridBuilder {
            rows: 1,
            cols: 1,
            orientation: AxisPlaneOrientation::ZX,
        }
        .build();

        let offset = contact_tolerance as f64;

        surface.scale([3.0, 1.0, 1.5]);
        surface.translate([-1.5, -0.5 - offset, 0.0]);
        surface.rotate([0.0, 0.0, 1.0], pi / 18.0);

        for h in [0.02] {
            let init_speed = 0.1; // - 1.7034886229125867 * h;
            let init_vel = [
                -init_speed * (pi / 18.0).cos(),
                -init_speed * (pi / 18.0).sin(),
                0.0,
            ];
            tetmesh.set_attrib_data::<VelType, VertexIndex>(
                VELOCITY_ATTRIB,
                vec![init_vel; tetmesh.num_vertices()],
            )?;
            let steps = ((0.1 / 0.006502015185220986) / h as f64).ceil() as u32;
            params.time_step = Some(h as f32);
            friction_experiment(
                material,
                fc_params,
                params.clone(),
                tetmesh.clone(),
                surface.clone(),
                false,
                steps,
                &format!("experiment2_h{}", h),
            )?;
        }
    }
    Ok(())
}
