mod test_utils;

use softy::*;
pub use test_utils::*;
use utils::*;

fn friction_tester(
    material: SolidMaterial,
    fc_params: FrictionalContactParams,
    tetmesh: TetMesh,
    surface: PolyMesh,
    implicit_tetmesh: bool,
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
        time_step: Some(0.01),
        print_level: 5,
        friction_iterations,
        ..DYNAMIC_PARAMS
    };

    let coupling = if implicit_tetmesh { (0, 1) } else { (1, 0) };

    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(tetmesh, material.with_id(0))
        .add_fixed(surface, 1)
        .add_frictional_contact(fc_params, coupling)
        .build()?;

    for i in 0..50 {
        let res = solver.step()?;
        println!("res = {:?}", res);
        geo::io::save_tetmesh(
            &solver.solid(0).tetmesh,
            &format!("./out/solid_mesh_{}.vtk", i),
        )?;
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );
    }

    Ok(())
}

/// A regular tetrahedron sliding on a flat surface.
#[test]
fn sliding_tet_on_points() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(1e5, 0.4));
    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.5,
            tolerance: 0.001,
        },
        friction_params: Some(FrictionParams {
            dynamic_friction: 0.2,
            inner_iterations: 100,
            tolerance: 1e-6,
            print_level: 5,
        }),
    };

    let tetmesh = make_regular_tet();
    let mut surface = make_grid(Grid {
        rows: 4,
        cols: 4,
        orientation: AxisPlaneOrientation::ZX,
    });
    rotate(&mut surface, [1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
    translate(&mut surface, [0.0, -0.7, 0.0].into());

    friction_tester(material, fc_params, tetmesh, surface, true)
}

/// A regular tetrahedron sliding on a flat surface.
#[test]
fn sliding_tet_on_implicit() -> Result<(), Error> {
    let material =
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_young_poisson(1e5, 0.4));

    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 1.5,
            tolerance: 0.001,
        },
        friction_params: Some(FrictionParams {
            dynamic_friction: 0.2,
            inner_iterations: 100,
            tolerance: 1e-6,
            print_level: 5,
        }),
    };

    let tetmesh = make_regular_tet();
    let mut surface = make_grid(Grid {
        rows: 1,
        cols: 1,
        orientation: AxisPlaneOrientation::ZX,
    });
    rotate(&mut surface, [1.0, 0.0, 0.0], std::f64::consts::PI / 16.0);
    translate(&mut surface, [0.0, -0.7, 0.0].into());

    //geo::io::save_polymesh(&surface, "./out/polymesh.vtk")?;

    friction_tester(material, fc_params, tetmesh, surface, false)
}
