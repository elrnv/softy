mod test_utils;

use log::debug;
use softy::opt_fem::*;
use softy::*;
pub use test_utils::*;

#[test]
fn tube_cloth() -> Result<(), Error> {
    init_logger();
    let fc_params = FrictionalContactParams {
        contact_type: ContactType::LinearizedPoint,
        kernel: KernelType::Cubic {
            radius_multiplier: 1.0,
        },
        contact_offset: -0.02,
        use_fixed: false,
        friction_params: Some(FrictionParams {
            smoothing_weight: 0.5,
            friction_forwarding: 0.0,
            dynamic_friction: 0.15,
            inner_iterations: 100,
            tolerance: 1e-6,
            print_level: 0,
        }),
    };

    let tube = geo::io::load_polymesh("./assets/tube.vtk")?;
    let cloth = geo::io::load_polymesh("./assets/tube_cloth.vtk")?;

    let params = SimParams {
        max_iterations: 200,
        outer_tolerance: 0.1,
        max_outer_iterations: 20,
        gravity: [0.0f32, -9.81, 0.0],
        time_step: Some(0.01),
        print_level: 0,
        max_gradient_scaling: 1e-3,
        friction_iterations: 10,
        ..DYNAMIC_OPT_PARAMS
    };

    let cloth_material = SoftShellMaterial::new(1)
        .with_density(1000.0)
        .with_bending_stiffness(5.2)
        .with_elasticity(ElasticityParameters::from_bulk_shear(1000.0, 100.0));

    let mut solver = SolverBuilder::new(params.clone())
        .add_fixed(tube, 0)
        .add_soft_shell(cloth, cloth_material)
        .add_frictional_contact(fc_params, (0, 1))
        .build()?;

    for _ in 0..2 {
        let res = solver.step()?;
        debug!("res = {:?}", res);
        //geo::io::save_tetmesh(
        //    &solver.solid(0).tetmesh,
        //    &format!("./out/solid_mesh_{}.vtk", i),
        //)?;
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );
    }
    Ok(())
}
