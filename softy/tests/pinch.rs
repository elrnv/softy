mod test_utils;

use geo::attrib::Attrib;
use geo::mesh::*;
use softy::opt_fem::*;
use softy::*;
use std::path::PathBuf;
pub use test_utils::*;

/// Pinch a box between two probes.
/// Given sufficient friction, the box should not fall.
fn pinch_tester(fc_params: FrictionalContactParams) -> Result<(), Error> {
    use geo::mesh::topology::*;

    let params = SimParams {
        max_iterations: 200,
        max_outer_iterations: 20,
        gravity: [0.0f32, -9.81, 0.0],
        time_step: Some(0.01),
        print_level: 5,
        friction_iterations: 1,
        ..DYNAMIC_OPT_PARAMS
    };

    let material = default_solid()
        .with_id(0)
        .with_elasticity(ElasticityParameters::from_young_poisson(1e6, 0.45));

    let clamps = geo::io::load_polymesh(&PathBuf::from("assets/clamps.vtk"))?;
    let mut box_mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk"))?;
    box_mesh.remove_attrib::<VertexIndex>("fixed")?;

    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(box_mesh, material)
        .add_fixed(clamps, 1)
        .add_frictional_contact(fc_params, (0, 1))
        .build()?;

    for iter in 0..50 {
        let res = solver.step()?;

        //println!("res = {:?}", res);
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );

        // Check that the mesh hasn't fallen.
        let tetmesh = &solver.solid(0).tetmesh;

        geo::io::save_tetmesh(tetmesh, &PathBuf::from(&format!("out/mesh_{}.vtk", iter)))?;

        for v in tetmesh.vertex_position_iter() {
            assert!(v[1] > -0.6);
        }
    }

    Ok(())
}

/// Pinch a box against a couple of implicit surfaces.
#[allow(dead_code)]
fn pinch_against_implicit() -> Result<(), Error> {
    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Cubic {
            radius_multiplier: 1.5,
        },
        contact_offset: 0.0,
        use_fixed: true,
        friction_params: Some(FrictionParams {
            smoothing_weight: 0.0,
            friction_forwarding: 1.0,
            dynamic_friction: 0.4,
            inner_iterations: 40,
            tolerance: 1e-5,
            print_level: 5,
        }),
    };

    pinch_tester(fc_params)
}
