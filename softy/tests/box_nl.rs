mod test_utils;

use softy::fem::nl::SimParams as NLParams;
use softy::fem::nl::*;
use softy::{ElasticityParameters, Error, SolidMaterial, TetMesh};
use std::path::PathBuf;
pub use test_utils::*;

const STRETCH_NL_PARAMS: NLParams = NLParams {
    gravity: [0.0f32, 0.0, 0.0],
    ..STATIC_NL_PARAMS
};

pub fn medium_solid_material() -> SolidMaterial {
    default_solid().with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
}

/// Test that the solver produces no change for an equilibrium configuration for a
/// tetrahedralized box. This example also uses a softer material and a momentum term
/// (dynamics enabled), which is more sensitive to perturbations.
#[test]
fn equilibrium() {
    init_logger();
    let params = SimParams {
        max_iterations: 1,
        ..DYNAMIC_NL_PARAMS
    };

    let soft_material = SolidMaterial::new(0)
        .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.49))
        .with_volume_preservation(false)
        .with_density(1000.0);

    // Box in equilibrium configuration should stay in equilibrium configuration
    let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk")).unwrap();

    let mut solver = SolverBuilder::new(params)
        .add_solid(mesh.clone(), soft_material)
        .build::<f64>()
        .expect("Failed to create solver for soft box equilibrium test");
    assert!(solver.step().is_ok());

    // Expect the box to remain in original configuration
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &mesh, 1e-6);
}

#[test]
fn stretch_plain() -> Result<(), Error> {
    init_logger();
    let mesh = make_stretched_box(4);
    let mut solver = SolverBuilder::new(NLParams {
        ..STRETCH_NL_PARAMS
    })
    .add_solid(mesh, medium_solid_material())
    .build::<f64>()?;
    solver.step()?;
    let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched.vtk"))?;
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &expected, 1e-2);
    Ok(())
}

/*
#[test]
fn stretch_volume_constraint() -> Result<(), Error> {
    init_logger();
    let incompressible_material = medium_solid_material().with_volume_preservation(true);
    let mesh = make_stretched_box(4);
    let mut solver = SolverBuilder::new(NLParams {
        ..STRETCH_NL_PARAMS
    })
    .add_solid(mesh, incompressible_material)
    .build()?;
    solver.step()?;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_stretched_const_volume.vtk"))?;
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &expected, 1e-3);
    Ok(())
}
*/

#[test]
fn twist_plain() -> Result<(), Error> {
    init_logger();
    let material = medium_solid_material()
        .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0));
    let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
    let params = NLParams {
        ..STRETCH_NL_PARAMS
    };
    let mut solver = SolverBuilder::new(params)
        .add_solid(mesh, material)
        .build::<f64>()?;
    solver.step()?;
    let expected: TetMesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted.vtk"))?;
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &expected, 1e-3);
    Ok(())
}

/*
#[test]
fn twist_dynamic_volume_constraint() -> Result<(), Error> {
    init_logger();
    let material = medium_solid_material()
        .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0))
        .with_volume_preservation(true);

    // We use a large time step to get the simulation to settle to the static sim with less
    // iterations.
    let params = NLParams {
        time_step: Some(2.0),
        ..DYNAMIC_NL_PARAMS
    };

    let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(mesh, material)
        .build()?;

    // The dynamic sim needs to settle
    for _ in 1u32..15 {
        let result = solver.step()?;
        assert!(
            result.iterations <= params.max_iterations,
            "Unconstrained solver ran out of outer iterations."
        );
    }

    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
    let solution = &solver.solid(0).tetmesh;
    //geo::io::save_tetmesh(solution, &PathBuf::from("out/box_twisted_const_volume.vtk"))?;
    compare_meshes(solution, &expected, 1e-2);
    Ok(())
}

#[test]
fn twist_volume_constraint() -> Result<(), Error> {
    init_logger();
    let material = medium_solid_material()
        .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0))
        .with_volume_preservation(true);
    let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk")).unwrap();
    let params = NLParams {
        ..STRETCH_NL_PARAMS
    };
    let mut solver = SolverBuilder::new(params)
        .add_solid(mesh, material)
        .build()?;
    solver.step()?;
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &expected, 1e-6);
    Ok(())
}

/// This test insures that a non-linearized constraint like volume doesn't cause multiple outer
/// iterations, and converges after the first solve.
#[test]
fn twist_volume_constraint_consistent_outer_iterations() -> Result<(), Error> {
    init_logger();
    let material = medium_solid_material()
        .with_elasticity(ElasticityParameters::from_young_poisson(1000.0, 0.0))
        .with_volume_preservation(true);

    let params = NLParams {
        tolerance: 1e-5, // This is a fairly strict tolerance.
        ..STRETCH_NL_PARAMS
    };

    let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_twist.vtk"))?;
    let mut solver = SolverBuilder::new(params)
        .add_solid(mesh, material)
        .build()?;
    let solve_result = solver.step()?;
    assert_eq!(solve_result.iterations, 1);

    // This test should produce the exact same mesh as the original
    // box_twist_volume_constraint_test
    let expected: TetMesh =
        geo::io::load_tetmesh(&PathBuf::from("assets/box_twisted_const_volume.vtk"))?;
    let solution = &solver.solid(0).tetmesh;
    compare_meshes(solution, &expected, 1e-6);
    Ok(())
}
*/
