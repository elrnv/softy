use geo::mesh::VertexPositions;
pub use softy::test_utils::*;
use softy::*;

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
    max_gradient_scaling: 0.0001,
    log_file: None,
};

//pub(crate) const QUASI_STATIC_PARAMS: SimParams = SimParams {
//    gravity: [0.0f32, 0.0, 0.0],
//    time_step: Some(0.01),
//    clear_velocity: true,
//    ..STATIC_PARAMS
//};

// Note: The key to getting reliable simulations here is to keep bulk_modulus, shear_modulus
// (mu) and density in the same range of magnitude. Higher stiffnesses compared to denisty will
// produce highly oscillatory configurations and keep the solver from converging fast.
// As an example if we increase the moduli below by 1000, the solver can't converge, even in
// 300 steps.
pub const SOLID_MATERIAL: SolidMaterial = Material {
    id: 0,
    properties: SolidProperties {
        volume_preservation: false,
        deformable: DeformableProperties {
            elasticity: Some(ElasticityParameters {
                lambda: 93333.33,
                mu: 10e3,
            }),
            density: Some(1000.0),
            damping: 0.0,
            scale: 1.0,
        },
    },
};

pub const DYNAMIC_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, 0.0, 0.0],
    time_step: Some(0.01),
    ..STATIC_PARAMS
};

/// Utility function to compare positions of two meshes.
pub fn compare_meshes(solution: &TetMesh, expected: &TetMesh, tol: f64) {
    use approx::*;
    for (pos, expected_pos) in solution
        .vertex_positions()
        .iter()
        .zip(expected.vertex_positions().iter())
    {
        for j in 0..3 {
            assert_relative_eq!(pos[j], expected_pos[j], max_relative = tol, epsilon = 1e-7);
        }
    }
}

pub fn init_logger() {
    let _ = env_logger::Builder::from_env("SOFTY_LOG")
        .is_test(true)
        .try_init();
}
