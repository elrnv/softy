use geo::mesh::VertexPositions;
pub use softy::test_utils::*;
use softy::*;

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
