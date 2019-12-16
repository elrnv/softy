use geo::mesh::VertexPositions;
pub use softy::test_utils::*;

/// Utility function to compare positions of two meshes.
pub fn compare_meshes<M>(solution: &M, expected: &M, tol: f64)
where
    M: VertexPositions<Element = [f64; 3]>,
{
    use approx::*;
    for (pos, expected_pos) in solution
        .vertex_positions()
        .iter()
        .zip(expected.vertex_positions().iter())
    {
        assert_relative_eq!(pos[0], expected_pos[0], max_relative = tol, epsilon = 5e-6);
        assert_relative_eq!(pos[1], expected_pos[1], max_relative = tol, epsilon = 5e-6);
        assert_relative_eq!(pos[2], expected_pos[2], max_relative = tol, epsilon = 5e-6);
    }
}

pub fn init_logger() {
    let _ = env_logger::Builder::from_env("SOFTY_LOG")
        .is_test(true)
        .try_init();
}
