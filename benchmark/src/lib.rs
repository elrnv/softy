use utils::soap::*;

/// Generate a random vector of float values between -1 and 1.
pub fn random_vec(n: usize) -> Vec<f64> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n)
        .map(move |_| rng.sample(range))
        .collect()
}
pub fn lazy_expr(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    (m.expr() * v.expr()).eval()
}
pub fn outer_read_local(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (col, rhs) in m.iter().zip(v.iter()) {
        for (out, &val) in out.iter_mut().zip(col.iter()) {
            *out += val*rhs;
        }
    }
    out
}
