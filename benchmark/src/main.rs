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
fn main() {
    let n = 2000;
    let mut m = ChunkedN::from_flat_with_stride(random_vec(n*n), n);
    let v = random_vec(n);
    //// symmetrize matrix to make consistent results
    //for i in 0..n {
    //    for j in 0..n {
    //        if j > i {
    //            let other = *m.view().at(j).at(i);
    //            let val = m.view_mut().isolate(i).isolate(j);
    //            *val = other;
    //        }
    //    }
    //}
    let mut size = 0;
    for _ in 0..1000 {
        size += lazy_expr(m.view(), v.view()).len();
        eprint!(".");
    }
    eprintln!("{}", size);
}
