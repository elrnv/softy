#[cfg(features = "packed_simd")]
use packed_simd::*;
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

#[cfg(features = "packed_simd")]
pub fn inner_simd_unchecked(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (col, rhs) in row.chunks_exact(4).zip(v.chunks_exact(4)) {
            unsafe {
                let col4 = f64x4::new(*col.get_unchecked(0), *col.get_unchecked(1), *col.get_unchecked(2), *col.get_unchecked(3));
                let rhs4 = f64x4::new(*rhs.get_unchecked(0), *rhs.get_unchecked(1), *rhs.get_unchecked(2), *rhs.get_unchecked(3));
                *out += (col4 * rhs4).sum();
            }
        }
    }
    out
}

#[cfg(features = "packed_simd")]
pub fn inner_simd(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (col, rhs) in row.chunks_exact(4).zip(v.chunks_exact(4)) {
            unsafe {
                let col4 = f64x4::new(col[0], col[1], col[2], col[3]);
                let rhs4 = f64x4::new(rhs[0], rhs[1], rhs[2], rhs[3]);
                *out += (col4 * rhs4).sum();
            }
        }
    }
    out
}

pub fn inner(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (&col, &rhs) in Chunked4::from_flat(row).iter().zip(Chunked4::from_flat(v).iter()) {
            *out += Vector4::new(col).dot(Vector4::new(rhs));
        }
    }
    out
}
