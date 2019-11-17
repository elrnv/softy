use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
use utils::soap::*;
use rayon::prelude::*;
use approx::assert_relative_eq;

/// Generate a random vector of float values between -1 and 1.
pub fn random_vec(n: usize) -> Vec<f64> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n)
        .map(move |_| rng.sample(range))
        .collect()
}

pub fn outer_write_local(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (i, out) in out.iter_mut().enumerate() {
        for (col, &rhs) in m.iter().zip(v.iter()) {
            *out += col[i]*rhs;
        }
    }
    out
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
pub fn inner(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (&col, &rhs) in row.iter().zip(v.iter()) {
            *out += col*rhs;
        }
    }
    out
}

pub fn outer_read_local_par(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let size = v.len();
    let n = 24;
    let mut out_24 = vec![vec![0.0; size]; n+1];
    let m = m.into_flat();
    let chunk_size = size / n;
    m.par_chunks(size*chunk_size).zip(v.par_chunks(chunk_size)).zip(out_24.par_iter_mut()).for_each(|((m, v), out)| {
        m.chunks(size).zip(v.iter()).for_each(|(col, rhs)| {
            out.iter_mut().zip(col.iter()).for_each(|(out_val, &val)| {
                *out_val += val*rhs;
            });
        });
    });
    let mut out = out_24.pop().unwrap();
    for i in 0..n {
        out.iter_mut().zip(out_24[i].iter()).for_each(|(out, outi)|  {
            *out += outi;
        });
    }
    out
}
pub fn inner_par(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    m.into_par_iter().zip(out.par_iter_mut()).for_each(|(row, out_val)|  {
        row.iter().zip(v.iter()).for_each(|(&col, &rhs)|  {
            *out_val += col*rhs;
        });
    });
    out
}

fn matrix_vector_mul_order_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Vector Multiply Order");

    for &n in &[100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000] {
        let mut m = ChunkedN::from_flat_with_stride(random_vec(n*n), n);
        let v = random_vec(n);
        // symmetrize matrix to make consistent results
        for i in 0..n {
            for j in 0..n {
                if j > i {
                    let other = *m.view().at(j).at(i);
                    let val = m.view_mut().isolate(i).isolate(j);
                    *val = other;
                }
            }
        }

        assert_eq!(outer_read_local(m.view(), v.view()), outer_write_local(m.view(), v.view()));
        assert_eq!(inner(m.view(), v.view()), outer_write_local(m.view(), v.view()));
        assert_eq!(inner_par(m.view(), v.view()), outer_write_local(m.view(), v.view()));
        { 
            let out1 = outer_read_local_par(m.view(), v.view());
            let out2 = outer_write_local(m.view(), v.view());
            for (a, b) in out1.iter().zip(out2.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-7, epsilon = 1e-7);
            }
        }

        if n < 5000 {
            group.bench_with_input(BenchmarkId::new("Outer Write Local", n), &(&m, &v),
                |b, (m, v)| {
                    b.iter_batched(|| (m.view(), v.view()), |(m,v)| outer_write_local(m, v), BatchSize::LargeInput)
                });
        }

        group.bench_with_input(BenchmarkId::new("Outer Read Local", n), &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(|| (m.view(), v.view()), |(m,v)| outer_read_local(m, v), BatchSize::LargeInput)
            });

        group.bench_with_input(BenchmarkId::new("Inner", n), &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(|| (m.view(), v.view()), |(m,v)| inner(m, v), BatchSize::LargeInput)
            });

        group.bench_with_input(BenchmarkId::new("Outer Read Local Parallel", n), &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(|| (m.view(), v.view()), |(m,v)| outer_read_local_par(m, v), BatchSize::LargeInput)
            });

        group.bench_with_input(BenchmarkId::new("Inner Parallel", n), &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(|| (m.view(), v.view()), |(m,v)| inner_par(m, v), BatchSize::LargeInput)
            });
    }
        
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = matrix_vector_mul_order_benchmark
);
criterion_main!(benches);
