use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use utils::soap::*;
//use rayon::prelude::*;
//use approx::assert_relative_eq;

/// Generate a random vector of float values between -1 and 1.
pub fn random_vec(n: usize) -> Vec<f64> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n).map(move |_| rng.sample(range)).collect()
}

pub fn lazy_expr(a: ChunkedN<&[f64]>, b: ChunkedN<&[f64]>) -> ChunkedN<Vec<f64>> {
    (a.expr() + b.expr()).eval()
}

pub fn manual(a: ChunkedN<&[f64]>, b: ChunkedN<&[f64]>) -> ChunkedN<Vec<f64>> {
    let mut out = Vec::with_capacity(a.view().into_flat().len());
    for (a, b) in a.iter().zip(b.iter()) {
        for (a, b) in a.iter().zip(b.iter()) {
            out.push(a + b);
        }
    }
    ChunkedN::from_flat_with_stride(out, a.len())
}

pub fn manual_init(a: ChunkedN<&[f64]>, b: ChunkedN<&[f64]>) -> ChunkedN<Vec<f64>> {
    let mut out = ChunkedN::from_flat_with_stride(vec![0.0; a.view().into_flat().len()], a.len());
    for ((a, b), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        for ((a, b), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out = a + b;
        }
    }
    out
}

pub fn manual_assign(mut a: ChunkedN<Vec<f64>>, b: ChunkedN<&[f64]>) -> ChunkedN<Vec<f64>> {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        for (a, b) in a.iter_mut().zip(b.iter()) {
            *a += b;
        }
    }
    a
}

fn matrix_matrix_add_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Matrix Add");

    for &n in &[100, 250, 500, 750, 1000, 2500, 5000] {
        let a = ChunkedN::from_flat_with_stride(random_vec(n * n), n);
        let b = ChunkedN::from_flat_with_stride(random_vec(n * n), n);

        group.bench_with_input(
            BenchmarkId::new("Lazy Expr", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || (a.view(), b.view()),
                    |(a, b)| lazy_expr(a, b),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(BenchmarkId::new("Manual", n), &(&a, &b), |bench, (a, b)| {
            bench.iter_batched(
                || (a.view(), b.view()),
                |(a, b)| manual(a, b),
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(
            BenchmarkId::new("Manual Init", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || (a.view(), b.view()),
                    |(a, b)| manual_init(a, b),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Manual Assign", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter_batched(
                    || ((*a).clone(), b.view()),
                    |(a, b)| manual_assign(a, b),
                    BatchSize::LargeInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(15);
    targets = matrix_matrix_add_benchmark
);
criterion_main!(benches);
