mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn mtx_mtx_mul_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix*Matrix");

    let a2 = matrix2();
    let b2 = matrix2();
    group.bench_with_input(BenchmarkId::new("soap", 2), &(a2, b2), |bench, &(a, b)| {
        bench.iter(|| a * b)
    });
    let a2 = matrix2_cgmath();
    let b2 = matrix2_cgmath();
    group.bench_with_input(
        BenchmarkId::new("cgmath", 2),
        &(a2, b2),
        |bench, &(a, b)| bench.iter(|| a * b),
    );

    let a3 = matrix3();
    let b3 = matrix3();
    group.bench_with_input(BenchmarkId::new("soap", 3), &(a3, b3), |bench, &(a, b)| {
        bench.iter(|| a * b)
    });
    let a3 = matrix3_cgmath();
    let b3 = matrix3_cgmath();
    group.bench_with_input(
        BenchmarkId::new("cgmath", 3),
        &(a3, b3),
        |bench, &(a, b)| bench.iter(|| a * b),
    );

    let a4 = matrix4();
    let b4 = matrix4();
    group.bench_with_input(BenchmarkId::new("soap", 4), &(a4, b4), |bench, &(a, b)| {
        bench.iter(|| a * b)
    });
    let a4 = matrix4_cgmath();
    let b4 = matrix4_cgmath();
    group.bench_with_input(
        BenchmarkId::new("cgmath", 4),
        &(a4, b4),
        |bench, &(a, b)| bench.iter(|| a * b),
    );

    group.finish();
}

criterion_group!(benches, mtx_mtx_mul_benchmark);
criterion_main!(benches);
