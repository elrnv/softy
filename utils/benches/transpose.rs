mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn transpose_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transpose");

    group.bench_function(BenchmarkId::new("soap", 3), |b| {
        let m = matrix3();
        b.iter(|| m.transpose())
    });

    group.bench_function(BenchmarkId::new("cgmath_ref", 3), |b| {
        let m = matrix3();
        b.iter(|| m.transpose())
    });

    group.bench_function(BenchmarkId::new("soap", 4), |b| {
        let m = matrix4();
        b.iter(|| m.transpose())
    });

    group.bench_function(BenchmarkId::new("cgmath_ref", 4), |b| {
        let m = matrix4();
        b.iter(|| m.transpose())
    });

    group.finish()
}

criterion_group!(benches, transpose_benchmark);
criterion_main!(benches);
