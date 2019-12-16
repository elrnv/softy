mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn determinant_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Determinant");

    group.bench_function(BenchmarkId::new("soap", 2), |b| {
        let m = matrix2();
        b.iter(|| m.determinant())
    });

    group.bench_function(BenchmarkId::new("cgmath", 2), |b| {
        let m = matrix2_cgmath();
        b.iter(|| m.determinant())
    });

    group.bench_function(BenchmarkId::new("soap", 3), |b| {
        let m = matrix3();
        b.iter(|| m.determinant())
    });

    group.bench_function(BenchmarkId::new("cgmath", 3), |b| {
        let m = matrix3_cgmath();
        b.iter(|| m.determinant())
    });

    group.bench_function(BenchmarkId::new("soap", 4), |b| {
        let m = matrix4();
        b.iter(|| m.determinant())
    });

    group.bench_function(BenchmarkId::new("cgmath", 4), |b| {
        let m = matrix4_cgmath();
        b.iter(|| m.determinant())
    });

    group.finish();
}

criterion_group!(benches, determinant_benchmark);
criterion_main!(benches);
