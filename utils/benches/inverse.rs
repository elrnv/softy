mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn inverse_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inverse");

    group.bench_function(BenchmarkId::new("soap", 2), |b| {
        let m2 = matrix2();
        b.iter(|| m2.inverse())
    });

    group.bench_function(BenchmarkId::new("soap_inplace", 2), |b| {
        let mut m2 = matrix2();
        b.iter(|| m2.invert())
    });

    group.bench_function(BenchmarkId::new("cgmath", 2), |b| {
        let m2 = matrix2_cgmath();
        b.iter(|| m2.invert())
    });

    group.bench_function(BenchmarkId::new("soap", 3), |b| {
        let m3 = matrix3();
        b.iter(|| m3.inverse());
    });

    group.bench_function(BenchmarkId::new("soap_inplace", 3), |b| {
        let mut m3 = matrix3();
        b.iter(|| m3.invert())
    });

    group.bench_function(BenchmarkId::new("cgmath", 3), |b| {
        let m3 = matrix3_cgmath();
        b.iter(|| m3.invert());
    });

    group.finish();
}

criterion_group!(benches, inverse_benchmark);
criterion_main!(benches);
