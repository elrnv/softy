mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn trace_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Trace");

    group.bench_function(BenchmarkId::new("soap", 2), |b| {
        let m2 = matrix2();
        b.iter(|| m2.trace())
    });

    group.bench_function(BenchmarkId::new("cgmath", 2), |b| {
        let m2 = matrix2_cgmath();
        b.iter(|| m2.trace())
    });

    group.bench_function(BenchmarkId::new("soap", 3), |b| {
        let m3 = matrix3();
        b.iter(|| m3.trace());
    });

    group.bench_function(BenchmarkId::new("cgmath", 3), |b| {
        let m3 = matrix3_cgmath();
        b.iter(|| m3.trace());
    });

    group.bench_function(BenchmarkId::new("soap", 4), |b| {
        let m4 = matrix4();
        b.iter(|| m4.trace());
    });

    group.bench_function(BenchmarkId::new("cgmath", 4), |b| {
        let m4 = matrix4_cgmath();
        b.iter(|| m4.trace());
    });

    group.finish();
}

criterion_group!(benches, trace_benchmark);
criterion_main!(benches);
