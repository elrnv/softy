mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn mtx_vec_mul_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix*Vector");

    let m2 = matrix2();
    let v2 = vector2();
    group.bench_with_input(BenchmarkId::new("soap", 2), &(m2, v2), |b, &(m, v)| {
        b.iter(|| m * v)
    });
    let m2 = matrix2_cgmath();
    let v2 = vector2_cgmath();
    group.bench_with_input(BenchmarkId::new("cgmath", 2), &(m2, v2), |b, &(m, v)| {
        b.iter(|| m * v)
    });

    let m3 = matrix3();
    let v3 = vector3();
    group.bench_with_input(BenchmarkId::new("soap", 3), &(m3, v3), |b, &(m, v)| {
        b.iter(|| m * v)
    });
    let m3 = matrix3_cgmath();
    let v3 = vector3_cgmath();
    group.bench_with_input(BenchmarkId::new("cgmath", 3), &(m3, v3), |b, &(m, v)| {
        b.iter(|| m * v)
    });

    let m4 = matrix4();
    let v4 = vector4();
    group.bench_with_input(BenchmarkId::new("soap", 4), &(m4, v4), |b, &(m, v)| {
        b.iter(|| m * v)
    });
    let m4 = matrix4_cgmath();
    let v4 = vector4_cgmath();
    group.bench_with_input(BenchmarkId::new("cgmath", 4), &(m4, v4), |b, &(m, v)| {
        b.iter(|| m * v)
    });

    group.finish();
}

criterion_group!(benches, mtx_vec_mul_benchmark);
criterion_main!(benches);
