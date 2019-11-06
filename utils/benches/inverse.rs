#[macro_use]
mod preamble;
use preamble::*;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

// Benchmark inverse computation against cgmath
//bench_uniop_ref!(mat2_inverse, soap::Matrix2::inverse, matrix2);
//bench_uniop_ref_mut!(mat2_invert, soap::Matrix2::invert, matrix2);
//bench_uniop_ref!(mat2_inverse_cgmath, cgmath::SquareMatrix::invert, matrix2_cgmath);
//bench_uniop_ref!(mat3_inverse, soap::Matrix3::inverse, matrix3);
//bench_uniop_ref_mut!(mat3_invert, soap::Matrix3::invert, matrix3);
//bench_uniop_ref!(mat3_inverse_cgmath, cgmath::SquareMatrix::invert, matrix3_cgmath);
//bench_uniop!(mat4_inverse, inverse, matrix4);
//bench_uniop!(mat4_inverse_cgmath, inverse, matrix4_cgmath);
fn inverse_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inverse");

    let m2 = &matrix2();
    group.bench_with_input(BenchmarkId::new("soap", "Matrix2"), m2,
        |b, m2| b.iter(|| m2.inverse()));

    group.bench_function(BenchmarkId::new("soap_inplace", "Matrix2"),
        |b| {
            let mut m2 = matrix2();
            b.iter(|| m2.invert())
        });

    let m2_cg = &matrix2_cgmath();
    group.bench_with_input(BenchmarkId::new("cgmath", "Matrix2"), m2_cg,
        |b, m2| b.iter(|| m2.invert()));

    let m3 = &matrix3();
    group.bench_with_input(BenchmarkId::new("soap", "Matrix3"), m3,
        |b, m3| b.iter(|| m3.inverse()));

    group.bench_function(BenchmarkId::new("soap_inplace", "Matrix3"),
        |b| {
            let mut m3 = matrix3();
            b.iter(|| m3.invert())
        });

    let m3_cg = &matrix3_cgmath();
    group.bench_with_input(BenchmarkId::new("cgmath", "Matrix3"), m3_cg,
        |b, m3| b.iter(|| m3.invert()));
        
    group.finish();
}

criterion_group!(benches, inverse_benchmark);
criterion_main!(benches);
