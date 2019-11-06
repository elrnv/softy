#[macro_use]
mod preamble;
use preamble::*;
use utils::soap;
use criterion::{criterion_group, criterion_main, Bencher, Criterion, Fun};

// Benchmark determinant computation against cgmath
bench_uniop_ref!(mat2_det, soap::Matrix2::determinant, matrix2);
bench_uniop_ref!(mat2_det_cgmath, cgmath::Matrix2::determinant, matrix2_cgmath);
bench_uniop_ref!(mat3_det, soap::Matrix3::determinant, matrix3);
bench_uniop_ref!(mat3_det_cgmath, cgmath::Matrix3::determinant, matrix3_cgmath);
bench_uniop_ref!(mat4_det, soap::Matrix4::determinant, matrix4);
bench_uniop_ref!(mat4_det_cgmath, cgmath::Matrix4::determinant, matrix4_cgmath);

fn determinant_benchmark(c: &mut Criterion) {
    let mat2_det = Fun::new("Matrix2 determinant", move |b, _| {
        mat2_det(b)
    });
    let mat2_det_cgmath = Fun::new("Matrix2 determinant cgmath", move |b, _| {
        mat2_det_cgmath(b)
    });
    let mat3_det = Fun::new("Matrix3 determinant", move |b, _| {
        mat3_det(b)
    });
    let mat3_det_cgmath = Fun::new("Matrix3 determinant cgmath", move |b, _| {
        mat3_det_cgmath(b)
    });
    let mat4_det = Fun::new("Matrix4 determinant", move |b, _| {
        mat4_det(b)
    });
    let mat4_det_cgmath = Fun::new("Matrix4 determinant cgmath", move |b, _| {
        mat4_det_cgmath(b)
    });
        
    let fns = vec![
        mat2_det, mat2_det_cgmath,
        mat3_det, mat3_det_cgmath,
        mat4_det, mat4_det_cgmath
    ];
    c.bench_functions("Inverse", fns, ());
}

criterion_group!(benches, determinant_benchmark);
criterion_main!(benches);
