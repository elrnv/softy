#[macro_use]
mod preamble;
use preamble::*;
use std::ops::Mul;
use criterion::{criterion_group, criterion_main, Bencher, Criterion, Fun};

// Bench local math library against cgmath. The performance should be similar if loop unrolling
// works correctly.
bench_binop!(mat4_mul_mat4_cgmath, Mul::mul, matrix4_cgmath, matrix4_cgmath);
bench_binop!(mat3_mul_mat3_cgmath, Mul::mul, matrix3_cgmath, matrix3_cgmath);
bench_binop!(mat2_mul_mat2_cgmath, Mul::mul, matrix2_cgmath, matrix2_cgmath);

bench_binop!(mat4_mul_mat4, Mul::mul, matrix4, matrix4);
bench_binop!(mat3_mul_mat3, Mul::mul, matrix3, matrix3);
bench_binop!(mat2_mul_mat2, Mul::mul, matrix2, matrix2);

bench_binop_ref!(mat4_mul_mat4_ref, Mul::mul, matrix4, matrix4);
bench_binop_ref!(mat3_mul_mat3_ref, Mul::mul, matrix3, matrix3);
bench_binop_ref!(mat2_mul_mat2_ref, Mul::mul, matrix2, matrix2);

fn mtx_mtx_mul_benchmark(c: &mut Criterion) {
    let mat2_mul_mat2_cgmath = Fun::new("Matrix2 * Matrix2 cgmath", move |b, _| {
        mat2_mul_mat2_cgmath(b)
    });
    let mat3_mul_mat3_cgmath = Fun::new("Matrix3 * Matrix3 cgmath", move |b, _| {
        mat3_mul_mat3_cgmath(b)
    });
    let mat4_mul_mat4_cgmath = Fun::new("Matrix4 * Matrix4 cgmath", move |b, _| {
        mat4_mul_mat4_cgmath(b)
    });
    let mat2_mul_mat2 = Fun::new("Matrix2 * Matrix2", move |b, _| {
        mat2_mul_mat2(b)
    });
    let mat3_mul_mat3 = Fun::new("Matrix3 * Matrix3", move |b, _| {
        mat3_mul_mat3(b)
    });
    let mat4_mul_mat4 = Fun::new("Matrix4 * Matrix4", move |b, _| {
        mat4_mul_mat4(b)
    });
    let mat2_mul_mat2_ref = Fun::new("Matrix2 * Matrix2 Ref", move |b, _| {
        mat2_mul_mat2_ref(b)
    });
    let mat3_mul_mat3_ref = Fun::new("Matrix3 * Matrix3 Ref", move |b, _| {
        mat3_mul_mat3_ref(b)
    });
    let mat4_mul_mat4_ref = Fun::new("Matrix4 * Matrix4 Ref", move |b, _| {
        mat4_mul_mat4_ref(b)
    });
        
    let fns = vec![
        mat2_mul_mat2_cgmath,
        mat3_mul_mat3_cgmath,
        mat4_mul_mat4_cgmath,
        mat2_mul_mat2,
        mat3_mul_mat3,
        mat4_mul_mat4,
        mat2_mul_mat2_ref,
        mat3_mul_mat3_ref,
        mat4_mul_mat4_ref,
    ];
    c.bench_functions("Matrix*Matrix Multiplication", fns, ());
}

criterion_group!(benches, mtx_mtx_mul_benchmark);
criterion_main!(benches);
