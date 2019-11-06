#[macro_use]
mod preamble;

use preamble::*;
use std::ops::Mul;
use criterion::{criterion_group, criterion_main, Bencher, Criterion, Fun};

// Bench local math library against cgmath. The performance should be similar if loop unrolling
// works correctly.
bench_binop!(mat4_mul_vec4_cgmath, Mul::mul, matrix4_cgmath, vector4_cgmath);
bench_binop!(mat3_mul_vec3_cgmath, Mul::mul, matrix3_cgmath, vector3_cgmath);
bench_binop!(mat2_mul_vec2_cgmath, Mul::mul, matrix2_cgmath, vector2_cgmath);

bench_binop!(mat4_mul_vec4, Mul::mul, matrix4, vector4);
bench_binop!(mat3_mul_vec3, Mul::mul, matrix3, vector3);
bench_binop!(mat2_mul_vec2, Mul::mul, matrix2, vector2);

fn mtx_vec_mul_benchmark(c: &mut Criterion) {
    let mat2_mul_vec2_cgmath = Fun::new("Matrix2 * Vector2 cgmath", move |b, _| {
        mat2_mul_vec2_cgmath(b)
    });
    let mat3_mul_vec3_cgmath = Fun::new("Matrix3 * Vector3 cgmath", move |b, _| {
        mat3_mul_vec3_cgmath(b)
    });
    let mat4_mul_vec4_cgmath = Fun::new("Matrix4 * Vector4 cgmath", move |b, _| {
        mat4_mul_vec4_cgmath(b)
    });
    let mat2_mul_vec2 = Fun::new("Matrix2 * Vector2", move |b, _| {
        mat2_mul_vec2(b)
    });
    let mat3_mul_vec3 = Fun::new("Matrix3 * Vector3", move |b, _| {
        mat3_mul_vec3(b)
    });
    let mat4_mul_vec4 = Fun::new("Matrix4 * Vector4", move |b, _| {
        mat4_mul_vec4(b)
    });
        
    let fns = vec![
        mat2_mul_vec2_cgmath,
        mat3_mul_vec3_cgmath,
        mat4_mul_vec4_cgmath,
        mat2_mul_vec2,
        mat3_mul_vec3,
        mat4_mul_vec4,
    ];
    c.bench_functions("Matrix*Vector Multiplication", fns, ());
}

criterion_group!(benches, mtx_vec_mul_benchmark);
criterion_main!(benches);
