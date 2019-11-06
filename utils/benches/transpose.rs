#[macro_use]
mod preamble;
use preamble::*;
use criterion::{criterion_group, criterion_main, Bencher, Criterion, Fun};

// Benchmark transpose computation against cgmath
bench_uniop!(mat2_transpose, soap::Matrix::transpose, matrix2);
bench_uniop_ref!(mat2_transpose_cgmath, cgmath::Matrix::transpose, matrix2_cgmath);
bench_uniop!(mat3_transpose, soap::Matrix::transpose, matrix3);
bench_uniop_ref!(mat3_transpose_cgmath, cgmath::Matrix::transpose, matrix3_cgmath);
bench_uniop!(mat4_transpose, soap::Matrix::transpose, matrix4);
bench_uniop_ref!(mat4_transpose_cgmath, cgmath::Matrix::transpose, matrix4_cgmath);

fn transpose_benchmark(c: &mut Criterion) {
    let mat2_transpose = Fun::new("Matrix2 transpose", move |b, _| {
        mat2_transpose(b)
    });
    let mat2_transpose_cgmath = Fun::new("Matrix2 transpose ref cgmath", move |b, _| {
        mat2_transpose_cgmath(b)
    });
    let mat3_transpose = Fun::new("Matrix3 transpose", move |b, _| {
        mat3_transpose(b)
    });
    let mat3_transpose_cgmath = Fun::new("Matrix3 transpose ref cgmath", move |b, _| {
        mat3_transpose_cgmath(b)
    });
    let mat4_transpose = Fun::new("Matrix4 transpose", move |b, _| {
        mat4_transpose(b)
    });
    let mat4_transpose_cgmath = Fun::new("Matrix4 transpose ref cgmath", move |b, _| {
        mat4_transpose_cgmath(b)
    });
        
    let fns = vec![
        mat2_transpose, mat2_transpose_cgmath,
        mat3_transpose, mat3_transpose_cgmath,
        mat4_transpose, mat4_transpose_cgmath
    ];
    c.bench_functions("Transpose", fns, ());
}

criterion_group!(benches, transpose_benchmark);
criterion_main!(benches);
