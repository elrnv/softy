/**
 * This module benchmarks the performance of internal maths against other popular libraries.
 */

use rand::{FromEntropy, IsaacRng, Rng};
use crate::soap::*;
use std::ops::Mul;

fn matrix3() -> Matrix3<f64> {
    let mut rng = IsaacRng::from_entropy();
    Matrix3([
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
    ])
}

fn matrix4() -> Matrix4<f64> {
    let mut rng = IsaacRng::from_entropy();
    Matrix4([
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
    ])
}

macro_rules! bench_uniop(
            ($name: ident,  $op: ident, $gen: ident) => {
                    fn $name(bench: &mut Bencher) {
                            let a = $gen();

                            bench.iter(|| {
                                    a.$op()
                            })
                    }
            }
            );

macro_rules! bench_uniop_ref(
            ($name: ident,  $op: ident, $gen: ident) => {
                    fn $name(bench: &mut Bencher) {
                            let a = &$gen();

                            bench.iter(|| {
                                    a.$op()
                            })
                    }
            }
            );

// Benchmark vectorize pass by value vs pass by reference
bench_uniop_ref!(mat3_vectorize, vec, matrix3);
bench_uniop_ref!(mat3_vectorize_ref, vec_ref, matrix3);
bench_uniop_ref!(mat4_vectorize, vec, matrix4);
bench_uniop_ref!(mat4_vectorize_ref, vec_ref, matrix4);

fn vectorize_benchmark(c: &mut Criterion) {
    let mat3_vectorize = Fun::new("Matrix3 vectorize", move |b, _| {
        mat3_vectorize(b)
    });
    let mat3_vectorize_ref = Fun::new("Matrix3 vectorize ref", move |b, _| {
        mat3_vectorize_ref(b)
    });
    let mat4_vectorize = Fun::new("Matrix4 vectorize", move |b, _| {
        mat4_vectorize(b)
    });
    let mat4_vectorize_ref = Fun::new("Matrix4 vectorize ref", move |b, _| {
        mat4_vectorize_ref(b)
    });
        
    let fns = vec![mat3_vectorize, mat3_vectorize_ref, mat4_vectorize, mat4_vectorize_ref];
    c.bench_functions("Vectorize", fns, ());
}

criterion_group!(benches, vectorize_benchmark);
criterion_main!(benches);
