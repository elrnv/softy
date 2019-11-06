mod preamble;

use preamble::*;
use criterion::{criterion_group, criterion_main, Criterion, Fun};

fn norm_squared_benchmark(c: &mut Criterion) {
    let mat4_norm_squared = Fun::new("Matrix4 norm squared", move |b, _| {
        let a = matrix4();
        b.iter(|| a.frob_norm_squared())
    });
    let mat4_map_norm_squared = Fun::new("Matrix4 map norm squared", move |b, _| {
        let a = matrix4();
        b.iter(|| a.clone().map_inner(|x| x * x).sum_inner())
    });
    let mat4_optimal_norm_squared = Fun::new("Matrix4 optimal norm squared", move |b, _| {
        let a = matrix4();
        b.iter(|| {
                a[0][0] * a[0][0]
                    + a[0][1] * a[0][1]
                    + a[0][2] * a[0][2]
                    + a[0][3] * a[0][3]
                    + a[1][0] * a[1][0]
                    + a[1][1] * a[1][1]
                    + a[1][2] * a[1][2]
                    + a[1][3] * a[1][3]
                    + a[2][0] * a[2][0]
                    + a[2][1] * a[2][1]
                    + a[2][2] * a[2][2]
                    + a[2][3] * a[2][3]
                    + a[3][0] * a[3][0]
                    + a[3][1] * a[3][1]
                    + a[3][2] * a[3][2]
                    + a[3][3] * a[3][3]
        })
    });
        
    let fns = vec![mat4_norm_squared, mat4_map_norm_squared, mat4_optimal_norm_squared];
    c.bench_functions("Norm Squared", fns, ());
}

criterion_group!(benches, norm_squared_benchmark);
criterion_main!(benches);
