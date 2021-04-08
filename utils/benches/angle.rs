//!
//! This benchmark compares the performance of different methods for computing the angle between
//! two vectors.
//!

mod preamble;
use approx::*;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;
use tensr::Vector3;

// NOTE: the cross product approach has multiple advantages:
// 1. doesn't require the edge vector e
// 2. doesn't require normalizing any vectors:
//    - This means there are no divisions, hence no possibility of division by 0
//    - This makes it more robust to degeneracies.
// 3. is faster (measured by this benchmark)

// n0 and n1 are normals, e is the edge, and t0 is another edge vector for triangle 0
fn via_cross_product(
    mut n0: Vector3<f64>,
    mut n1: Vector3<f64>,
    t0: Vector3<f64>,
    _e: Vector3<f64>,
) -> f64 {
    n0.cross(n1).norm().atan2(n0.dot(n1)) * n1.dot(t0).signum()
}

fn via_dot_products(
    mut n0: Vector3<f64>,
    n1: Vector3<f64>,
    mut t0: Vector3<f64>,
    mut e: Vector3<f64>,
) -> f64 {
    n0.normalize();
    e.normalize();
    t0 -= e * t0.dot(e);
    t0.normalize();

    n1.dot(t0).atan2(n0.dot(n1))
}

fn angle_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Angle");

    let moc = || {
        let n0 = vector3() - Vector3::new([0.5; 3]);
        let n1 = vector3() - Vector3::new([0.5; 3]);
        let mut t0 = vector3() - Vector3::new([0.5; 3]);
        t0 -= n0.normalized() * n0.normalized().dot(t0);
        let e = n0.cross(n1).normalized() * 0.5;
        assert_relative_eq!(n0.dot(t0), 0.0, max_relative = 1e-8, epsilon = 1e-8);
        assert_relative_eq!(e.dot(n0), 0.0, max_relative = 1e-8, epsilon = 1e-8);
        assert_relative_eq!(e.dot(n1), 0.0, max_relative = 1e-8, epsilon = 1e-8);
        assert_relative_eq!(
            via_cross_product(n0, n1, t0, e),
            via_dot_products(n0, n1, t0, e),
            max_relative = 1e-5,
            epsilon = 1e-4,
        );
        (n0, n1, t0, e)
    };

    group.bench_function(BenchmarkId::new("Cross and Dot", 1), |bench| {
        let (a, b, c, e) = moc();
        bench.iter(|| via_cross_product(a, b, c, e))
    });

    group.bench_function(BenchmarkId::new("Two Dots and Project", 1), |bench| {
        let (a, b, c, e) = moc();
        bench.iter(|| via_dot_products(a, b, c, e))
    });

    group.finish();
}

criterion_group!(benches, angle_benchmark);
criterion_main!(benches);
