mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn frob_norm_squared_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Frob Norm Squared");

    let m = matrix4();
    group.bench_with_input(BenchmarkId::new("frob_norm_squared", 4), &m, |b, &m| {
        b.iter(|| m.frob_norm_squared())
    });

    let m = matrix4();
    group.bench_with_input(BenchmarkId::new("map_inner", 4), &m, |b, &m| {
        b.iter(|| m.map_inner(|x| x * x).sum_inner())
    });

    let m = matrix4().into_data();
    group.bench_with_input(BenchmarkId::new("benchmark", 4), &m, |b, &m| {
        b.iter(|| unsafe {
            m.get_unchecked(0).get_unchecked(0) * m.get_unchecked(0).get_unchecked(0)
                + m.get_unchecked(0).get_unchecked(1) * m.get_unchecked(0).get_unchecked(1)
                + m.get_unchecked(0).get_unchecked(2) * m.get_unchecked(0).get_unchecked(2)
                + m.get_unchecked(0).get_unchecked(3) * m.get_unchecked(0).get_unchecked(3)
                + m.get_unchecked(1).get_unchecked(0) * m.get_unchecked(1).get_unchecked(0)
                + m.get_unchecked(1).get_unchecked(1) * m.get_unchecked(1).get_unchecked(1)
                + m.get_unchecked(1).get_unchecked(2) * m.get_unchecked(1).get_unchecked(2)
                + m.get_unchecked(1).get_unchecked(3) * m.get_unchecked(1).get_unchecked(3)
                + m.get_unchecked(2).get_unchecked(0) * m.get_unchecked(2).get_unchecked(0)
                + m.get_unchecked(2).get_unchecked(1) * m.get_unchecked(2).get_unchecked(1)
                + m.get_unchecked(2).get_unchecked(2) * m.get_unchecked(2).get_unchecked(2)
                + m.get_unchecked(2).get_unchecked(3) * m.get_unchecked(2).get_unchecked(3)
                + m.get_unchecked(3).get_unchecked(0) * m.get_unchecked(3).get_unchecked(0)
                + m.get_unchecked(3).get_unchecked(1) * m.get_unchecked(3).get_unchecked(1)
                + m.get_unchecked(3).get_unchecked(2) * m.get_unchecked(3).get_unchecked(2)
                + m.get_unchecked(3).get_unchecked(3) * m.get_unchecked(3).get_unchecked(3)
        })
    });

    group.finish();
}

criterion_group!(benches, frob_norm_squared_benchmark);
criterion_main!(benches);
