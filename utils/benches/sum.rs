mod preamble;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use preamble::*;

fn sum_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sum");

    group.bench_function(BenchmarkId::new("fold_inner", 4), |b| {
        let m = matrix4();
        b.iter(|| m.fold_inner(0.0, |acc, x| x + acc))
    });

    group.bench_function(BenchmarkId::new("sum_inner", 4), |b| {
        let m = matrix4();
        b.iter(|| m.sum_inner())
    });

    group.bench_function(BenchmarkId::new("benchmark", 4), |b| {
        let m = matrix4().into_data();
        b.iter(|| unsafe {
            m.get_unchecked(0).get_unchecked(0)
                + m.get_unchecked(0).get_unchecked(1)
                + m.get_unchecked(0).get_unchecked(2)
                + m.get_unchecked(0).get_unchecked(3)
                + m.get_unchecked(1).get_unchecked(0)
                + m.get_unchecked(1).get_unchecked(1)
                + m.get_unchecked(1).get_unchecked(2)
                + m.get_unchecked(1).get_unchecked(3)
                + m.get_unchecked(2).get_unchecked(0)
                + m.get_unchecked(2).get_unchecked(1)
                + m.get_unchecked(2).get_unchecked(2)
                + m.get_unchecked(2).get_unchecked(3)
                + m.get_unchecked(3).get_unchecked(0)
                + m.get_unchecked(3).get_unchecked(1)
                + m.get_unchecked(3).get_unchecked(2)
                + m.get_unchecked(3).get_unchecked(3)
        })
    });
    group.finish();
}

criterion_group!(benches, sum_benchmark);
criterion_main!(benches);
