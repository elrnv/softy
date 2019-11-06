mod preamble;
use preamble::*;
use criterion::{criterion_group, criterion_main, Criterion, Fun};

fn sum_benchmark(c: &mut Criterion) {
    // Benchmark fold vs sum
    let mat4_fold_sum = Fun::new("Matrix4 fold sum", move |b, _| {
        let a = matrix4();
        b.iter(|| a.fold_inner(0.0, |acc, x| x + acc))
    });
    let mat4_sum_inner = Fun::new("Matrix4 sum inner", move |b, _| {
        let a = matrix4();
        b.iter(|| a.sum_inner())
    });
    let mat4_optimal_sum = Fun::new("Matrix4 optimal sum", move |b, _| {
        let a = matrix4().into_inner();
        b.iter(|| {
            unsafe {
                a.get_unchecked(0).get_unchecked(0)
                    + a.get_unchecked(0).get_unchecked(1)
                    + a.get_unchecked(0).get_unchecked(2)
                    + a.get_unchecked(0).get_unchecked(3)
                    + a.get_unchecked(1).get_unchecked(0)
                    + a.get_unchecked(1).get_unchecked(1)
                    + a.get_unchecked(1).get_unchecked(2)
                    + a.get_unchecked(1).get_unchecked(3)
                    + a.get_unchecked(2).get_unchecked(0)
                    + a.get_unchecked(2).get_unchecked(1)
                    + a.get_unchecked(2).get_unchecked(2)
                    + a.get_unchecked(2).get_unchecked(3)
                    + a.get_unchecked(3).get_unchecked(0)
                    + a.get_unchecked(3).get_unchecked(1)
                    + a.get_unchecked(3).get_unchecked(2)
                    + a.get_unchecked(3).get_unchecked(3)
            }
        })
    });
    let fns = vec![mat4_fold_sum, mat4_sum_inner, mat4_optimal_sum];
    c.bench_functions("Sum", fns, ());
}

criterion_group!(benches, sum_benchmark);
criterion_main!(benches);
