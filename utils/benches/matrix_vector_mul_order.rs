use approx::assert_relative_eq;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rayon::prelude::*;
use utils::soap::*;
use utils::zip;

#[cfg(feature = "packed_simd")]
use packed_simd::*;

/// Generate a random vector of float values between -1 and 1.
pub fn random_vec(n: usize) -> Vec<f64> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-1.0, 1.0);
    (0..n).map(move |_| rng.sample(range)).collect()
}

pub fn outer_write_local(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (i, out) in out.iter_mut().enumerate() {
        for (col, &rhs) in m.iter().zip(v.iter()) {
            *out += col[i] * rhs;
        }
    }
    out
}

pub fn lazy_expr4(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let n = v.len() / 4;
    let m4 =
        ChunkedN::from_flat_with_stride(Chunked4::from_flat(Chunked4::from_flat(m.into_flat())), n);
    let v4 = Chunked4::from_flat(v);
    let mut out4 = Chunked4::from_flat(vec![0.0; v.len()]);
    for (row, out) in m4.iter().zip(out4.iter_mut()) {
        for (col, &rhs) in row.iter().zip(v4.iter()) {
            *out.as_mut_tensor() += (*col.as_matrix()) * Vector4::new(rhs);
        }
    }
    out4.into_flat()
}

pub fn lazy_expr(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    (m.expr() * v.expr()).eval()
}

pub fn outer_read_local(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (col, rhs) in m.iter().zip(v.iter()) {
        for (out, &val) in out.iter_mut().zip(col.iter()) {
            *out += val * rhs;
        }
    }
    out
}
pub fn inner(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (&col, &rhs) in row.iter().zip(v.iter()) {
            *out += col * rhs;
        }
    }
    out
}

#[cfg(feature = "packed_simd")]
pub fn inner_simd(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (col, rhs) in row.chunks_exact(4).zip(v.chunks_exact(4)) {
            unsafe {
                let col4 = f64x4::new(
                    *col.get_unchecked(0),
                    *col.get_unchecked(1),
                    *col.get_unchecked(2),
                    *col.get_unchecked(3),
                );
                let rhs4 = f64x4::new(
                    *rhs.get_unchecked(0),
                    *rhs.get_unchecked(1),
                    *rhs.get_unchecked(2),
                    *rhs.get_unchecked(3),
                );
                *out += (col4 * rhs4).sum();
            }
        }
    }
    out
}

#[cfg(feature = "packed_simd")]
pub fn inner_outer_simd(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    let n = v.len();
    let m = m.into_flat();
    for (rows4, outs4) in m.chunks_exact(4 * n).zip(out.chunks_exact_mut(4)) {
        unsafe {
            let row0 = rows4.get_unchecked(0..n);
            let row1 = rows4.get_unchecked(n..2 * n);
            let row2 = rows4.get_unchecked(2 * n..3 * n);
            let row3 = rows4.get_unchecked(3 * n..);
            let mut outsimd = f64x4::splat(0.0);
            for (&r0, &r1, &r2, &r3, &col) in
                zip!(row0.iter(), row1.iter(), row2.iter(), row3.iter(), v.iter())
            {
                let r = f64x4::new(r0, r1, r2, r3);
                outsimd += r * col;
            }
            outsimd.write_to_slice_aligned_unchecked(outs4);
        }
    }
    out
}

pub fn inner_chunked(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (row, out) in m.iter().zip(out.iter_mut()) {
        for (&col, &rhs) in Chunked4::from_flat(row)
            .iter()
            .zip(Chunked4::from_flat(v).iter())
        {
            *out += Vector4::new(col).dot(Vector4::new(rhs));
        }
    }
    out
}

pub fn outer_read_local_par(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let size = v.len();
    let n = 24;
    let mut out_24 = vec![vec![0.0; size]; n + 1];
    let m = m.into_flat();
    let chunk_size = size / n;
    m.par_chunks(size * chunk_size)
        .zip(v.par_chunks(chunk_size))
        .zip(out_24.par_iter_mut())
        .for_each(|((m, v), out)| {
            m.chunks(size).zip(v.iter()).for_each(|(col, rhs)| {
                out.iter_mut().zip(col.iter()).for_each(|(out_val, &val)| {
                    *out_val += val * rhs;
                });
            });
        });
    let mut out = out_24.pop().unwrap();
    for i in 0..n {
        out.iter_mut()
            .zip(out_24[i].iter())
            .for_each(|(out, outi)| {
                *out += outi;
            });
    }
    out
}
pub fn inner_par(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    m.into_par_iter()
        .zip(out.par_iter_mut())
        .for_each(|(row, out_val)| {
            row.iter().zip(v.iter()).for_each(|(&col, &rhs)| {
                *out_val += col * rhs;
            });
        });
    out
}
pub fn inner_par_chunked(m: ChunkedN<&[f64]>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    let size = v.len();
    let n = 24;
    let chunk_size = size / n;
    let m = m.into_flat();
    m.par_chunks(size * chunk_size)
        .zip(out.par_chunks_mut(chunk_size))
        .for_each(|(m, out)| {
            m.chunks(size)
                .zip(out.iter_mut())
                .for_each(|(row, out_val)| {
                    row.iter().zip(v.iter()).for_each(|(&col, &rhs)| {
                        *out_val += col * rhs;
                    });
                });
        });
    out
}

fn matrix_vector_mul_order_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Vector Multiply Order");

    for &n in &[100, 200, 400, 800, 1600, 3200 /*, 5000, 7500, 10000*/] {
        let mut m = ChunkedN::from_flat_with_stride(random_vec(n * n), n);
        let v = random_vec(n);
        // symmetrize matrix to make consistent results
        for i in 0..n {
            for j in 0..n {
                if j > i {
                    let other = *m.view().at(j).at(i);
                    let val = m.view_mut().isolate(i).isolate(j);
                    *val = other;
                }
            }
        }

        assert_eq!(
            outer_read_local(m.view(), v.view()),
            lazy_expr(m.view(), v.view())
        );
        //assert_eq!(outer_read_local(m.view(), v.view()), lazy_expr4(m.view(), v.view()));
        assert_eq!(
            outer_read_local(m.view(), v.view()),
            outer_write_local(m.view(), v.view())
        );
        assert_eq!(
            inner(m.view(), v.view()),
            outer_read_local(m.view(), v.view())
        );
        assert_eq!(
            inner_par(m.view(), v.view()),
            outer_read_local(m.view(), v.view())
        );

        let approx_equal = |a: Vec<f64>, b: Vec<f64>| {
            for (a, b) in a.iter().zip(b.iter()) {
                assert_relative_eq!(a, b, max_relative = 1e-7, epsilon = 1e-7);
            }
        };

        approx_equal(
            outer_read_local_par(m.view(), v.view()),
            outer_write_local(m.view(), v.view()),
        );

        //if n < 5000 {
        //    group.bench_with_input(BenchmarkId::new("Outer Write Local", n), &(&m, &v),
        //        |b, (m, v)| {
        //            b.iter_batched(|| (m.view(), v.view()), |(m,v)| outer_write_local(m, v), BatchSize::LargeInput)
        //        });
        //}

        group.bench_with_input(BenchmarkId::new("Lazy Expr", n), &(&m, &v), |b, (m, v)| {
            b.iter_batched(
                || (m.view(), v.view()),
                |(m, v)| lazy_expr(m, v),
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(
            BenchmarkId::new("Lazy Expr Blocks", n),
            &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(
                    || (m.view(), v.view()),
                    |(m, v)| lazy_expr4(m, v),
                    BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Outer Read Local", n),
            &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(
                    || (m.view(), v.view()),
                    |(m, v)| outer_read_local(m, v),
                    BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(BenchmarkId::new("Inner", n), &(&m, &v), |b, (m, v)| {
            b.iter_batched(
                || (m.view(), v.view()),
                |(m, v)| inner(m, v),
                BatchSize::LargeInput,
            )
        });

        group.bench_with_input(
            BenchmarkId::new("Outer Read Local Parallel", n),
            &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(
                    || (m.view(), v.view()),
                    |(m, v)| outer_read_local_par(m, v),
                    BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Inner Parallel", n),
            &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(
                    || (m.view(), v.view()),
                    |(m, v)| inner_par(m, v),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Inner Parallel Chunked", n),
            &(&m, &v),
            |b, (m, v)| {
                b.iter_batched(
                    || (m.view(), v.view()),
                    |(m, v)| inner_par_chunked(m, v),
                    BatchSize::LargeInput,
                )
            },
        );
        #[cfg(feature = "packed_simd")]
        {
            approx_equal(
                inner_par(m.view(), v.view()),
                inner_simd(m.view(), v.view()),
            );
            approx_equal(
                inner_par(m.view(), v.view()),
                inner_outer_simd(m.view(), v.view()),
            );
            group.bench_with_input(BenchmarkId::new("Inner SIMD", n), &(&m, &v), |b, (m, v)| {
                b.iter_batched(
                    || (m.view(), v.view()),
                    |(m, v)| inner_simd(m, v),
                    BatchSize::LargeInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("Inner Outer SIMD", n),
                &(&m, &v),
                |b, (m, v)| {
                    b.iter_batched(
                        || (m.view(), v.view()),
                        |(m, v)| inner_outer_simd(m, v),
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = matrix_vector_mul_order_benchmark
);
criterion_main!(benches);
