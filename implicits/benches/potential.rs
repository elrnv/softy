use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use geo::mesh::{builder::*, VertexPositions};
use implicits::{
    surface_from_polymesh, ImplicitSurface, KernelType, Params, QueryTopo, SampleType,
};
use rayon::prelude::*;

fn potential_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallelism");

    let params = Params {
        kernel: KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 1e-5,
        },
        sample_type: SampleType::Face,
        ..Params::default()
    };
    let torus = TorusBuilder {
        outer_radius: 0.5,
        inner_radius: 0.25,
        outer_divs: 100,
        inner_divs: 100,
    }
    .build();
    let mut grid = GridBuilder {
        rows: 100,
        cols: 100,
        orientation: AxisPlaneOrientation::ZX,
    }
    .build();

    let torus_surface = surface_from_polymesh(&torus, params).unwrap();

    let torus_mls = if let ImplicitSurface::MLS(mls) = torus_surface.clone() {
        mls
    } else {
        panic!("Invalid surface.")
    };

    let mut pos = grid.vertex_positions().to_vec();
    let query_surf = QueryTopo::new(&pos, torus_mls);

    let jac1 = query_surf
        .surface_jacobian_indexed_block_par_iter(&mut pos)
        .collect::<Vec<_>>();
    let jac2 = query_surf
        .surface_jacobian_block_iter(&mut pos)
        .collect::<Vec<_>>();
    for (j2, (_, _, j1)) in jac2.iter().zip(jac1.iter()) {
        assert!((j1[0] - j2[0]).abs() < 1e-10);
        assert!((j1[1] - j2[1]).abs() < 1e-10);
        assert!((j1[2] - j2[2]).abs() < 1e-10);
    }

    group.bench_function(BenchmarkId::new("Potential", 100), |b| {
        b.iter_batched(
            || torus_surface.clone(),
            |torus_surface| torus_surface.compute_potential_on_mesh(&mut grid, || false),
            BatchSize::LargeInput,
        )
    });

    group.bench_function(BenchmarkId::new("Surface Jacobian", 100), |b| {
        b.iter_batched(
            || query_surf.clone(),
            |query_surf| {
                query_surf
                    .surface_jacobian_block_iter(&mut pos)
                    .collect::<Vec<_>>()
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function(BenchmarkId::new("Surface Jacobian Parallel", 100), |b| {
        b.iter_batched(
            || query_surf.clone(),
            |query_surf| {
                query_surf
                    .surface_jacobian_indexed_block_par_iter(&mut pos)
                    .collect::<Vec<_>>()
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = potential_benchmark
);
criterion_main!(benches);
