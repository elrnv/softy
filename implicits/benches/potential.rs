use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use implicits::{surface_from_polymesh, KernelType, Params, SampleType};
use utils::*;
use geo::mesh::builder::*;

fn potential_benchmark(c: &mut Criterion) {
    let params = Params {
        kernel: KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 1e-5,
        },
        sample_type: SampleType::Face,
        ..Params::default()
    };
    let torus = TorusBuilder { outer_radius: 0.5, inner_radius: 0.25, outer_divs: 100, inner_divs: 100 }.build();
    let mut grid = GridBuilder {
        rows: 100,
        cols: 100,
        orientation: AxisPlaneOrientation::ZX,
    }.build();
    let torus_surface = surface_from_polymesh(&torus, params).unwrap();

    c.bench_function("Potential", |b| {
        b.iter_batched(
            || torus_surface.clone(),
            |torus_surface| torus_surface.compute_potential_on_mesh(&mut grid, || false),
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
