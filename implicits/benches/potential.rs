use criterion::{criterion_group, criterion_main, Criterion, BatchSize};

use implicits::{KernelType, SampleType, surface_from_polymesh, Params};
use utils::*;

fn potential_benchmark(c: &mut Criterion) {
    let params = Params {
        kernel: KernelType::Approximate { radius_multiplier: 2.0, tolerance: 1e-5 },    
        sample_type: SampleType::Face,
        ..Params::default()
    };
    let torus = make_torus(0.5, 0.25, 100, 100);
    let mut grid = make_grid(Grid { rows: 100, cols: 100, orientation: AxisPlaneOrientation::ZX });
    let torus_surface = surface_from_polymesh(&torus, params).unwrap();

    c.bench_function("Potential", |b| b.iter_batched(|| torus_surface.clone(), |torus_surface| {
        torus_surface.compute_potential_on_mesh(&mut grid,  || false)
    }, BatchSize::LargeInput));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = potential_benchmark
);
criterion_main!(benches);

