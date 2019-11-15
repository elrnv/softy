use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use geo::mesh::PolyMesh;
use geo::ops::*;
use softy::{test_utils::*, *};

const STRETCH_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, 0.0, 0.0],
    ..STATIC_PARAMS
};

fn stretch_material() -> SolidMaterial {
    default_solid().with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
}

fn make_grids(i: usize) -> (PolyMesh<f64>, PolyMesh<f64>) {
    let mut bottom_grid = make_grid(i);
    bottom_grid.translate([0.0, -0.39, 0.0]);

    let mut top_grid = make_grid(i);
    top_grid.reverse();
    top_grid.translate([0.0, 0.39, 0.0]);
    (top_grid, bottom_grid)
}

fn box_stretch(c: &mut Criterion) {
    let mut group = c.benchmark_group("Box Stretch");

    let const_volume_stretch_material = stretch_material().with_volume_preservation(true);

    for i in (2..31).step_by(4) {
        let box_mesh = make_stretched_box(i);

        group.bench_function(BenchmarkId::new("Simple", i), |b| {
            let mut engine = SolverBuilder::new(STRETCH_PARAMS)
                .add_solid(box_mesh.clone(), stretch_material())
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        group.bench_function(BenchmarkId::new("Const Volume", i), |b| {
            let mut engine = SolverBuilder::new(STRETCH_PARAMS)
                .add_solid(box_mesh.clone(), const_volume_stretch_material)
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = box_stretch
);
criterion_main!(benches);
