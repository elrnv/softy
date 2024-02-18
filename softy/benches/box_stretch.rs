use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use softy::fem::nl::{SimParams, SolverBuilder};
use softy::{test_utils::*, *};

fn stretch_params() -> SimParams {
    SimParams {
        gravity: [0.0f32, 0.0, 0.0],
        ..static_nl_params(0)
    }
}

fn stretch_material() -> SolidMaterial {
    default_solid().with_elasticity(Elasticity::from_bulk_shear(300e6, 100e6))
}

fn box_stretch(c: &mut Criterion) {
    let mut group = c.benchmark_group("Box Stretch");

    let const_volume_stretch_material = stretch_material().with_volume_preservation(true);

    for i in (2..18).step_by(4) {
        let box_mesh = make_stretched_box(i);

        group.bench_function(BenchmarkId::new("Simple", i), |b| {
            let mut engine = SolverBuilder::new(stretch_params())
                .set_mesh(box_mesh.clone())
                .set_material(stretch_material())
                .build::<f64>()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        group.bench_function(BenchmarkId::new("Const Volume", i), |b| {
            let mut engine = SolverBuilder::new(stretch_params())
                .set_mesh(box_mesh.clone())
                .set_material(const_volume_stretch_material)
                .build::<f64>()
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
