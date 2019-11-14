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

fn box_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Box Stretch");

    let const_volume_stretch_material = stretch_material().with_volume_preservation(true);

    for i in (2..32).step_by(2) {
        let box_mesh = make_stretched_box(i);

        group.bench_function(BenchmarkId::new("Box Stretch", i), |b| {
            let mut engine = SolverBuilder::new(STRETCH_PARAMS)
                .add_solid(box_mesh.clone(), stretch_material())
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        group.bench_function(BenchmarkId::new("Box Stretch Const Volume", i), |b| {
            let mut engine = SolverBuilder::new(STRETCH_PARAMS)
                .add_solid(box_mesh.clone(), const_volume_stretch_material)
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        // Frictionless contact benches
        let mut contact_box = make_box(i);
        contact_box.scale([1.0, 0.8, 1.0]);

        let fc_point = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.1,
                tolerance: 0.001,
            },
            friction_params: None,
        };

        let fc_lin_point = FrictionalContactParams {
            contact_type: ContactType::LinearizedPoint,
            ..fc_point
        };

        // 10x10 grids
        let (top_grid, bottom_grid) = make_grids(10);

        group.bench_function(BenchmarkId::new("Box Point Contact Grid 10", i), |b| {
            let mut engine = SolverBuilder::new(DYNAMIC_PARAMS)
                .add_solid(contact_box.clone(), stretch_material())
                .add_fixed(top_grid.clone(), 1)
                .add_fixed(bottom_grid.clone(), 1)
                .add_frictional_contact(fc_point, (1, 0))
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        group.bench_function(BenchmarkId::new("Box Linearized Point Contact Grid 10", i), |b| {
            let mut engine = SolverBuilder::new(DYNAMIC_PARAMS)
                .add_solid(contact_box.clone(), stretch_material())
                .add_fixed(top_grid.clone(), 1)
                .add_fixed(bottom_grid.clone(), 1)
                .add_frictional_contact(fc_lin_point, (1, 0))
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        // 20x20 grids
        let (top_grid, bottom_grid) = make_grids(20);

        group.bench_function(BenchmarkId::new("Box Point Contact Grid 20", i), |b| {
            let mut engine = SolverBuilder::new(DYNAMIC_PARAMS)
                .add_solid(contact_box.clone(), stretch_material())
                .add_fixed(top_grid.clone(), 1)
                .add_fixed(bottom_grid.clone(), 1)
                .add_frictional_contact(fc_point, (1, 0))
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });

        group.bench_function(BenchmarkId::new("Box Linearized Point Contact Grid 20", i), |b| {
            let mut engine = SolverBuilder::new(DYNAMIC_PARAMS)
                .add_solid(contact_box.clone(), stretch_material())
                .add_fixed(top_grid.clone(), 1)
                .add_fixed(bottom_grid.clone(), 1)
                .add_frictional_contact(fc_lin_point, (1, 0))
                .build()
                .unwrap();
            b.iter(|| engine.step().is_ok())
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(15);
    targets = box_bench
);
criterion_main!(benches);
