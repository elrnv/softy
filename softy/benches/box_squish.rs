use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use geo::mesh::PolyMesh;
use geo::ops::*;
use softy::{test_utils::*, *};

const SQUISH_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, 0.0, 0.0],
    time_step: Some(0.001),
    ..DYNAMIC_PARAMS
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

fn box_squish(c: &mut Criterion) {
    let mut group = c.benchmark_group("Box Squish");

    for i in (2..20).step_by(4) {
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

        let mut builder = SolverBuilder::new(SQUISH_PARAMS);
        builder
            .add_solid(contact_box.clone(), stretch_material())
            .add_fixed(top_grid.clone(), 1)
            .add_fixed(bottom_grid.clone(), 1)
            .add_frictional_contact(fc_point, (1, 0));

        let mut builder_lin = SolverBuilder::new(SQUISH_PARAMS);
        builder_lin
            .add_solid(contact_box.clone(), stretch_material())
            .add_fixed(top_grid.clone(), 1)
            .add_fixed(bottom_grid.clone(), 1)
            .add_frictional_contact(fc_lin_point, (1, 0));

        group.bench_function(BenchmarkId::new("Point Contact Grid 10", i), |b| {
            b.iter_batched_ref(
                || builder.build().unwrap(),
                |engine| engine.step().is_ok(),
                BatchSize::NumIterations(15),
            )
        });

        group.bench_function(
            BenchmarkId::new("Linearized Point Contact Grid 10", i),
            |b| {
                b.iter_batched_ref(
                    || builder_lin.build().unwrap(),
                    |engine| engine.step().is_ok(),
                    BatchSize::NumIterations(15),
                )
            },
        );

        // 20x20 grids
        let (top_grid, bottom_grid) = make_grids(20);

        let mut builder = SolverBuilder::new(SQUISH_PARAMS);
        builder
            .add_solid(contact_box.clone(), stretch_material())
            .add_fixed(top_grid.clone(), 1)
            .add_fixed(bottom_grid.clone(), 1)
            .add_frictional_contact(fc_point, (1, 0));

        let mut builder_lin = SolverBuilder::new(SQUISH_PARAMS);
        builder_lin
            .add_solid(contact_box.clone(), stretch_material())
            .add_fixed(top_grid.clone(), 1)
            .add_fixed(bottom_grid.clone(), 1)
            .add_frictional_contact(fc_lin_point, (1, 0));

        group.bench_function(BenchmarkId::new("Point Contact Grid 20", i), |b| {
            b.iter_batched_ref(
                || builder.build().unwrap(),
                |engine| engine.step().is_ok(),
                BatchSize::NumIterations(15),
            )
        });

        group.bench_function(
            BenchmarkId::new("Linearized Point Contact Grid 20", i),
            |b| {
                b.iter_batched_ref(
                    || builder_lin.build().unwrap(),
                    |engine| engine.step().is_ok(),
                    BatchSize::NumIterations(15),
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = box_squish
);
criterion_main!(benches);
