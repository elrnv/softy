use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use geo::mesh::{attrib::Attrib, builder::SolidBoxBuilder, topology::*, TetMesh, VertexPositions};
use softy::{test_utils::*, *};

const STRETCH_PARAMS: SimParams = SimParams {
    gravity: [0.0f32, 0.0, 0.0],
    ..STATIC_PARAMS
};

fn stretch_material() -> SolidMaterial {
    default_solid().with_elasticity(ElasticityParameters::from_bulk_shear(300e6, 100e6))
}

fn make_box(i: usize) -> TetMesh<f64> {
    let mut box_mesh = SolidBoxBuilder { res: [i, i, i] }.build();
    let mut fixed = vec![0i8; box_mesh.vertex_positions().len()];

    // Stretch along the x axis
    for (v, f) in box_mesh.vertex_position_iter_mut().zip(fixed.iter_mut()) {
        if v[0] == 0.5 {
            *f = 1;
            v[0] = 1.5;
        }
        if v[0] == -0.5 {
            *f = 1;
            v[0] = -1.5;
        }
    }

    box_mesh
        .add_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, fixed)
        .unwrap();

    box_mesh
}

fn box_stretch(c: &mut Criterion) {
    let mut group = c.benchmark_group("Box Stretch");

    let const_volume_stretch_material = stretch_material().with_volume_preservation(true);

    for i in (2..35).step_by(4) {
        let box_mesh = make_box(i);

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
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(15);
    targets = box_stretch
);
criterion_main!(benches);
