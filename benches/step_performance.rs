#[macro_use]
extern crate criterion;
extern crate geometry as geo;
extern crate softy;

use criterion::{Criterion, Fun};
use geo::mesh::{topology::*, Attrib, TetMesh};
use softy::{FemEngine, MaterialProperties, SimParams};
use std::path::PathBuf;

const DYNAMIC_PARAMS: SimParams = SimParams {
    material: MaterialProperties {
        bulk_modulus: 1e6,
        shear_modulus: 1e5,
        density: 1000.0,
        damping: 0.0,
    },
    gravity: [0.0f32, 0.0, 0.0],
    time_step: Some(0.01),
    tolerance: 1e-9,
};

fn step_performance(c: &mut Criterion) {
    let three_tets = {
        Fun::new("Three Tets", move |b, _| {
            let verts = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
            ];
            let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
            let mut mesh = TetMesh::new(verts, indices);

            let ref_verts = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ];

            mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
                .ok();

            let mut engine = FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap();

            b.iter(|| {
                engine.solve_step().is_ok();
            })
        })
    };

    let torus_medium = {
        Fun::new("Torus 30K", move |b, _| {
            let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
            let mut engine = FemEngine::new(mesh, DYNAMIC_PARAMS).unwrap();
            b.iter(|| engine.solve_step().is_ok())
        })
    };

    let fns = vec![three_tets, torus_medium];
    c.bench_functions("step_performance", fns, ());
}

criterion_group!(benches, step_performance);
criterion_main!(benches);
