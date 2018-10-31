#[macro_use]
extern crate criterion;
extern crate geometry as geo;
extern crate softy;

use criterion::{Criterion, Fun};
use crate::geo::mesh::{topology::*, Attrib, TetMesh};
use softy::fem::{self, MaterialProperties, ElasticityProperties, SimParams};
use std::path::PathBuf;

const DYNAMIC_PARAMS: SimParams = SimParams {
    material: MaterialProperties {
        elasticity: ElasticityProperties {
            bulk_modulus: 1e6,
            shear_modulus: 1e5,
        },
        density: 1000.0,
        damping: 0.0,
    },
    gravity: [0.0f32, 0.0, 0.0],
    time_step: Some(0.01),
    tolerance: 1e-9,
    max_iterations: 800,
    volume_constraint: false,
};

const STRETCH_PARAMS: SimParams = SimParams {
    material: MaterialProperties {
        elasticity: ElasticityProperties {
            bulk_modulus: 300e6,
            shear_modulus: 100e6,
        },
        density: 1000.0,
        damping: 0.0,
    },
    gravity: [0.0f32, 0.0, 0.0],
    time_step: None,
    tolerance: 1e-9,
    max_iterations: 800,
    volume_constraint: false,
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

            let mut engine = fem::Solver::new(mesh, DYNAMIC_PARAMS).unwrap();

            b.iter(|| {
                engine.solve_step().is_ok();
            })
        })
    };

    let box_stretch = {
        Fun::new("Box Stretch", move |b, _| {
            let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk")).unwrap();
            let mut engine = fem::Solver::new(mesh, STRETCH_PARAMS).unwrap();
            b.iter(|| engine.solve_step().is_ok())
        })
    };

    let box_stretch_const_volume = {
        let params = SimParams {
            volume_constraint: true,
            ..STRETCH_PARAMS
        };
        Fun::new("Box Stretch Const Volume", move |b, _| {
            let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box_stretch.vtk")).unwrap();
            let mut engine = fem::Solver::new(mesh, params).unwrap();
            b.iter(|| engine.solve_step().is_ok())
        })
    };

    let fns = vec![three_tets, box_stretch, box_stretch_const_volume];
    c.bench_functions("step_performance", fns, ());
}

criterion_group!(benches, step_performance);
criterion_main!(benches);
