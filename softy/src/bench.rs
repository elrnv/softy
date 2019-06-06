/**
 * This module benchmarks the performance of the FEM engine.
 */

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use crate::fem::{self, ElasticityProperties, MaterialProperties, SimParams};
    use geo::mesh::topology::VertexIndex;
    use geo::mesh::{Attrib, TetMesh};

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
        clear_velocity: false,
        tolerance: 1e-9,
        max_iterations: 800,
        volume_constraint: false,
    };

    #[bench]
    fn three_tets_bench(b: &mut Bencher) {
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

        b.iter(|| {
            assert!(fem::Solver::new(mesh.clone(), DYNAMIC_PARAMS)
                .unwrap()
                .step()
                .is_ok())
        })
    }

}
