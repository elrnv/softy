/**
 * This module benchmarks the performance of the FEM engine.
 */

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use fem::run;
    use geo::mesh::{Attrib, TetMesh};
    use geo::topology::VertexIndex;
    use self::test::{Bencher};

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

        b.iter(|| run(&mut mesh, || true))
    }
}
