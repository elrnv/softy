mod test_utils;
pub use test_utils::*;

// Execute the following for release builds only since they are computationally intensive.
#[cfg(not(debug_assertions))]
pub mod complex_tests {
    use super::*;
    use geo;
    use softy::*;
    use std::path::PathBuf;

    fn stiff_material() -> SolidMaterial {
        SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(1750e6, 10e6))
    }

    #[test]
    fn torus_medium_test() -> Result<(), Error> {
        init_logger();
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        let mut solver = SolverBuilder::new(SimParams {
            print_level: 0,
            ..DYNAMIC_PARAMS
        })
        .add_solid(mesh, stiff_material())
        .build()
        .unwrap();
        solver.step()?;
        Ok(())
    }

    #[test]
    fn torus_long_test() -> Result<(), Error> {
        init_logger();
        let mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk"))?;
        let mut solver = SolverBuilder::new(DYNAMIC_PARAMS)
            .add_solid(mesh, stiff_material())
            .build()?;

        for _i in 0..10 {
            //geo::io::save_tetmesh_ascii(
            //    &solver.borrow_mesh(),
            //    &PathBuf::from(format!("./out/mesh_{}.vtk", 1)),
            //    ).unwrap();
            solver.step()?;
        }
        Ok(())
    }
}
