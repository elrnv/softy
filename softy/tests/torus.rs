mod test_utils;
pub use test_utils::*;

// Execute the following for release builds only since they are computationally intensive.
#[cfg(not(debug_assertions))]
pub mod complex_tests {
    use super::*;
    use geo;
    use softy::fem::nl;
    use softy::*;

    fn stiff_material() -> SolidMaterial {
        default_solid().with_elasticity(Elasticity::from_bulk_shear(1750e6, 10e6))
    }

    #[test]
    #[cfg(feature = "optsolver")]
    fn torus_medium_test() -> Result<(), Error> {
        use softy::fem::opt;
        init_logger();
        let mesh = geo::io::load_tetmesh("assets/torus_tets.vtk").unwrap();
        let mut solver = opt::SolverBuilder::new(opt::SimParams {
            print_level: 0,
            ..DYNAMIC_OPT_PARAMS
        })
        .add_solid(mesh, stiff_material())
        .build()
        .unwrap();
        solver.step()?;
        Ok(())
    }

    #[test]
    #[cfg(feature = "optsolver")]
    fn torus_long_test() -> Result<(), Error> {
        use softy::fem::opt;
        init_logger();
        let mesh = geo::io::load_tetmesh("assets/torus_tets.vtk")?;
        let mut solver = opt::SolverBuilder::new(DYNAMIC_OPT_PARAMS)
            .add_solid(mesh, stiff_material())
            .build()?;

        for _i in 0..10 {
            //geo::io::save_tetmesh_ascii(
            //    &solver.borrow_mesh(),
            //    &format!("./out/mesh_{}.vtk", 1),
            //    ).unwrap();
            solver.step()?;
        }
        Ok(())
    }

    #[test]
    fn torus_medium_nl_test() -> Result<(), Error> {
        init_logger();
        let mesh = geo::io::load_tetmesh("assets/torus_tets.vtk")?;
        let params = nl::SimParams {
            time_step: Some(0.01),
            ..static_nl_params(0)
        };
        let mut solver = nl::SolverBuilder::new(params)
            .set_mesh(Mesh::from(mesh))
            .set_materials(vec![stiff_material().into()])
            .build::<f64>()?;
        solver.step()?;
        Ok(())
    }

    #[test]
    fn torus_long_nl_test() -> Result<(), Error> {
        init_logger();
        let mesh = geo::io::load_tetmesh("assets/torus_tets.vtk")?;
        let params = nl::SimParams {
            time_step: Some(0.01),
            ..static_nl_params(0)
        };
        let mut solver = nl::SolverBuilder::new(params)
            .set_mesh(Mesh::from(mesh))
            .set_materials(vec![stiff_material().into()])
            .build::<f64>()?;

        for _i in 0..10 {
            //geo::io::save_tetmesh_ascii(
            //    &solver.borrow_mesh(),
            //    &format!("./out/mesh_{}.vtk", 1),
            //    ).unwrap();
            solver.step()?;
        }
        Ok(())
    }
}
