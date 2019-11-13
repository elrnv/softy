mod test_utils;

use softy::*;
//use std::path::PathBuf;
use geo::mesh::builder::*;
use geo::ops::transform::*;
pub use test_utils::*;

pub fn medium_solid_material() -> SolidMaterial {
    SOLID_MATERIAL.with_elasticity(ElasticityParameters::from_bulk_shear(10e6, 1e6))
}

#[test]
fn stacking_boxes() -> Result<(), Error> {
    let material = medium_solid_material();
    let params = SimParams {
        max_iterations: 200,
        outer_tolerance: 0.1,
        max_outer_iterations: 20,
        gravity: [0.0f32, -9.81, 0.0],
        time_step: Some(0.0208333),
        friction_iterations: 1,
        print_level: 0,
        ..DYNAMIC_PARAMS
    };

    let mut grid = GridBuilder {
        rows: 8,
        cols: 8,
        orientation: AxisPlaneOrientation::ZX,
    }.build();

    grid.scale([3.0, 1.0, 3.0]);
    grid.translate([0.0, -1.4, 0.0]);

    let box_bottom = SolidBoxBuilder { res: [2, 2, 2] }.build();
    //let mut box_top = make_box([2, 2, 2]);
    let mut box_top = PlatonicSolidBuilder::build_tetrahedron();
    box_top.translate([0.0, 2.2, 0.0]);

    let fc_params = FrictionalContactParams {
        contact_type: ContactType::Point,
        kernel: KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 0.0001,
        },
        friction_params: Some(FrictionParams {
            dynamic_friction: 0.5,
            inner_iterations: 100,
            tolerance: 1e-5,
            print_level: 0,
        }),
    };

    let mut solver = SolverBuilder::new(params.clone())
        .add_solid(box_bottom, material.with_id(1))
        .add_solid(box_top, material.with_id(2))
        .add_fixed(grid, 0)
        .add_frictional_contact(fc_params, (0, 1))
        .add_frictional_contact(fc_params, (0, 2))
        //.add_frictional_contact(fc_params, (1, 2))
        //.add_frictional_contact(fc_params, (2, 1))
        .build()?;

    //geo::io::save_polymesh(
    //    &PolyMesh::from(solver.shell(0).trimesh.clone()),
    //    "./out/grid.vtk",
    //)
    //.unwrap();

    for i in 0..50 {
        let res = solver.step()?;
        println!("res = {:?}; rame = {:?}", res, i);
        //geo::io::save_tetmesh(
        //    &solver.solid(0).tetmesh,
        //    &PathBuf::from(format!("./out/box_bottom_{}.vtk", i)),
        //)
        //.unwrap();
        //geo::io::save_tetmesh(
        //    &solver.solid(1).tetmesh,
        //    &PathBuf::from(format!("./out/box_top_{}.vtk", i)),
        //)
        //.unwrap();
        assert!(
            res.iterations <= params.max_outer_iterations,
            "Exceeded max outer iterations."
        );
    }

    Ok(())
}
