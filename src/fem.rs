use energy_model::NeohookeanEnergyModel;
use ipopt::{self, Ipopt};
use geo::math::{Matrix3, Vector3};
use geo::topology::*;
use geo::prim::Tetrahedron;
use geo::mesh::{self, attrib, Attrib};
use geo::mesh::tetmesh::TetCell;
use geo::ops::{ShapeMatrix, Volume};
use geo::reinterpret::*;

pub type TetMesh = mesh::TetMesh<f64>;
pub type Tet = Tetrahedron<f64>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialProperties {
    /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
    /// incompressibility. The larger the value, the more incompressible the material is.
    /// Think of this as "Volume Stiffness"
    pub bulk_modulus: f32,
    /// Shear modulus measures the material's resistance to shear deformation. The larger the
    /// value, the more it resists changes in shape. Think of this as "Shape Stiffness"
    pub shear_modulus: f32,
    /// The density of the material.
    pub density: f32,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model.
    pub damping: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SimParams {
    pub material: MaterialProperties,
    pub gravity: [f32;3],
    pub time_step: Option<f32>,
}

/// Get reference tetrahedron.
/// This routine assumes that there is a vertex attribute called `ref` of type `[f64;3]`.
pub fn ref_tet(tetmesh: &TetMesh, indices: &TetCell) -> Tet {
    let attrib = tetmesh.attrib::<VertexIndex>("ref").unwrap();
    Tetrahedron(
        (attrib.get::<[f64; 3], _>(indices[0]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[1]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[2]).unwrap()).into(),
        (attrib.get::<[f64; 3], _>(indices[3]).unwrap()).into(),
    )
}

pub fn run<F>(mesh: &mut TetMesh, params: SimParams, check_interrupt: F) -> Result<(), Error>
where
    F: FnMut() -> bool + Sync,
{
    // Prepare tet mesh for simulation.

    let verts = mesh.vertex_positions().to_vec();
    mesh.attrib_or_add_data::<_, VertexIndex>("ref", verts.as_slice())?;

    mesh.attrib_or_add::<_, VertexIndex>("vel", [0.0; 3])?;

    //mesh.set_attrib::<_, CellVertexIndex>("cell_vertex_forces", Vector3([0.0; 3]))?;

    {
        // compute reference element signed volumes
        let ref_volumes: Vec<f64> = mesh.cell_iter()
            .map(|cell| ref_tet(mesh, cell).signed_volume())
            .collect();
        if ref_volumes.iter().find(|&&x| x <= 0.0).is_some() {
            return Err(Error::InvertedReferenceElement);
        }
        mesh.set_attrib_data::<_, CellIndex>("ref_volume", ref_volumes.as_slice())?;
    }

    {
        // compute reference shape matrix inverses
        let ref_shape_mtx_inverses: Vec<Matrix3<f64>> = mesh.cell_iter()
            .map(|cell| {
                let ref_shape_matrix = ref_tet(mesh, cell).shape_matrix();
                // We assume that ref_shape_matrices are non-singular.
                ref_shape_matrix.inverse().unwrap()
            })
            .collect();
        mesh.set_attrib_data::<_, CellIndex>(
            "ref_shape_mtx_inv",
            ref_shape_mtx_inverses.as_slice(),
        )?;
    }

    let prev_pos: Vec<Vector3<f64>> = reinterpret_slice(mesh.vertex_positions()).to_vec();
    let (r, new_pos) = {
        let mu = params.material.shear_modulus as f64;
        let lambda = params.material.bulk_modulus as f64 - 2.0*params.material.shear_modulus as f64/3.0;
        let density = params.material.density as f64;
        let damping = params.material.damping as f64;
        let gravity = [params.gravity[0] as f64, params.gravity[1] as f64, params.gravity[2] as f64];
        let mut nlp = NeohookeanEnergyModel::new(mesh, check_interrupt)
            .material(lambda, mu, density, damping)
            .gravity(gravity);
        if let Some(dt) = params.time_step {
            nlp = nlp.time_step(dt as f64);
        }
        let mut ipopt = Ipopt::new_newton(nlp);

        ipopt.set_option("tol", 1e-9);
        ipopt.set_option("acceptable_tol", 1e-8);
        ipopt.set_option("max_iter", 800);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("sb", "yes"); // removes the Ipopt welcome message
        ipopt.set_option("print_level", 0);
        //ipopt.set_option("print_timing_statistics", "yes");
        //ipopt.set_option("hessian_approximation", "limited-memory");
        //ipopt.set_option("derivative_test", "second-order");
        //ipopt.set_option("derivative_test_tol", 1e-4);
        //ipopt.set_option("point_perturbation_radius", 0.01);
        ipopt.set_option("nlp_scaling_max_gradient", 1e-5);
        ipopt.set_intermediate_callback(Some(NeohookeanEnergyModel::intermediate_cb));
        let (r, _obj) = ipopt.solve();
        (r, ipopt.solution().to_vec())
    };

    match r {
        ipopt::ReturnStatus::SolveSucceeded | ipopt::ReturnStatus::SolvedToAcceptableLevel => {
            // Write back the velocity for the next iteration.
            let new_pos_slice: &[Vector3<f64>] = reinterpret_slice(new_pos.as_slice());
            for ((vel, &prev_x), &x) in mesh.attrib_iter_mut::<[f64; 3], VertexIndex>("vel")
                .unwrap()
                .zip(prev_pos.iter())
                .zip(new_pos_slice.iter())
            {
                *vel = (x - prev_x).into();
            }

            Ok(())
        }
        e => Err(Error::SolveError(e)),
    }
}

#[derive(Debug)]
pub enum Error {
    AttribError(attrib::Error),
    InvertedReferenceElement,
    SolveError(ipopt::ReturnStatus),
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo;
    use std::path::PathBuf;
    const DYNAMIC_PARAMS: SimParams = SimParams {
        material: 
            MaterialProperties {
                bulk_modulus: 1750e6,
                shear_modulus: 10e6,
                density: 1000.0,
                damping: 0.0,
            },
        gravity: [0.0f32, 0.0, 0.0],
        time_step: Some(0.1),
    };

    const STATIC_PARAMS: SimParams = SimParams {
        material: 
            MaterialProperties {
                bulk_modulus: 1750e6,
                shear_modulus: 10e6,
                density: 1000.0,
                damping: 0.0,
            },
        gravity: [0.0f32, -9.81, 0.0],
        time_step: None,
    };

    #[test]
    fn simple_tet_test() {
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
        mesh.add_attrib_data::<i8, VertexIndex>("fixed", vec![0,0,1,1,0,0]).ok();

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

        assert!(run(&mut mesh, STATIC_PARAMS, || true).is_ok());
    }

    #[test]
    fn torus_medium_test() {
        let mut mesh = geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets.vtk")).unwrap();
        assert!(run(&mut mesh, DYNAMIC_PARAMS, || true).is_ok());
    }

    #[test]
    #[ignore]
    fn torus_large_test() {
        let mut mesh =
            geo::io::load_tetmesh(&PathBuf::from("assets/torus_tets_large.vtk")).unwrap();
        assert!(run(&mut mesh, DYNAMIC_PARAMS, || true).is_ok());
    }
}
