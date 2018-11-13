#![cfg_attr(feature = "unstable", feature(test))]

extern crate geometry as geo;
extern crate ipopt;
extern crate nalgebra as na;
extern crate rayon;
extern crate reinterpret;
extern crate implicits;

#[cfg(test)]
#[macro_use]
extern crate approx;

mod attrib_names;
mod bench;
mod constraint;
mod constraints;
mod energy;
mod energy_models;
pub mod fem;
mod matrix;

pub type PointCloud = geo::mesh::PointCloud<f64>;
pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;
pub type TriMesh = geo::mesh::TriMesh<f64>;

pub use self::fem::{ElasticityParameters, Material, SimParams, SolveResult};
use crate::geo::mesh::attrib;

#[derive(Debug)]
pub enum Error {
    SizeMismatch,
    AttribError(attrib::Error),
    InvertedReferenceElement,
    SolveError(ipopt::SolveStatus, SolveResult), // Iterations and objective value
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError(err)
    }
}

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<Error> for SimResult {
    fn from(err: Error) -> SimResult {
        match err {
            Error::SizeMismatch => SimResult::Error(format!("Size mismatch error.").into()),
            Error::AttribError(e) => SimResult::Error(format!("Attribute error: {:?}", e).into()),
            Error::InvertedReferenceElement => {
                SimResult::Error(format!("Inverted reference element detected.").into())
            }
            Error::SolveError(
                e,
                SolveResult {
                    iterations,
                    objective_value,
                },
            ) => SimResult::Error(
                format!(
                    "Solve failed: {:?}\nIterations: {}\nObjective: {}",
                    e, iterations, objective_value
                )
                .into(),
            ),
        }
    }
}

impl Into<SimResult> for Result<SolveResult, Error> {
    fn into(self) -> SimResult {
        match self {
            Ok(SolveResult {
                iterations,
                objective_value,
                ..
            }) => SimResult::Success(
                format!("Iterations: {}\nObjective: {}", iterations, objective_value).into(),
            ),
            Err(err) => err.into(),
        }
    }
}

pub fn sim(
    tetmesh: Option<TetMesh>,
    material: Material,
    _polymesh: Option<PolyMesh>,
    sim_params: SimParams,
    interrupter: Option<Box<FnMut() -> bool>>,
) -> SimResult {
    if let Some(mesh) = tetmesh {
        match fem::SolverBuilder::new(sim_params).add_solid(mesh).solid_material(material).build() {
            Ok(mut engine) => {
                if let Some(interrupter) = interrupter {
                    engine.set_interrupter(interrupter);
                }
                engine.step().into()
            }
            Err(e) => e.into(),
        }
    } else {
        SimResult::Error("Tetmesh not found".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::mesh::{topology::*, Attrib, TetMesh};

    const STATIC_PARAMS: SimParams = SimParams {
        gravity: [0.0f32, -9.81, 0.0],
        time_step: None,
        tolerance: 1e-9,
        max_iterations: 800,
    };

    const MATERIAL: Material = Material {
        elasticity: ElasticityParameters {
            bulk_modulus: 1750e6,
            shear_modulus: 10e6,
        },
        incompressibility: false,
        density: 1000.0,
        damping: 0.0,
    };

    #[test]
    fn sim_test() {
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
        mesh.add_attrib_data::<i8, VertexIndex>("fixed", vec![0, 0, 1, 1, 0, 0])
            .unwrap();

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
            .unwrap();

        assert!(match sim(Some(mesh), MATERIAL, None, STATIC_PARAMS, None) {
            SimResult::Success(_) => true,
            _ => false,
        });
    }
}