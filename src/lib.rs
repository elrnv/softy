#![cfg_attr(feature = "unstable", feature(test))]

extern crate geometry as geo;
extern crate ipopt;
extern crate nalgebra as na;
extern crate rayon;
extern crate reinterpret;

mod attrib_names;
mod matrix;
mod bench;
mod energy;
mod energy_model;
mod constraint;
mod volume_constraint;
mod fem;

pub type PointCloud = geo::mesh::PointCloud<f64>;
pub type TetMesh = geo::mesh::TetMesh<f64>;
pub type PolyMesh = geo::mesh::PolyMesh<f64>;

// reexport params structs for interfacing.
pub use fem::{Error, FemEngine, MaterialProperties, SimParams, SolveResult};

pub enum SimResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<fem::Error> for SimResult {
    fn from(err: fem::Error) -> SimResult {
        match err {
            fem::Error::AttribError(e) => {
                SimResult::Error(format!("Attribute error: {:?}", e).into())
            }
            fem::Error::InvertedReferenceElement => {
                SimResult::Error(format!("Inverted reference element detected.").into())
            }
            fem::Error::SolveError(
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

impl Into<SimResult> for Result<fem::SolveResult, fem::Error> {
    fn into(self) -> SimResult {
        match self {
            Ok(fem::SolveResult {
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
    _polymesh: Option<PolyMesh>,
    sim_params: SimParams,
    interrupter: Option<Box<FnMut() -> bool>>,
) -> SimResult {
    if let Some(mesh) = tetmesh {
        match FemEngine::new(mesh, sim_params) {
            Ok(mut engine) => {
                if let Some(interrupter) = interrupter {
                    engine.set_interrupter(interrupter);
                }
                match engine.step() {
                    Err(e) => e.into(),
                    Ok(fem::SolveResult {
                        iterations,
                        objective_value,
                    }) => SimResult::Success(
                        format!("Iterations: {}\nObjective: {}", iterations, objective_value)
                            .into(),
                    ),
                }
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
    use geo::mesh::{topology::*, Attrib, TetMesh};

    const STATIC_PARAMS: SimParams = SimParams {
        material: MaterialProperties {
            bulk_modulus: 1750e6,
            shear_modulus: 10e6,
            density: 1000.0,
            damping: 0.0,
        },
        gravity: [0.0f32, -9.81, 0.0],
        time_step: None,
        tolerance: 1e-9,
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

        assert!(match sim(Some(mesh), None, STATIC_PARAMS, None) {
            SimResult::Success(_) => true,
            _ => false,
        });
    }
}
