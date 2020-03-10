//mod elastic_solver;
pub mod contact_solver;
pub mod polar_solver;
pub mod proj_solver;
pub mod solver;

use crate::contact::ContactBasis;
use tensr::{Chunked3, Sparse};
//pub use elastic_solver::*;
//pub use polar_solver::*;

/// Result from one inner friction step.
#[derive(Clone, Debug, PartialEq)]
pub struct FrictionSolveResult {
    /// The value of the dissipation objective at the end of the step.
    pub objective_value: f64,
    /// Resultant friction impulse in contact space.
    pub solution: Vec<[f64; 2]>,
    /// Iteration count for the resulting solve.
    pub iterations: u32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionParams {
    pub smoothing_weight: f64,
    pub friction_forwarding: f64,
    pub dynamic_friction: f64,
    pub inner_iterations: usize,
    pub tolerance: f64,
    pub print_level: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrictionalContact {
    pub params: FrictionParams,
    pub contact_basis: ContactBasis,
    /// The impulse required to correct the velocities from the elasticity solve on the object
    /// along with the true applied friction impulse.
    pub object_impulse: Chunked3<(Vec<f64>, Vec<f64>)>,
    /// The impulse required to correct the velocities from the elasticity solve on the collider
    /// along with the true applied friction impulse.
    pub collider_impulse: Sparse<Chunked3<(Vec<f64>, Vec<f64>)>, std::ops::RangeTo<usize>>,
}

impl FrictionalContact {
    pub fn new(params: FrictionParams) -> FrictionalContact {
        FrictionalContact {
            params,
            contact_basis: ContactBasis::new(),
            object_impulse: Chunked3::new(),
            collider_impulse: Sparse::from_dim(vec![], 0, Chunked3::new()),
        }
    }
}
