//mod elastic_solver;
mod polar_solver;
mod proj_solver;
mod solver;

use crate::contact::ContactBasis;
//pub use elastic_solver::*;
pub use polar_solver::*;
pub use proj_solver::*;

/// Result from one inner friction step.
#[derive(Clone, Debug, PartialEq)]
pub struct FrictionSolveResult {
    /// The value of the dissipation objective at the end of the step.
    pub objective_value: f64,
    /// Resultant friction force in contact space.
    pub solution: Vec<[f64; 2]>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionParams {
    pub dynamic_friction: f64,
    pub inner_iterations: usize,
    pub tolerance: f64,
    pub print_level: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrictionalContact {
    pub params: FrictionParams,
    pub contact_basis: ContactBasis,
    pub object_impulse: Vec<[f64; 3]>,
    pub collider_impulse: Vec<[f64; 3]>,
}

impl FrictionalContact {
    pub fn new(params: FrictionParams) -> FrictionalContact {
        FrictionalContact {
            params,
            contact_basis: ContactBasis::new(),
            impulse: Vec::new(),
        }
    }
}
