mod proj_solver;
mod polar_solver;
mod solver;

use crate::contact::ContactBasis;
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
    pub density: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Friction {
    pub params: FrictionParams,
    pub contact_basis: ContactBasis,
    pub impulse: Vec<[f64; 3]>,
}

impl Friction {
    pub fn new(params: FrictionParams) -> Friction {
        Friction {
            params,
            impulse: Vec::new(),
            contact_basis: ContactBasis::new()
        }
    }
}
