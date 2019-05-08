mod polar_solver;
mod solver;

use crate::contact::ContactBasis;
pub use polar_solver::*;
pub use solver::*;

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
    pub force: Vec<[f64; 3]>,
}

impl Friction {
    pub fn new(params: FrictionParams) -> Friction {
        Friction {
            params,
            force: Vec::new(),
            contact_basis: ContactBasis::new()
        }
    }
}
