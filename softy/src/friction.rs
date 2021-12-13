//mod elastic_solver;
#[cfg(feature = "optsolver")]
pub mod contact_solver;
#[cfg(feature = "optsolver")]
pub mod polar_solver;
#[cfg(feature = "optsolver")]
pub mod proj_solver;
#[cfg(feature = "optsolver")]
pub mod solver;

use crate::contact::ContactBasis;
use crate::Real;
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
pub struct FrictionImpulses<T> {
    pub params: FrictionParams,
    pub contact_basis: ContactBasis<T>,
    /// The impulse required to correct the velocities from the elasticity solve on the object
    /// along with the true applied friction impulse.
    pub object_impulse: Chunked3<(Vec<T>, Vec<T>)>,
    /// The impulse required to correct the velocities from the elasticity solve on the collider
    /// along with the true applied friction impulse.
    pub collider_impulse: Sparse<Chunked3<(Vec<T>, Vec<T>)>, std::ops::RangeTo<usize>>,
}

impl<T: Real> FrictionImpulses<T> {
    pub fn clone_cast<S: Real>(&self) -> FrictionImpulses<S> {
        use tensr::{AsTensor, Storage};
        FrictionImpulses {
            params: self.params,
            contact_basis: self.contact_basis.clone_cast(),
            object_impulse: self
                .object_impulse
                .iter()
                .map(|(a, b)| {
                    (
                        a.as_tensor().cast::<S>().into(),
                        b.as_tensor().cast::<S>().into(),
                    )
                })
                .collect(),
            collider_impulse: Sparse::from_dim(
                self.collider_impulse.indices().to_vec(),
                self.collider_impulse.selection().target.end,
                Chunked3::from_flat((
                    self.collider_impulse
                        .storage()
                        .0
                        .iter()
                        .map(|&x| S::from(x).unwrap())
                        .collect(),
                    self.collider_impulse
                        .storage()
                        .1
                        .iter()
                        .map(|&x| S::from(x).unwrap())
                        .collect(),
                )),
            ),
        }
    }
    pub fn new(params: FrictionParams) -> FrictionImpulses<T> {
        FrictionImpulses {
            params,
            contact_basis: ContactBasis::new(),
            object_impulse: Chunked3::new(),
            collider_impulse: Sparse::from_dim(vec![], 0, Chunked3::new()),
        }
    }
}
