use serde::{Deserialize, Serialize};
use tensr::{Chunked3, Sparse};

use crate::contact::ContactBasis;
use crate::Real;
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

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrictionParams {
    pub smoothing_weight: f64,
    pub friction_forwarding: f64,
    pub dynamic_friction: f64,
    pub inner_iterations: usize,
    pub tolerance: f64,
    pub print_level: u8,
}

impl From<crate::constraints::penalty_point_contact::FrictionParams> for Option<FrictionParams> {
    fn from(fp: crate::constraints::FrictionParams) -> Self {
        if fp.is_none() {
            None
        } else {
            Some(FrictionParams {
                smoothing_weight: 0.0,
                friction_forwarding: 0.0,
                dynamic_friction: fp.dynamic_friction,
                inner_iterations: 0,
                tolerance: 0.0,
                print_level: 0,
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrictionWorkspace<T> {
    pub params: FrictionParams,
    pub contact_basis: ContactBasis<T>,

    // Forces are used in the NL solve whereas impulses in the original Opt solver.
    /// Friction force on the object.
    pub object_force: Chunked3<Vec<T>>,
    /// Friction force on the collider.
    pub collider_force: Sparse<Chunked3<Vec<T>>, std::ops::RangeTo<usize>>,
    /// The impulse required to correct the velocities from the elasticity solve on the object
    /// along with the true applied friction impulse.
    pub object_impulse: Chunked3<(Vec<T>, Vec<T>)>,
    /// The impulse required to correct the velocities from the elasticity solve on the collider
    /// along with the true applied friction impulse.
    pub collider_impulse: Sparse<Chunked3<(Vec<T>, Vec<T>)>, std::ops::RangeTo<usize>>,
}

impl<T: Real> FrictionWorkspace<T> {
    pub fn clone_cast<S: Real>(&self) -> FrictionWorkspace<S> {
        use tensr::{AsTensor, Storage};
        FrictionWorkspace {
            params: self.params,
            contact_basis: self.contact_basis.clone_cast(),
            object_force: Chunked3::default(),
            collider_force: Sparse::from_dim(vec![], 0, Chunked3::default()),
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
    pub fn new(params: FrictionParams) -> FrictionWorkspace<T> {
        FrictionWorkspace {
            params,
            contact_basis: ContactBasis::new(),
            object_force: Chunked3::new(),
            collider_force: Sparse::from_dim(vec![], 0, Chunked3::new()),
            object_impulse: Chunked3::new(),
            collider_impulse: Sparse::from_dim(vec![], 0, Chunked3::new()),
        }
    }
}
