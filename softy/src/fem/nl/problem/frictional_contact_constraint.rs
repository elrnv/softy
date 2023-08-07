use crate::constraints::penalty_point_contact::PenaltyPointContactConstraint;
use crate::Real;
use autodiff as ad;
use std::cell::RefCell;

/// The id of the object subject to the appropriate contact constraint.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ObjectId {
    pub obj_id: usize,
    pub include_fixed: bool,
}

/// A struct that keeps track of which objects are being affected by the contact
/// constraints.
#[derive(Clone, Debug)]
pub struct FrictionalContactConstraint<T: Real> {
    pub object_id: ObjectId,
    pub collider_id: ObjectId,
    pub constraint: RefCell<PenaltyPointContactConstraint<T>>,
}

impl<T: Real> FrictionalContactConstraint<T> {
    pub fn clone_as_autodiff<S: Real>(&self) -> FrictionalContactConstraint<ad::FT<S>> {
        let FrictionalContactConstraint {
            object_id,
            collider_id,
            ref constraint,
        } = *self;

        FrictionalContactConstraint {
            object_id,
            collider_id,
            constraint: RefCell::new(constraint.borrow().clone_as_autodiff()),
        }
    }
}
