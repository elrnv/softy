use crate::friction::FrictionParams;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactType {
    Implicit,
    Point,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SmoothContactParams {
    pub kernel: implicits::KernelType,
    pub contact_type: ContactType,
    pub friction_params: Option<FrictionParams>,
}

