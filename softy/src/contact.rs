
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactType {
    Implicit,
    Point,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SmoothContactParams {
    pub kernel: implicits::KernelType,
    pub contact_type: ContactType,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionParams {
    pub dynamic_friction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Friction {
    pub params: FrictionParams,
    pub impulse: Vec<f64>,
}
