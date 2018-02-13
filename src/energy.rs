use na::{Vector3, Scalar};

/// Energy trait. This describes relationship between strain and stress.
pub trait Energy<T: Scalar> {
    fn energy(&self) -> T;
    fn energy_gradient(&self) -> Vec<Vector3<T>>;
}
