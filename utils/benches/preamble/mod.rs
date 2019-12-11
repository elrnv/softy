#![allow(dead_code)]
pub use cgmath::prelude::*;
pub use rand::{FromEntropy, IsaacRng, Rng};
pub use std::ops::Mul;
pub use utils::soap;
pub use utils::soap::{Expr, IntoData, Matrix};

// Cgmath

pub fn matrix2_cgmath() -> cgmath::Matrix2<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Matrix2::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

pub fn matrix3_cgmath() -> cgmath::Matrix3<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Matrix3::new(
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
    )
}

pub fn matrix4_cgmath() -> cgmath::Matrix4<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Matrix4::new(
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
    )
}

pub fn vector2_cgmath() -> cgmath::Vector2<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Vector2::new(rng.gen(), rng.gen())
}

pub fn vector3_cgmath() -> cgmath::Vector3<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Vector3::new(rng.gen(), rng.gen(), rng.gen())
}

pub fn vector4_cgmath() -> cgmath::Vector4<f64> {
    let mut rng = IsaacRng::from_entropy();
    cgmath::Vector4::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
}

// Local maths

pub fn matrix2() -> soap::Matrix2<f64> {
    let mut rng = IsaacRng::from_entropy();
    soap::Matrix2::new([[rng.gen(), rng.gen()], [rng.gen(), rng.gen()]])
}

pub fn matrix3() -> soap::Matrix3<f64> {
    let mut rng = IsaacRng::from_entropy();
    soap::Matrix3::new([
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen()],
    ])
}

pub fn matrix4() -> soap::Matrix4<f64> {
    let mut rng = IsaacRng::from_entropy();
    soap::Matrix4::new([
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
    ])
}

pub fn vector2() -> soap::Vector2<f64> {
    let mut rng = IsaacRng::from_entropy();
    soap::Vector2::new([rng.gen(), rng.gen()])
}

pub fn vector3() -> soap::Vector3<f64> {
    let mut rng = IsaacRng::from_entropy();
    soap::Vector3::new([rng.gen(), rng.gen(), rng.gen()])
}

pub fn vector4() -> soap::Vector4<f64> {
    let mut rng = IsaacRng::from_entropy();
    soap::Vector4::new([rng.gen(), rng.gen(), rng.gen(), rng.gen()])
}
